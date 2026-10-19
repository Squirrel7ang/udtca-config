#!/usr/bin/env python3
"""Qwen14B 一键功能性能测试：固定跑 4 个 case，polar vs DDP baseline。

目的：让测试人员一条命令就能验证 POLAR + bitscom 到底有没有真的优化通信。

固定 4 个 case（写死在 run_cases.json，也可 --cases 指定别的 json）：
    2Gb: baseline / polar        1Gb: baseline / polar

一条命令跑完两机 16 rank（每机 8 卡 × 2 芯粒 = 每机 16 process）全套流程
（GPU 占用检查 → 生成脚本 → tc 限速 → 起训 → 收日志），然后：

  * 训练时终端实时打 `[rt-metric]` 行：每步 loss、DP-PP 重叠率、DP 通信量降低量
    （只在 polar 打，baseline 不打；mm 见 polar-sgd 的 wrapper）。
  * 4 个 case 跑完后汇总：每个 case 的总 loss、用时、端到端吞吐、通信量、
    DP-PP 重叠率；并给出 polar 相对 baseline 的对比。

幂等：每次运行新建 `run_log/run_<时间戳>/`，从不覆盖旧结果；本脚本永远只对比
本次运行里的这 4 个 case，不管 run_log 下已经有多少次历史运行。

用法：
    python run.py                       # 跑固定的 4 个 case，max-steps 用 json 里的值
    python run.py --max-steps 3         # 冒烟：每个 case 只跑 3 步
    python run.py --cases my_cases.json # 用别的 case 定义
"""

import argparse
import json
import re
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

CONFIG_DIR = Path(__file__).resolve().parent          # udtca-config/qwen14b
CONFIG_ROOT = CONFIG_DIR.parent                        # udtca-config
REPO_ROOT = CONFIG_ROOT.parent                         # udtca
sys.path.insert(0, str(CONFIG_ROOT / "common"))
sys.path.insert(0, str(CONFIG_ROOT / "utils"))

import case_common as cc  # noqa: E402
import orchestrator as orch  # noqa: E402

DEFAULT_CASES_JSON = CONFIG_DIR / "run_cases.json"
DEFAULT_RUN_ROOT = REPO_ROOT / "run_log"

QWEN14B_SPEC = cc.ExperimentSpec(
    name="qwen14b",
    experiment_dir=REPO_ROOT / "experiments" / "qwen14b",
    entry_script="run_qwen14b_polar_dp_pp_tp.py",
    config_dir=CONFIG_DIR,
    remote_experiment_dir=f"{cc.REMOTE_BASE_DIR}/experiments/qwen14b",
)

SCRIPT_PREFIX = "train_qwen14b"
BASELINE_SCRIPTS = (
    "0_train_qwen14b_baseline_ddp_1f1b_tp.sh",
    "1_train_qwen14b_baseline_ddp_1f1b_tp.sh",
)
BASELINE_RUN_LABEL = "baseline_ddp_1f1b_tp"


# ---------------------------------------------------------------- 启动参数

def build_launch_params(config: Dict[str, Any], node_rank: int,
                        case_dir: "Optional[Path]") -> "OrderedDict":
    """POLAR + bitscom 这条路径的启动参数（与 generate_and_run.py 保持一致）。"""
    micro_batches = config["micro-batches"]
    params = OrderedDict(
        [
            ("model-name", "Qwen/Qwen2.5-14B-Instruct"),
            ("pp-size", config["pp-size"]),
            ("tp-size", config["tp-size"]),
            ("micro-batches", micro_batches),
            ("comm-timing", 8),
            ("max-steps", config["max-steps"]),
            ("per-device-batch-size", micro_batches),
            ("seq-len", config["seq-len"]),
            ("lr", "2e-4"),
            ("dataset-name-or-path", "HuggingFaceFW/fineweb"),
            ("text-field", "text"),
            ("using-polar", True),
            ("run-label", "polar_bitscom_1f1b_tp"),
            ("polar-hook", "ef_lowmem"),
            ("polar-bucket-numel", 64000000),
            ("polar-max-inflight-buckets", 4 if node_rank == 0 else 1),
            ("method", "bitscom"),
            ("bitwidth", config["bit-width"]),
            ("log-interval", 10),
        ]
    )
    if case_dir is not None:
        params["step-log-dir"] = str(case_dir / "step_csv")
    params["disable-profiler"] = False
    return params


# ---------------------------------------------------------------- run_log 重定向
#
# orchestrator 的 allocate_case() 固定往 <repo>/runtime_log/ 下编号。run.py 要独立
# 到 run_log/run_<ts>/，所以这里把 case_common 的两个模块级常量重绑，并把
# orchestrator 命名空间里的 allocate_case 换成 run 作用域的版本（case 编号、本地
# case 目录、远程暂存目录都在本次 run 下）。

RUN_DIR: Optional[Path] = None
RUN_NAME: Optional[str] = None


def _run_allocate_case() -> tuple:
    cases = cc.load_cases()
    case = cc.next_case_id(cases)
    case_dir = RUN_DIR / case
    case_dir.mkdir(parents=True, exist_ok=True)
    remote_case_dir = f"{cc.REMOTE_BASE_DIR}/run_log/{RUN_NAME}/{case}"
    return case, case_dir, remote_case_dir, cases


class _Tee:
    """把输出同时写到终端和文件（run.log）。

    只实现 print 用到的几个方法；isatty 返回 False，让上层当作非终端处理。
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        return False


def _tee_stdout(run_dir: Path) -> Path:
    """把 stdout/stderr 同时落一份到 <run_dir>/run.log，返回日志路径。"""
    log_path = run_dir / "run.log"
    # 行缓冲：训练卡住时也能实时 tail 到最后的输出
    fh = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    return log_path


def _bind_run_dir(run_dir: Path) -> None:
    global RUN_DIR, RUN_NAME
    RUN_DIR = run_dir
    RUN_NAME = run_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)
    cc.LOG_DIR = run_dir
    cc.CASE_JSON_PATH = run_dir / "case.json"
    orch.allocate_case = _run_allocate_case


# ---------------------------------------------------------------- 结果解析

_RT_RE = re.compile(r"\[rt-metric\]\s+(.*)")
_KV_RE = re.compile(r"(\w+)=([^\s]+)")


def _parse_rt_metrics(log_path: Path) -> list:
    """从 train_node0.log 里抽出每一步的 [rt-metric] 行。"""
    rows = []
    if not log_path.exists():
        return rows
    for line in log_path.read_text(errors="ignore").splitlines():
        m = _RT_RE.search(line)
        if not m:
            continue
        rec = {}
        for k, v in _KV_RE.findall(m.group(1)):
            rec[k] = v
        rows.append(rec)
    return rows


def _parse_step_csv(case_dir: Path) -> Dict[str, Any]:
    """读 step_csv/*.csv，算平均步时与总训练时间。"""
    import csv

    csv_dir = case_dir / "step_csv"
    files = sorted(csv_dir.glob("*.csv")) if csv_dir.is_dir() else []
    if not files:
        return {}
    # 两个 DP rank 的最后一个 stage 会写同一个 csv 文件（run_id 相同），
    # 同一步会有多行、步时略有差异；按 step 去重，取慢的那个（瓶颈）。
    per_step = {}
    with open(files[0], newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                step = int(float(row["step"]))
                step_time = float(row["step_time_s"])
            except (KeyError, ValueError):
                continue
            if step not in per_step or step_time > per_step[step]:
                per_step[step] = step_time
    if not per_step:
        return {}
    times = [per_step[s] for s in sorted(per_step)]
    return {
        "num_steps": len(times),
        "avg_step_s": sum(times) / len(times),
        "total_step_s": sum(times),
    }


def _parse_losses(log_path: Path) -> list:
    losses = []
    if not log_path.exists():
        return losses
    pat = re.compile(r"Step (\d+), Loss:\s*([0-9.]+)")
    for line in log_path.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            losses.append((int(m.group(1)), float(m.group(2))))
    return losses


def _offline_overlap(case_dir: Path) -> Dict[str, Any]:
    """按 utils/overlap_interval.py 的口径，从 profiler trace 算 DP-PP 重叠率。"""
    try:
        from analyze_overlap import rank0_trace_path
        import overlap_interval as oi
    except Exception:
        return {}
    trace = rank0_trace_path(case_dir)
    if trace is None:
        return {}
    res = oi.analyze(trace, "last")
    if not res:
        return {}
    return {
        "overlap": res["overlap"],
        "dp": res["dp"],
        "pp": res["pp"],
        "iter_s": res["iter_s"],
    }


def _gb(num_bytes: Optional[float]) -> Optional[float]:
    if num_bytes is None:
        return None
    return num_bytes / 1e9


def _collect_case(record: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """把一个 case 的记录 + 日志解析成汇总用的结构。"""
    case_dir = Path(record["log_dir"])
    params = (record.get("launch_params") or {}).get("node0") or {}

    def _int(key, fallback):
        try:
            return int(params.get(key, fallback))
        except (TypeError, ValueError):
            return fallback

    seq_len = _int("seq-len", default_config.get("seq-len", 256))
    per_dev_batch = _int("per-device-batch-size", default_config.get("micro-batches", 32))

    step = _parse_step_csv(case_dir)
    losses = _parse_losses(case_dir / "train_node0.log")
    rt = _parse_rt_metrics(case_dir / "train_node0.log")
    overlap = _offline_overlap(case_dir)

    # dp_world_size 从 step_csv 拿不到就在这里用默认 2
    dp_world = 2
    tokens = None
    throughput = None
    if step.get("total_step_s"):
        tokens = step["num_steps"] * dp_world * per_dev_batch * seq_len
        throughput = tokens / step["total_step_s"]

    # 端到端墙钟：case.json 里记的是启动/收尾时间戳（秒级），
    # 包含建库、加载模型、编译、运行全部环节，和 total_step_s（只算训练循环）不是一个口径。
    elapsed_s = None
    try:
        fmt = "%Y-%m-%d_%H-%M-%S"
        t0 = datetime.strptime(record["started_at"], fmt)
        if record.get("finished_at"):
            t1 = datetime.strptime(record["finished_at"], fmt)
            elapsed_s = (t1 - t0).total_seconds()
    except (KeyError, TypeError, ValueError):
        elapsed_s = None

    # 每步重叠率 / 通信量（polar 才有），取均值
    ov_vals, saved_vals = [], []
    for r in rt:
        try:
            ov_vals.append(float(r["dp_pp_overlap"].rstrip("%")) / 100.0)
        except (KeyError, ValueError):
            pass
        try:
            saved_vals.append(_parse_bytes(r["comm_saved"]))
        except (KeyError, ValueError):
            pass

    return {
        "case": record["case"],
        "test_index": record.get("test_index"),
        "kind": record.get("kind"),
        "rate": record.get("rate"),
        "exit_code": record.get("exit_code"),
        "comm_gb": record.get("comm_gb"),
        "comm_bytes": record.get("comm_bytes"),
        "final_loss": losses[-1][1] if losses else None,
        "num_steps": step.get("num_steps"),
        "avg_step_s": step.get("avg_step_s"),
        "total_step_s": step.get("total_step_s"),
        "elapsed_s": elapsed_s,
        "throughput_tok_s": throughput,
        "offline_overlap": overlap.get("overlap"),
        "rt_steps": len(rt),
        "rt_overlap_mean": (sum(ov_vals) / len(ov_vals)) if ov_vals else None,
        "rt_comm_saved_mean": (sum(saved_vals) / len(saved_vals)) if saved_vals else None,
    }


_BYTE_UNITS = {"GB": 1e9, "MB": 1e6, "KB": 1e3, "B": 1.0}


def _parse_bytes(text: str) -> float:
    for unit, mul in _BYTE_UNITS.items():
        if text.endswith(unit):
            return float(text[: -len(unit)]) * mul
    return float(text)


# ---------------------------------------------------------------- 汇总

def _fmt_pct(v):
    return f"{v * 100:.1f}%" if v is not None else "-"


def _fmt_gb(v):
    return f"{v:.3f}" if v is not None else "-"


def _fmt(v, nd=2):
    return f"{v:.{nd}f}" if v is not None else "-"


def _summary_table(rows: list) -> str:
    hdr = (f"{'case':>5} {'kind':8} {'rate':6} {'loss':>9} {'steps':>6} "
           f"{'avg_step_ms':>12} {'e2e_s':>9} {'tok/s':>10} {'comm_GB':>9} "
           f"{'overlap':>8} {'rt_overlap':>10}")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(
            f"{r['case']:>5} {str(r['kind']):8} {str(r['rate']):6} "
            f"{_fmt(r['final_loss'], 4):>9} {str(r['num_steps'] or '-'):>6} "
            f"{_fmt((r['avg_step_s'] or 0) * 1000, 1):>12} "
            f"{_fmt(r['elapsed_s'], 0):>9} "
            f"{_fmt(r['throughput_tok_s'], 1):>10} "
            f"{_fmt_gb(r['comm_gb']):>9} "
            f"{_fmt_pct(r['offline_overlap']):>8} "
            f"{_fmt_pct(r['rt_overlap_mean']):>10}"
        )
    return "\n".join(lines)


def _pair_compare(rows: list) -> str:
    """同一限速下 polar 相对 baseline 的对比。"""
    by_rate = {}
    for r in rows:
        by_rate.setdefault(r["rate"], {})[r["kind"]] = r
    out = []
    for rate in sorted(by_rate):
        pair = by_rate[rate]
        b, p = pair.get("baseline"), pair.get("polar")
        if not b or not p:
            continue
        comm_ratio = None
        if b.get("comm_gb") and p.get("comm_gb"):
            comm_ratio = b["comm_gb"] / p["comm_gb"]
        tp_ratio = None
        if b.get("throughput_tok_s") and p.get("throughput_tok_s"):
            tp_ratio = p["throughput_tok_s"] / b["throughput_tok_s"]
        saved_gb = None
        if b.get("comm_gb") is not None and p.get("comm_gb") is not None:
            saved_gb = b["comm_gb"] - p["comm_gb"]
        # 端到端时间优先用 case.json 的墙钟（含启动/建库/编译），拿不到退回归到
        # step_csv 的训练循环耗时（total_step_s），两者口径不同，标签里区分开。
        e2e_b, e2e_p = b.get("elapsed_s"), p.get("elapsed_s")
        e2e_label = "端到端耗时"
        if not e2e_b or not e2e_p:
            e2e_b, e2e_p = b.get("total_step_s"), p.get("total_step_s")
            e2e_label = "循环耗时"
        speedup = (e2e_b / e2e_p) if (e2e_b and e2e_p) else None
        out.append(
            f"[{rate}] "
            f"通信量 baseline={_fmt_gb(b.get('comm_gb'))}GB "
            f"polar={_fmt_gb(p.get('comm_gb'))}GB "
            f"降低={_fmt_gb(saved_gb)}GB ({_fmt(comm_ratio)}x) | "
            f"{e2e_label} baseline={_fmt(e2e_b, 0)}s "
            f"polar={_fmt(e2e_p, 0)}s "
            f"polar 快 {_fmt(speedup)}x | "
            f"重叠率 baseline={_fmt_pct(b.get('offline_overlap'))} "
            f"polar={_fmt_pct(p.get('offline_overlap'))} "
            f"(实时 {_fmt_pct(b.get('rt_overlap_mean'))} / "
            f"{_fmt_pct(p.get('rt_overlap_mean'))}) | "
            f"吞吐 polar/baseline={_fmt(tp_ratio)}x"
        )
    return "\n".join(out)


# ---------------------------------------------------------------- 主流程

def make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Qwen14B 一键 4-case 功能性能测试")
    p.add_argument("--cases", default=str(DEFAULT_CASES_JSON),
                   help="case 定义 json（默认 run_cases.json，写死 4 个 case）")
    p.add_argument("--default-config", default=None,
                   help="默认超参 json（默认用同目录 default.json）")
    p.add_argument("--max-steps", type=int, default=None,
                   help="临时覆盖每个 case 的 max-steps（默认用 case json / default.json 的值）")
    p.add_argument("--run-log-root", default=str(DEFAULT_RUN_ROOT),
                   help=f"运行结果根目录（默认 {DEFAULT_RUN_ROOT}）")
    p.add_argument("--runtime-log", choices=["on", "off"], default="on",
                   help="off 时只跑训练不归档（默认 on）")
    p.add_argument("--no-tc", action="store_true",
                   help="不下发 tc 限速（共享机器被抢、只想验证能否跑通时用；"
                        "此时 2gbit/1gbit 两组结果不再有带宽差异）")
    return p


def main():
    args = make_arg_parser().parse_args()

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 必须是绝对路径：case 目录会作为训练进程的 cwd，日志/step_csv 都用它拼路径，
    # 传相对路径时 tee 的目标在 cwd 里找不到同名目录（日志直接丢）。
    run_dir = Path(args.run_log_root).expanduser().resolve() / f"run_{ts}"
    _bind_run_dir(run_dir)
    run_log_path = _tee_stdout(run_dir)

    cc.set_runtime_log_enabled(args.runtime_log == "on")
    orch.use_experiment(QWEN14B_SPEC)
    orch.install_signal_handlers(QWEN14B_SPEC)
    orch.TC_ENABLED = not args.no_tc

    print(f"{'=' * 80}")
    print(f"Qwen14B 一键测试：run_dir = {run_dir}")
    print(f"标准输出实时写入: {run_log_path}")
    print(f"{'=' * 80}")

    # kube-proxy 会 REJECT 一批临时端口，NCCL 的 bind(0) 一旦撞上就卡初始化。
    # 每次跑之前检查一遍，并把冲突端口从内核的自动分配范围里排除掉。
    if not cc.check_port_conflicts(fix=True):
        print("\n临时端口仍与被 REJECT 的端口冲突，训练可能卡在初始化，本次退出。")
        sys.exit(1)

    if args.runtime_log == "on":
        print(f"Environment info written to: {cc.write_environment_info()}")

    default_config = cc.load_json(
        Path(args.default_config) if args.default_config
        else QWEN14B_SPEC.default_config_path
    )
    cases = cc.load_json(Path(args.cases))
    print(f"Default config: {default_config}")
    print(f"Cases ({len(cases)}): {[c.get('index') for c in cases]}")

    if not cc.check_gpus_free():
        print("\nGPU 被占用，本次退出。等卡空出来再重跑。")
        sys.exit(1)

    for i, case_cfg in enumerate(cases):
        merged = cc.merge_config(default_config, case_cfg)
        if args.max_steps is not None:
            merged["max-steps"] = int(args.max_steps)
        print(f"\n{'=' * 80}\nCase {i + 1}/{len(cases)}: {merged}\n{'=' * 80}")
        if cc.is_baseline(case_cfg, default_config):
            orch.run_baseline_case(
                QWEN14B_SPEC, merged, BASELINE_SCRIPTS, BASELINE_RUN_LABEL
            )
        else:
            orch.run_polar_case(
                QWEN14B_SPEC, merged, build_launch_params,
                SCRIPT_PREFIX, "POLAR + bitscom",
            )

    # ---- 汇总：只对比本次运行的这几个 case ----
    print(f"\n{'=' * 80}\n汇总（只对比本次运行）\n{'=' * 80}")
    records = cc.load_cases()
    rows = [_collect_case(rec, default_config) for rec in records]
    table = _summary_table(rows)
    compare = _pair_compare(rows)
    print("\n" + table + "\n")
    print(compare)

    if args.runtime_log == "on":
        summary = {"run_dir": str(run_dir), "cases": rows}
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (run_dir / "summary.md").write_text(
            "# Qwen14B 一键测试汇总\n\n"
            f"run_dir: {run_dir}\n\n```\n{table}\n```\n\n```\n{compare}\n```\n",
            encoding="utf-8",
        )
        print(f"\n汇总已写入: {run_dir/'summary.json'} 和 {run_dir/'summary.md'}")


if __name__ == "__main__":
    main()
