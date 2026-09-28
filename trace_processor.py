#!/usr/bin/env python3
"""从 runtime_log 里提取每个 case 的性能数据，做对比。

数据来源（按优先级）：
  1. `<case>/step_csv/*.csv` —— polar-sgd 的 wrapper 每一步写一行，
     列有 step / loss / step_time_s / elapsed_s / dp_world_size / pp_size / tp_size ...
     这是最直接的耗时来源
  2. `<case>/tb_scalars/` 里的 tfevent —— 没有 step_csv 时退回到读事件的 wall_time

每个 case 的元信息（rate / POLAR 还是 baseline / 配置）从 `runtime_log/case.json` 拿。

每步 token 数 = per-device-batch-size × dp_world_size × seq_len，
吞吐 = 每步 token 数 / 平均步时。和最早那版写死 16384 的做法一致，
只是现在从 case 自己的配置里算，换配置也不会算错。

用法：
    python trace_processor.py                      # 对比 runtime_log 下所有 case
    python trace_processor.py --case 0001 0002     # 只看指定的
    python trace_processor.py --csv out.csv        # 顺便导出成 CSV
    python trace_processor.py --runtime-log /别的/路径
"""

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

DEFAULT_RUNTIME_LOG = Path(__file__).resolve().parent.parent / "runtime_log"


# ---------------------------------------------------------------- 读元信息


def load_case_records(runtime_log: Path) -> dict:
    """读 runtime_log/case.json，返回 {case_id: record}。"""
    path = runtime_log / "case.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {str(c.get("case")): c for c in data.get("cases", [])}


def tokens_per_step(record: dict, fallback: int = 0) -> int:
    """每步处理的 token 数 = per-device-batch-size × dp × seq-len。

    从 case.json 记的启动参数里取；取不到就返回 fallback（0 表示算不出吞吐）。
    """
    params = (record or {}).get("launch_params", {}).get("node0", {})
    try:
        batch = int(params["per-device-batch-size"])
        seq = int(params["seq-len"])
    except (KeyError, TypeError, ValueError):
        return fallback
    # dp = 节点数（每个节点一个 DP 副本）
    dp = 2
    try:
        dp = int(params.get("dp-world-size", dp))
    except (TypeError, ValueError):
        pass
    return batch * dp * seq


# ---------------------------------------------------------------- 数据来源


def read_step_csv(case_dir: Path):
    """返回 (steps, rows)，steps 是 [{step, loss, step_time_s, ...}]，按 step 排序。

    一个 case 目录下正常只有一个 csv（每次跑生成一个带时间戳的）。
    有多个就取文件名最大（最晚）的那个。
    """
    d = case_dir / "step_csv"
    if not d.is_dir():
        return None
    files = sorted(p for p in d.glob("*.csv") if p.is_file())
    if not files:
        return None
    path = files[-1]
    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    rows.append({
                        "step": int(r["step"]),
                        "loss": float(r["loss"]) if r.get("loss") not in (None, "") else None,
                        "step_time_s": float(r["step_time_s"])
                        if r.get("step_time_s") not in (None, "") else None,
                    })
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        return None
    if not rows:
        return None

    # 同一个 step 会出现多行：pp 最后一级的多个 rank 都会往同一个 csv 写
    # （wrapper 里只挡了 is_last 和 dp_local_rank==0，TP>1 时多个 rank 都通过）。
    # 实测 TP=2 时每个 step 有 2 行、且两行的 step_time_s 略有差异。
    # 这里先按 step 聚合，每步取一行（多个值取平均），否则平均值会被重复行带偏。
    grouped = {}
    for r in rows:
        grouped.setdefault(r["step"], []).append(r)
    rows = []
    for step in sorted(grouped):
        group = grouped[step]
        times = [g["step_time_s"] for g in group if g["step_time_s"] is not None]
        losses = [g["loss"] for g in group if g["loss"] is not None]
        rows.append({
            "step": step,
            "step_time_s": statistics.fmean(times) if times else None,
            "loss": statistics.fmean(losses) if losses else None,
            "dup_rows": len(group),
        })
    return rows, path.name


def read_tfevents(case_dir: Path):
    """退回方案：从 tb_scalars 的 tfevent 里按 wall_time 推每步耗时。"""
    tb = case_dir / "tb_scalars"
    if not tb.is_dir():
        return None
    files = sorted(p for p in tb.rglob("events.out.tfevents*") if p.is_file())
    if not files:
        return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None, None

    step_times = {}
    for path in files:
        try:
            acc = EventAccumulator(str(path))
            acc.Reload()
            for tag in acc.Tags().get("scalars", []):
                for ev in acc.Scalars(tag):
                    step_times.setdefault(ev.step, ev.wall_time)
        except Exception:
            continue
    if not step_times:
        return None
    times = sorted(step_times.items())
    rows = []
    for i, (step, t) in enumerate(times):
        dur = None if i == 0 else t - times[i - 1][1]
        rows.append({"step": step, "loss": None, "step_time_s": dur})
    return rows, f"tfevent x{len(files)}"


# ---------------------------------------------------------------- 单 case 分析


def analyze_case(case_id: str, case_dir: Path, record: dict) -> dict:
    """算一个 case 的平均步时和吞吐。第一轮（step 0）不计入，和原来一致。"""
    result = {
        "case": case_id,
        "kind": (record or {}).get("kind", "?"),
        "rate": (record or {}).get("rate", "?"),
        "bitwidth": None,
        "steps": 0,
        "avg_step_time_s": None,
        "throughput": None,
        "first_loss": None,
        "last_loss": None,
        "source": None,
        "note": "",
    }

    cfg = (record or {}).get("config", {}) or {}
    result["bitwidth"] = cfg.get("bit-width")

    loaded = read_step_csv(case_dir)
    if loaded:
        rows, name = loaded
        result["source"] = f"step_csv/{name}"
    else:
        loaded = read_tfevents(case_dir)
        if not loaded:
            result["note"] = "没有 step_csv 也没有 tfevent"
            return result
        rows, name = loaded
        result["source"] = name

    result["steps"] = len(rows)

    losses = [r["loss"] for r in rows if r["loss"] is not None]
    if losses:
        result["first_loss"] = losses[0]
        result["last_loss"] = losses[-1]

    # 排除第一轮：第一轮含模型初始化/编译等一次性开销
    durations = [r["step_time_s"] for r in rows[1:] if r["step_time_s"] is not None]
    if not durations:
        result["note"] = "有效步时数据不足（只有一轮或全是空值）"
        return result

    avg = statistics.fmean(durations)
    result["avg_step_time_s"] = avg

    tokens = tokens_per_step(record)
    if tokens:
        result["throughput"] = tokens / avg
    else:
        result["note"] = "case.json 里缺 per-device-batch-size / seq-len，算不出吞吐"
    return result


# ---------------------------------------------------------------- 输出


def hr_time(sec) -> str:
    return "—" if sec is None else f"{sec:.3f}s"


def hr_tp(tp) -> str:
    return "—" if tp is None else f"{tp:,.0f} tok/s"


def hr_loss(x) -> str:
    return "—" if x is None else f"{x:.4f}"


def print_table(results: list) -> None:
    if not results:
        print("没找到任何 case。")
        return

    name_w = max(len("case"), max(len(r["case"]) for r in results))
    kind_w = max(len("类型"), max(len(f'{r["kind"]}') for r in results))
    rate_w = max(len("网速"), max(len(str(r["rate"])) for r in results))

    header = (f"{'case':<{name_w}}  {'类型':<{kind_w}}  {'网速':>{rate_w}}  "
              f"{'块':>4}  {'轮次':>4}  {'平均步时':>10}  {'吞吐':>14}  "
              f"{'首 loss':>9}  {'末 loss':>9}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['case']:<{name_w}}  {r['kind']:<{kind_w}}  {str(r['rate']):>{rate_w}}  "
              f"{str(r['bitwidth'] or '—'):>4}  {r['steps']:>4}  "
              f"{hr_time(r['avg_step_time_s']):>10}  {hr_tp(r['throughput']):>14}  "
              f"{hr_loss(r['first_loss']):>9}  {hr_loss(r['last_loss']):>9}")

    # 同网速下 baseline 和 POLAR 的对比
    by_rate = {}
    for r in results:
        by_rate.setdefault(str(r["rate"]), {})[r["kind"]] = r
    rows = []
    for rate, pair in sorted(by_rate.items()):
        polar, base = pair.get("polar"), pair.get("baseline")
        if polar and base and polar["throughput"] and base["throughput"]:
            rows.append((rate, polar, base))
    if rows:
        print("\n=== 同网速对比（POLAR 相对 baseline）===")
        for rate, polar, base in rows:
            speedup = polar["throughput"] / base["throughput"]
            t_ratio = base["avg_step_time_s"] / polar["avg_step_time_s"]
            print(f"  {rate:>7}: 吞吐 {hr_tp(polar['throughput'])} vs "
                  f"{hr_tp(base['throughput'])}  →  快 {speedup:.2f}x"
                  f"（步时 {hr_time(polar['avg_step_time_s'])} vs "
                  f"{hr_time(base['avg_step_time_s'])}，{t_ratio:.2f}x）")


def export_csv(results: list, path: Path) -> None:
    fields = ["case", "kind", "rate", "bitwidth", "steps",
              "avg_step_time_s", "throughput", "first_loss", "last_loss",
              "source", "note"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\n已导出: {path}")


def parse_args():
    p = argparse.ArgumentParser(description="从 runtime_log 提取并对比各 case 的性能")
    p.add_argument("--runtime-log", type=Path, default=DEFAULT_RUNTIME_LOG,
                   help=f"runtime_log 目录（默认 {DEFAULT_RUNTIME_LOG}）")
    p.add_argument("--case", nargs="+", default=None,
                   help="只看这些 case，例如 --case 0001 0002")
    p.add_argument("--csv", type=Path, default=None, help="顺便导出成 CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runtime_log: Path = args.runtime_log.expanduser().resolve()

    if not runtime_log.is_dir():
        print(f"[ERROR] 目录不存在: {runtime_log}")
        sys.exit(1)

    records = load_case_records(runtime_log)

    # case 目录 = runtime_log 下的纯数字目录（四位编号）
    case_dirs = sorted(
        d for d in runtime_log.iterdir()
        if d.is_dir() and d.name.isdigit()
    )
    if args.case:
        wanted = {c.zfill(4) for c in args.case}
        case_dirs = [d for d in case_dirs if d.name in wanted]

    print(f"runtime_log: {runtime_log}")
    print(f"找到 {len(case_dirs)} 个 case 目录\n")
    if not case_dirs:
        print("（还没有跑过任何 case）")
        return

    results = []
    for d in case_dirs:
        r = analyze_case(d.name, d, records.get(d.name, {}))
        results.append(r)
        flag = f"  [{r['note']}]" if r["note"] else ""
        print(f"  {r['case']}  轮次={r['steps']:<4} 平均步时={hr_time(r['avg_step_time_s']):>9}  "
              f"吞吐={hr_tp(r['throughput']):>14}   <- {r['source']}{flag}")

    print()
    print_table(results)

    if args.csv:
        export_csv(results, args.csv)


if __name__ == "__main__":
    main()
