#!/usr/bin/env python3
"""跨机 Qwen2.5-14B 端到端训练 + profiler 采集 + 自动分析（容器版）。

**必须在 powerone1 的宿主机上运行**（容器里没有 docker / docker.sock，脚本会直接拒绝启动）：
  * node0 = 本机，宿主机执行 `docker exec trj-udtca ...`
  * node1 = powerone2，宿主机执行 `ssh <node1> 'docker exec trj-udtca ...'`

每个 mode（默认 mixed 先、baseline 后）依次做一遍：

  1. 在 run 目录下给两台机器各生成一份训练脚本
     `<run>/scripts/node{0,1}_<mode>.sh`（自包含：环境、METHOD_ARGS、torchrun 全在里面）
  2. 先起 node1（后台，输出落 node1.ssh.log），warmup 后起 node0（实时回显 + 落盘）
  3. 等两边退出，收集 train.log / step_csv / profiler trace
  4. 全部跑完自动做 trace 分析（每步耗时、通信/计算占比、DP/PP 标注），
     结果同时打到终端和 `<run>/analysis.log`、`<run>/summary.md`

profiler：每个 mode 跑 10 iter，只录第 8、9 iter
（`--profiler-wait-steps 7 --profiler-warmup-steps 0 --profiler-active-steps 2`）。
不做 tc 限速。

中途退出（Ctrl-C / 异常）会像 generate_and_run.py 那样把两台机器上的训练进程都清掉：
按「命令行里带 run_qwen14b_polar_dp_pp_tp.py 或 torchrun」精确匹配，先 SIGTERM，
8 s 后仍在的再 SIGKILL。注意 Corex 上打断在途 NCCL 有 XID 24 风险，这是你要的行为。

用法：
    python3 run_docker.py                      # mixed 先跑、再 baseline，各 10 iter
    python3 run_docker.py --modes mixed        # 只跑 mixed
    python3 run_docker.py --max-steps 3        # 冒烟（此时 profile 不到 8/9 iter）
    python3 run_docker.py --dry-run            # 只打印生成的脚本，什么都不启动
    python3 run_docker.py --preflight-only     # 只检查两台机器/容器/ssh 是否就绪
"""

import argparse
import base64
import csv
import io
import json
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

LOCAL_HOST = "local"
DEFAULT_REPO = "/data/tangruijing/udtca"
ENTRY_SUBPATH = "experiments/qwen14b/run_qwen14b_polar_dp_pp_tp.py"

NODES = {
    0: {"host": LOCAL_HOST, "ip": "10.205.200.14", "ifname": "ens120f0", "name": "powerone1"},
    1: {"host": "powerone2", "ip": "10.205.200.16", "ifname": "enx00e04c680003",
        "name": "powerone2"},
}

# ssh 到 node1 的参数，main() 里按命令行覆盖。
# 默认 BatchMode：无人值守时绝不能停在密码提示上（通了就通，不通 preflight 会报出来）。
SSH = {"target": "powerone2",
       "opts": ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]}

RUN_LABELS = {
    "baseline": "baseline_ddp_xnode",
    "polar_dense": "polar_dense_xnode",
    "bitscom": "bitscom_quant_xnode",
    "mixed": "polar_bitscom_xnode",
}

# 四种 mode 各自要追加的训练参数（与 cloud/edge 的 run_xnode_*.sh 一致）。
METHOD_ARGS = {
    "baseline": [
        "--baseline-mode manual",
        "--using-polar false --polar-hook none --comm-timing 0",
        "--method none",
    ],
    "polar_dense": [
        "--baseline-mode manual",
        "--using-polar true --polar-hook ef_lowmem --comm-timing 8",
        "--polar-bucket-numel 64000000 --polar-max-inflight-buckets 1",
        "--method none",
    ],
    "bitscom": [
        "--baseline-mode manual",
        "--using-polar false --polar-hook none --comm-timing 0",
        "--method bitscom --bitwidth 2",
    ],
    "mixed": [
        "--baseline-mode manual",
        "--using-polar true --polar-hook ef_lowmem --comm-timing 8",
        "--polar-bucket-numel 64000000 --polar-max-inflight-buckets 1",
        "--method bitscom --bitwidth 2",
    ],
}


# ---------------------------------------------------------------- 训练脚本模板

SCRIPT_TEMPLATE = r"""#!/usr/bin/env bash
# run_docker.py 生成 —— mode=@@mode@@ node=@@rank@@  (@@ts@@)
set -uo pipefail
REPO_ROOT=@@repo@@
ENTRY="$REPO_ROOT/@@entry@@"
CASE_DIR=@@case_dir@@
mkdir -p "$CASE_DIR/logs"
RUN_LOG="$CASE_DIR/run.log"
TEE=(tee -a "$CASE_DIR/train.log" "$RUN_LOG")

# 环境变量对齐 run.py / generate_and_run.py（common/case_common.py 的 SETUP_CMDS +
# orchestrator 的 NCCL 集合通信日志）。排查性能瓶颈期间只留这些，不额外加诊断/revoke 变量。

# ---- 容器内 Corex 环境（两台强制一致）----
export COREX_PATH=/usr/local/corex
export CUDA_HOME="$COREX_PATH"
export CPATH="$COREX_PATH/include"
export LIBRARY_PATH="$COREX_PATH/lib64"
export TRITON_CUDA_SYSROOT="$COREX_PATH"
export BITSCOM_CUDA_COMPILER="$COREX_PATH/bin/clang++"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PATH=/usr/local/corex/bin:/usr/local/corex/lib64/python3/dist-packages/bin:$PATH
export LD_LIBRARY_PATH="$COREX_PATH/lib64:/usr/local/openmpi/lib:/usr/local/lib"
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages

# ---- 通信（其余一律 NCCL 默认值）----
export NCCL_SOCKET_IFNAME=@@ifname@@
export NCCL_SOCKET_RETRY_CNT="${NCCL_SOCKET_RETRY_CNT:-600}"
export NCCL_SOCKET_RETRY_SLEEP_MSEC="${NCCL_SOCKET_RETRY_SLEEP_MSEC:-1000}"

# ---- 数据 ----
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY NO_PROXY no_proxy
export HF_ENDPOINT=https://hf-mirror.com

# ---- 输出 ----
export PYTHONUNBUFFERED=1

# 低比特 DP 同步严格串行（同 case_common：多 bucket 同时在飞会让两个 DP rank 的
# p2p 投递顺序分叉，偶发卡死）。
export POLAR_EF_LOWMEM_SERIAL_BUCKETS="${POLAR_EF_LOWMEM_SERIAL_BUCKETS:-1}"

# ---- NCCL 集合通信日志（同 orchestrator._nccl_debug_env）----
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL
export NCCL_DEBUG_FILE="$CASE_DIR/logs/nccl_%h_%p.log"

cat > "$CASE_DIR/PRE_RUN.md" <<PRE
# PRE_RUN @@ts@@
- mode=@@mode@@ node_rank=@@rank@@/2 master=@@master_addr@@:@@master_port@@
- GPUs=@@gpus@@ PP=@@pp@@ TP=@@tp@@
- steps=@@max_steps@@ micro_batches=@@micro_batches@@ seq_len=@@seq_len@@ seed=@@seed@@
- profiler: disable=false wait=@@prof_wait@@ warmup=0 active=@@prof_active@@ repeat=1
- generated by run_docker.py
PRE

echo "=== [XNODE] @@mode@@ node@@rank@@ start $(date -Is)" | "${TEE[@]}"
echo "    case_dir=$CASE_DIR" | "${TEE[@]}"

find "$REPO_ROOT/experiments/qwen14b/cache" -maxdepth 1 \
  -name '*_labels_unshifted_v2.pt' -size 0 -delete 2>/dev/null || true

cd "$REPO_ROOT"
torchrun \
  --nproc_per_node=@@gpus@@ \
  --nnodes=2 \
  --node_rank=@@rank@@ \
  --master_addr=@@master_addr@@ \
  --master_port=@@master_port@@ \
  "$ENTRY" \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --pp-size @@pp@@ \
  --tp-size @@tp@@ \
  --micro-batches @@micro_batches@@ \
  --per-device-batch-size @@micro_batches@@ \
  --seq-len @@seq_len@@ \
  --max-steps @@max_steps@@ \
  --log-interval 1 \
  --seed @@seed@@ \
  --lr 2e-4 \
  --dataset-name-or-path HuggingFaceFW/fineweb \
  --dataset-config @@dataset_config@@ \
  --text-field text \
  --run-label @@run_label@@ \
  @@method_args@@ \
  --num-workers 0 \
  --disable-profiler false \
  --profiler-wait-steps @@prof_wait@@ \
  --profiler-warmup-steps 0 \
  --profiler-active-steps @@prof_active@@ \
  --profiler-repeat 1 \
  --debug-nan-steps 2 \
  --save-dir "$CASE_DIR/ckpt" \
  --step-log-dir "$CASE_DIR/step_csv" \
  2>&1 | "${TEE[@]}"
RC=${PIPESTATUS[0]}
echo "EXIT_CODE=$RC" | tee -a "$CASE_DIR/EXIT_CODE" "$CASE_DIR/train.log" "$RUN_LOG"
exit "$RC"
"""


def build_script(mode, rank, args, case_dir, ts):
    prof_active = 2
    prof_wait = max(int(args.max_steps) - (prof_active + 1), 0)
    values = {
        "mode": mode,
        "rank": rank,
        "ts": ts,
        "repo": args.repo,
        "entry": ENTRY_SUBPATH,
        "case_dir": case_dir,
        "ifname": NODES[rank]["ifname"],
        "master_addr": NODES[0]["ip"],
        "master_port": args.master_port,
        "gpus": args.gpus,
        "pp": args.pp,
        "tp": args.tp,
        "max_steps": args.max_steps,
        "micro_batches": args.micro_batches,
        "seq_len": args.seq_len,
        "dataset_config": args.dataset_config,
        "seed": args.seed,
        "method_args": " \\\n  ".join(METHOD_ARGS[mode]),
        "run_label": RUN_LABELS[mode],
        "prof_wait": prof_wait,
        "prof_active": prof_active,
    }
    body = SCRIPT_TEMPLATE
    for key, val in values.items():
        body = body.replace(f"@@{key}@@", str(val))
    leftovers = re.findall(r"@@[a-z_]+@@", body)
    if leftovers:
        raise ValueError(f"模板里有未替换的占位符: {sorted(set(leftovers))}")
    return body


# ---------------------------------------------------------------- 远端执行

def _on(host, inner):
    """host=local 直接本地跑，否则 ssh 到 node1 的宿主机上跑。"""
    if host == LOCAL_HOST:
        return ["bash", "-lc", inner]
    return ["ssh", *SSH["opts"], SSH["target"], inner]


def _docker(host, container, inner):
    return _on(host, f"docker exec {container} bash -c {shlex.quote(inner)}")


def put_file(host, container, path, content):
    """把内容写到（远端）容器的 path 下。base64 避免多层引号展开。"""
    b64 = base64.b64encode(content.encode()).decode()
    inner = (f"mkdir -p {shlex.quote(str(Path(path).parent))} && "
             f"echo {b64} | base64 -d > {shlex.quote(str(path))} && "
             f"chmod +x {shlex.quote(str(path))}")
    subprocess.run(_docker(host, container, inner), check=True,
                   capture_output=True, text=True, timeout=120)


def read_remote(host, container, path):
    out = subprocess.run(_docker(host, container, f"cat {shlex.quote(str(path))} 2>/dev/null"),
                         capture_output=True, text=True, timeout=60)
    return out.stdout.strip()


def _spawn(argv, log_path, live):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "w", encoding="utf-8", buffering=1)
    if live:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, start_new_session=True)
    else:
        proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
    proc._log_fh = fh
    return proc


def stream_node(host, container, script, log_path, live=False):
    return _spawn(_docker(host, container, f"bash {shlex.quote(str(script))}"), log_path, live)


def pump(proc):
    """把 node0 的输出边打屏边落盘，直到进程结束。"""
    for line in proc.stdout:
        print(line, end="", flush=True)
        try:
            proc._log_fh.write(line)
            proc._log_fh.flush()
        except ValueError:            # 主线程已经关掉日志句柄（进程已结束）
            pass
    proc.wait()


def wait_proc(proc, timeout_s, label):
    try:
        return proc.wait(timeout=timeout_s), False
    except subprocess.TimeoutExpired:
        print(f"\n[{label}] 超过 {timeout_s}s 仍未结束，停止等待（不发任何信号）。")
        return None, True


# ---------------------------------------------------------------- 启动前检查

_KILL_SNIPPET = r"""
term=0
match() {
  cmd=$(tr '\0' ' ' < /proc/"$1"/cmdline 2>/dev/null) || return 1
  case "$cmd" in
    *run_qwen14b_polar_dp_pp_tp.py*) return 0;;
    *torchrun*run_qwen14b*) return 0;;
  esac
  return 1
}
list_pids() {
  for pid in $(ls /proc 2>/dev/null); do
    case "$pid" in ''|*[!0-9]*) continue;; esac
    [ "$pid" = "$$" ] && continue
    match "$pid" && echo "$pid"
  done
}
for pid in $(list_pids); do
  kill -TERM "$pid" 2>/dev/null && term=$((term + 1))
done
[ "$term" -gt 0 ] && sleep 8
hard=0
for pid in $(list_pids); do
  kill -KILL "$pid" 2>/dev/null && hard=$((hard + 1))
done
echo "term=$term kill=$hard"
"""


def preflight(args):
    """上机前检查：宿主机 docker、两台容器、ssh、仓库路径、可写性。"""
    print("=" * 78)
    print("启动前检查")
    print("=" * 78)
    problems = []

    def check(label, argv, essential=True, timeout=60):
        try:
            out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        except Exception as exc:                               # noqa: BLE001
            msg = f"{exc.__class__.__name__}: {exc}"
            out = None
        else:
            msg = (out.stdout + out.stderr).strip().splitlines()
            msg = msg[-1] if msg else ""
        ok = out is not None and out.returncode == 0
        print(f"  [{'OK ' if ok else 'FAIL'}] {label}" + ("" if ok else f"  -> {msg}"))
        if not ok and essential:
            problems.append(label)
        return ok

    if shutil.which("docker") is None:
        print("  [FAIL] 宿主机没有 docker 命令")
        print("\n这个脚本要在 powerone1 的**宿主机**上跑，不能在容器里跑：")
        print("  容器内没有 docker / /var/run/docker.sock，无法启动 node0，也起不了 node1。")
        print("  请在宿主机上执行：cd /data1/tangruijing/udtca/udtca-config/qwen14b && "
              "python3 run_docker.py")
        return False

    for rank in (0, 1):
        node = NODES[rank]
        check(f"node{rank} {node['name']} 容器 {args.container} 在运行",
              _on(node["host"], f"docker inspect -f '{{{{.State.Running}}}}' {args.container}"))
        check(f"node{rank} 容器里仓库存在 ({ENTRY_SUBPATH})",
              _docker(node["host"], args.container, f"test -f {args.repo}/{ENTRY_SUBPATH} "
                                                   f"&& echo ok"))
        check(f"node{rank} 容器里 run_log 可写",
              _docker(node["host"], args.container,
                      f"touch {args.repo}/run_log/.preflight && rm -f "
                      f"{args.repo}/run_log/.preflight && echo ok"))
        check(f"node{rank} 容器里能看到 8 张卡",
              _docker(node["host"], args.container,
                      "test $(ls /dev/ix* 2>/dev/null | wc -l) -ge 8 && echo ok || echo cards"),
              essential=False)

    if problems:
        print("\n有致命项没通过，先修好再跑：")
        for p in problems:
            print(f"  - {p}")
        if any("容器" in p and "node1" in p for p in problems):
            print("\nnode1 起不来最常见的原因是宿主机之间没有免密 ssh。检查：")
            print(f"  ssh -o BatchMode=yes {SSH['target']} hostname")
            print("  不通就先做免密：ssh-copy-id <node1>（或用 --ssh-target/--ssh-opts 指定）")
        return False
    print("\n检查全部通过。")
    return True


# ---------------------------------------------------------------- 中断清理

_STATE = {"args": None, "active": False, "proc0": None, "proc1": None, "cleaned": False}


def cleanup_nodes(args, reason=""):
    """中断/异常时把两台机器上的训练进程清掉（同 generate_and_run.py 的策略）。"""
    if _STATE["cleaned"] or args is None:
        return
    _STATE["cleaned"] = True
    print(f"\n[cleanup] {reason or '退出'}：清理两台机器上的训练进程")
    print("          （先 SIGTERM，8s 后仍在的 SIGKILL；Corex 上打断在途 NCCL 有 XID 24 风险）")
    for rank in (0, 1):
        node = NODES[rank]
        try:
            out = subprocess.run(_docker(node["host"], args.container, _KILL_SNIPPET),
                                 capture_output=True, text=True, timeout=120)
            print(f"  node{rank} {node['name']}: {out.stdout.strip() or out.stderr.strip()}")
        except Exception as exc:                               # noqa: BLE001
            print(f"  node{rank} {node['name']}: 清理失败 {exc}")


def _on_signal(signum, _frame):
    args = _STATE["args"]
    if _STATE["active"]:
        cleanup_nodes(args, f"收到信号 {signum}")
        _STATE["active"] = False
    sys.exit(130)


def install_signal_handlers(args):
    _STATE["args"] = args
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)


# ---------------------------------------------------------------- 结果解析

_LOSS_RE = re.compile(r"Step (\d+), Loss:\s*([0-9.]+)")


def read_text(host, container, path):
    """统一走 docker exec cat：宿主机上的路径不一定是容器里的那个（如 /data1 与 /data）。"""
    return read_remote(host, container, path)


def _losses_from_text(text):
    rows = []
    for line in text.splitlines():
        m = _LOSS_RE.search(line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2))))
    return rows


def parse_step_csv(host, container, case_dir):
    """step_csv 在容器里，从容器里 cat 出来再解析（宿主机未必能直接看到该路径）。"""
    csv_dir = Path(case_dir) / "step_csv"
    inner = (f"cd {shlex.quote(str(csv_dir))} 2>/dev/null && "
             f"for f in *.csv; do [ -f \"$f\" ] && cat \"$f\"; done || true")
    out = subprocess.run(_docker(host, container, inner),
                         capture_output=True, text=True, timeout=120)
    per_step = {}
    for row in csv.DictReader(io.StringIO(out.stdout)):
        try:
            step = int(float(row["step"]))
            t = float(row["step_time_s"])
        except (KeyError, TypeError, ValueError):
            continue
        per_step[step] = max(per_step.get(step, 0.0), t)
    if not per_step:
        return {}
    times = [per_step[s] for s in sorted(per_step)]
    return {"num_steps": len(times), "avg_step_s": sum(times) / len(times),
            "total_step_s": sum(times), "per_step": times}


def find_traces(host, container, repo, since_epoch):
    inner = (f"find {repo}/log -name '*.pt.trace.json' -newermt @{int(since_epoch)} "
             f"-printf '%s %p\\n' 2>/dev/null | sort -k2")
    out = subprocess.run(_docker(host, container, inner),
                         capture_output=True, text=True, timeout=180)
    rows = []
    for line in out.stdout.splitlines():
        size, _, path = line.partition(" ")
        if path:
            rows.append({"path": path, "mb": round(int(size) / 1e6, 1)})
    return rows


# ---------------------------------------------------------------- trace 分析
#
# 在容器里跑（trace 有几 GB，不搬回宿主机）。输出 JSON，宿主机侧汇总成表格。

ANALYZE_CODE = r"""
import json, sys

def iter_events(path):
    text = open(path, encoding="utf-8", errors="ignore").read()
    i = text.index('"traceEvents"')
    i = text.index("[", i + len('"traceEvents"'))
    dec = json.JSONDecoder()
    # 快路径：整个数组一次解出来；尾部被 encoder 写坏时退回逐条 raw_decode。
    try:
        obj, _ = dec.raw_decode(text, i)
    except (json.JSONDecodeError, MemoryError):
        pass
    else:
        yield from walk(obj)
        return
    i += 1
    n = len(text)
    while True:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n or text[i] == "]":
            break
        try:
            obj, i = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            break
        yield from walk(obj)

def walk(obj):
    if isinstance(obj, dict):
        yield obj
    elif isinstance(obj, list):
        for sub in obj:
            yield from walk(sub)

def merge_len(iv):
    if not iv:
        return 0.0
    iv.sort()
    total = 0.0
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return total + (ce - cs)

def cat_of(name):
    low = name.lower()
    if "sendrecv" in low or "send_recv" in low:
        return "sendrecv"
    if "allreduce" in low or "all_reduce" in low:
        return "allreduce"
    if "nccl" in low or "ixccl" in low:
        return "comm_other"
    if "gemm" in low:
        return "gemm"
    if "elementwise" in low:
        return "elementwise"
    if "reduce_kernel" in low:
        return "reduce"
    if "flash" in low or "attn" in low or "softmax" in low:
        return "attn"
    return "other"

def analyze(path):
    steps, kern, ann = [], [], {}
    for ev in iter_events(path):
        name = ev.get("name", "")
        ts = ev.get("ts")
        dur = ev.get("dur") or 0.0
        if ts is None:
            continue
        if name.startswith("ProfilerStep"):
            steps.append([name, ts, ts + dur])
        elif ev.get("cat") == "kernel":
            kern.append((ts, ts + dur, name))
        elif name in ("dp_pp_metric/DP", "dp_pp_metric/PP"):
            ann.setdefault(name, []).append((ts, ts + dur))
    steps.sort(key=lambda x: x[1])
    out = {"path": path, "steps": [], "n_steps": len(steps)}
    for nm, s, e in steps:
        w = [(a, b, n) for a, b, n in kern if s <= a < e]
        per = {}
        for a, b, n in w:
            per[cat_of(n)] = per.get(cat_of(n), 0.0) + (b - a)
        busy = merge_len([(a, b) for a, b, _ in w])
        rec = {"name": nm, "window_s": (e - s) / 1e6, "busy_s": busy / 1e6,
               "kernels": len(w), "cat_s": {k: v / 1e6 for k, v in per.items()}}
        for k, iv in ann.items():
            sel = [x for x in iv if s <= x[0] < e]
            rec[k] = merge_len(sel) / 1e6
        out["steps"].append(rec)
    return out

print(json.dumps([analyze(p) for p in sys.argv[1:] if p]))
"""


def analyze_node(args, host, container, traces):
    if not traces:
        return []
    paths = [t["path"] for t in traces]
    try:
        proc = subprocess.run(_on(host, f"docker exec -i {args.container} python3 - "
                                        + " ".join(shlex.quote(p) for p in paths)),
                              input=ANALYZE_CODE, capture_output=True, text=True, timeout=1800)
    except Exception as exc:                                   # noqa: BLE001
        print(f"  [分析失败] {host}: {exc}")
        return []
    if proc.returncode != 0:
        print(f"  [分析失败] {host}: {(proc.stderr or '').strip().splitlines()[-1:] }")
        return []
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(f"  [分析失败] {host}: 输出不是 JSON")
        return []


def _agg_step(step_reports):
    """把同一 mode 多个 trace 的稳态步平均掉（第一个步是暖机，剔除）。"""
    steady = [s for r in step_reports for s in r["steps"][1:]] or \
             [s for r in step_reports for s in r["steps"]]
    if not steady:
        return None
    n = len(steady)
    cats = {}
    for s in steady:
        for k, v in s["cat_s"].items():
            cats[k] = cats.get(k, 0.0) + v
    return {
        "n": n,
        "window_s": sum(s["window_s"] for s in steady) / n,
        "busy_s": sum(s["busy_s"] for s in steady) / n,
        "kernels": sum(s["kernels"] for s in steady) // n,
        "cat_s": {k: v / n for k, v in cats.items()},
        "dp_s": sum(s.get("dp_pp_metric/DP", 0.0) for s in steady) / n,
        "pp_s": sum(s.get("dp_pp_metric/PP", 0.0) for s in steady) / n,
    }


def analysis_report(rows, args):
    """对每个 mode 的 trace 做分析，返回可直接打印/写日志的文本。"""
    lines = []
    add = lines.append
    add("=" * 78)
    add("训练性能分析（trace 口径：ProfilerStep 窗口内的 GPU kernel 累加）")
    add("=" * 78)
    for r in rows:
        add(f"\n## {r['mode']}   两端 EXIT_CODE: node0={r['exit0']} node1={r['exit1']}")
        agg_all = []
        for rank in (0, 1):
            traces = r["traces_by_node"][rank]
            if not traces:
                add(f"  node{rank} {NODES[rank]['name']}: 没有找到 trace")
                continue
            print(f"  [分析] {r['mode']} node{rank} {NODES[rank]['name']}: "
                  f"{len(traces)} 个 trace, "
                  f"{sum(t['mb'] for t in traces) / 1024:.1f} GB …", flush=True)
            reports = analyze_node(args, NODES[rank]["host"], args.container, traces)
            agg = _agg_step(reports)
            if agg is None:
                add(f"  node{rank} {NODES[rank]['name']}: trace 里没有 ProfilerStep")
                continue
            agg_all.append(agg)
            cats = agg["cat_s"]
            tot = sum(cats.values()) or 1.0
            add(f"  node{rank} {NODES[rank]['name']}  ({len(traces)} 个 trace, "
                f"稳态 {agg['n']} 步平均)")
            add(f"    步时长       {agg['window_s']:8.2f} s")
            add(f"    GPU busy     {agg['busy_s']:8.2f} s  "
                f"({100 * agg['busy_s'] / agg['window_s']:5.1f}%)   kernels/step={agg['kernels']}")
            add(f"    DP 同步标注  {agg['dp_s']:8.2f} s  "
                f"({100 * agg['dp_s'] / agg['window_s']:5.1f}%)")
            add(f"    PP 区间标注  {agg['pp_s']:8.2f} s  "
                f"({100 * agg['pp_s'] / agg['window_s']:5.1f}%)")
            order = ["sendrecv", "allreduce", "comm_other", "gemm", "elementwise",
                     "reduce", "attn", "other"]
            parts = [f"{k}={cats.get(k, 0.0):.2f}s" for k in order if cats.get(k, 0.0) > 0.005]
            add(f"    kernel 分类  " + "  ".join(parts))
            comm = sum(cats.get(k, 0.0) for k in ("sendrecv", "allreduce", "comm_other"))
            compute = sum(cats.get(k, 0.0) for k in ("gemm", "elementwise", "reduce", "attn"))
            add(f"    通信 vs 计算  comm={comm:.2f}s ({100 * comm / agg['window_s']:.1f}%)  "
                f"compute={compute:.2f}s ({100 * compute / agg['window_s']:.1f}%)")
        if r.get("avg_step_s"):
            add(f"  step_csv（最后一级 stage）: {r['num_steps']} 步, "
                f"均值 {r['avg_step_s']:.2f} s, 逐步 "
                + " ".join(f"{t:.1f}" for t in r.get("per_step", [])))
        if agg_all:
            best = min(a["window_s"] for a in agg_all)
            add(f"  → 本 mode 稳态步时长 {best:.2f} s（两节点取快的一侧）")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------- 单个 mode

def run_mode(mode, args, run_dir):
    ts = args._ts
    mode_dir = run_dir / mode
    scripts = run_dir / "scripts"

    print(f"\n{'=' * 78}\n{mode}: 生成脚本并启动\n{'=' * 78}")
    mode_dir.mkdir(parents=True, exist_ok=True)
    script_paths = {}
    for rank in (0, 1):
        node = NODES[rank]
        path = scripts / f"node{rank}_{mode}.sh"
        body = build_script(mode, rank, args, str(mode_dir / f"node{rank}"), ts)
        put_file(node["host"], args.container, path, body)
        script_paths[rank] = path
        print(f"  node{rank} {node['name']:10s} -> {path}")

    started = time.time()

    print(f"\n[1/3] 起 node1（后台，{NODES[1]['name']}）…")
    proc1 = stream_node(NODES[1]["host"], args.container, script_paths[1],
                        mode_dir / "node1.ssh.log", live=False)
    _STATE["proc1"] = proc1
    print(f"      输出: {mode_dir}/node1.ssh.log")

    print(f"[2/3] 等 {args.warmup_s}s 让 node1 进入 rendezvous …")
    time.sleep(args.warmup_s)

    if proc1.poll() is not None:
        print(f"\n[!] node1 在 warmup 期间就退出了，返回码 {proc1.returncode}。node1 日志最后几行：")
        tail = (mode_dir / "node1.ssh.log").read_text(errors="ignore").splitlines()[-20:]
        for line in tail:
            print("    " + line)
        print("    node0 不启动，跳过本 mode 剩下的内容。")
        return {"mode": mode, "case_dir": str(mode_dir), "started": started,
                "exit0": "not-started", "exit1": str(proc1.returncode),
                "node0": str(mode_dir / "node0"), "node1": str(mode_dir / "node1"),
                "timed_out": False, "node1_died_early": True}

    print("[3/3] 起 node0（前台，实时回显）…")
    proc0 = stream_node(NODES[0]["host"], args.container, script_paths[0],
                        mode_dir / "node0.ssh.log", live=True)
    _STATE["proc0"] = proc0
    threading.Thread(target=pump, args=(proc0,), daemon=True).start()
    rc0, timed_out0 = wait_proc(proc0, args.timeout_s, "node0")
    rc1, timed_out1 = wait_proc(proc1, args.timeout_s, "node1")
    _STATE["proc0"] = _STATE["proc1"] = None

    for p in (proc0, proc1):
        fh = getattr(p, "_log_fh", None)
        if fh:
            fh.close()

    exit0 = read_remote(NODES[0]["host"], args.container,
                        mode_dir / "node0" / "EXIT_CODE") or f"EXIT_CODE={rc0}"
    exit1 = read_remote(NODES[1]["host"], args.container,
                        mode_dir / "node1" / "EXIT_CODE") or f"EXIT_CODE={rc1}"
    return {
        "mode": mode,
        "case_dir": str(mode_dir),
        "started": started,
        "exit0": exit0.replace("EXIT_CODE=", "").strip() or str(rc0),
        "exit1": exit1.replace("EXIT_CODE=", "").strip() or str(rc1),
        "node0": str(mode_dir / "node0"),
        "node1": str(mode_dir / "node1"),
        "timed_out": timed_out0 or timed_out1,
        "node1_died_early": False,
    }


def collect(rec, args):
    node0 = Path(rec["node0"])
    node1 = Path(rec["node1"])
    losses0 = _losses_from_text(read_text(NODES[0]["host"], args.container, node0 / "train.log"))
    losses1 = _losses_from_text(read_text(NODES[1]["host"], args.container, node1 / "train.log"))
    step = parse_step_csv(NODES[0]["host"], args.container, node0)

    dp_world = 2
    thr = None
    if step.get("total_step_s"):
        thr = step["num_steps"] * dp_world * args.micro_batches * args.seq_len / step["total_step_s"]

    traces = {0: find_traces(NODES[0]["host"], args.container, args.repo, rec["started"]),
              1: find_traces(NODES[1]["host"], args.container, args.repo, rec["started"])}
    return {
        **rec,
        "final_loss_node0": losses0[-1][1] if losses0 else None,
        "final_loss_node1": losses1[-1][1] if losses1 else None,
        "num_steps": step.get("num_steps"),
        "avg_step_s": step.get("avg_step_s"),
        "total_step_s": step.get("total_step_s"),
        "per_step": step.get("per_step"),
        "throughput_tok_s": thr,
        "traces_by_node": traces,
        "traces": traces[0] + traces[1],
        "trace_mb": round(sum(t["mb"] for t in traces[0] + traces[1]), 1),
        "mode_dir": str(node0.parent),
    }


# ---------------------------------------------------------------- 汇总输出

def _fmt(v, nd=2):
    return f"{v:.{nd}f}" if v is not None else "-"


def summary_table(rows):
    hdr = (f"{'mode':10} {'exit0':>9} {'exit1':>9} {'loss0':>8} {'loss1':>8} "
           f"{'steps':>5} {'avg_step_s':>10} {'tok/s':>9} {'traces':>6} {'GB':>7}")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(
            f"{r['mode']:10} {str(r['exit0']):>9} {str(r['exit1']):>9} "
            f"{_fmt(r['final_loss_node0'], 4):>8} {_fmt(r['final_loss_node1'], 4):>8} "
            f"{str(r['num_steps'] or '-'):>5} {_fmt(r['avg_step_s']):>10} "
            f"{_fmt(r['throughput_tok_s'], 1):>9} "
            f"{len(r['traces']):>6} {_fmt(r['trace_mb'] / 1024):>7}"
        )
    return "\n".join(lines)


def compare(rows):
    by = {r["mode"]: r for r in rows}
    b, m = by.get("baseline"), by.get("mixed")
    out = []
    if b and m and b.get("avg_step_s") and m.get("avg_step_s"):
        out.append(f"步时   baseline={b['avg_step_s']:.2f}s  mixed={m['avg_step_s']:.2f}s  "
                   f"加速={b['avg_step_s'] / m['avg_step_s']:.2f}x")
    if b and m and b.get("throughput_tok_s") and m.get("throughput_tok_s"):
        out.append(f"吞吐   baseline={b['throughput_tok_s']:.1f} tok/s  "
                   f"mixed={m['throughput_tok_s']:.1f} tok/s  "
                   f"ratio={m['throughput_tok_s'] / b['throughput_tok_s']:.2f}x")
    if b and m:
        bit = [f"loss baseline={_fmt(b.get('final_loss_node0'), 4)} "
               f"mixed={_fmt(m.get('final_loss_node0'), 4)}"]
        out.extend(bit)
    if not out:
        out.append("（缺少 baseline 或 mixed 的步时数据，不做对比）")
    return "\n".join(out)


# ---------------------------------------------------------------- main

def make_arg_parser():
    p = argparse.ArgumentParser(
        description="跨机 Qwen14B：mixed/baseline 各 10 iter + profiler 8/9 iter + 自动分析",
        epilog="必须在 powerone1 的宿主机上运行（容器里没有 docker）。")
    p.add_argument("--modes", default="mixed,baseline",
                   help="逗号分隔，可选 baseline|polar_dense|bitscom|mixed（按顺序跑）")
    p.add_argument("--repo", default=DEFAULT_REPO, help="两台机器上相同的仓库根路径")
    p.add_argument("--container", default="trj-udtca")
    p.add_argument("--ssh-target", default=SSH["target"], help="ssh 到 node1 的目标（默认 powerone2）")
    p.add_argument("--ssh-opts", default="",
                   help='ssh 参数（给了就整体替换默认的 "-o BatchMode=yes -o ConnectTimeout=10"），'
                        '如 "-p 7122 -i /root/.ssh/id_rsa"')
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--micro-batches", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--gpus", type=int, default=8, help="每台机器的卡数")
    p.add_argument("--pp", type=int, default=4)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--master-port", type=int, default=29630)
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--warmup-s", type=int, default=25, help="node1 起来后等多少秒再起 node0")
    p.add_argument("--timeout-s", type=int, default=None,
                   help="单节点最长等待（默认不限；到点只停止等待，不杀进程）")
    p.add_argument("--no-analyze", action="store_true", help="跑完不做 trace 分析")
    p.add_argument("--skip-preflight", action="store_true", help="跳过启动前检查")
    p.add_argument("--preflight-only", action="store_true", help="只做启动前检查")
    p.add_argument("--dry-run", action="store_true", help="只打印生成的脚本")
    return p


def main():
    args = make_arg_parser().parse_args()
    args._ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    SSH["target"] = args.ssh_target
    if args.ssh_opts.strip():
        SSH["opts"] = shlex.split(args.ssh_opts)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in METHOD_ARGS:
            print(f"未知 mode: {m}", file=sys.stderr)
            return 2

    run_dir = Path(args.repo) / "run_log" / f"xnode_{args._ts}"
    print("=" * 78)
    print(f"跨机训练  modes={modes}  steps={args.max_steps}  "
          f"GPUS={args.gpus}/节点 DP={2 * args.gpus // (args.pp * args.tp)} "
          f"PP={args.pp} TP={args.tp}  profiler=第 8/9 iter")
    print(f"  node0 {NODES[0]['name']}(本机) {NODES[0]['ip']}   "
          f"node1 {NODES[1]['name']} {NODES[1]['ip']}   容器={args.container}")
    print(f"  master {NODES[0]['ip']}:{args.master_port}    run_dir {run_dir}")
    print("=" * 78)

    if args.dry_run:
        for mode in modes:
            for rank in (0, 1):
                print(f"\n----- {mode} node{rank} -----")
                print(build_script(mode, rank, args,
                                   str(run_dir / mode / f"node{rank}"), args._ts))
        return 0

    if not args.skip_preflight and not preflight(args):
        return 2
    if args.preflight_only:
        return 0

    install_signal_handlers(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for mode in modes:
            _STATE["active"] = True
            rec = run_mode(mode, args, run_dir)
            rows.append(collect(rec, args))
            _STATE["active"] = False
    except BaseException as exc:                               # noqa: BLE001
        _STATE["active"] = False
        cleanup_nodes(args, f"异常 {exc.__class__.__name__}: {exc}")
        raise

    table = summary_table(rows)
    cmp_text = compare(rows)
    analysis = "" if args.no_analyze else analysis_report(rows, args)

    print(f"\n{'=' * 78}\n汇总\n{'=' * 78}")
    print(table, "\n")
    print(cmp_text)
    if analysis:
        print("\n" + analysis)

    header = (f"# 跨机 Qwen14B 汇总\n\nrun_dir: {run_dir}\n\n"
              f"## 汇总表\n```\n{table}\n```\n\n## 对比\n```\n{cmp_text}\n```\n")
    details = []
    for r in rows:
        details.append(f"\n## {r['mode']}\n- 退出码 node0={r['exit0']} node1={r['exit1']}")
        details.append(f"- case: {r['case_dir']}")
        for t in r["traces"]:
            details.append(f"- trace {t['mb']} MB: {t['path']}")
    md = header + "\n## 分析\n```\n" + analysis + "\n```\n" + "\n".join(details) + "\n"

    (run_dir / "summary.json").write_text(
        json.dumps({"run_dir": str(run_dir), "modes": rows, "compare": cmp_text},
                   indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(md, encoding="utf-8")
    if analysis:
        (run_dir / "analysis.log").write_text(analysis + "\n", encoding="utf-8")

    print(f"\n结果已写入:")
    print(f"  {run_dir}/summary.md")
    print(f"  {run_dir}/summary.json")
    if analysis:
        print(f"  {run_dir}/analysis.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
