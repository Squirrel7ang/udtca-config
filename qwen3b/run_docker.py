#!/usr/bin/env python3
"""跨机 Qwen2.5-3B 端到端训练 + profiler 采集 + 自动分析（容器版）。

**必须在 powerone1 的宿主机上运行**（容器里没有 docker / docker.sock，脚本会直接拒绝启动）：
  * node0 = 本机，宿主机执行 `docker exec trj-udtca ...`
  * node1 = powerone2，宿主机执行 `ssh <node1> 'docker exec trj-udtca ...'`

每个 mode（默认 polar 先、baseline 后）跑之前先过一道卡检查（见 card_precheck）：
  * 占用：两台节点容器里 `ixsmi --query-gpu/--query-compute-apps`，有人在用就不启动
  * 健康：宿主机 `journalctl -k -b` 里出现 page fault / vmid（Corex XID 24）就不启动
  任一台不干净，本轮（含后续 mode）直接跳过，不启动训练。

每个 mode 依次做一遍：

  1. 在 run 目录下给两台机器各生成一份训练脚本
     `<run>/scripts/node{0,1}_<mode>.sh`（自包含：环境、METHOD_ARGS、torchrun 全在里面）
  2. 先起 node1（后台，输出落 node1.ssh.log），warmup 后起 node0（实时回显 + 落盘）
  3. 等两边退出，收集 train.log / step_csv / [rt-metric] / NCCL 日志
     （trace 一律不查：Corex 上 profiler 跑不出 trace，分析只按 step_csv 口径）
  4. 全部跑完自动分析：
     * 步时口径：step_csv 里最后一级 stage 的逐步耗时；
     * 实时口径：每步 `[rt-metric]` 的 DP-PP 重叠率、pp_ms/dp_ms、通信
       dense/compressed/saved、压缩比（只有开 polar 的 mode 有）；
     * 通信量：`logs/nccl_*.log` 里 NCCL 自报的集合通信 payload 字节，两节点合计
       （口径同 udtca-config/utils/parse_nccl_bytes.py）。
     结果同时打到终端和 `<run>/analysis.log`、`<run>/metrics.log`、`<run>/summary.md`。
     baseline vs polar 的对比里包含步时/吞吐、端到端通信量与降低倍数、重叠率、压缩比。

profiler：默认关闭（`--disable-profiler true`）。Corex 上开 profiler 会让第 1 步
卡在 CUPTI attach（此前 10 iter 复跑即卡死），稳定性优先。确实要抓 trace 时用
`--enable-profiler`，此时按 `--profiler-active-steps`（默认 2）在最后两步采样；
但跑完的分析**不读 trace**，无论 trace 有没有生成都只按 step_csv 口径出结论。
不做 tc 限速。

中途退出（Ctrl-C / 异常）会像 generate_and_run.py 那样把两台机器上的训练进程都清掉：
按「命令行里带 run_qwen3b_polar_dp_pp_tp.py 或 torchrun」精确匹配，先 SIGTERM，
8 s 后仍在的再 SIGKILL。注意 Corex 上打断在途 NCCL 有 XID 24 风险，这是你要的行为。

用法：
    python3 run_docker.py                      # polar 先跑、再 baseline，各 5 iter，不开 profiler
    python3 run_docker.py --only-baseline      # 只跑 baseline，不跑 polar
    python3 run_docker.py --modes polar        # 只跑 polar
    python3 run_docker.py --max-steps 3        # 冒烟
    python3 run_docker.py --enable-profiler    # 采 trace（默认不开，分析也不读 trace）
    python3 run_docker.py --dry-run            # 只打印生成的脚本，什么都不启动
    python3 run_docker.py --preflight-only     # 只检查两台机器/容器/ssh/卡占用/卡健康
    python3 run_docker.py --skip-card-check    # 卡检查有历史日志噪音时临时跳过
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
ENTRY_SUBPATH = "experiments/qwen3b/run_qwen3b_polar_dp_pp_tp.py"

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
    "polar": "polar_bitscom_xnode",
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
    "polar": [
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
# orchestrator 的 NCCL 集合通信日志）。排查性能瓶颈期间只留这些，不额外加诊断变量。

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

# ---- 通信（与 cloud/run_xnode.sh、成功 run 的 PRE_RUN 一致）----
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=@@ifname@@
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SOCKET_RETRY_CNT="${NCCL_SOCKET_RETRY_CNT:-600}"
export NCCL_SOCKET_RETRY_SLEEP_MSEC="${NCCL_SOCKET_RETRY_SLEEP_MSEC:-1000}"

# ---- 数据 ----
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY NO_PROXY no_proxy
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-15}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-30}"

# ---- 输出 / 诊断 ----
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_CPP_LOG_LEVEL="${TORCH_CPP_LOG_LEVEL:-WARNING}"
export KINETO_LOG_LEVEL="${KINETO_LOG_LEVEL:-5}"
export TORCH_DISABLE_ADDR2LINE=1
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-off}"
# torch 的 NCCL watchdog 会 abort 在途通信（Corex 上就是 XID 24 的路径），保持关闭。
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-0}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_FR_BUFFER_SIZE=1048576
# 通信超时基线；卡住时由外部 kill -9 处理，torch 自己不 abort。跨机首步慢，放宽到 600 s。
export UDTCA_COMM_WATCHDOG_S="${UDTCA_COMM_WATCHDOG_S:-600}"
export UDTCA_HTTP_READ_TIMEOUT_S="${UDTCA_HTTP_READ_TIMEOUT_S:-30}"
export UDTCA_DATASET_RETRIES="${UDTCA_DATASET_RETRIES:-3}"
export UDTCA_DATASET_STALL_S="${UDTCA_DATASET_STALL_S:-120}"

# 低比特 DP 同步严格串行（同 case_common：多 bucket 同时在飞会让两个 DP rank 的
# p2p 投递顺序分叉，偶发卡死）。
export POLAR_EF_LOWMEM_SERIAL_BUCKETS="${POLAR_EF_LOWMEM_SERIAL_BUCKETS:-1}"

# ---- NCCL 集合通信日志（同 orchestrator._nccl_debug_env，subsys 对齐成功 run）----
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG_FILE="$CASE_DIR/logs/nccl_%h_%p.log"

cat > "$CASE_DIR/PRE_RUN.md" <<PRE
# PRE_RUN @@ts@@
- mode=@@mode@@ node_rank=@@rank@@/2 master=@@master_addr@@:@@master_port@@
- GPUs=@@gpus@@ PP=@@pp@@ TP=@@tp@@
- steps=@@max_steps@@ micro_batches=@@micro_batches@@ seq_len=@@seq_len@@ seed=@@seed@@
- profiler: disable=@@prof_disable@@ wait=@@prof_wait@@ warmup=@@prof_warmup@@ active=@@prof_active@@ repeat=1
- nccl: debug=$NCCL_DEBUG subsys=$NCCL_DEBUG_SUBSYS ib_disable=$NCCL_IB_DISABLE ifname=$NCCL_SOCKET_IFNAME p2p_disable=$NCCL_P2P_DISABLE
- torch: cpp_log=$TORCH_CPP_LOG_LEVEL dist_debug=$TORCH_DISTRIBUTED_DEBUG faulthandler=$PYTHONFAULTHANDLER
- generated by run_docker.py
PRE

echo "=== [XNODE] @@mode@@ node@@rank@@ start $(date -Is)" | "${TEE[@]}"
echo "    case_dir=$CASE_DIR" | "${TEE[@]}"

find "$REPO_ROOT/experiments/qwen3b/cache" -maxdepth 1 \
  -name '*_labels_unshifted_v2.pt' -size 0 -delete 2>/dev/null || true

cd "$REPO_ROOT"
torchrun \
  --nproc_per_node=@@gpus@@ \
  --nnodes=2 \
  --node_rank=@@rank@@ \
  --master_addr=@@master_addr@@ \
  --master_port=@@master_port@@ \
  "$ENTRY" \
  --model-name Qwen/Qwen2.5-3B-Instruct \
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
  --disable-profiler @@prof_disable@@ \
  --profiler-wait-steps @@prof_wait@@ \
  --profiler-warmup-steps @@prof_warmup@@ \
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
    if args.enable_profiler:
        prof_disable = "false"
        prof_warmup = 0
        prof_active = int(args.profiler_active_steps)
        prof_wait = max(int(args.max_steps) - (prof_active + 1), 0)
    else:
        # profiler 关掉；entry 仍校验 active>0，所以给 1 这个占位值。
        prof_disable = "true"
        prof_wait = 0
        prof_warmup = 0
        prof_active = 1
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
        "prof_disable": prof_disable,
        "prof_wait": prof_wait,
        "prof_warmup": prof_warmup,
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


def _tail_rt_lines(host, container, log_path, n=3):
    """取日志里最后 n 条 [rt-metric]（tail 只读文件尾，不会把整份日志拖过来）。"""
    inner = (f"grep '\\[rt-metric\\]' {shlex.quote(str(log_path))} 2>/dev/null | tail -n {n}")
    try:
        out = subprocess.run(_docker(host, container, inner), capture_output=True,
                             text=True, timeout=30)
    except Exception:                                          # noqa: BLE001
        return []
    return [ln for ln in out.stdout.splitlines() if "[rt-metric]" in ln]


def live_metrics_loop(mode, container, node_dirs, stop_event, poll_s=20.0):
    """训练进行中，每隔 poll_s 把两台新出的 [rt-metric] 摘要打到终端。

    数据本来每步就落进 train.log（print flush + tee），这里只是把它压缩成一行打出来，
    让「跑到第几步、重叠率多少、压缩多少」在训练当中就能看到，而不用等整轮跑完。
    """
    last_step = {0: 0, 1: 0}
    while not stop_event.wait(poll_s):
        for rank in (0, 1):
            lines = _tail_rt_lines(NODES[rank]["host"], container,
                                   Path(node_dirs[rank]) / "train.log", n=5)
            for line in lines:
                kv = dict(_KV_RE.findall(line.split("[rt-metric]", 1)[1]))
                try:
                    step = int(kv.get("step", "0"))
                except ValueError:
                    continue
                if step <= last_step[rank]:          # 已经打过的步
                    continue
                last_step[rank] = step
                bits = [f"step={step}", f"loss={kv.get('loss', '?')}"]
                if kv.get("dp_pp_overlap"):
                    bits.append(f"DP∩PP={kv['dp_pp_overlap']}")
                if kv.get("comm_reduction"):
                    bits.append(f"压缩={kv['comm_reduction']}"
                                f" (dense {kv.get('comm_dense', '?')} → "
                                f"{kv.get('comm_compressed', '?')})")
                print(f"  [metric] {mode} node{rank} " + "  ".join(bits), flush=True)


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
    *run_qwen3b_polar_dp_pp_tp.py*) return 0;;
    *torchrun*run_qwen3b*) return 0;;
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
        print("  请在宿主机上执行：cd /data1/tangruijing/udtca/udtca-config/qwen3b && "
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
        check(f"node{rank} 容器里能看到 {args.gpus} 张卡",
              _docker(node["host"], args.container,
                      f"test $(ls /dev/ix* 2>/dev/null | wc -l) -ge {args.gpus} "
                      f"&& echo ok || echo cards"),
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

# ---------------------------------------------------------------- 卡占用 / 卡健康检查
#
# 两道闸，都在启动训练之前：
#   1) 占用：两台节点各自的容器里跑 ixsmi，有显存超阈值或 --query-compute-apps
#      非空，就认为卡被占着（别人在用、或者上一轮没退干净），本次不启动训练。
#      对齐 udtca-config/common/case_common.py 的 check_gpus_free()。
#   2) 健康：仿照 powerone1:~/check.sh，宿主机本 boot 的内核日志里只要有
#      "page fault" 或 "vmid" 就认为卡不干净（Corex 的 XID 24 MMU 挂死就是这么报的）。

# 空载时每张卡约 68 MiB（驱动本身），超过阈值就认为有人在用
GPU_MEM_THRESHOLD_MIB = 256

GPU_OCCUPANCY_SNIPPET = r"""
THRESHOLD=__THRESHOLD__
export LD_LIBRARY_PATH=/usr/local/corex/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

SMI=""
for c in ixsmi /usr/local/bin/ixsmi /usr/local/corex/bin/ixsmi \
         /data/tangruijing/udtca/.corex-4.5.0/bin/ixsmi /usr/local/corex-4.5.0/bin/ixsmi nvidia-smi; do
  if command -v "$c" >/dev/null 2>&1; then SMI="$c"; break; fi
done
if [ -z "$SMI" ]; then
  echo "NO_SMI"
  exit 2
fi

BUSY=$("$SMI" --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
  | awk -F, -v t="$THRESHOLD" '{
      gsub(/[ \t]/,"",$1); gsub(/[ \t]/,"",$2);
      if ($1 != "" && $2+0 > t) printf "GPU%s(%sMiB) ", $1, $2 }')
APPS=$("$SMI" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
  | sed '/^[[:space:]]*$/d')

if [ -n "$BUSY" ] || [ -n "$APPS" ]; then
  echo "BUSY $SMI"
  [ -n "$BUSY" ] && echo "  超过 ${THRESHOLD}MiB 的卡: $BUSY"
  [ -n "$APPS" ] && { echo "  正在跑的进程:"; echo "$APPS" | cut -c1-200 | sed 's/^/    /'; }
  exit 1
fi

echo "FREE $SMI"
exit 0
"""

# 只在宿主机上跑（容器里读不到 journald）；读不到内核日志时返回 2，不阻塞
HEALTH_SNIPPET = r"""
N=$(journalctl -k -b 2>/dev/null | wc -l)
if [ "$N" -lt 50 ]; then
  echo "NO_JOURNALD"
  exit 2
fi
FAULTS=$(journalctl -k -b 2>/dev/null | grep -i "page fault")
VMIDS=$(journalctl -k -b 2>/dev/null | grep -i "vmid")
if [ -n "$FAULTS$VMIDS" ]; then
  echo "DIRTY"
  [ -n "$FAULTS" ] && { echo "  page fault:"; echo "$FAULTS" | cut -c1-220 | tail -10 | sed 's/^/    /'; }
  [ -n "$VMIDS" ] && { echo "  vmid:"; echo "$VMIDS" | cut -c1-220 | tail -10 | sed 's/^/    /'; }
  exit 1
fi
echo "CLEAN"
exit 0
"""


def _run_capture(argv, timeout=90):
    """跑一条命令，返回 (returncode | None, 输出文本)。异常只当作失败，不抛。"""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:                                   # noqa: BLE001
        return None, f"{exc.__class__.__name__}: {exc}"
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _print_block(text, indent="      "):
    for line in text.splitlines():
        print(indent + line)


def check_card_occupancy(args, node):
    """在一台节点的容器里看卡有没有被占用。返回 'free' / 'busy' / 'unknown'。"""
    snippet = GPU_OCCUPANCY_SNIPPET.replace("__THRESHOLD__", str(args.gpu_mem_threshold))
    rc, out = _run_capture(_docker(node["host"], args.container, snippet))
    if rc == 0:
        print(f"    [OK  ] 空闲（{out.splitlines()[0] if out else 'FREE'}）")
        return "free"
    if rc == 1:
        lines = out.splitlines()
        print(f"    [占用] {lines[0] if lines else 'BUSY'}")
        _print_block("\n".join(lines[1:]))
        return "busy"
    if rc == 2:
        print(f"    [警告] 没找到 ixsmi，跳过占用检查（{out}）")
        return "unknown"
    print(f"    [警告] 占用检查执行失败（rc={rc}）：{out}")
    return "unknown"


def check_card_health(node):
    """在宿主机上看本 boot 内核日志有没有 page fault / vmid。返回 'clean'/'dirty'/'unknown'。"""
    rc, out = _run_capture(_on(node["host"], HEALTH_SNIPPET))
    if rc == 0:
        print("    [OK  ] 内核日志无 page fault / vmid")
        return "clean"
    if rc == 1:
        lines = out.splitlines()
        print(f"    [异常] {lines[0] if lines else 'DIRTY'}")
        _print_block("\n".join(lines[1:]))
        return "dirty"
    if rc == 2:
        print("    [警告] 读不到内核日志（需要 systemd-journald 读权限），跳过健康检查")
        return "unknown"
    print(f"    [警告] 健康检查执行失败（rc={rc}）：{out}")
    return "unknown"


def card_precheck(args, when=""):
    """启动训练前的两道闸：卡有没有被占用 + 卡有没有报错。

    任一台节点「被子占用」或「内核日志里有 page fault / vmid」就返回 False，
    调用方不要再启动训练。
    """
    title = "启动前检查：卡占用 / 卡健康" + (f"（{when}）" if when else "")
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)
    problems = []
    for rank in (0, 1):
        node = NODES[rank]
        print(f"  node{rank} {node['name']}:")
        occ = check_card_occupancy(args, node)
        health = check_card_health(node)
        if occ == "busy":
            problems.append(f"node{rank} {node['name']} 的卡正被占用")
        if health == "dirty":
            problems.append(f"node{rank} {node['name']} 内核日志里有 page fault / vmid")
    if problems:
        print("\n[卡不可用] 以下问题没解决，本次不启动训练：")
        for p in problems:
            print(f"  - {p}")
        print("\n处理建议：")
        print("  · 被占用：等占用进程结束再重跑；若是上一轮遗留的训练，先确认它真的停了")
        print("    （ps -ef | grep run_qwen3b_polar_dp_pp_tp），不要在 NCCL 在途时强杀。")
        print("  · page fault / vmid：Corex 上通常是 XID 24 MMU 挂死，只能重启机器恢复；")
        print("    先看 ixsmi 与 journalctl -k -b | grep -E 'page fault|vmid' 确认是哪张卡。")
        print("  · 确认无误、只是历史日志噪音时，可用 --skip-card-check 跳过这两道闸。")
        return False
    print("  卡检查通过。")
    return True


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


# ---------------------------------------------------------------- 实时指标（[rt-metric]）
#
# polar-sgd 的 wrapper._emit_rt_metric 每步打一行（不受 --log-interval 限制），
# 是 DP/PP 重叠率和通信压缩量的**在线**口径，和 run.py / metrics_api.py 同一个源：
#
#   [rt-metric] step=3 loss=11.7841 pp_ms=1234.50 dp_ms=321.00 dp_pp_overlap=21.3%
#               comm_dense=2.4303GB comm_compressed=313.2745MB
#               comm_saved=2.1170GB comm_reduction=7.76x
#
# 只有开了 polar 的 mode（polar_dense / polar）才有这行；baseline / bitscom 没有，
# 这时不能当成 0，要如实报「未测到」。

_RT_RE = re.compile(r"\[rt-metric\]\s+(.*)")
_KV_RE = re.compile(r"(\w+)=([^\s]+)")
_BYTE_UNITS = {"GB": 1e9, "MB": 1e6, "KB": 1e3, "B": 1.0}


def _parse_bytes(text):
    for unit, mul in _BYTE_UNITS.items():
        if text.endswith(unit):
            try:
                return float(text[: -len(unit)]) * mul
            except ValueError:
                return None
    return None


def parse_rt_metrics(text):
    """从 train.log 里抽出每一步的 [rt-metric] 记录（原样 kv，后面再换算）。"""
    rows = []
    for line in text.splitlines():
        m = _RT_RE.search(line)
        if not m:
            continue
        rec = dict(_KV_RE.findall(m.group(1)))
        rows.append(rec)
    return rows


def _rt_summary(rt):
    """把每步的 [rt-metric] 归并成均值/极值；没有该行就返回 None（不是 0）。"""
    if not rt:
        return None

    def vals(key, strip_pct=False):
        out = []
        for r in rt:
            v = r.get(key)
            if v is None:
                continue
            try:
                out.append(float(v.rstrip("%") if strip_pct else v.rstrip("x")))
            except ValueError:
                pass
        return out

    ov = vals("dp_pp_overlap", strip_pct=True)
    dense = [_parse_bytes(r["comm_dense"]) for r in rt if r.get("comm_dense")]
    comp = [_parse_bytes(r["comm_compressed"]) for r in rt if r.get("comm_compressed")]
    saved = [_parse_bytes(r["comm_saved"]) for r in rt if r.get("comm_saved")]
    dense = [v for v in dense if v is not None]
    comp = [v for v in comp if v is not None]
    saved = [v for v in saved if v is not None]
    mean = lambda xs: (sum(xs) / len(xs)) if xs else None  # noqa: E731
    return {
        "rt_steps": len(rt),
        "rt_overlap_mean": mean(ov) / 100.0 if ov else None,   # 0..1
        "rt_pp_ms_mean": mean(vals("pp_ms")),
        "rt_dp_ms_mean": mean(vals("dp_ms")),
        "rt_comm_dense_gb": (mean(dense) / 1e9) if dense else None,
        "rt_comm_compressed_gb": (mean(comp) / 1e9) if comp else None,
        "rt_comm_saved_gb": (mean(saved) / 1e9) if saved else None,
        "rt_comm_reduction_mean": mean(vals("comm_reduction")),
        "rt_steps_detail": [
            {
                "step": r.get("step"),
                "loss": r.get("loss"),
                "pp_ms": r.get("pp_ms"),
                "dp_ms": r.get("dp_ms"),
                "overlap": r.get("dp_pp_overlap"),
                "comm_dense": r.get("comm_dense"),
                "comm_compressed": r.get("comm_compressed"),
                "comm_saved": r.get("comm_saved"),
                "comm_reduction": r.get("comm_reduction"),
            }
            for r in rt
        ],
    }


# ---------------------------------------------------------------- NCCL 集合通信字节
#
# 口径与 udtca-config/utils/parse_nccl_bytes.py 一致：NCCL 的集合通信行（COLL 子
# 系统）都带 count 和 datatype，乘起来就是「交给网络的那块 buffer」的字节数。这里
# 用 SUBSYS=ALL 是为了诊断信息齐全，只匹配 "<Op> ... count ... datatype" 行，不受影响。把该 case
# 所有 rank 加起来 = 这一跑端到端真正通信的量（压缩过的 DP + 未压缩的 TP/PP 都在内）。
# 比读网卡计数器干净：k8s 节点上还有别的租户流量。
#
# 日志在容器里，脚本也在容器里跑，只把聚合结果带回来。

NCCL_COLL_CODE = r'''
import glob, json, re, sys
from collections import defaultdict

COLL_RE = re.compile(r"NCCL INFO (AllReduce|AllGather|ReduceScatter|AllToAll|"
                     r"Broadcast|Send|Recv|Reduce|Gather|Scatter)\b")
COUNT_RE = re.compile(r"\bcount (\d+)")
DTYPE_RE = re.compile(r"\bdatatype (\d+)")
DTYPE_BYTES = {0: 1, 1: 1, 2: 4, 3: 4, 4: 8, 5: 8, 6: 2, 7: 4, 8: 8, 9: 2}

paths = sorted(glob.glob(sys.argv[1] + "/logs/nccl*.log"))
total = 0
by_op = defaultdict(int)
unknown = 0
for path in paths:
    with open(path, errors="ignore") as fh:
        for line in fh:
            if "NCCL INFO" not in line:
                continue
            m = COLL_RE.search(line)
            if not m:
                continue
            mc = COUNT_RE.search(line)
            if not mc:
                continue
            count = int(mc.group(1))
            md = DTYPE_RE.search(line)
            size = DTYPE_BYTES.get(int(md.group(1)) if md else -1)
            if size is None:
                unknown += count
                continue
            nbytes = count * size
            total += nbytes
            by_op[m.group(1)] += nbytes

print(json.dumps({
    "logs": len(paths),
    "total_bytes": total,
    "unknown_elements": unknown,
    "by_op_gb": {k: round(v / 1e9, 6) for k, v in sorted(by_op.items(), key=lambda x: -x[1])},
}))
'''


def parse_nccl_bytes(host, container, case_dir):
    """在容器里扫 logs/nccl*.log，返回端到端集合通信字节量。扫不到就返回 logs=0。"""
    argv = _on(host, f"docker exec -i {container} python3 - {shlex.quote(str(case_dir))}")
    try:
        proc = subprocess.run(argv, input=NCCL_COLL_CODE, capture_output=True,
                              text=True, timeout=600)
    except Exception as exc:                                   # noqa: BLE001
        return {"error": f"{exc.__class__.__name__}: {exc}"}
    out = (proc.stdout or "").strip().splitlines()
    for line in reversed(out):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                break
    return {"error": (proc.stderr or "no output").strip().splitlines()[-1:] or "no output"}


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
    add("训练性能分析（step_csv 口径：不读 trace）")
    add("=" * 78)
    for r in rows:
        add(f"\n## {r['mode']}   两端 EXIT_CODE: node0={r['exit0']} node1={r['exit1']}")
        add("  trace: 不采集也不分析（Corex 上 profiler 出不了 trace）")
        agg_all = []
        for rank in (0, 1):
            traces = r["traces_by_node"][rank]
            if not traces:
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
            # 重叠率按 utils/overlap_interval.py 的口径：|DP∩PP| / (|DP|+|PP|-|DP∩PP|)。
            # 这里只有标注区间的累计秒数，交集用 dp+pp-window 估计（假设并集是整步）。
            ovl = max(0.0, agg["dp_s"] + agg["pp_s"] - agg["window_s"])
            denom = agg["dp_s"] + agg["pp_s"] - ovl
            if denom > 0:
                add(f"    DP∩PP 重叠    {ovl:8.2f} s  ({100 * ovl / denom:5.1f}%)   "
                    f"← 标注区间估计")
        if r.get("avg_step_s"):
            add(f"  step_csv（最后一级 stage）: {r['num_steps']} 步, "
                f"均值 {r['avg_step_s']:.2f} s, 逐步 "
                + " ".join(f"{t:.1f}" for t in r.get("per_step", [])))
        if agg_all:
            best = min(a["window_s"] for a in agg_all)
            add(f"  → 本 mode 稳态步时长 {best:.2f} s（两节点取快的一侧）")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------- 实时指标报告

def _fmt_gb(v, nd=3):
    return f"{v:.{nd}f}" if v is not None else "-"


def _fmt_pct(v, nd=1):
    return f"{100 * v:.{nd}f}%" if v is not None else "-"


def metrics_report(rows, args):
    """每步实时指标（[rt-metric]）+ 端到端 NCCL 通信量，返回可直接打印/落盘的文本。"""
    lines = []
    add = lines.append
    add("=" * 78)
    add("实时指标（[rt-metric] 每步一条）与端到端集合通信量")
    add("=" * 78)
    add("  [rt-metric] 来自 polar wrapper（polar-sgd/src/psgd/parallelism/polar/wrapper.py），")
    add("  和 run.py / metrics_api.py 同源；只有开了 polar 的 mode（polar_dense/polar）才有。")

    for r in rows:
        add(f"\n## {r['mode']}   EXIT_CODE node0={r['exit0']} node1={r['exit1']}")
        for rank in (0, 1):
            rt = r["rt"][rank]
            name = NODES[rank]["name"]
            if not rt:
                add(f"  node{rank} {name}: 没有 [rt-metric]"
                    + ("（baseline/bitscom 不开 polar，DP 重叠率与压缩量无从谈起）"
                       if r["mode"] in ("baseline", "bitscom") else "（训练没进到打印指标那一步）"))
                continue
            add(f"  node{rank} {name}: {rt['rt_steps']} 步有指标")
            for d in rt["rt_steps_detail"]:
                add(f"    step={str(d['step']):>3}  loss={str(d['loss']):>8}  "
                    f"pp_ms={str(d['pp_ms']):>9}  dp_ms={str(d['dp_ms']):>8}  "
                    f"DP∩PP={str(d['overlap']):>7}  "
                    f"dense={str(d['comm_dense']):>9}  comp={str(d['comm_compressed']):>9}  "
                    f"saved={str(d['comm_saved']):>9}  压缩={str(d['comm_reduction']):>7}")
            add(f"    均值: DP-PP 重叠={_fmt_pct(rt['rt_overlap_mean'])}  "
                f"pp={_fmt(rt['rt_pp_ms_mean'], 1)}ms  dp={_fmt(rt['rt_dp_ms_mean'], 1)}ms  "
                f"每步通信 dense={_fmt_gb(rt['rt_comm_dense_gb'])}GB "
                f"compressed={_fmt_gb(rt['rt_comm_compressed_gb'])}GB "
                f"saved={_fmt_gb(rt['rt_comm_saved_gb'])}GB "
                f"压缩比={_fmt(rt['rt_comm_reduction_mean'], 2)}x")

        nc0, nc1 = r["nccl"][0], r["nccl"][1]
        total = r.get("nccl_total_bytes")
        if total is None:
            why = (nc0.get("error") or nc1.get("error")
                   or "没有 logs/nccl*.log（NCCL_DEBUG 未生效或训练没起来）")
            add(f"  端到端集合通信量: 未测得（{why}）")
        else:
            add(f"  端到端集合通信量（NCCL COLL payload，两节点合计）: "
                f"{_fmt_gb(total / 1e9)} GB  "
                f"(node0 {_fmt_gb(nc0['total_bytes'] / 1e9)} GB / "
                f"node1 {_fmt_gb(nc1['total_bytes'] / 1e9)} GB, "
                f"{nc0.get('logs', 0) + nc1.get('logs', 0)} 个 rank 日志)")
            ops = {}
            for nc in (nc0, nc1):
                for op, gb in (nc.get("by_op_gb") or {}).items():
                    ops[op] = ops.get(op, 0.0) + gb
            if ops:
                add("    按算子: " + "  ".join(f"{k}={v:.3f}GB"
                                             for k, v in sorted(ops.items(), key=lambda x: -x[1])))
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
    # 训练当中就把每步的 DP-PP 重叠率 / 通信压缩量打出来（node0 的原始行也会实时回显）
    metrics_stop = threading.Event()
    threading.Thread(target=live_metrics_loop,
                     args=(mode, args.container,
                           {0: mode_dir / "node0", 1: mode_dir / "node1"},
                           metrics_stop),
                     daemon=True).start()
    rc0, timed_out0 = wait_proc(proc0, args.timeout_s, "node0")
    rc1, timed_out1 = wait_proc(proc1, args.timeout_s, "node1")
    metrics_stop.set()
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

    dp_world = (args.gpus * 2) // max(args.pp * args.tp, 1)
    thr = None
    if step.get("total_step_s"):
        thr = step["num_steps"] * dp_world * args.micro_batches * args.seq_len / step["total_step_s"]

    # trace 一律不找：Corex 上 profiler 跑不出 trace，扫 log 目录只会拿到别轮的旧文件。
    traces = {0: [], 1: []}

    # 实时指标（每步一行，只有开 polar 的 mode 有）——node0/node1 各一份
    rt0 = _rt_summary(parse_rt_metrics(
        read_text(NODES[0]["host"], args.container, node0 / "train.log")))
    rt1 = _rt_summary(parse_rt_metrics(
        read_text(NODES[1]["host"], args.container, node1 / "train.log")))

    # 端到端集合通信字节量（NCCL 自报口径），两节点各自统计
    nccl0 = parse_nccl_bytes(NODES[0]["host"], args.container, node0)
    nccl1 = parse_nccl_bytes(NODES[1]["host"], args.container, node1)
    nccl_total = None
    if (nccl0.get("total_bytes") is not None and nccl1.get("total_bytes") is not None
            and nccl0.get("logs", 0) + nccl1.get("logs", 0) > 0):
        nccl_total = nccl0["total_bytes"] + nccl1["total_bytes"]

    return {
        **rec,
        "final_loss_node0": losses0[-1][1] if losses0 else None,
        "final_loss_node1": losses1[-1][1] if losses1 else None,
        "rt": {0: rt0, 1: rt1},
        "nccl": {0: nccl0, 1: nccl1},
        "nccl_total_bytes": nccl_total,
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


def _rt_mean(r, key):
    """两节点里能测到的那一侧的均值（node0 优先）。"""
    for rank in (0, 1):
        rt = (r.get("rt") or {}).get(rank)
        if rt and rt.get(key) is not None:
            return rt[key]
    return None


def summary_table(rows):
    hdr = (f"{'mode':10} {'exit0':>9} {'exit1':>9} {'loss0':>8} {'loss1':>8} "
           f"{'steps':>5} {'avg_step_s':>10} {'tok/s':>9} {'traces':>6} {'GB':>7} "
           f"{'overlap':>8} {'comm_GB':>9} {'压缩比':>8}")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        nccl_total = r.get("nccl_total_bytes")
        lines.append(
            f"{r['mode']:10} {str(r['exit0']):>9} {str(r['exit1']):>9} "
            f"{_fmt(r['final_loss_node0'], 4):>8} {_fmt(r['final_loss_node1'], 4):>8} "
            f"{str(r['num_steps'] or '-'):>5} {_fmt(r['avg_step_s']):>10} "
            f"{_fmt(r['throughput_tok_s'], 1):>9} "
            f"{len(r['traces']):>6} {_fmt(r['trace_mb'] / 1024):>7} "
            f"{_fmt_pct(_rt_mean(r, 'rt_overlap_mean')):>8} "
            f"{_fmt_gb(None if nccl_total is None else nccl_total / 1e9):>9} "
            f"{_fmt(_rt_mean(r, 'rt_comm_reduction_mean')):>8}"
        )
    return "\n".join(lines)


def compare(rows):
    by = {r["mode"]: r for r in rows}
    b, m = by.get("baseline"), by.get("polar")
    out = []
    if b and m and b.get("avg_step_s") and m.get("avg_step_s"):
        out.append(f"步时   baseline={b['avg_step_s']:.2f}s  polar={m['avg_step_s']:.2f}s  "
                   f"加速={b['avg_step_s'] / m['avg_step_s']:.2f}x")
    if b and m and b.get("throughput_tok_s") and m.get("throughput_tok_s"):
        out.append(f"吞吐   baseline={b['throughput_tok_s']:.1f} tok/s  "
                   f"polar={m['throughput_tok_s']:.1f} tok/s  "
                   f"ratio={m['throughput_tok_s'] / b['throughput_tok_s']:.2f}x")
    # 通信量：NCCL 自报的端到端 payload（压缩过的 DP + 未压缩的 TP/PP 都在内）
    nb, nm = b.get("nccl_total_bytes") if b else None, m.get("nccl_total_bytes") if m else None
    if nb and nm:
        out.append(f"通信量 端到端 NCCL payload: baseline={_fmt_gb(nb / 1e9)}GB  "
                   f"polar={_fmt_gb(nm / 1e9)}GB  "
                   f"降低={_fmt_gb((nb - nm) / 1e9)}GB ({nb / nm:.2f}x)")
    # 实时口径：DP-PP 重叠率 / 压缩比（只有开 polar 的 mode 才测得到）
    ov_b, ov_m = (_rt_mean(b, "rt_overlap_mean") if b else None), \
                 (_rt_mean(m, "rt_overlap_mean") if m else None)
    if ov_m is not None:
        out.append(f"重叠率 DP-PP（实时均值）: baseline={_fmt_pct(ov_b)}  polar={_fmt_pct(ov_m)}")
    elif b and m:
        out.append("重叠率 DP-PP: 实时口径未测到（baseline 不开 polar），无 trace 可参考")
    red = _rt_mean(m, "rt_comm_reduction_mean") if m else None
    if red is not None:
        out.append(f"压缩比（实时均值）: polar {red:.2f}x   "
                   f"每步 dense={_fmt_gb(_rt_mean(m, 'rt_comm_dense_gb'))}GB → "
                   f"compressed={_fmt_gb(_rt_mean(m, 'rt_comm_compressed_gb'))}GB")
    if b and m:
        bit = [f"loss baseline={_fmt(b.get('final_loss_node0'), 4)} "
               f"polar={_fmt(m.get('final_loss_node0'), 4)}"]
        out.extend(bit)
    if not out:
        out.append("（缺少 baseline 或 polar 的步时数据，不做对比）")
    return "\n".join(out)


# ---------------------------------------------------------------- main

def make_arg_parser():
    p = argparse.ArgumentParser(
        description="跨机 Qwen3B：polar/baseline 各 5 iter（默认不开 profiler）+ 自动分析",
        epilog="必须在 powerone1 的宿主机上运行（容器里没有 docker）。")
    p.add_argument("--modes", default=None,
                   help="逗号分隔，可选 baseline|polar|polar_dense|bitscom（按顺序跑；"
                        "不写则默认 polar,baseline）")
    p.add_argument("--only-baseline", action="store_true",
                   help="只跑 baseline、不跑 polar（等价于 --modes baseline，二者同时给出时以本参数为准）")
    p.add_argument("--repo", default=DEFAULT_REPO, help="两台机器上相同的仓库根路径")
    p.add_argument("--container", default="trj-udtca")
    p.add_argument("--ssh-target", default=SSH["target"], help="ssh 到 node1 的目标（默认 powerone2）")
    p.add_argument("--ssh-opts", default="",
                   help='ssh 参数（给了就整体替换默认的 "-o BatchMode=yes -o ConnectTimeout=10"），'
                        '如 "-p 7122 -i /root/.ssh/id_rsa"')
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--micro-batches", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--gpus", type=int, default=16, help="每台机器的卡数")
    p.add_argument("--pp", type=int, default=8)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    # 29630 在 powerone1 上会被 kube-apiserver 的临时端口占用，默认换成两机都空的端口；
    # 29694 与 14B 用的 29693 错开，大小模型可以同时起。
    p.add_argument("--master-port", type=int, default=29694)
    p.add_argument("--enable-profiler", action="store_true",
                   help="打开 torch profiler（默认关闭：Corex 上开着跑会在第 1 步卡 CUPTI）")
    p.add_argument("--profiler-active-steps", type=int, default=2,
                   help="打开 profiler 时最后几步采样（默认 2）")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--warmup-s", type=int, default=25, help="node1 起来后等多少秒再起 node0")
    p.add_argument("--timeout-s", type=int, default=None,
                   help="单节点最长等待（默认不限；到点只停止等待，不杀进程）")
    p.add_argument("--no-analyze", action="store_true", help="跑完不做分析")
    p.add_argument("--skip-preflight", action="store_true", help="跳过启动前检查")
    p.add_argument("--skip-card-check", action="store_true",
                   help="跳过卡占用 / page fault / vmid 检查（默认不跳过）")
    p.add_argument("--gpu-mem-threshold", type=int, default=GPU_MEM_THRESHOLD_MIB,
                   help=f"单卡显存超过多少 MiB 算被占用（默认 {GPU_MEM_THRESHOLD_MIB}）")
    p.add_argument("--preflight-only", action="store_true",
                   help="只做启动前检查（含卡占用 / 卡健康）")
    p.add_argument("--dry-run", action="store_true", help="只打印生成的脚本")
    return p


def main():
    args = make_arg_parser().parse_args()
    args._ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    SSH["target"] = args.ssh_target
    if args.ssh_opts.strip():
        SSH["opts"] = shlex.split(args.ssh_opts)

    if args.only_baseline:
        if args.modes and [m.strip() for m in args.modes.split(",") if m.strip()] != ["baseline"]:
            print(f"--only-baseline 与 --modes {args.modes} 同时给出，以 --only-baseline 为准（只跑 baseline）",
                  file=sys.stderr)
        modes = ["baseline"]
    else:
        modes = [m.strip() for m in (args.modes or "polar,baseline").split(",") if m.strip()]
    for m in modes:
        if m not in METHOD_ARGS:
            print(f"未知 mode: {m}", file=sys.stderr)
            return 2

    run_dir = Path(args.repo) / "run_log" / f"xnode_{args._ts}"
    print("=" * 78)
    print(f"跨机训练  modes={modes}  steps={args.max_steps}  "
          f"GPUS={args.gpus}/节点 DP={2 * args.gpus // (args.pp * args.tp)} "
          f"PP={args.pp} TP={args.tp}  "
          f"profiler={'最后 %d iter' % args.profiler_active_steps if args.enable_profiler else '关闭'}")
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
        if args.skip_card_check:
            return 0
        return 0 if card_precheck(args) else 3

    install_signal_handlers(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for mode in modes:
            # 每个 mode 起训前都过一遍：卡被占 / 有 page fault·vmid 就整轮不跑
            if not args.skip_card_check and not card_precheck(args, when=mode):
                print(f"\n卡不可用，{mode} 及后续 mode 都不启动。")
                break
            _STATE["active"] = True
            rec = run_mode(mode, args, run_dir)
            rows.append(collect(rec, args))
            _STATE["active"] = False
    except BaseException as exc:                               # noqa: BLE001
        _STATE["active"] = False
        cleanup_nodes(args, f"异常 {exc.__class__.__name__}: {exc}")
        raise

    if not rows:
        print("\n所有 mode 都因卡不可用被跳过，没有产生任何训练结果。")
        return 3

    table = summary_table(rows)
    cmp_text = compare(rows)
    analysis = "" if args.no_analyze else analysis_report(rows, args)
    metrics = metrics_report(rows, args)

    print(f"\n{'=' * 78}\n汇总\n{'=' * 78}")
    print(table, "\n")
    print(cmp_text)
    if analysis:
        print("\n" + analysis)
    print("\n" + metrics)

    header = (f"# 跨机 Qwen3B 汇总\n\nrun_dir: {run_dir}\n\n"
              f"## 汇总表\n```\n{table}\n```\n\n## 对比\n```\n{cmp_text}\n```\n")
    details = []
    for r in rows:
        details.append(f"\n## {r['mode']}\n- 退出码 node0={r['exit0']} node1={r['exit1']}")
        details.append(f"- case: {r['case_dir']}")
        for t in r["traces"]:
            details.append(f"- trace {t['mb']} MB: {t['path']}")
    md = (header + "\n## 实时指标与通信量\n```\n" + metrics + "\n```\n"
          + ("\n## 分析\n```\n" + analysis + "\n```\n" if analysis else "")
          + "\n".join(details) + "\n")

    (run_dir / "summary.json").write_text(
        json.dumps({"run_dir": str(run_dir), "modes": rows, "compare": cmp_text},
                   indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(md, encoding="utf-8")
    (run_dir / "metrics.log").write_text(metrics + "\n", encoding="utf-8")
    if analysis:
        (run_dir / "analysis.log").write_text(analysis + "\n", encoding="utf-8")

    print(f"\n结果已写入:")
    print(f"  {run_dir}/summary.md")
    print(f"  {run_dir}/summary.json")
    print(f"  {run_dir}/metrics.log")
    if analysis:
        print(f"  {run_dir}/analysis.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
