#!/usr/bin/env bash
# =============================================================================
# Qwen2.5-14B 训练启动器 —— 仅用于【单机】
#
# 2x2 实验矩阵（POLAR 提前 DP 同步 × 是否量化）：
#   MODE=baseline     POLAR off, dense     —— DDP baseline
#   MODE=polar_dense  POLAR on,  dense     —— 与 baseline 的差别只有 POLAR
#   MODE=bitscom      POLAR off, bitscom   —— 与 baseline 的差别只有量化
#   MODE=mixed        POLAR on,  bitscom   —— 与 polar_dense 的差别只有量化
#
#   单机 8 卡 : GPUS=8 ...
#
#   双机（powerone1 + powerone2）不走这个脚本：请用 qwen14b/run.py
#   （orchestrator 会按 config 自动拉起两台机器）。本脚本只连 127.0.0.1。
#
# 默认拓扑（PP*TP*DP = NPROC）：
#   GPUS=16 -> PP=8, TP=1, DP=2
#   GPUS=8  -> PP=2, TP=2, DP=2
#   DP=1 的旧配置（PP=8,TP=2）需显式传 PP_SIZE=8。
#
# 常用覆盖项：
#   MODE=baseline|polar_dense|bitscom|mixed   GPUS=16|8
#   PP_SIZE TP_SIZE NPROC_PER_NODE CUDA_VISIBLE_DEVICES
#   MAX_STEPS MICRO_BATCHES SEQ_LEN DATASET_CONFIG MASTER_PORT CASE_DIR
#   COMM_TIMING POLAR_MAX_INFLIGHT SKIP_PRECHECK=1
#   UDTCA_COMM_WATCHDOG_S（默认 300；自动回滚，设 0 关闭）
#
# 产出：$CASE_DIR（默认 run_log/single_qwen14b_<mode>_g<GPUS>_<时间戳>）
#   train.log  nccl.*.log  PRE_RUN.md  ixsmi_before.log  EXIT_CODE
# =============================================================================
set -uo pipefail

REPO_ROOT=/data/tangruijing/udtca
ENTRY="$REPO_ROOT/experiments/qwen14b/run_qwen14b_polar_dp_pp_tp.py"

MODE="${MODE:-baseline}"
GPUS="${GPUS:-16}"
MAX_STEPS="${MAX_STEPS:-3}"
MICRO_BATCHES="${MICRO_BATCHES:-16}"
SEQ_LEN="${SEQ_LEN:-256}"
DATASET_CONFIG="${DATASET_CONFIG:-sample-10BT}"
MASTER_PORT="${MASTER_PORT:-29630}"

case "$GPUS" in
  16) DEF_PP=8; DEF_TP=1 ;;
  8)  DEF_PP=2; DEF_TP=2 ;;
  *)  DEF_PP=2; DEF_TP=2 ;;
esac
PP_SIZE="${PP_SIZE:-$DEF_PP}"
TP_SIZE="${TP_SIZE:-$DEF_TP}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$GPUS}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS - 1)))}"

# 2x2 矩阵只由两个开关决定；其余参数在四种模式下完全一致。
case "$MODE" in
  baseline)    POLAR_ON=0; QUANT_ON=0; RUN_LABEL=baseline_ddp_single ;;
  polar_dense) POLAR_ON=1; QUANT_ON=0; RUN_LABEL=polar_dense_single ;;
  bitscom)     POLAR_ON=0; QUANT_ON=1; RUN_LABEL=bitscom_quant_single ;;
  mixed|polar) POLAR_ON=1; QUANT_ON=1; RUN_LABEL=polar_bitscom_single ;;
  *)
    echo "ERROR: 未知 MODE=$MODE（可选 baseline|polar_dense|bitscom|mixed）" >&2
    exit 2 ;;
esac

METHOD_ARGS=(--baseline-mode manual)
if [ "$POLAR_ON" = "1" ]; then
  # POLAR：预测式提前 DP 同步。dense/bitscom 两臂这里完全相同。
  METHOD_ARGS+=(--using-polar true --polar-hook ef_lowmem
                --comm-timing "${COMM_TIMING:-8}"
                --polar-bucket-numel 64000000
                --polar-max-inflight-buckets "${POLAR_MAX_INFLIGHT:-1}")
else
  METHOD_ARGS+=(--using-polar false --polar-hook none --comm-timing 0)
fi
if [ "$QUANT_ON" = "1" ]; then
  # 量化：梯度同步走 bitscom lowbit 后端（2bit），其余不变。
  METHOD_ARGS+=(--method bitscom --bitwidth "${BITWIDTH:-2}")
else
  METHOD_ARGS+=(--method none)
fi

# ---- 拓扑检查：DP = NPROC / (PP*TP) ----
if [ $((PP_SIZE * TP_SIZE)) -eq 0 ] || [ $((NPROC_PER_NODE % (PP_SIZE * TP_SIZE))) -ne 0 ]; then
  echo "ERROR: 拓扑不整除: NPROC=$NPROC_PER_NODE PP=$PP_SIZE TP=$TP_SIZE" >&2
  exit 2
fi
DP_SIZE=$((NPROC_PER_NODE / (PP_SIZE * TP_SIZE)))
VISIBLE_COUNT="$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"
if [ "$VISIBLE_COUNT" != "$NPROC_PER_NODE" ]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES 有 $VISIBLE_COUNT 张卡，" \
       "与 NPROC_PER_NODE=$NPROC_PER_NODE 不一致" >&2
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
CASE_DIR="${CASE_DIR:-$REPO_ROOT/run_log/single_qwen14b_${MODE}_g${GPUS}_${TS}}"
CACHE_DIR="$REPO_ROOT/run_log/single_cache"
mkdir -p "$CASE_DIR/logs" "$CACHE_DIR"

# ---- 占卡检查（在改 LD_LIBRARY_PATH 前用系统 ixsmi）----
if [ "${SKIP_PRECHECK:-0}" != "1" ] && ! /usr/bin/ixsmi 2>&1 | grep -q "No running processes found"; then
  echo "ERROR: GPU 上已有进程，退出" | tee -a "$CASE_DIR/precheck.log"
  /usr/bin/ixsmi | tee -a "$CASE_DIR/precheck.log"
  exit 4
fi
/usr/bin/ixsmi > "$CASE_DIR/ixsmi_before.log" 2>&1

{
  echo "# PRE_RUN $TS"
  echo "- topology: SINGLE-NODE GPUs=$GPUS DP=$DP_SIZE PP=$PP_SIZE TP=$TP_SIZE MODE=$MODE"
  echo "- matrix: POLAR_ON=$POLAR_ON QUANT_ON=$QUANT_ON (bitscom bitwidth=${BITWIDTH:-2})"
  echo "- devices: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "- host: $(hostname) uptime: $(uptime -p)"
  echo "- xid_since_boot: $(journalctl -k -b 2>/dev/null | grep -icE 'xid|mmu page fault|va space')"
  echo "- steps=$MAX_STEPS micro_batches=$MICRO_BATCHES seq_len=$SEQ_LEN dataset=$DATASET_CONFIG"
  echo "- case_dir=$CASE_DIR"
} > "$CASE_DIR/PRE_RUN.md"

# ---- 机器初始化 + Corex/CUDA 变量 + conda（与两机编排脚本一致）----
if [ -f ~/.bashrc ]; then
  PS1="${PS1:-ustca-init}"; _flags="$-"; set +eu
  . ~/.bashrc >/dev/null 2>&1 || true
  case "$_flags" in *e*) set -e;; esac; case "$_flags" in *u*) set -u;; esac
  unset _flags PS1
fi
# /usr/local/corex 指向本机实际 Corex 版本（当前集群 4.4.0，新机器 4.5.0）
export COREX_PATH="${UDTCA_COREX_PATH:-/usr/local/corex}"
export COREX_TOOLKIT=/data/tianshu/20260720/corex/corex/corex-toolkit
export CUDA_HOME="$COREX_PATH"
export CPATH="$COREX_PATH/include:${CPATH:-}"
export LIBRARY_PATH="$COREX_PATH/lib64:${LIBRARY_PATH:-}"
export TRITON_CUDA_SYSROOT="$COREX_PATH"
export BITSCOM_CUDA_COMPILER="$COREX_PATH/bin/clang++"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_RETRY_CNT="${NCCL_SOCKET_RETRY_CNT:-600}"
export NCCL_SOCKET_RETRY_SLEEP_MSEC="${NCCL_SOCKET_RETRY_SLEEP_MSEC:-1000}"
export PATH=/usr/local/corex/bin:/usr/local/corex/lib64/python3/dist-packages/bin:$PATH
export LD_LIBRARY_PATH=$COREX_TOOLKIT/ixblas/lib64:$COREX_TOOLKIT/ixinfer/lib64:$COREX_PATH/lib64:/usr/local/corex/lib64:$COREX_TOOLKIT/ixdriver/lib64:$COREX_TOOLKIT/ixccl/lib64:/usr/mpi/gcc/openmpi-4.1.7rc1/lib
if [ -e "$COREX_TOOLKIT/ixccl/lib64/libnccl.so.2" ]; then
  export LD_PRELOAD="$COREX_TOOLKIT/ixccl/lib64/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-package
# conda 根目录按机器探测，兼容 powerone（/home/powerone）和当前机器（/root）
CONDA_ROOT="${UDTCA_CONDA_ROOT:-}"
if [ -z "$CONDA_ROOT" ]; then
  for c in /root/miniconda3 /home/powerone/miniconda3; do
    [ -d "$c" ] && { CONDA_ROOT="$c"; break; }
  done
fi
source "${CONDA_ROOT:?找不到 conda 根目录，请设置 UDTCA_CONDA_ROOT}/etc/profile.d/conda.sh"
export PATH="$CONDA_ROOT/bin:$PATH"
(conda deactivate >/dev/null 2>&1 || true)
conda activate udtca
export PYTHONUNBUFFERED=1

# ---- 网络 / HF ----
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens120f0
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY NO_PROXY no_proxy
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES

# ---- 全量诊断日志 + 自动回滚（revoke 后各 rank 自行退出）----
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISABLE_ADDR2LINE=1
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-off}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-0}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
export UDTCA_COMM_WATCHDOG_S="${UDTCA_COMM_WATCHDOG_S:-300}"
export UDTCA_REVOKE_QUIESCE_S="${UDTCA_REVOKE_QUIESCE_S:-10}"
export UDTCA_REVOKE_BARRIER_S="${UDTCA_REVOKE_BARRIER_S:-5}"
# 手动触发回滚用的停止文件；默认不设，避免上次残留文件误触发。
export UDTCA_REVOKE_FILE="${UDTCA_REVOKE_FILE:-}"
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_FR_BUFFER_SIZE=1048576
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG_FILE="$CASE_DIR/logs/nccl.%h.%p.log"

echo "=== [SINGLE-NODE] qwen14b start $(date -Is)" | tee -a "$CASE_DIR/train.log"
echo "    mode=$MODE POLAR_ON=$POLAR_ON QUANT_ON=$QUANT_ON GPUs=$GPUS DP=$DP_SIZE PP=$PP_SIZE TP=$TP_SIZE" | tee -a "$CASE_DIR/train.log"
echo "    steps=$MAX_STEPS micro_batches=$MICRO_BATCHES seq_len=$SEQ_LEN" | tee -a "$CASE_DIR/train.log"
echo "    case_dir=$CASE_DIR" | tee -a "$CASE_DIR/train.log"
echo "    watchdog=${UDTCA_COMM_WATCHDOG_S}s (0=off)" | tee -a "$CASE_DIR/train.log"

cd "$REPO_ROOT"
torchrun \
  --nproc_per_node="$NPROC_PER_NODE" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port="$MASTER_PORT" \
  "$ENTRY" \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --pp-size "$PP_SIZE" \
  --tp-size "$TP_SIZE" \
  --micro-batches "$MICRO_BATCHES" \
  --per-device-batch-size "$MICRO_BATCHES" \
  --seq-len "$SEQ_LEN" \
  --max-steps "$MAX_STEPS" \
  --log-interval 1 \
  --lr 2e-4 \
  --dataset-name-or-path HuggingFaceFW/fineweb \
  --dataset-config "$DATASET_CONFIG" \
  --text-field text \
  --dataset-cache-dir "$CACHE_DIR" \
  --run-label "$RUN_LABEL" \
  "${METHOD_ARGS[@]}" \
  --num-workers 0 \
  --disable-profiler false \
  --debug-nan-steps 2 \
  --save-dir "$CASE_DIR/ckpt" \
  --step-log-dir "$CASE_DIR/step_csv" \
  2>&1 | tee -a "$CASE_DIR/train.log"
RC=${PIPESTATUS[0]}
echo "EXIT_CODE=$RC" | tee -a "$CASE_DIR/EXIT_CODE" "$CASE_DIR/train.log"
exit "$RC"
