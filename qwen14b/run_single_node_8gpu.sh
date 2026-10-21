#!/usr/bin/env bash
# 单机 16 卡 Qwen2.5-14B 训练（本机 16 个 BI-D300 设备）
# 拓扑：DP1 x PP8 x TP2，与 udtca-config/qwen14b 的两机配置保持同一套参数。
#
# 用法：
#   MODE=baseline bash run_single_node.sh      # 默认，DDP baseline
#   MODE=polar MAX_STEPS=3 bash run_single_node.sh
set -euo pipefail

REPO_ROOT=/data/tangruijing/udtca
ENTRY="$REPO_ROOT/experiments/qwen14b/run_qwen14b_polar_dp_pp_tp.py"

MODE="${MODE:-baseline}"
MAX_STEPS="${MAX_STEPS:-3}"
MICRO_BATCHES="${MICRO_BATCHES:-16}"
PP_SIZE="${PP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-2}"
SEQ_LEN="${SEQ_LEN:-256}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29630}"
DATASET_CONFIG="${DATASET_CONFIG:-sample-10BT}"
TS="$(date +%Y%m%d_%H%M%S)"
CASE_DIR="${CASE_DIR:-$REPO_ROOT/run_log/single_qwen14b_8gpu_${MODE}_${TS}}"
CACHE_DIR="$REPO_ROOT/run_log/single_cache"
mkdir -p "$CASE_DIR/logs" "$CACHE_DIR"

# ---- 占卡检查（在改 LD_LIBRARY_PATH 前用系统 ixsmi）----
if [ "${SKIP_PRECHECK:-0}" != "1" ] && ! /usr/bin/ixsmi 2>&1 | grep -q "No running processes found"; then
  echo "ERROR: GPU 上已有进程，退出" | tee -a "$CASE_DIR/precheck.log"
  /usr/bin/ixsmi | tee -a "$CASE_DIR/precheck.log"
  exit 4
fi
/usr/bin/ixsmi > "$CASE_DIR/ixsmi_before.log" 2>&1

# ---- 机器初始化 + Corex/CUDA 变量 + conda，与两机编排脚本一致 ----
if [ -f ~/.bashrc ]; then
  PS1="${PS1:-ustca-init}"; _flags="$-"; set +eu
  . ~/.bashrc >/dev/null 2>&1 || true
  case "$_flags" in *e*) set -e;; esac; case "$_flags" in *u*) set -u;; esac
  unset _flags PS1
fi
export COREX_PATH=/data/tangruijing/udtca/.corex-4.5.0
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
export LD_PRELOAD="$COREX_TOOLKIT/ixccl/lib64/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-package
source /home/powerone/miniconda3/etc/profile.d/conda.sh
export PATH="/home/powerone/miniconda3/bin:$PATH"
(conda deactivate >/dev/null 2>&1 || true)
conda activate udtca
export PYTHONUNBUFFERED=1

# ---- 网络 / HF ----
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens120f0
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY NO_PROXY no_proxy
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15

# ---- 全量诊断日志 ----
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISABLE_ADDR2LINE=1
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_FR_BUFFER_SIZE=1048576
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG_FILE="$CASE_DIR/logs/nccl.%h.%p.log"

if [ "$MODE" = "polar" ] || [ "$MODE" = "bitscom" ]; then
  METHOD_ARGS=(--using-polar true --polar-hook ef_lowmem --method bitscom --bitwidth 2 --comm-timing 8 --polar-bucket-numel 64000000 --polar-max-inflight-buckets 4)
  RUN_LABEL=polar_bitscom_single
elif [ "$MODE" = "polar_dense" ]; then
  METHOD_ARGS=(--using-polar true --polar-hook ef_lowmem --method none --comm-timing 8 --polar-bucket-numel 64000000 --polar-max-inflight-buckets 4)
  RUN_LABEL=polar_dense_single
else
  METHOD_ARGS=(--using-polar false --baseline-mode manual --polar-hook none --method none --comm-timing 0)
  RUN_LABEL=baseline_ddp_single
fi

echo "=== single-node qwen14b start $(date -Is) mode=$MODE pp=$PP_SIZE tp=$TP_SIZE nproc=$NPROC_PER_NODE max_steps=$MAX_STEPS micro_batches=$MICRO_BATCHES dataset_config=$DATASET_CONFIG case_dir=$CASE_DIR ===" | tee -a "$CASE_DIR/train.log"

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
