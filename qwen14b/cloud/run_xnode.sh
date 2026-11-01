#!/usr/bin/env bash
# 跨机 Qwen2.5-14B / 云侧 powerone1（node0）—— 容器内执行，自包含，不调用别的脚本。
#
#   bash cloud/run_xnode.sh
#   MODE=mixed MAX_STEPS=1 bash cloud/run_xnode.sh
#
# 先起边侧 edge/run_xnode.sh，再起本脚本（rendezvous 在 master 上等对端）。
# 拓扑：两台各 16 卡（合计 32），PP=8 TP=2 → 每台就是一个完整的 PP×TP 副本，
# 全局 DP=2；PP/TP 组都落在单机内，跨机只有 DP 的 all-reduce。
#
#   产物在 $CASE_DIR：train.log、logs/nccl.*.log、PRE_RUN.md、EXIT_CODE。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${UDTCA_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
ENTRY="$REPO_ROOT/experiments/qwen14b/run_qwen14b_polar_dp_pp_tp.py"

# ---- 跨机参数（本机是 node0）----
NNODES=2
NODE_RANK=0
MASTER_ADDR="${MASTER_ADDR:-10.205.200.14}"
MASTER_PORT="${MASTER_PORT:-29630}"

GPUS="${GPUS:-16}"
PP_SIZE="${PP_SIZE:-8}"
TP_SIZE="${TP_SIZE:-2}"
MODE="${MODE:-baseline}"
MAX_STEPS="${MAX_STEPS:-3}"
MICRO_BATCHES="${MICRO_BATCHES:-16}"
SEQ_LEN="${SEQ_LEN:-256}"
DATASET_CONFIG="${DATASET_CONFIG:-sample-10BT}"
DP_SIZE=$((GPUS / (PP_SIZE * TP_SIZE)))
# profiler：全程抓 trace（没有 wait/warmup，active=max_steps），落在仓库根 log/<...>/tb_trace。
DISABLE_PROFILER="${DISABLE_PROFILER:-false}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS - 1)))}"

# 低比特 DP 同步严格串行（正确性优先）。lowbit 的 phase1/phase2/restore 共用
# 同一个 comm，多 bucket 同时在飞时两个 DP rank 的 p2p 投递顺序会分叉，偶发卡死。
export POLAR_EF_LOWMEM_SERIAL_BUCKETS="${POLAR_EF_LOWMEM_SERIAL_BUCKETS:-1}"

# 四种模式各自需要追加的训练参数，全放在 METHOD_ARGS 里，torchrun 末尾统一展开。
case "$MODE" in
  baseline)
    RUN_LABEL=baseline_ddp_xnode
    METHOD_ARGS=(--baseline-mode manual
                 --using-polar false --polar-hook none --comm-timing 0
                 --method none)
    ;;
  polar_dense)
    RUN_LABEL=polar_dense_xnode
    METHOD_ARGS=(--baseline-mode manual
                 --using-polar true --polar-hook ef_lowmem --comm-timing 8
                 --polar-bucket-numel 64000000 --polar-max-inflight-buckets 1
                 --method none)
    ;;
  bitscom)
    RUN_LABEL=bitscom_quant_xnode
    METHOD_ARGS=(--baseline-mode manual
                 --using-polar false --polar-hook none --comm-timing 0
                 --method bitscom --bitwidth 2)
    ;;
  mixed)
    RUN_LABEL=polar_bitscom_xnode
    METHOD_ARGS=(--baseline-mode manual
                 --using-polar true --polar-hook ef_lowmem --comm-timing 8
                 --polar-bucket-numel 64000000 --polar-max-inflight-buckets 1
                 --method bitscom --bitwidth 2)
    ;;
  *) echo "ERROR: 未知 MODE=$MODE（baseline|polar_dense|bitscom|mixed）" >&2; exit 2 ;;
esac

TS="$(date +%Y%m%d_%H%M%S)"
CASE_DIR="${CASE_DIR:-$REPO_ROOT/run_log/xnode_${MODE}_node0_${TS}}"
mkdir -p "$CASE_DIR/logs"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/log}"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/$(basename "$CASE_DIR").log"
TEE=(tee -a "$CASE_DIR/train.log" "$RUN_LOG")

{
  echo "# PRE_RUN $TS"
  echo "- runner: cloud/run_xnode.sh (container corex 4.5.0)"
  echo "- topology: GPUs=$GPUS PP=$PP_SIZE TP=$TP_SIZE MODE=$MODE"
  echo "- distributed: nnodes=$NNODES node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT"
  echo "- mode_args: ${METHOD_ARGS[*]}"
  echo "- devices: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "- host: $(hostname)"
  echo "- steps=$MAX_STEPS micro_batches=$MICRO_BATCHES seq_len=$SEQ_LEN seed=${SEED:-1234}"
  echo "- dataset=HuggingFaceFW/fineweb config=$DATASET_CONFIG"
} > "$CASE_DIR/PRE_RUN.md"

# ---- 容器内 Corex 环境 ----
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

# ---- 通信 ----
export NCCL_IB_DISABLE=1
# 10.205.200.14 挂在 ens120f0 上，跨机建连必须钉到这块。
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens120f0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SOCKET_RETRY_CNT="${NCCL_SOCKET_RETRY_CNT:-600}"
export NCCL_SOCKET_RETRY_SLEEP_MSEC="${NCCL_SOCKET_RETRY_SLEEP_MSEC:-1000}"

# ---- 数据 ----
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY NO_PROXY no_proxy
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-15}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-30}"

# ---- 日志 / 诊断环境 ----
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
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG_FILE="$CASE_DIR/logs/nccl.%h.%p.log"

{
  echo "- logs: case=$CASE_DIR logs=$CASE_DIR/logs run_log=$RUN_LOG"
  echo "- nccl: debug=$NCCL_DEBUG subsys=$NCCL_DEBUG_SUBSYS file=$NCCL_DEBUG_FILE ib_disable=$NCCL_IB_DISABLE ifname=${NCCL_SOCKET_IFNAME:-<auto>} p2p_disable=$NCCL_P2P_DISABLE"
  echo "- torch: cpp_log=$TORCH_CPP_LOG_LEVEL dist_debug=$TORCH_DISTRIBUTED_DEBUG faulthandler=$PYTHONFAULTHANDLER trace_buf=$TORCH_NCCL_TRACE_BUFFER_SIZE dump_on_timeout=$TORCH_NCCL_DUMP_ON_TIMEOUT"
  echo "- corex: path=$COREX_PATH pythonpath=$PYTHONPATH ld_library_path=$LD_LIBRARY_PATH"
} >> "$CASE_DIR/PRE_RUN.md"

echo "=== [XNODE/DOCKER] qwen14b start $(date -Is)" | "${TEE[@]}"
echo "    mode=$MODE node_rank=$NODE_RANK/$NNODES master=$MASTER_ADDR:$MASTER_PORT GPUs=$GPUS PP=$PP_SIZE TP=$TP_SIZE" | "${TEE[@]}"
echo "    case_dir=$CASE_DIR" | "${TEE[@]}"

# 崩溃残留的 0 字节缓存：entry 里用 cache_path.exists() 判断，0 字节也算存在，
# 于是跳过重建、随后 torch.load 直接失败。
find "$REPO_ROOT/experiments/qwen14b/cache" -maxdepth 1 \
  -name '*_labels_unshifted_v2.pt' -size 0 -delete 2>/dev/null || true

cd "$REPO_ROOT"
torchrun \
  --nproc_per_node="$GPUS" \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
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
  --seed "${SEED:-1234}" \
  --lr 2e-4 \
  --dataset-name-or-path HuggingFaceFW/fineweb \
  --dataset-config "$DATASET_CONFIG" \
  --text-field text \
  --run-label "$RUN_LABEL" \
  "${METHOD_ARGS[@]}" \
  --num-workers 0 \
  --disable-profiler "$DISABLE_PROFILER" \
  --profiler-start-step "${PROFILER_START_STEP:-1}" \
  --profiler-wait-steps "${PROFILER_WAIT_STEPS:-0}" \
  --profiler-warmup-steps "${PROFILER_WARMUP_STEPS:-0}" \
  --profiler-active-steps "${PROFILER_ACTIVE_STEPS:-$MAX_STEPS}" \
  --profiler-repeat 1 \
  --debug-nan-steps 2 \
  --save-dir "$CASE_DIR/ckpt" \
  --step-log-dir "$CASE_DIR/step_csv" \
  2>&1 | "${TEE[@]}"
RC=${PIPESTATUS[0]}
echo "EXIT_CODE=$RC" | tee -a "$CASE_DIR/EXIT_CODE" "$CASE_DIR/train.log" "$RUN_LOG"
exit "$RC"
