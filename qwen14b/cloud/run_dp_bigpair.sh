#!/usr/bin/env bash
# 跨机 32 卡：只有 {0,16} 和 {1,17} 两对做大数据量 all-reduce，其余 28 卡直接进 barrier。
#
#   bash cloud/run_dp_bigpair.sh
#   ROUNDS=8 BYTES=$((1610612736)) bash cloud/run_dp_bigpair.sh
#
# 每轮：pair 0、1 各做一次 2 卡 all-reduce（默认 2 GiB），另外 28 个 rank 不做任何通信，
#       直接进 world(32) barrier；两对传完之后一起放行，共 ROUNDS(默认 8) 轮。
# 注意：P2P 保持打开（NCCL_P2P_DISABLE=0）。
# 产物在 $CASE_DIR：train.log、logs/nccl.*.log、PRE_RUN.md、EXIT_CODE。

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${UDTCA_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
ENTRY="$REPO_ROOT/udtca-config/qwen14b/dp_test_bigpair.py"

# ---- 跨机参数（本机是 node0）----
NNODES=2
NODE_RANK=0
MASTER_ADDR="${MASTER_ADDR:-10.205.200.14}"
MASTER_PORT="${MASTER_PORT:-29641}"

GPUS="${GPUS:-16}"
ROUNDS="${ROUNDS:-8}"
BYTES="${BYTES:-$((2*2**30))}"
DTYPE="${DTYPE:-bf16}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS - 1)))}"

TS="$(date +%Y%m%d_%H%M%S)"
CASE_DIR="${CASE_DIR:-$REPO_ROOT/run_log/bigpair_node0_${TS}}"
mkdir -p "$CASE_DIR/logs"

RUN_LOG="$REPO_ROOT/log/$(basename "$CASE_DIR").log"
mkdir -p "$REPO_ROOT/log"
TEE=(tee -a "$CASE_DIR/train.log" "$RUN_LOG")

{
  echo "# PRE_RUN $TS"
  echo "- runner: cloud/run_dp_bigpair.sh"
  echo "- test: rounds=$ROUNDS bytes=$BYTES dtype=$DTYPE"
  echo "- distributed: nnodes=$NNODES node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT"
  echo "- devices: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "- host: $(hostname)"
} > "$CASE_DIR/PRE_RUN.md"

# ---- 容器内 Corex 环境 ----
export COREX_PATH=/usr/local/corex
export CUDA_HOME="$COREX_PATH"
export CPATH="$COREX_PATH/include"
export LIBRARY_PATH="$COREX_PATH/lib64"
export PATH=/usr/local/corex/bin:/usr/local/corex/lib64/python3/dist-packages/bin:$PATH
export LD_LIBRARY_PATH="$COREX_PATH/lib64:/usr/local/openmpi/lib:/usr/local/lib"
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages

# ---- 通信 ----
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens120f0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SOCKET_RETRY_CNT="${NCCL_SOCKET_RETRY_CNT:-600}"
export NCCL_SOCKET_RETRY_SLEEP_MSEC="${NCCL_SOCKET_RETRY_SLEEP_MSEC:-1000}"

# ---- 日志 / 诊断环境 ----
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_CPP_LOG_LEVEL="${TORCH_CPP_LOG_LEVEL:-WARNING}"
export TORCH_DISABLE_ADDR2LINE=1
# 与训练一致：不让 torch 的 NCCL watchdog abort 在途通信。
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-0}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG_FILE="$CASE_DIR/logs/nccl.%h.%p.log"

{
  echo "- logs: case=$CASE_DIR logs=$CASE_DIR/logs run_log=$RUN_LOG"
  echo "- nccl: debug=$NCCL_DEBUG subsys=$NCCL_DEBUG_SUBSYS file=$NCCL_DEBUG_FILE ib_disable=$NCCL_IB_DISABLE ifname=$NCCL_SOCKET_IFNAME p2p_disable=$NCCL_P2P_DISABLE"
} >> "$CASE_DIR/PRE_RUN.md"

echo "=== [DPTEST] start $(date -Is)" | "${TEE[@]}"
echo "    node_rank=$NODE_RANK/$NNODES master=$MASTER_ADDR:$MASTER_PORT GPUs=$GPUS rounds=$ROUNDS bytes=$BYTES dtype=$DTYPE" | "${TEE[@]}"
echo "    case_dir=$CASE_DIR" | "${TEE[@]}"

cd "$REPO_ROOT"
torchrun \
  --nproc_per_node="$GPUS" \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$ENTRY" \
  --rounds "$ROUNDS" \
  --bytes "$BYTES" \
  --dtype "$DTYPE" \
  2>&1 | "${TEE[@]}"
RC=${PIPESTATUS[0]}
echo "EXIT_CODE=$RC" | tee -a "$CASE_DIR/EXIT_CODE" "$CASE_DIR/train.log" "$RUN_LOG"
exit "$RC"
