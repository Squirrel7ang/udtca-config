#!/usr/bin/env bash
# 云侧 powerone1 = node0：跨机 NCCL / bitscom 连通性自检，每机 1 卡（world=2）。
#
#   先在本机跑这个，再去 powerone2 跑 edge/run_nccl_xnode_edge.sh。
#   顺序反了也能跑，rendezvous 会等对端。
#
#   bash run_nccl_xnode_cloud.sh                 # 全部用例
#   bash run_nccl_xnode_cloud.sh --skip-lowbit   # 只测 NCCL
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_PY="$SCRIPT_DIR/../nccl_xnode_test.py"

export MASTER_ADDR="${MASTER_ADDR:-10.205.200.14}"
export MASTER_PORT="${MASTER_PORT:-29631}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens120f0}"
export NCCL_IB_DISABLE=1
export NCCL_SYM_KERNEL="${NCCL_SYM_KERNEL:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"

export COREX_PATH=/usr/local/corex
export CUDA_HOME="$COREX_PATH"
export LD_LIBRARY_PATH="$COREX_PATH/lib64:/usr/local/openmpi/lib:/usr/local/lib"
export PYTHONPATH="$COREX_PATH/lib64/python3/dist-packages"
export PATH="$COREX_PATH/bin:$PATH"
export PYTHONUNBUFFERED=1

exec torchrun \
  --nnodes=2 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$TEST_PY" "$@"
