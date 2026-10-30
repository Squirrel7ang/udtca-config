#!/usr/bin/env bash
# 云侧 powerone1：单机 8 卡，DP=2 x PP=4 x TP=1，在容器内执行。
#
#   MODE=baseline bash run_single_node_8gpu.sh
#
# 其余可覆盖项（MAX_STEPS MICRO_BATCHES SEQ_LEN SEED MASTER_PORT CASE_DIR 等）
# 与 run_docker_single_node.sh 一致。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GPUS=8
export PP_SIZE=4
export TP_SIZE=1
export SKIP_PRECHECK=1
# 这台机器的 P2P 直连不稳，统一走间接中继；要对照直连时显式设成 0。
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
# 10.205.200.14 挂在 ens120f0 上，跨机建连必须钉到这块。
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens120f0}"
export UDTCA_NODE0_IP="${UDTCA_NODE0_IP:-10.205.200.14}"
export UDTCA_NODE1_HOST="${UDTCA_NODE1_HOST:-10.205.200.16}"
# 镜像自带 torch 2.13，psgd 的版本检查默认要求 2.10。
export PSGD_CHECK_TORCH_VERSION="${PSGD_CHECK_TORCH_VERSION:-0}"

exec bash "$SCRIPT_DIR/../run_docker_single_node.sh" "$@"
