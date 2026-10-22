#!/usr/bin/env bash
# 单机 8 卡启动器：DP=2 × PP=4 × TP=1。
#
# 与 run_single_node.sh 的区别只有默认拓扑和占卡检查：
#   - GPUS=8，PP_SIZE=4，TP_SIZE=1（run_single_node.sh 的 8 卡默认是 PP2×TP2）
#   - 默认 SKIP_PRECHECK=1：powerone1 的 ixsmi 读数不可信（8 张物理卡 = 16 个
#     芯粒，部分卡掉线后设备号会错位），不能拿它判断占卡。
#   - 不包 timeout。powerone1 上 NCCL 进程被 SIGTERM 打断会让整机卡死，
#     必须靠脚本自身的通信看门狗 + revoke 路径退出。
#
#   MODE=baseline bash run_single_node_8gpu.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 是默认值；可用同名环境变量覆盖。
#   其余覆盖项（MAX_STEPS MICRO_BATCHES SEQ_LEN SEED MASTER_PORT CASE_DIR 等）
#   与 run_single_node.sh 完全一致。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GPUS="${GPUS:-8}"
export PP_SIZE="${PP_SIZE:-4}"
export TP_SIZE="${TP_SIZE:-1}"
export SKIP_PRECHECK="${SKIP_PRECHECK:-1}"

# exec：让 torchrun 直接顶替本 shell，这样 PID/信号链里不会多一层 bash。
exec bash "$SCRIPT_DIR/run_single_node.sh" "$@"
