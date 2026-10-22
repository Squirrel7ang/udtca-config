#!/usr/bin/env bash
# 16 rank allreduce + barrier（正常退出，不做中途终止）
set -uo pipefail
REPO_ROOT=/data/tangruijing/udtca
TS="$(date +%Y%m%d_%H%M%S)"
CASE_DIR="$REPO_ROOT/run_log/diag_comm16_$TS"
mkdir -p "$CASE_DIR/logs"

export COREX_PATH=/data/tangruijing/udtca/.corex-4.5.0
export COREX_TOOLKIT=/data/tianshu/20260720/corex/corex/corex-toolkit
export CUDA_HOME="$COREX_PATH"
export LD_LIBRARY_PATH=$COREX_TOOLKIT/ixblas/lib64:$COREX_TOOLKIT/ixinfer/lib64:$COREX_PATH/lib64:/usr/local/corex/lib64:$COREX_TOOLKIT/ixdriver/lib64:$COREX_TOOLKIT/ixccl/lib64:/usr/mpi/gcc/openmpi-4.1.7rc1/lib
export LD_PRELOAD="$COREX_TOOLKIT/ixccl/lib64/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages
source /home/powerone/miniconda3/etc/profile.d/conda.sh
conda activate udtca
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
unset CUDA_VISIBLE_DEVICES
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens120f0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG_FILE="$CASE_DIR/logs/nccl.%h.%p.log"
export TORCH_SHOW_CPP_STACKTRACES=0
export TORCH_DISABLE_ADDR2LINE=1

KERNEL_START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "=== $(date -Is) 16-rank allreduce+barrier, case=$CASE_DIR ==="
torchrun --nproc_per_node=16 --master_port="${MASTER_PORT:-29642}" \
  "$REPO_ROOT/run_log/diag_comm_16.py" > "$CASE_DIR/train.log" 2>&1
RC=$?
echo "torchrun rc=$RC"
echo "=== kernel since $KERNEL_START ==="
journalctl -k --since "$KERNEL_START" 2>&1 | grep -iE "XID|mmu|va space" || echo "no XID/mmu/va-space messages"
echo "=== tail train.log ==="
tail -20 "$CASE_DIR/train.log"
echo "=== out=$CASE_DIR ==="
