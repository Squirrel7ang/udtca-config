#!/usr/bin/env bash
# 探针 B 的运行壳（容器内执行）。单机 2 卡，不碰其它卡、不碰在跑的作业。
#
#   bash diag/run_probe_ar_sync.sh
#   NGPUS=2 NCCL_P2P_DISABLE=1 bash diag/run_probe_ar_sync.sh    # powerone2 必须加
#
# 只用 CUDA_VISIBLE_DEVICES 指定的卡；日志落在 run_log/probe_ar_sync_<host>_<ts>/
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"

NGPUS="${NGPUS:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export COREX_PATH=/usr/local/corex
export CUDA_HOME="$COREX_PATH"
export PATH=/usr/local/corex/bin:$PATH
export LD_LIBRARY_PATH="$COREX_PATH/lib64:/usr/local/openmpi/lib:/usr/local/lib"
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1

TS="$(date +%Y%m%d_%H%M%S)"
OUT="$REPO_ROOT/run_log/probe_ar_sync_$(hostname)_${TS}"
mkdir -p "$OUT"
echo "devices=$CUDA_VISIBLE_DEVICES p2p_disable=$NCCL_P2P_DISABLE out=$OUT"

torchrun --nproc_per_node="$NGPUS" --master_port=29677 \
  "$HERE/probe_ar_sync.py" 2>&1 | tee "$OUT/probe.log"
echo "EXIT=${PIPESTATUS[0]}" | tee -a "$OUT/probe.log"
echo "OUT=$OUT"
