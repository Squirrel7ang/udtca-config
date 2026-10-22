#!/usr/bin/env bash
# 16 die 并行健康扫描：每张卡独立进程，单卡 12s 超时
set -uo pipefail
REPO_ROOT=/data/tangruijing/udtca
TS="$(date +%Y%m%d_%H%M%S)"
OUT="${1:-$REPO_ROOT/run_log/recovery_$TS}"
mkdir -p "$OUT"

export COREX_PATH=/data/tangruijing/udtca/.corex-4.5.0
export COREX_TOOLKIT=/data/tianshu/20260720/corex/corex/corex-toolkit
export CUDA_HOME="$COREX_PATH"
export LD_LIBRARY_PATH=$COREX_TOOLKIT/ixblas/lib64:$COREX_TOOLKIT/ixinfer/lib64:$COREX_PATH/lib64:/usr/local/corex/lib64:$COREX_TOOLKIT/ixdriver/lib64:$COREX_TOOLKIT/ixccl/lib64:/usr/mpi/gcc/openmpi-4.1.7rc1/lib
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-package
source /home/powerone/miniconda3/etc/profile.d/conda.sh
conda activate udtca
export CUDA_DEVICE_MAX_CONNECTIONS=1

{
  echo "=== $(date -Is) uptime ==="; uptime
  echo "=== processes ==="; pgrep -af "torchrun|diag_nccl_teardown|run_qwen14b" || echo none
  echo "=== ixsmi ==="; /usr/bin/ixsmi 2>&1
  echo "=== kernel XID/mmu baseline (this boot) ==="
  journalctl -k -b 2>&1 | grep -iE "XID|mmu|va space" || echo none
} > "$OUT/baseline.log" 2>&1

# 只等待，不发任何信号：单卡 12s 没结束就标记 WEDGED，进程保留给人工处理
# （杀掉在途的 GPU 进程可能把 die 打挂，见 XID 24 的经验）。
PIDS=()
DEVS=()
for dev in $(seq 0 15); do
  (
    CUDA_DEVICE_MAX_CONNECTIONS=1 python -c "import torch; torch.cuda.set_device($dev); x=torch.ones(8,device='cuda'); torch.cuda.synchronize(); print('OK', x.sum().item())" > "$OUT/dev${dev}.log" 2>&1
    echo "$?" > "$OUT/dev${dev}.rc"
  ) &
  PIDS+=("$!")
  DEVS+=("$dev")
done

DEADLINE=$((SECONDS + 12))
while :; do
  alive=0
  for pid in "${PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null && alive=$((alive + 1))
  done
  [ "$alive" -eq 0 ] && break
  [ "$SECONDS" -ge "$DEADLINE" ] && break
  sleep 0.2
done

for i in "${!DEVS[@]}"; do
  dev="${DEVS[$i]}"
  if [ -f "$OUT/dev${dev}.rc" ]; then
    rc="$(cat "$OUT/dev${dev}.rc")"
    msg="$(tail -1 "$OUT/dev${dev}.log" 2>/dev/null)"
  else
    rc="WEDGED"
    msg="no result in 12s; probe pid=${PIDS[$i]} left running (not killed)"
  fi
  echo "dev$dev rc=$rc :: $msg"
done | tee "$OUT/health.log"

echo "out=$OUT"
