#!/usr/bin/env bash
# 诊断：NCCL 运行中 SIGTERM 是否会把 Corex GPU 打成 MMU 故障
# !!! 危险：term 模式会故意在 collective 在途时发 SIGTERM（会打出 XID 24）。
# 只在专用/待重启的机器上跑，不要在有人用的共享机器上随手执行。
# 用法: bash diag_nccl_teardown.sh normal|term
# 只使用 TEST_DEVICES（默认 2,3，两张健康 die）
set -uo pipefail

MODE="${1:-normal}"
REPO_ROOT=/data/tangruijing/udtca
TEST_DEVICES="${TEST_DEVICES:-2,3}"
TS="$(date +%Y%m%d_%H%M%S)"
CASE_DIR="$REPO_ROOT/run_log/diag_nccl_teardown_${MODE}_${TS}"
mkdir -p "$CASE_DIR/logs"
LOG="$CASE_DIR/test.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -Is) mode=$MODE devices=$TEST_DEVICES case=$CASE_DIR ==="

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
export PATH=/usr/local/corex/bin:/usr/local/corex/lib64/python3/dist-packages/bin:$PATH
export LD_LIBRARY_PATH=$COREX_TOOLKIT/ixblas/lib64:$COREX_TOOLKIT/ixinfer/lib64:$COREX_PATH/lib64:/usr/local/corex/lib64:$COREX_TOOLKIT/ixdriver/lib64:$COREX_TOOLKIT/ixccl/lib64:/usr/mpi/gcc/openmpi-4.1.7rc1/lib
export LD_PRELOAD="$COREX_TOOLKIT/ixccl/lib64/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-package
source /home/powerone/miniconda3/etc/profile.d/conda.sh
export PATH="/home/powerone/miniconda3/bin:$PATH"
(conda deactivate >/dev/null 2>&1 || true)
conda activate udtca
export PYTHONUNBUFFERED=1
unset CUDA_VISIBLE_DEVICES
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens120f0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG_FILE="$CASE_DIR/logs/nccl.%h.%p.log"
export TORCH_SHOW_CPP_STACKTRACES=0
export TORCH_DISABLE_ADDR2LINE=1
export PYTHONFAULTHANDLER=1

if /usr/bin/ixsmi 2>&1 | grep -q "No running processes found"; then
  echo "gpus idle"
else
  echo "WARN: ixsmi reports running processes"; /usr/bin/ixsmi >> "$CASE_DIR/ixsmi_before.log" 2>&1
fi
/usr/bin/ixsmi > "$CASE_DIR/ixsmi_before.log" 2>&1

KERNEL_START="$(date '+%Y-%m-%d %H:%M:%S')"
journalctl -k -n 3 > "$CASE_DIR/kernel_before.log" 2>&1 || true

export TEST_DEVICES TEST_MODE="$MODE"
torchrun --nproc_per_node=2 --master_port="${MASTER_PORT:-29641}" \
  "$REPO_ROOT/udtca-config/qwen14b/diag/diag_nccl_teardown.py" > "$CASE_DIR/train.log" 2>&1 &
TPID=$!
echo "torchrun pid=$TPID"

READY=0
for _ in $(seq 1 60); do
  if grep -q "first allreduce ok" "$CASE_DIR/train.log" 2>/dev/null; then READY=1; break; fi
  if ! kill -0 "$TPID" 2>/dev/null; then break; fi
  sleep 0.5
done
echo "ready=$READY"
if [ "$READY" != "1" ]; then
  echo "ERROR: NCCL 未就绪"
  tail -40 "$CASE_DIR/train.log" || true
fi

if [ "$MODE" = "term" ]; then
  sleep 3
  echo "=== $(date -Is) SIGTERM torchrun $TPID (allreduce in flight) ==="
  kill -TERM "$TPID" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$TPID" 2>/dev/null || break; sleep 0.5; done
  if kill -0 "$TPID" 2>/dev/null; then
    echo "WARN: torchrun 15s 未退出，SIGKILL"
    kill -KILL "$TPID" 2>/dev/null || true
  fi
else
  echo "=== 等待正常退出 ==="
  for _ in $(seq 1 60); do kill -0 "$TPID" 2>/dev/null || break; sleep 0.5; done
fi
wait "$TPID" 2>/dev/null
echo "torchrun rc=$?"
echo "=== $(date -Is) leftover processes ==="
pgrep -af "diag_nccl_teardown.py" || echo "none"

echo "=== health probe ==="
for DEV in ${TEST_DEVICES//,/ }; do
  OUT=$(CUDA_DEVICE_MAX_CONNECTIONS=1 timeout -k 5 15 python -c "import torch; torch.cuda.set_device($DEV); x=torch.ones(8,device='cuda'); torch.cuda.synchronize(); print('OK', x.sum().item())" 2>&1)
  echo "dev$DEV rc=$? :: $(echo "$OUT" | tail -2 | tr '\n' ' ')"
done

echo "=== kernel log since $KERNEL_START ==="
journalctl -k --since "$KERNEL_START" 2>&1 | grep -iE "XID|mmu|va space" || echo "no XID/mmu/va-space messages"
echo "=== done $(date -Is) ==="
