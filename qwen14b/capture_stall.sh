#!/usr/bin/env bash
# 卡死现场抓取（容器内执行）。只读：不 kill、不 abort、不改任何进程状态。
#
#   bash capture_stall.sh [输出目录]
#
# 抓什么：
#   1. 当天 xnode_baseline_* 的 train.log 原样拷贝
#   2. 每个训练 rank 的 py-spy Python 栈（含 locals）
#   3. NCCL 日志尾部
#   4. ixsmi / ps 快照
#
# 抓完之后再决定重启。重启前再在宿主机补一份 dmesg。
set -uo pipefail

TS="$(date +%Y%m%d_%H%M%S)"
OUT="${1:-/data/tangruijing/udtca/run_log/stall_$(hostname)_${TS}}"
mkdir -p "$OUT/logs"

d="$(ls -dt /data/tangruijing/udtca/run_log/xnode_baseline* 2>/dev/null | head -1)"
echo "source_case_dir=$d" | tee "$OUT/source_case_dir.txt"

if [ -n "${d:-}" ] && [ -f "$d/train.log" ]; then
  cp -a "$d/train.log" "$OUT/train.log"
  if [ -d "$d/logs" ]; then
    for f in "$d"/logs/nccl.*.log; do
      [ -f "$f" ] || continue
      tail -c 2000000 "$f" > "$OUT/logs/$(basename "$f").tail" 2>/dev/null
    done
  fi
fi

# 每个 rank 的 Python 栈
pids=$(ps -ef | grep 'run_qwen14b_polar_dp_pp_tp.py' | grep -v grep | awk '{print $2}')
echo "rank pids: $pids" | tee "$OUT/pids.txt"
for pid in $pids; do
  ( py-spy dump --pid "$pid" --locals > "$OUT/pyspy.rank_$pid.txt" 2>&1 ) &
done
wait

ixsmi > "$OUT/ixsmi.txt" 2>&1
ps -ef > "$OUT/ps.txt" 2>&1

echo "CAPTURED=$OUT"
