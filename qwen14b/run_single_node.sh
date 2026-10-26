#!/usr/bin/env bash
# 单机 Qwen2.5-14B 启动器。四种模式是 2x2 矩阵（POLAR × 量化），
# 模式之间只差下面那两个开关，其余参数完全一致：
#   baseline     POLAR off, 不量化
#   polar_dense  POLAR on,  不量化      （与 baseline 只差 POLAR）
#   bitscom      POLAR off, 量化        （与 baseline 只差量化）
#   mixed        POLAR on,  量化        （与 polar_dense 只差量化）
#
#   MODE=baseline GPUS=16 bash run_single_node.sh
#   GPUS=16 -> PP=8 TP=1 DP=2；GPUS=8 -> PP=2 TP=2 DP=2，可用 PP_SIZE/TP_SIZE 覆盖。
#   产物在 $CASE_DIR：train.log、logs/nccl.*.log、PRE_RUN.md、EXIT_CODE。
#   数据集缓存默认用 entry 的默认值（全量 fineweb），一个参数都不用带；
#   DATASET_CONFIG / DATASET_FILES 只是可选的缩小范围开关。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${UDTCA_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
ENTRY="$REPO_ROOT/experiments/qwen14b/run_qwen14b_polar_dp_pp_tp.py"

# 低比特 DP 同步严格串行（正确性优先，默认打开）。lowbit 的 phase1/phase2/
# restore 共用同一个 comm，多个 bucket 同时在飞时两个 DP rank 的 p2p 投递
# 顺序会分叉，偶发卡死；串行后一个 bucket 完全 restore 才发下一个。
# 性能对照时用 POLAR_EF_LOWMEM_SERIAL_BUCKETS=0 关掉（2gbit 实测快约 12%）。
export POLAR_EF_LOWMEM_SERIAL_BUCKETS="${POLAR_EF_LOWMEM_SERIAL_BUCKETS:-1}"

MODE="${MODE:-baseline}"
GPUS="${GPUS:-16}"
MAX_STEPS="${MAX_STEPS:-3}"
MICRO_BATCHES="${MICRO_BATCHES:-16}"
SEQ_LEN="${SEQ_LEN:-256}"
MASTER_PORT="${MASTER_PORT:-29630}"
# 数据集默认用 fineweb 的 sample-10BT（10B token 子集，冒烟足够）。entry 会按
# 这个 config 去卡片里取 data_files，只列一个小目录；不用再递归扫整个仓库。
# 换数据集用 DATASET_CONFIG/DATASET_FILES，或直接改这里。
DATASET_CONFIG="${DATASET_CONFIG-sample-10BT}"
DATASET_FILES="${DATASET_FILES:-}"

DS_ARGS=(--dataset-name-or-path HuggingFaceFW/fineweb)
[ -n "$DATASET_CONFIG" ] && DS_ARGS+=(--dataset-config "$DATASET_CONFIG")
[ -n "$DATASET_FILES" ] && DS_ARGS+=(--data-files "$DATASET_FILES")

case "$GPUS" in
  16) DEF_PP=8; DEF_TP=1 ;;
  *)  DEF_PP=2; DEF_TP=2 ;;
esac
PP_SIZE="${PP_SIZE:-$DEF_PP}"
TP_SIZE="${TP_SIZE:-$DEF_TP}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$GPUS}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS - 1)))}"

case "$MODE" in
  baseline)    POLAR_ON=0; QUANT_ON=0; RUN_LABEL=baseline_ddp_single ;;
  polar_dense) POLAR_ON=1; QUANT_ON=0; RUN_LABEL=polar_dense_single ;;
  bitscom)     POLAR_ON=0; QUANT_ON=1; RUN_LABEL=bitscom_quant_single ;;
  mixed|polar) POLAR_ON=1; QUANT_ON=1; RUN_LABEL=polar_bitscom_single ;;
  *) echo "ERROR: 未知 MODE=$MODE（baseline|polar_dense|bitscom|mixed）" >&2; exit 2 ;;
esac

METHOD_ARGS=(--baseline-mode manual)
if [ "$POLAR_ON" = "1" ]; then
  METHOD_ARGS+=(--using-polar true --polar-hook ef_lowmem
                --comm-timing "${COMM_TIMING:-8}"
                --polar-bucket-numel 64000000
                --polar-max-inflight-buckets "${POLAR_MAX_INFLIGHT:-1}")
else
  METHOD_ARGS+=(--using-polar false --polar-hook none --comm-timing 0)
fi
if [ "$QUANT_ON" = "1" ]; then
  METHOD_ARGS+=(--method bitscom --bitwidth "${BITWIDTH:-2}")
else
  METHOD_ARGS+=(--method none)
fi

# 只打印参数、不启动训练：DRY_RUN=1 MODE=xxx bash run_single_node.sh
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "MODE=$MODE POLAR_ON=$POLAR_ON QUANT_ON=$QUANT_ON"
  echo "ARGS=${METHOD_ARGS[*]}"
  exit 0
fi

if [ $((PP_SIZE * TP_SIZE)) -eq 0 ] || [ $((NPROC_PER_NODE % (PP_SIZE * TP_SIZE))) -ne 0 ]; then
  echo "ERROR: 拓扑不整除: NPROC=$NPROC_PER_NODE PP=$PP_SIZE TP=$TP_SIZE" >&2
  exit 2
fi
DP_SIZE=$((NPROC_PER_NODE / (PP_SIZE * TP_SIZE)))
VISIBLE_COUNT="$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"
if [ "$VISIBLE_COUNT" != "$NPROC_PER_NODE" ]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES 只有 $VISIBLE_COUNT 张卡" >&2
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
CASE_DIR="${CASE_DIR:-$REPO_ROOT/run_log/single_qwen14b_${MODE}_g${GPUS}_${TS}}"
mkdir -p "$CASE_DIR/logs"

# 每次运行的控制台输出额外落一份到 $LOG_DIR，省掉外面手写 `| tee`。
# 默认是仓库下的 log/（不是根目录 /udtca/log：那需要 root 才能创建）。
LOG_DIR="${LOG_DIR:-$REPO_ROOT/log}"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/$(basename "$CASE_DIR").log"
TEE=(tee -a "$CASE_DIR/train.log" "$RUN_LOG")

# 占卡检查。ixsmi 位置随机型不同，先找一遍。
IXSMI_BIN="${IXSMI_BIN:-}"
if [ -z "$IXSMI_BIN" ]; then
  for c in "$(command -v ixsmi 2>/dev/null)" /usr/bin/ixsmi /usr/local/corex/bin/ixsmi; do
    [ -n "$c" ] && [ -x "$c" ] && { IXSMI_BIN="$c"; break; }
  done
fi
# powerone1 上 ixsmi 的读数不可信（8 张物理卡 16 个芯粒，掉过卡后设备号错位，
# 永远看不到 "No running processes found"），在这台机器上默认跳过，免得每次都要
# 手动带 SKIP_PRECHECK=1；占卡情况由用户看 NOTE 文件和 ixsmi 原始输出确认。
if [ "${SKIP_PRECHECK:-$(case "$(hostname)" in powerone*) echo 1;; *) echo 0;; esac)}" != "1" ]; then
  if [ -z "$IXSMI_BIN" ]; then
    echo "WARN: 找不到 ixsmi，跳过占卡检查（可设 IXSMI_BIN）" >&2
  elif ! "$IXSMI_BIN" 2>&1 | grep -q "No running processes found"; then
    echo "ERROR: GPU 上已有进程，退出" >&2
    exit 4
  fi
fi

{
  echo "# PRE_RUN $TS"
  echo "- topology: GPUs=$GPUS DP=$DP_SIZE PP=$PP_SIZE TP=$TP_SIZE MODE=$MODE"
  echo "- matrix: POLAR_ON=$POLAR_ON QUANT_ON=$QUANT_ON (bitwidth=${BITWIDTH:-2})"
  echo "- devices: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "- host: $(hostname)"
  echo "- steps=$MAX_STEPS micro_batches=$MICRO_BATCHES seq_len=$SEQ_LEN seed=${SEED:-1234}"
  echo "- dataset=HuggingFaceFW/fineweb config=${DATASET_CONFIG:-<none>} files=${DATASET_FILES:-<none>}"
} > "$CASE_DIR/PRE_RUN.md"

if [ -f ~/.bashrc ]; then
  PS1="${PS1:-ustca-init}"; _flags="$-"; set +eu
  . ~/.bashrc >/dev/null 2>&1 || true
  case "$_flags" in *e*) set -e;; esac; case "$_flags" in *u*) set -u;; esac
  unset _flags PS1
fi
# /usr/local/corex 指向本机实际的 Corex 版本（4.4.0 / 4.5.0 都适用）
export COREX_PATH="${UDTCA_COREX_PATH:-/usr/local/corex}"
export COREX_TOOLKIT=/data/tianshu/20260720/corex/corex/corex-toolkit
export CUDA_HOME="$COREX_PATH"
export CPATH="$COREX_PATH/include:${CPATH:-}"
export LIBRARY_PATH="$COREX_PATH/lib64:${LIBRARY_PATH:-}"
export TRITON_CUDA_SYSROOT="$COREX_PATH"
export BITSCOM_CUDA_COMPILER="$COREX_PATH/bin/clang++"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_RETRY_CNT="${NCCL_SOCKET_RETRY_CNT:-600}"
export NCCL_SOCKET_RETRY_SLEEP_MSEC="${NCCL_SOCKET_RETRY_SLEEP_MSEC:-1000}"
export PATH=/usr/local/corex/bin:/usr/local/corex/lib64/python3/dist-packages/bin:$PATH
export LD_LIBRARY_PATH=$COREX_TOOLKIT/ixblas/lib64:$COREX_TOOLKIT/ixinfer/lib64:$COREX_PATH/lib64:/usr/local/corex/lib64:$COREX_TOOLKIT/ixdriver/lib64:$COREX_TOOLKIT/ixccl/lib64:/usr/mpi/gcc/openmpi-4.1.7rc1/lib
if [ -e "$COREX_TOOLKIT/ixccl/lib64/libnccl.so.2" ]; then
  export LD_PRELOAD="$COREX_TOOLKIT/ixccl/lib64/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages
CONDA_ROOT="${UDTCA_CONDA_ROOT:-}"
if [ -z "$CONDA_ROOT" ]; then
  for c in /root/miniconda3 /home/powerone/miniconda3; do
    [ -d "$c" ] && { CONDA_ROOT="$c"; break; }
  done
fi
source "${CONDA_ROOT:?找不到 conda 根目录，请设置 UDTCA_CONDA_ROOT}/etc/profile.d/conda.sh"
export PATH="$CONDA_ROOT/bin:$PATH"
(conda deactivate >/dev/null 2>&1 || true)
conda activate udtca
export PYTHONUNBUFFERED=1

export NCCL_IB_DISABLE=1
# 网卡名各机器不同（u13/u177 是 ens1f0），默认取默认路由的网卡，可用
# NCCL_SOCKET_IFNAME 覆盖。写死一个不存在的网卡会让 NCCL 在 bootstrap 阶段
# 直接失败（"no socket interface found"），rank0 退出码 1 且没有 traceback。
if [ -z "${NCCL_SOCKET_IFNAME:-}" ]; then
  NCCL_SOCKET_IFNAME="$(ip -o -4 route show default 2>/dev/null | awk '{print $5; exit}')"
  NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$(ip -o -4 addr show scope global 2>/dev/null | awk '$2!="lo"{sub("@.*","",$2); print $2; exit}')}"
fi
export NCCL_SOCKET_IFNAME
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY NO_PROXY no_proxy
export HF_ENDPOINT=https://hf-mirror.com
# HF 的元数据 HEAD 请求默认没有超时：镜像不回包也不断链时会永久挂住
# （2026-10-23 在 powerone1 上卡了 11 分钟，模型都没开始建）。给个上限。
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-15}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-30}"
export CUDA_VISIBLE_DEVICES

# 诊断日志 + 卡死自动回滚（revoke 后各 rank 自行退出）
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
# C++ 侧日志默认 WARNING。开成 INFO 会把 CoreX kernel 内部的逐算子诊断
# （如 COREX::is_valid_transpose_io_tensor_view）全打出来，量大且无从下手；
# 排查时临时用 TORCH_CPP_LOG_LEVEL=INFO 覆盖即可。
export TORCH_CPP_LOG_LEVEL="${TORCH_CPP_LOG_LEVEL:-WARNING}"
# kineto 有自己的一套 logger，不受 TORCH_CPP_LOG_LEVEL 控制。CoreX 的 occupancy
# 计算器只认 compute 3/5/6/7，TG 卡（8.1）上每个 kernel 都算失败报
# "Failed to calculate occupancy, status = 2"（每 1000 次打一条）；profiler 本身
# 正常，只是 trace 里少了 occupancy 指标，纯噪声。kineto 级别 ERROR=4，设 5 静音；
# 要排查 kineto 自身的问题时设回 0/1。
export KINETO_LOG_LEVEL="${KINETO_LOG_LEVEL:-5}"
export TORCH_DISABLE_ADDR2LINE=1
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-off}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-0}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
export UDTCA_COMM_WATCHDOG_S="${UDTCA_COMM_WATCHDOG_S:-60}"
# 解析数据集目录时 huggingface_hub 的 client 没有读超时，镜像断流会无限挂起；
# entry 会给每个 httpx 请求补一个读超时并做有限次重试，这里是对应的上限。
export UDTCA_HTTP_READ_TIMEOUT_S="${UDTCA_HTTP_READ_TIMEOUT_S:-30}"
export UDTCA_DATASET_RETRIES="${UDTCA_DATASET_RETRIES:-3}"
# 单次解析超过这个秒数没返回，就把所有线程栈打进日志（定位用，不中断训练）
export UDTCA_DATASET_STALL_S="${UDTCA_DATASET_STALL_S:-120}"
export UDTCA_REVOKE_QUIESCE_S="${UDTCA_REVOKE_QUIESCE_S:-10}"
export UDTCA_REVOKE_BARRIER_S="${UDTCA_REVOKE_BARRIER_S:-5}"
export UDTCA_REVOKE_FILE="${UDTCA_REVOKE_FILE:-}"
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_FR_BUFFER_SIZE=1048576
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG_FILE="$CASE_DIR/logs/nccl.%h.%p.log"

echo "=== [SINGLE-NODE] qwen14b start $(date -Is)" | "${TEE[@]}"
echo "    mode=$MODE POLAR_ON=$POLAR_ON QUANT_ON=$QUANT_ON GPUs=$GPUS DP=$DP_SIZE PP=$PP_SIZE TP=$TP_SIZE" | "${TEE[@]}"
echo "    case_dir=$CASE_DIR" | "${TEE[@]}"
echo "    run_log=$RUN_LOG" | "${TEE[@]}"

# 崩溃残留的 0 字节缓存：entry 里用 cache_path.exists() 判断，0 字节也算存在，
# 于是跳过重建、随后 torch.load 直接失败。正常缓存有几十 MB，清掉不算误伤。
find "$REPO_ROOT/experiments/qwen14b/cache" -maxdepth 1 \
  -name '*_labels_unshifted_v2.pt' -size 0 -delete 2>/dev/null || true

cd "$REPO_ROOT"
torchrun \
  --nproc_per_node="$NPROC_PER_NODE" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
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
  "${DS_ARGS[@]}" \
  --text-field text \
  --run-label "$RUN_LABEL" \
  "${METHOD_ARGS[@]}" \
  --num-workers 0 \
  --disable-profiler false \
  --debug-nan-steps 2 \
  --save-dir "$CASE_DIR/ckpt" \
  --step-log-dir "$CASE_DIR/step_csv" \
  2>&1 | "${TEE[@]}"
RC=${PIPESTATUS[0]}
echo "EXIT_CODE=$RC" | tee -a "$CASE_DIR/EXIT_CODE" "$CASE_DIR/train.log" "$RUN_LOG"
exit "$RC"
