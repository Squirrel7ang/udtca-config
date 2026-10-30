#!/usr/bin/env bash
# 单机 Qwen2.5-14B 启动器 —— 容器内版本（corex:4.5.0 镜像）。
#
# 按机器区分的入口：
#   cloud/run_single_node_8gpu.sh  云侧 powerone1
#   edge/run_single_node_8gpu.sh    边侧 powerone2
# 两者都转到本脚本，只在本脚本之上预设 GPUS/PP/TP、网卡名和互联 IP。
#
# 四种模式是 2x2 矩阵（POLAR × 量化），各自需要追加的训练参数直接写在下面的
# case 里：
#   baseline     POLAR off, 不量化
#   polar_dense  POLAR on,  不量化      （与 baseline 只差 POLAR）
#   bitscom      POLAR off, 量化        （与 baseline 只差量化）
#   mixed        POLAR on,  量化        （与 polar_dense 只差量化）
#
#   产物在 $CASE_DIR：train.log、logs/nccl.*.log、PRE_RUN.md、EXIT_CODE。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${UDTCA_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
ENTRY="$REPO_ROOT/experiments/qwen14b/run_qwen14b_polar_dp_pp_tp.py"

# 低比特 DP 同步严格串行（正确性优先）。lowbit 的 phase1/phase2/restore 共用
# 同一个 comm，多 bucket 同时在飞时两个 DP rank 的 p2p 投递顺序会分叉，偶发卡死。
export POLAR_EF_LOWMEM_SERIAL_BUCKETS="${POLAR_EF_LOWMEM_SERIAL_BUCKETS:-1}"

MODE="${MODE:-baseline}"
GPUS="${GPUS:-8}"
PP_SIZE="${PP_SIZE:-4}"
TP_SIZE="${TP_SIZE:-1}"
MAX_STEPS="${MAX_STEPS:-3}"
MICRO_BATCHES="${MICRO_BATCHES:-16}"
SEQ_LEN="${SEQ_LEN:-256}"
MASTER_PORT="${MASTER_PORT:-29630}"
DATASET_CONFIG="${DATASET_CONFIG:-sample-10BT}"

DP_SIZE=$((GPUS / (PP_SIZE * TP_SIZE)))
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS - 1)))}"

# 四种模式各自需要追加的训练参数，全放在 METHOD_ARGS 里，torchrun 末尾统一展开。
case "$MODE" in
  baseline)
    RUN_LABEL=baseline_ddp_single
    METHOD_ARGS=(--baseline-mode manual
                 --using-polar false --polar-hook none --comm-timing 0
                 --method none)
    ;;
  polar_dense)
    RUN_LABEL=polar_dense_single
    METHOD_ARGS=(--baseline-mode manual
                 --using-polar true --polar-hook ef_lowmem --comm-timing 8
                 --polar-bucket-numel 64000000 --polar-max-inflight-buckets 1
                 --method none)
    ;;
  bitscom)
    RUN_LABEL=bitscom_quant_single
    METHOD_ARGS=(--baseline-mode manual
                 --using-polar false --polar-hook none --comm-timing 0
                 --method bitscom --bitwidth 2)
    ;;
  mixed)
    RUN_LABEL=polar_bitscom_single
    METHOD_ARGS=(--baseline-mode manual
                 --using-polar true --polar-hook ef_lowmem --comm-timing 8
                 --polar-bucket-numel 64000000 --polar-max-inflight-buckets 1
                 --method bitscom --bitwidth 2)
    ;;
  *) echo "ERROR: 未知 MODE=$MODE（baseline|polar_dense|bitscom|mixed）" >&2; exit 2 ;;
esac

TS="$(date +%Y%m%d_%H%M%S)"
CASE_DIR="${CASE_DIR:-$REPO_ROOT/run_log/single_qwen14b_${MODE}_g${GPUS}_${TS}}"
mkdir -p "$CASE_DIR/logs"

# 每次运行的控制台输出额外落一份到 $LOG_DIR。
LOG_DIR="${LOG_DIR:-$REPO_ROOT/log}"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/$(basename "$CASE_DIR").log"
TEE=(tee -a "$CASE_DIR/train.log" "$RUN_LOG")

{
  echo "# PRE_RUN $TS"
  echo "- runner: run_docker_single_node.sh (container corex 4.5.0)"
  echo "- topology: GPUs=$GPUS DP=$DP_SIZE PP=$PP_SIZE TP=$TP_SIZE MODE=$MODE"
  echo "- mode_args: ${METHOD_ARGS[*]}"
  echo "- devices: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "- host: $(hostname)"
  echo "- steps=$MAX_STEPS micro_batches=$MICRO_BATCHES seq_len=$SEQ_LEN seed=${SEED:-1234}"
  echo "- dataset=HuggingFaceFW/fineweb config=$DATASET_CONFIG"
} > "$CASE_DIR/PRE_RUN.md"

# ---- 容器内 Corex 环境 ----
# 全部用镜像自带的 4.5.0：lib64 下的 cublas/cuinfer/ixattnbkd/nccl，以及
# dist-packages 里对着 libnccl.so.2 编的 torch 2.13。不引入宿主机路径。
export COREX_PATH=/usr/local/corex
export CUDA_HOME="$COREX_PATH"
export CPATH="$COREX_PATH/include"
export LIBRARY_PATH="$COREX_PATH/lib64"
export TRITON_CUDA_SYSROOT="$COREX_PATH"
export BITSCOM_CUDA_COMPILER="$COREX_PATH/bin/clang++"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PATH=/usr/local/corex/bin:/usr/local/corex/lib64/python3/dist-packages/bin:$PATH
export LD_LIBRARY_PATH="$COREX_PATH/lib64:/usr/local/openmpi/lib:/usr/local/lib"
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages

# ---- 通信 ----
export NCCL_IB_DISABLE=1
# 两台机器都由 cloud/edge 脚本给出；跨机建连必须钉到内网那块网卡。
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SOCKET_RETRY_CNT="${NCCL_SOCKET_RETRY_CNT:-600}"
export NCCL_SOCKET_RETRY_SLEEP_MSEC="${NCCL_SOCKET_RETRY_SLEEP_MSEC:-1000}"

# ---- 数据 ----
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY NO_PROXY no_proxy
export HF_ENDPOINT=https://hf-mirror.com
# HF 的元数据 HEAD 请求默认没有超时，镜像不回包时会永久挂住。给个上限。
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-15}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-30}"

# ---- 日志 / 诊断环境（全部写死在脚本里，不依赖 shell 或 ~/.bashrc）----
export PYTHONUNBUFFERED=1
# 崩溃诊断：段错误/abort 直接打 Python 栈；C++ 异常带栈。
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
# C++ 侧默认 WARNING；INFO 会把 CoreX kernel 内部逐算子诊断全打出来，
# 排查时临时用 TORCH_CPP_LOG_LEVEL=INFO 覆盖。
export TORCH_CPP_LOG_LEVEL="${TORCH_CPP_LOG_LEVEL:-WARNING}"
# kineto 的 occupancy 计算器不认 TG 卡（compute 8.1），每 1000 次刷一条失败，
# 纯噪声。级别 ERROR=4，设 5 静音。
export KINETO_LOG_LEVEL="${KINETO_LOG_LEVEL:-5}"
export TORCH_DISABLE_ADDR2LINE=1
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-off}"
# torch 的 NCCL watchdog 会 abort 在途通信（Corex 上就是 XID 24 的路径），
# 进程组超时由 entry 自己管，所以保持关闭。
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-0}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_FR_BUFFER_SIZE=1048576
# 看门狗只管“多久没进度就 revoke 收尾”，不做 kill。首步本来就慢，放宽到 600 s。
export UDTCA_COMM_WATCHDOG_S="${UDTCA_COMM_WATCHDOG_S:-600}"
# 解析数据集目录时 huggingface_hub 的 client 没有读超时，镜像断流会无限挂起；
# entry 会给每个 httpx 请求补读超时并做有限次重试，这里是对应的上限。
export UDTCA_HTTP_READ_TIMEOUT_S="${UDTCA_HTTP_READ_TIMEOUT_S:-30}"
export UDTCA_DATASET_RETRIES="${UDTCA_DATASET_RETRIES:-3}"
# 单次解析超过这个秒数没返回，就把所有线程栈打进日志（定位用，不中断训练）
export UDTCA_DATASET_STALL_S="${UDTCA_DATASET_STALL_S:-120}"
export UDTCA_REVOKE_QUIESCE_S="${UDTCA_REVOKE_QUIESCE_S:-10}"
export UDTCA_REVOKE_BARRIER_S="${UDTCA_REVOKE_BARRIER_S:-5}"
export UDTCA_REVOKE_FILE="${UDTCA_REVOKE_FILE:-}"
# NCCL 全量 INFO 落文件（每 rank 一个）。NCCL_DEBUG_FILE 生效后 NCCL 不再往
# 控制台打，控制台没有 NCCL 行 ≠ 变量没设上 —— 看 $CASE_DIR/logs/。
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG_FILE="$CASE_DIR/logs/nccl.%h.%p.log"

# 把实际生效的环境写进 PRE_RUN.md，事后不用猜当时是什么环境。
{
  echo "- logs: case=$CASE_DIR logs=$CASE_DIR/logs run_log=$RUN_LOG"
  echo "- nccl: debug=$NCCL_DEBUG subsys=$NCCL_DEBUG_SUBSYS file=$NCCL_DEBUG_FILE ib_disable=$NCCL_IB_DISABLE ifname=${NCCL_SOCKET_IFNAME:-<auto>} sym_kernel=$NCCL_SYM_KERNEL p2p_disable=$NCCL_P2P_DISABLE"
  echo "- torch: cpp_log=$TORCH_CPP_LOG_LEVEL dist_debug=$TORCH_DISTRIBUTED_DEBUG faulthandler=$PYTHONFAULTHANDLER trace_buf=$TORCH_NCCL_TRACE_BUFFER_SIZE dump_on_timeout=$TORCH_NCCL_DUMP_ON_TIMEOUT"
  echo "- corex: path=$COREX_PATH pythonpath=$PYTHONPATH ld_library_path=$LD_LIBRARY_PATH"
} >> "$CASE_DIR/PRE_RUN.md"

echo "=== [SINGLE-NODE/DOCKER] qwen14b start $(date -Is)" | "${TEE[@]}"
echo "    mode=$MODE GPUs=$GPUS DP=$DP_SIZE PP=$PP_SIZE TP=$TP_SIZE" | "${TEE[@]}"
echo "    case_dir=$CASE_DIR" | "${TEE[@]}"
echo "    run_log=$RUN_LOG" | "${TEE[@]}"

# 崩溃残留的 0 字节缓存：entry 里用 cache_path.exists() 判断，0 字节也算存在，
# 于是跳过重建、随后 torch.load 直接失败。正常缓存有几十 MB，清掉不算误伤。
find "$REPO_ROOT/experiments/qwen14b/cache" -maxdepth 1 \
  -name '*_labels_unshifted_v2.pt' -size 0 -delete 2>/dev/null || true

cd "$REPO_ROOT"
torchrun \
  --nproc_per_node="$GPUS" \
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
  --dataset-name-or-path HuggingFaceFW/fineweb \
  --dataset-config "$DATASET_CONFIG" \
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
