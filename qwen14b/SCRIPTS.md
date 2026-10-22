# qwen14b 脚本索引

| 类别 | 文件 | 说明 |
|---|---|---|
| 双机 / API（原有，勿改） | `run.py`、`generate_and_run.py`、`_run_baseline.py`、`metrics_api.py`、`start_metrics_api.py`、`API_USAGE.md`、`*.json` | 两机编排 |
| 单机 / 诊断 | `run_single_node.sh`、`run_single_node_8gpu.sh`、`diag/` | 单机训练和排障 |

## 1. 单机训练

唯一入口 `run_single_node.sh`，只连 127.0.0.1：

```bash
cd <repo>/udtca-config/qwen14b
MODE=baseline GPUS=16 bash run_single_node.sh       # 16 卡 DDP baseline
MODE=mixed    GPUS=16 bash run_single_node.sh       # 16 卡 POLAR + bitscom 2bit
MODE=baseline bash run_single_node_8gpu.sh          # 8 卡 DP2×PP4×TP1
```

`run_single_node_8gpu.sh` 只是把默认拓扑设成 `GPUS=8 PP_SIZE=4 TP_SIZE=1`，
并默认 `SKIP_PRECHECK=1`（powerone1 的 ixsmi 读数不可信：8 张物理卡 = 16 个
芯粒，掉卡后设备号会错位）。其余参数透传给 `run_single_node.sh`。
**不要用 `timeout` 或 SIGTERM 停它**：powerone1 上 NCCL 进程被打断会整机卡死，
要停就让脚本自己的通信看门狗走 revoke 路径退出。

四种模式是 2x2 矩阵（POLAR × 量化），脚本里只有 `POLAR_ON`/`QUANT_ON` 两个开关：

| MODE | POLAR | 量化 | 与谁只差一个开关 |
|---|---|---|---|
| `baseline`（默认） | off | off | — |
| `polar_dense` | on | off | 与 baseline 只差 POLAR |
| `bitscom` | off | on | 与 baseline 只差量化 |
| `mixed`（`polar` 是别名） | on | on | 与 polar_dense 只差量化 |

- POLAR on：`--using-polar true --polar-hook ef_lowmem --comm-timing 8 --polar-bucket-numel 64000000 --polar-max-inflight-buckets 1`
- POLAR off：`--using-polar false --polar-hook none --comm-timing 0`
- 量化 on：`--method bitscom --bitwidth 2`；量化 off：`--method none`
- `baseline` 与 `bitscom` 走同一个 `_allreduce_dp_grads_`（逐参数 SUM 后除以 DP），只有后端不同。

拓扑：`GPUS=16 -> PP8×TP1×DP2`，`GPUS=8 -> PP2×TP2×DP2`（8 卡 DP2×PP4×TP1 用
`run_single_node_8gpu.sh`）；可用 `PP_SIZE/TP_SIZE`
覆盖（要求 `PP*TP` 整除 `NPROC_PER_NODE`，默认用 0..GPUS-1 号卡，可用
`CUDA_VISIBLE_DEVICES` 覆盖）。数据集缓存目录和 `--dataset-config` 用 entry 默认值
（`experiments/qwen14b/cache`，config 为空）；新机器没有缓存时会走下载，需要外网。

常用覆盖项：

```bash
MAX_STEPS=3 MICRO_BATCHES=16 SEQ_LEN=256 MASTER_PORT=29630 \
CASE_DIR=/path/to/case COMM_TIMING=8 POLAR_MAX_INFLIGHT=4 \
SKIP_PRECHECK=1 UDTCA_COMM_WATCHDOG_S=60 bash run_single_node.sh
```

- `DRY_RUN=1`：只打印该模式的参数，不启动训练。
- `UDTCA_COMM_WATCHDOG_S`：卡死自动回滚超时（默认 300s，0 关闭）。
- `SKIP_PRECHECK=1`：跳过 `ixsmi` 占卡检查。

产物在 `$CASE_DIR`（默认 `run_log/single_qwen14b_<mode>_g<GPUS>_<时间戳>/`）：
`train.log`、`logs/nccl.*.log`、`PRE_RUN.md`、`EXIT_CODE`。

## 2. 诊断脚本（`diag/`）

| 脚本 | 机器 / 卡数 | 用途 |
|---|---|---|
| `diag/diag_comm_16.sh` | 单机 16 卡 | 16 rank `allreduce + barrier` 冒烟 |
| `diag/diag_health_16.sh` | 单机 16 die | 每 die 独立健康扫描 |
| `diag/diag_nccl_teardown.sh normal\|term` | 2 卡（`TEST_DEVICES`） | 危险：故意在 NCCL 在途时 SIGTERM，复现 XID 24 |
| `diag/diag_lowbit_teardown.py` | 2 rank | lowbit PG all-reduce 正确性 + 干净 teardown |

## 3. 双机

双机走 `python run.py ...`（见 `API_USAGE.md`）。当前 powerone2 不可用，先不要跑。
