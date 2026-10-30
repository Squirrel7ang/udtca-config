# qwen14b 脚本索引

| 类别 | 文件 | 说明 |
|---|---|---|
| 双机 / API（原有，勿改） | `run.py`、`generate_and_run.py`、`_run_baseline.py`、`metrics_api.py`、`start_metrics_api.py`、`API_USAGE.md`、`*.json` | 两机编排 |
| 单机入口 | `cloud/run_single_node_8gpu.sh`、`edge/run_single_node_8gpu.sh` | 按机器选一个，内部转调 `run_docker_single_node.sh` |
| 单机主体 | `run_docker_single_node.sh` | 容器内实际启动 torchrun 的脚本 |
| 诊断 | `diag/` | 单机排障 |

## 1. 单机训练（容器内执行）

容器 `trj-udtca` 里，先进到脚本目录，再按机器选入口：

```bash
cd <repo>/udtca-config/qwen14b
MODE=baseline bash cloud/run_single_node_8gpu.sh   # 云侧 powerone1
MODE=baseline bash edge/run_single_node_8gpu.sh     # 边侧 powerone2
```

两个入口只预设这几项，其余全部透传给 `run_docker_single_node.sh`：

| 变量 | cloud（powerone1） | edge（powerone2） |
|---|---|---|
| `GPUS` / `PP_SIZE` / `TP_SIZE` | 8 / 4 / 1（DP=2） | 8 / 4 / 1（DP=2） |
| `NCCL_SOCKET_IFNAME` | `ens120f0` | `enx00e04c680003` |
| `UDTCA_NODE0_IP` | `10.205.200.14` | `10.205.200.14` |
| `UDTCA_NODE1_HOST` | `10.205.200.16` | `10.205.200.16` |
| `NCCL_P2P_DISABLE` | `1` | `1` |

`NCCL_P2P_DISABLE=1` 是因为两台机器的 P2P 直连都不稳，统一走间接中继。

四种模式是 2x2 矩阵（POLAR × 量化），一次 `MODE` 直接决定 `METHOD_ARGS` 和
`RUN_LABEL`，脚本里没有中间开关：

| MODE | POLAR | 量化 | RUN_LABEL |
|---|---|---|---|
| `baseline`（默认） | off | off | `baseline_ddp_single` |
| `polar_dense` | on | off | `polar_dense_single` |
| `bitscom` | off | on | `bitscom_quant_single` |
| `mixed` | on | on | `polar_bitscom_single` |

低比特 DP 同步默认**严格串行**（`POLAR_EF_LOWMEM_SERIAL_BUCKETS=1`）。lowbit 的
phase1/phase2/restore 共用同一个 comm，多个 bucket 同时在飞时两个 DP rank 的 p2p
投递顺序会分叉、偶发卡死；串行后一个 bucket 完全 restore 才发下一个。正确性优先，
关掉大约快 12%（u13+u177 2gbit 实测）。

**不要用 `timeout` 或 SIGTERM 停它**：Corex 上 NCCL 进程被打断会让卡永久挂死
（XID 24），只能重启机器。要停就让 entry 自己的通信看门狗走 revoke 路径退出。

常用覆盖项（都能放在命令行前缀）：

```bash
MAX_STEPS=3 MICRO_BATCHES=16 SEQ_LEN=256 MASTER_PORT=29630 \
CASE_DIR=/path/to/case DATASET_CONFIG=sample-10BT \
MODE=mixed bash cloud/run_single_node_8gpu.sh
```

- `UDTCA_COMM_WATCHDOG_S`：卡死自动回滚超时（默认 600s）。

数据集固定为 `HuggingFaceFW/fineweb` + `--dataset-config sample-10BT`（10B token
子集，冒烟足够）；模型是本地随机初始化，不下载权重。镜像走 `HF_ENDPOINT=https://hf-mirror.com`。

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

双机走 `python run.py ...`（见 `API_USAGE.md`）。
