# qwen14b 脚本索引

本目录混着两套东西，先分清：

| 类别 | 文件 | 说明 |
|---|---|---|
| **双机 / API（原有，未改动）** | `run.py`、`generate_and_run.py`、`_run_baseline.py`、`metrics_api.py`、`start_metrics_api.py`、`API_USAGE.md`、`*.json` | 两机编排走 `run.py`（orchestrator 自动拉起两台机器），不要再改 |
| **单机 / 诊断（本次整理）** | `run_single_node.sh`、`diag/` | 单机训练和排障用 |

---

## 1. 单机训练启动

只有一个入口：**`run_single_node.sh`**（只连 127.0.0.1，不跨机）。

```bash
cd /data/tangruijing/udtca/udtca-config/qwen14b

# 单机 16 卡，DDP baseline
MODE=baseline GPUS=16 bash run_single_node.sh

# 单机 16 卡，POLAR + bitscom 2bit
MODE=polar GPUS=16 bash run_single_node.sh

# 单机 8 卡（例如只用 8~15 卡）
MODE=polar GPUS=8 CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 bash run_single_node.sh
```

### 模式（MODE）

| MODE | 含义 | 关键参数 |
|---|---|---|
| `baseline`（默认） | DDP baseline，不用 POLAR | `--using-polar false --baseline-mode manual --method none` |
| `polar` / `bitscom` | POLAR + bitscom 2bit 低比特 DP 通信 | `--using-polar true --polar-hook ef_lowmem --method bitscom --bitwidth 2` |
| `polar_dense` | POLAR 框架 + dense 通信（对照组） | `--using-polar true --method none` |

### GPU 数 / 拓扑

| GPUS | 默认 PP | 默认 TP | DP |
|---|---|---|---|
| 16 | 4 | 2 | 2 |
| 8 | 2 | 2 | 2 |

- 默认用第 `0..GPUS-1` 张卡，可用 `CUDA_VISIBLE_DEVICES` 覆盖（数量必须等于 `NPROC_PER_NODE`）。
- 想复现旧的 DP=1 配置：`PP_SIZE=8 GPUS=16`（PP8×TP2×DP1）。
- 脚本会检查 `PP*TP` 能整除 `NPROC_PER_NODE`，并打印/写下拓扑。

### 常用覆盖项

```bash
MAX_STEPS=3 MICRO_BATCHES=16 SEQ_LEN=256 DATASET_CONFIG=sample-10BT \
MASTER_PORT=29630 CASE_DIR=/path/to/case \
COMM_TIMING=8 POLAR_MAX_INFLIGHT=4 \
UDTCA_COMM_WATCHDOG_S=60 \
bash run_single_node.sh
```

- `UDTCA_COMM_WATCHDOG_S`：自动回滚超时（默认 60s；设 `0` 关闭，卡死时就只能人工处理）。
- `SKIP_PRECHECK=1`：跳过 `ixsmi` 占卡检查（不建议）。

### 产出（`$CASE_DIR`）

默认 `run_log/single_qwen14b_<mode>_g<GPUS>_<时间戳>/`：

```
PRE_RUN.md       机器状态 + 拓扑 + 模式
train.log        训练输出
logs/nccl.*.log  全量 NCCL 日志
ixsmi_before.log 启动前整卡状态
EXIT_CODE        进程退出码（不依赖外部信号）
```

---

## 2. 诊断脚本（`diag/`）

| 脚本 | 机器 / 卡数 | 用途 |
|---|---|---|
| `diag/diag_comm_16.sh` | 单机 16 卡 | 16 rank `allreduce + barrier` 冒烟测试，正常退出 |
| `diag/diag_health_16.sh` | 单机 16 die | 每 die 独立进程健康扫描（单卡 12s 超时），重启后确认卡是否恢复 |
| `diag/diag_nccl_teardown.sh normal\|term` | 单机 2 卡（`TEST_DEVICES`，默认 2,3） | **危险**：`term` 模式故意在 NCCL 在途时发 SIGTERM，复现 XID 24。只用于验证问题，不要在共享机器上随手跑 |
| `diag/diag_nccl_teardown.py` | 2 rank | 上面脚本的 worker，一般不单独调用 |
| `diag/diag_lowbit_teardown.py` | 2 rank | bitscom lowbit PG 的 all-reduce 正确性 + 干净 teardown 验证 |

诊断脚本的输出都在 `run_log/` 下，带时间戳。

---

## 3. 双机（不要用单机脚本）

双机请走原有 API：

```bash
cd /data/tangruijing/udtca/udtca-config/qwen14b
python run.py ...     # 详见 API_USAGE.md
```

- 机器、网卡、目录等在两机编排脚本（`common/case_common.py`、`common/orchestrator.py`）里统一配置。
- 当前 powerone2 不可用，跨机用例先不要跑。
