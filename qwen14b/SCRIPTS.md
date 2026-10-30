# qwen14b 脚本索引

| 类别 | 文件 | 说明 |
|---|---|---|
| 双机 / API（原有，勿改） | `run.py`、`generate_and_run.py`、`_run_baseline.py`、`metrics_api.py`、`start_metrics_api.py`、`API_USAGE.md`、`*.json` | 两机编排 |
| 单机入口 | `cloud/run_single_node_8gpu.sh`、`edge/run_single_node_8gpu.sh` | 按机器选一个，内部转调 `run_docker_single_node.sh` |
| 单机主体 | `run_docker_single_node.sh` | 容器内实际启动 torchrun 的脚本 |
| 跨机连通性 | `cloud/run_nccl_xnode.sh`、`edge/run_nccl_xnode.sh` + `nccl_xnode_test.py` | 两机各 1 卡，先测 NCCL 再测 bitscom |
| 跨机训练（手动） | `cloud/run_xnode.sh`、`edge/run_xnode.sh` | 两台各 16 卡、PP=8 TP=2；各自容器内执行，见 `XNODE-USAGE.md` |
| 跨机训练（一键） | `run_docker.py` | 在 powerone1 **宿主机**执行，自动起两机容器跑 mixed/baseline 并分析 trace |
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
| `NCCL_P2P_DISABLE` | `0` | `0` |

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

## 2. 跨机连通性（两机各 1 卡，world=2）

两边各跑一个入口，先起哪边都行（rendezvous 会等对端）：

```bash
# powerone1（node0）
bash cloud/run_nccl_xnode.sh --skip-lowbit  # 只测 NCCL，不 import bitscom
# powerone2（node1）
bash edge/run_nccl_xnode.sh
```

`nccl_xnode_test.py` 的用例顺序：barrier → 阻塞 send/recv（环形）→ 异步
isend/irecv（bf16 PP 激活形状）→ all_reduce → all_gather → all_to_all_single →
bitscom 低比特 all_reduce（fp32 + bf16，跟 NCCL 全精度比相对误差）。
全部通过会打 `=== ALL CHECKS PASSED ===`；失败打 `[FAIL] STEP n` 加栈。

进程组自带 `--timeout-s`（默认 300s），超时由 c10d 自己抛异常。**不要 kill。**

## 3. 跨机训练（`run_docker.py`）

在能 ssh 到两台机器的机器上跑（不是容器里）：

```bash
cd udtca-config/qwen14b
python run_docker.py --mode baseline                 # 2×8 卡，DP=4 PP=4 TP=1
python run_docker.py --mode mixed --max-steps 3
python run_docker.py --dry-run                       # 只打印两边命令
```

它做的事：远端建 case 目录 → ssh 起 node1（后台，日志落盘）→ 等 20s →
ssh 起 node0（实时回显）→ 拉两边的 `train.log`/`EXIT_CODE` → 打汇总。
启动仍走 `run_docker_single_node.sh`，本脚本只下发
`NNODES=2 / NODE_RANK / MASTER_ADDR / CASE_DIR`。
`--timeout-s` 只是停止等待，**不发信号**。

## 4. 诊断脚本（`diag/`）

| 脚本 | 机器 / 卡数 | 用途 |
|---|---|---|
| `diag/diag_comm_16.sh` | 单机 16 卡 | 16 rank `allreduce + barrier` 冒烟 |
| `diag/diag_health_16.sh` | 单机 16 die | 每 die 独立健康扫描 |
| `diag/diag_nccl_teardown.sh normal\|term` | 2 卡（`TEST_DEVICES`） | 危险：故意在 NCCL 在途时 SIGTERM，复现 XID 24 |
| `diag/diag_lowbit_teardown.py` | 2 rank | lowbit PG all-reduce 正确性 + 干净 teardown |

## 5. 双机（旧路径）

宿主机 conda 那套 `python run.py ...` 仍保留（见 `API_USAGE.md`）。
