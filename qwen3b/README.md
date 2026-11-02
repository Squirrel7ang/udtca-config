# qwen3b —— Qwen2.5-3B 的编排/启动器

`udtca-config/qwen14b/` 的**同构副本**，模型换成 `Qwen/Qwen2.5-3B-Instruct`。
用途：同一套 PP/TP/DP 代码、同一个编排流程，用更小的模型把对照实验跑快，
便于在**不同卡数 / 不同并行度**上快速扫结果，再拿结论回到 14B 上验证。

## 和 qwen14b 目录的对应关系

| 文件 | 作用 | 与 14B 的差别 |
|---|---|---|
| `default.json` | 默认超参（pp8/tp2、micro-batches 32、max-steps 30、seq 256） | 无 |
| `test.json` / `run_cases.json` | 实验矩阵（2gbit/1gbit × polar/baseline） | 无 |
| `generate_and_run.py` | **主入口**：按 `test.json` 分派 polar / baseline | 模型名、`master_port=29520` |
| `run.py` | 固定 4 case 一键跑 + 汇总（写 `run_log/run_<ts>/summary.*`） | 同上 |
| `_run_baseline.py` | 内部脚本：只跑 baseline 条目 | 同上 |
| `run_docker.py` | **跨机容器版启动器**（宿主机上跑）：生成 node 脚本 → 起训 → 分析 | 入口路径、模型名、`--master-port 29694`、清理匹配 `run_qwen3b*` |
| `run_docker_single_node.sh` | 单机容器版启动器 | 入口路径、`MASTER_PORT 29730` |
| `cloud/` `edge/` | 云侧(powerone1,node0) / 边侧(powerone2,node1) 的单机与跨机脚本 | 入口路径、端口 +100 |
| `dp_test.py` `dp_test_bigpair.py` `nccl_xnode_test.py` | 与模型无关的通信自检 | 仅路径/端口 |
| `metrics_api.py` `start_metrics_api.py` | 实时指标 API（读 `[rt-metric]` 日志） | 仅路径/命名 |
| `capture_stall.sh` | 卡死现场抓取（只读） | 匹配 `run_qwen3b*` |

**故意没有复制**：`qwen14b/diag/`（14B 专用的探针/补丁脚本）、
`API_USAGE.md` / `SCRIPTS.md` / `XNODE-USAGE.md`（14B 的历史运行说明）。
需要那一套时看 `udtca-config/qwen14b/`。

## 端口分配（和 14B 错开，大小模型可以同时跑）

| 用途 | qwen14b | qwen3b |
|---|---|---|
| `generate_and_run.py` / `run.py` / 手写 `0_\|1_train_*.sh` | 29500 / 29501 | 29520 / 29521 |
| `run_docker.py` | 29693 | 29694 |
| `run_docker_single_node.sh` / `cloud\|edge/run_xnode.sh` | 29630 | 29730 |
| `run_nccl_xnode.sh` | 29631 | 29731 |
| `run_dp_test.sh` | 29640 | 29740 |
| `run_dp_bigpair.sh` | 29641 | 29741 |

> 29630 在 powerone1 上会被 kube-apiserver 的临时端口占用过，3B 里一并挪开了。

## 首跑前：把模型拉到两台节点的本地缓存

训练入口固定 `local_files_only=True`，模型不在本地 HF 缓存里会直接报错。
两台节点（各自的容器里）各拉一次即可（约 6.2 GB）：

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen2.5-3B-Instruct
```

## 模型约束（3B 特有）

- **TP ⊆ {1, 2}**：Qwen2.5-3B 是 GQA（16 个 attention 头 / 2 个 KV 头），
  TP>2 时每卡分不到整 KV 头，入口脚本启动时直接报错退出。
- **PP 不要求整除层数**：36 层，pp=8 时前 4 个 stage 各 5 层、后 4 个各 4 层。
- DP 由 `world_size / (pp*tp)` 自动推出，16 卡/机、两机 32 卡时
  pp8×tp2 → dp2，pp4×tp2 → dp4，pp8×tp1 → dp4。

## 怎么跑

主流程（自动生成脚本 → 下发 → 起训 → 收日志 → 汇总）：

```bash
python udtca-config/qwen3b/generate_and_run.py            # test.json 全矩阵
python udtca-config/qwen3b/generate_and_run.py --only 1 --runtime-log off   # 冒烟
python udtca-config/qwen3b/run.py --max-steps 3 --no-tc   # 固定 4 case + 汇总
```

跨机 32 卡（每机 16 卡，在 **powerone1 宿主机**上跑，容器里没有 docker）：

```bash
cd /data1/tangruijing/udtca/udtca-config/qwen3b
python3 run_docker.py --modes polar --max-steps 5            # 默认不开 profiler
python3 run_docker.py --modes baseline,polar --max-steps 5 --gpus 8 --pp 4 --tp 2
python3 run_docker.py --dry-run                              # 只看生成的脚本
```

不同卡数/并行度就调 `--gpus/--pp/--tp`（同一台机器的两台节点卡数必须相同），
例如 8 卡/机时 `--gpus 8 --pp 4 --tp 2`（dp=2）、`--gpus 8 --pp 4 --tp 1`（dp=4）。
单机自测用容器内的 `run_docker_single_node.sh`（`GPUS=8 PP=4 TP=2 MODE=baseline bash ...`）。

运行前的检查照旧：先读 `~/NOTE-NOTE-NOTE-Please-Readme-First.txt` 看有没有人预约，
再看卡有没有被别的租户占用（`ixsmi`、`docker ps` 里的 `k8s_llm*`、`journalctl -k` 的
XID 24 / page fault），有占用就停下汇报，别用 `--skip-card-check` 硬起。
