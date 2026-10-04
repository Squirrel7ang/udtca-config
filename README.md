# udtca-config

两节点分布式训练实验的**编排器**（不是训练代码）。按实验矩阵生成启动脚本 → 下发到
u62/u210 → `tc` 限速 → 起训 → 回收日志到 `<仓库根>/runtime_log/<case>/`。

只需在 **u62** 上跑一条命令，它会同时操控 u62（node 0）和 u210（node 1）。

## 目录结构

```
udtca-config/
├── common/
│   ├── case_common.py       公共库：常量、两节点命令封装、case 编号、GPU 检查、日志收集
│   └── orchestrator.py      通用编排流程：生成脚本/下发/限速/起训/收日志
├── qwen14b/                 Qwen2.5-14B 实验（PP=8 TP=2 DP=2）
│   ├── default.json         默认超参，被 test.json 的条目覆盖
│   ├── test.json            实验矩阵（index / rate / bit-width / baseline）
│   ├── generate_and_run.py  主入口
│   └── generate_and_run_baseline.py   只跑 baseline 的入口
├── qwenvl8b/                Qwen3-VL-8B 实验（PP=8 TP=1 DP=4），结构与 qwen14b 一一对应
│   ├── default.json  test.json  generate_and_run.py  generate_and_run_baseline.py
└── utils/                   与具体实验无关的辅助脚本
    ├── watch_bandwidth.py   实时看两机之间的网卡带宽
    ├── collect_case_logs.py 把深层 ./log/... 拍平成 tb_scalars/tb_trace/step_csv
    ├── trace_processor.py   从 step_csv/tfevent 算每步耗时与吞吐，做 case 间对比
    ├── download_fineweb.py  按顺序把 fineweb 的前缀 parquet 下到本地
    └── build_comm_opt_data.py  把实验结果整理成交付目录树并打包

> `run_qwen14b_polar_dp_pp_tp.py` 是 `experiments/qwen14b/` 下同名文件的**旧副本**
> （只差一行 print），没有任何代码引用它。留着没删，确认无用后可以删掉。
```

## 前置条件

| 项 | 值 |
|---|---|
| 仓库路径 | `/data1/tangruijing/udtca`（**两节点必须相同**） |
| 调度节点 | u62 = `10.31.10.62`，**必须是 node 0**（torchrun 要 bind `MASTER_ADDR`） |
| 训练节点 | u210 = `10.31.10.210`，通过 `ssh u210` 访问 |
| conda 环境 | `trj-test`（两节点都要有，且 bitscom / polar-sgd 已 `pip install -e .`） |
| 互联 | `ens1f0`，理论带宽 12.5 Gb/s |
| HF 下载 | 走 `HF_ENDPOINT=https://hf-mirror.com`，不需要代理 |

```bash
ssh u210 hostname                    # 免密 ssh 通
ixsmi | grep MiB                     # 两节点的卡都空闲（runner 自己也会查，占用直接退出）
```

## 跑起来

```bash
cd /data1/tangruijing/udtca
python udtca-config/qwen14b/generate_and_run.py          # Qwen14B 全矩阵
python udtca-config/qwenvl8b/generate_and_run.py         # QwenVL8B 全矩阵
```

常用参数（两个实验通用）：

```bash
--only 1,3           # 只跑指定 index
--runtime-log off    # 冒烟：不占 case 编号、不建 case 目录、不归档，只验证能跑通
```

也可以用环境变量 `UDTCA_RUNTIME_LOG=0`（优先级：命令行 > 环境变量 > 默认 on）。
只跑 baseline：

```bash
python udtca-config/qwen14b/generate_and_run_baseline.py
```

每条配置按自己的 `baseline` 字段分派：为真跑仓库里现成的 `0/1_*_baseline_*.sh`，
否则生成 `0/1_train_<exp>_exp{index}.sh`（POLAR + bitscom）再跑。

## 实验矩阵

`test.json` 的 `index` 就是运行顺序，没写的字段从同目录 `default.json` 继承。

| | qwen14b | qwenvl8b |
|---|---|---|
| 模型 | `Qwen/Qwen2.5-14B-Instruct` | `Qwen/Qwen3-VL-8B-Instruct` |
| 并行 | pp 8 / tp 2 / dp 2 | pp 8 / tp 1 / dp 4 |
| 训练入口 | `run_qwen14b_polar_dp_pp_tp.py` | `run_qwenvl8b_polar_dp_pp.py` |
| 数据 | `HuggingFaceFW/fineweb`（流式 + 节点本地 token 缓存） | 随机张量（`--data-seed` 可复现） |
| 当前 case 数 | 4（rate 2/2/1/1 gbit，POLAR 与 baseline 交替） | 4，同上 |

当前超参（`default.json`）：`max-steps=30`、`seq-len=256`、
qwen14b `micro-batches=32`、qwenvl8b `micro-batches=8`；`--log-interval 10`，每 10 步打一行。

## 脚本用法

| 脚本 | 作用 | 用法 |
|---|---|---|
| `qwen14b/generate_and_run.py` | 主入口，按 `test.json` 分派 polar / baseline | `python .../qwen14b/generate_and_run.py [--only 1,3] [--runtime-log off]` |
| `qwen14b/generate_and_run_baseline.py` | 只跑 `baseline=True` 的条目，共用同一套 case 编号 | 同上 |
| `qwenvl8b/generate_and_run.py` | QwenVL8B 主入口，行为同上 | 同上 |
| `qwenvl8b/generate_and_run_baseline.py` | QwenVL8B 只跑 baseline | 同上 |
| `common/case_common.py` | 被上面 4 个 import。定义 `ExperimentSpec`、`SETUP_CMDS`（两节点环境对齐）、`ssh_execute`（base64 传远程脚本）、`check_gpus_free`、`allocate_case`、`write_launch_script` 等 | 不单独运行 |
| `common/orchestrator.py` | 被上面 4 个 import。`run_polar_case` / `run_baseline_case` / `run_experiments`，以及训练脚本模板 `write_train_script` | 不单独运行 |
| `utils/watch_bandwidth.py` | 读两节点 `ens1f0` 的 rx/tx 计数算实时速率（只读，不干扰训练） | `python utils/watch_bandwidth.py [-i 5] [--once] [--iface ens1f0]` |
| `utils/collect_case_logs.py` | 把训练侧写的深层 `./log/{...}/{rank}/tb_scalars` 拍平到 case 根下 | 由 `case_common.finish_case` 自动调用 |
| `utils/trace_processor.py` | 从 `step_csv` 或 tfevent 算平均步时/吞吐，做 case 间对比 | `python utils/trace_processor.py [--case 0001 0002] [--csv out.csv]` |
| `utils/download_fineweb.py` | 按仓库顺序下 fineweb 前缀 parquet，供离线训练 | `python utils/download_fineweb.py [--target-gb 20] [--dry-run]` |
| `utils/build_comm_opt_data.py` | 把实验结果整理成交付目录树并打 zip（只复制不移动） | `python utils/build_comm_opt_data.py [--out /path]` |

## 产物：`runtime_log/`

```
runtime_log/
├── case.json            每个 case 的启动脚本与启动参数
├── environment.json     机器与网络环境说明（交数据时用）
└── 0001/
    ├── launch.sh                      自包含的复现脚本（不依赖 runner）
    ├── 0_train_<exp>_exp1.sh          node0(u62) 实际跑的
    ├── 1_train_<exp>_exp1.sh          node1(u210) 实际跑的（从远程拉回）
    ├── traffic_control_exp1.sh        限速脚本
    ├── train_node0.log / train_node1.log
    ├── tb_scalars/  tb_trace/  step_csv/
```

`case` 编号按**实际运行顺序**递增，不是 `test.json` 的 index。u210 的输出只写
`train_node1.log`，不回显终端（两节点的输出交织太乱）；PP 最后一级在 u62 上有副本，
loss 在终端照样看得到。

手动复现某个 case（两个终端各跑一条）：

```bash
bash runtime_log/0001/launch.sh 0     # u62
bash runtime_log/0001/launch.sh 1     # u210
```

## 注意事项

- **u62 必须是 MASTER**。`node_rank=0` 的 torchrun 会 bind `MASTER_ADDR`，绑别人的地址
  会 `EADDRNOTAVAIL` 然后两边干等。`SETUP_CMDS` 和仓库里的启动脚本默认值都指向
  `10.31.10.62`，双保险。
- **`tc` 只限出向**，两个节点都要执行，跑完必须撤（runner 会自动撤；Ctrl-C 中断时也会
  杀远程训练进程并撤限速）。留着的限速规则会影响这台共享机器上的其他人。
- **日志目录叫 `runtime_log` 不叫 `log`**：仓库根下的 `log/` 是早期实验留下的几十 GB 旧结构。
- **只 copy 不 move**：仓库里已有的 baseline 脚本受版本管理，原地保留；runner 生成的
  polar 脚本归集后会从原件移走，下次同 index 重新生成。
- **GPU 占用检查**在入口最开头跑：`--query-compute-apps` 有进程，或显存 > 256 MiB
  （空载基线 68 MiB）就判定被占用并退出。u210 的 `ixsmi` 不在 PATH 里，走绝对路径
  `/usr/local/corex-4.4.0/bin/ixsmi`。
