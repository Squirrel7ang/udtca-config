# udtca-config

Qwen2.5-14B 两节点分布式训练的实验编排脚本，对应仓库
[github.com/Aerithy/udtca](https://github.com/Aerithy/udtca)。

本目录**不是**训练代码本身，而是"调度器"：它按 `test.json` 的矩阵生成启动脚本、
下发到两个节点、用 `tc` 限速、起训、回收日志，最后把每个 case 的产物归集到
`<仓库根>/runtime_log/<case>/`，方便整包上交。

---

## 1. 运行环境

| 项 | 值 |
|---|---|
| 调度节点 | **u62**（`10.31.10.62`），本机 |
| 训练节点 | **u210**（`10.31.10.210`），通过 `ssh u210` 访问（`~/.ssh/config` 里有别名） |

### 节点角色：u62 是 MASTER

跑之前必须保证这三件事一致，否则会**一个 worker 都不起、两边干等**（踩过）：

| | u62 | u210 |
|---|---|---|
| 跑哪个脚本 | `0_train_*`（`--node_rank=0`） | `1_train_*`（`--node_rank=1`） |
| 角色 | **MASTER_NODE**：绑定 `MASTER_ADDR:MASTER_PORT` 当 rendezvous 服务端 | 客户端，连过去 |
| `MASTER_ADDR` | `10.31.10.62` | `10.31.10.62` |
| `MASTER_PORT` | polar 用 29500，baseline 用 29501 | 同左 |

**为什么**：`node_rank=0` 的 torchrun 会 **bind** `MASTER_ADDR`，绑的是别人的地址会
`EADDRNOTAVAIL`，然后两边永远等下去。所以 `MASTER_ADDR` 必须指向跑 `node_rank=0`
的那台机器，而 `node_rank=0` 就是本机 u62。

对齐方式有两层，双保险：
1. `case_common.SETUP_CMDS` 里 `export MASTER_ADDR=10.31.10.62`，runner 跑训练前注入；
2. `experiments/qwen14b/` 下那 4 个训练脚本的 `${MASTER_ADDR:-...}` 默认值也改成了
   `10.31.10.62`，手动跑也不会错。
| 仓库路径 | `/data1/tangruijing/udtca`（**两个节点上必须相同**） |
| conda 环境 | `trj-test`（两节点都要有） |
| 互联 | `ens1f0` 接入交换机，理论带宽 12.5 Gb/s |
| 模型 / 数据集 | `Qwen/Qwen2.5-14B-Instruct` / `HuggingFaceFW/fineweb` |
| 并行 | 2 节点 x 16 卡，DP=2, PP=8, TP=2 |
| HF 下载 | 走 `HF_ENDPOINT=https://hf-mirror.com`，不需要任何代理工具 |

**只需要在 u62 上跑一条命令**，它会同时操控 u62（node 0）和 u210（node 1）。

### 前置检查清单

```bash
# 1) 免密 ssh 通
ssh u210 hostname

# 2) 两节点的 GPU 都空闲（脚本也会自动查，占用会直接中止）
ixsmi --query-gpu=index,memory.used --format=csv,noheader

# 3) 两节点的 bitscom / polar-sgd 都已编译安装（trj-test 环境里）
#    装了才有的 .pth：__editable__.bitscom-0.1.0.pth / __editable__.psgd-0.1.0.pth
```

子模块需要时重新编译（两边都要做）：

```bash
# bitscom —— 必须给 CUDA_HOME + 编译器；u210 上还要额外给 NCCL 路径，
# 否则 conda 的 compiler_compat/ld 找不到 -lnccl
cd /data1/tangruijing/udtca/bitscom
CUDA_HOME=/usr/local/corex-4.4.0 \
BITSCOM_CUDA_COMPILER=/usr/local/corex-4.4.0/bin/clang++ \
NCCL_LIB_DIR=/usr/local/corex-4.4.0/lib64 \
NCCL_INCLUDE_DIR=/usr/local/corex-4.4.0/include \
/root/miniconda3/envs/trj-test/bin/python -m pip install -e . --no-build-isolation

# polar-sgd —— 分支必须是 refactor
cd /data1/tangruijing/udtca/polar-sgd
/root/miniconda3/envs/trj-test/bin/python -m pip install -e . --no-build-isolation
```

---

## 2. 启动步骤

### 正常跑一批实验

在 **u62** 上：

```bash
cd /data1/tangruijing/udtca
python udtca-config/generate_and_run.py
```

脚本会按 `test.json` 的顺序逐条跑，每条配置**按它自己的 `baseline` 标记**分派：

- `baseline` 为真 → 复用仓库里现成的 `0/1_train_qwen14b_baseline_ddp_1f1b_tp.sh`
- 否则 → 生成 POLAR + bitscom 的 `0/1_train_qwen14b_exp{index}.sh` 再跑

开始之前会先检查两个节点的 GPU 有没有被占用，**被占用就直接报错退出，不会启动任何训练**。

### 冒烟测试（只验证环境能不能跑通）

```bash
python udtca-config/generate_and_run.py --runtime-log off
```

`--runtime-log off` 时不占 case 编号、不建 `runtime_log/000N/`、不写 `case.json`、
不归档任何脚本和日志，训练产物落在训练进程自己的 cwd 下。跑训练本身照旧，所以验证
环境是准的。

也可以用环境变量代替命令行开关（优先级：命令行 > 环境变量 > 默认 on）：

```bash
UDTCA_RUNTIME_LOG=0 python udtca-config/generate_and_run.py
```

### 只跑 baseline

```bash
python udtca-config/generate_and_run_baseline.py            # 只跑 test.json 里 baseline=True 的
python udtca-config/generate_and_run_baseline.py --runtime-log off
```

两个入口共用同一套 case 编号和同一个 `runtime_log/case.json`。

### 实验矩阵（`test.json`）

共 8 个 case，**按网速从快到慢**排，同一档网速下 POLAR 和 baseline 紧挨着跑
（减少机器状态漂移对对比的干扰）：

| 运行顺序 | case 目录 | rate | 类型 |
|---|---|---|---|
| 1 | `runtime_log/0001/` | 10 gbit | POLAR + bitscom |
| 2 | `runtime_log/0002/` | 10 gbit | baseline |
| 3 | `runtime_log/0003/` | 5 gbit | POLAR + bitscom |
| 4 | `runtime_log/0004/` | 5 gbit | baseline |
| 5 | `runtime_log/0005/` | 2 gbit | POLAR + bitscom |
| 6 | `runtime_log/0006/` | 2 gbit | baseline |
| 7 | `runtime_log/0007/` | 1 gbit | POLAR + bitscom |
| 8 | `runtime_log/0008/` | 1 gbit | baseline |

`test.json` 里 `index` 1-8 就是上表顺序。没写的字段从 `default.json` 继承。

**当前训练超参**（`default.json`）：`max-steps=100`、`micro-batches=32`、
`per-device-batch-size=32`、`seq-len=256`、`pp=8`、`tp=2`、`lr=2e-4`；
`--log-interval 10`，所以**每 10 步打一行 step/loss**。

---

## 3. 产物：`runtime_log/`

```
<仓库根>/runtime_log/
├── case.json              每个 case 的启动脚本与启动参数
├── environment.json       机器与网络环境说明（交数据时用）
├── 0001/
│   ├── launch.sh          完整自包含的复现脚本
│   ├── 0_train_qwen14b_exp1.sh      node0(u62) 实际跑的
│   ├── 1_train_qwen14b_exp1.sh      node1(u210) 实际跑的（从远程拉回来）
│   ├── traffic_control_exp1.sh      网络限速脚本
│   ├── train_node0.log    node0 的 torchrun 全部输出
│   ├── train_node1.log    node1 的 torchrun 全部输出（从远程拉回来）
│   ├── tb_scalars/        tensorboard 事件文件（两节点合并）
│   ├── tb_trace/          profiler trace
│   └── step_csv/          每步耗时 CSV
├── 0002/ ...
```

`case` 编号按**实际运行顺序**递增（四位，跨多次运行连续），不是 `test.json` 的 index。

### 训练日志

两个节点的 `torchrun` 输出（stdout + stderr）**各自重定向**到 case 目录下的
`train_node0.log` / `train_node1.log`，不再刷屏。node1 的那份先写在 u210 上，
跑完由 `rsync` 拉回同一个 case 目录。

`--runtime-log off` 时不重定向，输出直接打在终端——冒烟测试本来就要看实时输出。

**u210 不往终端回显**（只写它自己的文件）：两个节点的 torchrun 输出交织在一起太乱。
所以终端上看到的是 runner 自己的编排信息和 **u62** 的训练输出；u210 的去
`train_node1.log` 里看。PP 最后一级在 u62 上有 rank 14/15，loss 在终端照样能看到。

### 中断（Ctrl-C）

收到 SIGINT/SIGTERM 时会：
1. 杀掉本机训练进程，并 **ssh 过去把 u210 上的训练进程也杀掉**
   —— 中断 runner 时 ssh 链路断了，但 u210 上的 torchrun 不会跟着死，会一直占着卡
2. 撤销两台机器上的 `tc` 限速 —— 不撤的话会给共享机器留一个限速规则

清理用的是「解释器路径 + 训练入口脚本名」精确匹配，**不会误杀共享机器上别人的任务**。

`launch.sh` 手动复现时行为一致，也写进同样的文件；想看实时输出就另开一个终端：

```bash
tail -f runtime_log/0001/train_node0.log
```

### `launch.sh` —— 完整可复现

每个 case 目录里的 `launch.sh` 是**自包含**的，不依赖本目录的 runner：

```bash
# node 0 (u62)
bash runtime_log/0001/launch.sh 0
# node 1 (u210)
bash runtime_log/0001/launch.sh 1
```

里面包含环境准备（shell 初始化 → Corex/CUDA 变量 → conda 环境 → HF 镜像）、
限速命令说明、以及两个节点各自的完整 `torchrun` 命令。

### `case.json` 结构

```json
{
  "cases": [
    {
      "case": "0001",
      "test_index": 1,
      "kind": "polar",                    // polar | baseline
      "started_at": "2026-09-22_08-15-03",
      "finished_at": "2026-09-22_08-21-47",
      "exit_code": 0,
      "log_dir": "/data1/tangruijing/udtca/runtime_log/0001",
      "run_label": "polar_bitscom_1f1b_tp",
      "rate": "1gbit",
      "config": { "...合并后的完整配置..." },
      "scripts": { "node0": "...", "node1": "...", "traffic_control": "..." },
      "launch_params": { "node0": { "--key": "value", "...": "..." }, "node1": { "..." } }
    }
  ]
}
```

`launch_params` 里 node0 / node1 **分开记录**，因为 `--polar-max-inflight-buckets` 两边不同
（node0 = 4，node1 = 1）。polar 的参数由 `build_launch_params()` 渲染，**和实际下发的脚本同源**，
不会对不上；baseline 的脚本是仓库里已有的文件，改成从脚本里解析，node1 那份还会**从 u210 上读**。

---

## 4. 网络限速

`traffic_control_exp{index}.sh` 由 runner 生成并下发到两个节点，两边都要执行（需要 root）：

```bash
bash traffic_control_exp1.sh start   # 按 rate 限速
bash traffic_control_exp1.sh stop    # 恢复
bash traffic_control_exp1.sh status  # 查看
```

只用到 `RATE`（取值来自 `test.json` 的 `rate`），脚本里的 `BURST` / `LATENCY` / `DELAY`
是保留的可调项，当前 `DELAY="0ms"` 表示不模拟时延。

---

## 5. 文件说明

| 文件 | 作用 |
|---|---|
| `generate_and_run.py` | 主编排入口，按 config 分派 polar / baseline |
| `generate_and_run_baseline.py` | 只跑 baseline 的入口，同时对外提供 `run_baseline_case()` 供上面那个 import |
| `case_common.py` | 两节点命令封装、`SETUP_CMDS`、case 编号与 `case.json` 读写、GPU 检查、日志收集 |
| `collect_case_logs.py` | 把训练侧写的深层 `./log/...` 目录拍平成 `tb_scalars` / `tb_trace` / `step_csv` |
| `default.json` / `test.json` | 默认配置 / 实验矩阵 |
| `run_qwen14b_polar_dp_pp_tp.py` | 训练入口（仓库里 `experiments/qwen14b/` 下那份的副本） |
| `trace_processor.py` | 从 tfevent 里算平均迭代时间和吞吐 |

---

## 6. 注意事项

**日志目录为什么叫 `runtime_log` 而不是 `log`**
仓库根下的 `log/` 是早期实验留下的旧结构（`log/True/...`、`log/False/...`），有几十 GB，
本目录的产物一律走 `runtime_log/`，互不干扰。

**脚本只 copy 不 move**
`experiments/qwen14b/` 里那些脚本不会因为跑实验而消失：仓库里已有的 baseline 脚本
受版本管理必须原地保留；runner 自己生成的 polar 脚本在归集后会从原地移走，下次同 index
运行会重新生成。

**两节点的 shell 环境是对齐过的**
本地跑时进程继承的是交互 shell 已 source 过 `.bashrc` 的环境，而 `ssh u210` 是非交互的、
默认不 source 任何东西，且两台的 `.bashrc` 内容并不一样（u210 缺 `COREX_PATH` /
`CPATH` / `LIBRARY_PATH` / `TRITON_CUDA_SYSROOT` / `CUDA_DEVICE_MAX_CONNECTIONS`）。
所以 `SETUP_CMDS` 会先手动 source `~/.bashrc`（绕过它开头的非交互保护），
再把这些变量显式补齐，保证两边一致。

**远程命令用 base64 传**
`ssh host '/bin/bash -c "..."'` 这种两层引号会让远程登录 shell 抢先把 `$VAR` 展开掉，
而这些变量正是脚本里要 export 的。所以 `ssh_execute` 把整条脚本 base64 后送过去
（`ssh host 'echo <b64> | base64 -d | bash'`），避免这一整类引号问题。

**GPU 占用检查**
`check_gpus_free()` 在两个入口的 `main()` 最开头跑，判定条件是
`--query-compute-apps` 有进程 **或** `--query-gpu=memory.used` 超过 256 MiB（空载基线 68 MiB）。
SMI 工具按候选路径找：u62 的 `ixsmi` 在 PATH 里，**u210 的 ixsmi 不在 PATH 里**，
要靠 `/usr/local/corex-4.4.0/bin/ixsmi`；两边都没有 `nvidia-smi`。找不到工具时只告警不阻塞。
