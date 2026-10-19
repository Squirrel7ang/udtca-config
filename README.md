# udtca-config

两节点分布式训练实验的**编排器**（不是训练代码）。按实验矩阵生成启动脚本 → 下发到
u13/u177 → `tc` 限速 → 起训 → 回收日志到 `<仓库根>/runtime_log/<case>/`。

只需在 **u13** 上跑一条命令，它会同时操控 u13（node 0）和 u177（node 1）。

## 目录结构

```
udtca-config/
├── common/
│   ├── case_common.py       公共库：常量、两节点命令封装、case 编号、GPU 检查、日志收集
│   └── orchestrator.py      通用编排流程：生成脚本/下发/限速/起训/收日志
├── qwen14b/                 Qwen2.5-14B 实验（PP=8 TP=2 DP=2）
│   ├── default.json         默认超参，被 test.json 的条目覆盖
│   ├── test.json            实验矩阵（index / rate / bit-width / baseline）
│   ├── generate_and_run.py  主入口（唯一需要直接跑的）
│   ├── run.py               4-case 一键跑 + 汇总（读 run_cases.json，产物在 run_log/）
│   ├── run_cases.json       run.py 的 case 列表（2gbit/1gbit × polar/baseline）
│   ├── start_metrics_api.py 实时指标 API 入口（可同时拉起 run.py），见 API_USAGE.md
│   ├── metrics_api.py       API 实现：扫日志、解析 [rt-metric]、HTTP/SSE
│   └── _run_baseline.py     内部脚本：只跑 baseline 的入口（一般不用直接调用）
├── qwenvl8b/                Qwen3-VL-8B 实验（PP=8 TP=1 DP=4），结构与 qwen14b 一一对应
│   ├── default.json  test.json  generate_and_run.py  _run_baseline.py
└── utils/                   与具体实验无关的辅助脚本
    ├── watch_bandwidth.py   实时看两机之间的网卡带宽
    ├── collect_case_logs.py 把深层 ./log/... 拍平成 tb_scalars/tb_trace/step_csv
    ├── trace_processor.py   从 step_csv/tfevent 算每步耗时与吞吐，做 case 间对比
    ├── download_fineweb.py  按顺序把 fineweb 的前缀 parquet 下到本地
    ├── analyze_overlap.py   从 profiler trace 算计算-通信重叠率（三种口径）
    ├── overlap_interval.py  从 profiler trace 算 DP 区间与 PP 区间的交并比重叠率
    ├── parse_nccl_bytes.py  从 NCCL 集合通信日志算通信量，给出压缩比
    └── build_comm_opt_data.py  把实验结果整理成交付目录树并打包

> `run_qwen14b_polar_dp_pp_tp.py` 是 `experiments/qwen14b/` 下同名文件的**旧副本**
> （只差一行 print），没有任何代码引用它。留着没删，确认无用后可以删掉。
```

## 前置条件

| 项 | 值 |
|---|---|
| 仓库路径 | `/data1/tangruijing/udtca`（**两节点必须相同**） |
| 调度节点 | u13 = `10.31.10.13`，**必须是 node 0**（torchrun 要 bind `MASTER_ADDR`） |
| 训练节点 | u177 = `10.31.10.177`，通过 `ssh 10.31.10.177` 访问 |
| conda 环境 | `udtca`（两节点都要有，且 bitscom / polar-sgd 已 `pip install -e .`） |
| 互联 | `ens1f0`，理论带宽 12.5 Gb/s |
| HF 下载 | 走 `HF_ENDPOINT=https://hf-mirror.com`，不需要代理 |

上面这些都不是写死的，换机器时导出环境变量即可（代码里默认就是当前这两台）：

| 环境变量 | 含义 | 默认 |
|---|---|---|
| `UDTCA_NODE0_HOST` / `UDTCA_NODE0_IP` | node 0（本机，MASTER_ADDR 指向它） | `13` / `10.31.10.13` |
| `UDTCA_NODE1_HOST` / `UDTCA_NODE1_IP` | node 1（ssh 目标，IP 或 ssh 别名） | `10.31.10.177` |
| `UDTCA_CONDA_ENV` | 两侧同名的 conda 环境 | `udtca` |
| `UDTCA_NIC_IFACE` | 两机互联网卡 | `ens1f0` |

实时指标 API（给第三方测试平台）：`qwen14b/start_metrics_api.py`，跑在 node 0 上，
提供 `/api/v1/metrics/latest`、`/stream`（SSE）、`/overview` 等接口。
启动方式、接口字段、前端调用示例见 [`qwen14b/API_USAGE.md`](qwen14b/API_USAGE.md)。

```bash
ssh 10.31.10.177 hostname                    # 免密 ssh 通
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

另一条入口是 **`run.py`**：固定按 `qwen14b/run_cases.json` 跑 4 个 case
（2gbit / 1gbit × polar / baseline），产物在 `run_log/run_<时间戳>/`，跑完在终端
打印对比表并写 `summary.md` / `summary.json`。整个标准输出（含最后的 4-case 对比）
同时实时写到 `run_log/run_<时间戳>/run.log`，训练中途卡住也可以直接 tail 它。
`--max-steps 3` 冒烟，`--no-tc` 不下限速（只验证能否跑通，不能用来比带宽）：

```bash
python udtca-config/qwen14b/run.py --max-steps 3 --no-tc
```

只跑 baseline 是 `generate_and_run.py` 的一个子集，一般直接用它加 `--only` 即可，
不需要单独跑别的脚本。确有需要时再用内部脚本 `_run_baseline.py`：

```bash
python udtca-config/qwen14b/_run_baseline.py    # 内部脚本，只跑 test.json 里 baseline 的条目
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
| `qwen14b/generate_and_run.py` | **唯一对外入口**，按 `test.json` 分派 polar / baseline | `python .../qwen14b/generate_and_run.py [--only 1,3] [--runtime-log off]` |
| `qwen14b/run.py` | 固定跑 `run_cases.json` 里的 4 个 case，跑完打印对比表 + `summary.md/json` | `python .../qwen14b/run.py [--max-steps 3] [--no-tc]` |
| `qwen14b/start_metrics_api.py` | 起实时指标 API，`--with-training` 可同时拉起 `run.py` | `python .../qwen14b/start_metrics_api.py --port 8000 [--with-training]` |
| `qwen14b/_run_baseline.py` | 内部脚本：只跑 `baseline=True` 的条目，共用同一套 case 编号 | 一般不用直接调用，`generate_and_run.py --only` 即可 |
| `qwenvl8b/generate_and_run.py` | QwenVL8B 主入口，行为同上 | 同上 |
| `qwenvl8b/_run_baseline.py` | QwenVL8B 内部脚本：只跑 baseline | 同上 |
| `common/case_common.py` | 被上面 4 个 import。定义 `ExperimentSpec`、`SETUP_CMDS`（两节点环境对齐）、`ssh_execute`（base64 传远程脚本）、`check_gpus_free`、`allocate_case`、`write_launch_script` 等 | 不单独运行 |
| `common/orchestrator.py` | 被上面 4 个 import。`run_polar_case` / `run_baseline_case` / `run_experiments`，以及训练脚本模板 `write_train_script` | 不单独运行 |
| `utils/watch_bandwidth.py` | 读两节点 `ens1f0` 的 rx/tx 计数算实时速率（只读，不干扰训练） | `python utils/watch_bandwidth.py [-i 5] [--once] [--iface ens1f0]` |
| `utils/check_k8s_port_conflict.py` | 检查 kube-proxy 的 REJECT 端口是否落在内核临时端口范围里（会导致 NCCL 初始化卡死），可 `--fix` 自动排除；run.py / generate_and_run.py 已自动调用 | `python utils/check_k8s_port_conflict.py [--fix] [--json] [--probe]` |
| `utils/collect_case_logs.py` | 把训练侧写的深层 `./log/{...}/{rank}/tb_scalars` 拍平到 case 根下 | 由 `case_common.finish_case` 自动调用 |
| `utils/trace_processor.py` | 从 `step_csv` 或 tfevent 算平均步时/吞吐，做 case 间对比 | `python utils/trace_processor.py [--case 0001 0002] [--csv out.csv]` |
| `utils/download_fineweb.py` | 按仓库顺序下 fineweb 前缀 parquet，供离线训练 | `python utils/download_fineweb.py [--target-gb 20] [--dry-run]` |
| `utils/build_comm_opt_data.py` | 把实验结果整理成交付目录树并打 zip（只复制不移动） | `python utils/build_comm_opt_data.py [--out /path]` |

## 产物：`runtime_log/`

```
runtime_log/
├── case.json            所有 case 的索引（见下）
├── environment.json     机器与网络环境说明（交数据时用）
└── 0021/                一个 case 一个目录，编号按实际运行顺序递增
    ├── launch.sh                       自包含的复现脚本（不依赖 runner）
    ├── 0_train_<exp>_exp1.sh           node0(u13) 实际跑的
    ├── 1_train_<exp>_exp1.sh           node1(u177) 实际跑的（跑完从远程拉回）
    ├── traffic_control_exp1.sh         限速脚本（start / stop）
    ├── train_node0.log                 node0 的完整输出
    ├── train_node1.log                 node1 的完整输出
    ├── nccl_<host>_<pid>.log           每个 rank 一份的 NCCL 集合通信日志
    ├── tb_scalars/                     tensorboard 标量（loss 等）
    ├── tb_trace/                       每个 rank 一份 profiler trace
    └── step_csv/                       每步耗时/吞吐的 csv
```

各项说明：

| 项 | 说明 |
|---|---|
| `case.json` | 每个 case 一条：编号、`test_index`、`kind`（polar / baseline）、限速、实际下发的启动参数、`exit_code`、以及**训练窗口内两机之间的通信量** `comm_bytes` / `comm_gb`（1GB=1e9B）。 |
| case 编号 | 按**实际运行顺序**递增，不是 `test.json` 的 index；同一个 index 重跑会拿到新编号。 |
| `launch.sh` | 把该 case 的下发、限速、起训打包好，两个终端各跑 `bash launch.sh 0` / `bash launch.sh 1` 就能复现，不依赖 runner。 |
| `*_train_<exp>_exp1.sh` | 实际下发的训练脚本；`0_` 是 u13 的，`1_` 是 u177 的。 |
| `train_node*.log` | u177 的输出只写 `train_node1.log`、不回显终端（两节点输出交织太乱）；PP 最后一级在 u13 上有副本，loss 在终端照样看得到。 |
| `nccl_*.log` | 训练时开 `NCCL_DEBUG` 落下来的集合通信日志，每个 rank 一个文件。通信量的另一套口径从这里读（`count × dtype`），对应 `utils/parse_nccl_bytes.py`。 |
| `tb_trace/` | profiler trace，每个 rank 一份 json。重叠率就是从这里算：取 rank0 那份，对应 `utils/analyze_overlap.py`、`utils/overlap_interval.py`。 |

被中断的 case 目录会留在原地（通常没有 `tb_trace/`），方便事后查为什么没跑完。

手动复现某个 case（两个终端各跑一条）：

```bash
bash runtime_log/0001/launch.sh 0     # u13
bash runtime_log/0001/launch.sh 1     # u177
```

## 注意事项

- **u13 必须是 MASTER**。`node_rank=0` 的 torchrun 会 bind `MASTER_ADDR`，绑别人的地址
  会 `EADDRNOTAVAIL` 然后两边干等。`SETUP_CMDS` 和仓库里的启动脚本默认值都指向
  `10.31.10.13`，双保险。
- **`tc` 只限出向**，两个节点都要执行，跑完必须撤（runner 会自动撤；Ctrl-C 中断时也会
  杀远程训练进程并撤限速）。留着的限速规则会影响这台共享机器上的其他人。
- **日志目录叫 `runtime_log` 不叫 `log`**：仓库根下的 `log/` 是早期实验留下的几十 GB 旧结构。
- **只 copy 不 move**：仓库里已有的 baseline 脚本受版本管理，原地保留；runner 生成的
  polar 脚本归集后会从原件移走，下次同 index 重新生成。
- **GPU 占用检查**在入口最开头跑：`--query-compute-apps` 有进程，或显存 > 256 MiB
  （空载基线 68 MiB）就判定被占用并退出。u177 的 `ixsmi` 不在 PATH 里，走绝对路径
  `/usr/local/corex-4.4.0/bin/ixsmi`。

## 排障（Troubleshooting）

### 训练迟迟没有日志、卡显存不涨，像是"压根没启动"

先看 node0 日志里有没有反复刷这一行：

```
[W ... socket.cpp:200] [c10d] The hostname of the client socket cannot be retrieved. err=-3
```

**出错的阶段**：不是训练，而是最早的 rendezvous：node0 的 `TCPStore` 在
`MASTER_PORT`（29500）上 accept node1 的客户端连接时，会对客户端地址做一次反向 DNS
（`getnameinfo`）。`err=-3` 是 `EAI_AGAIN`，即解析超时/临时失败。

**为什么会导致"没启动"**：16 个 rank 的连接是串行 accept 的，每个反向解析都卡在
DNS 超时上（这台机器上实测 **单次 10 秒**），于是一两分钟内没有任何训练日志、
显存也不上涨，看起来就像进程没起来。之前正是这个原因。

**确认方法**（在 node0 / u13 上）：

```bash
time getent hosts 10.31.10.177     # 正常应 <10ms；要 10s 且无输出就是命中了
```

`/etc/resolv.conf` 指向 `127.0.0.53`（systemd-resolved），上游 DNS 不可达时会
先等超时再返回 `EAI_AGAIN`。

**修复**：给两台机器的 `/etc/hosts` 加上对方（以及自己）的映射，让反向解析走
本地文件、不碰 DNS：

```bash
# 两台机器都执行（已做过一次，换成别的机器时照做）
printf '10.31.10.13 u13\n10.31.10.177 u177\n' | sudo tee -a /etc/hosts
time getent hosts 10.31.10.177     # 验证：应立刻返回
```

修复后从启动到进入训练循环约 30–40 秒；修复前同样配置 1 分钟仍毫无输出。

### 初始化阶段随机失败：NCCL `Connection refused` / `free(): double free detected`

共享机器上 4 个 case 连跑时，偶发单个 case 在初始化阶段卡住或崩掉，三种表现：

1. 整个 case **无限期卡住**（日志停在模型加载之后，几十个 rank 停在 `dist.barrier()`，
   个别 rank 停在 `create_lowbit_dp_group` → `ncclCommInitRank`）。根因见下一节。
2. `ncclRemoteError ... socketPollConnect: connect returned Connection refused,
   exceeded error retry count (35)`，随后另一个 rank 在 TCPStore 上 `wait timeout`。
   这是 NCCL bootstrap 默认只重试 34 次 × 100 ms（约 3 秒）导致的：16 个进程同时
   从本地盘加载分片权重，两节点进入 TP 切分的时间可能相差几十秒，先到的 rank
   连不上后到的 peer 的 listener 就放弃。`SETUP_CMDS` 里已把重试放宽到最长约
   10 分钟作为兜底（实测 60×100 ms 只有约 6 秒、120×1000 ms 仍出现过 4 分钟的
   两节点启动时间差；正常跑通时不会触发这个等待）：
3. `free(): double free detected in tcache 2` + `Signal 6 (SIGABRT)`。这是
   Iluvatar 运行时/第三方库里的堆损坏，代码层面没有稳定触发条件，只能重跑该 case；
   快速反复 kill/重启会提高触发概率——尽量让 4 个 case 一次跑完，中断后等几秒、
   确认 `ixsmi` 显示 16×68 MiB 再重跑。

```bash
export NCCL_SOCKET_RETRY_CNT=600
export NCCL_SOCKET_RETRY_SLEEP_MSEC=1000
```

### case 卡死在初始化、既不报错也不进训练循环（kube-proxy 把临时端口 REJECT 掉）

**现象**：某个 case 的日志停在模型加载后不再前进，卡几十分钟也不报错。用 py-spy
看两个 rank（两台机器上各一个）会看到它们在同一个 `ncclCommInitRank` 里：

```bash
/tmp/pyspy_pkg/bin/py-spy dump --pid <pid> --native
# 一侧：bootstrapRecv → ncclSocketAccept → accept()      （在等对方连进来）
# 另一侧：bootstrapSend → ncclSocketConnect → socketPollConnect → sleep（连不上）
```

**根因**：这两台机器同时是 k8s 节点（两边都有 kube-proxy：u13 PID 3271724、
u177 PID 4420，`/etc/kubernetes`、`/var/lib/kubelet` 都在）。**u177** 上
kube-proxy 在 `KUBE-EXTERNAL-SERVICES` 里给"没有后端的 ssh 服务"挂了
`-j REJECT --reject-with icmp-port-unreachable`，匹配条件是
`ADDRTYPE match dst-type LOCAL`，也就是说**不管目标是哪个本机地址都会被拒**。
被拒的端口里有一段正好落在 **Linux 临时端口范围** `32768–60999` 内：

| 机器 | 被 REJECT 且在临时端口范围内的端口 | 是否真会挡我们 |
|---|---|---|
| node 0（u13） | `38000`、`50007` | 否：这两条在 `KUBE-SERVICES` 里限定了 `-d 10.43.x.x`（虚 IP），连 `10.31.10.13:<port>` 不命中 |
| node 1（u177） | `40001–40016` | **是**：`dst-type LOCAL`，本机任意地址都挡 |

NCCL 的 bootstrap 监听 socket 是 `bind(port=0)` 由内核分配端口的，一旦分到
40001–40016，对端连过来时包还没到监听 socket 就被 iptables REJECT 掉，内核回
RST → NCCL 看到 `connect returned Connection refused`。`socketPollConnect` 会按
`NCCL_SOCKET_RETRY_CNT` 重试，**每次睡眠时间线性增长**（第 n 次睡 n 秒，日志里
能看到 `retrying (61/600) after sleep for 61000 msec`），累计到 600 次要几十小时，
所以表现出来就是"永远卡住"。其余 rank 都在 `dist.barrier()` 等它，case 永不结束。
每个 case 要开上百个监听端口，16/28232 的命中率 → 每个 case 约 5%–10%，
和"4 个 case 里偶发一个初始化卡死"对得上。

**确认方法**（卡住时执行）：

```bash
# 1. 找出被 REJECT 且落在临时端口范围内的端口（只看 dst-type LOCAL 那种规则）
iptables-save | grep 'dst-type LOCAL' | grep REJECT | grep -oE 'dport [0-9]+' \
  | awk '{print $2}' | sort -n
sysctl net.ipv4.ip_local_port_range          # 32768 60999
grep -rh 'service-node-port-range' /etc/kubernetes/manifests/ 2>/dev/null   # NodePort 池

# 2. 卡住时看我们自己的进程有没有监听在这些端口上
ssh 10.31.10.177 "ss -ltnp | grep ':4000[0-9]\\|:4001[0-6]'"

# 3. 复现验证（几秒钟）：在 u177 上 bind 某个被拒端口，再从 u13 连它
#    有监听 socket 也照样 Connection refused；同时 iptables 该条规则的计数 +1
```

**修复**：把这几段端口从"内核自动分配"的池子里摘出去（不影响别人显式 bind，
不动任何 kube 规则，随时可回退）：

```bash
# 只需在 u177 上做（u13 的那两条规则限定虚 IP，挡不到我们）；需要 root
sudo sysctl -w net.ipv4.ip_local_reserved_ports=40000-45000
```

> 预留的是**整个 NodePort 池**（40000–45000），不只是当前已分配的 40001–40016：
> 每新建一个 `dev-ssh-dev-userNN` 服务，kube-proxy 就会在池子里多分配一个端口并
> 下发 REJECT 规则，只预留"当前 16 个"的话下次还会中招。临时端口还剩
> 32768–39999（7232 个），对训练足够。
>
> 已在 u177 上应用（2026-10-18 首次应用 40001–40016，2026-10-19 扩到整个池）。
> 这是运行时配置，重启后失效；每次跑训练前脚本会自动重新检查/写入，也可以写进
> `/etc/sysctl.d/99-udtca-reserved-ports.conf` 持久化。验证方式：连续 `bind(0)`
> 4000 次，改前会命中 40001–40016（实测命中过 40003、40009），改后命中 0 次，
> 且其它临时端口（如 38000–38010）不受影响。

**为什么 u62 / u210 上从来没这个问题**（2026-10-19 实测对比）：

| 机器 | kube-proxy | NodePort 池 | 落在临时端口范围（32768–60999）里的 REJECT 端口 |
|---|---|---|---|
| u13 | RKE 集群（`--cluster-cidr=10.42.0.0/16`） | 30000–32765 | **0**（737 条规则，但都在 32768 以下） |
| u62 | 同 u13 的集群 | 30000–32765 | **0**（737 条，同上） |
| u177 | 另一个集群（`/usr/local/bin/kube-proxy --config=/var/lib/kube-proxy/config.conf`） | **40000–45000** | **16**：40001–40016（`dev-ssh-dev-user01..16`，服务没有 endpoints） |
| u210 | 没有 kube-proxy | — | **0** |

关键差别是 **NodePort 池和临时端口范围有没有交集**：u13/u62 的集群用默认
30000–32767（在临时端口范围之下），u210 根本不是 k8s 节点；只有 u177 加入了
一个把 NodePort 池配到 `40000-45000` 的集群（`/etc/kubernetes/manifests/kube-apiserver.yaml`
里的 `--service-node-port-range=40000-45000`），这段正好落在 32768–60999 中间。
所以同一个训练脚本在 u62/u210 上怎么跑都不会撞，在 u177 上则会偶发卡死。

不打算改系统配置时，直接重跑这个 case 也可以——换个 run 端口就躲开了，只是仍会偶发。

**自动检查脚本**：`utils/check_k8s_port_conflict.py` 把上面的判断做成了两步：
读 `iptables-save` 找「REJECT + `dst-type LOCAL`」的端口，同时读出 k8s 声明的
NodePort 池；两者与 `ip_local_port_range` / `ip_local_reserved_ports` 对比，
列出还没排除的风险端口。

```bash
python udtca-config/utils/check_k8s_port_conflict.py          # 只检查
python udtca-config/utils/check_k8s_port_conflict.py --fix    # 检查 + 写 sysctl（root）
python udtca-config/utils/check_k8s_port_conflict.py --json   # 机器可读
python udtca-config/utils/check_k8s_port_conflict.py --probe  # 采样 4000 次 bind(0) 验证
```

输出示例（u177）：

```
[port-check] u177: 临时端口范围 32768-60999
[port-check] u177: 被 kube-proxy REJECT 的端口 16 个，落在临时端口范围里的 16 个: 40001-40016
[port-check] u177: OK（冲突端口都已从自动分配里排除）
[port-check] u177: bind(0) × 4000，落到被 REJECT 端口的次数 = 0
```

退出码：`0` 没冲突（或已修复）、`1` 仍有冲突、`2` 读不到 iptables（非 root 等，跳过）。

**已经接进运行流程**：`run.py` 和 `generate_and_run.py`（`orchestrator.run_experiments`）
在开跑前会调用 `case_common.check_port_conflicts(fix=True)`，对**本机和 u177**
各跑一次上面这个脚本。正常情况下每台机器会打印一行
`[port-check] ... OK`；如果修复后仍有冲突，直接退出，不会让训练卡在初始化。
所以不需要每次手动 `sysctl`。

第二种是 Iluvatar 运行时/第三方库里的堆损坏，代码层面没有稳定触发条件，
只能重跑该 case；快速反复 kill/重启会提高触发概率——尽量让 4 个 case 一次跑完，
中断后等几秒、确认 `ixsmi` 显示 16×68 MiB 再重跑。

另外注意：runner 以前把失败 case 的 `exit_code` 记成 0（`bash -o pipefail` 只包了
内层命令，外层管道取的是 `tee` 的返回码）。已修正为整条管道都带 pipefail，
现在失败会如实反映在 `case.json` 的 `exit_code` 和终端输出里。

### 日志要等很久才出现

训练输出不是 tty，Python 默认块缓冲（约 8KB 才刷一次）。`SETUP_CMDS` 里已统一
`export PYTHONUNBUFFERED=1`，`launch.sh` 和 runner 两条路径都生效。自己手写启动
脚本时记得也加上，否则看不到实时的初始化进度。

### 重新起跑报 `Address already in use`（MASTER_PORT 被占）

上一次中断如果没清干净，残留的 torchrun 会继续占着 29500/29501：

```bash
ss -ltnp | grep -E '29500|29501'   # 看是哪个 pid
```

确认是自己这次实验的进程后精确 kill 那个 pid（**不要** `pkill -f torchrun`，
共享机器上会误杀别人）。runner 的 Ctrl-C 清理逻辑（`kill_local_training` /
`kill_remote_training`）会按"解释器路径 + 脚本名"精确匹配，正常中断不需要手清。
