# 双机 32 卡训练用法（cloud / edge）

两台机器各 16 卡，合计 32 卡：**PP=8、TP=2、全局 DP=2**。
每台机器上就是一个完整的 PP×TP 副本，跨机的只有 DP 的 all-reduce。

| 目录 | 机器 | torchrun | 分支 |
|------|------|----------|------|
| `cloud/` | powerone1（云侧，10.205.200.14） | `node_rank=0`，master | `cloud/run_xnode.sh` |
| `edge/` | powerone2（边侧，10.205.200.16） | `node_rank=1` | `edge/run_xnode.sh` |

两个脚本都是**自包含**的（环境变量、模式参数、torchrun 全在文件里），
要在各自机器的 `trj-udtca` 容器里执行，不调用别的脚本。

## 起法（顺序不能反）

```bash
# 1) 边侧先起，它会等 master（边侧机器容器内）
cd /data/tangruijing/udtca/udtca-config/qwen14b
MODE=mixed bash edge/run_xnode.sh

# 2) 云侧再起（云侧机器容器内）
cd /data/tangruijing/udtca/udtca-config/qwen14b
MODE=mixed bash cloud/run_xnode.sh
```

两边 `MODE` 必须一致，其余可覆盖项（`MAX_STEPS`、`MICRO_BATCHES`、`SEQ_LEN`、
`GPUS`、`PP_SIZE`、`TP_SIZE`、`SEED`、`MASTER_PORT`）也必须一致，否则各 rank
起的拓扑不同，rendezvous 后直接挂。

建议在 tmux / screen 里跑，云侧那条命令会一直占着前台输出。

## MODE

| MODE | 说明 |
|------|------|
| `baseline` | 纯 DDP（DP 走 dense all-reduce） |
| `polar_dense` | Polar EF 低内存桶，dense all-reduce |
| `bitscom` | 低比特压缩，不用 polar |
| `mixed` | Polar + bitscom（默认关注的对象） |

## 常用覆盖

```bash
MODE=mixed MAX_STEPS=1 bash edge/run_xnode.sh          # 冒烟
MODE=mixed DISABLE_PROFILER=true bash edge/run_xnode.sh # 不抓 trace
MODE=baseline MASTER_PORT=29631 ...                     # 端口冲突时换
```

- 默认 `MAX_STEPS=3`、`MICRO_BATCHES=16`、`PER_DEVICE_BATCH_SIZE=16`、`SEQ_LEN=256`、`SEED=1234`。
- `MICRO_BATCHES` 必须 ≥ `PP_SIZE`（默认 16 ≥ 8，OK）；`MAX_STEPS=3` 只够冒烟，
  看性能用 10 步以上。
- profiler 默认开（`DISABLE_PROFILER=false`），抓 `MAX_STEPS` 步；32 卡每步
  trace 有几百 MB，冒烟时就关掉。

## 产物

每台机器各自写到自己的仓库下：

```
run_log/xnode_<MODE>_node{0,1}_<时间戳>/
├── train.log            训练输出（同时 tee 到 log/xnode_...log）
├── EXIT_CODE            最后一行 EXIT_CODE=<rc>
├── PRE_RUN.md           本次参数快照
├── logs/nccl.<host>.<pid>.log
├── step_csv/            逐步 step_time_s（最后一级 stage 才有）
└── ckpt/
```

profiler trace 落在 `<repo>/log/<run-label>/.../tb_trace/*.pt.trace.json`。

## 注意

- 两台机器的 `ifname` 不同（云侧 `ens120f0`、边侧 `enx00e04c680003`），脚本里已各自写死，
  不要用同一个值去覆盖两台。
- 中途 Ctrl-C 会打断在途 NCCL，Corex 上有 XID 24（MMU 挂死）风险；要停就两边一起停，
  停完用 `ixsmi` 确认卡状态。
- 宿主机侧还有 `qwen14b/run_docker.py`（在 powerone1 宿主机跑，自动起两台容器、
  跑 mixed/baseline 并做 trace 分析），见该文件头部注释。
