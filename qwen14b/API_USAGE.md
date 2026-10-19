# 实时指标 API 使用说明

给前端/测试平台用：一条命令拉起 4-case 训练和实时指标服务，前端通过 HTTP / SSE
读取 **DP-PP 通信重叠率** 和 **通信压缩量**。

涉及两件事，互相独立，可以一起起也可以分开起：

| 组件 | 入口 | 作用 |
|---|---|---|
| 训练 | `udtca-config/qwen14b/run.py` | 一键跑 4 个 case（2gbit / 1gbit × baseline / polar），跑完打印对比 |
| API | `udtca-config/qwen14b/start_metrics_api.py` | **只读**日志，提供重叠率/压缩量的 HTTP + SSE 接口 |

API 是纯只读的：只 tail `<run_dir>/<case>/train_node0.log` 和 `step_csv/*.csv`，
不碰训练进程、不碰 GPU、不占卡。

---

## 1. 先跑训练（4 个 case）

所有命令都在 **node 0（u13）** 上执行，编排器自己通过 ssh 拉起 u177。

```bash
cd /data1/tangruijing/udtca

# 完整跑：每个 case 30 步（default.json / run_cases.json）
python udtca-config/qwen14b/run.py

# 冒烟：每个 case 3 步
python udtca-config/qwen14b/run.py --max-steps 3

# 冒烟且不限速（共享机器被抢、只想验证能否跑通时最快）
python udtca-config/qwen14b/run.py --max-steps 3 --no-tc
```

> `--no-tc` 会让不同 `rate` 的 case 不再有带宽差异，只能用来验证流程，不能用来做
> 2gbit vs 1gbit 的对比。

跑哪几个 case 由 `udtca-config/qwen14b/run_cases.json` 决定，默认 4 个：

```json
[
  { "index": 1, "rate": "2gbit", "bit-width": 2, "baseline": "False" },   ← POLAR + bitscom
  { "index": 2, "rate": "2gbit", "baseline": "True" },                    ← 普通 DP baseline
  { "index": 3, "rate": "1gbit", "bit-width": 2, "baseline": "False" },
  { "index": 4, "rate": "1gbit", "baseline": "True" }
]
```

脚本自己会做：GPU 占用检查（被占直接退出，不会抢卡）→ 生成两节点脚本 → 下发 →
`tc` 限速 → 起训 → 收日志 → 撤限速 → 打印汇总。

产物在 `run_log/run_<时间戳>/`，**每次新建、从不覆盖**：

```
run_log/run_2026-10-17_03-14-12/
├── run.log                 run.py 的完整标准输出/错误（含最后的 4-case 对比），跑的过程中实时写
├── case.json               case 索引（编号 / kind / rate / 日志目录 / 通信量）
├── environment.json        机器与网络环境
├── summary.json/.md        4-case 汇总（跑完才有）
└── 0001/                   一个 case 一个目录，编号按实际运行顺序
    ├── train_node0.log     node0 的完整输出（含每步 [rt-metric] 行）— API 的主数据源
    ├── train_node1.log     node1 的输出（跑完从 u177 拉回）
    ├── step_csv/           每步耗时
    └── tb_trace/           profiler trace
```

> 用了 `--run-log-root` 的话传**绝对路径**（脚本会 `resolve()`，但别依赖这个）。

---

## 2. 再起 API

两种方式，任选其一：

### 2.1 一条命令：训练 + API 一起起（推荐给测试平台）

```bash
cd /data1/tangruijing/udtca
python udtca-config/qwen14b/start_metrics_api.py --with-training --max-steps 30
```

- 训练由 `run.py` 拉起，`start_metrics_api.py` 退出时（Ctrl-C）会一并收掉训练进程；
- API 自动盯**本次**新建的 run 目录，不用手填 `--run-dir`。

### 2.2 分开起：训练已经在跑，只起 API

```bash
cd /data1/tangruijing/udtca

# 默认自动挑 run_log 下最新的那个 run
python udtca-config/qwen14b/start_metrics_api.py --port 8000

# 或明确指定某次 run
python udtca-config/qwen14b/start_metrics_api.py \
    --run-dir /data1/tangruijing/udtca/run_log/run_2026-10-17_03-14-12 --port 8000

# 手工起的训练（没有 run 目录）：直接盯一个日志文件
python udtca-config/qwen14b/start_metrics_api.py --log-file /tmp/train_node0.log
```

起起来后终端会打印一行 `run_dir`，确认指向的是本次 run（不是上一次）。

### 2.3 跨机访问

API 只需要在 **node 0** 上起，也只读 node 0 本地的 `run_log/`（node 1 的日志是每个
case 跑完后才拉回来的），所以不用配 node 1 的地址：

```bash
# 监听所有网卡（默认），其他机器用 node 0 的 IP 访问
python udtca-config/qwen14b/start_metrics_api.py --host 0.0.0.0 --port 8000
#   http://10.31.10.13:8000/docs

# 只允许本机访问
python udtca-config/qwen14b/start_metrics_api.py --host 127.0.0.1
```

`--with-training` 拉起的 `run.py` 会继承当前 shell 的 `UDTCA_*` 变量（node0/node1
IP、网卡、远程仓库路径），换机器时按 [`../README.md`](../README.md) 的
「换机器 / 改 IP、网卡、目录」一节先 `source udtca-config/cluster.env` 即可。

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--host` / `--port` | `0.0.0.0` / `8000` | 监听地址与端口 |
| `--run-dir` | 自动取最新 | 只服务指定的某次 run |
| `--run-log-root` | `<repo>/run_log` | run 根目录 |
| `--log-file` | 无 | 直连单个训练日志（手工起训练时用） |
| `--poll-interval` | `1.0` | 扫日志间隔（秒），越小越实时 |
| `--window` | `20` | 滚动平均窗口步数 |
| `--with-training` | 关 | 同时拉起 `run.py` 的 4-case 训练 |
| `--max-steps` | 无 | 传给 `run.py`，冒烟用 |
| `--no-tc` | 关 | 传给 `run.py`，不限速；只验证流程时用 |
| `--install-deps` | 关 | 缺 fastapi/uvicorn 时装到项目内 `.api_deps/` |
| `--demo` / `--demo-live` | 关 | 造一份假 run（`--demo-live` 每 2 秒追加一步），没卡也能给前端联调 |

依赖：只需要 `fastapi` + `uvicorn`，且只有起 HTTP 服务时才需要。装过一次
`.api_deps/` 之后直接启动即可，不必再带 `--install-deps`；也可以用已装好依赖的
`megatron` 环境跑本服务（训练仍用自己的 `udtca` 环境）：

```bash
/root/miniconda3/envs/megatron/bin/python udtca-config/qwen14b/start_metrics_api.py
```

> **用哪个 python**：本机 PATH 里的 `python3` 是系统自带的 3.8，fastapi 的新版
> 依赖装不上也跑不起来。请显式用 conda 环境里的解释器：
>
> ```bash
> /root/miniconda3/envs/udtca/bin/python3.10 \
>     udtca-config/qwen14b/start_metrics_api.py --install-deps ...
> ```
>
> `.api_deps/` 里的包和解释器版本是绑定的：如果之前用别的 python 装过又报
> `module 'typing' has no attribute '_SpecialGenericAlias'` / `'type' object is
> not subscriptable`，删掉 `udtca-config/qwen14b/.api_deps` 用正确的解释器重装一次。

验证：

```bash
curl http://127.0.0.1:8000/api/v1/health
# {"status":"ok","run_dir":"...","cases_tracked":1,"error":null, ...}
```

---

## 3. 前端怎么调

CORS 已放开（`allow_origins=["*"]`，只允许 GET），浏览器可直接跨域请求。
注意端口要能被前端访问到（内网直连或经反向代理）。

### 3.1 接口一览

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/v1/health` | 健康检查：run_dir、已跟踪 case 数、错误信息 |
| GET | `/api/v1/cases` | 本次 run 的 case 列表，以及各 case 是否还在写日志 |
| GET | `/api/v1/metrics/latest?case=` | **最新一步**的指标（单个 case，默认正在跑的那个） |
| GET | `/api/v1/metrics/history?case=&limit=` | 最近 N 步的指标数组（默认 200，最多 5000） |
| GET | `/api/v1/metrics/stream?case=` | **SSE 实时推送**（推荐前端用这个） |
| GET | `/api/v1/overview` | 4-case 汇总对比（端到端时间 / 通信量 / 重叠率） |
| GET | `/docs` | FastAPI 自动生成的交互式文档 |

### 3.2 实时：SSE（推荐）

```js
const es = new EventSource("http://10.31.10.13:8000/api/v1/metrics/stream");

// 连上先收到一次当前状态，字段与 /metrics/latest 相同
es.addEventListener("snapshot", (e) => {
  const s = JSON.parse(e.data);
  if (s.available) render(s.metrics, s.rolling);
});

// 之后每发现新的训练步推一条；一次可能带多步，所以 metrics 是数组
es.addEventListener("metrics", (e) => {
  const { case, metrics } = JSON.parse(e.data);
  for (const m of metrics) render(m, null);
});
```

- 15 秒没有新数据，服务端发一行 `: keep-alive` 心跳（`EventSource` 会自动忽略，
  但经 Nginx 之类的代理时记得关缓冲：响应头已带 `X-Accel-Buffering: no`）。
- 断线重连由浏览器自动做；也可以用 `?case=0001` 只看某个 case。
- 端到端延迟 ≈ `--poll-interval` + 一步训练时间，不是逐 token 级别。

> **不带 `case` 时取的是"当前正在跑的那个 case"**。跑到 baseline case 时它没有
> `[rt-metric]` 行，接口会返回 `available=false`；想看刚跑完的 polar case 就显式
> 传 `?case=0001`、`?case=0003`。前端做 4-case 对比时建议一律带 `case`。

### 3.3 轮询（简单场景）

```js
const r = await fetch("http://10.31.10.13:8000/api/v1/metrics/latest");
const { available, case: caseId, metrics, rolling } = await r.json();
```

### 3.4 前端要展示的两个数

**① 重叠率**：`dp_pp_overlap_pct`（百分数，如 `69.1`）、`dp_pp_overlap`（0–1）。
口径是 Jaccard：`|DP ∩ PP| / (|DP| + |PP| − |DP ∩ PP|)`，与
`utils/overlap_interval.py` 一致。

**② 压缩量**：一组字节数与压缩比

| 字段 | 含义 |
|---|---|
| `comm_dense_bytes` / `comm_dense_gb` | 不压缩时每步要发的字节数 |
| `comm_compressed_bytes` / `comm_compressed_gb` | bitscom 压缩后每步实际发的字节数 |
| `comm_saved_bytes` / `comm_saved_gb` | 省下的量 |
| `comm_reduction_x` | 压缩比（dense ÷ compressed，示例 `7.76`） |

**速率**（每步字节数 ÷ 该步耗时，分母优先用 `pp_ms`，见 `rate_basis`）：
`comm_dense_mb_s`、`comm_compressed_mb_s`、`comm_saved_mb_s`（以及 `*_gb_s`）。

滚动平均在 `rolling` 里：`dp_pp_overlap_avg_pct`、`comm_compressed_mb_s_avg`、
`comm_reduction_x_avg` 等，适合做大数字展示；单步值适合画曲线。

完整字段示例：

```json
{
  "available": true,
  "case": "0001",
  "kind": "polar",
  "rate": "2gbit",
  "steps_seen": 1,
  "metrics": {
    "step": 0,
    "loss": 12.2479,
    "dp_pp_overlap": 0.691,
    "dp_pp_overlap_pct": 69.1,
    "overlap_available": true,
    "overlap_reason": null,
    "pp_ms": 28978.49,
    "dp_ms": 20034.36,
    "comm_dense_gb": 2.4303,
    "comm_compressed_gb": 0.3132745,
    "comm_saved_gb": 2.117,
    "comm_reduction_x": 7.76,
    "rate_basis": "pp_ms",
    "comm_dense_mb_s": 83.87,
    "comm_compressed_mb_s": 10.81,
    "comm_saved_mb_s": 73.05,
    "step_time_s": 30.0
  },
  "rolling": { "window": 1, "dp_pp_overlap_avg_pct": 69.1, "comm_reduction_x_avg": 7.76 }
}
```

> **必须处理 `overlap_available=false`**：此时 `overlap_reason` 说明原因。
> 前端应显示"暂不可用"，不要显示 0——否则看起来像重叠率为零。

### 3.5 汇总对比：`/api/v1/overview`

4 个 case 都跑完后用它做对比页。返回：

| 字段 | 说明 |
|---|---|
| `cases[]` | 每个 case 一行：`kind` / `rate` / `elapsed_s`（端到端墙钟）/ `avg_step_s` / `comm_gb` / `offline_overlap` / `rt_overlap_mean` 等 |
| `table` | 已排版好的等宽文本表（可原样展示或忽略） |
| `comparison` | 按带宽成对的文字对比（通信量、端到端耗时、重叠率） |
| `summary_json` | 若 `summary.json` 已生成则原样带上 |

`offline_overlap` 要解析 rank0 的 profiler trace（单个文件约 100 MB），第一次调用
可能要十几秒；结果按 trace 的 `(路径, mtime, size)` 缓存，整批 case 汇总也按
`case.json` 的 mtime 缓存 10 秒，所以除第一次外基本是毫秒级。解析在工作线程里做，
不会卡住 SSE 或其它接口（第一次调用期间 `/health` 仍然是 3 ms 返回）。

注意 baseline 的 case 没有 `[rt-metric]` 行，所以 `rt_overlap_mean` 为 `null`，
这是预期的——baseline 不做 DP-PP 通信重叠统计。

---

## 4. 排障

| 现象 | 处理 |
|---|---|
| 训练日志里反复刷 `[c10d] ... client socket ... err=-3`、久不进训练循环、显存不涨 | rendezvous 阶段反向 DNS 超时，不是训练代码问题。在**两台机器** `/etc/hosts` 加 `10.31.10.13 u13` / `10.31.10.177 u177`，详见 `udtca-config/README.md` 的「排障」 |
| 重新起跑报 `Address already in use` | 上次中断留下的 torchrun 还占着 29500/29501：`ss -ltnp \| grep -E '29500\|29501'` 找到 pid 后**精确** kill |
| 启动报缺 `fastapi` / `uvicorn` | 加一次 `--install-deps`（装到 `.api_deps/`，不动训练环境），或改用 `megatron` 环境 |
| 启动即报 `module 'typing' has no attribute '_SpecialGenericAlias'` / `'type' object is not subscriptable` | 用了系统 Python 3.8。改用 `/root/miniconda3/envs/udtca/bin/python3.10`，必要时删掉 `.api_deps` 重装 |
| `run_dir` 指向了上一次的 run | 传 `--run-dir` 明确指定本次目录 |
| `available=false`、`steps_seen=0` | 确认训练已进入第一个 step（模型初始化/编译要几分钟）；确认 `--run-dir` 正确 |
| `overlap_available=false` | 训练侧没算出 CUDA event，看 `overlap_reason`；训练日志里应有 `[rt-metric]` 行 |
| 前端连不上 | 确认监听 `0.0.0.0`、端口 8000 未被防火墙拦截；可先在同机 `curl 127.0.0.1:8000/api/v1/health` |
| SSE 卡住不更新 | 中间代理要关缓冲（`X-Accel-Buffering: no`），或改用轮询 `/metrics/latest` |
