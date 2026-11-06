# third-test 冻结镜像（云侧 / 边侧分开打包）

两台机器的环境**不是**完全一样，所以分两个镜像：

| | 云侧 powerone1 (10.205.200.14) | 边侧 powerone2 (10.205.200.16) |
|---|---|---|
| Corex 运行时库来源 | 宿主机 `/usr/local/corex` 现役的那一套 | `/data1/tangruijing/docker_libs` 下的 `libcuda_a.so.1` / `libixml.so` / `ixsmi` |
| `NCCL_SOCKET_IFNAME` | `ens120f0` | `enx00e04c680003` |

两份运行时库都**已烧进各自的镜像**，运行期不再 bind mount，也不再联网。

## 三测的顺序

```bash
# 1. 建容器（云侧宿主机上跑，会顺带 ssh 到边侧建边侧的容器 + 两边自检）
bash /data1/tangruijing/third-test/create_docker.sh

# 2. 跑训练（三选一，都在云侧宿主机上跑）
python3 /data1/tangruijing/third-test/run_docker.py     # polar vs polar 的 baseline（torch DDP），各 20 iter
python3 /data1/tangruijing/third-test/run_megatron.py   # polar vs Megatron-Core baseline，各 10 iter
python3 udtca-config/qwen14b/third-test/run_api.py      # 从仓库目录起：训练 + 实时指标 API（默认只跑 polar）
```

三个脚本的分工：

| 脚本 | 默认 modes | 默认步数 | 说明 |
|---|---|---|---|
| `run_api.py` | polar | 跟入口脚本（默认 20） | 在 `run_docker.py` 之上挂 `/api/v1/*` 实时指标 API（从仓库目录运行，需 `qwen14b/metrics_api.py`） |
| `run_docker.py` | `polar,baseline` | 20 | baseline = polar 的 torch-DDP 对照，不跑 Megatron |
| `run_megatron.py` | `polar,megatron` | 10 | baseline = Megatron-Core；复用 `run_docker.py` 的实现，只是默认值不同 |

`run_megatron.py` 起训前会在**两台机器上各单进程预构建一次 GPTDataset 索引缓存**
（缓存目录是容器内本地盘，只有 global rank 0 建的那份另一台读不到，node1 的 building
rank 会并发重建并撞 `FileNotFoundError`；预热后所有 rank 都是 cache_hit）。
缓存 key 随 `--max-steps`/`--micro-batches`/`--seq-len`/`--seed` 变化，脚本会自动重建。

`run_docker.py` / `run_megatron.py` 默认 16 卡/节点、双机 32 卡（`--gpus 16 --pp 8 --tp 2`
→ DP=2 跨机），开跑前做全卡健康检查（`--skip-card-check` 才会跳过）。日志落
`/data1/tangruijing/third-test/run_log/xnode_<ts>/`。

## 镜像里有什么

```
/workspace/udtca              仓库（third-test 分支的代码）
/workspace/megatron-runtime   Megatron-Core 运行时
/workspace/megatron-data      fineweb sample-10BT 的 bin/idx 语料（17 MB）
/workspace/hf-cache           Qwen2.5 tokenizer 的 HF 缓存（离线解析）
/workspace/udtca_manifest.txt        冻结时逐文件 md5
/workspace/megatron_manifest.txt     同上
/workspace/megatron_data_manifest.txt
```

构建时还会：

- 补齐基础镜像缺的运行时依赖：`tensorboard==2.21.0 datasets==3.2.0 pyarrow==18.1.0`
  （pyarrow 必须 ≤18，19+ 要 numpy≥2，而基础镜像 numpy 是 1.26.4）；
- `pip install --no-build-isolation --no-index --no-deps -e` 安装
  `/workspace/udtca/polar-sgd` 和 `/workspace/udtca/bitscom`：site-packages 里落下指向
  这两个目录的 `.pth`，`bitscom` 的 C++ 扩展（`_lowbit_*`）**在各机器上单独编译**，
  不跨机器复用；
- 设 `NCCL_LIB_DIR=/usr/local/corex/lib64`（`-lnccl` 才能解析）、
  `HF_HOME=/workspace/hf-cache`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。

运行期除了系统核心软件栈（驱动/Corex 运行时）和 bind mount 进来的 `/data`、`/home`、
`/dev`，不再依赖镜像外的任何代码、语料、脚本。

## 构建

```bash
# 云端 powerone1
docker build -t udtca-third-test:cloud-20261105 \
    -f /data1/tangruijing/third-test/build/Dockerfile.cloud \
    /data1/tangruijing/third-test/build

# 边端 powerone2
docker build -t udtca-third-test:edge-20261105 \
    -f /data1/tangruijing/third-test/build/Dockerfile.edge \
    /data1/tangruijing/third-test/build
```

构建上下文 `build/ctx`：`udtca`（仓库）+ `megatron-runtime` + `megatron-data` +
`hf-cache`，另外云侧有 `host_libs`、边侧有 `docker_libs`（各自的 Corex 运行时库）。

导出：

```bash
docker save udtca-third-test:cloud-20261105 | gzip > /data1/tangruijing/third-test/images/udtca-third-test-cloud-20261105.tar.gz
docker save udtca-third-test:edge-20261105  | gzip > /data1/tangruijing/third-test/images/udtca-third-test-edge-20261105.tar.gz
```

## 起容器

- 云侧：`bash docker/run_cloud_container.sh`（powerone1 宿主机）
- 边侧：`bash docker/run_edge_container.sh`（powerone2 宿主机）
- 两边一起：`bash create_docker.sh`（云侧宿主机，ssh 到边侧）

容器名都是 `third-test`，`--restart unless-stopped`。
