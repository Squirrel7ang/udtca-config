# third-test 冻结镜像（云侧 / 边侧分开打包）

两台机器的环境**不是**完全一样，所以分两个镜像：

| | 云侧 powerone1 (10.205.200.14) | 边侧 powerone2 (10.205.200.16) |
|---|---|---|
| Corex 运行时库 | 宿主机 `/usr/local/corex` 自带，运行期 bind mount 进去 | 用 `/data1/tangruijing/docker_libs` 下的 `libcuda_a.so.1` / `libixml.so` / `ixsmi` 替换，**已烧进镜像** |
| `NCCL_SOCKET_IFNAME` | `ens120f0` | `enx00e04c680003` |

除这两点，两边镜像内容完全一致（同一份基础镜像 + 同一份代码，构建时用 md5 清单核对过）。

## 镜像里有什么

```
/workspace/udtca              仓库（third-test 分支的代码）
/workspace/megatron-runtime   Megatron-Core 运行时
/workspace/megatron-data      fineweb sample-10BT 的 bin/idx 语料（17 MB）
/workspace/udtca_manifest.txt      冻结时逐文件 md5
/workspace/megatron_manifest.txt   同上
```

另外构建时执行 `pip install --no-build-isolation --no-index -e` 安装
`/workspace/udtca/polar-sgd` 和 `/workspace/udtca/bitscom`：site-packages 里会落下
指向这两个目录的 `.pth`，`bitscom` 的 C++ 扩展（`_lowbit_*`）**在各自机器上单独编译**，
不跨机器复用。

运行期除了系统核心软件栈（Corex 驱动/运行时）和 bind mount 进来的 `/data`、`/home`、
`/dev`，不再依赖镜像外的任何东西——代码、语料、Python 包全部在镜像里。

日志不放进镜像：运行期 `/data` bind mount 进来，日志落宿主机
`/data1/tangruijing/third-test/run_log/`。

## 构建

在对应机器上执行（构建上下文 `build/ctx` 由仓库 + megatron-runtime 暂存）：

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

导出：

```bash
docker save udtca-third-test:cloud-20261105 | gzip > /data1/tangruijing/third-test/images/udtca-third-test-cloud-20261105.tar.gz
docker save udtca-third-test:edge-20261105  | gzip > /data1/tangruijing/third-test/images/udtca-third-test-edge-20261105.tar.gz
```

## 起容器

- 云侧：`bash docker/run_cloud_container.sh`（powerone1 宿主机）
- 边侧：`bash docker/run_edge_container.sh`（powerone2 宿主机）

容器名都是 `third-test`，`--restart unless-stopped`。

## 跑训练

云侧宿主机：

```bash
python3 /data1/tangruijing/third-test/run_docker.py     # polar + megatron 各 20 iter，自动对比
```

默认就是 16 卡/节点、双机 32 卡（`--gpus 16 --pp 8 --tp 2` → DP=2 跨机），
开跑前做全卡健康检查（`--skip-card-check` 才会跳过）。日志落在
`/data1/tangruijing/third-test/run_log/xnode_<ts>/`。
