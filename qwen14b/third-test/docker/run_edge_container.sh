#!/usr/bin/env bash
# 边侧（powerone2 / 10.205.200.16）容器启动命令 —— 冻结镜像 udtca-third-test:edge-20261105
#
# 与云侧的差异只有两处，都已烧进镜像，运行期不需要挂 /data1/tangruijing/docker_libs，
# 也不需要联网拉 tokenizer：
#   1. /usr/local/corex/lib64/{libcuda.so,libcuda.so.1,libixml.so} 和 bin/ixsmi 用
#      docker_libs 那一套（云侧用宿主机自带的）；
#   2. NCCL_SOCKET_IFNAME 是 enx00e04c680003（云侧 ens120f0）。
set -euo pipefail

docker rm -f third-test 2>/dev/null || true

docker run -d --name third-test \
  --privileged --cap-add ALL \
  --network host --ipc host \
  --restart unless-stopped \
  -v /dev:/dev \
  -v /lib/modules:/lib/modules \
  -v /home:/home \
  -v /data:/data \
  -v /usr/src:/usr/src \
  -v /etc/localtime:/etc/localtime:ro \
  -e NCCL_SOCKET_IFNAME=enx00e04c680003 \
  -e UDTCA_NODE0_IP=10.205.200.14 \
  -e UDTCA_NODE1_HOST=10.205.200.16 \
  -w /workspace/udtca \
  udtca-third-test:edge-20261105 \
  bash -c "ln -sfn /data /data1; exec sleep infinity"

docker exec third-test bash -c 'ixsmi --query-gpu=index --format=csv,noheader | wc -l; python -c "import torch;print(torch.__version__)"; ls /workspace/udtca | head -5'
