#!/usr/bin/env bash
# 云侧（powerone1 / 10.205.200.14）容器启动命令 —— 冻结镜像 udtca-third-test:cloud-20261105
#
# Corex 运行时库、代码、语料、Python 依赖和 tokenizer 缓存都在镜像里，
# 这里只挂宿主资源（设备、/data 落日志、/home、/usr/src）。
set -euo pipefail

docker rm -f third-test 2>/dev/null || true

docker run -d --name third-test \
  --privileged --cap-add ALL \
  --network host --ipc host \
  --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
  --restart unless-stopped \
  -v /dev:/dev \
  -v /lib/modules:/lib/modules \
  -v /home:/home \
  -v /data:/data \
  -v /usr/src:/usr/src \
  -v /etc/localtime:/etc/localtime:ro \
  -e NCCL_SOCKET_IFNAME=ens120f0 \
  -e UDTCA_NODE0_IP=10.205.200.14 \
  -e UDTCA_NODE1_HOST=10.205.200.16 \
  -w /workspace/udtca \
  udtca-third-test:cloud-20261105 \
  bash -c "ln -sfn /data /data1; exec sleep infinity"

docker exec third-test bash -c 'ixsmi --query-gpu=index --format=csv,noheader | wc -l; python -c "import torch;print(torch.__version__)"; ls /workspace/udtca | head -5'
