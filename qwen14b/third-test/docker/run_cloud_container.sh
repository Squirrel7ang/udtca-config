#!/usr/bin/env bash
# 云侧（powerone1 / 10.205.200.14）容器启动命令 —— 冻结镜像 udtca-third-test:cloud-20261105
# 与之前手动起的 udtca 容器参数一致，只是镜像换了、另外挂了一个缓存的活目录。
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
  -v /usr/local/corex/bin/ixsmi:/usr/local/corex/bin/ixsmi:ro \
  -v /usr/local/corex/lib64/libcuda.so.1:/usr/local/corex/lib64/libcuda.so.1:ro \
  -v /usr/local/corex/lib64/libcuda.so.1:/usr/local/corex/lib64/libcuda.so:ro \
  -v /usr/local/corex/lib64/libixml.so:/usr/local/corex/lib64/libixml.so:ro \
  -v /data1/tangruijing/third-test/cache:/workspace/udtca/experiments/qwen14b/cache \
  -e NCCL_SOCKET_IFNAME=ens120f0 \
  -e UDTCA_NODE0_IP=10.205.200.14 \
  -e UDTCA_NODE1_HOST=10.205.200.16 \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e HF_HOME=/home/powerone/.cache/huggingface \
  -w /workspace/udtca \
  udtca-third-test:cloud-20261105 \
  bash -c "ln -sfn /data /data1; exec sleep infinity"

docker exec third-test bash -c 'ixsmi --query-gpu=index --format=csv,noheader | wc -l; python -c "import torch;print(torch.__version__)"; ls /workspace/udtca | head -5'
