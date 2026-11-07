#!/usr/bin/env python3
"""third-test：Megatron-Core **DP=4 / PP=2 / TP=4，PP 跨机**（两机各 16 卡，共 32 卡），默认 20 iter。

和 `run_megatron.py` 是同一个实现的第三个入口（实现全在 `run_docker.py`），差别只有并行切分：

    run_megatron.py      PP=8 TP=2 DP=2  + --use-tp-pp-dp-mapping
                         rank 顺序 tp-cp-ep-pp-dp，DP 在最外层 ⇒ **DP 跨机**，PP/TP 在单机内
    run_megatron_pp.py   PP=2 TP=4 DP=4  + --megatron-pp-cross-node（= 不加 mapping 参数）
                         Megatron 默认 rank 顺序 tp-cp-ep-dp-pp，PP 在最外层
                         ⇒ **stage 0 = rank 0-15（powerone1）、stage 1 = rank 16-31（powerone2），
                         PP 跨机**；TP=4 与 DP=4 都落在单机内

所以两种切分对跨机网络的用法完全不同：一个走 DP 梯度 all-reduce，一个走 PP 的
activation/梯度点对点（P2P）流水。对比「DP 跨机 vs PP 跨机」时这一点是关键。

**必须在 powerone1 的宿主机上运行**（容器里没有 docker / docker.sock）：

    python3 /data1/tangruijing/third-test/run_megatron_pp.py               # 20 iter
    python3 .../run_megatron_pp.py --max-steps 3                           # 冒烟
    python3 .../run_megatron_pp.py --dry-run                               # 只打印生成的脚本
    python3 .../run_megatron_pp.py --modes polar,megatron                  # 同一切分也跑一遍 polar
    python3 .../run_megatron_pp.py --megatron-extra-args "--use-flash-attn" # 自己加开关

优化默认打开（都是这台 Corex 栈上跑通过的组合，见
`megatron-baseline/run_qwen3_trj.sh` / Janus `run_qwen3_trj.sh`）：

    --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather
        （distributed optimizer；梯度 reduce 与参数 all-gather 与反传重叠）
    --sequence-parallel（TP 的 all-reduce 按 sequence 切开，少一半通信量）
    --log-throughput（每步打 per-GPU 吞吐）

刻意**没开**的（这个栈上没验证过，失败代价是整轮 20 iter 白跑）：

    --use-flash-attn        已知能跑的是 --attention-backend unfused，flash-attn 与它冲突
    --tp-comm-overlap       需要 userbuffers / CUDA graph，Corex 上没跑通过
    --recompute-*           省显存但变慢，PP=2/TP=4 每卡只放 24 层，用不着
    需要就透传：--megatron-extra-args "--use-flash-attn"
    （注意 --megatron-extra-args 整串覆盖，自己写全，别和上面的 --sequence-parallel 打架）

其余默认与 `run_megatron.py` 一致：`--grad-reduce-in-bf16`（与 polar 同口径）、
NCCL 固定 Socket/Ring/Simple、起训前两台各自预构建 GPTDataset 索引缓存
（gbs = micro_batches × DP = 16 × 4 = 64，20 步 ⇒ num_samples=1280，缓存 key 与
run_megatron.py 不同，会自动重建一次）。

数据：默认 `--mock-data`（随机 token，只比时间和通信）；要跟 polar 同语料就
`--megatron-data-path <prefix>`（镜像里已指向 fineweb sample-10BT）。
"""

import sys

import run_docker

# 默认值放前面，用户命令行给的参数在后面：argparse 对同名选项取最后一次出现，
# 所以命令行永远覆盖这里的默认值（--megatron-extra-args 除外，它是整串覆盖）。
DEFAULTS = [
    "--pp", "2",
    "--tp", "4",
    "--megatron-pp-cross-node",
    "--megatron-dist-opt",
    "--megatron-extra-args", "--sequence-parallel --log-throughput",
]


def main():
    sys.argv = [sys.argv[0]] + DEFAULTS + sys.argv[1:]
    return run_docker.main(default_modes="megatron",
                           default_max_steps=20,
                           baseline_mode="megatron")


if __name__ == "__main__":
    raise SystemExit(main())
