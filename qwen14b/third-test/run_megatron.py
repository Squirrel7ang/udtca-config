#!/usr/bin/env python3
"""third-test：polar vs **Megatron-Core baseline**（跨机，默认 10 iter）。

和 `run_docker.py` 是同一份实现的第二个入口：实现、指标口径、输出格式全在
`run_docker.py` 里，这里只把默认值改掉——

    run_docker.py   ->  --modes polar,baseline   --max-steps 20   （baseline = torch DDP）
    run_megatron.py ->  --modes polar,megatron   --max-steps 10   （baseline = Megatron-Core）

**必须在 powerone1 的宿主机上运行**（容器里没有 docker / docker.sock）：

    python3 run_megatron.py                       # polar 先跑，再跑 Megatron baseline，各 10 iter
    python3 run_megatron.py --only-baseline       # 只跑 Megatron baseline
    python3 run_megatron.py --max-steps 3         # 冒烟
    python3 run_megatron.py --dry-run             # 只打印生成的脚本
    python3 run_megatron.py --megatron-fp32-grad-reduce   # 复现 Megatron 默认的 fp32 梯度 all-reduce

数据集与环境（都是踩过的坑，这里已经固化成默认行为）：

  * **数据集缓存预热**：Megatron 只在 global rank 0 先建索引缓存、其余 building rank
    过 barrier 后再建，前提是 `--data-cache-path` 是共享盘。两机都是容器内本地盘，
    node1 读不到 node0 的缓存，它的两个 building rank（first/last pipeline stage × tp0）
    会同时重建同一份索引，抢跑的 rank 撞
    `FileNotFoundError: ...-GPTDataset-train-document_index.npy`，随后一串 NCCL 连带错误。
    所以本脚本在起训前会在**两台机器上各单进程预构建一次**（见
    `run_docker.warm_megatron_index_cache`），正式跑时所有 rank 都是 cache_hit。
    缓存的 key 由 `num_samples(=max-steps×global-batch)`/`seq-len`/`seed`/`split`/
    `data-path`/tokenizer 决定，**改这几个参数就会换 key，脚本会自动重建**，不用手动处理。
    想跳过：`--skip-megatron-cache-warm`。
  * **环境变量**：`--grad-reduce-in-bf16` 默认打开（`--bf16` 在 Megatron 里会自动把梯度
    all-reduce 变成 fp32，字节数翻倍，与 polar 的 bf16 all-reduce 不同口径）；
    NCCL 固定 `NCCL_NET=Socket / ALGO=Ring / PROTO=Simple / P2P_LL_THRESHOLD=0 /
    ALLOC_P2P_NET_LL_BUFFERS=0`，与已知能跑的那份 baseline 对齐。
    要复现 Megatron 原始默认：`--megatron-fp32-grad-reduce`。
  * **数据**：默认 `--mock-data`；要真实语料先 `--megatron-prepare-data`，
    再 `--megatron-data-path <prefix>`（默认已指向镜像里的 fineweb sample-10BT）。
"""

import run_docker

if __name__ == "__main__":
    raise SystemExit(run_docker.main(default_modes="polar,megatron",
                                     default_max_steps=10,
                                     baseline_mode="megatron"))
