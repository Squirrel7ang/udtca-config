#!/usr/bin/env python3
"""探针 B：sync all_reduce 之后，同一流上的读到底等不等通信完成。

PyTorch 上游语义：dist.all_reduce(t)（async_op=False）在 host 上立即返回，
但 work.wait() 会把**当前流** block 在 NCCL 流的事件上，所以紧接着在同一
流上启动的 kernel 一定排在 all_reduce 之后 —— 读到的一定是规约后的值。

如果这个后端（ixccl）没做这一步，紧接着读就会读到**本地的旧值**，
在真实训练里就表现为 `torch.isfinite(grad)` 读到写了一半的梯度。

做法：两卡各持不同常数做 SUM，all_reduce 返回后立刻 .item() 读，
期望值 = 各 rank 常数之和。若读到本地值 → 缺少当前流同步。

用法（容器内，32 卡机器上任取 2 卡）：
  NGPUS=2 bash diag/run_probe_ar_sync.sh
"""
import os
import sys

import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)

    miss = 0
    total = 0

    # 1) 标量式：每个 rank 拿不同的常数，SUM 是确定的
    for it in range(20):
        t = torch.full((1,), float(rank + 1), device=dev, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        # 不再插入任何同步，直接 host 读
        got = float(t.item())
        want = float(world * (world + 1) / 2)
        total += 1
        if abs(got - want) > 1e-6:
            miss += 1
            print(f"[scalar] it={it} rank={rank} got={got} want={want} "
                  f"*** 读到旧值，缺当前流同步 ***", flush=True)

    # 2) 模拟真实路径：大张量 all_reduce 后立刻做 isfinite 判定
    n = 40_000_000  # 80 MB bf16
    for it in range(10):
        base = torch.full((n,), 1.0 / world, device=dev, dtype=torch.bfloat16)
        dist.all_reduce(base, op=dist.ReduceOp.SUM)  # SUM 应为 1.0
        finite = bool(torch.isfinite(base).all().item())
        v = float(base.float().mean().item())
        total += 1
        if (not finite) or abs(v - 1.0) > 1e-2:
            miss += 1
            print(f"[big] it={it} rank={rank} mean={v:.6f} finite={finite} "
                  f"*** 读到的不是规约后的值 ***", flush=True)

    if rank == 0:
        print(f"[result] total={total} miss={miss}", flush=True)
        print("[verdict] miss>0 → sync all_reduce 未阻塞当前流，竞态成立；"
              "miss==0 → 竞态排除，问题在 isfinite kernel 一侧", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
