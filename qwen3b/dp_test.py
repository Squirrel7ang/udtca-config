#!/usr/bin/env python3
"""跨机 32 卡 DP 测试（复刻训练里卡死的那条通信，但规模和顺序故意打乱）。

拓扑（与训练一致）：
    两台机器各 16 rank，全局 rank = node_rank*16 + local_rank，即 0~15 / 16~31。
    DP 组两两配对：{0,16}, {1,17}, ..., {15,31}，跨机。

每一轮（默认 3 轮）：
    16 个 DP 对同时各做 1 次 2 卡 all-reduce，**每一对的数据量都不一样**，
    而且按固定种子打乱后**不再单调递增**——谁大谁小是乱的。
    量级：最大的一对 2 GiB，最小的一对 64 MiB，一轮 payload 合计约 5.25 GiB，
    和"两机 1 Gbps、正常单步 ~40 s ⇒ 单步大概搬了 5 GB"对得上。
    每次 all-reduce 做完，立刻做一次 world(32) barrier：
        数据量小的对很快就到 barrier 上等，数据量大的对最后到。

启动（两边各一条，先起 node1 再起 node0）：

    # powerone2（node1）
    cd /data/tangruijing/udtca/udtca-config/qwen3b
    NCCL_SOCKET_IFNAME=enx00e04c680003 NCCL_P2P_DISABLE=0 \
      torchrun --nnodes=2 --nproc_per_node=16 --node_rank=1 \
               --master_addr=10.205.200.14 --master_port=29640 dp_test.py

    # powerone1（node0）
    cd /data/tangruijing/udtca/udtca-config/qwen3b
    NCCL_SOCKET_IFNAME=ens120f0 NCCL_P2P_DISABLE=0 \
      torchrun --nnodes=2 --nproc_per_node=16 --node_rank=0 \
               --master_addr=10.205.200.14 --master_port=29640 dp_test.py

注意：
  * 不要用 timeout / kill 停。进程卡住时它自己不会退出，先抓栈再决定。
  * 想留 NCCL 全量日志就加 NCCL_DEBUG=INFO NCCL_DEBUG_FILE=/path/nccl.%h.%p.log。
  * 打印格式固定：卡住时看最后一行的 round / pair / numel 就知道卡在哪一对上。
"""

import argparse
import datetime
import os
import random
import socket
import sys
import time

import torch
import torch.distributed as dist


DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# 16 对的数据量（MiB），合计 5376 MiB ≈ 5.25 GiB。
# 最大 2 GiB、最小 64 MiB；分到哪一对由固定种子打乱，不按 pair 序号递增。
BASE_SIZE_MIB = [2048, 1024, 512, 256, 256, 192, 192,
                 128, 128, 128, 128, 96, 96, 64, 64, 64]
SHUFFLE_SEED = 20261031


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=3, help="跑几轮（每轮=每个 DP 对各 1 次 all-reduce + barrier）")
    p.add_argument("--max-bytes", type=int, default=2 * 2 ** 30,
                   help="最大那一对的字节数（默认 2 GiB）；整张表按这个值等比缩放")
    p.add_argument("--dtype", default="bf16", choices=list(DTYPE_MAP))
    p.add_argument("--timeout-s", type=int, default=1800,
                   help="process group 超时（c10d 自己的超时，不发信号）")
    return p.parse_args()


def log(msg: str) -> None:
    rank = dist.get_rank() if dist.is_initialized() else "?"
    print(f"[{time.strftime('%H:%M:%S')}][rank {rank}"
          f"/lrank {os.environ.get('LOCAL_RANK', '?')}"
          f"/{socket.gethostname()}] {msg}", flush=True)


def human(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    for unit in ("KiB", "MiB", "GiB"):
        n /= 1024.0
        if n < 1024 or unit == "GiB":
            return f"{n:.1f}{unit}"
    return f"{n}B"


def build_sizes(n_pairs: int, max_bytes: int) -> list:
    """返回每个 pair 的 numel 列表（顺序 = pair 序号），由固定种子打乱。"""
    scale = max_bytes / (BASE_SIZE_MIB[0] * 2 ** 20)
    sched = [m * 2 ** 20 for m in BASE_SIZE_MIB]
    if n_pairs <= len(sched):
        sched = sched[:n_pairs]
    else:                                   # 只有 16 对的场景，兜底
        sched = (sched * (n_pairs // len(sched) + 1))[:n_pairs]
    sched = [int(b * scale) for b in sched]
    random.Random(SHUFFLE_SEED).shuffle(sched)
    return sched


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dtype = DTYPE_MAP[args.dtype]
    itemsize = torch.tensor([], dtype=dtype).element_size()

    dist.init_process_group(
        "nccl", timeout=datetime.timedelta(seconds=args.timeout_s))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world % 2:
        log("world 必须是偶数")
        return 2

    # DP 组：{r, r+world/2}  → 32 卡时就是 {0,16}...[15,31]
    half = world // 2
    groups = [dist.new_group([r, r + half]) for r in range(half)]
    my_idx = rank % half
    my_group = groups[my_idx]
    my_pair = dist.get_process_group_ranks(my_group)

    sizes = [max(1, b // itemsize) for b in build_sizes(half, args.max_bytes)]

    total = sum(sizes) * itemsize
    log(f"init 完成 world={world} device=cuda:{local_rank} my_pair={my_pair} "
        f"dtype={args.dtype} rounds={args.rounds}")
    log(f"本对 pair={my_idx} numel={sizes[my_idx]} ({human(sizes[my_idx] * itemsize)})")
    log("全表(已打乱): " + ", ".join(
        f"{i}:{human(sizes[i] * itemsize)}" for i in range(half)))
    log(f"一轮 payload 合计={human(total)}；按 1 Gbps 满速估算 ≈{total * 8 / 1e9:.0f}s/轮")

    # 起跑线
    dist.barrier()
    log("START")

    t_all = time.monotonic()
    for rnd in range(args.rounds):
        numel = sizes[my_idx]
        t = torch.ones(numel, dtype=dtype, device=f"cuda:{local_rank}")
        expected = 1.0 * len(my_pair)

        t0 = time.monotonic()
        log(f"round={rnd} pair={my_idx} 发起 all_reduce numel={numel} "
            f"({human(numel * itemsize)})")
        dist.all_reduce(t, group=my_group)                       # ← 训练里卡死的那条通信
        t1 = time.monotonic()
        lo, hi = float(t.min().item()), float(t.max().item())
        ok = abs(lo - expected) < 1e-2 and abs(hi - expected) < 1e-2
        log(f"round={rnd} pair={my_idx} all_reduce 完成 耗时={(t1 - t0) * 1000.0:.1f}ms "
            f"min={lo:.3f} max={hi:.3f} {'OK' if ok else 'MISMATCH'}")
        del t
        torch.cuda.empty_cache()
        if not ok:
            log("结果不对，停止")
            dist.barrier()
            return 3

        t2 = time.monotonic()
        log(f"round={rnd} pair={my_idx} 进入 world barrier")
        dist.barrier()                                           # ← 每次 all-reduce 后都来一次
        log(f"round={rnd} pair={my_idx} barrier 通过 等待={(time.monotonic() - t2) * 1000.0:.1f}ms")

    log(f"全部 {args.rounds} 轮通过，总耗时={time.monotonic() - t_all:.2f}s")
    dist.barrier()
    log("DONE")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
