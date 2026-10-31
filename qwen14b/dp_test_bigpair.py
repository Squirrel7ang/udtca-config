#!/usr/bin/env python3
"""跨机 32 卡：只有 2 对做大数据量 all-reduce，其余 28 张卡直接进 barrier。

场景（用户指定）：
    只让 {0,16} 和 {1,17} 这两对做一次大数据量 all-reduce（默认 2 GiB），
    其他 28 个 rank **不做任何 all-reduce**，直接进入 world(32) barrier 等它们；
    两对做完大 all-reduce 之后才进 barrier。每轮重复一次。

    想验证的是："少数几对在传大包、其余卡都在 barrier 上等" 这种情况会不会卡死。

启动（两边各一条，先起 node1 再起 node0）：

    # powerone2（node1）
    cd /data/tangruijing/udtca/udtca-config/qwen14b
    NCCL_SOCKET_IFNAME=enx00e04c680003 NCCL_P2P_DISABLE=0 \
      torchrun --nnodes=2 --nproc_per_node=16 --node_rank=1 \
               --master_addr=10.205.200.14 --master_port=29641 dp_test_bigpair.py

    # powerone1（node0）
    cd /data/tangruijing/udtca/udtca-config/qwen14b
    NCCL_SOCKET_IFNAME=ens120f0 NCCL_P2P_DISABLE=0 \
      torchrun --nnodes=2 --nproc_per_node=16 --node_rank=0 \
               --master_addr=10.205.200.14 --master_port=29641 dp_test_bigpair.py

注意：
  * 不要用 timeout / kill 停。卡住时先抓栈。
  * 打印格式固定：卡住时看最后一行就知道是谁在等谁。
"""

import argparse
import datetime
import os
import socket
import sys
import time

import torch
import torch.distributed as dist


DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# 做大数据量 all-reduce 的 pair 序号（pair i = {i, i + world/2}）
BIG_PAIRS = (0, 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=8, help="跑几轮")
    p.add_argument("--bytes", type=int, default=2 * 2 ** 30,
                   help="大数据量那一对每次传多少字节（默认 2 GiB）")
    p.add_argument("--dtype", default="bf16", choices=list(DTYPE_MAP))
    p.add_argument("--timeout-s", type=int, default=1800)
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


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dtype = DTYPE_MAP[args.dtype]
    itemsize = torch.tensor([], dtype=dtype).element_size()

    dist.init_process_group(
        "nccl", timeout=datetime.timedelta(seconds=args.timeout_s))
    rank, world = dist.get_rank(), dist.get_world_size()
    half = world // 2
    if world != 32:
        log(f"本测试按 32 卡写的，当前 world={world}")
        return 2

    # 只有 BIG_PAIRS 这两对建组做 all-reduce，其余 rank 拿到 None
    groups = {}
    my_pair = None
    for i in BIG_PAIRS:
        g = dist.new_group([i, i + half])
        groups[i] = g
        if rank % half == i:
            my_pair = i
    my_group = groups[my_pair] if my_pair is not None else None

    numel = max(1, args.bytes // itemsize)
    log(f"init 完成 world={world} device=cuda:{local_rank} "
        f"{'参与 all_reduce 的 pair=' + str(my_pair) + ' ' + str([my_pair, my_pair + half]) if my_pair is not None else '不进任何 all_reduce 组'}")
    log(f"大数据量 pair={list(BIG_PAIRS)} 每次 numel={numel} ({human(numel * itemsize)})，"
        f"其余 {world - 2 * len(BIG_PAIRS)} 个 rank 直接进 barrier；rounds={args.rounds}")

    dist.barrier()
    log("START")

    t_all = time.monotonic()
    t = None
    if my_pair is not None:
        t = torch.ones(numel, dtype=dtype, device=f"cuda:{local_rank}")

    for rnd in range(args.rounds):
        if my_pair is not None:
            expected = 2.0
            t.fill_(1.0)
            t0 = time.monotonic()
            log(f"round={rnd} pair={my_pair} 发起 all_reduce numel={numel} "
                f"({human(numel * itemsize)})")
            dist.all_reduce(t, group=my_group)
            t1 = time.monotonic()
            lo, hi = float(t.min().item()), float(t.max().item())
            ok = abs(lo - expected) < 1e-2 and abs(hi - expected) < 1e-2
            log(f"round={rnd} pair={my_pair} all_reduce 完成 提交耗时={(t1 - t0) * 1000.0:.1f}ms "
                f"min={lo:.3f} max={hi:.3f} {'OK' if ok else 'MISMATCH'}")
            if not ok:
                log("结果不对，直接进 barrier 后退出")
                dist.barrier()
                return 3
        else:
            log(f"round={rnd} 不参与 all_reduce，直接进 barrier")

        t2 = time.monotonic()
        log(f"round={rnd} 进入 world barrier")
        dist.barrier()
        log(f"round={rnd} barrier 通过 等待={(time.monotonic() - t2) * 1000.0:.1f}ms")

    log(f"全部 {args.rounds} 轮通过，总耗时={time.monotonic() - t_all:.2f}s")
    dist.barrier()
    log("DONE")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
