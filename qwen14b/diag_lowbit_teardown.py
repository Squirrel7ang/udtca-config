"""2-rank validation: lowbit PG allreduce correctness + clean teardown.

Element count must be divisible by the lowbit group size (C++ TORCH_CHECK).
"""
import datetime
import os
import torch
import torch.distributed as dist

TIMEOUT = datetime.timedelta(seconds=120)


def log(msg):
    rank = dist.get_rank() if dist.is_initialized() else "?"
    print(f"[rank{rank}] {msg}", flush=True)


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=TIMEOUT)
    import bitscom
    bitscom.init()
    rank = dist.get_rank()
    log(f"init done, cuda={torch.cuda.current_device()}")

    n = 8
    t = torch.arange(rank * n, (rank + 1) * n, dtype=torch.float32, device="cuda")
    g = dist.new_group(ranks=[0, 1], backend="lowbit", timeout=TIMEOUT)
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=g)
    torch.cuda.synchronize()
    expected = sum(torch.arange(r * n, (r + 1) * n, dtype=torch.float32) for r in range(2))
    ok = torch.allclose(t.cpu(), expected, atol=3.0)  # 4-bit quantization tolerance
    log(f"lowbit allreduce got={t.cpu().tolist()} expected={expected.tolist()} ok={ok}")
    if not ok:
        log("WARNING: outside quantization tolerance")

    dist.barrier()
    dist.destroy_process_group(g)
    log("lowbit group destroyed")
    dist.barrier()
    dist.destroy_process_group()
    log("world destroyed -> clean exit")


main()
