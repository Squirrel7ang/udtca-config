import os
import time

import torch
import torch.distributed as dist


def log(msg):
    print(f"[rank{os.environ.get('RANK')}] {msg}", flush=True)


def main():
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    devs = [int(x) for x in os.environ.get("TEST_DEVICES", "2,3").split(",")]
    dev = devs[local]
    torch.cuda.set_device(dev)
    log(f"device={dev} set")
    dist.init_process_group(backend="nccl", init_method="env://")
    log("process group ready")

    t = torch.ones(1 << 20, device="cuda")
    dist.all_reduce(t)
    torch.cuda.synchronize()
    log("first allreduce ok")

    if os.environ.get("TEST_MODE", "term") == "normal":
        for _ in range(50):
            dist.all_reduce(t)
        torch.cuda.synchronize()
        dist.barrier()
        dist.destroy_process_group()
        log("clean exit")
        return

    # 保持最多 64 个 allreduce 在途，随时可以被 SIGTERM 打断
    handles = []
    i = 0
    while True:
        handles.append(dist.all_reduce(t, async_op=True))
        if len(handles) > 64:
            handles.pop(0).wait()
        i += 1
        if i % 1000 == 0:
            log(f"allreduce loop {i}, <=65 in flight")


if __name__ == "__main__":
    main()
