"""跨机 NCCL / bitscom 连通性自检（torchrun --nnodes=2 启动，每机 1 卡）。

跑在容器 `trj-udtca` 里，两台机器各起 1 个进程（DP=2 跨机）。顺序：

  1  barrier
  2  send / recv（阻塞，双向）
  3  isend / irecv（异步，PP 激活形状 bf16，双向）
  4  all_reduce（SUM，逐元素校验）
  5  all_gather（每个 rank 的内容落到正确槽位）
  6  all_to_all_single（等分，内容校验）
  7  bitscom 低比特 all_reduce（backend_allreduce=True，fp32 + bf16，
     和同 world 的 NCCL 全精度结果比相对误差）

用法（两台机器各跑一次，cloud 先起、edge 后起；顺序反了也没事，
rendezvous 会等对端）：

    # powerone1（node0）
    bash cloud/run_nccl_xnode_cloud.sh
    # powerone2（node1）
    bash edge/run_nccl_xnode_edge.sh

**不要用 timeout / SIGTERM / kill 停它**：Corex 4.5.0 下打断在途 NCCL 会触发
XID 24 MMU page fault，卡永久挂死只能重启。进程组带着 timeout，任何集合通信
超过 `--timeout-s` 没完成，c10d 自己抛异常退出。
"""

import argparse
import os
import socket
import sys
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist

_SEQ = [0]


class _Done(Exception):
    """--only-p2p 跑完后用来直接跳出，不走后面的集合通信用例。"""


def _step(name: str, rank: int) -> None:
    _SEQ[0] += 1
    if rank == 0:
        print(f"[STEP {_SEQ[0]}] {name}", flush=True)


def _ok(tag: str, rank: int, detail: str = "") -> None:
    if rank == 0:
        print(f"[PASS] {tag} {detail}".rstrip(), flush=True)


def _barrier() -> None:
    torch.cuda.synchronize()
    dist.barrier()


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------- 1. barrier

def check_barrier(rank: int, world: int) -> None:
    _step(f"barrier（world_size={world}）", rank)
    dist.barrier()
    _ok("1) barrier", rank)


# ---------------------------------------------------------------- 2. 阻塞 send/recv

def check_p2p_blocking(device: torch.device, rank: int, world: int) -> None:
    _step("阻塞 send/recv 双向", rank)
    numel = 4096
    nxt = (rank + 1) % world
    prv = (rank - 1) % world

    # 环形：每个 rank 发给下家，从上一家收。world=2 时就是 0<->1 两个方向。
    payload = torch.full((numel,), float(rank + 1), dtype=torch.float32, device=device)
    buf = torch.zeros(numel, dtype=torch.float32, device=device)
    send_work = dist.isend(payload, dst=nxt)
    dist.recv(buf, src=prv)
    send_work.wait()
    expected = float(prv + 1)
    _assert(
        bool(torch.allclose(buf, torch.full_like(buf, expected))),
        f"rank {rank}: 从 {prv} 收的内容不对（期望全 {expected}）",
    )
    _barrier()
    _ok("2) send/recv 阻塞环形", rank, f"numel={numel}")


# ---------------------------------------------------------------- 3. 异步 isend/irecv

def check_p2p_async(device: torch.device, rank: int, world: int,
                    seq_len: int, hidden: int) -> None:
    _step("异步 isend/irecv（PP 激活形状 bf16）", rank)
    shape = (1, seq_len, hidden)
    nxt = (rank + 1) % world
    prv = (rank - 1) % world

    gen = torch.Generator(device="cpu")
    gen.manual_seed(1234)
    base = torch.randn(shape, generator=gen, dtype=torch.float32).to(torch.bfloat16)

    payload = (base + rank).to(device)
    buf = torch.zeros(shape, dtype=torch.bfloat16, device=device)

    recv_work = dist.irecv(buf, src=prv)
    send_work = dist.isend(payload, dst=nxt)
    recv_work.wait()
    send_work.wait()

    expected = (base + prv).to(device)
    _assert(bool(torch.equal(buf, expected)), f"rank {rank}: 异步 isend/irecv 内容不一致")
    _barrier()
    _ok("3) isend/irecv 异步", rank, f"shape={shape} bf16")


# ---------------------------------------------------------------- 4. all_reduce

def check_all_reduce(device: torch.device, rank: int, world: int) -> None:
    _step("all_reduce SUM", rank)
    numel = 4096
    x = torch.full((numel,), float(rank + 1), dtype=torch.float32, device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expected = float(sum(range(1, world + 1)))
    got = float(x[0].item())
    _assert(abs(got - expected) / expected < 1e-6,
            f"all_reduce 结果错: got={got} expected={expected}")
    _barrier()
    _ok("4) all_reduce SUM", rank, f"got={got} expected={expected}")


# ---------------------------------------------------------------- 5. all_gather

def check_all_gather(device: torch.device, rank: int, world: int) -> None:
    _step("all_gather", rank)
    numel = 1024
    mine = torch.full((numel,), float(rank + 1), dtype=torch.float32, device=device)
    out = [torch.zeros(numel, dtype=torch.float32, device=device) for _ in range(world)]
    dist.all_gather(out, mine)
    for i, chunk in enumerate(out):
        _assert(
            bool(torch.allclose(chunk, torch.full_like(chunk, float(i + 1)))),
            f"rank {rank}: all_gather 第 {i} 槽内容不对（期望全 {i + 1}）",
        )
    _barrier()
    _ok("5) all_gather", rank, f"world={world} numel={numel}")


# ---------------------------------------------------------------- 6. all_to_all

def check_all_to_all(device: torch.device, rank: int, world: int) -> None:
    _step("all_to_all_single 等分", rank)
    per = 256
    # rank i 发给 rank j 的那一份，全部填 i*100 + j，收回来按 j 逐段校验。
    send = torch.cat([
        torch.full((per,), float(rank * 100 + j), dtype=torch.float32, device=device)
        for j in range(world)
    ])
    recv = torch.zeros(world * per, dtype=torch.float32, device=device)
    dist.all_to_all_single(recv, send)

    for j in range(world):
        seg = recv[j * per:(j + 1) * per]
        expected = float(j * 100 + rank)
        _assert(
            bool(torch.allclose(seg, torch.full_like(seg, expected))),
            f"rank {rank}: all_to_all 第 {j} 段内容不对（期望全 {expected}）",
        )
    _barrier()
    _ok("6) all_to_all_single", rank, f"world={world} per_rank={per}")


# ---------------------------------------------------------------- 7. bitscom 低比特

def check_lowbit(device: torch.device, rank: int, world: int, lowbit_pg,
                 numel: int, bitwidth: int, dtype: torch.dtype, tag: str) -> None:
    from bitscom.api import LowBitGroup

    _step(f"{tag}（bitscom 低比特 all_reduce）", rank)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(7 + rank)
    x32 = (torch.randn(numel, generator=gen, dtype=torch.float32) * 0.01
           + 0.001 * rank).to(device)

    # 全精度参照（NCCL，始终 fp32）
    ref = x32.clone()
    dist.all_reduce(ref, op=dist.ReduceOp.SUM)

    # C++ backend 路径：训练里 DP 梯度同步走的就是这条
    x = x32.to(dtype)
    backend_group = LowBitGroup(bitwidth=bitwidth, process_group=lowbit_pg,
                                backend_allreduce=True)
    out = x.clone()
    work = backend_group.all_reduce(out, op=dist.ReduceOp.SUM, async_op=True)
    _assert(work is not None, "async_op=True 必须返回 Work")
    work.wait()

    # Python alltoall 路径作为 oracle：同一套低比特算法，只是不走 C++ backend。
    # 低比特量化本身有损，跟全精度比没有意义，真正要判的是"两条实现路径一致"。
    py_group = LowBitGroup(bitwidth=bitwidth, process_group=lowbit_pg,
                           backend_allreduce=False)
    out_py = x.clone()
    py_group.all_reduce(out_py, op=dist.ReduceOp.SUM, async_op=True).wait()

    out32 = out.to(torch.float32)
    py32 = out_py.to(torch.float32)

    # 各 rank 结果必须一致
    gathered = [torch.empty_like(out32) for _ in range(world)]
    dist.all_gather(gathered, out32)
    cross = max(float((g - gathered[0]).abs().max().item()) for g in gathered)

    finite = torch.isfinite(out32).all().to(torch.float32)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)

    # 1) backend 与 python 路径的偏差（判定项）
    scale = ref.abs().max().clamp_min(1e-12)
    vs_py = ((out32 - py32).abs().max() / scale).to(torch.float32)
    dist.all_reduce(vs_py, op=dist.ReduceOp.MAX)

    # 2) 与全精度结果的平均相对误差（只报告，低比特本来就有损）
    dense_rel = ((out32 - ref).abs().mean()
                 / ref.abs().mean().clamp_min(1e-12)).to(torch.float32)
    dist.all_reduce(dense_rel, op=dist.ReduceOp.MAX)

    tol = 1e-2 if dtype == torch.float32 else 5e-2
    _assert(float(finite.item()) == 1.0, f"{tag}: 结果里有非有限值")
    _assert(cross < 1e-5, f"{tag}: 各 rank 结果不一致，cross={cross:.3e}")
    _assert(float(vs_py.item()) < tol,
            f"{tag}: backend 路径与 Python 路径差太多 vs_py={float(vs_py.item()):.3e} "
            f"(tol={tol:.0e})")

    _barrier()
    _ok(f"7) {tag}", rank,
        f"numel={numel} dtype={dtype} bw={bitwidth} cross={cross:.3e} "
        f"backend_vs_python={float(vs_py.item()):.3e} "
        f"mean_rel_err_vs_dense={float(dense_rel.item()):.3f}")


# ---------------------------------------------------------------- main

def main() -> int:
    p = argparse.ArgumentParser(description="跨机 NCCL / bitscom 连通性自检")
    p.add_argument("--bitwidth", type=int, default=2)
    p.add_argument("--numel", type=int, default=1 << 20,
                   help="低比特 all_reduce fp32 用例的元素数（默认 1M）")
    p.add_argument("--bf16-numel", type=int, default=1 << 18,
                   help="低比特 all_reduce bf16 用例的元素数（默认 256K）")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--hidden", type=int, default=5120)
    p.add_argument("--timeout-s", type=int, default=300,
                   help="进程组超时（秒）。超时由 c10d 自己抛异常，不用外部 kill")
    p.add_argument("--skip-lowbit", action="store_true",
                   help="只测 NCCL，不 import bitscom")
    p.add_argument("--skip-p2p", action="store_true",
                   help="跳过 send/recv 两项，只测集合通信（P2P 建链有问题时用）")
    p.add_argument("--only-p2p", action="store_true",
                   help="只跑 barrier + send/recv（阻塞 + 异步），不测集合通信和低比特")
    p.add_argument("--no-eager-init", action="store_true",
                   help="init_process_group 不传 device_id（惰性初始化）。训练里是 eager，"
                        "跨机 P2P 若在 eager 下报错，用它对照")
    args = p.parse_args()
    if args.only_p2p and args.skip_p2p:
        print("ERROR: --only-p2p 与 --skip-p2p 互斥", file=sys.stderr)
        return 2

    if "LOCAL_RANK" not in os.environ:
        print("ERROR: 必须用 torchrun 启动，见文件头部的用法。", file=sys.stderr)
        return 2

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)          # Corex: 必须在建进程组前绑卡
    device = torch.device(f"cuda:{local_rank}")

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world < 2:
        print(f"ERROR: 至少需要 2 个进程（跨机），当前 WORLD_SIZE={world}", file=sys.stderr)
        return 2

    if not args.skip_lowbit:
        import bitscom
        bitscom.init(bitwidth=args.bitwidth)   # 注册 lowbit backend

    if args.no_eager_init:
        dist.init_process_group(backend="nccl",
                                timeout=timedelta(seconds=args.timeout_s))
    else:
        dist.init_process_group(backend="nccl",
                                timeout=timedelta(seconds=args.timeout_s),
                                device_id=device)

    host = socket.gethostname()
    print(f"[rank {rank}/{world}] host={host} device={torch.cuda.get_device_name(local_rank)} "
          f"torch={torch.__version__} lowbit={'off' if args.skip_lowbit else 'on'} "
          f"init={'lazy' if args.no_eager_init else 'eager'} "
          f"p2p_disable={os.environ.get('NCCL_P2P_DISABLE', '<unset>')} "
          f"ifname={os.environ.get('NCCL_SOCKET_IFNAME', '<unset>')}",
          flush=True)
    dist.barrier()
    if rank == 0:
        print(f"=== 跨机连通性自检开始（world={world}, timeout={args.timeout_s}s）===", flush=True)

    rc = 0
    try:
        check_barrier(rank, world)
        if args.skip_p2p:
            _step("send/recv 跳过（--skip-p2p）", rank)
            _barrier()
        else:
            check_p2p_blocking(device, rank, world)
            check_p2p_async(device, rank, world, args.seq_len, args.hidden)

        if args.only_p2p:
            _step("其余用例跳过（--only-p2p）", rank)
            _barrier()
            if rank == 0:
                print("=== P2P CHECKS PASSED ===", flush=True)
            raise _Done()

        check_all_reduce(device, rank, world)
        check_all_gather(device, rank, world)
        check_all_to_all(device, rank, world)

        if args.skip_lowbit:
            _step("低比特 all_reduce 跳过（--skip-lowbit）", rank)
            _barrier()
        else:
            lowbit_pg = dist.new_group(
                ranks=list(range(world)), backend="lowbit",
                timeout=timedelta(seconds=args.timeout_s),
            )
            check_lowbit(device, rank, world, lowbit_pg, args.numel,
                         args.bitwidth, torch.float32, "lowbit all_reduce fp32")
            check_lowbit(device, rank, world, lowbit_pg, args.bf16_numel,
                         args.bitwidth, torch.bfloat16, "lowbit all_reduce bf16")
            dist.destroy_process_group(lowbit_pg)

        _barrier()
        if rank == 0:
            print("=== ALL CHECKS PASSED ===", flush=True)

    except _Done:
        pass
    except Exception:
        rc = 1
        _SEQ[0] += 1
        print(f"[FAIL] rank {rank} 在 STEP {_SEQ[0]} 失败：", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
    finally:
        # 收尾顺序：barrier -> destroy。任何异常都吞掉，退出码由上面的 rc 决定。
        try:
            torch.cuda.synchronize()
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass

    return rc


if __name__ == "__main__":
    sys.exit(main())
