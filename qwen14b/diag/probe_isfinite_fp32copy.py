#!/usr/bin/env python3
"""探针 A2：复现诊断路径的判定不一致。

真实诊断里这段代码跑出来的结论互相矛盾：
    gf = grad.float()                              # 778M -> 3.1 GB fp32
    n_nan  = isnan(gf).sum().item()                # 0
    n_pinf = isposinf(gf).sum().item()             # 0
    n_ninf = isneginf(gf).sum().item()             # 0
    bad    = (~isfinite(gf)).nonzero()             # 非空！且 first/last 倒序

探针 A 只测了 bf16 原张量、坏元素 <=3 个，结果完全自洽。
这里补两个变量：**fp32 拷贝** + **坏元素数量很多**。

用法（容器内，指定一张空闲卡）：
  CUDA_VISIBLE_DEVICES=2 python3 diag/probe_isfinite_fp32copy.py
"""
import torch

DEV = "cuda:0"
ROWS, COLS = 76032, 10240


def probe(tag, base, n_bad, dtype_out):
    """在 base 上注入 n_bad 个 NaN 和 n_bad 个 +inf，然后在 dtype_out 上做判定。"""
    t = base.clone()
    if n_bad:
        flat = t.view(-1)
        idx_nan = torch.arange(0, n_bad, device=DEV) * 7 % flat.numel()
        idx_inf = (torch.arange(0, n_bad, device=DEV) * 11 + 5) % flat.numel()
        flat[idx_nan] = float("nan")
        flat[idx_inf] = float("inf")
        del flat
    g = t if dtype_out == torch.bfloat16 else t.float()

    n_nan = int(torch.isnan(g).sum().item())
    n_pinf = int(torch.isposinf(g).sum().item())
    n_ninf = int(torch.isneginf(g).sum().item())
    n_not = int((~torch.isfinite(g)).sum().item())
    allf = bool(torch.isfinite(g).all().item())
    idx = (~torch.isfinite(g)).nonzero()
    rows = idx.shape[0]

    ok = (n_nan + n_pinf + n_ninf == n_not == rows) and (allf == (n_not == 0))
    print(f"[{tag}] dtype={g.dtype} injected={n_bad}nan+{n_bad}inf "
          f"isnan={n_nan} +inf={n_pinf} -inf={n_ninf} sum~finite={n_not} "
          f"allfinite={allf} nonzero={rows} -> {'OK' if ok else '*** MISMATCH ***'}",
          flush=True)
    if rows:
        first, last = idx[0].tolist(), idx[-1].tolist()
        ascending = first <= last
        print(f"[{tag}] first={first} last={last} "
              f"{'' if ascending else '*** 下标倒序（nonzero 未按行主序输出）***'}", flush=True)
    del g
    return ok


def main():
    torch.cuda.set_device(0)
    print(f"torch={torch.__version__}", flush=True)
    base = (torch.randn(ROWS, COLS, device=DEV, dtype=torch.bfloat16) * 1e-4)
    print(f"base {ROWS}x{COLS} bf16 = "
          f"{base.numel() * base.element_size() / 2**30:.2f} GiB", flush=True)

    bad = 0
    # 诊断里实际走的是 fp32 拷贝 + 未知数量的坏元素，这里扫一遍
    for n in (0, 1, 1000, 1_000_000):
        for dt in (torch.bfloat16, torch.float32):
            bad += 0 if probe(f"n={n}", base, n, dt) else 1
    print(f"[done] 不一致次数={bad}", flush=True)


if __name__ == "__main__":
    main()
