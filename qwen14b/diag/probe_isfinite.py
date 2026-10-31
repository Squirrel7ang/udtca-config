#!/usr/bin/env python3
"""探针 A：Corex 上 isfinite / isnan+isinf / nonzero 的自洽性。

目的：rank17 的诊断里，同一个 fp32 快照上
    isnan=0  isposinf=0  isneginf=0   但 (~isfinite).nonzero() 非空
这四个结论互相矛盾。这里在单卡上复现，判断是 kernel 判定不一致，
还是 nonzero 排序/内容问题。

用法（容器内）：
  python3 diag/probe_isfinite.py
"""
import os
import torch

DEV = "cuda:0"


def check(tag, t):
    """对同一份张量做四种判定，全部 host 同步后比较。"""
    f = torch.isfinite(t)
    n_nan = int(torch.isnan(t).sum().item())
    n_pinf = int(torch.isposinf(t).sum().item())
    n_ninf = int(torch.isneginf(t).sum().item())
    n_notfinite = int(f.logical_not().sum().item())
    all_finite = bool(f.all().item())
    idx = (~torch.isfinite(t)).nonzero()
    print(
        f"[{tag}] numel={t.numel()} dtype={t.dtype} "
        f"isnan={n_nan} isposinf={n_pinf} isneginf={n_ninf} "
        f"sum(~isfinite)={n_notfinite} all_isfinite={all_finite} "
        f"nonzero_rows={idx.shape[0]}",
        flush=True,
    )
    consistent = (n_nan + n_pinf + n_ninf == n_notfinite == idx.shape[0]) and (
        all_finite == (n_notfinite == 0)
    )
    if not consistent:
        print(f"[{tag}] *** INCONSISTENT ***", flush=True)
    if idx.shape[0]:
        print(f"[{tag}] first={idx[0].tolist()} last={idx[-1].tolist()}", flush=True)
    return consistent


def main():
    torch.cuda.set_device(0)
    print(f"torch={torch.__version__} device={torch.cuda.get_device_name(0)}", flush=True)

    # --- 1) 小张量：能直接看出来对不对 ---
    base = torch.randn(1024, 1024, device=DEV, dtype=torch.float32)
    check("small_fp32_clean", base)

    bad = base.clone()
    bad[0, 0] = float("inf")
    bad[1, 1] = float("-inf")
    bad[2, 2] = float("nan")
    check("small_fp32_3bad", bad)

    bad_bf16 = bad.to(torch.bfloat16)
    check("small_bf16_3bad", bad_bf16)

    # --- 2) embed grad 同尺寸：76032 x 10240 bf16（与 Qwen2.5-14B vocab 一致）---
    rows, cols = 76032, 10240
    big = torch.randn(rows, cols, device=DEV, dtype=torch.bfloat16) * 1e-4
    print(f"[big] allocated {big.numel() * big.element_size() / 2**30:.2f} GiB", flush=True)
    check("big_bf16_clean", big)

    big_bad = big.clone()
    big_bad[120217 % rows, 3232] = float("nan")
    check("big_bf16_1nan", big_bad)
    del big_bad

    # --- 3) 重复性：同一份张量反复判 100 次，看结论是否漂移 ---
    drift = 0
    for i in range(100):
        f = torch.isfinite(big)
        a = bool(f.all().item())
        n = int(f.logical_not().sum().item())
        if (not a) or n != 0:
            drift += 1
            print(f"[loop] iter={i} all_isfinite={a} notfinite={n}", flush=True)
    print(f"[loop] 100 轮里异常 {drift} 次", flush=True)

    # --- 4) 结论 ---
    print("[done] 若 big_bf16_clean / loop 出现 INCONSISTENT 或异常，"
          "则 kernel 侧问题成立", flush=True)


if __name__ == "__main__":
    main()
