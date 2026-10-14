#!/usr/bin/env python3
"""按「DP 区间」和「PP 区间」的交并比算重叠率。

区间定义
--------
每个 iter 取两段墙钟区间：

    DP: [t1, t2]   DP 同步通信的起止（该 stream 上第一个 kernel 的开始 ~ 最后一个 kernel 的结束）
    PP: [t3, t4]   流水线阶段的起止（TP all-reduce 所在 stream 的跨度，跟着流水线进度走）

重叠率 = |DP ∩ PP| / (|DP| + |PP| - |DP ∩ PP|)

就是两个区间的 Jaccard 交并比：两段完全错开是 0，一段被另一段完全包住是
短的那段 / 长的那段。

只算一个 iter
-------------
trace 里只有一个 `ProfilerStep#N` 标记（3 步训练里最后落盘的那个），统一用它，
保证所有 case 口径一致。区间按这个 iter 的窗口裁剪。

用法
    python overlap_interval.py                       # runtime_log 下所有带 trace 的 case
    python overlap_interval.py --cases 0015 0016
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_overlap import (  # noqa: E402
    DEFAULT_RUNTIME_LOG, load_case_records, rank0_trace_path,
)

STEP_RE = re.compile(r"^ProfilerStep#(\d+)$")

# 代码里打的 profiler 标注名（见 polar-sgd wrapper.py / hooks.py）。脚本优先认这两个，
# 和运行时每步打印的重叠率用同一对区间；没有时才退回按 kernel 名挑流的启发式。
DP_ANN = "dp_pp_metric/DP"
PP_ANN = "dp_pp_metric/PP"


def parse_trace(path: Path):
    """返回 (steps, kernels, anns)，单位 us。

    steps   = [(name, ts, dur)]                 ProfilerStep#N 标记
    kernels = [(ts, end, name, tid)]            所有 kernel
    anns    = [(name, ts, end)]                 dp_pp_metric/DP、dp_pp_metric/PP 标注
    """
    buf = path.read_text(errors="ignore")
    start = buf.index('"traceEvents"')
    start = buf.index("[", start) + 1
    dec = json.JSONDecoder()
    n = len(buf)
    pos = start
    steps, kernels, anns = [], [], []
    while pos < n:
        while pos < n and buf[pos] in " \t\r\n,":
            pos += 1
        if pos >= n or buf[pos] == "]":
            break
        try:
            ev, end = dec.raw_decode(buf, pos)
        except ValueError:
            break
        pos = end
        if not isinstance(ev, dict):
            continue
        if ev.get("cat") == "kernel":
            kernels.append((ev.get("ts", 0.0),
                            ev.get("ts", 0.0) + ev.get("dur", 0.0),
                            ev.get("name", ""), ev.get("tid")))
        elif ev.get("cat") == "user_annotation":
            nm = ev.get("name", "")
            if STEP_RE.match(nm):
                steps.append((nm, ev.get("ts", 0.0), ev.get("dur", 0.0)))
            elif nm in (DP_ANN, PP_ANN):
                anns.append((nm, ev.get("ts", 0.0),
                             ev.get("ts", 0.0) + ev.get("dur", 0.0)))
    return steps, kernels, anns


def pick_step(steps, which):
    """which = "last" 取最后一个标记；否则按 ProfilerStep#N 的 N 取。"""
    if not steps:
        return None
    if which == "last":
        return steps[-1]
    for name, ts, dur in steps:
        if int(STEP_RE.match(name).group(1)) == int(which):
            return (name, ts, dur)
    return None


def span(intervals, lo, hi):
    """把区间裁到 [lo, hi]，返回 (起点, 终点)（相对 lo，单位 s）；没有交集返回 None。"""
    iv = [(max(a, lo), min(b, hi)) for a, b in intervals if a < hi and b > lo]
    if not iv:
        return None
    return ((min(a for a, _ in iv) - lo) / 1e6, (max(b for _, b in iv) - lo) / 1e6)


def _ann_span(anns, name, lo, hi):
    """代码打的标注区间（绝对 ts），裁到迭代窗口；没有返回 None。"""
    iv = [(a, b) for nm, a, b in anns if nm == name and a < hi and b > lo]
    if not iv:
        return None
    return (min(a for a, _ in iv), max(b for _, b in iv))


def analyze(path: Path, step_sel):
    steps, kernels, anns = parse_trace(path)
    step = pick_step(steps, step_sel)
    if step is None:
        return None
    _, ts, dur = step
    t0, t1 = ts, ts + dur
    L = (t1 - t0) / 1e6

    # 优先：代码里用 record_function 打的 dp_pp_metric/DP、dp_pp_metric/PP 标注，
    # 和运行时每步的重叠率是同一对区间（这是对齐口径的关键）。
    dpa = _ann_span(anns, DP_ANN, t0, t1)
    ppa = _ann_span(anns, PP_ANN, t0, t1)
    if dpa is not None and ppa is not None:
        dp = ((dpa[0] - t0) / 1e6, (dpa[1] - t0) / 1e6)
        pp = ((ppa[0] - t0) / 1e6, (ppa[1] - t0) / 1e6)
        mode, dp_tid, pp_tid = "annotation", None, None
    else:
        # 兜底：旧口径，按 kernel 名挑 DP/PP 流，取首尾 kernel 的跨度
        by_tid = defaultdict(list)
        for a, b, nm, tid in kernels:
            if a < t1 and b > t0:
                by_tid[tid].append((a, b, nm))

        def cnt(tid, pat):
            return sum(1 for a, b, nm in by_tid[tid] if pat in nm)

        def busy(tid):
            return sum(b - a for a, b, nm in by_tid[tid])

        # PP 流 = TP all-reduce（每层每 microbatch 一次，kernel 数最多），跨度跟着流水线走
        pp_tid = max(by_tid, key=lambda t: cnt(t, "AllReduce_Sum_bf16"), default=None)
        # DP 流：bitscom 低比特路径带 AllGather；baseline 是纯 AllReduce
        dp_tid = None
        ag = [t for t in by_tid if t != pp_tid and cnt(t, "AllGather_RING") > 0]
        if ag:
            dp_tid = max(ag, key=busy)
        else:
            cand = [t for t in by_tid if t != pp_tid and
                    (cnt(t, "AllReduce_Sum_bf16") > 0 or cnt(t, "SendRecv") > 0)]
            dp_tid = max(cand, key=busy, default=None)
        if dp_tid is None or pp_tid is None:
            return None

        dp = span([(a, b) for a, b, _ in by_tid[dp_tid]], t0, t1)
        pp = span([(a, b) for a, b, _ in by_tid[pp_tid]], t0, t1)
        if dp is None or pp is None:
            return None
        mode = "kernel"

    (d1, d2), (p1, p2) = dp, pp
    inter = max(0.0, min(d2, p2) - max(d1, p1))
    union = (d2 - d1) + (p2 - p1) - inter
    return {
        "iter_s": L,
        "dp": (d1, d2), "pp": (p1, p2),
        "dp_s": d2 - d1, "pp_s": p2 - p1, "inter_s": inter, "union_s": union,
        "overlap": inter / union if union > 0 else None,
        "dp_tid": dp_tid, "pp_tid": pp_tid,
        "step": step[0],
        "mode": mode,
    }


def load_meta(runtime_log: Path):
    return load_case_records(runtime_log)


def main():
    ap = argparse.ArgumentParser(description="DP 区间与 PP 区间的交并比重叠率")
    ap.add_argument("--runtime-log", default=str(DEFAULT_RUNTIME_LOG))
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--step", default="last", help="'last' 或 ProfilerStep 的编号")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    runtime_log = Path(args.runtime_log)
    meta = load_meta(runtime_log)
    results = []
    for case_dir in sorted(p for p in runtime_log.iterdir()
                           if p.is_dir() and p.name.isdigit()):
        if args.cases and case_dir.name not in args.cases:
            continue
        trace = rank0_trace_path(case_dir)
        if trace is None:
            continue
        m = analyze(trace, args.step)
        if m is None:
            continue
        rec = meta.get(case_dir.name, {})
        m.update({"case": case_dir.name, "kind": rec.get("kind", "?"),
                  "rate": rec.get("rate", "?"), "index": rec.get("test_index", "?"),
                  "comm_gb": rec.get("comm_gb")})
        results.append(m)

    if not results:
        print("没有找到可分析的 trace")
        return 1

    print(f"runtime_log = {runtime_log}   step = {args.step}\n")
    # 同一 rate 下 baseline/polar 成对时的通信量降低比（baseline / polar）
    by_rate = defaultdict(dict)
    for r in results:
        by_rate[r["rate"]][r["kind"]] = r

    def comm_ratio(r):
        pair = by_rate.get(r["rate"], {})
        b, p = pair.get("baseline"), pair.get("polar")
        if not b or not p:
            return None
        bg, pg = b.get("comm_gb"), p.get("comm_gb")
        if not bg or not pg:
            return None
        return bg / pg

    hdr = (f"{'case':>5} {'idx':>3} {'kind':8} {'rate':7} {'iter_s':>8} "
           f"{'DP_s':>17} {'PP_s':>17} {'通信量GB':>10} {'通信量降低比':>12} {'重叠率%':>8} {'口径':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        f = lambda iv: f"[{iv[0]:6.1f},{iv[1]:6.1f}]"
        ov = f"{r['overlap'] * 100:.1f}" if r["overlap"] is not None else "-"
        cg = f"{r['comm_gb']:.3f}" if r.get("comm_gb") is not None else "-"
        cr = comm_ratio(r)
        crs = f"{cr:.2f}x" if cr is not None else "-"
        mode = "标注" if r.get("mode") == "annotation" else "kernel兜底"
        print(f"{r['case']:>5} {str(r['index']):>3} {r['kind']:8} {str(r['rate']):7} "
              f"{r['iter_s']:>8.2f} {f(r['dp']):>17} {f(r['pp']):>17} "
              f"{cg:>10} {crs:>12} {ov:>8} {mode:>10}")
    print()
    print("DP_s / PP_s = 该 iter 内 DP 同步区间、流水线阶段区间（秒）")
    print("通信量 = 训练窗口内两机链路上的端到端字节总量（1GB=1e9B，取自 case 记录）")
    print("通信量降低比 = 同一限速下 baseline 通信量 / polar 通信量")
    print("区间单位是秒，起点相对该 iter 的 ProfilerStep 标记")
    print("重叠率 = |DP ∩ PP| / (|DP| + |PP| - |DP ∩ PP|)")
    print("口径 = 标注：用代码里 record_function 打的 dp_pp_metric/DP、dp_pp_metric/PP，"
          "与运行时每步一致；kernel兜底：按 kernel 名挑流（无标注时）")
    if args.json:
        print()
        print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
