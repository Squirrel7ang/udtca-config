#!/usr/bin/env python3
"""算一个 case 的计算-通信重叠率（方案 A / B / D）和 iter/comm/comp 用时。

为什么不用最朴素的 (comp+comm-iter)/comm
----------------------------------------
朴素口径里 comp = "CUDA compute kernel 总时长"，但这个 workload 里大量时间是
流水线在等网络、没有 kernel 在跑，comp+comm 永远小于 iter，公式必然出负数。
正确做法是让 **comp = PP 阶段的墙钟时间**（含等待），这正是下面方案 A。

POLAR 的原理（决定了怎么切）
----------------------------
POLAR 在 PP 还没跑完就把 DP 同步提前发出去，没同步完的部分滚到下一个 iter。
所以每个 iter 可以拆成：
  * PP 阶段：流水线执行（compute + PP p2p + TP all-reduce）
  * DP 同步：DP 集合通信
baseline 的 DP 同步要等 PP 阶段结束才发（整块补在尾巴上）；
polar 的 DP 同步从 PP 阶段一开始就发，摊在 PP 里。

三个方案
--------
  A: (comp + comm - iter) / comm     # 你原来的公式，comp = PP 阶段墙钟
  B: 1 - exposed / comm              # exposed = DP busy 落在 PP 阶段之后的部分
  D: comm/iter 和 DP起点/iter（不做任何近似，最稳的两个比值）

怎么切 PP 阶段和 DP 流
----------------------
trace 里每个 GPU 事件带 tid（CUDA stream）。同一 process group 用一个 stream：
  * TP all-reduce 每层每 microbatch 一次 -> kernel 数最多的那个 stream，它的跨度
    正好跟着流水线进度走，用来当 **PP 阶段的起止**；
  * DP 流：baseline 取"含 AllReduce 且 busy 最大"的 stream；polar（bitscom）取
    "含 AllGather"的 stream —— 注意 polar 里另一个含 SendRecv 的 stream 是 PP p2p
    （kernel 数和 baseline 的 PP p2p 一样都是 39），不能算进 DP。

用法
    python analyze_overlap.py                       # runtime_log 下所有带 trace 的 case
    python analyze_overlap.py --cases 0016 0015
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# utils/ -> udtca-config/ -> <repo>/runtime_log
DEFAULT_RUNTIME_LOG = Path(__file__).resolve().parents[2] / "runtime_log"

STEP_RE = re.compile(r"^ProfilerStep#(\d+)$")
RANK0_PID_RE = re.compile(r"rank=0 local_rank=0 pid=(\d+)")


def rank0_trace_path(case_dir: Path):
    log = case_dir / "train_node0.log"
    pid = None
    if log.exists():
        with open(log, "r", errors="ignore") as f:
            for line in f:
                m = RANK0_PID_RE.search(line)
                if m:
                    pid = m.group(1)
                    break
    trace_dir = case_dir / "tb_trace"
    if not trace_dir.is_dir():
        return None
    # trace 文件名是 "<hostname>_<pid>.<ts>.json"，hostname 会随机器变（以前写死了
    # u62_，换到 u13 上就找不到），所以只按 pid 匹配，不认主机名前缀。
    if pid is not None:
        matches = sorted(trace_dir.glob(f"*_{pid}.*.json")) or sorted(trace_dir.glob(f"*_{pid}.json"))
        if matches:
            return matches[0]
    matches = sorted(trace_dir.glob("*.json"))
    return matches[0] if matches else None


def parse_trace(path: Path):
    """返回 (step, kernels)，kernels=[(ts, end, name, tid)]，单位 us。"""
    buf = path.read_text(errors="ignore")
    start = buf.index('"traceEvents"')
    start = buf.index("[", start) + 1
    dec = json.JSONDecoder()
    n = len(buf)
    pos = start
    steps = []
    kernels = []
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
        elif ev.get("cat") == "user_annotation" and STEP_RE.match(ev.get("name", "")):
            steps.append((ev["name"], ev.get("ts", 0.0), ev.get("dur", 0.0)))
    return (steps[-1] if steps else None), kernels


def _busy(intervals, lo, hi):
    iv = sorted((max(a, lo), min(b, hi)) for a, b in intervals if a < hi and b > lo)
    tot = 0.0
    cs = ce = None
    for s, e in iv:
        if cs is None:
            cs, ce = s, e
        elif s <= ce:
            ce = max(ce, e)
        else:
            tot += ce - cs
            cs, ce = s, e
    if cs is not None:
        tot += ce - cs
    return tot


def analyze_trace(path: Path, kind: str):
    step, kernels = parse_trace(path)
    if step is None:
        return None
    _, ts, dur = step
    t0, t1 = ts, ts + dur
    L = (t1 - t0) / 1e6

    by_tid = defaultdict(list)
    for a, b, nm, tid in kernels:
        by_tid[tid].append((a, b, nm))

    def cnt(tid, pat):
        return sum(1 for a, b, nm in by_tid[tid] if pat in nm)

    def busy(tid):
        return sum(b - a for a, b, nm in by_tid[tid] if a < t1 and b > t0)

    # TP 流 = AllReduce_bf16 kernel 数最多（每层每 microbatch 一次）
    tp_tid = max(by_tid, key=lambda t: cnt(t, "AllReduce_Sum_bf16"), default=None)
    # DP 流
    dp_tid = None
    ag = [t for t in by_tid if t != tp_tid and cnt(t, "AllGather_RING") > 0]
    if ag:
        dp_tid = max(ag, key=busy)                      # bitscom 的 DP 低比特路径
    else:
        cand = [t for t in by_tid if t != tp_tid and
                (cnt(t, "AllReduce_Sum_bf16") > 0 or cnt(t, "SendRecv") > 0)]
        dp_tid = max(cand, key=busy, default=None)
    if dp_tid is None or tp_tid is None:
        return None

    dp = [(a, b) for a, b, nm in by_tid[dp_tid]]
    tp = [(a, b) for a, b, nm in by_tid[tp_tid]]
    comm = _busy(dp, t0, t1) / 1e6
    pp_start, pp_end = min(a for a, b in tp), max(b for a, b in tp)
    comp = (pp_end - pp_start) / 1e6
    exposed = _busy(dp, pp_end, t1) / 1e6
    dp_first = min(a for a, b in dp)
    return {
        "iter_s": L, "comm_s": comm, "comp_s": comp, "exposed_s": exposed,
        "dp_tid": dp_tid, "tp_tid": tp_tid, "n_dp": len(dp),
        "dp_span": ((min(a for a, b in dp) - t0) / 1e6, (max(b for a, b in dp) - t0) / 1e6),
        "pp_span": ((pp_start - t0) / 1e6, (pp_end - t0) / 1e6),
        "A": (comp + comm - L) / comm * 100 if comm else None,
        "B": (1 - exposed / comm) * 100 if comm else None,
        "D_comm_iter": comm / L * 100 if L else None,
        "D_dpstart_iter": (dp_first - t0) / 1e6 / L * 100 if L else None,
    }


def load_case_records(runtime_log: Path):
    path = runtime_log / "case.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    cases = data.get("cases", data) if isinstance(data, dict) else data
    return {c["case"]: c for c in cases}


def collect(runtime_log: Path, only_cases=None):
    records = load_case_records(runtime_log)
    out = []
    for case_dir in sorted(p for p in runtime_log.iterdir()
                           if p.is_dir() and p.name.isdigit()):
        if only_cases and case_dir.name not in only_cases:
            continue
        trace = rank0_trace_path(case_dir)
        if trace is None:
            continue
        rec = records.get(case_dir.name, {})
        m = analyze_trace(trace, rec.get("kind", "?"))
        if m is None:
            continue
        m.update({"case": case_dir.name, "kind": rec.get("kind", "?"),
                  "rate": rec.get("rate", "?"), "index": rec.get("test_index", "?"),
                  "comm_gb": rec.get("comm_gb")})
        out.append(m)
    return out


def print_table(results):
    print(f"{'case':>5} {'idx':>3} {'kind':8} {'rate':7} {'iter_s':>8} {'comm_s':>8} "
          f"{'comp_s':>7} {'DP流':>5} {'PP跨度':>14} {'DP跨度':>14} "
          f"{'A%':>7} {'B%':>7} {'comm/iter%':>11} {'DP起点%':>8}")
    print("-" * 132)
    for r in results:
        g = lambda v: f"{v:.2f}" if v is not None else "-"
        print(f"{r['case']:>5} {str(r['index']):>3} {r['kind']:8} {str(r['rate']):7} "
              f"{g(r['iter_s']):>8} {g(r['comm_s']):>8} {g(r['comp_s']):>7} "
              f"tid{r['dp_tid']:<2} "
              f"[{r['pp_span'][0]:5.1f},{r['pp_span'][1]:5.1f}] "
              f"[{r['dp_span'][0]:5.1f},{r['dp_span'][1]:5.1f}] "
              f"{r['A']:>7.1f} {r['B']:>7.1f} {r['D_comm_iter']:>11.1f} {r['D_dpstart_iter']:>8.1f}")


def main():
    ap = argparse.ArgumentParser(description="算 iter/comp/comm 与计算-通信重叠率（A/B/D）")
    ap.add_argument("--runtime-log", default=str(DEFAULT_RUNTIME_LOG))
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    runtime_log = Path(args.runtime_log)
    results = collect(runtime_log, args.cases)
    if not results:
        print("没有找到可分析的 trace（tb_trace 为空？）")
        return 1
    print(f"runtime_log = {runtime_log}\n")
    print_table(results)
    print()
    print("comp = PP 阶段墙钟（TP all-reduce 流的跨度）; comm = DP 同步 busy")
    print("A = (comp+comm-iter)/comm     —— 你原来的公式，comp 换成 PP 阶段墙钟")
    print("B = 1 - exposed/comm          —— exposed = DP 落在 PP 阶段结束之后的部分")
    print("D = 不做近似的两个比值：comm/iter、DP 起点/iter")
    if args.json:
        print()
        print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
