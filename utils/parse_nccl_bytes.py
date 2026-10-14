#!/usr/bin/env python3
"""从 NCCL 集合通信日志里算每个 case **端到端**的通信量，并给出压缩率提升。

口径
----
`runtime_log/<case>/nccl_<host>_<pid>.log` 是训练时用
`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL NCCL_DEBUG_FILE=...` 落下来的日志。
每一行集合通信都带 `count`(元素数) 和 `datatype`，乘起来就是**交给网络的那块
buffer 的字节数**。

**把该 case 所有 rank、所有通信子、所有算子加起来**，就是这一跑端到端真正通信的
字节量：压缩过的 DP 梯度同步，和没压缩的参数广播 / TP all-reduce / PP p2p，全都在
里面 —— 这正是"端到端压缩率提升"要的口径。

为什么不用网卡计数：这台是 k8s 节点，网卡和别的租户共用，而且限速的重传会把
字节数放大，端到端比值会被污染。NCCL 自报的是"交给网络的那块 buffer"，免疫这些。

用法
    python parse_nccl_bytes.py                       # 分析 runtime_log 下所有 case
    python parse_nccl_bytes.py --cases 0016 0015
    python parse_nccl_bytes.py --detail              # 顺带打出按通信子的明细
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

DEFAULT_RUNTIME_LOG = Path(__file__).resolve().parents[2] / "runtime_log"

COLL_RE = re.compile(
    r"NCCL INFO (AllReduce|AllGather|ReduceScatter|AllToAll|Broadcast|"
    r"Send|Recv|Reduce|Gather|Scatter)\b"
)
COUNT_RE = re.compile(r"\bcount (\d+)")
DTYPE_RE = re.compile(r"\bdatatype (\d+)")
NRANKS_RE = re.compile(r"\[nranks=(\d+)\]")
COMM_RE = re.compile(r"\bcomm (0x[0-9a-fA-F]+)")

# NCCL 2.x ncclDataType_t：编号 -> (名字, 字节/元素)
NCCL_DTYPE = {
    0: ("int8", 1),
    1: ("uint8", 1),
    2: ("int32", 4),
    3: ("uint32", 4),
    4: ("int64", 8),
    5: ("uint64", 8),
    6: ("float16", 2),
    7: ("float32", 4),
    8: ("float64", 8),
    9: ("bfloat16", 2),
}


def parse_log(path: Path):
    """返回 {(op, comm): [调用次数, 元素总数, {dtype: 元素数}]}。"""
    groups = defaultdict(lambda: [0, 0, defaultdict(int)])
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if "NCCL INFO" not in line:
                continue
            m = COLL_RE.search(line)
            if not m:
                continue
            op = m.group(1)
            mc = COUNT_RE.search(line)
            if not mc:
                continue
            count = int(mc.group(1))
            md = DTYPE_RE.search(line)
            dtype = int(md.group(1)) if md else -1
            mcomm = COMM_RE.search(line)
            comm = mcomm.group(1) if mcomm else "?"
            g = groups[(op, comm)]
            g[0] += 1
            g[1] += count
            g[2][dtype] += count
    return groups


def group_bytes(counts_by_dtype):
    """把 {dtype: 元素数} 换算成字节；未知 dtype 按 0 计并单独报告。"""
    total = 0
    unknown = 0
    for dtype, cnt in counts_by_dtype.items():
        if dtype in NCCL_DTYPE:
            total += cnt * NCCL_DTYPE[dtype][1]
        else:
            unknown += cnt
    return total, unknown


def analyze_case(case_dir: Path):
    logs = sorted(case_dir.glob("nccl_*.log"))
    if not logs:
        return None
    total_bytes = 0
    rows = []
    unknown_total = 0
    for path in logs:
        for (op, comm), (calls, count_sum, by_dtype) in parse_log(path).items():
            nbytes, unknown = group_bytes(by_dtype)
            total_bytes += nbytes
            unknown_total += unknown
            dtxt = ",".join(
                f"{NCCL_DTYPE.get(d, (str(d), 0))[0]}:{c}"
                for d, c in sorted(by_dtype.items(), key=lambda x: -x[1])
            )
            rows.append((op, comm, calls, count_sum, nbytes, dtxt))
    # 合并不同 rank 的同一 (op, comm) —— comm 句柄每个 rank 不同，这里按 (op, 字节量级) 汇总
    return {"logs": len(logs), "total_bytes": total_bytes,
            "unknown_elements": unknown_total, "rows": rows}


def summarize(rows):
    """按 comm 句柄归并。

    一个通信子可能用多个 NCCL 算子：baseline 的 DP 就是纯 AllReduce，而 bitscom 的
    低比特 DP 路径会同时用 AllReduce / AllGather / SendRecv。所以必须**按 comm 汇总**，
    只取单个算子会漏掉协议开销，把压缩比算虚高。
    """
    agg = defaultdict(lambda: {"bytes": 0, "calls": 0, "ops": defaultdict(int)})
    for op, comm, calls, count_sum, nbytes, _ in rows:
        a = agg[comm]
        a["bytes"] += nbytes
        a["calls"] += calls
        a["ops"][op] += nbytes
    return agg


def fmt_gb(b):
    return f"{b / 1e9:.3f}"


def main():
    ap = argparse.ArgumentParser(description="从 NCCL 日志算通信 payload 字节（压缩率口径）")
    ap.add_argument("--runtime-log", default=str(DEFAULT_RUNTIME_LOG))
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--detail", action="store_true", help="按通信子打印明细")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    runtime_log = Path(args.runtime_log)
    case_json = runtime_log / "case.json"
    meta = {}
    if case_json.exists():
        data = json.loads(case_json.read_text())
        cases = data.get("cases", data) if isinstance(data, dict) else data
        meta = {c["case"]: c for c in cases}

    results = []
    for case_dir in sorted(p for p in runtime_log.iterdir()
                           if p.is_dir() and p.name.isdigit()):
        if args.cases and case_dir.name not in args.cases:
            continue
        info = analyze_case(case_dir)
        if info is None:
            continue
        rec = meta.get(case_dir.name, {})
        agg = summarize(info["rows"])
        # DP 通信子 = 字节最多的那个 comm
        dp = max(agg.items(), key=lambda kv: kv[1]["bytes"], default=None)
        results.append({
            "case": case_dir.name,
            "kind": rec.get("kind", "?"),
            "rate": rec.get("rate", "?"),
            "index": rec.get("test_index", "?"),
            "n_logs": info["logs"],
            "total_gb": info["total_bytes"] / 1e9,
            "dp_comm": dp[0] if dp else None,
            "dp_gb": (dp[1]["bytes"] / 1e9) if dp else None,
            "dp_ops": (dict(dp[1]["ops"]) if dp else {}),
            "dp_calls": dp[1]["calls"] if dp else 0,
            "comm_groups": sorted(
                [(k, v["bytes"] / 1e9, v["calls"], dict(v["ops"]))
                 for k, v in agg.items()],
                key=lambda x: -x[1],
            ),
        })

    if not results:
        print("没有找到 nccl_*.log（训练时没开 NCCL_DEBUG？）")
        return 1

    print(f"runtime_log = {runtime_log}")
    print("（字节 = NCCL 报的 count × dtype 大小，就是交给网络的那块 buffer）\n")
    hdr = (f"{'case':>5} {'idx':>3} {'kind':8} {'rate':7} {'logs':>5} "
           f"{'总字节GB':>10} {'DP字节GB':>10} {'DP调用':>7} {'DP算子构成':>26}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        ops = "+".join(f"{k}:{v / 1e9:.3f}" for k, v in
                       sorted(r["dp_ops"].items(), key=lambda x: -x[1]))
        dp_gb = r["dp_gb"] if r["dp_gb"] is not None else float("nan")
        print(f"{r['case']:>5} {str(r['index']):>3} {r['kind']:8} {str(r['rate']):7} "
              f"{r['n_logs']:>5} {r['total_gb']:>10.3f} {dp_gb:>10.3f} "
              f"{r['dp_calls']:>7} {ops:>26}")

    # 同 rate 的 baseline/polar 配对算压缩比
    print()
    by_rate = defaultdict(dict)
    for r in results:
        by_rate[r["rate"]][r["kind"]] = r
    if not any(len(v) == 2 for v in by_rate.values()):
        print("（还没凑齐同一 rate 的 baseline+polar 一对）")
    for rate, pair in sorted(by_rate.items()):
        if "baseline" in pair and "polar" in pair and pair["baseline"]["dp_gb"]:
            b, p = pair["baseline"]["dp_gb"], pair["polar"]["dp_gb"]
            pt = pair["polar"]["total_gb"]
            bt = pair["baseline"]["total_gb"]
            print(f"[{rate}] DP payload 压缩比 = {b:.3f} / {p:.3f} = "
                  f"{b / p if p else float('nan'):.2f}x   "
                  f"(全部通信口径 {bt:.3f} / {pt:.3f} = {bt / pt if pt else float('nan'):.2f}x)")

    if args.json:
        print()
        print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
