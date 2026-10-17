#!/usr/bin/env python3
"""Qwen14B 训练实时指标 API：给前端平台查询 DP-PP 重叠率与通信压缩量。

本服务**只读**，数据全部来自训练进程已经落盘的东西，不会碰训练、不会碰 GPU：

    <run_dir>/<case>/train_node0.log   polar-sgd 每步打的 `[rt-metric]` 行（主数据源）
    <run_dir>/<case>/step_csv/*.csv    每步耗时（算 MB/s 的分母）
    <run_dir>/case.json                本次 run 的 case 索引（kind / rate / log_dir）
    <run_dir>/summary.json             run.py 跑完写的 4-case 汇总（存在就直接用）

为什么「读文件」也能做到实时（这条链子断了实时性就没了，改动训练侧时要注意）：

  1. 训练进程：wrapper._emit_rt_metric 用 `print(..., flush=True)` 打印，每步一行
     （不受 --log-interval 控制）。flush=True 把 Python 侧缓冲推进内核，这是第 1 个必要条件。
  2. 编排器：case_common.execute_command 用
     `bash -o pipefail -c '<cmd>' 2>&1 | tee -a <case>/train_node0.log`
     起训练，tee 每收到一行就写文件，这是第 2 个必要条件。
     实测（本机 GNU tee）：行在 print 之后 ~0.1s 内就出现在文件里。
  3. step_csv：wrapper._append_step_csv 每步 `with open(path, "a")` 写一行，
     with 退出即 close → 立刻落盘。

所以本服务按 `--poll-interval`（默认 1.0s）增量读文件，端到端延迟 ≈ poll 间隔 + 一步的
训练时间，不额外引入缓冲。手工起训练（没有 runner 的 tee 包裹）时用 `--log-file` 直连
一个日志文件即可，但那样必须自己保证日志是实时的（`| tee` 或 `stdbuf -oL`）。

训练侧实时打的那行长这样（polar-sgd/src/psgd/parallelism/polar/wrapper.py
的 `_emit_rt_metric`）：

    [rt-metric] step=3 loss=11.7841 pp_ms=1234.50 dp_ms=321.00 dp_pp_overlap=21.3%
                comm_dense=2.4303GB comm_compressed=313.2745MB
                comm_saved=2.1170GB comm_reduction=7.76x

本服务把它展开成结构化 JSON，并补上两样训练侧没直接给的东西：

  * 通信压缩量的**速率**（MB/s、GB/s）：每步字节数 ÷ 该步耗时，分母优先用
    `pp_ms`（整条流水线一步的墙钟），拿不到就退回 `step_csv` 的 `step_time_s`；
  * **滚动平均**（默认最近 20 步），避免前端只看到单步抖动。

口径与 udtca-config/utils/overlap_interval.py 完全一致：

    DP-PP 重叠率 = |DP ∩ PP| / (|DP| + |PP| − |DP ∩ PP|)

字段缺失不会让接口报错：`dp_pp_overlap` 拿不到时 `overlap_available=false`，
并在 `overlap_reason` 里说明缺在哪（例如训练侧那条 `except Exception` 吞掉了
CUDA event 的错误），方便平台如实展示而不是显示一个假的 0。

接口一览（默认 http://<host>:8000）：

    GET /api/v1/health              服务与数据源状态
    GET /api/v1/cases               本次 run 里的 case 列表与训练是否还在跑
    GET /api/v1/metrics/latest      最新一步的指标（含 MB/s、GB/s、滚动平均）
    GET /api/v1/metrics/history     最近 N 步的指标序列
    GET /api/v1/metrics/stream      SSE，实时推送每一步
    GET /api/v1/overview            4 个 case 的性能比对 + 通信量端到端优化
    GET /docs                       FastAPI 自动生成的接口文档

SSE 的两种事件（前端照这个解析；15s 没新数据会发一行 `: keep-alive` 心跳）：

    event: snapshot           连上就先推一次当前状态，等于 /metrics/latest 的 body
    data: {"available":..., "case":..., "metrics": {...}, "rolling": {...}}

    event: metrics             之后每次轮询到新步就推一条；一次可能带多步，所以是数组
    data: {"case": "0001", "metrics": [ {...}, {...} ]}

用法（在跑训练的那台机器上，即 node 0 / u62 / u13）：

    # 只起 API，服务已经在别处跑着的训练
    python udtca-config/qwen14b/metrics_api.py --port 8000

    # 指定某次 run（默认自动挑 run_log 下最新的那个）
    python udtca-config/qwen14b/metrics_api.py --run-dir run_log/run_2026-10-14_09-21-51

    # 手工起的训练，没有 run 目录：直接盯日志文件
    python udtca-config/qwen14b/metrics_api.py --log-file /tmp/train_node0.log

    # 没有卡也能演示：造一份假的 run 数据，并每 2 秒追加一步模拟训练
    python udtca-config/qwen14b/metrics_api.py --demo --demo-live

一键「起 API + 起训练」用同目录的 start_metrics_api.py。

依赖：只需要 fastapi + uvicorn，而且只有起 HTTP 服务时才需要——解析/汇总那一层是纯标准库。
本机（u13）上 `megatron` 环境里已经有（fastapi 0.128.0 / uvicorn 0.40.0），可以直接用它跑本
服务，训练仍用自己的环境，两个进程互不影响：
    /root/miniconda3/envs/megatron/bin/python udtca-config/qwen14b/metrics_api.py
也可以用 start_metrics_api.py --install-deps 把依赖装到项目内的 .api_deps/。
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import re
import sys
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

CONFIG_DIR = Path(__file__).resolve().parent          # udtca-config/qwen14b
CONFIG_ROOT = CONFIG_DIR.parent                        # udtca-config
REPO_ROOT = CONFIG_ROOT.parent                         # udtca
for _p in (CONFIG_ROOT / "common", CONFIG_ROOT / "utils", CONFIG_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# 复用 run.py 的解析与汇总口径，保证 API 和命令行汇总看到的是同一套数
import run as qrun  # noqa: E402

# fastapi 只在真正起服务时（create_app / main）才 import：
# 这样解析与采集这一层不依赖任何三方库，可以直接 import 出来对着真实日志做单测。
_FASTAPI_HINT = (
    "缺少依赖 {name}。二选一：\n"
    "  pip install fastapi uvicorn\n"
    "  或者用同目录的 start_metrics_api.py —— 它会把依赖装到项目内的 .api_deps/，"
    "不动 trj-test 环境"
)


DEFAULT_RUN_LOG_ROOT = REPO_ROOT / "run_log"
DEFAULT_CASE_JSON = "case.json"

# 与 run.py 保持同一套：先长单位后短单位，否则 "1.5GB" 会被 "B" 先吃掉
_BYTE_UNITS: Tuple[Tuple[str, float], ...] = (
    ("GB", 1e9),
    ("MB", 1e6),
    ("KB", 1e3),
    ("B", 1.0),
)

_RT_RE = re.compile(r"\[rt-metric\]\s+(.*)")
_KV_RE = re.compile(r"(\w+)=([^\s]+)")
# 训练还活着吗：日志最后一行的时间戳，或者进程还在写文件
_LOSS_RE = re.compile(r"Step (\d+), Loss:\s*([0-9.]+)")


# ---------------------------------------------------------------- 解析工具

def parse_bytes(text: Any) -> Optional[float]:
    """把 `313.2745MB` / `2.4303GB` 解析成字节数；解析不了返回 None。"""
    if text is None:
        return None
    s = str(text).strip()
    for unit, mul in _BYTE_UNITS:
        if s.endswith(unit):
            try:
                return float(s[: -len(unit)]) * mul
            except ValueError:
                return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_float(text: Any, *, strip_pct: bool = False) -> Optional[float]:
    if text is None:
        return None
    s = str(text).strip()
    if strip_pct and s.endswith("%"):
        s = s[:-1]
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def step_times_of(case_dir: Path) -> Dict[int, float]:
    """读 case 目录下所有 step_csv，返回 {step: 步耗时秒}。

    同一个 step 可能有多行（TP/DP 各写一份自己的进程），按 step 取**最慢**的那个，
    瓶颈才是这一步的真实耗时。run.py 只读字典序第一个文件，这里把全部文件都并进来，
    换机器/换 rank 布局后更稳。
    """
    csv_dir = case_dir / "step_csv"
    files = sorted(csv_dir.glob("*.csv")) if csv_dir.is_dir() else []
    per_step: Dict[int, float] = {}
    for path in files:
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    try:
                        step = int(float(row["step"]))
                        cost = float(row["step_time_s"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if step not in per_step or cost > per_step[step]:
                        per_step[step] = cost
        except OSError:
            continue
    return per_step


def rank0_trace_path(case_dir: Path) -> Optional[Path]:
    """定位 rank0 的 profiler trace，**不依赖主机名**。

    utils/analyze_overlap.py 里写死了 `u62_<pid>.*.json`；这套代码现在要能在
    u13/u177 上跑，所以这里改成：先从日志里取 rank0 的 pid，再按 `*_<pid>.*.json`
    匹配，匹配不到就退回目录里任意一个 trace。
    """
    trace_dir = case_dir / "tb_trace"
    if not trace_dir.is_dir():
        return None
    pid = None
    log = case_dir / "train_node0.log"
    if log.exists():
        m = re.search(r"rank=0 local_rank=0 pid=(\d+)",
                      log.read_text(errors="ignore"))
        if m:
            pid = m.group(1)
    if pid:
        hits = sorted(trace_dir.glob(f"*_{pid}.*.json"))
        if hits:
            return hits[0]
    hits = sorted(trace_dir.glob("*.json"))
    return hits[0] if hits else None


# ---------------------------------------------------------------- 单步指标

@dataclass
class MetricSnapshot:
    """一步的指标。字节数是该步的**增量**（训练侧每步 reset 累加器）。"""

    case: str
    step: int
    ts: float
    loss: Optional[float] = None
    pp_ms: Optional[float] = None
    dp_ms: Optional[float] = None
    dp_pp_overlap: Optional[float] = None          # 0..1
    comm_dense_bytes: Optional[float] = None
    comm_compressed_bytes: Optional[float] = None
    comm_saved_bytes: Optional[float] = None
    comm_reduction: Optional[float] = None
    step_time_s: Optional[float] = None            # 来自 step_csv
    rate_basis: Optional[str] = None               # "pp_ms" | "step_time_s" | None
    overlap_reason: Optional[str] = None
    raw: str = ""

    @property
    def rate_denom_s(self) -> Optional[float]:
        if self.pp_ms and self.pp_ms > 0:
            return self.pp_ms / 1000.0
        if self.step_time_s and self.step_time_s > 0:
            return self.step_time_s
        return None

    def _per_s(self, num_bytes: Optional[float]) -> Optional[float]:
        d = self.rate_denom_s
        if num_bytes is None or d is None:
            return None
        return num_bytes / d

    def to_dict(self) -> Dict[str, Any]:
        dense_ps = self._per_s(self.comm_dense_bytes)
        comp_ps = self._per_s(self.comm_compressed_bytes)
        saved_ps = self._per_s(self.comm_saved_bytes)

        def gbps(v: Optional[float]) -> Optional[float]:
            return None if v is None else v / 1e9

        def mbps(v: Optional[float]) -> Optional[float]:
            return None if v is None else v / 1e6

        return OrderedDict(
            [
                ("case", self.case),
                ("step", self.step),
                ("timestamp", datetime.fromtimestamp(self.ts).isoformat(timespec="seconds")),
                ("loss", self.loss),
                # ---- DP-PP 重叠率（0..1 和百分比都给，前端爱用哪个用哪个）----
                ("dp_pp_overlap", self.dp_pp_overlap),
                ("dp_pp_overlap_pct",
                 None if self.dp_pp_overlap is None else round(self.dp_pp_overlap * 100.0, 2)),
                ("overlap_available", self.dp_pp_overlap is not None),
                ("overlap_reason", self.overlap_reason),
                ("pp_ms", self.pp_ms),
                ("dp_ms", self.dp_ms),
                # ---- 通信压缩量：每步字节 ----
                ("comm_dense_bytes", self.comm_dense_bytes),
                ("comm_compressed_bytes", self.comm_compressed_bytes),
                ("comm_saved_bytes", self.comm_saved_bytes),
                ("comm_dense_gb",
                 None if self.comm_dense_bytes is None else self.comm_dense_bytes / 1e9),
                ("comm_compressed_gb",
                 None if self.comm_compressed_bytes is None else self.comm_compressed_bytes / 1e9),
                ("comm_saved_gb",
                 None if self.comm_saved_bytes is None else self.comm_saved_bytes / 1e9),
                ("comm_reduction_x", self.comm_reduction),
                # ---- 通信压缩量：速率（前端主要看这几个）----
                ("rate_basis", self.rate_basis),
                ("comm_dense_gb_s", gbps(dense_ps)),
                ("comm_compressed_gb_s", gbps(comp_ps)),
                ("comm_saved_gb_s", gbps(saved_ps)),
                ("comm_dense_mb_s", mbps(dense_ps)),
                ("comm_compressed_mb_s", mbps(comp_ps)),
                ("comm_saved_mb_s", mbps(saved_ps)),
                ("step_time_s", self.step_time_s),
            ]
        )


def snapshot_from_kv(kv: Dict[str, str], case: str, ts: float,
                     step_times: Dict[int, float]) -> Optional[MetricSnapshot]:
    step = parse_float(kv.get("step"))
    if step is None:
        return None
    step_i = int(step)

    snap = MetricSnapshot(case=case, step=step_i, ts=ts, raw=kv.get("__raw__", ""))
    snap.loss = parse_float(kv.get("loss"))
    snap.pp_ms = parse_float(kv.get("pp_ms"))
    snap.dp_ms = parse_float(kv.get("dp_ms"))

    overlap = parse_float(kv.get("dp_pp_overlap"), strip_pct=True)
    if overlap is not None:
        snap.dp_pp_overlap = overlap / 100.0
    else:
        # 训练侧那行没带 dp_pp_overlap：不是「重叠率 0」，而是没测出来。
        # 如实告诉前端，别让它显示一个假的 0。
        if kv.get("pp_ms") is None:
            snap.overlap_reason = (
                "训练侧未输出 pp_ms/dp_ms/dp_pp_overlap：wrapper._emit_rt_metric 里 "
                "CUDA event 的 elapsed_time 抛了异常被 except 吞掉（同一步的 comm_* 字段正常）"
            )
        else:
            snap.overlap_reason = "该步未测量 DP-PP 重叠率"

    snap.comm_dense_bytes = parse_bytes(kv.get("comm_dense"))
    snap.comm_compressed_bytes = parse_bytes(kv.get("comm_compressed"))
    snap.comm_saved_bytes = parse_bytes(kv.get("comm_saved"))
    if snap.comm_saved_bytes is None and snap.comm_dense_bytes is not None \
            and snap.comm_compressed_bytes is not None:
        snap.comm_saved_bytes = snap.comm_dense_bytes - snap.comm_compressed_bytes
    red = kv.get("comm_reduction")
    if red is not None:
        snap.comm_reduction = parse_float(red.rstrip("x"))
    if snap.comm_reduction is None and snap.comm_dense_bytes and snap.comm_compressed_bytes:
        snap.comm_reduction = snap.comm_dense_bytes / snap.comm_compressed_bytes

    snap.step_time_s = step_times.get(step_i)
    if snap.pp_ms and snap.pp_ms > 0:
        snap.rate_basis = "pp_ms"
    elif snap.step_time_s and snap.step_time_s > 0:
        snap.rate_basis = "step_time_s"
    return snap


# ---------------------------------------------------------------- 增量读日志

class LogTail:
    """按行增量读一个正在被追加的日志文件。

    训练日志可能有几十 MB 且每步都在长，整文件重读既慢又抖；这里只读新增的字节，
    并且把最后一行不完整的残行留到下次（tee/缓冲可能把一行切成两半）。
    """

    def __init__(self, path: Path):
        self.path = path
        self._offset = 0
        self._partial = ""
        self._inode: Optional[int] = None

    def read_new_lines(self) -> List[str]:
        try:
            st = self.path.stat()
        except OSError:
            return []
        # 日志被轮转/重建过就从头读
        if self._inode is not None and st.st_ino != self._inode:
            self._offset = 0
            self._partial = ""
        self._inode = st.st_ino
        if st.st_size < self._offset:
            self._offset = 0
            self._partial = ""
        if st.st_size == self._offset:
            return []
        try:
            with open(self.path, "r", errors="ignore") as fh:
                fh.seek(self._offset)
                chunk = fh.read()
                self._offset = fh.tell()
        except OSError:
            return []
        text = self._partial + chunk
        lines = text.split("\n")
        self._partial = lines.pop()          # 最后一段可能不完整，留到下次
        return lines


# ---------------------------------------------------------------- 数据仓库

@dataclass
class CaseState:
    case: str
    case_dir: Path
    kind: Optional[str] = None
    rate: Optional[str] = None
    tail: Optional[LogTail] = None
    step_times: Dict[int, float] = field(default_factory=dict)
    step_times_mtime: float = 0.0
    history: Deque[MetricSnapshot] = field(default_factory=lambda: deque(maxlen=5000))
    seen: set = field(default_factory=set)
    log_mtime: float = 0.0


class MetricStore:
    """把某个 run 目录下的 `[rt-metric]` 行维护成内存里的时间序列。"""

    def __init__(self, run_log_root: Path, run_dir: Optional[Path] = None,
                 window: int = 20, poll_interval: float = 1.0,
                 log_file: Optional[Path] = None):
        self.run_log_root = Path(run_log_root)
        self.run_dir: Optional[Path] = Path(run_dir) if run_dir else None
        # 手工起训练（没有 runner 的 tee 包裹）时直接盯一个日志文件
        self.log_file: Optional[Path] = Path(log_file) if log_file else None
        self.window = max(1, int(window))
        self.poll_interval = float(poll_interval)
        self.cases: "OrderedDict[str, CaseState]" = OrderedDict()
        self._lock = threading.Lock()
        self._subscribers: List[asyncio.Queue] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.error: Optional[str] = None
        self.last_scan = 0.0

    # ---- 发现 run / case ----

    def resolve_run_dir(self) -> Optional[Path]:
        if self.run_dir and (self.run_dir / DEFAULT_CASE_JSON).exists():
            return self.run_dir
        if self.run_dir and self.run_dir.is_dir() and any(self.run_dir.glob("*/")):
            return self.run_dir
        if not self.run_log_root.is_dir():
            return None
        runs = sorted((p for p in self.run_log_root.glob("run_*") if p.is_dir()),
                      key=lambda p: p.name)
        return runs[-1] if runs else None

    def _case_meta(self) -> Dict[str, Dict[str, Any]]:
        run_dir = self.resolve_run_dir()
        if run_dir is None:
            return {}
        path = run_dir / DEFAULT_CASE_JSON
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        cases = data.get("cases", []) if isinstance(data, dict) else []
        out: Dict[str, Dict[str, Any]] = {}
        for rec in cases:
            if isinstance(rec, dict) and rec.get("case"):
                out[str(rec["case"])] = rec
        return out

    def _discover_cases(self) -> None:
        run_dir = self.resolve_run_dir()
        self.run_dir = run_dir
        if run_dir is None:
            self._discover_adhoc()
            return
        meta = self._case_meta()
        for case_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            case = case_dir.name
            if not case.isdigit():
                continue
            rec = meta.get(case, {})
            state = self.cases.get(case)
            if state is None:
                state = CaseState(case=case, case_dir=case_dir)
                # 训练侧只在 node0(rank14) 打 [rt-metric]；node1 也一起读，
                # 万一以后换 rank 布局不至于看不见。
                for name in ("train_node0.log", "train_node1.log"):
                    state.tail = state.tail or LogTail(case_dir / name)
                state.tails = [LogTail(case_dir / n)
                               for n in ("train_node0.log", "train_node1.log")]
                self.cases[case] = state
            state.kind = rec.get("kind", state.kind)
            state.rate = rec.get("rate", state.rate)
            state.record = rec
        self._discover_adhoc()

    def _discover_adhoc(self) -> None:
        """`--log-file` 指定的日志：当成一个独立的 case 跟踪。"""
        if self.log_file is None:
            return
        state = self.cases.get("adhoc")
        if state is None:
            state = CaseState(case="adhoc", case_dir=self.log_file.parent)
            state.tails = [LogTail(self.log_file)]
            state.kind = "polar"
            self.cases["adhoc"] = state

    # ---- 增量采样 ----

    def refresh(self) -> None:
        run_dir = self.resolve_run_dir()
        if run_dir is None and self.log_file is None:
            self.error = f"{self.run_log_root} 下没有 run_* 目录"
            return
        self.error = None
        if run_dir != self.run_dir or not self.cases:
            with self._lock:
                self._discover_cases()

        for state in list(self.cases.values()):
            # step_csv 会在训练过程中不断增长，mtime 变了才重读
            csv_dir = state.case_dir / "step_csv"
            try:
                csv_mtime = max((p.stat().st_mtime for p in csv_dir.glob("*.csv")),
                                default=0.0)
            except OSError:
                csv_mtime = 0.0
            if csv_mtime > state.step_times_mtime:
                state.step_times = step_times_of(state.case_dir)
                state.step_times_mtime = csv_mtime

            new: List[MetricSnapshot] = []
            for tail in getattr(state, "tails", []):
                for line in tail.read_new_lines():
                    m = _RT_RE.search(line)
                    if not m:
                        continue
                    kv: Dict[str, str] = {}
                    for k, v in _KV_RE.findall(m.group(1)):
                        kv[k] = v
                    snap = snapshot_from_kv(kv, state.case, time.time(), state.step_times)
                    if snap is None:
                        continue
                    key = (snap.step, kv.get("comm_compressed"), kv.get("loss"))
                    if key in state.seen:
                        # 同一 step 被两个 rank/tp 各打一行且数值相同，只留一条；
                        # 数值不同的（真·两 rank）保留，前端能看到差异
                        continue
                    state.seen.add(key)
                    new.append(snap)
                try:
                    state.log_mtime = max(state.log_mtime, tail.path.stat().st_mtime)
                except OSError:
                    pass
            if new:
                state.history.extend(new)
                self._publish(state, new)

    def _publish(self, state: CaseState, snaps: List[MetricSnapshot]) -> None:
        if self._loop is None:
            return
        payload = {
            "case": state.case,
            "metrics": [s.to_dict() for s in snaps],
        }
        for q in list(self._subscribers):
            try:
                self._loop.call_soon_threadsafe(q.put_nowait, payload)
            except Exception:
                pass

    # ---- 读取 ----

    def active_case(self, case: Optional[str] = None) -> Optional[CaseState]:
        if case:
            return self.cases.get(case)
        best = None
        for state in self.cases.values():
            if not state.history:
                # 没数据的 case 也可能是正在跑的（第一行还没打出来）
                pass
            if best is None:
                best = state
                continue
            if state.log_mtime > best.log_mtime:
                best = state
        return best

    @staticmethod
    def _rolling(snaps: List[MetricSnapshot], window: int) -> Dict[str, Any]:
        tail = snaps[-window:]
        ov = [s.dp_pp_overlap for s in tail if s.dp_pp_overlap is not None]
        comp = [s.to_dict()["comm_compressed_mb_s"] for s in tail
                if s.to_dict()["comm_compressed_mb_s"] is not None]
        saved = [s.to_dict()["comm_saved_mb_s"] for s in tail
                 if s.to_dict()["comm_saved_mb_s"] is not None]
        dense = [s.to_dict()["comm_dense_mb_s"] for s in tail
                 if s.to_dict()["comm_dense_mb_s"] is not None]

        def avg(xs):
            return round(sum(xs) / len(xs), 4) if xs else None

        return OrderedDict(
            [
                ("window", len(tail)),
                ("dp_pp_overlap_avg", avg(ov)),
                ("dp_pp_overlap_avg_pct", None if not ov else round(avg(ov) * 100.0, 2)),
                ("comm_compressed_mb_s_avg", avg(comp)),
                ("comm_compressed_gb_s_avg", None if avg(comp) is None else round(avg(comp) / 1000.0, 6)),
                ("comm_dense_mb_s_avg", avg(dense)),
                ("comm_saved_mb_s_avg", avg(saved)),
                ("comm_reduction_x_avg",
                 avg([s.comm_reduction for s in tail if s.comm_reduction is not None])),
            ]
        )

    def latest(self, case: Optional[str] = None) -> Dict[str, Any]:
        state = self.active_case(case)
        if state is None or not state.history:
            return OrderedDict([
                ("available", False),
                ("reason", self.error or "还没有采集到 [rt-metric] 行"
                           "（baseline case 不打这行；polar case 前几步之后才会出现）"),
                ("run_dir", str(self.run_dir) if self.run_dir else None),
            ])
        snaps = list(state.history)
        latest = snaps[-1]
        return OrderedDict([
            ("available", True),
            ("run_dir", str(self.run_dir)),
            ("case", state.case),
            ("kind", state.kind),
            ("rate", state.rate),
            ("steps_seen", len(snaps)),
            ("metrics", latest.to_dict()),
            ("rolling", self._rolling(snaps, self.window)),
        ])

    def history(self, case: Optional[str], limit: int) -> Dict[str, Any]:
        state = self.active_case(case)
        if state is None:
            return OrderedDict([("available", False), ("metrics", [])])
        snaps = list(state.history)[-limit:]
        return OrderedDict([
            ("available", bool(snaps)),
            ("case", state.case),
            ("count", len(snaps)),
            ("metrics", [s.to_dict() for s in snaps]),
        ])

    def case_list(self) -> List[Dict[str, Any]]:
        now = time.time()
        out = []
        for state in self.cases.values():
            last = state.history[-1] if state.history else None
            out.append(OrderedDict([
                ("case", state.case),
                ("kind", state.kind),
                ("rate", state.rate),
                ("steps_seen", len(state.history)),
                ("last_step", last.step if last else None),
                ("last_metric_ago_s",
                 None if last is None else round(now - last.ts, 1)),
                ("log_idle_s",
                 None if not state.log_mtime else round(now - state.log_mtime, 1)),
                ("log_dir", str(state.case_dir)),
            ]))
        return out

    # ---- 订阅（SSE） ----

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass


# ---------------------------------------------------------------- FastAPI

def overview_payload(store: MetricStore) -> Dict[str, Any]:
    """4 个 case 的性能比对 + 通信量端到端优化。

    直接复用 run.py 的 `_collect_case` / `_summary_table` / `_pair_compare`，
    保证 API 与命令行汇总口径一致；顺带把每条 case 的实时指标也带上。
    """
    run_dir = store.resolve_run_dir()
    if run_dir is None:
        return OrderedDict([("available", False), ("reason", "找不到 run 目录")])
    case_json = run_dir / DEFAULT_CASE_JSON
    if not case_json.exists():
        return OrderedDict([("available", False),
                            ("reason", f"{case_json} 不存在")])
    try:
        records = json.loads(case_json.read_text(encoding="utf-8")).get("cases", [])
    except (OSError, json.JSONDecodeError) as exc:
        return OrderedDict([("available", False), ("reason", f"读 case.json 失败: {exc}")])
    if not records:
        return OrderedDict([("available", False), ("reason", "case.json 里没有 case")])

    try:
        default_config = json.loads(
            qrun.QWEN14B_SPEC.default_config_path.read_text(encoding="utf-8"))
    except OSError:
        default_config = {}

    rows = [qrun._collect_case(rec, default_config) for rec in records]

    # run.py 只读字典序第一个 csv，这里用更全的 step_csv 口径把吞吐补准
    for row in rows:
        state = store.cases.get(str(row["case"]))
        if state and state.step_times:
            times = [state.step_times[s] for s in sorted(state.step_times)]
            params = {}
            for rec in records:
                if str(rec.get("case")) == str(row["case"]):
                    params = (rec.get("launch_params") or {}).get("node0") or {}
                    break

            def _int(key, fallback):
                try:
                    return int(params.get(key, fallback))
                except (TypeError, ValueError):
                    return fallback

            seq_len = _int("seq-len", default_config.get("seq-len", 256))
            per_dev = _int("per-device-batch-size", default_config.get("micro-batches", 32))
            dp_world = _int("dp-world-size", 2)
            tokens = len(times) * dp_world * per_dev * seq_len
            row["num_steps"] = len(times)
            row["avg_step_s"] = sum(times) / len(times)
            row["total_step_s"] = sum(times)
            row["throughput_tok_s"] = tokens / sum(times)

    # 离线重叠率：走本文件里主机名无关的 trace 定位
    for row, rec in zip(rows, records):
        case_dir = Path(rec.get("log_dir", ""))
        if not case_dir.is_dir():
            continue
        try:
            import overlap_interval as oi
            trace = rank0_trace_path(case_dir)
            if trace is not None:
                res = oi.analyze(trace, "last")
                if res:
                    row["offline_overlap"] = res["overlap"]
        except Exception:                       # noqa: BLE001 - 展示用，不能因此 500
            pass

    payload = OrderedDict([
        ("available", True),
        ("run_dir", str(run_dir)),
        ("cases", rows),
        ("table", qrun._summary_table(rows)),
        ("comparison", qrun._pair_compare(rows)),
    ])
    summary_file = run_dir / "summary.json"
    if summary_file.exists():
        try:
            payload["summary_json"] = json.loads(summary_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return payload


def create_app(store: MetricStore):
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import HTMLResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(_FASTAPI_HINT.format(name=exc.name)) from exc

    app = FastAPI(
        title="UDTCA Qwen14B 通信指标 API",
        description=(
            "实时返回 POLAR + bitscom 训练的 DP-PP 重叠率与通信压缩量"
            "（MB/s、GB/s），并提供 4-case 端到端通信量优化比对。"
        ),
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=False,
        allow_methods=["GET"], allow_headers=["*"],
    )
    app.state.store = store

    @app.on_event("startup")
    async def _startup() -> None:
        store._loop = asyncio.get_running_loop()   # noqa: SLF001 - 单进程内共享

        async def _poller() -> None:
            while True:
                try:
                    await asyncio.to_thread(store.refresh)
                except Exception:                  # noqa: BLE001 - 采集不许把服务打挂
                    pass
                await asyncio.sleep(store.poll_interval)

        app.state.poller = asyncio.create_task(_poller())

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        task = getattr(app.state, "poller", None)
        if task:
            task.cancel()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        links = [
            ("/api/v1/health", "服务与数据源状态"),
            ("/api/v1/cases", "本次 run 的 case 列表"),
            ("/api/v1/metrics/latest", "最新一步指标（含 MB/s / GB/s）"),
            ("/api/v1/metrics/history?limit=200", "最近 N 步指标序列"),
            ("/api/v1/metrics/stream", "SSE 实时推送"),
            ("/api/v1/overview", "4-case 性能比对 + 通信量端到端优化"),
            ("/docs", "OpenAPI 文档"),
        ]
        items = "\n".join(f'<li><a href="{u}"><code>{u}</code></a> — {d}</li>'
                          for u, d in links)
        return (f"<html><head><meta charset='utf-8'><title>UDTCA 通信指标 API</title>"
                f"</head><body><h2>UDTCA Qwen14B 通信指标 API</h2>"
                f"<p>run_dir: <code>{store.resolve_run_dir()}</code></p>"
                f"<ul>{items}</ul></body></html>")

    @app.get("/api/v1/health")
    async def health() -> Dict[str, Any]:
        run_dir = store.resolve_run_dir()
        return OrderedDict([
            ("status", "ok"),
            ("run_log_root", str(store.run_log_root)),
            ("run_dir", str(run_dir) if run_dir else None),
            ("cases_tracked", len(store.cases)),
            ("subscribers", len(store._subscribers)),   # noqa: SLF001
            ("poll_interval_s", store.poll_interval),
            ("window", store.window),
            ("error", store.error),
        ])

    @app.get("/api/v1/cases")
    async def cases() -> Dict[str, Any]:
        store.refresh()
        return OrderedDict([
            ("run_dir", str(store.run_dir) if store.run_dir else None),
            ("cases", store.case_list()),
        ])

    @app.get("/api/v1/metrics/latest")
    async def latest(case: Optional[str] = Query(None, description="case 编号，默认取正在跑的那个")):
        store.refresh()
        return store.latest(case)

    @app.get("/api/v1/metrics/history")
    async def history(case: Optional[str] = Query(None),
                      limit: int = Query(200, ge=1, le=5000)) -> Dict[str, Any]:
        store.refresh()
        return store.history(case, limit)

    @app.get("/api/v1/metrics/stream")
    async def stream(case: Optional[str] = Query(None)) -> StreamingResponse:
        q = store.subscribe()

        async def gen():
            try:
                # 先把当前状态推一次，前端连上就有东西显示
                yield f"event: snapshot\ndata: {json.dumps(store.latest(case), ensure_ascii=False)}\n\n"
                while True:
                    try:
                        payload = await asyncio.wait_for(q.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"          # 心跳，防中间层断连
                        continue
                    if case and payload.get("case") != case:
                        continue
                    yield f"event: metrics\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            finally:
                store.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.get("/api/v1/overview")
    async def overview() -> Dict[str, Any]:
        store.refresh()
        payload = overview_payload(store)
        if not payload.get("available"):
            raise HTTPException(status_code=404, detail=payload.get("reason"))
        return payload

    return app


# ---------------------------------------------------------------- demo 数据

# 数字全部对齐 run_log/run_2026-10-14_09-21-51 那次真实 4-case 跑出来的量级，
# 免得演示时算出一个比网卡还快的 MB/s（原脚本用 28ms 的步时配 313MB，是 11GB/s）。
_DEMO_SPEC = [
    # (case, kind, rate, dense/compressed 倍数, 重叠率区间, 每步秒数)
    ("0001", "polar", "2gbit", 7.76, (0.183, 0.221), 33.5),
    ("0002", "baseline", "2gbit", None, None, 142.2),
    ("0003", "polar", "1gbit", 7.74, (0.191, 0.234), 49.3),
    ("0004", "baseline", "1gbit", None, None, 286.9),
]
_DEMO_STEPS = 6
_DEMO_DENSE_BYTES = 2.4303e9
# 真实日志里 pp_ms / step_time ≈ 21.3s / 39.5s
_DEMO_PP_FRACTION = 0.54


def build_demo_run(run_root: Path, run_name: str = "run_demo_healthcheck") -> Path:
    """造一份假的 run 目录，没卡也能给平台演示接口。

    结构和真实 run 一致：case.json + 每个 case 的 train_node0.log / step_csv。
    """
    run_dir = Path(run_root) / run_name
    records = []
    for case, kind, rate, ratio, ov, speed in _DEMO_SPEC:
        case_dir = run_dir / case
        (case_dir / "step_csv").mkdir(parents=True, exist_ok=True)
        lines = []
        with open(case_dir / "step_csv" / f"demo_{case}.csv", "w",
                  newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "loss", "step_time_s", "elapsed_s", "train_mode",
                        "baseline_mode", "polar_hook", "ablation", "local_sgd_mode",
                        "local_sgd_steps", "dp_world_size", "pp_size", "tp_size"])
            for step in range(_DEMO_STEPS):
                cost = speed * (1.0 + 0.03 * ((step * 7) % 5))
                loss = 12.27 - step * 0.09
                w.writerow([step, f"{loss:.4f}", f"{cost:.4f}", f"{(step + 1) * cost:.4f}",
                            "polar" if kind == "polar" else "baseline",
                            "dense", "ef_lowmem", "", "", 0, 2, 8, 2])
                if kind == "polar":
                    pp_ms = cost * _DEMO_PP_FRACTION * 1000.0
                    dp_ms = pp_ms * 0.86
                    overlap = ov[0] + (ov[1] - ov[0]) * (step / max(1, _DEMO_STEPS - 1))
                    dense = _DEMO_DENSE_BYTES
                    comp = dense / ratio
                    lines.append(
                        f"[rt-metric] step={step} loss={loss:.4f} pp_ms={pp_ms:.2f} "
                        f"dp_ms={dp_ms:.2f} dp_pp_overlap={overlap * 100:.1f}% "
                        f"comm_dense={dense / 1e9:.4f}GB "
                        f"comm_compressed={comp / 1e6:.4f}MB "
                        f"comm_saved={(dense - comp) / 1e9:.4f}GB "
                        f"comm_reduction={ratio:.2f}x")
        (case_dir / "train_node0.log").write_text(
            "\n".join(lines + [""]), encoding="utf-8")
        records.append(OrderedDict([
            ("case", case), ("test_index", int(case)), ("kind", kind),
            ("started_at", "demo"), ("log_dir", str(case_dir)),
            ("rate", rate),
            ("config", {"index": int(case), "rate": rate, "baseline":
                        "True" if kind == "baseline" else "False"}),
            ("launch_params", {"node0": {
                "seq-len": 256, "per-device-batch-size": 32,
                "max-steps": _DEMO_STEPS,
            }}),
            ("finished_at", "demo"), ("exit_code", 0),
            ("comm_bytes", 227.452e9 if kind == "baseline" else 55.190e9),
            ("comm_gb", 227.452 if kind == "baseline" else 55.190),
        ]))
    (run_dir / DEFAULT_CASE_JSON).write_text(
        json.dumps({"cases": records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return run_dir


def start_demo_live(run_dir: Path, interval: float = 2.0) -> threading.Event:
    """每 interval 秒往 polar 的 demo 日志追加一步，让 SSE 有东西可推。"""
    stop = threading.Event()

    def _loop() -> None:
        step = _DEMO_STEPS
        while not stop.wait(interval):
            for case, kind, _rate, ratio, ov, speed in _DEMO_SPEC:
                if kind != "polar":
                    continue
                log = run_dir / case / "train_node0.log"
                pp_ms = speed * _DEMO_PP_FRACTION * 1000.0 * (1.0 + (step % 5) / 100.0)
                dp_ms = pp_ms * 0.83
                overlap = ov[0] + (ov[1] - ov[0]) * ((step % 17) / 17.0)
                dense = _DEMO_DENSE_BYTES
                comp = dense / ratio
                with open(log, "a", encoding="utf-8") as fh:
                    fh.write(
                        f"[rt-metric] step={step} loss={12.27 - step * 0.02:.4f} "
                        f"pp_ms={pp_ms:.2f} dp_ms={dp_ms:.2f} "
                        f"dp_pp_overlap={overlap * 100:.1f}% "
                        f"comm_dense={dense / 1e9:.4f}GB "
                        f"comm_compressed={comp / 1e6:.4f}MB "
                        f"comm_saved={(dense - comp) / 1e9:.4f}GB "
                        f"comm_reduction={ratio:.2f}x\n")
            step += 1

    t = threading.Thread(target=_loop, name="demo-live", daemon=True)
    t.start()
    return stop


# ---------------------------------------------------------------- 入口

def make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Qwen14B 训练实时指标 API（DP-PP 重叠率 + 通信压缩量）")
    p.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    p.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")
    p.add_argument("--run-log-root", default=str(DEFAULT_RUN_LOG_ROOT),
                   help=f"run 根目录（默认 {DEFAULT_RUN_LOG_ROOT}）")
    p.add_argument("--run-dir", default=None,
                   help="指定某次 run（默认自动取 run_log 下最新的 run_*）")
    p.add_argument("--log-file", default=None,
                   help="直连一个训练日志文件（手工起训练、没有 run 目录时用；"
                        "该文件必须是实时写的，即训练被 `| tee` 或 `stdbuf -oL` 包着）")
    p.add_argument("--poll-interval", type=float, default=1.0,
                   help="扫日志间隔秒数（默认 1.0）")
    p.add_argument("--window", type=int, default=20,
                   help="滚动平均的窗口步数（默认 20）")
    p.add_argument("--demo", action="store_true",
                   help="先用假数据造一份 run，方便没卡时演示接口")
    p.add_argument("--demo-live", action="store_true",
                   help="配合 --demo：每 2 秒追加一步，模拟训练在跑")
    p.add_argument("--print-plan", action="store_true",
                   help="只打印将要监听的地址与数据源，不起服务")
    return p


def main() -> int:
    args = make_arg_parser().parse_args()
    run_log_root = Path(args.run_log_root)
    run_dir = Path(args.run_dir) if args.run_dir else None

    if args.demo:
        run_dir = build_demo_run(run_log_root)
        print(f"[demo] 假 run 目录: {run_dir}")
        if args.demo_live:
            start_demo_live(run_dir)
            print("[demo] 每 2 秒追加一步，SSE 会有实时数据")

    store = MetricStore(run_log_root, run_dir,
                        window=args.window, poll_interval=args.poll_interval,
                        log_file=Path(args.log_file) if args.log_file else None)
    app = create_app(store)

    plan = OrderedDict([
        ("listen", f"http://{args.host}:{args.port}"),
        ("docs", f"http://{args.host}:{args.port}/docs"),
        ("run_log_root", str(run_log_root)),
        ("run_dir", str(store.resolve_run_dir())),
        ("log_file", str(store.log_file) if store.log_file else None),
        ("poll_interval_s", args.poll_interval),
        ("window", args.window),
    ])
    print("=" * 76)
    for k, v in plan.items():
        print(f"  {k:16} {v}")
    print("=" * 76)
    if args.print_plan:
        return 0

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(_FASTAPI_HINT.format(name=exc.name)) from exc
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
