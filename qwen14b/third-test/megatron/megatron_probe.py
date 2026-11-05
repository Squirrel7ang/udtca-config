#!/usr/bin/env python3
"""Megatron-Core baseline 运行时探针（third-test 用）。

把 Megatron 当成 polar 之外的一条 baseline 跑时，run_docker.py 要从训练日志里读到
和 polar **同一套口径**的每步指标，否则两边没法比。polar 的指标来自
`polar-sgd/src/psgd/parallelism/polar/wrapper.py::_emit_rt_metric`：

    [rt-metric] step=3 loss=11.7841 pp_ms=1234.50 dp_ms=321.00 dp_pp_overlap=21.3%
                comm_dense=2.4303GB ...

这个文件不改 Megatron 源码，只在启动前 monkey-patch 三个位置，产出同样格式的行：

  1. `megatron.training.training.train_step`：一次 optimizer step 的边界，在这里
     开/关整步计时事件、写 step_csv、打 `Step N, Loss: x`（run_docker 的
     `_losses_from_text` 认这个格式）和 `[rt-metric]`。
  2. 传进 train_step 的 `forward_backward_func`：包一层，量 **PP 窗口**
     （一次流水线 forward+backward 的 GPU 事件跨度）。
  3. `_ParamAndGradBucketGroup.start_grad_sync/finish_grad_sync`：量 **DP 窗口**
     （第一笔 DP 集合通信入队 → 这笔通信真正完成），并累计本步 DP 负载字节。

指标定义（与 polar 对齐，另外补了一个「串行段」修正）：

  pp_ms         = PP 窗口的 GPU 事件跨度。
  dp_ms         = DP 窗口的 GPU 事件跨度。
  dp_pp_overlap = 被计算真正掩盖掉的 DP 时间 / dp_ms。

重叠率的算法：Megatron 不开 `--overlap-grad-reduce` 时，all-reduce 是在**计算流上
同步**发起的（CPU 阻塞到通信完成），它在时间线上嵌在 PP 窗口里面，直接套
|DP∩PP|/(|DP|+|PP|−|DP∩PP|) 会得到接近 100% 的假重叠。所以先把同步段从 PP 窗口里扣掉：

    pp_excl    = pp_ms − dp_sync_ms     # dp_sync_ms 只在同步路径上非零
    overlap_ms = clip(pp_excl + dp_ms − step_ms, 0, dp_ms)

串行时 pp_excl + dp − step ≈ 0；只有通信真的在别的流上与计算并行时才会 > 0，
和 polar baseline（DP 同步在 schedule.step 之后串行做，重叠率 0%）口径一致。

用法（由 run_docker.py 生成的脚本调用）：

    python megatron_probe.py --megatron-entry /data1/tangruijing/megatron-runtime/pretrain_gpt.py \
        -- --num-layers 48 ...            # `--` 之后原样交给 Megatron

环境变量：
    MEGATRON_RT_CSV_DIR   每步 step_csv 的落盘目录（不给就不写 csv）
    MEGATRON_RT_TAG       文件名后缀，默认 rank<rank>
"""

from __future__ import annotations

import csv
import os
import runpy
import sys
import time
from pathlib import Path

_CSV_DIR = os.environ.get("MEGATRON_RT_CSV_DIR", "")
_TAG = os.environ.get("MEGATRON_RT_TAG", "")


def _log(msg):
    print(f"[megatron-rt-probe] {msg}", flush=True)


class _IterState:
    """一次 train_step 的计时状态（每条 rank 自己的，不跨 rank）。"""

    __slots__ = ("step", "t0", "t1", "pp_start", "pp_end", "dp_start", "dp_end",
                 "call_start", "call_end", "dp_open", "dp_done", "dp_bytes",
                 "sync_ms", "pp_seen")

    def __init__(self, step, torch):
        self.step = step
        self.t0 = torch.cuda.Event(enable_timing=True)
        self.t1 = torch.cuda.Event(enable_timing=True)
        self.pp_start = torch.cuda.Event(enable_timing=True)
        self.pp_end = torch.cuda.Event(enable_timing=True)
        self.dp_start = torch.cuda.Event(enable_timing=True)
        self.dp_end = torch.cuda.Event(enable_timing=True)
        self.call_start = torch.cuda.Event(enable_timing=True)
        self.call_end = torch.cuda.Event(enable_timing=True)
        self.dp_open = False
        self.dp_done = False
        self.dp_bytes = 0
        self.sync_ms = 0.0
        self.pp_seen = False


class Probe:
    def __init__(self):
        self.state = None
        self.step = 0
        self.torch = None
        self._csv_path = None

    # ------------------------------------------------------------ 安装
    def install(self):
        import torch  # noqa: F401  (Corex 上必须先把 torch 加载起来)

        self.torch = torch
        from megatron.core.distributed import param_and_grad_buffer as pgb
        from megatron.training import training as mg_training

        group_cls = pgb._ParamAndGradBucketGroup
        orig_start = group_cls.start_grad_sync
        orig_finish = group_cls.finish_grad_sync
        probe = self

        def start_grad_sync(self_grp, *args, **kwargs):
            st = probe.state
            if st is not None and not st.dp_open:
                st.dp_open = True
                st.dp_start.record()
                try:
                    st.dp_bytes += sum(
                        int(b.grad_data.numel()) * int(b.grad_data.element_size())
                        for b in self_grp.buckets
                    )
                except Exception:                             # noqa: BLE001
                    pass
            if st is not None:
                st.call_start.record()
            try:
                return orig_start(self_grp, *args, **kwargs)
            finally:
                # grad_reduce_handle 为 None ⇒ 这次是同步调用，通信在这里已经做完。
                if st is not None and getattr(self_grp, "grad_reduce_handle", None) is None:
                    st.call_end.record()
                    probe._close_dp(st, sync=True)

        def finish_grad_sync(self_grp, *args, **kwargs):
            try:
                return orig_finish(self_grp, *args, **kwargs)
            finally:
                if probe.state is not None:
                    probe._close_dp(probe.state, sync=False)

        group_cls.start_grad_sync = start_grad_sync
        group_cls.finish_grad_sync = finish_grad_sync

        orig_train_step = mg_training.train_step

        def train_step(forward_step_func, data_iterator, model, optimizer,
                       opt_param_scheduler, config, forward_backward_func):
            probe.step += 1
            st = _IterState(probe.step, torch)
            probe.state = st
            st.t0.record()

            def timed_forward_backward(**kw):
                st.pp_start.record()
                st.pp_seen = True
                try:
                    return forward_backward_func(**kw)
                finally:
                    st.pp_end.record()

            try:
                out = orig_train_step(
                    forward_step_func=forward_step_func,
                    data_iterator=data_iterator,
                    model=model,
                    optimizer=optimizer,
                    opt_param_scheduler=opt_param_scheduler,
                    config=config,
                    forward_backward_func=timed_forward_backward,
                )
            finally:
                st.t1.record()
                probe._close_dp(st, sync=False)
                probe.state = None
            try:
                probe._emit(st, out)
            except Exception as exc:                          # noqa: BLE001
                _log(f"step {st.step} 指标输出失败（不影响训练）: {exc.__class__.__name__}: {exc}")
            return out

        mg_training.train_step = train_step
        _log(f"installed (csv_dir={_CSV_DIR or '-'}, tag={_TAG or '-'}, "
             f"rank={os.environ.get('RANK', '?')})")

    # ------------------------------------------------------------ 采集
    def _close_dp(self, st, *, sync):
        if not st.dp_open:
            return
        st.dp_open = False
        st.dp_done = True
        st.dp_end.record()
        if sync:
            try:
                st.sync_ms += st.call_start.elapsed_time(st.call_end)
            except Exception:                                 # noqa: BLE001
                pass

    def _emit(self, st, out):
        torch = self.torch
        torch.cuda.synchronize()

        pp_ms = st.pp_start.elapsed_time(st.pp_end) if st.pp_seen else None
        dp_ms = st.dp_start.elapsed_time(st.dp_end) if st.dp_done else None
        step_ms = st.t0.elapsed_time(st.t1)

        overlap_ratio = None
        if pp_ms is not None and dp_ms is not None and dp_ms > 0:
            pp_excl = max(0.0, pp_ms - st.sync_ms)
            overlap_ms = max(0.0, min(dp_ms, pp_excl + dp_ms - step_ms))
            overlap_ratio = overlap_ms / dp_ms

        loss = _extract_loss(out)

        parts = [f"[rt-metric] step={st.step}"]
        if loss is not None:
            parts.append(f"loss={loss:.4f}")
        parts.append(f"step_ms={step_ms:.2f}")
        if pp_ms is not None:
            parts.append(f"pp_ms={pp_ms:.2f}")
        if dp_ms is not None:
            parts.append(f"dp_ms={dp_ms:.2f}")
        if overlap_ratio is not None:
            parts.append(f"dp_pp_overlap={overlap_ratio * 100.0:.1f}%")
        parts.append(f"dp_sync_ms={st.sync_ms:.2f}")
        if st.dp_bytes:
            parts.append(f"comm_dense={st.dp_bytes / 1e9:.4f}GB")
        print(" ".join(parts), flush=True)

        # run_docker 的 _losses_from_text 只认 "Step N, Loss: x"（polar 也是这么打的）。
        # 非最后一级流水线 stage 拿不到 loss，那就不打，免得把别的值覆盖掉。
        if loss is not None:
            print(f"Step {st.step}, Loss: {loss:.4f}", flush=True)

        self._write_csv(st.step, loss, step_ms / 1000.0)

    def _write_csv(self, step, loss, step_time_s):
        if not _CSV_DIR:
            return
        if self._csv_path is None:
            try:
                d = Path(_CSV_DIR)
                d.mkdir(parents=True, exist_ok=True)
                tag = _TAG or f"rank{os.environ.get('RANK', '0')}"
                self._csv_path = d / f"megatron_{tag}.csv"
            except Exception:                                 # noqa: BLE001
                return
        try:
            new = not self._csv_path.exists()
            with open(self._csv_path, "a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["step", "loss", "step_time_s", "logged_at"])
                w.writerow([step, "" if loss is None else f"{loss:.6f}",
                            f"{step_time_s:.6f}", f"{time.time():.3f}"])
        except Exception:                                     # noqa: BLE001
            pass


def _extract_loss(out):
    if not isinstance(out, tuple) or not out:
        return None
    d = out[0]
    if not isinstance(d, dict):
        return None
    v = d.get("lm loss")
    if v is None:
        return None
    try:
        return float(v.item() if hasattr(v, "item") else v)
    except Exception:                                         # noqa: BLE001
        return None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("用法: megatron_probe.py --megatron-entry <pretrain_gpt.py> -- <Megatron 参数>")
    cut = argv.index("--")
    head, tail = argv[:cut], argv[cut + 1:]
    entry = None
    for i, tok in enumerate(head):
        if tok == "--megatron-entry" and i + 1 < len(head):
            entry = head[i + 1]
    if not entry or not Path(entry).is_file():
        raise SystemExit(f"找不到 Megatron 入口: {entry!r}")

    Probe().install()
    sys.argv = [entry] + tail
    runpy.run_path(entry, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
