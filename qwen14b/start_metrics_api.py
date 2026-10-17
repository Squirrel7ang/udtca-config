#!/usr/bin/env python3
"""一键：起实时指标 API（+ 可选地把 run.py 的 4-case 测试一起拉起来）。

给第三方测试平台的接入方式就是这一个脚本。它在**跑 node 0 的那台机器**上执行
（原来的 u62，现在这套代码要跑的话是 u13），前端只需要访问：

    GET http://<node0-ip>:8000/api/v1/metrics/latest      最新一步：重叠率 + 压缩量(MB/s, GB/s)
    GET http://<node0-ip>:8000/api/v1/metrics/stream      SSE，实时推每一步
    GET http://<node0-ip>:8000/api/v1/overview            4-case 汇总与通信量端到端优化
    GET http://<node0-ip>:8000/docs                       OpenAPI 文档（前端可以直接照着写）

两种用法：

    # 1) 训练已经在跑（比如你手动 bash 了 0_train_*.sh，或者 run.py 正在跑），
    #    只把 API 挂上去服务这次 run
    python udtca-config/qwen14b/start_metrics_api.py --run-log-root run_log

    # 2) 一条命令把 4-case 测试和 API 一起拉起来
    python udtca-config/qwen14b/start_metrics_api.py --with-training --max-steps 3

    # 3) 没有卡 / 还没开跑，先给平台演示接口长什么样
    python udtca-config/qwen14b/start_metrics_api.py --demo --demo-live

关于依赖：本脚本**不会**擅自往共享环境里装东西。缺 fastapi/uvicorn 时它会打印
要装什么，并给出两个选项——
  * 加 `--install-deps`：装到本项目内的 `udtca-config/qwen14b/.api_deps/`，
    只影响本脚本（它会把该目录塞进 sys.path），trj-test 环境一个包都不动；
  * 或者你自己 `pip install fastapi uvicorn` 到想用的环境里。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

CONFIG_DIR = Path(__file__).resolve().parent          # udtca-config/qwen14b
CONFIG_ROOT = CONFIG_DIR.parent
REPO_ROOT = CONFIG_ROOT.parent
DEPS_DIR = CONFIG_DIR / ".api_deps"
RUN_PY = CONFIG_DIR / "run.py"


def _ensure_deps(install: bool) -> None:
    """确认 fastapi/uvicorn 可用；--install-deps 时装到项目内的 .api_deps。

    .api_deps 已经装过就直接用，不必再带 --install-deps（训练环境里默认没有
    fastapi，装过一次之后每次启动都还要求 flag 很容易踩）。
    """
    if DEPS_DIR.is_dir() and str(DEPS_DIR) not in sys.path:
        sys.path.insert(0, str(DEPS_DIR))
    missing: List[str] = []
    for mod in ("fastapi", "uvicorn"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if not missing:
        return

    if not install:
        print(f"[依赖] 缺 {' / '.join(missing)}。", file=sys.stderr)
        print(f"       加 --install-deps 装到 {DEPS_DIR}（不动 trj-test 环境），",
              file=sys.stderr)
        print(f"       或者自己 pip install {' '.join(missing)}", file=sys.stderr)
        raise SystemExit(2)

    print(f"[依赖] 装 {missing} 到 {DEPS_DIR} ...")
    DEPS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "install", "--quiet",
           "--target", str(DEPS_DIR), *missing]
    subprocess.run(cmd, check=True)
    sys.path.insert(0, str(DEPS_DIR))


def _start_training(args: argparse.Namespace) -> Optional[subprocess.Popen]:
    """把 run.py 的 4-case 测试拉起来，日志同时落盘并回显。"""
    if not RUN_PY.exists():
        print(f"[训练] 找不到 {RUN_PY}，跳过", file=sys.stderr)
        return None
    cmd = [sys.executable, str(RUN_PY),
           "--run-log-root", str(args.run_log_root)]
    if args.cases:
        cmd += ["--cases", str(args.cases)]
    if args.max_steps is not None:
        cmd += ["--max-steps", str(args.max_steps)]
    if args.runtime_log:
        cmd += ["--runtime-log", args.runtime_log]
    log_dir = Path(args.training_log_dir) if args.training_log_dir else REPO_ROOT / "run_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"metrics_api_training_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    print(f"[训练] {' '.join(cmd)}")
    print(f"[训练] 输出同时写入 {log_path}")
    fh = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)

    import threading

    def _pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            fh.flush()                     # 日志本身也要实时，方便和 API 对照
            sys.stdout.write(line)
        fh.close()
        proc.wait()

    threading.Thread(target=_pump, name="train-log", daemon=True).start()
    return proc


def make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="起 UDTCA 实时指标 API（可选同时拉起 run.py 的 4-case 测试）")
    p.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    p.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")
    p.add_argument("--run-log-root", default=str(REPO_ROOT / "run_log"),
                   help="run 根目录")
    p.add_argument("--run-dir", default=None, help="只服务指定的某次 run")
    p.add_argument("--log-file", default=None,
                   help="直连一个实时写着的训练日志（手工起训练时用）")
    p.add_argument("--poll-interval", type=float, default=1.0,
                   help="扫日志间隔秒数，越小越实时（默认 1.0）")
    p.add_argument("--window", type=int, default=20, help="滚动平均窗口步数（默认 20）")

    p.add_argument("--install-deps", action="store_true",
                   help=f"缺依赖时装到 {DEPS_DIR}（不影响 trj-test 环境）")
    p.add_argument("--with-training", action="store_true",
                   help="同时把 run.py 的 4-case 测试拉起来（默认只起 API）")
    p.add_argument("--cases", default=None, help="传给 run.py 的 --cases")
    p.add_argument("--max-steps", type=int, default=None,
                   help="传给 run.py 的 --max-steps（冒烟用）")
    p.add_argument("--runtime-log", choices=["on", "off"], default=None,
                   help="传给 run.py 的 --runtime-log")
    p.add_argument("--training-log-dir", default=None,
                   help="训练 stdout 的落盘目录（默认 <repo>/run_log）")
    p.add_argument("--demo", action="store_true", help="先造一份假 run 供演示")
    p.add_argument("--demo-live", action="store_true", help="配合 --demo：每 2 秒追加一步")
    p.add_argument("--print-plan", action="store_true", help="只打印计划，不起服务")
    return p


def main() -> int:
    args = make_arg_parser().parse_args()
    _ensure_deps(args.install_deps)

    # 让 metrics_api 能 import 到 run.py / case_common
    sys.path.insert(0, str(CONFIG_DIR))
    import metrics_api as m

    training: Optional[subprocess.Popen] = None

    if args.demo:
        run_dir = m.build_demo_run(Path(args.run_log_root))
        print(f"[demo] 假 run 目录: {run_dir}")
        if args.demo_live:
            m.start_demo_live(run_dir)
            print("[demo] 每 2 秒追加一步，SSE 会有实时数据")
        args.run_dir = str(run_dir)

    if args.with_training:
        training = _start_training(args)

    store = m.MetricStore(
        Path(args.run_log_root),
        Path(args.run_dir) if args.run_dir else None,
        window=args.window,
        poll_interval=args.poll_interval,
        log_file=Path(args.log_file) if args.log_file else None,
    )

    print("=" * 78)
    print("  UDTCA Qwen14B 实时通信指标 API")
    print(f"  监听              http://{args.host}:{args.port}")
    print(f"  最新指标          http://{args.host}:{args.port}/api/v1/metrics/latest")
    print(f"  SSE 实时推送      http://{args.host}:{args.port}/api/v1/metrics/stream")
    print(f"  4-case 汇总       http://{args.host}:{args.port}/api/v1/overview")
    print(f"  OpenAPI 文档      http://{args.host}:{args.port}/docs")
    print(f"  run_log_root      {args.run_log_root}")
    print(f"  run_dir           {store.resolve_run_dir()}")
    print(f"  训练              {'已拉起 pid=' + str(training.pid) if training else '未拉起（只起 API）'}")
    print(f"  扫日志间隔        {args.poll_interval}s   （端到端延迟 ≈ 该间隔 + 一步训练时间）")
    print("=" * 78)

    if args.print_plan:
        if training is not None:
            training.terminate()
        return 0

    app = m.create_app(store)
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(m._FASTAPI_HINT.format(name=exc.name)) from exc

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        if training is not None and training.poll() is None:
            print("[训练] 收到退出信号，收掉训练进程 ...")
            training.terminate()
            try:
                training.wait(timeout=30)
            except subprocess.TimeoutExpired:
                training.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
