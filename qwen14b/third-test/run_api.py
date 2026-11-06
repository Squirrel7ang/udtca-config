#!/usr/bin/env python3
"""third-test：一键起跨机训练（run_docker.py）+ 实时指标 API（与 metrics_api.py 同接口）。

给前端的接口与 `udtca-config/qwen14b/start_metrics_api.py` 起的那套**完全相同**
（字段、口径、SSE 事件名都一样）：

    GET /api/v1/health                      服务与数据源状态
    GET /api/v1/cases                       本次 run 的 case 列表
    GET /api/v1/metrics/latest?case=        最新一步：DP-PP 重叠率 + 通信压缩量(MB/s, GB/s)
    GET /api/v1/metrics/history?case=&limit= 最近 N 步
    GET /api/v1/metrics/stream?case=        SSE 实时推送（推荐）
    GET /api/v1/overview                    汇总对比
    GET /docs                               OpenAPI 文档

两件事：

1. **启动训练**：把同目录的 run_docker.py 作为子进程拉起（默认只跑
   `--modes polar`；要 polar vs baseline 传 `--modes polar,baseline` 或直接跑 run_docker.py，
   要看 Megatron baseline 用 `--docker-script .../run_megatron.py --modes polar,megatron`）。它的 stdout 会实时回显，同时由 run_docker 自己
   tee 到 `<run_dir>/run_docker.stdout.log`。
2. **实时输出重叠率 / 压缩量**：run_docker 的产物是
   `run_log/xnode_<ts>/<mode>/node{0,1}/train.log`，而 metrics_api 只认
   `run_<ts>/<case>/train_node0.log`。这里不复制日志、也不重写解析逻辑：建一个
   兼容目录 `run_log/run_third_<ts>/<case>/`（symlink 指向 run_docker 的产物，
   再写一个 case.json），然后原样复用 `metrics_api.MetricStore` /
   `metrics_api.create_app`。所以接口定义只有一处，不会漂。

用法（powerone1 宿主机上执行，容器里没有 docker）：

    # 起训练（默认只跑 polar）+ API
    python3 udtca-config/qwen14b/third-test/run_api.py --max-steps 5

    # 只跑训练里的某几个 mode、换端口
    python3 udtca-config/qwen14b/third-test/run_api.py --modes polar --max-steps 30 --port 8001

    # 训练已经在跑：只把 API 挂到某次 xnode run 上
    python3 udtca-config/qwen14b/third-test/run_api.py --no-training \
        --run-dir run_log/xnode_20261102_030500_9887c182

    传额外参数给 run_docker.py：在 `--` 后面写，原样透传
    python3 udtca-config/qwen14b/third-test/run_api.py --max-steps 3 -- --warmup-s 40

依赖：只有起 HTTP 服务需要 fastapi + uvicorn；缺了加 `--install-deps`（装到
`udtca-config/qwen14b/.api_deps/`，和 start_metrics_api.py 共用同一个目录）。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

THIRD_DIR = Path(__file__).resolve().parent          # udtca-config/qwen14b/third-test
CONFIG_DIR = THIRD_DIR.parent                        # udtca-config/qwen14b
REPO_ROOT = CONFIG_DIR.parent.parent                 # udtca
DEFAULT_DOCKER = THIRD_DIR / "run_docker.py"
DEPS_DIR = CONFIG_DIR / ".api_deps"

# metrics_api 在 qwen14b 目录下，import 它要用同一个 sys.path 设置
if str(CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIG_DIR))

# 注意：mode "polar" 原来的名字是 "mixed"（POLAR + 2bit 量化），2026-11-02 起统一改名，
#       旧的 --modes mixed 已不再接受；这里 kind 仍叫 "polar"。
# 各 mode 在 API 里叫什么 kind（前端只拿来做区分/上色）
MODE_KIND = {
    "baseline": "baseline",
    "polar_dense": "polar",
    "bitscom": "bitscom",
    "polar": "polar",
}


# ---------------------------------------------------------------- 依赖

def _ensure_deps(install: bool) -> None:
    """确认 fastapi/uvicorn 可用；--install-deps 时装到共享的 .api_deps。"""
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
        print(f"       加 --install-deps 装到 {DEPS_DIR}，"
              f"或者自己 pip install {' '.join(missing)}", file=sys.stderr)
        raise SystemExit(2)
    print(f"[依赖] 装 {missing} 到 {DEPS_DIR} ...")
    DEPS_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    "--target", str(DEPS_DIR), *missing], check=True)
    sys.path.insert(0, str(DEPS_DIR))


# ---------------------------------------------------------------- 起训练

def start_training(args: argparse.Namespace) -> subprocess.Popen:
    """拉起 run_docker.py。stdout 直接继承（实时回显 + 它自己 tee 到 run_dir）。"""
    script = Path(args.docker_script)
    if not script.is_absolute():
        script = (REPO_ROOT / script).resolve()
    if not script.exists():
        raise SystemExit(f"[训练] 找不到 {script}")
    cmd = [sys.executable, str(script),
           "--modes", args.modes,
           "--repo", args.repo]
    if args.max_steps is not None:
        cmd += ["--max-steps", str(args.max_steps)]
    cmd += list(args.docker_args or [])
    print(f"[训练] {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def wait_for_new_run_dir(root: Path, before: set, proc: subprocess.Popen,
                         timeout_s: float = 180.0) -> Optional[Path]:
    """等 run_docker 建出新 run 目录；它提前退出（预检不过等）就返回 None。"""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        new = sorted(p for p in root.glob("xnode_*")
                     if p.is_dir() and p.name not in before)
        if new:
            return new[-1]
        if proc.poll() is not None:
            return None
        time.sleep(1.0)
    return None


# ---------------------------------------------------------------- 布局桥接

class RunBridge:
    """把 `xnode_<ts>/<mode>/node{0,1}` 映射成 metrics_api 认识的 case 目录。

    metrics_api 只认
        run_<ts>/<case>/train_node0.log
        run_<ts>/<case>/train_node1.log
        run_<ts>/<case>/step_csv/
        run_<ts>/case.json   {"cases": [{case, kind, rate, log_dir, ...}]}
    这里用 symlink 让 `<case>/train_node0.log` 直接指向 run_docker 写的
    `xnode_<ts>/<mode>/node0/train.log`（不复制、不占额外空间，读取端照常 tail）。
    """

    def __init__(self, xnode_dir: Path, compat_dir: Path, modes: List[str],
                 launch_params: Dict[str, Any]):
        self.xnode_dir = xnode_dir
        self.compat_dir = compat_dir
        self.modes = modes
        self.launch_params = launch_params
        self.records: Dict[str, Dict[str, Any]] = {}
        self._written = None
        self.compat_dir.mkdir(parents=True, exist_ok=True)

    # ---- 建链接 ----
    def _make_case(self, case_dir: Path, mode_dir: Path, case: str) -> None:
        case_dir.mkdir(parents=True, exist_ok=True)
        links = {
            "train_node0.log": mode_dir / "node0" / "train.log",
            "train_node1.log": mode_dir / "node1" / "train.log",
            "step_csv": mode_dir / "node0" / "step_csv",
        }
        for name, target in links.items():
            link = case_dir / name
            if link.exists() or link.is_symlink():
                continue
            try:
                os.symlink(target, link)   # 允许悬空：目标稍后才出现
            except OSError as exc:
                print(f"[桥接] 建链接失败 {link} -> {target}: {exc}", file=sys.stderr)

    # ---- 幂等同步 ----
    def sync(self) -> None:
        changed = False
        for idx, mode in enumerate(self.modes):
            mode_dir = self.xnode_dir / mode
            if not mode_dir.is_dir():
                continue
            case = f"{idx + 1:04d}"
            case_dir = self.compat_dir / case
            if case not in self.records:
                self._make_case(case_dir, mode_dir, case)
                self.records[case] = {
                    "case": case,
                    "kind": MODE_KIND.get(mode, mode),
                    "rate": "xnode",
                    "mode": mode,
                    "log_dir": str(case_dir),
                    "started_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    "finished_at": None,
                    "launch_params": {"node0": dict(self.launch_params)},
                }
                changed = True
            rec = self.records[case]
            # mode 收尾标记：run_docker 的脚本会写 node0/EXIT_CODE
            if rec["finished_at"] is None and (mode_dir / "node0" / "EXIT_CODE").exists():
                rec["finished_at"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                changed = True

        if changed:
            self._write_case_json()

    def _write_case_json(self) -> None:
        payload = {"cases": [self.records[k] for k in sorted(self.records)]}
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if text == self._written:
            return
        tmp = self.compat_dir / ".case.json.tmp"
        tmp.write_text(text + "\n", encoding="utf-8")
        os.replace(tmp, self.compat_dir / "case.json")   # 原子替换，读端不会读到半个文件
        self._written = text


def make_store(metrics_api, run_log_root: Path, compat_dir: Path,
               bridge: RunBridge, window: int, poll_interval: float):
    """MetricStore 子类：每轮 refresh 之前先同步一次桥接目录。"""

    class XnodeMetricStore(metrics_api.MetricStore):
        def refresh(self) -> None:          # noqa: D401 - 同名覆盖
            try:
                bridge.sync()
            except Exception as exc:        # noqa: BLE001 - 桥接出错不许把服务打挂
                print(f"[桥接] sync 失败: {exc}", file=sys.stderr)
            super().refresh()

    return XnodeMetricStore(run_log_root, compat_dir, window=window,
                            poll_interval=poll_interval)


# ---------------------------------------------------------------- 入口

def make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="起跨机训练（run_docker.py）+ 实时指标 API（与 metrics_api.py 同接口）",
        epilog="必须在 powerone1 宿主机上执行（容器里没有 docker）。")
    p.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    p.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")
    p.add_argument("--run-log-root", default=str(REPO_ROOT / "run_log"),
                   help="run 根目录（默认 <repo>/run_log，要和 run_docker 的 --repo 对上）")
    p.add_argument("--run-dir", default=None,
                   help="只服务指定的那次 xnode run（隐含 --no-training）")
    p.add_argument("--poll-interval", type=float, default=1.0,
                   help="扫日志间隔秒数（默认 1.0）")
    p.add_argument("--window", type=int, default=20, help="滚动平均窗口步数（默认 20）")

    p.add_argument("--no-training", action="store_true",
                   help="不起训练，只把 API 挂到已有的 xnode run 上")
    p.add_argument("--modes", default="polar",
                   help="传给 run_docker/run_megatron 的 --modes（默认只跑 polar；"
                        "要对比 polar,baseline 或 polar,megatron）")
    p.add_argument("--max-steps", type=int, default=None,
                   help="传给 run_docker/run_megatron 的 --max-steps（不写则用对应脚本的默认值："
                        "run_docker 20、run_megatron 10）")
    p.add_argument("--repo", default=str(REPO_ROOT),
                   help="两台机器上的仓库根路径（传给 run_docker）")
    p.add_argument("--docker-script", default=str(DEFAULT_DOCKER),
                   help="被拉起的入口脚本位置（默认同目录 run_docker.py；"
                        "要看 Megatron baseline 就指到 run_megatron.py）")
    p.add_argument("--install-deps", action="store_true",
                   help=f"缺依赖时装到 {DEPS_DIR}")
    p.add_argument("docker_args", nargs=argparse.REMAINDER,
                   help="`--` 之后的内容原样传给 run_docker.py")
    return p


def _resolve_xnode_dir(args: argparse.Namespace, root: Path) -> Optional[Path]:
    if args.run_dir:
        p = Path(args.run_dir)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return p if p.is_dir() else None
    runs = sorted(p for p in root.glob("xnode_*") if p.is_dir())
    return runs[-1] if runs else None


def main() -> int:
    args = make_arg_parser().parse_args()
    if args.docker_args and args.docker_args[0] == "--":
        args.docker_args = args.docker_args[1:]
    if args.run_dir:
        args.no_training = True

    _ensure_deps(args.install_deps)
    import metrics_api as m

    root = Path(args.run_log_root)
    root.mkdir(parents=True, exist_ok=True)
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    training: Optional[subprocess.Popen] = None
    if args.no_training:
        xnode_dir = _resolve_xnode_dir(args, root)
        if xnode_dir is None:
            print(f"[训练] 没找到 xnode run（--run-dir 未给且 {root} 下没有 xnode_*）",
                  file=sys.stderr)
            return 2
    else:
        before = {p.name for p in root.glob("xnode_*")}
        training = start_training(args)
        print("[训练] 等 run_docker 建出 run 目录 …")
        xnode_dir = wait_for_new_run_dir(root, before, training)
        if xnode_dir is None:
            rc = training.poll()
            print(f"[训练] 没等到 run 目录（run_docker 返回 {rc}）；"
                  "看它上面的输出/预检报告", file=sys.stderr)
            return 2
    print(f"[训练] xnode run 目录: {xnode_dir}")

    compat_dir = root / f"run_third_{xnode_dir.name.removeprefix('xnode_')}"
    bridge = RunBridge(xnode_dir, compat_dir, modes,
                       launch_params={"seq-len": 256, "per-device-batch-size": 16})
    bridge.sync()
    store = make_store(m, root, compat_dir, bridge,
                       window=args.window, poll_interval=args.poll_interval)

    print("=" * 78)
    print("  UDTCA Qwen14B 跨机实时通信指标 API（third-test）")
    print(f"  监听              http://{args.host}:{args.port}")
    print(f"  最新指标          http://{args.host}:{args.port}/api/v1/metrics/latest")
    print(f"  SSE 实时推送      http://{args.host}:{args.port}/api/v1/metrics/stream")
    print(f"  case 列表         http://{args.host}:{args.port}/api/v1/cases")
    print(f"  OpenAPI 文档      http://{args.host}:{args.port}/docs")
    print(f"  xnode run_dir     {xnode_dir}")
    print(f"  兼容 run_dir      {compat_dir}（symlink -> xnode 产物，不复制日志）")
    print(f"  case 映射         " + ", ".join(
        f"{i + 1:04d}={mode}" for i, mode in enumerate(modes)))
    print(f"  训练              " + (f"已拉起 pid={training.pid}" if training
                                     else "未拉起（--no-training）"))
    print("=" * 78)

    app = m.create_app(store)
    try:
        import uvicorn
    except ImportError as exc:      # pragma: no cover
        raise SystemExit(m._FASTAPI_HINT.format(name=exc.name)) from exc

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        # 只收编排进程；它自己的 SIGINT 处理会走 cleanup_nodes 收训练。
        if training is not None and training.poll() is None:
            print("[训练] API 退出，收掉 run_docker …")
            training.terminate()
            try:
                training.wait(timeout=30)
            except subprocess.TimeoutExpired:
                training.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
