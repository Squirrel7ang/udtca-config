#!/usr/bin/env python3
"""把一次实验写下的深层日志目录拍平到 case 目录下。

训练代码（polar-sgd 的 wrapper）把 tensorboard 和 profiler trace 写到相对路径

    ./log/{using_polar}/{dataset_config}/{optimizer}/{comm_timing}
         /{datetime}-{dp}-{pp}-{tp}/{rank}/tb_scalars
    ./log/{using_polar}/{dataset_config}/{optimizer}/{comm_timing}
         /{datetime}-{dp}-{pp}-{tp}/{rank}/tb_trace

跑训练时 cwd 被设成 case 目录，所以这一整棵树都落在 case 目录里面，但中间那些
层级对复盘没有意义。这里把它们收拢成

    <case>/tb_scalars
    <case>/tb_trace
    <case>/step_csv

再把剩下的空目录删掉。两个节点的日志会被合并到同一个 case 目录，事件文件名里
自带 hostname 和 pid，不会互相覆盖。

用法：
    python collect_case_logs.py <case_dir>
"""

import shutil
import sys
from pathlib import Path

# 需要收拢的叶子目录名
COLLECT_DIRS = ("tb_scalars", "tb_trace", "step_csv")


def collect(case_dir) -> dict:
    """把 case_dir 下嵌套的日志目录收拢到顶层，返回每个目录移动的文件数。"""
    case_dir = Path(case_dir).resolve()
    if not case_dir.is_dir():
        raise SystemExit(f"case 目录不存在: {case_dir}")

    moved = {name: 0 for name in COLLECT_DIRS}

    for name in COLLECT_DIRS:
        target = case_dir / name
        # 注意：收拢之后 target 自身也会被 rglob 命中，所以要跳过
        for src in sorted(case_dir.rglob(name)):
            if src == target or not src.is_dir():
                continue
            for entry in sorted(src.iterdir()):
                if not entry.is_file():
                    continue
                target.mkdir(parents=True, exist_ok=True)
                dest = target / entry.name
                if dest.exists():
                    # 理论上不会撞名（事件文件名带 hostname/pid），留个兜底
                    dest = target / f"{src.parent.parent.name}-{entry.name}"
                shutil.move(str(entry), str(dest))
                moved[name] += 1

    # 自底向上删空目录；rmdir 对非空目录会失败，正好保证不会误删
    for path in sorted(case_dir.rglob("*"), key=lambda p: -len(p.parts)):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass

    # 即使没收到东西也把标准目录建出来，方便后续脚本直接读取
    for name in COLLECT_DIRS:
        (case_dir / name).mkdir(exist_ok=True)

    return moved


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__.strip())
    moved = collect(sys.argv[1])
    summary = ", ".join(f"{name}={count}" for name, count in moved.items())
    print(f"[collect_case_logs] {sys.argv[1]}: {summary}")


if __name__ == "__main__":
    main()
