#!/usr/bin/env python3
"""把 HuggingFaceFW/fineweb 的一部分 parquet 下到本地。

按仓库里的顺序挨个下（`data/CC-MAIN-2013-20/...` 开头），累计到指定大小就停。
训练侧是 `load_dataset(..., streaming=True)` 从流的开头顺序读，所以按仓库顺序下
下来的这份正好是训练会用到的那段前缀。

用法：
    python download_fineweb.py --dry-run          # 只看看会下哪些、多大，不下载
    python download_fineweb.py                    # 真下，默认 20 GiB
    python download_fineweb.py --target-gb 5      # 想小一点

已经下好的文件会跳过（按远端大小判断），中断了直接重跑即可续上。
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# 国内走镜像；没有的话退回官方（会很慢）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

REPO_ID = "HuggingFaceFW/fineweb"
# 默认放在 udtca 目录里，避免动仓库外面的东西
DEFAULT_DEST = Path("/data1/tangruijing/udtca/fineweb_data")
# 只下这个前缀下的文件。`data/` = 完整数据集；也可以换成 `sample/10BT/` 等
DEFAULT_SUBDIR = "data/"
GIB = 1024 ** 3


def human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0


def parse_args():
    p = argparse.ArgumentParser(description="下载 fineweb 的一部分到本地")
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                   help=f"下载到哪个目录（默认 {DEFAULT_DEST}）")
    p.add_argument("--target-gb", type=float, default=20.0,
                   help="下到大约多少 GiB 就停（默认 20）")
    p.add_argument("--subdir", default=DEFAULT_SUBDIR,
                   help=f"只下这个前缀下的文件（默认 {DEFAULT_SUBDIR}）")
    p.add_argument("--dry-run", action="store_true",
                   help="只列出会下哪些文件，不真的下载")
    p.add_argument("--margin-gb", type=float, default=5.0,
                   help="磁盘剩余空间至少要比目标多这么多才开下（默认 5）")
    return p.parse_args()


def check_disk(dest: Path, need_bytes: int, margin_bytes: int) -> None:
    """磁盘不够就直接退出，别下一半爆了。"""
    probe = dest
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    print(f"目标目录所在分区: {probe}  已用 {human(shutil.disk_usage(probe).used)}  "
          f"剩余 {human(free)}")
    print(f"本次计划下载: 约 {human(need_bytes)}（另留 {human(margin_bytes)} 余量）")
    if free < need_bytes + margin_bytes:
        print(f"\n[ERROR] 空间不够：剩余 {human(free)} < "
              f"需要 {human(need_bytes + margin_bytes)}")
        sys.exit(1)
    print("空间充足。\n")


def main() -> None:
    args = parse_args()
    target_bytes = int(args.target_gb * GIB)
    dest: Path = args.dest.expanduser().resolve()

    print(f"仓库      : {REPO_ID}")
    print(f"HF_ENDPOINT: {os.environ['HF_ENDPOINT']}")
    print(f"下载前缀  : {args.subdir}")
    print(f"目标大小  : {human(target_bytes)}")
    print(f"目标目录  : {dest}")
    print()

    check_disk(dest, target_bytes, int(args.margin_gb * GIB))

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    print("正在获取仓库文件列表...")
    entries = [
        e for e in api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True)
        if getattr(e, "size", None) is not None
        and e.path.startswith(args.subdir)
        and e.path.endswith(".parquet")
    ]
    # 仓库返回的顺序就是它的自然顺序（CC-MAIN-2013-20 开始），按这个下
    entries.sort(key=lambda e: e.path)
    total_available = sum(e.size for e in entries)
    print(f"该前缀下共 {len(entries)} 个 parquet，合计 {human(total_available)}")
    print()

    # 挑出要下的文件。
    # 注意是「加上这个文件会超过目标就停」，不是「够了才停」——parquet 单个就 2GiB 左右，
    # 后者会多下一个文件，结果比目标大 10%。
    planned, acc = [], 0
    for e in entries:
        local = dest / e.path
        if local.exists() and local.stat().st_size == e.size:
            continue  # 已下好
        if acc + e.size > target_bytes and acc > 0:
            break
        planned.append(e)
        acc += e.size

    already = sum(1 for e in entries
                  if (dest / e.path).exists() and (dest / e.path).stat().st_size == e.size)
    print(f"本地已有: {already} 个文件")
    print(f"本次要下: {len(planned)} 个文件，合计 {human(acc)}")
    if planned:
        print(f"  第一个: {planned[0].path}")
        print(f"  最后一个: {planned[-1].path}")

    if args.dry_run:
        print("\n--dry-run，不下载。")
        return
    if not planned:
        print("\n已经够了，不用下。")
        return

    print()
    dest.mkdir(parents=True, exist_ok=True)
    got = 0
    for i, e in enumerate(planned, 1):
        target = dest / e.path
        try:
            hf_hub_download(
                REPO_ID, e.path,
                repo_type="dataset",
                local_dir=str(dest),
            )
        except Exception as exc:
            print(f"  [{i}/{len(planned)}] 失败 {e.path}: {exc}")
            print("  中断了就直接重跑本脚本，已下好的会跳过。")
            sys.exit(1)
        got += e.size
        print(f"  [{i}/{len(planned)}] {e.path}  {human(e.size)}  "
              f"累计 {human(got)}")

    print(f"\n完成：{len(planned)} 个文件，{human(got)}")
    print(f"位置：{dest}")
    print(f"当前该目录共占用：{human(sum(f.stat().st_size for f in dest.rglob('*') if f.is_file()))}")


if __name__ == "__main__":
    main()
