#!/usr/bin/env python3
"""把这次的实验数据整理成一份可以直接交出去的目录树，然后打包成 zip。

输出结构（默认 /data1/tangruijing/comm-opt-data）：

    comm-opt-data/
      README.txt                       结构说明
      baseline/                        1. 支撑指标数据（不考虑通信调度优化）
        dataset/                       训练测试数据（3 个 parquet，polar/baseline 各一份）
        config/<rate>/                 配置数据（case 记录 + launch.sh + 训练/限速脚本）
        run_logs/<rate>/               运行日志（train_nodeN.log 原样）
        run_data/<rate>/               运行数据（从运行日志里洗出来的，只留 loss 相关）
      polar/                           2. 分布式训练通信优化
        dataset/ config/ run_logs/ run_data/
      performance/                     3. 模型训练性能对比
        performance_compare.txt / .csv

两条硬规则：
  * **只做复制 / 新建，绝不移动、删除或修改任何源文件**（源目录只读）
  * 增量：目标目录下已有、且大小一致的文件会跳过，重复跑不会重新拷一遍。
    想从零重建用 --clean（只会清掉 comm-opt-data 自己）

用法：
    python build_comm_opt_data.py --dry-run     # 只打印计划
    python build_comm_opt_data.py               # 真做
    python build_comm_opt_data.py --no-zip      # 不打包
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

DEFAULT_RUNTIME_LOG = REPO_ROOT / "runtime_log"
DEFAULT_DATASET_SRC = REPO_ROOT / "fineweb_data"
DEFAULT_DEST = Path("/data1/tangruijing/comm-opt-data")
DEFAULT_ZIP = Path("/data1/tangruijing/comm-opt-data.zip")

# 数据集取前 5 个文件（按文件名排序）：每个约 2.0 GiB，polar / baseline 各一份，
# 合计约 20 GiB —— 交付要求的数据量主要靠这部分撑起来。
DATASET_FILES = [
    "000_00000.parquet", "000_00001.parquet", "000_00002.parquet",
    "000_00003.parquet", "000_00004.parquet",
]

CATEGORY_DIR = {"baseline": "baseline", "polar": "polar"}

# 从运行日志里洗出「运行数据」时，只提取匹配到的东西，不做整行保留——
# 日志里 tqdm 进度条和 [stage_mem] 之类的 debug 会挤在同一个物理行，
# 整行保留会把 debug 一起带进来。
EXTRACT_PATTERNS = [
    # 训练侧显式打印的：「Step 10, Loss: 10.2625」
    re.compile(r"Step\s+\d+,\s*Loss:\s*[\d.eE+-]+"),
    # tqdm 进度：「10/30 [04:26<07:43, 23.16s/it, loss=10.2625]」
    re.compile(r"\d+/\d+\s*\[[^\]]*loss=[^\]]*\]"),
]


# ---------------------------------------------------------------- 工具


def log(msg: str) -> None:
    print(msg, flush=True)


def copy_file(src: Path, dst: Path, dry: bool) -> bool:
    """只复制。源文件不存在、或目标已有同样大小的文件就跳过（增量）。

    返回是否真的复制了。
    """
    if not src.exists():
        log(f"    [跳过] 源文件不存在: {src}")
        return False
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return False  # 已经在目标里了
    if dry:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_tree(src: Path, dst: Path, dry: bool, patterns=None) -> int:
    """递归复制目录（可只挑匹配的文件）。返回实际复制的文件数。"""
    if not src.is_dir():
        return 0
    n = 0
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        if patterns and not any(f.match(p) for p in patterns):
            continue
        if copy_file(f, dst / f.relative_to(src), dry):
            n += 1
    return n


def load_cases(runtime_log: Path) -> list:
    p = runtime_log / "case.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return sorted(data.get("cases", []), key=lambda c: str(c.get("case")))


# ---------------------------------------------------------------- 运行数据清洗


def clean_log(text: str) -> list:
    """从运行日志里抽出只跟 loss / 进度有关的行。

    日志里 tqdm 的进度条是用 \\r 刷新的，整条进度会被拼在一行里，
    所以先按 \\r 和 \\n 都切开，再逐段挑。
    """
    out = []
    for seg in re.split(r"[\r\n]+", text):
        if not seg.strip():
            continue
        for pattern in EXTRACT_PATTERNS:
            for m in pattern.finditer(seg):
                out.append(m.group(0).strip())
    # 去掉完全重复的相邻行
    dedup = []
    for line in out:
        if not dedup or dedup[-1] != line:
            dedup.append(line)
    return dedup


def build_run_data(src_logs: dict, dst: Path, dry: bool) -> None:
    """src_logs: {标签: 日志路径}；把它们洗干净后写到 dst。"""
    if dry:
        return
    dst.mkdir(parents=True, exist_ok=True)
    for label, path in src_logs.items():
        if not path.exists():
            continue
        lines = clean_log(path.read_text(encoding="utf-8", errors="replace"))
        header = (
            f"# 运行数据（从 {path.name} 清洗得到）\n"
            f"# 只保留 loss / 训练进度相关行；加载日志、warning、debug 信息已去掉\n"
            f"# 原始文件请见同级的 run_logs/\n"
            f"# 行数: {len(lines)}\n"
        )
        (dst / f"{label}.cleaned.txt").write_text(
            header + "\n".join(lines) + "\n", encoding="utf-8")
        log(f"      {label}.cleaned.txt  ({len(lines)} 行)")


# ---------------------------------------------------------------- 主要流程


def build_case(case: dict, runtime_log: Path, dest: Path, dry: bool) -> None:
    """整理一个 case 的 配置数据 / 运行日志 / 运行数据。"""
    case_id = str(case["case"])
    kind = case.get("kind", "?")
    rate = case.get("rate", "?")
    cat = CATEGORY_DIR.get(kind, kind)
    src = runtime_log / case_id
    if not src.is_dir():
        log(f"  [跳过] {case_id}: 找不到目录 {src}")
        return

    log(f"  --- case {case_id}  {kind}/{rate} ---")

    # 1) 配置数据：case 记录 + launch.sh + 训练/限速脚本
    cfg_dst = dest / cat / "config" / rate
    if not dry:
        cfg_dst.mkdir(parents=True, exist_ok=True)
        # 单独把这个 case 的记录写成一个 json（源 case.json 是合集，只读不改）
        (cfg_dst / f"case_{case_id}.json").write_text(
            json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    copy_file(src / "launch.sh", cfg_dst / "launch.sh", dry)
    for f in sorted(src.glob("*_train_*.sh")):
        copy_file(f, cfg_dst / f.name, dry)
    for f in sorted(src.glob("traffic_control_*.sh")):
        copy_file(f, cfg_dst / f.name, dry)
    log(f"    配置数据 -> {cfg_dst.relative_to(dest)}")

    # 2) 运行日志：原样复制
    logs_dst = dest / cat / "run_logs" / rate
    for f in sorted(src.glob("train_node*.log")):
        copy_file(f, logs_dst / f.name, dry)
    log(f"    运行日志 -> {logs_dst.relative_to(dest)}")

    # 3) 运行数据：洗净后新建
    data_dst = dest / cat / "run_data" / rate
    build_run_data(
        {f.stem: f for f in sorted(src.glob("train_node*.log"))},
        data_dst, dry,
    )
    log(f"    运行数据 -> {data_dst.relative_to(dest)}")

    # 4) 运行产物：训练侧自己写出来的原始输出，整份带过去
    #    tb_trace 是 profiler trace（最大头），tb_scalars 是 tensorboard 事件，
    #    step_csv 是每步 loss/耗时，debug_logs 是调试信息
    art_dst = dest / cat / "run_artifacts" / rate
    for sub in ("tb_trace", "tb_scalars", "step_csv", "debug_logs"):
        n = copy_tree(src / sub, art_dst / sub, dry)
        if n:
            log(f"    运行产物 -> {(art_dst / sub).relative_to(dest)}  ({n} 个文件)")


def build_dataset(dataset_src: Path, dest: Path, dry: bool) -> None:
    """数据集：前 3 个 parquet，polar / baseline 各复制一份。"""
    sub = "data/CC-MAIN-2013-20"
    src_dir = dataset_src / sub
    if not src_dir.is_dir():
        log(f"  [警告] 找不到数据集目录 {src_dir}，跳过")
        return
    for cat in ("baseline", "polar"):
        dst_dir = dest / cat / "dataset" / "CC-MAIN-2013-20"
        copied = sum(1 for name in DATASET_FILES
                     if copy_file(src_dir / name, dst_dir / name, dry))
        total = len(list(dst_dir.glob("*.parquet"))) if dst_dir.exists() else 0
        log(f"  数据集 -> {dst_dir.relative_to(dest)}  "
            f"(本次新增 {copied} 个，目录内共 {total} 个)")


def build_performance(runtime_log: Path, dest: Path, dry: bool) -> None:
    """性能对比：跑一遍 trace_processor，把输出存下来。"""
    out_dir = dest / "performance"
    if dry:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    script = HERE / "trace_processor.py"
    cmd = [sys.executable, str(script), "--runtime-log", str(runtime_log),
           "--csv", str(out_dir / "performance_compare.csv")]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE))
    text = proc.stdout + (("\n[stderr]\n" + proc.stderr) if proc.stderr.strip() else "")
    (out_dir / "performance_compare.txt").write_text(text, encoding="utf-8")
    log(f"  性能对比 -> {out_dir.relative_to(dest)}")
    if proc.returncode != 0:
        log(f"    [警告] trace_processor 退出码 {proc.returncode}")


def build_readme(dest: Path, cases: list, dry: bool) -> None:
    if dry:
        return
    lines = [
        "分布式训练通信优化（POLAR + bitscom）实验数据",
        "=" * 52,
        "",
        "目录结构：",
        "",
        "  baseline/    1. 支撑指标数据（不考虑通信调度优化）",
        "  polar/       2. 分布式训练通信优化（POLAR + bitscom）",
        "",
        "  两者各有四份：",
        "    dataset/          训练测试数据（fineweb 前 5 个 parquet，约 10 GiB，两边内容相同）",
        "    config/<rate>/    配置数据：case 记录、launch.sh、训练脚本、限速脚本",
        "    run_logs/<rate>/  运行日志：两个节点 torchrun 的原始输出",
        "    run_data/<rate>/  运行数据：从运行日志里洗出来的，只留 loss/进度相关行",
        "    run_artifacts/<rate>/  运行产物：训练侧写出来的原始输出",
        "        tb_trace/     profiler trace（这部分最大）",
        "        tb_scalars/   tensorboard 事件文件",
        "        step_csv/     每步的 loss 与耗时 CSV",
        "        debug_logs/   调试信息（只有部分 case 有）",
        "",
        "  performance/  3. 模型训练性能对比（trace_processor.py 的输出）",
        "    performance_compare.txt   汇总表",
        "    performance_compare.csv   同样内容的 CSV",
        "",
        "本次包含的 case：",
    ]
    for c in cases:
        lines.append(
            f"  {c['case']}  {c.get('kind'):<9} rate={c.get('rate'):<7} "
            f"bit-width={c.get('config', {}).get('bit-width')}"
        )
    lines += [
        "",
        "说明：",
        "  - 每个 case 一轮 = 30 步（max-steps=30），per-device-batch-size=32，seq-len=256",
        "  - 网速用 tc 限在 ens1f0 上，只在 case 运行期间生效",
        "  - 运行日志的 train_node0.log 来自 u62、train_node1.log 来自 u210",
        "  - step_csv 里每步有 2 行重复，是训练侧已知的写入问题；",
        "    performance_compare 里已按 step 去重",
        "",
    ]
    (dest / "README.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"  README.txt")


def make_zip(dest: Path, zip_path: Path, dry: bool) -> None:
    if dry:
        return
    if zip_path.exists():
        zip_path.unlink()
    base = dest.parent
    n = 0
    # parquet 本身就是压缩格式，再 deflate 几乎压不动、只会慢得离谱（12G 要跑很久）。
    # 这类文件直接 STORED 存进去，只有文本才压缩。
    store_suffixes = {".parquet", ".zip", ".gz", ".bz2", ".xz", ".pt", ".png", ".jpg"}
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in sorted(dest.rglob("*")):
            if not f.is_file():
                continue
            if f.suffix.lower() in store_suffixes:
                zf.write(f, f.relative_to(base), compress_type=zipfile.ZIP_STORED)
            else:
                zf.write(f, f.relative_to(base),
                         compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
            n += 1
    size = zip_path.stat().st_size
    log(f"  打包完成: {zip_path}  ({n} 个文件, {size / 1024**3:.2f} GiB)")


# ---------------------------------------------------------------- 入口


def parse_args():
    p = argparse.ArgumentParser(description="整理 comm-opt 实验数据并打包")
    p.add_argument("--runtime-log", type=Path, default=DEFAULT_RUNTIME_LOG)
    p.add_argument("--dataset-src", type=Path, default=DEFAULT_DATASET_SRC)
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    p.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    p.add_argument("--no-zip", action="store_true", help="不打包")
    p.add_argument("--dry-run", action="store_true", help="只打印计划，不落盘")
    p.add_argument("--clean", action="store_true",
                   help="先清空目标目录再重建（默认增量：已存在且大小一致的文件跳过）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runtime_log = args.runtime_log.expanduser().resolve()
    dataset_src = args.dataset_src.expanduser().resolve()
    dest = args.dest.expanduser().resolve()

    log("=" * 60)
    log(f"runtime_log : {runtime_log}")
    log(f"数据集源目录: {dataset_src}")
    log(f"输出目录    : {dest}")
    log(f"打包        : {'否' if args.no_zip else args.zip}")
    log("=" * 60)

    if not runtime_log.is_dir():
        log(f"[ERROR] 找不到 {runtime_log}")
        sys.exit(1)

    cases = load_cases(runtime_log)
    if not cases:
        log("[ERROR] runtime_log/case.json 里没有 case 记录，先跑实验再来。")
        sys.exit(1)
    log(f"\n共 {len(cases)} 个 case\n")

    if args.dry_run:
        log("（--dry-run：只打印计划，不写任何东西）\n")
    else:
        if args.clean and dest.exists():
            log(f"--clean：先清掉目标目录 {dest}")
            shutil.rmtree(dest)
        # 只动 comm-opt-data 自己，不碰任何源目录
        dest.mkdir(parents=True, exist_ok=True)

    log("[3] 性能对比数据")
    build_performance(runtime_log, dest, args.dry_run)

    log("\n[数据集]")
    build_dataset(dataset_src, dest, args.dry_run)

    for case in cases:
        build_case(case, runtime_log, dest, args.dry_run)

    build_readme(dest, cases, args.dry_run)

    if not args.no_zip:
        log("\n[打包]")
        make_zip(dest, args.zip.expanduser().resolve(), args.dry_run)

    log("\n完成。")


if __name__ == "__main__":
    main()
