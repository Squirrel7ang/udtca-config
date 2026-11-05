#!/usr/bin/env python3
"""准备 Megatron baseline 用的真实数据集（和 polar 同一份语料）。

polar 的训练数据来自 HuggingFace：`--dataset-name-or-path HuggingFaceFW/fineweb
--dataset-config sample-10BT`，用 `datasets` 流式下载、Qwen2.5 的 tokenizer 切词
（见 experiments/qwen14b/run_qwen14b_polar_dp_pp_tp.py::get_dataloader）。
Megatron 不认 streaming dataset，要预先把文本切成 token 存成 bin/idx。

这个脚本干两件事：
  1. 用 `datasets` 流式抓同一份 fineweb sample-10BT，写成 jsonl（每行 {"text": ...}）；
  2. 调 Megatron 自己的转换脚本 `tools/preprocess_data.py` 把它切成
     `<out-dir>/<prefix>_text_document.bin/.idx`。

tools/preprocess_data.py 是 vendor 进来的 Megatron-LM（core_v0.16.0）原版，
因为 Corex 的 megatron 包只装了 Python 包、没带 tools/。

用法（容器内，两台机器各跑一次，数据不共享）：

    python prepare_data.py \
        --out-dir /data1/tangruijing/megatron-data/fineweb_sample-10BT \
        --tokenizer-model Qwen/Qwen2.5-14B-Instruct    # 与 polar 的 model_name 一致

    # 冒烟：只抓 20 篇、切 10 万 token
    python prepare_data.py --out-dir /tmp/x --max-docs 20 --target-tokens 100000

约定：
  * 已存在 `.idx` 且没给 --force 时直接跳过（幂等）。
  * jsonl 默认保留（--keep-jsonl 默认开），重跑转换不用再下载。
  * 目标规模默认 ~2M token（20 步 × 8192 token/步 ≈ 16 万，留足余量）。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_DATASET = "HuggingFaceFW/fineweb"
DEFAULT_CONFIG = "sample-10BT"
DEFAULT_TEXT_FIELD = "text"
# HF repo id（不是本地目录）：powerone2 上没有本地模型目录，两边统一从 hf-mirror 拉。
DEFAULT_TOKENIZER = "Qwen/Qwen2.5-14B-Instruct"
TOOLS_DIR = Path(__file__).resolve().parent / "tools"


def _log(msg):
    print(f"[prepare-data] {msg}", flush=True)


def download_jsonl(args, out_jsonl: Path) -> dict:
    """流式抓 fineweb，写到 jsonl。返回统计信息。"""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    _log(f"加载 tokenizer: {args.tokenizer_model}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer_model, trust_remote_code=True)

    ds = load_dataset(
        args.dataset,
        name=args.dataset_config or None,
        split="train",
        streaming=True,
        cache_dir=args.hf_cache or None,
    )

    docs = tokens = chars = 0
    tmp = out_jsonl.with_suffix(".jsonl.part")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in ds:
            text = row.get(args.text_field)
            if not text:
                continue
            n_tok = len(tok.encode(text, add_special_tokens=False))
            fh.write(json.dumps({args.text_field: text}, ensure_ascii=False) + "\n")
            docs += 1
            tokens += n_tok
            chars += len(text)
            if docs % 200 == 0:
                _log(f"  已下载 {docs} 篇 / {tokens / 1e6:.2f}M token")
            if docs >= args.max_docs:
                break
            if args.target_tokens and tokens >= args.target_tokens:
                break
    tmp.replace(out_jsonl)
    _log(f"下载完成: {docs} 篇, {tokens / 1e6:.2f}M token, {chars / 1e6:.2f}M 字符 -> {out_jsonl}")
    return {"docs": docs, "tokens": tokens, "chars": chars}


def run_megatron_preprocess(args, out_jsonl: Path, out_prefix: Path) -> None:
    """调 Megatron-LM tools/preprocess_data.py 转成 bin/idx。"""
    script = TOOLS_DIR / "preprocess_data.py"
    if not script.is_file():
        raise SystemExit(f"找不到 Megatron 转换脚本: {script}")

    cmd = [
        sys.executable, str(script),
        "--input", str(out_jsonl),
        "--output-prefix", str(out_prefix),
        "--tokenizer-type", "HuggingFaceTokenizer",
        "--tokenizer-model", args.tokenizer_model,
        "--json-keys", args.text_field,
        "--workers", str(args.workers),
        "--partitions", "1",
        "--log-interval", "500",
    ]
    if args.append_eod:
        cmd.append("--append-eod")
    _log("执行 Megatron preprocess: " + " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"preprocess_data.py 失败（exit={rc}）")


def summarize(out_prefix: Path) -> None:
    """读回 .idx 打印规模，顺便验证 Megatron 能解析。"""
    idx = out_prefix.parent / f"{out_prefix.name}_text_document.idx"
    bin_ = out_prefix.parent / f"{out_prefix.name}_text_document.bin"
    if not idx.is_file():
        return
    _log(f"产物: {bin_.name} {bin_.stat().st_size / 1e6:.1f}MB, "
         f"{idx.name} {idx.stat().st_size / 1e6:.2f}MB")
    try:
        import numpy as np
        from megatron.core.datasets.indexed_dataset import IndexedDataset

        ds = IndexedDataset(str(out_prefix) + "_text_document", mmap=True)
        sizes = [ds[i].size for i in range(min(1000, len(ds)))]
        _log(f"IndexedDataset 可读: {len(ds)} 篇, 前 {len(sizes)} 篇平均 "
             f"{np.mean(sizes):.0f} token")
    except Exception as exc:                                     # noqa: BLE001
        _log(f"（IndexedDataset 校验跳过: {exc.__class__.__name__}: {exc}）")


def main(argv=None):
    p = argparse.ArgumentParser(description="ping 一下 fineweb -> Megatron bin/idx")
    p.add_argument("--out-dir", required=True,
                   help="输出目录（容器内路径，两台机器各自本地）")
    p.add_argument("--prefix", default="fineweb_sample-10BT",
                   help="输出前缀，最终文件是 <prefix>_text_document.bin/.idx")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--dataset-config", default=DEFAULT_CONFIG)
    p.add_argument("--text-field", default=DEFAULT_TEXT_FIELD)
    p.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER,
                   help="Qwen2.5 tokenizer（本地目录或 HF 名字），与 polar 一致")
    p.add_argument("--target-tokens", type=int, default=2_000_000,
                   help="抓到多少 token 就停（0 = 不限，靠 --max-docs）")
    p.add_argument("--max-docs", type=int, default=20000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--hf-cache", default="",
                   help="datasets 的缓存目录（默认沿用 HF 的 ~/.cache/huggingface）")
    p.add_argument("--hf-endpoint", default="https://hf-mirror.com",
                   help="HF 端点；内网直连 huggingface.co 会被 reset，polar 的脚本也是走 hf-mirror")
    p.add_argument("--append-eod", action="store_true", default=True)
    p.add_argument("--no-append-eod", dest="append_eod", action="store_false")
    p.add_argument("--keep-jsonl", action="store_true", default=True)
    p.add_argument("--drop-jsonl", dest="keep_jsonl", action="store_false")
    p.add_argument("--force", action="store_true", help="已有产物也重做")
    args = p.parse_args(argv)

    # 直连 huggingface.co 会被 reset（polar 的启动脚本同样走 hf-mirror，并清掉代理）。
    for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY",
                "all_proxy", "ALL_PROXY"):
        os.environ.pop(var, None)
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "corpus.jsonl"
    out_prefix = out_dir / args.prefix
    done_idx = out_dir / f"{args.prefix}_text_document.idx"

    if done_idx.is_file() and not args.force:
        _log(f"已存在 {done_idx}，跳过（--force 可重做）")
        summarize(out_prefix)
        return 0

    if out_jsonl.is_file() and not args.force:
        _log(f"复用已有 jsonl: {out_jsonl}")
    else:
        download_jsonl(args, out_jsonl)

    run_megatron_preprocess(args, out_jsonl, out_prefix)
    summarize(out_prefix)

    if not args.keep_jsonl:
        out_jsonl.unlink(missing_ok=True)
        _log(f"已删除 {out_jsonl}")

    _log(f"完成。训练时加：--megatron-data-path {out_prefix}_text_document")
    return 0


if __name__ == "__main__":
    sys.exit(main())
