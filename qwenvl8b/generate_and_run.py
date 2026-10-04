#!/usr/bin/env python3
"""按 test.json 依次跑 QwenVL8B 实验（POLAR + bitscom / baseline 两条路径）。

和 qwen14b/generate_and_run.py 一一对应，区别只在模型、并行拓扑和数据集参数：

  * 模型 Qwen/Qwen3-VL-8B-Instruct，PP=8 / TP=1 / DP=4（TP 暂时不支持，见脚本内校验）
  * 数据是随机搓的，用 images-per-sample / image-grid-h / image-grid-w 描述图片形状
  * 训练入口是 run_qwenvl8b_polar_dp_pp.py

baseline 为真的配置走仓库里现成的 0/1_train_qwenvl8b_baseline_ddp_1f1b.sh，
否则生成 0/1_train_qwenvl8b_exp{index}.sh（POLAR + bitscom）。

用法：
    python generate_and_run.py                    # 跑 test.json 全部
    python generate_and_run.py --only 1,3         # 只跑指定 index
    python generate_and_run.py --runtime-log off  # 冒烟：不归档，只验证能跑通
"""

import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import case_common as cc  # noqa: E402
from orchestrator import make_arg_parser, run_experiments  # noqa: E402

QWENVL8B_SPEC = cc.ExperimentSpec(
    name="qwenvl8b",
    experiment_dir=cc.REPO_ROOT / "experiments" / "qwenvl8b",
    entry_script="run_qwenvl8b_polar_dp_pp.py",
    config_dir=cc.CONFIG_ROOT / "qwenvl8b",
    remote_experiment_dir=f"{cc.REMOTE_BASE_DIR}/experiments/qwenvl8b",
    # 和 experiments/qwenvl8b/ 下手写脚本的默认端口一致
    master_port="29510",
)

SCRIPT_PREFIX = "train_qwenvl8b"
BASELINE_SCRIPTS = (
    "0_train_qwenvl8b_baseline_ddp_1f1b.sh",
    "1_train_qwenvl8b_baseline_ddp_1f1b.sh",
)
BASELINE_RUN_LABEL = "baseline_ddp_1f1b_qwenvl8b"


def build_launch_params(config: Dict[str, Any], node_rank: int,
                        case_dir: "Optional[Path]") -> "OrderedDict":
    """POLAR + bitscom 这条路径的启动参数，生成的脚本和 case.json 都从这里渲染。

    参数顺序和 experiments/qwenvl8b/0_train_qwenvl8b_polar_dp_pp.sh 一字不差，
    只把跟着 test.json/default.json 走的值换掉，别的都跟手写脚本保持一致。
    """
    micro_batches = config["micro-batches"]
    params = OrderedDict(
        [
            ("model-name", "Qwen/Qwen3-VL-8B-Instruct"),
            ("pp-size", config["pp-size"]),
            ("tp-size", config["tp-size"]),
            ("micro-batches", micro_batches),
            ("per-device-batch-size", micro_batches),
            ("seq-len", config["seq-len"]),
            ("images-per-sample", config.get("images-per-sample", 1)),
            ("image-grid-h", config.get("image-grid-h", 4)),
            ("image-grid-w", config.get("image-grid-w", 4)),
            # comm-timing 是"第几个 micro-batch 触发 POLAR 通信"，必须 < micro-batches
            ("comm-timing", max(1, micro_batches // 2)),
            ("max-steps", config["max-steps"]),
            ("lr", "2e-4"),
            ("using-polar", True),
            ("run-label", "polar_bitscom_qwenvl8b"),
            ("polar-hook", "ef_lowmem"),
            ("polar-bucket-numel", 64000000),
            # 必须两个节点一致：同一个 DP 组 {j, j+8, j+16, j+24} 跨两个节点，
            # 这个值决定 lowbit 调度器的窗口，节点间不一致会让桶的发射顺序错开而卡死。
            ("polar-max-inflight-buckets", 4),
            ("method", "bitscom"),
            ("bitwidth", config["bit-width"]),
        ]
    )
    # 只有归档时才指定；不指定的话 step csv 会落到 cwd 下的默认目录
    if case_dir is not None:
        params["step-log-dir"] = str(case_dir / "step_csv")
    return params


def main():
    args = make_arg_parser("按 test.json 跑 QwenVL8B 实验").parse_args()
    run_experiments(
        QWENVL8B_SPEC,
        args,
        build_params=build_launch_params,
        script_prefix=SCRIPT_PREFIX,
        baseline_scripts=BASELINE_SCRIPTS,
        baseline_run_label=BASELINE_RUN_LABEL,
    )


if __name__ == "__main__":
    main()
