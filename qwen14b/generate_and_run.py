#!/usr/bin/env python3
"""按 test.json 依次跑 Qwen14B 实验（POLAR + bitscom / baseline 两条路径）。

每条配置按它自己的 `baseline` 标记分派：
  - baseline 为真  -> 复用仓库里现成的 0/1_train_qwen14b_baseline_ddp_1f1b_tp.sh
  - 否则           -> 生成 POLAR + bitscom 的 0/1_train_qwen14b_exp{index}.sh 再跑

每次运行占用一个递增的 case 编号，tensorboard / profiler trace / step csv 都收到
<repo>/runtime_log/<case>/ 下，启动脚本与参数记在 runtime_log/case.json。

生成脚本 / 下发 / 限速 / 起训 / 收日志这些与实验无关的流程在 common/orchestrator.py，
这里只留 Qwen14B 自己的：spec + 启动参数。

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

QWEN14B_SPEC = cc.ExperimentSpec(
    name="qwen14b",
    experiment_dir=cc.REPO_ROOT / "experiments" / "qwen14b",
    entry_script="run_qwen14b_polar_dp_pp_tp.py",
    config_dir=cc.CONFIG_ROOT / "qwen14b",
    remote_experiment_dir=f"{cc.REMOTE_BASE_DIR}/experiments/qwen14b",
)

# 生成的 POLAR 脚本叫 0_{SCRIPT_PREFIX}_exp{index}.sh
SCRIPT_PREFIX = "train_qwen14b"
# 仓库里现成的 baseline 脚本（node0 本地 / node1 远程）
BASELINE_SCRIPTS = (
    "0_train_qwen14b_baseline_ddp_1f1b_tp.sh",
    "1_train_qwen14b_baseline_ddp_1f1b_tp.sh",
)
BASELINE_RUN_LABEL = "baseline_ddp_1f1b_tp"


def build_launch_params(config: Dict[str, Any], node_rank: int,
                        case_dir: "Optional[Path]") -> "OrderedDict":
    """POLAR + bitscom 这条路径的启动参数，生成的脚本和 case.json 都从这里渲染。"""
    micro_batches = config["micro-batches"]
    params = OrderedDict(
        [
            ("model-name", "Qwen/Qwen2.5-14B-Instruct"),
            ("pp-size", config["pp-size"]),
            ("tp-size", config["tp-size"]),
            ("micro-batches", micro_batches),
            ("comm-timing", 8),
            ("max-steps", config["max-steps"]),
            ("per-device-batch-size", micro_batches),
            ("seq-len", config["seq-len"]),
            ("lr", "2e-4"),
            ("dataset-name-or-path", "HuggingFaceFW/fineweb"),
            ("text-field", "text"),
            ("using-polar", True),
            ("run-label", "polar_bitscom_1f1b_tp"),
            ("polar-hook", "ef_lowmem"),
            ("polar-bucket-numel", 64000000),
            ("polar-max-inflight-buckets", 4 if node_rank == 0 else 1),
            ("method", "bitscom"),
            ("bitwidth", config["bit-width"]),
            # 每 10 步打一次 step/loss（polar-sgd 的 wrapper 读这个值）
            ("log-interval", 10),
        ]
    )
    if case_dir is not None:
        params["step-log-dir"] = str(case_dir / "step_csv")
    # profiler 必须开：显式写 False（入口脚本的默认值也是 False），
    # 这样生成的脚本和 case.json 里都能直接看到这个设置。
    params["disable-profiler"] = False
    return params


def main():
    args = make_arg_parser("按 test.json 跑 Qwen14B 实验").parse_args()
    run_experiments(
        QWEN14B_SPEC,
        args,
        build_params=build_launch_params,
        script_prefix=SCRIPT_PREFIX,
        baseline_scripts=BASELINE_SCRIPTS,
        baseline_run_label=BASELINE_RUN_LABEL,
    )


if __name__ == "__main__":
    main()
