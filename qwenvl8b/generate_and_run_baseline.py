#!/usr/bin/env python3
"""只跑 qwenvl8b/test.json 里标记为 baseline 的配置（稠密 DP + 1F1B PP）。

训练用的是仓库里现成的
experiments/qwenvl8b/0|1_train_qwenvl8b_baseline_ddp_1f1b.sh，
本脚本只负责生成 traffic_control 脚本、下发、限速、起停、收日志。

和 generate_and_run.py 共用同一套 case 编号与 runtime_log/case.json。

用法：
    python generate_and_run_baseline.py                    # 跑全部 baseline 配置
    python generate_and_run_baseline.py --only 2,4         # 只跑指定 index
    python generate_and_run_baseline.py --runtime-log off  # 冒烟：不归档
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from case_common import (  # noqa: E402
    check_gpus_free,
    is_baseline,
    load_json,
    merge_config,
    parse_only,
    runtime_log_enabled,
    set_runtime_log_enabled,
    write_environment_info,
)
from orchestrator import (  # noqa: E402
    install_signal_handlers,
    make_arg_parser,
    run_baseline_case,
)

from generate_and_run import (  # noqa: E402
    BASELINE_RUN_LABEL,
    BASELINE_SCRIPTS,
    QWENVL8B_SPEC,
)


def main():
    print("=" * 60)
    print("QwenVL8B Baseline Training Runner")
    print("=" * 60)

    args = make_arg_parser("只跑 qwenvl8b/test.json 里 baseline 的配置").parse_args()
    if args.runtime_log is not None:
        set_runtime_log_enabled(args.runtime_log == "on")

    install_signal_handlers(QWENVL8B_SPEC)

    if not check_gpus_free():
        print("\nGPU 被占用，本次退出。等卡空出来再重跑。")
        sys.exit(1)

    if runtime_log_enabled():
        print(f"\nEnvironment info written to: {write_environment_info()}")
    else:
        print("\nruntime_log 已关闭：不写 environment.json / case.json，不归档脚本与日志")

    default_config = load_json(QWENVL8B_SPEC.default_config_path)
    test_configs = load_json(QWENVL8B_SPEC.test_config_path)
    only = parse_only(args.only)
    if only is not None:
        test_configs = [c for c in test_configs if c.get("index") in only]
        print(f"\n只跑 index {sorted(only)}")

    baseline_configs = [
        merge_config(default_config, config)
        for config in test_configs
        if is_baseline(config, default_config)
    ]

    if not baseline_configs:
        print("\nNo experiments with baseline=True found in test.json")
        return

    print(f"\nFound {len(baseline_configs)} baseline experiments to run:")
    for config in baseline_configs:
        print(f"  index={config['index']}, rate={config['rate']}, max-steps={config['max-steps']}")

    for config in baseline_configs:
        run_baseline_case(QWENVL8B_SPEC, config, BASELINE_SCRIPTS, BASELINE_RUN_LABEL)

    print("\n" + "=" * 60)
    print("All baseline experiments completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
