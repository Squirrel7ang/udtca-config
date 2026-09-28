#!/usr/bin/env python3
"""只跑 test.json 里标记为 baseline 的配置（稠密 DP + 1F1B PP + TP）。

训练用的是仓库里现成的 experiments/qwen14b/0|1_train_qwen14b_baseline_ddp_1f1b_tp.sh，
本脚本只负责生成 traffic_control 脚本、下发、限速、起停，以及收集日志。

generate_and_run.py 会 import 这里的 run_baseline_case()，所以同一套 case 编号和
runtime_log/case.json 是两边共用的。
"""

import argparse
import json
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from case_common import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    QWEN_DIR,
    REMOTE_BASE_DIR,
    REMOTE_HOST,
    REMOTE_QWEN_DIR,
    SETUP_CMDS,
    TEST_CONFIG_PATH,
    allocate_case,
    check_gpus_free,
    kill_local_training,
    kill_remote_training,
    stop_traffic_control,
    execute_command,
    finalize_process_log,
    finish_case,
    fetch_remote_text,
    generate_traffic_control_script,
    is_baseline,
    load_json,
    merge_config,
    parse_only,
    parse_launch_params,
    parse_launch_params_text,
    runtime_log_enabled,
    set_runtime_log_enabled,
    ssh_execute,
    write_case_record,
    write_environment_info,
    write_launch_script,
)

# 现成的 baseline 启动脚本（分别在本节点和远程节点上跑）
BASELINE_SCRIPT_0 = "0_train_qwen14b_baseline_ddp_1f1b_tp.sh"  # 本地节点 (u62)
BASELINE_SCRIPT_1 = "1_train_qwen14b_baseline_ddp_1f1b_tp.sh"  # 远程节点 (u210)

LOCAL_PROCESS = None
REMOTE_PROCESS = None
CURRENT_TC_SCRIPT = None  # 当前 case 的限速脚本路径，中断时用来撤销


def cleanup_and_exit(signum, frame):
    """清理并退出"""
    print("\n\nReceived signal, cleaning up...")
    print("NOTE: Baseline scripts are not moved, manual cleanup may be required.")

    if LOCAL_PROCESS is not None:
        print("[LOCAL] Sending SIGTERM to local process...")
        LOCAL_PROCESS.terminate()
        time.sleep(2)
        if LOCAL_PROCESS.poll() is None:
            print("[LOCAL] Force killing local process...")
            LOCAL_PROCESS.kill()

    # 中断时 ssh 链路断了，但 u210 上的 torchrun 不会跟着死，会一直占着卡；
    # 限速规则也会留在两台机器上。这里一并收拾干净。
    print("\nCleaning up remote processes and traffic control...")
    kill_remote_training()
    kill_local_training()
    stop_traffic_control(CURRENT_TC_SCRIPT)

    print("\nCleanup completed. Exiting.")
    exit(1)


def run_baseline_case(config: dict) -> bool:
    """跑一条 baseline 配置：生成 tc -> 下发 -> 限速 -> 两节点起训 -> 收日志。"""
    global LOCAL_PROCESS, REMOTE_PROCESS, CURRENT_TC_SCRIPT

    index = config["index"]
    rate = config["rate"]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # baseline 的脚本是仓库里已有的文件，先确认存在再占用 case 编号
    script_path_0 = QWEN_DIR / BASELINE_SCRIPT_0
    script_path_1 = QWEN_DIR / BASELINE_SCRIPT_1
    if not script_path_0.exists():
        print(f"[ERROR] Local script not found: {script_path_0}")
        return False

    case = case_dir = remote_case_dir = cases = None
    record = None

    if runtime_log_enabled():
        case, case_dir, remote_case_dir, cases = allocate_case()

    print(f"\n{'='*60}")
    if runtime_log_enabled():
        print(f"Running Baseline Experiment index={index} as case {case}")
        print(f"Log dir: {case_dir}")
    else:
        print(f"Running Baseline Experiment index={index}")
        print("runtime_log 已关闭：只验证环境能否跑通，不归档脚本与日志")
    print(f"Config: {json.dumps(config, indent=2)}")
    print(f"{'='*60}")

    tc_script_path = QWEN_DIR / f"traffic_control_exp{index}.sh"
    CURRENT_TC_SCRIPT = tc_script_path

    if runtime_log_enabled():
        # node0 跑的是本地这份；node1 跑的是 u210 上那份，必须从远程读，
        # 本地那份可能是改过的旧版（这两份本地确实不一致）
        record = {
            "case": case,
            "test_index": index,
            "kind": "baseline",
            "started_at": timestamp,
            "log_dir": str(case_dir),
            "run_label": "baseline_ddp_1f1b_tp",
            "rate": rate,
            "config": config,
            "scripts": {
                "node0": BASELINE_SCRIPT_0,
                "node1": BASELINE_SCRIPT_1,
                "traffic_control": tc_script_path.name,
            },
            "launch_params": {
                "node0": parse_launch_params(script_path_0),
                "node1": parse_launch_params_text(
                    fetch_remote_text(f"{REMOTE_QWEN_DIR}/{BASELINE_SCRIPT_1}")
                ) or parse_launch_params(script_path_1),
            },
            "finished_at": None,
            "exit_code": None,
        }
        write_case_record(record, cases)

        # 完整、自包含、可独立复现的启动脚本（交数据时要连同 case 目录一起交出去）
        launch_script = write_launch_script(
            case_dir, case, "baseline (dense DP + 1F1B + TP)",
            {0: record["launch_params"]["node0"], 1: record["launch_params"]["node1"]},
            tc_script_path.name, rate, timestamp,
        )
        print(f"\n0. Wrote standalone launch script: {launch_script}")

    # 1. 生成 traffic control 脚本
    print("\n1. Generating traffic control script...")
    generate_traffic_control_script(tc_script_path, rate)
    print(f"   Generated: {tc_script_path}")

    # 2. 同步 traffic control 脚本到远程节点
    print("\n2. Syncing traffic control script to remote host...")
    execute_command(f"scp {tc_script_path} {REMOTE_HOST}:{REMOTE_QWEN_DIR}/", cwd=QWEN_DIR)

    # 3. 停止现有的 traffic control（本地和远程）
    print("\n3. Stopping existing traffic control [LOCAL]...")
    execute_command(f"bash {tc_script_path} stop", cwd=QWEN_DIR)
    print("\n3. Stopping existing traffic control [REMOTE]...")
    ssh_execute(f"bash {REMOTE_QWEN_DIR}/{tc_script_path.name} stop")

    # 4. 启动 traffic control（本地和远程）
    print("\n4. Starting traffic control [LOCAL]...")
    execute_command(f"bash {tc_script_path} start", cwd=QWEN_DIR)
    print("\n4. Starting traffic control [REMOTE]...")
    ssh_execute(f"bash {REMOTE_QWEN_DIR}/{tc_script_path.name} start")

    # 5. 执行本地脚本（cwd 设成 case 目录，训练侧写的 ./log/... 就落在 case 里）
    print(f"\n5. Executing local script ({BASELINE_SCRIPT_0})...")
    conda_cmd = (
        f"{SETUP_CMDS} && "
        "bash -c \"which torchrun\" && "
        f"bash {script_path_0}"
    )
    # 两边的 torchrun 输出各自落到 case 目录里，远程那份跑完由 rsync 拉回来
    node0_log = (case_dir / "train_node0.log") if case_dir else None
    node1_log = f"{remote_case_dir}/train_node1.log" if remote_case_dir else None

    LOCAL_PROCESS = execute_command(conda_cmd, cwd=case_dir or QWEN_DIR, background=True,
                                    log_path=node0_log)

    # 6. 执行远程脚本（开着 runtime_log 时先切到远程的 case 目录）
    print(f"\n6. Executing remote script ({BASELINE_SCRIPT_1}) via SSH...")
    remote_script = f"{REMOTE_QWEN_DIR}/{BASELINE_SCRIPT_1}"
    if remote_case_dir:
        remote_cmd = f"mkdir -p {remote_case_dir} && cd {remote_case_dir} && bash {remote_script}"
    else:
        remote_cmd = f"bash {remote_script}"
    REMOTE_PROCESS = ssh_execute(remote_cmd, background=True, log_path=node1_log)

    # 7. 等待本地进程完成
    print("\n7. Waiting for local process to complete...")
    LOCAL_PROCESS.wait()
    finalize_process_log(LOCAL_PROCESS)
    exit_code = LOCAL_PROCESS.returncode

    # 8. 检查本地进程退出码
    if exit_code != 0:
        print(f"[WARNING] Local process exited with code {exit_code}")

    # 9. 等待远程进程
    print("\n8. Waiting for remote process...")
    print("[REMOTE] Please monitor the remote node manually.")
    print(f"[REMOTE] SSH to {REMOTE_HOST} and check the process: ps aux | grep {BASELINE_SCRIPT_1}")

    # 10. 停止 traffic control（本地和远程）
    print("\n9. Stopping traffic control [LOCAL]...")
    execute_command(f"bash {tc_script_path} stop", cwd=QWEN_DIR)
    print("\n9. Stopping traffic control [REMOTE]...")
    ssh_execute(f"bash {REMOTE_QWEN_DIR}/{tc_script_path.name} stop")

    # 11. 收集日志：先留档两节点各自的脚本，再把远程日志拉回来统一拍平
    if runtime_log_enabled():
        finish_case(
            record, cases, case_dir, remote_case_dir,
            # 仓库里已有的 baseline 脚本是受版本管理的，必须原地保留，只能拷不能移；
            # 限速脚本是 runner 自己生成的，归集后原地不留
            local_scripts=[(script_path_0, False), (tc_script_path, True)],
            # node1 跑的是 u210 上那份，和本地同名文件内容可能不同，必须从远程拉；
            # 它同样是仓库文件，远程那份也保留
            remote_scripts=[(BASELINE_SCRIPT_1,
                             f"{REMOTE_QWEN_DIR}/{BASELINE_SCRIPT_1}", False)],
            exit_code=exit_code,
        )
    else:
        print("\n10. runtime_log 已关闭，跳过脚本与日志归档")

    return True


def parse_args():
    parser = argparse.ArgumentParser(description="只跑 test.json 里 baseline 的配置")
    parser.add_argument(
        "--runtime-log",
        choices=["on", "off"],
        default=None,
        help="是否把脚本/日志归集到 runtime_log/<case>/（默认 on，也可用环境变量 "
             "UDTCA_RUNTIME_LOG 设置）。off 时只跑训练验证环境能否跑通，不做任何归档",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="只跑指定 index，逗号分隔，例如 --only 2,4；不填则跑全部 baseline 配置",
    )
    return parser.parse_args()


def main():
    """只跑 baseline 配置"""
    print("=" * 60)
    print("Baseline Training Runner")
    print("=" * 60)

    args = parse_args()
    if args.runtime_log is not None:
        set_runtime_log_enabled(args.runtime_log == "on")

    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)

    # 跑之前确认两台机器的卡都没被占用；被占用就报错退出，不等
    if not check_gpus_free():
        print("\nGPU 被占用，本次退出。等卡空出来再重跑。")
        sys.exit(1)

    if runtime_log_enabled():
        print(f"\nEnvironment info written to: {write_environment_info()}")
    else:
        print("\nruntime_log 已关闭：不写 environment.json / case.json，不归档脚本与日志")

    default_config = load_json(DEFAULT_CONFIG_PATH)
    test_configs = load_json(TEST_CONFIG_PATH)
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
        run_baseline_case(config)

    print("\n" + "=" * 60)
    print("All baseline experiments completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
