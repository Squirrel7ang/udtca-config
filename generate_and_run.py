#!/usr/bin/env python3
"""按 test.json 依次跑 Qwen14B 实验。

每条配置按它自己的 `baseline` 标记分派：
  - baseline 为真  -> 复用仓库里现成的 0/1_train_qwen14b_baseline_ddp_1f1b_tp.sh
  - 否则           -> 生成 POLAR + bitscom 的 0/1_train_qwen14b_exp{index}.sh 再跑

每次运行占用一个递增的 case 编号，tensorboard / profiler trace / step csv 都收到
<repo>/runtime_log/<case>/ 下，启动脚本与参数记在 runtime_log/case.json。
"""

import argparse
import signal
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from case_common import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    HF_ENDPOINT,
    QWEN_DIR,
    REMOTE_BASE_DIR,
    REMOTE_HOST,
    REMOTE_QWEN_DIR,
    runtime_log_enabled,
    set_runtime_log_enabled,
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
    generate_traffic_control_script,
    is_baseline,
    load_json,
    merge_config,
    parse_only,
    ssh_execute,
    write_case_record,
    write_environment_info,
    write_launch_script,
)
from generate_and_run_baseline import run_baseline_case  # noqa: E402

LOCAL_PROCESS = None
CURRENT_EXP_INFO = None


def cleanup_and_exit(signum, frame):
    global LOCAL_PROCESS, CURRENT_EXP_INFO
    print(f"\n\nReceived signal {signum}, cleaning up...")

    if LOCAL_PROCESS is not None:
        print("Killing local process...")
        try:
            LOCAL_PROCESS.terminate()
        except Exception as e:
            print(f"Failed to terminate local process: {e}")
        try:
            LOCAL_PROCESS.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Force killing local process...")
            try:
                LOCAL_PROCESS.kill()
                LOCAL_PROCESS.wait(timeout=2)
            except Exception as e:
                print(f"Failed to kill local process: {e}")

    # 中断时 ssh 链路断了，但 u210 上的 torchrun 不会跟着死，会一直占着卡；
    # 限速规则也会留在两台机器上。这里一并收拾干净。
    tc_script = CURRENT_EXP_INFO[4] if CURRENT_EXP_INFO is not None else None
    print("\nCleaning up remote processes and traffic control...")
    kill_remote_training()
    kill_local_training()
    stop_traffic_control(tc_script)

    if CURRENT_EXP_INFO is not None:
        print("NOTE: 本次 case 的启动脚本留在 experiments/qwen14b/，没归档。")

    print("Cleanup complete, exiting...")
    sys.exit(1)


signal.signal(signal.SIGINT, cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)


# ---------------------------------------------------------------- POLAR 启动脚本


def build_launch_params(config: Dict[str, Any], node_rank: int,
                        case_dir: "Optional[Path]") -> "OrderedDict":
    """训练进程的启动参数。生成的脚本、case.json 都从这里渲染，保证两者一致。

    case_dir 为 None（runtime_log 关掉）时不指定 step-log-dir，让训练落到自己的 cwd 下。
    """
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


def render_cli_params(params: "OrderedDict", indent: str = "  ") -> str:
    """把参数字典渲染成 torchrun 后面的 `--k v \\` 行。"""
    lines = []
    for key, value in params.items():
        rendered = str(value).lower() if isinstance(value, bool) else value
        lines.append(f"{indent}--{key} {rendered}")
    return " \\\n".join(lines)


def generate_train_script(
    output_path: Path,
    node_rank: int,
    launch_params: "OrderedDict",
    case_dir: "Optional[Path]",
    hf_endpoint: str,
) -> None:
    # 两个节点都走 hf-mirror。ssh 和 conda 起的都是非交互非登录 shell，不会 source
    # ~/.bashrc，所以必须在这里显式导出 HF_ENDPOINT，不能依赖 .bashrc 或代理工具。
    hf_block = f""" \
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY && \
unset NO_PROXY no_proxy && \
export HF_ENDPOINT={hf_endpoint} \
"""

    # 关掉 runtime_log 时不做 cd，训练产物落在训练进程自己的 cwd 下
    if case_dir is None:
        case_block = ""
        cd_block = ""
    else:
        case_block = f"""
# 训练侧把 tb/trace 写成相对路径 ./log/...，切到这个 case 目录就落在 runtime_log/{case_dir.name}/
CASE_DIR="{case_dir}"
"""
        cd_block = 'mkdir -p "${CASE_DIR}"\ncd "${CASE_DIR}"\n'

    script_content = f"""\
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REPO_ROOT="$(cd "${{SCRIPT_DIR}}/../.." && pwd)"
{case_block}
MASTER_ADDR="${{MASTER_ADDR:-10.31.10.62}}"
MASTER_PORT="${{MASTER_PORT:-29500}}"
NNODES="${{NNODES:-2}}"
NPROC_PER_NODE="${{NPROC_PER_NODE:-16}}"
NCCL_SOCKET_IFNAME="${{NCCL_SOCKET_IFNAME:-ens1f0}}"
NCCL_IB_DISABLE="${{NCCL_IB_DISABLE:-1}}"

# export TORCH_DISTRIBUTED_DEBUG="${{TORCH_DISTRIBUTED_DEBUG:-DETAIL}}"
export NCCL_ASYNC_ERROR_HANDLING="${{NCCL_ASYNC_ERROR_HANDLING:-1}}"
export CUDA_LAUNCH_BLOCKING="${{CUDA_LAUNCH_BLOCKING:-0}}"
# export NCCL_DEBUG="${{NCCL_DEBUG:-INFO}}"
export NCCL_SOCKET_IFNAME
export NCCL_IB_DISABLE
# export PYTHONPATH="${{REPO_ROOT}}/polar-sgd/src:${{REPO_ROOT}}/bitscom/python:${{PYTHONPATH:-}}"

{hf_block}

{cd_block}
echo "`which torchrun` returns $(which torchrun)"

torchrun \\
  --nproc_per_node="${{NPROC_PER_NODE}}" \\
  --nnodes="${{NNODES}}" \\
  --node_rank={node_rank} \\
  --master_addr="${{MASTER_ADDR}}" \\
  --master_port="${{MASTER_PORT}}" \\
  "${{SCRIPT_DIR}}/run_qwen14b_polar_dp_pp_tp.py" \\
{render_cli_params(launch_params)}
"""

    with open(output_path, "w") as f:
        f.write(script_content)
    output_path.chmod(0o755)


# ---------------------------------------------------------------- 单个 case


def run_polar_case(config: Dict[str, Any]) -> None:
    """跑一条 POLAR + bitscom 配置：生成脚本 -> 下发 -> 限速 -> 训练 -> 收日志。"""
    global LOCAL_PROCESS, CURRENT_EXP_INFO

    index = config["index"]
    rate = config["rate"]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    case = case_dir = remote_case_dir = cases = None
    record = None

    if runtime_log_enabled():
        case, case_dir, remote_case_dir, cases = allocate_case()

    print(f"\n{'='*60}")
    if runtime_log_enabled():
        print(f"Running POLAR experiment index={index} as case {case}")
        print(f"Log dir: {case_dir}")
    else:
        print(f"Running POLAR experiment index={index}")
        print("runtime_log 已关闭：只验证环境能否跑通，不归档脚本与日志")
    print(f"Config: {config}")
    print(f"{'='*60}")

    script_0_path = QWEN_DIR / f"0_train_qwen14b_exp{index}.sh"
    script_1_path = QWEN_DIR / f"1_train_qwen14b_exp{index}.sh"
    tc_script_path = QWEN_DIR / f"traffic_control_exp{index}.sh"

    CURRENT_EXP_INFO = (index, timestamp, script_0_path, script_1_path,
                        tc_script_path, REMOTE_QWEN_DIR, REMOTE_BASE_DIR)

    # 先生成两份脚本的参数，再记录 case，保证 case.json 里写的就是实际下发的启动参数
    launch_0 = build_launch_params(config, 0, case_dir)
    launch_1 = build_launch_params(config, 1, case_dir)

    if runtime_log_enabled():
        record = {
            "case": case,
            "test_index": index,
            "kind": "polar",
            "started_at": timestamp,
            "log_dir": str(case_dir),
            "run_label": launch_0["run-label"],
            "rate": rate,
            "config": config,
            "scripts": {
                "node0": script_0_path.name,
                "node1": script_1_path.name,
                "traffic_control": tc_script_path.name,
            },
            "launch_params": {"node0": launch_0, "node1": launch_1},
            "finished_at": None,
            "exit_code": None,
        }
        write_case_record(record, cases)

        # 完整、自包含、可独立复现的启动脚本（交数据时要连同 case 目录一起交出去）
        launch_script = write_launch_script(
            case_dir, case, "POLAR + bitscom", {0: launch_0, 1: launch_1},
            tc_script_path.name, rate, timestamp,
        )
        print(f"\n0. Wrote standalone launch script: {launch_script}")

    print("\n1. Generating training scripts...")
    generate_train_script(script_0_path, 0, launch_0, case_dir, HF_ENDPOINT)
    generate_train_script(script_1_path, 1, launch_1, case_dir, HF_ENDPOINT)
    print(f"   Generated: {script_0_path}")
    print(f"   Generated: {script_1_path}")

    print("\n2. Generating traffic control script...")
    generate_traffic_control_script(tc_script_path, rate)
    print(f"   Generated: {tc_script_path}")

    print("\n3. Syncing scripts to remote host...")
    execute_command(
        f"scp {script_0_path} {script_1_path} {tc_script_path} "
        f"{REMOTE_HOST}:{REMOTE_QWEN_DIR}/",
        cwd=QWEN_DIR,
    )

    print("\n4. Stopping existing traffic control [LOCAL]...")
    execute_command(f"{tc_script_path} stop", cwd=QWEN_DIR)
    print("\n4. Stopping existing traffic control [REMOTE]...")
    ssh_execute(f"{REMOTE_QWEN_DIR}/{tc_script_path.name} stop")

    print("\n5. Starting traffic control [LOCAL]...")
    execute_command(f"{tc_script_path} start", cwd=QWEN_DIR)
    print("\n5. Starting traffic control [REMOTE]...")
    ssh_execute(f"{REMOTE_QWEN_DIR}/{tc_script_path.name} start")

    # 两边的 torchrun 输出各自落到 case 目录里，远程那份跑完由 rsync 拉回来
    node0_log = (case_dir / "train_node0.log") if case_dir else None
    node1_log = f"{remote_case_dir}/train_node1.log" if remote_case_dir else None

    print("\n6. Executing node 0 script locally (background)...")
    conda_cmd = (
        f"{SETUP_CMDS} && "
        "bash -c \"which torchrun\" && "
        f"bash {script_0_path}"
    )
    LOCAL_PROCESS = execute_command(conda_cmd, cwd=QWEN_DIR, background=True, log_path=node0_log)

    print("\n7. Executing node 1 script via SSH...")
    remote_script_1 = f"{REMOTE_QWEN_DIR}/{script_1_path.name}"
    ssh_execute(f"bash {remote_script_1}", log_path=node1_log)

    print("\n8. Waiting for training to complete...")
    print("   Waiting for local node 0...")
    exit_code = LOCAL_PROCESS.wait()
    finalize_process_log(LOCAL_PROCESS)
    print(f"   node 0 exited with code {exit_code}")

    print("\n9. Stopping traffic control [LOCAL]...")
    execute_command(f"{tc_script_path} stop")
    print("\n9. Stopping traffic control [REMOTE]...")
    ssh_execute(f"{REMOTE_QWEN_DIR}/{tc_script_path.name} stop")

    if runtime_log_enabled():
        print("\n10. Archiving scripts...")
        finish_case(
            record, cases, case_dir, remote_case_dir,
            # node0 的脚本和限速脚本都是 runner 自己生成的，归集后原地不留
            local_scripts=[(script_0_path, True), (tc_script_path, True)],
            # node1 实际跑的是 u210 上那份，必须从远程拉；拉完清掉远程的临时副本
            remote_scripts=[(script_1_path.name,
                             f"{REMOTE_QWEN_DIR}/{script_1_path.name}", True)],
            exit_code=exit_code,
        )
    else:
        print("\n10. runtime_log 已关闭，跳过脚本与日志归档")

    LOCAL_PROCESS = None
    CURRENT_EXP_INFO = None

    if runtime_log_enabled():
        print(f"\nExperiment index={index} (case {case}) completed!")
    else:
        print(f"\nExperiment index={index} completed (runtime_log off)!")


# ---------------------------------------------------------------- 入口


def parse_args():
    parser = argparse.ArgumentParser(description="按 test.json 跑 Qwen14B 实验")
    parser.add_argument(
        "--runtime-log",
        choices=["on", "off"],
        default=None,
        help="是否把脚本/日志归集到 runtime_log/<case>/（默认 on，也可用环境变量 "
             "UDTCA_RUNTIME_LOG 设置）。off 时只跑训练验证环境能否跑通，不建 case 目录、"
             "不写 case.json、不归档脚本与日志",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="只跑指定 index，逗号分隔，例如 --only 1,2,9；不填则跑 test.json 里全部",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.runtime_log is not None:
        set_runtime_log_enabled(args.runtime_log == "on")

    # 跑之前确认两台机器的卡都没被占用；被占用就报错退出，不等
    if not check_gpus_free():
        print("\nGPU 被占用，本次退出。等卡空出来再重跑。")
        sys.exit(1)

    if runtime_log_enabled():
        print(f"Environment info written to: {write_environment_info()}")
    else:
        print("runtime_log 已关闭：不写 environment.json / case.json，不归档脚本与日志")

    print("Loading default configuration...")
    default_config = load_json(DEFAULT_CONFIG_PATH)
    print(f"Default config: {default_config}")

    print("\nLoading test configurations...")
    test_configs = load_json(TEST_CONFIG_PATH)
    only = parse_only(args.only)
    if only is not None:
        test_configs = [c for c in test_configs if c.get("index") in only]
        print(f"只跑 index {sorted(only)}")
    print(f"Found {len(test_configs)} test configurations")

    for i, test_config in enumerate(test_configs):
        print(f"\n{'='*80}")
        print(f"Processing test configuration {i+1}/{len(test_configs)}")
        print(f"Raw config: {test_config}")

        merged_config = merge_config(default_config, test_config)
        if is_baseline(test_config, default_config):
            print("-> baseline 配置，走 generate_and_run_baseline 的 baseline 路径")
            run_baseline_case(merged_config)
        else:
            print("-> POLAR + bitscom 配置")
            run_polar_case(merged_config)


if __name__ == "__main__":
    main()
