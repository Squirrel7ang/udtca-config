#!/usr/bin/env python3
"""两节点实验的通用编排流程：生成脚本 -> 下发 -> 限速 -> 起训 -> 收日志。

每个实验目录（qwen14b/、qwenvl8b/）只需要提供三样东西：

  * 一个 `case_common.ExperimentSpec`（路径、训练入口文件名）
  * `build_launch_params(config, node_rank, case_dir)` —— 训练进程的启动参数
  * 脚本前缀 + baseline 用的现成脚本名（没有 baseline 路径就给 None）

其余逻辑这里全包了。case 编号、runtime_log/case.json、限速脚本、日志归集都和
实验无关，所以放公共库里，避免两个实验各抄一份。
"""

import argparse
import json
import signal
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from case_common import (
    ExperimentSpec,
    REMOTE_BASE_DIR,
    REMOTE_HOST,
    SETUP_CMDS,
    allocate_case,
    check_gpus_free,
    execute_command,
    finalize_process_log,
    finish_case,
    fetch_remote_text,
    generate_traffic_control_script,
    is_baseline,
    kill_local_training,
    kill_remote_training,
    load_json,
    merge_config,
    parse_only,
    parse_launch_params,
    parse_launch_params_text,
    runtime_log_enabled,
    set_runtime_log_enabled,
    ssh_execute,
    stop_traffic_control,
    use_experiment,
    write_case_record,
    write_environment_info,
    write_launch_script,
)
from case_common import HF_ENDPOINT

# 当前 case 的进程与脚本，中断时用来收拾干净
LOCAL_PROCESS = None
CURRENT_EXP_INFO = None  # (index, timestamp, script0, script1, tc_script, remote_dir, base_dir)


# ---------------------------------------------------------------- 中断清理


def cleanup_and_exit(signum, frame, spec: ExperimentSpec = None):
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
        name = spec.name if spec is not None else "实验"
        print(f"NOTE: 本次 case 的启动脚本留在 experiments/{name}/，没归档。")

    print("Cleanup complete, exiting...")
    sys.exit(1)


def install_signal_handlers(spec: ExperimentSpec) -> None:
    handler = lambda signum, frame: cleanup_and_exit(signum, frame, spec)  # noqa: E731
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


# ---------------------------------------------------------------- 启动脚本渲染


def build_launch_params(config: Dict[str, Any], node_rank: int,
                        case_dir: "Optional[Path]") -> "OrderedDict":
    """训练进程的启动参数。生成的脚本、case.json 都从这里渲染，保证两者一致。

    case_dir 为 None（runtime_log 关掉）时不指定 step-log-dir，让训练落到自己的 cwd 下。
    """
    raise NotImplementedError("由各实验目录提供")


def render_cli_params(params: "OrderedDict", indent: str = "  ") -> str:
    """把参数字典渲染成 torchrun 后面的 `--k v \\` 行。"""
    lines = []
    for key, value in params.items():
        rendered = str(value).lower() if isinstance(value, bool) else value
        lines.append(f"{indent}--{key} {rendered}")
    return " \\\n".join(lines)


def write_train_script(
    output_path: Path,
    node_rank: int,
    launch_params: "OrderedDict",
    case_dir: "Optional[Path]",
    entry_script: str,
    master_port: str = "29500",
    hf_endpoint: str = HF_ENDPOINT,
) -> None:
    """渲染并写出一份该实验的 torchrun 启动脚本。

    模板刻意和 experiments/<exp>/ 下手写的 0|1_train_*.sh 保持一致（同样的环境变量、
    同样的顺序、同样的默认端口），只把跟着配置走的参数换掉。生成的脚本和手写脚本
    长得一样，跑起来才一样。
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

# Node {node_rank} of 2. MASTER_ADDR 必须指向跑 node_rank=0 的那台机器（本机 u62）。
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REPO_ROOT="$(cd "${{SCRIPT_DIR}}/../.." && pwd)"
{case_block}
MASTER_ADDR="${{MASTER_ADDR:-10.31.10.62}}"
MASTER_PORT="${{MASTER_PORT:-{master_port}}}"
NNODES="${{NNODES:-2}}"
NPROC_PER_NODE="${{NPROC_PER_NODE:-16}}"
NCCL_SOCKET_IFNAME="${{NCCL_SOCKET_IFNAME:-ens1f0}}"
NCCL_IB_DISABLE="${{NCCL_IB_DISABLE:-1}}"

export NCCL_ASYNC_ERROR_HANDLING="${{NCCL_ASYNC_ERROR_HANDLING:-1}}"
export CUDA_LAUNCH_BLOCKING="${{CUDA_LAUNCH_BLOCKING:-0}}"
export NCCL_SOCKET_IFNAME
export NCCL_IB_DISABLE

# ssh / conda 起的都是非交互非登录 shell，不会 source ~/.bashrc，HF 镜像要显式导出
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY NO_PROXY no_proxy
export HF_ENDPOINT={hf_endpoint}

{cd_block}
torchrun \\
  --nproc_per_node="${{NPROC_PER_NODE}}" \\
  --nnodes="${{NNODES}}" \\
  --node_rank={node_rank} \\
  --master_addr="${{MASTER_ADDR}}" \\
  --master_port="${{MASTER_PORT}}" \\
  "${{SCRIPT_DIR}}/{entry_script}" \\
{render_cli_params(launch_params)}
"""

    with open(output_path, "w") as f:
        f.write(script_content)
    output_path.chmod(0o755)


# ---------------------------------------------------------------- POLAR / 自生成脚本路径


def run_polar_case(
    spec: ExperimentSpec,
    config: Dict[str, Any],
    build_params: Callable,
    script_prefix: str,
    kind_label: str = "POLAR + bitscom",
) -> None:
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

    script_0_path = spec.experiment_dir / f"0_{script_prefix}_exp{index}.sh"
    script_1_path = spec.experiment_dir / f"1_{script_prefix}_exp{index}.sh"
    tc_script_path = spec.experiment_dir / f"traffic_control_exp{index}.sh"

    CURRENT_EXP_INFO = (index, timestamp, script_0_path, script_1_path,
                        tc_script_path, spec.remote_experiment_dir, REMOTE_BASE_DIR)

    # 先生成两份脚本的参数，再记录 case，保证 case.json 里写的就是实际下发的启动参数
    launch_0 = build_params(config, 0, case_dir)
    launch_1 = build_params(config, 1, case_dir)

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
            case_dir, case, kind_label, {0: launch_0, 1: launch_1},
            tc_script_path.name, rate, timestamp,
        )
        print(f"\n0. Wrote standalone launch script: {launch_script}")

    print("\n1. Generating training scripts...")
    write_train_script(script_0_path, 0, launch_0, case_dir, spec.entry_script,
                       spec.master_port)
    write_train_script(script_1_path, 1, launch_1, case_dir, spec.entry_script,
                       spec.master_port)
    print(f"   Generated: {script_0_path}")
    print(f"   Generated: {script_1_path}")

    print("\n2. Generating traffic control script...")
    generate_traffic_control_script(tc_script_path, rate)
    print(f"   Generated: {tc_script_path}")

    print("\n3. Syncing scripts to remote host...")
    execute_command(
        f"scp {script_0_path} {script_1_path} {tc_script_path} "
        f"{REMOTE_HOST}:{spec.remote_experiment_dir}/",
        cwd=spec.experiment_dir,
    )

    print("\n4. Stopping existing traffic control [LOCAL]...")
    execute_command(f"{tc_script_path} stop", cwd=spec.experiment_dir)
    print("\n4. Stopping existing traffic control [REMOTE]...")
    ssh_execute(f"{spec.remote_experiment_dir}/{tc_script_path.name} stop")

    print("\n5. Starting traffic control [LOCAL]...")
    execute_command(f"{tc_script_path} start", cwd=spec.experiment_dir)
    print("\n5. Starting traffic control [REMOTE]...")
    ssh_execute(f"{spec.remote_experiment_dir}/{tc_script_path.name} start")

    # 两边的 torchrun 输出各自落到 case 目录里，远程那份跑完由 rsync 拉回来
    node0_log = (case_dir / "train_node0.log") if case_dir else None
    node1_log = f"{remote_case_dir}/train_node1.log" if remote_case_dir else None

    print("\n6. Executing node 0 script locally (background)...")
    conda_cmd = (
        f"{SETUP_CMDS} && "
        "bash -c \"which torchrun\" && "
        f"bash {script_0_path}"
    )
    LOCAL_PROCESS = execute_command(conda_cmd, cwd=spec.experiment_dir, background=True,
                                    log_path=node0_log)

    print("\n7. Executing node 1 script via SSH...")
    remote_script_1 = f"{spec.remote_experiment_dir}/{script_1_path.name}"
    ssh_execute(f"bash {remote_script_1}", log_path=node1_log)

    print("\n8. Waiting for training to complete...")
    print("   Waiting for local node 0...")
    exit_code = LOCAL_PROCESS.wait()
    finalize_process_log(LOCAL_PROCESS)
    print(f"   node 0 exited with code {exit_code}")

    print("\n9. Stopping traffic control [LOCAL]...")
    execute_command(f"{tc_script_path} stop")
    print("\n9. Stopping traffic control [REMOTE]...")
    ssh_execute(f"{spec.remote_experiment_dir}/{tc_script_path.name} stop")

    if runtime_log_enabled():
        print("\n10. Archiving scripts...")
        finish_case(
            record, cases, case_dir, remote_case_dir,
            # node0 的脚本和限速脚本都是 runner 自己生成的，归集后原地不留
            local_scripts=[(script_0_path, True), (tc_script_path, True)],
            # node1 实际跑的是 u210 上那份，必须从远程拉；拉完清掉远程的临时副本
            remote_scripts=[(script_1_path.name,
                             f"{spec.remote_experiment_dir}/{script_1_path.name}", True)],
            exit_code=exit_code,
        )
    else:
        print("\n10. runtime_log 已关闭，跳过脚本与日志归档")

    LOCAL_PROCESS = None
    CURRENT_EXP_INFO = None
    print(f"\nExperiment index={index} completed!")


# ---------------------------------------------------------------- baseline（现成脚本）


def run_baseline_case(
    spec: ExperimentSpec,
    config: Dict[str, Any],
    baseline_scripts: Tuple[str, str],
    run_label: str,
) -> bool:
    """跑一条 baseline 配置：生成 tc -> 下发 -> 限速 -> 两节点起训 -> 收日志。

    训练用的是仓库里现成的脚本（baseline_scripts 是 node0/node1 两个文件名），
    本函数只负责限速脚本、起停和收日志。
    """
    global LOCAL_PROCESS, CURRENT_EXP_INFO

    index = config["index"]
    rate = config["rate"]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # baseline 跑的是仓库里现成的手写 .sh，里面的参数都是写死的。这里只把
    # max-steps 用合并后的 config（default.json + test.json）盖掉，通过
    # MAX_STEPS 环境变量传给 .sh；其余参数一律保持 .sh 原样。
    max_steps = config.get("max-steps")
    max_steps_env = (
        f"export MAX_STEPS={int(max_steps)} && " if max_steps is not None else ""
    )

    script_name_0, script_name_1 = baseline_scripts
    script_path_0 = spec.experiment_dir / script_name_0
    script_path_1 = spec.experiment_dir / script_name_1
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

    tc_script_path = spec.experiment_dir / f"traffic_control_exp{index}.sh"
    CURRENT_EXP_INFO = (index, timestamp, script_path_0, script_path_1,
                        tc_script_path, spec.remote_experiment_dir, REMOTE_BASE_DIR)

    if runtime_log_enabled():
        # node0 跑的是本地这份；node1 跑的是 u210 上那份，必须从远程读，
        # 本地那份可能是改过的旧版（这两份本地确实不一致）
        record = {
            "case": case,
            "test_index": index,
            "kind": "baseline",
            "started_at": timestamp,
            "log_dir": str(case_dir),
            "run_label": run_label,
            "rate": rate,
            "config": config,
            "scripts": {
                "node0": script_name_0,
                "node1": script_name_1,
                "traffic_control": tc_script_path.name,
            },
            "launch_params": {
                "node0": parse_launch_params(script_path_0),
                "node1": parse_launch_params_text(
                    fetch_remote_text(f"{spec.remote_experiment_dir}/{script_name_1}")
                ) or parse_launch_params(script_path_1),
            },
            "finished_at": None,
            "exit_code": None,
        }
        # .sh 里 max-steps 写成 ${MAX_STEPS:-200}，解析出来是字面量；
        # case.json 里改成记实际生效的值，便于复查。
        if max_steps is not None:
            for node_key in ("node0", "node1"):
                params = record["launch_params"].get(node_key)
                if params is not None:
                    params["max-steps"] = int(max_steps)
        write_case_record(record, cases)

        # 完整、自包含、可独立复现的启动脚本（交数据时要连同 case 目录一起交出去）
        launch_script = write_launch_script(
            case_dir, case, "baseline (dense DP + 1F1B)",
            {0: record["launch_params"]["node0"], 1: record["launch_params"]["node1"]},
            tc_script_path.name, rate, timestamp,
        )
        print(f"\n0. Wrote standalone launch script: {launch_script}")

    print("\n1. Generating traffic control script...")
    generate_traffic_control_script(tc_script_path, rate)
    print(f"   Generated: {tc_script_path}")

    print("\n2. Syncing traffic control script to remote host...")
    execute_command(f"scp {tc_script_path} {REMOTE_HOST}:{spec.remote_experiment_dir}/",
                    cwd=spec.experiment_dir)

    print("\n3. Stopping existing traffic control [LOCAL]...")
    execute_command(f"bash {tc_script_path} stop", cwd=spec.experiment_dir)
    print("\n3. Stopping existing traffic control [REMOTE]...")
    ssh_execute(f"bash {spec.remote_experiment_dir}/{tc_script_path.name} stop")

    print("\n4. Starting traffic control [LOCAL]...")
    execute_command(f"bash {tc_script_path} start", cwd=spec.experiment_dir)
    print("\n4. Starting traffic control [REMOTE]...")
    ssh_execute(f"bash {spec.remote_experiment_dir}/{tc_script_path.name} start")

    # 两边的 torchrun 输出各自落到 case 目录里，远程那份跑完由 rsync 拉回来
    node0_log = (case_dir / "train_node0.log") if case_dir else None
    node1_log = f"{remote_case_dir}/train_node1.log" if remote_case_dir else None

    print(f"\n5. Executing local script ({script_name_0})...")
    conda_cmd = (
        f"{SETUP_CMDS} && "
        f"{max_steps_env}"
        "bash -c \"which torchrun\" && "
        f"bash {script_path_0}"
    )
    LOCAL_PROCESS = execute_command(conda_cmd, cwd=case_dir or spec.experiment_dir,
                                    background=True, log_path=node0_log)

    print(f"\n6. Executing remote script ({script_name_1}) via SSH...")
    remote_script = f"{spec.remote_experiment_dir}/{script_name_1}"
    if remote_case_dir:
        remote_cmd = (
            f"mkdir -p {remote_case_dir} && cd {remote_case_dir} && "
            f"{max_steps_env}bash {remote_script}"
        )
    else:
        remote_cmd = f"{max_steps_env}bash {remote_script}"
    ssh_execute(remote_cmd, background=True, log_path=node1_log)

    print("\n7. Waiting for local process to complete...")
    exit_code = LOCAL_PROCESS.wait()
    finalize_process_log(LOCAL_PROCESS)
    if exit_code != 0:
        print(f"[WARNING] Local process exited with code {exit_code}")

    print("\n8. Stopping traffic control [LOCAL]...")
    execute_command(f"bash {tc_script_path} stop", cwd=spec.experiment_dir)
    print("\n8. Stopping traffic control [REMOTE]...")
    ssh_execute(f"bash {spec.remote_experiment_dir}/{tc_script_path.name} stop")

    if runtime_log_enabled():
        print("\n9. Archiving scripts...")
        finish_case(
            record, cases, case_dir, remote_case_dir,
            # 仓库里已有的 baseline 脚本是受版本管理的，必须原地保留，只能拷不能移；
            # 限速脚本是 runner 自己生成的，归集后原地不留
            local_scripts=[(script_path_0, False), (tc_script_path, True)],
            # node1 跑的是 u210 上那份，和本地同名文件内容可能不同，必须从远程拉；
            # 它同样是仓库文件，远程那份也保留
            remote_scripts=[(script_name_1,
                             f"{spec.remote_experiment_dir}/{script_name_1}", False)],
            exit_code=exit_code,
        )
    else:
        print("\n9. runtime_log 已关闭，跳过脚本与日志归档")

    LOCAL_PROCESS = None
    CURRENT_EXP_INFO = None
    return True


# ---------------------------------------------------------------- 入口


def make_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
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
    return parser


def run_experiments(
    spec: ExperimentSpec,
    args: argparse.Namespace,
    build_params: Callable,
    script_prefix: str,
    baseline_scripts: "Optional[Tuple[str, str]]" = None,
    baseline_run_label: str = "baseline",
    kind_label: str = "POLAR + bitscom",
) -> None:
    """跑 test.json 里的全部配置，按每条的 baseline 标记分派到两条路径。"""
    # 把 case_common 里那几个路径常量（尤其是 ENTRYPOINT）绑到本次实验上。
    # 不调的话 ENTRYPOINT 还是模块级默认值（qwen14b），归档出来的 launch.sh
    # 会指向错误的训练入口。
    use_experiment(spec)

    if args.runtime_log is not None:
        set_runtime_log_enabled(args.runtime_log == "on")

    install_signal_handlers(spec)

    # 跑之前确认两台机器的卡都没被占用；被占用就报错退出，不等
    if not check_gpus_free():
        print("\nGPU 被占用，本次退出。等卡空出来再重跑。")
        sys.exit(1)

    if runtime_log_enabled():
        print(f"Environment info written to: {write_environment_info()}")
    else:
        print("runtime_log 已关闭：不写 environment.json / case.json，不归档脚本与日志")

    print("Loading default configuration...")
    default_config = load_json(spec.default_config_path)
    print(f"Default config: {default_config}")

    print("\nLoading test configurations...")
    test_configs = load_json(spec.test_config_path)
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
            if baseline_scripts is None:
                raise SystemExit(
                    f"{spec.name} 没有配 baseline 脚本，但 test.json 里有 baseline 配置"
                )
            print("-> baseline 配置")
            run_baseline_case(spec, merged_config, baseline_scripts, baseline_run_label)
        else:
            print(f"-> {kind_label} 配置")
            run_polar_case(spec, merged_config, build_params, script_prefix, kind_label)
