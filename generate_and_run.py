#!/usr/bin/env python3
import os
import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
QWEN_DIR = SCRIPT_DIR / "experiments" / "qwen14b"
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "default.json"
TEST_CONFIG_PATH = SCRIPT_DIR / "test.json"


def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def merge_config(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = default.copy()
    merged.update(override)
    return merged


def generate_traffic_control_script(output_path: Path, rate: str) -> None:
    script_content = f"""\
#!/bin/bash

# ==========================================
# tc 网络限速 + 时延模拟脚本
# ==========================================

DEV="ens1f0"
RATE="{rate}"
BURST="32kbit"
LATENCY="400ms"
DELAY="0ms"

start_tc() {{
    echo "[INFO] 开始配置 tc ..."
    tc qdisc del dev ${{DEV}} root 2>/dev/null
    tc qdisc add dev ${{DEV}} root handle 1: htb default 10
    tc class add dev ${{DEV}} parent 1: classid 1:10 \\
        htb rate ${{RATE}} ceil ${{RATE}}
    tc qdisc add dev ${{DEV}} parent 1:10 handle 10: \\
        netem delay ${{DELAY}}
    echo "[INFO] 配置完成"
}}

stop_tc() {{
    echo "[INFO] 删除 tc 配置 ..."
    tc qdisc del dev ${{DEV}} root 2>/dev/null
    echo "[INFO] tc 已恢复默认"
}}

status_tc() {{
    echo "========== qdisc =========="
    tc qdisc show dev ${{DEV}}
    echo
    echo "========== class =========="
    tc class show dev ${{DEV}}
}}

case "$1" in
    start)
        start_tc
        ;;
    stop)
        stop_tc
        ;;
    status)
        status_tc
        ;;
    *)
        echo "Usage:"
        echo "  sudo $0 start"
        echo "  sudo $0 stop"
        echo "  sudo $0 status"
        exit 1
        ;;
esac
"""
    with open(output_path, "w") as f:
        f.write(script_content)
    output_path.chmod(0o755)


def generate_train_script(
    output_path: Path,
    node_rank: int,
    pp_size: int,
    tp_size: int,
    bit_width: int,
    micro_batches: int,
    max_steps: int,
    seq_len: int,
) -> None:
    inflight_buckets = 4 if node_rank == 0 else 1
    
    script_content = f"""\
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
REPO_ROOT="$(cd "${{SCRIPT_DIR}}/../.." && pwd)"
MASTER_ADDR="${{MASTER_ADDR:-10.31.10.210}}"
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

export HF_ENDPOINT=https://hf-mirror.com
export http_proxy="http://10.31.10.20:7892"
export https_proxy="http://10.31.10.20:7892"
export NO_PROXY="localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,www.dogapi.cc"
export no_proxy="$NO_PROXY"


torchrun \\
  --nproc_per_node="${{NPROC_PER_NODE}}" \\
  --nnodes="${{NNODES}}" \\
  --node_rank={node_rank} \\
  --master_addr="${{MASTER_ADDR}}" \\
  --master_port="${{MASTER_PORT}}" \\
  "${{SCRIPT_DIR}}/run_qwen14b_polar_dp_pp_tp.py" \\
  --model-name Qwen/Qwen2.5-14B-Instruct \\
  --pp-size {pp_size} \\
  --tp-size {tp_size} \\
  --micro-batches {micro_batches} \\
  --comm-timing 8 \\
  --max-steps {max_steps} \\
  --per-device-batch-size 32 \\
  --seq-len {seq_len} \\
  --lr 2e-4 \\
  --dataset-name-or-path HuggingFaceFW/fineweb \\
  --text-field text \\
  --using-polar true \\
  --run-label polar_bitscom_1f1b_tp \\
  --polar-hook ef_lowmem \\
  --polar-bucket-numel 64000000 \\
  --polar-max-inflight-buckets {inflight_buckets} \\
  --method bitscom \\
  --bitwidth {bit_width}
"""
    
    with open(output_path, "w") as f:
        f.write(script_content)
    output_path.chmod(0o755)


def execute_command(cmd: str, cwd: Optional[Path] = None) -> int:
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(f"[local] STDOUT: {result.stdout}")
    if result.stderr:
        print(f"[local] STDERR: {result.stderr}")
    return result.returncode


def ssh_execute(
    cmd: str,
    host: str = "root@10.31.10.62",
    base_dir: str = "/home/tangruijing/udtca",
    setup_cmds: str = None,
) -> int:
    if setup_cmds is None:
        setup_cmds = (
            "source /root/miniconda3/etc/profile.d/conda.sh && "
            "export PATH='/root/miniconda3/bin:$PATH' && "
            "unset __conda_setup && "
            "conda activate tangruijing && "
            "source /root/clashctl/scripts/cmd/clashctl.sh && "
            "clashon && unset HF_ENDPOINT"
        )
    full_cmd = f"ssh {host} 'cd {base_dir} && {setup_cmds} && {cmd}'"
    print(f"Executing (SSH): {full_cmd}")
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(f"[{host}] STDOUT: {result.stdout}")
    if result.stderr:
        print(f"[{host}] STDERR: {result.stderr}")
    return result.returncode


def run_experiment(config: Dict[str, Any]) -> None:
    index = config["index"]
    pp_size = config["pp-size"]
    tp_size = config["tp-size"]
    rate = config["rate"]
    bit_width = config["bit-width"]
    micro_batches = config["micro-batches"]
    max_steps = config["max-steps"]
    seq_len = config["seq-len"]

    remote_host = "root@10.31.10.62"
    remote_base_dir = "/home/tangruijing/udtca"
    remote_qwen_dir = f"{remote_base_dir}/experiments/qwen14b"

    print(f"\n{'='*60}")
    print(f"Running experiment {index}")
    print(f"Config: {json.dumps(config, indent=2)}")
    print(f"{'='*60}")

    script_0_path = QWEN_DIR / f"0_train_qwen14b_exp{index}.sh"
    script_1_path = QWEN_DIR / f"1_train_qwen14b_exp{index}.sh"
    tc_script_path = QWEN_DIR / f"traffic_control_exp{index}.sh"

    print("\n1. Generating training scripts...")
    generate_train_script(script_0_path, 0, pp_size, tp_size, bit_width, micro_batches, max_steps, seq_len)
    generate_train_script(script_1_path, 1, pp_size, tp_size, bit_width, micro_batches, max_steps, seq_len)
    print(f"   Generated: {script_0_path}")
    print(f"   Generated: {script_1_path}")

    print("\n2. Generating traffic control script...")
    generate_traffic_control_script(tc_script_path, rate)
    print(f"   Generated: {tc_script_path}")

    print("\n3. Syncing scripts to remote host...")
    execute_command(f"scp {script_0_path} {script_1_path} {tc_script_path} {remote_host}:{remote_qwen_dir}/")

    print("\n4. Stopping existing traffic control [LOCAL]...")
    execute_command(f"{tc_script_path} stop")
    print("\n4. Stopping existing traffic control [REMOTE]...")
    ssh_execute(f"{remote_qwen_dir}/{tc_script_path.name} stop", setup_cmds="")

    print("\n5. Starting traffic control [LOCAL]...")
    execute_command(f"{tc_script_path} start")
    print("\n5. Starting traffic control [REMOTE]...")
    ssh_execute(f"{remote_qwen_dir}/{tc_script_path.name} start", setup_cmds="")

    print("\n6. Executing node 0 script locally...")
    conda_cmd = (
        "source /root/miniconda3/etc/profile.d/conda.sh && "
        "export PATH='/root/miniconda3/bin:$PATH' && "
        "unset __conda_setup && "
        f"conda activate tangruijing && {script_0_path}"
    )
    execute_command(conda_cmd, cwd=QWEN_DIR)

    print("\n7. Executing node 1 script via SSH...")
    remote_script_1 = f"{remote_qwen_dir}/{script_1_path.name}"
    ssh_execute(f"{remote_script_1}")

    print("\n8. Stopping traffic control [LOCAL]...")
    execute_command(f"{tc_script_path} stop")
    print("\n8. Stopping traffic control [REMOTE]...")
    ssh_execute(f"{remote_qwen_dir}/{tc_script_path.name} stop", setup_cmds="")

    print("\n9. Moving scripts to script_log directory [LOCAL]...")
    script_log_dir = SCRIPT_DIR / "script_log"
    script_log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_log_dir = script_log_dir / f"exp{index}_{timestamp}"
    exp_log_dir.mkdir(exist_ok=True)
    shutil.move(str(script_0_path), exp_log_dir / script_0_path.name)
    shutil.move(str(script_1_path), exp_log_dir / script_1_path.name)
    shutil.move(str(tc_script_path), exp_log_dir / tc_script_path.name)
    print(f"   Moved to: {exp_log_dir}")

    print("\n10. Moving scripts to script_log directory [REMOTE]...")
    remote_log_dir = f"{remote_base_dir}/script_log/exp{index}_{timestamp}"
    mv_cmd = (
        f"mkdir -p {remote_log_dir} && "
        f"mv {remote_qwen_dir}/{script_0_path.name} "
        f"{remote_qwen_dir}/{script_1_path.name} "
        f"{remote_qwen_dir}/{tc_script_path.name} "
        f"{remote_log_dir}/"
    )
    ssh_execute(mv_cmd, setup_cmds="")
    print(f"   Moved to: {remote_log_dir}")

    print(f"\nExperiment {index} completed!")


def main():
    print("Loading default configuration...")
    default_config = load_json(DEFAULT_CONFIG_PATH)
    print(f"Default config: {json.dumps(default_config, indent=2)}")

    print("\nLoading test configurations...")
    test_configs = load_json(TEST_CONFIG_PATH)
    print(f"Found {len(test_configs)} test configurations")

    for i, test_config in enumerate(test_configs):
        print(f"\n{'='*80}")
        print(f"Processing test configuration {i+1}/{len(test_configs)}")
        print(f"Raw config: {test_config}")
        
        merged_config = merge_config(default_config, test_config)
        print(f"Merged config: {json.dumps(merged_config, indent=2)}")
        
        run_experiment(merged_config)


if __name__ == "__main__":
    main()
