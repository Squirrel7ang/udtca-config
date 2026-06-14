#!/usr/bin/env python3
"""
运行 Baseline 训练的脚本
跳过 baseline 为 False 的配置，只运行 baseline 为 True 的配置
"""

import json
import time
import signal
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

# ============== 全局配置 ==============
LOCAL_HOST = "210"  # 本地节点标识
REMOTE_HOST = "62"  # 远程节点标识
REMOTE_IP = "10.31.10.62"
REMOTE_USER = "root"
REMOTE_BASE_DIR = "/home/tangruijing/udtca"
REMOTE_QWEN_DIR = f"{REMOTE_BASE_DIR}/experiments/qwen14b"
LOCAL_QWEN_DIR = Path(__file__).resolve().parent / "experiments" / "qwen14b"

# Baseline 脚本名称
BASELINE_SCRIPT_0 = "0_train_qwen14b_baseline_ddp_1f1b_tp.sh"  # 本地节点 (210)
BASELINE_SCRIPT_1 = "1_train_qwen14b_baseline_ddp_1f1b_tp.sh"  # 远程节点 (62)

LOCAL_PROCESS = None
REMOTE_PROCESS = None
QUIET_SSH = True
CMD_DELAY_SECONDS = 3

# ============== SSH 初始化命令 ==============
SETUP_CMDS = (
    "export PATH=/usr/local/corex/bin:/usr/local/corex/lib64/python3/dist-packages/bin:$PATH && "
    "export LD_LIBRARY_PATH=/usr/local/corex/lib64 && "
    "export PATH=/usr/local/corex/bin:$PATH && "
    "export PYTHONPATH=/usr/local/corex/lib64/python3/dist-package && "
    "export PATH=/usr/local/corex/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin && "
    "source /root/miniconda3/etc/profile.d/conda.sh && "
    "export PATH='/root/miniconda3/bin:$PATH' && "
    "unset __conda_setup && "
    "conda activate tangruijing && "
    "source /root/clashctl/scripts/cmd/clashctl.sh && "
    "clashon && unset HF_ENDPOINT"
)


def load_config():
    """加载配置"""
    script_dir = Path(__file__).resolve().parent
    default_config = json.loads((script_dir / "default.json").read_text())
    test_configs = json.loads((script_dir / "test.json").read_text())
    return default_config, test_configs


def merge_config(default, override):
    """合并配置"""
    merged = default.copy()
    merged.update(override)
    return merged


def generate_traffic_control_script(output_path: Path, rate: str) -> None:
    """生成 traffic control 脚本"""
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


def ssh_execute(cmd: str, setup_cmds: str = SETUP_CMDS, background: bool = False):
    """在远程主机上执行命令"""
    full_cmd = (
        f"ssh {REMOTE_USER}@{REMOTE_IP} '/bin/bash -c \""
        f"cd {REMOTE_BASE_DIR} && {setup_cmds} && {cmd}"
        f"\"'"
    )
    print(f"[REMOTE {REMOTE_HOST}] Executing: {full_cmd}")

    process = subprocess.Popen(full_cmd, shell=True, start_new_session=True)
    if not background:
        process.wait()
        if not QUIET_SSH:
            stdout, stderr = process.communicate()
            if stdout:
                for line in stdout.decode('utf-8').splitlines():
                    print(f"[REMOTE {REMOTE_HOST}] {line}")
            if stderr:
                for line in stderr.decode('utf-8').splitlines():
                    print(f"[REMOTE {REMOTE_HOST}] {line}")
        if CMD_DELAY_SECONDS > 0:
            time.sleep(CMD_DELAY_SECONDS)
    return process


def execute_command(cmd: str, cwd: Optional[Path] = None, background: bool = False):
    """在本地执行命令"""
    bash_cmd = f"/bin/bash -c '{cmd}'"
    print(f"[LOCAL {LOCAL_HOST}] Executing: {bash_cmd}")
    process = subprocess.Popen(bash_cmd, shell=True, cwd=cwd)
    if not background:
        process.wait()
        stdout, stderr = process.communicate()
        if stdout:
            for line in stdout.decode('utf-8').splitlines():
                print(f"[LOCAL {LOCAL_HOST}] {line}")
        if stderr:
            for line in stderr.decode('utf-8').splitlines():
                print(f"[LOCAL {LOCAL_HOST}] {line}")
        if CMD_DELAY_SECONDS > 0:
            time.sleep(CMD_DELAY_SECONDS)
    return process


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

    if REMOTE_PROCESS is not None:
        print("[REMOTE] Please manually kill the remote process on node 62")
        print(f"[REMOTE] Command: ssh root@{REMOTE_IP} 'pkill -f {BASELINE_SCRIPT_1}'")

    print("\nCleanup completed. Exiting.")
    exit(1)


def run_baseline_exp(config: dict):
    """运行单个 baseline 实验"""
    global LOCAL_PROCESS, REMOTE_PROCESS

    index = config["index"]
    rate = config["rate"]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(f"\n{'='*60}")
    print(f"Running Baseline Experiment index={index}")
    print(f"Config: {json.dumps(config, indent=2)}")
    print(f"{'='*60}")

    tc_script_path = LOCAL_QWEN_DIR / f"traffic_control_exp{index}.sh"

    # 1. 生成 traffic control 脚本
    print(f"\n1. Generating traffic control script...")
    generate_traffic_control_script(tc_script_path, rate)
    print(f"   Generated: {tc_script_path}")

    # 2. 同步 traffic control 脚本到远程节点
    print(f"\n2. Syncing traffic control script to remote host...")
    execute_command(f"scp {tc_script_path} root@{REMOTE_IP}:{REMOTE_QWEN_DIR}/", cwd=LOCAL_QWEN_DIR)

    # 3. 停止现有的 traffic control（本地和远程）
    print(f"\n3. Stopping existing traffic control [LOCAL]...")
    execute_command(f"bash {tc_script_path} stop", cwd=LOCAL_QWEN_DIR)
    print(f"\n3. Stopping existing traffic control [REMOTE]...")
    ssh_execute(f"bash {REMOTE_QWEN_DIR}/{tc_script_path.name} stop")

    # 4. 启动 traffic control（本地和远程）
    print(f"\n4. Starting traffic control [LOCAL]...")
    execute_command(f"bash {tc_script_path} start", cwd=LOCAL_QWEN_DIR)
    print(f"\n4. Starting traffic control [REMOTE]...")
    ssh_execute(f"bash {REMOTE_QWEN_DIR}/{tc_script_path.name} start")

    # 5. 执行本地脚本
    print(f"\n5. Executing local script ({BASELINE_SCRIPT_0})...")
    script_path_0 = LOCAL_QWEN_DIR / BASELINE_SCRIPT_0
    if not script_path_0.exists():
        print(f"[ERROR] Local script not found: {script_path_0}")
        return False

    print(f"[LOCAL {LOCAL_HOST}] Starting {BASELINE_SCRIPT_0}...")
    conda_cmd = (
        "export PATH=/usr/local/corex/bin:/usr/local/corex/lib64/python3/dist-packages/bin:$PATH && "
        "export LD_LIBRARY_PATH=/usr/local/corex/lib64 && "
        "export PATH=/usr/local/corex/bin:$PATH && "
        "export PYTHONPATH=/usr/local/corex/lib64/python3/dist-package && "
        "export PATH=/usr/local/corex/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin && "
        "source /root/miniconda3/etc/profile.d/conda.sh && "
        "export PATH='/root/miniconda3/bin:$PATH' && "
        "unset __conda_setup && "
        "conda deactivate && "
        "conda activate tangruijing && "
        "bash -c \"which torchrun\" && "
        f"bash {script_path_0}"
    )
    LOCAL_PROCESS = execute_command(conda_cmd, cwd=LOCAL_QWEN_DIR, background=True)

    # 6. 执行远程脚本
    print(f"\n6. Executing remote script ({BASELINE_SCRIPT_1}) via SSH...")
    remote_script = f"{REMOTE_QWEN_DIR}/{BASELINE_SCRIPT_1}"
    remote_cmd = f"bash {remote_script}"
    print(f"[REMOTE {REMOTE_HOST}] Starting {BASELINE_SCRIPT_1}...")
    REMOTE_PROCESS = ssh_execute(remote_cmd, background=True)

    # 7. 等待本地进程完成
    print("\n7. Waiting for local process to complete...")
    LOCAL_PROCESS.wait()

    # 8. 检查本地进程退出码
    if LOCAL_PROCESS.returncode != 0:
        print(f"[WARNING] Local process exited with code {LOCAL_PROCESS.returncode}")

    # 9. 等待远程进程
    print("\n8. Waiting for remote process...")
    print("[REMOTE] Please monitor the remote node manually.")
    print(f"[REMOTE] SSH to {REMOTE_IP} and check the process: ps aux | grep {BASELINE_SCRIPT_1}")

    # 10. 停止 traffic control（本地和远程）
    print(f"\n9. Stopping traffic control [LOCAL]...")
    execute_command(f"bash {tc_script_path} stop", cwd=LOCAL_QWEN_DIR)
    print(f"\n9. Stopping traffic control [REMOTE]...")
    ssh_execute(f"bash {REMOTE_QWEN_DIR}/{tc_script_path.name} stop")

    return True


def main():
    """主函数"""
    print("=" * 60)
    print("Baseline Training Runner")
    print("=" * 60)

    # 注册信号处理
    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)

    # 加载配置
    default_config, test_configs = load_config()

    # 筛选 baseline 为 True 的配置
    baseline_configs = [
        merge_config(default_config, config)
        for config in test_configs
        if config.get("baseline", default_config.get("baseline", False))
    ]

    if not baseline_configs:
        print("\nNo experiments with baseline=True found in test.json")
        return

    print(f"\nFound {len(baseline_configs)} baseline experiments to run:")
    for config in baseline_configs:
        print(f"  index={config['index']}, rate={config['rate']}, max-steps={config['max-steps']}")

    # 运行每个 baseline 实验
    for config in baseline_configs:
        run_baseline_exp(config)

    print("\n" + "=" * 60)
    print("All baseline experiments completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
