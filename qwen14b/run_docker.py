#!/usr/bin/env python3
"""跨机端到端训练启动器（Qwen2.5-14B，两台机器各 8 卡，共 16 rank）。

和 run.py 的区别：run.py 那套是宿主机 conda + 4.4.0 toolkit + tc 限速 + 4 个 case
的完整实验编排；这个只管一件事 —— 在容器 `trj-udtca` 里把跨机训练拉起来、把
两边的日志收回来、给出结论。机器和环境已经稳定，不再做占卡检查、端口检查、
拓扑校验这些预检。

实际启动仍然交给 `run_docker_single_node.sh`（NNODES/NODE_RANK/MASTER_ADDR 由本
脚本下发），本脚本只负责：

  1. 本地建 case 目录，写 PRE_RUN.md（记录参数和两边命令）
  2. ssh 到 node1 起 torchrun（node_rank=1，输出落 node1 本地日志）
  3. 等一会儿，再 ssh 到 node0 起 torchrun（node_rank=0，输出实时回显 + 落盘）
  4. node0 退出后把 node1 的日志和 EXIT_CODE 拉回来，打印结果

用法（在能 ssh 到两台机器的机器上跑，不是在容器里跑）：

    python run_docker.py --mode baseline
    python run_docker.py --mode mixed --max-steps 3 --gpus 8
    python run_docker.py --dry-run                 # 只打印两边的命令

停的方式：**不要 kill**。Corex 上打断在途 NCCL 会触发 XID 24，卡永久挂死只能
重启。要停就等它自己的通信看门狗走 revoke 路径退出。
"""

import argparse
import base64
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent           # udtca-config/qwen14b
REPO_ROOT = CONFIG_DIR.parent.parent                   # udtca

DEFAULT_NODE0 = {
    "ssh": "powerone1",
    "ip": "10.205.200.14",
    "entry": "cloud/run_single_node_8gpu.sh",
}
DEFAULT_NODE1 = {
    "ssh": "powerone2",
    "ip": "10.205.200.16",
    "entry": "edge/run_single_node_8gpu.sh",
}
DEFAULT_REMOTE_DIR = "/data/tangruijing/udtca"


# ---------------------------------------------------------------- 命令拼装

def _b64(script: str) -> str:
    return base64.b64encode(script.encode()).decode()


def _container_cmd(container: str, script: str) -> str:
    """在容器里跑一段脚本。

    脚本走 base64：整条命令里没有引号/`$` 这类字符，绕开 ssh + docker exec 的
    多层引号展开（外层展开会把脚本里要 export 的 $VAR 变空）。
    """
    return f"docker exec {container} bash -c 'echo {_b64(script)} | base64 -d | bash'"


def _launch_script(node_rank: int, node: dict, args, case_dir: str) -> str:
    """两台机器上各自要执行的脚本（容器内）。"""
    env = {
        "MODE": args.mode,
        "GPUS": str(args.gpus),
        "PP_SIZE": str(args.pp),
        "TP_SIZE": str(args.tp),
        "MAX_STEPS": str(args.max_steps),
        "MICRO_BATCHES": str(args.micro_batches),
        "SEQ_LEN": str(args.seq_len),
        "MASTER_PORT": str(args.master_port),
        "MASTER_ADDR": args.node0_ip,
        "NNODES": "2",
        "NODE_RANK": str(node_rank),
        "CASE_DIR": f"{case_dir}/node{node_rank}",
        "SEED": str(args.seed),
    }
    env.update(args.env or {})
    assigns = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    return (
        f"cd {args.remote_dir}/udtca-config/qwen14b && "
        f"{assigns} bash {node['entry']}"
    )


def _ssh_cmd(host: str, container: str, script: str) -> str:
    return f"ssh {host} {shlex.quote(_container_cmd(container, script))}"


# ---------------------------------------------------------------- 远端执行

def _run_remote(host: str, command: str, log_path: Path, tee: bool):
    """在远端跑一条命令，返回本地进程。

    tee=True 时输出实时回显并同时落 log_path（node0 用）；tee=False 时只落盘
    （node1 用，两个节点的输出交织在一起太乱，终端只看 node0）。
    两种情况都不等待，由调用方决定怎么等。
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = f"# {datetime.now().isoformat()}  {host}\n# {command}\n\n"
    if tee:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(header)
        # 整条管道（含 tee）交给 bash -o pipefail，退出码才是 ssh 的，不会被 tee 吞掉
        shell_cmd = f"{command} 2>&1 | tee -a {shlex.quote(str(log_path))}"
        return subprocess.Popen(["/bin/bash", "-o", "pipefail", "-c", shell_cmd],
                                start_new_session=True)

    fh = open(log_path, "w", encoding="utf-8", buffering=1)
    fh.write(header)
    proc = subprocess.Popen(["/bin/bash", "-c", command],
                            stdout=fh, stderr=subprocess.STDOUT,
                            start_new_session=True)
    proc._log_fh = fh
    return proc


def _wait_node0(proc, timeout_s):
    """等 node0 结束；给了 timeout 就只等这么久，到点仅停止等待，绝不发信号。"""
    if not timeout_s:
        return proc.wait(), False
    try:
        return proc.wait(timeout=timeout_s), False
    except subprocess.TimeoutExpired:
        return None, True


def _read_remote(path: str, host: str) -> str:
    try:
        out = subprocess.run(["ssh", host, f"cat {path} 2>/dev/null"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip()
    except Exception:
        return ""


def _pull(host: str, remote_dir: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--exclude", "ckpt",
                    f"{host}:{remote_dir}/", f"{local_dir}/"],
                   capture_output=True, text=True, timeout=300)


# ---------------------------------------------------------------- 结果汇总

def _tail_losses(train_log: Path, limit: int = 3) -> list:
    if not train_log.exists():
        return []
    rows = []
    for line in train_log.read_text(errors="ignore").splitlines():
        if "Loss:" in line and "Step" in line:
            rows.append(line.strip())
    return rows[-limit:]


def _summary(case_dir: Path, node0_rc: int, node1_rc: str, node1_running: bool) -> str:
    lines = [
        "",
        "=" * 72,
        f"跨机训练结束  case_dir={case_dir}",
        f"  node0 (powerone1) exit={node0_rc}",
        f"  node1 (powerone2) exit={node1_rc or ('仍在运行' if node1_running else '未知')}",
    ]
    for name in ("node0", "node1"):
        tail = _tail_losses(case_dir / name / "train.log")
        if tail:
            lines.append(f"  {name} 最后几行 loss:")
            lines.extend(f"    {t}" for t in tail)
    if node1_running:
        lines.append(
            "  注意：node1 在 node0 退出后仍在运行。不要 kill，"
            "等它自己的看门狗 revoke 后退出，或人工确认卡状态后再处理。"
        )
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------- main

def make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="跨机端到端训练（容器内，两机各 8 卡）")
    p.add_argument("--mode", default="baseline",
                   choices=["baseline", "polar_dense", "bitscom", "mixed"])
    p.add_argument("--gpus", type=int, default=8, help="每台机器的卡数")
    p.add_argument("--pp", type=int, default=4, help="pp-size")
    p.add_argument("--tp", type=int, default=1, help="tp-size")
    p.add_argument("--max-steps", type=int, default=3)
    p.add_argument("--micro-batches", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--master-port", type=int, default=29630)
    p.add_argument("--container", default="trj-udtca")
    p.add_argument("--node0-ssh", default=os.environ.get("UDTCA_NODE0_HOST", DEFAULT_NODE0["ssh"]))
    p.add_argument("--node0-ip", default=os.environ.get("UDTCA_NODE0_IP", DEFAULT_NODE0["ip"]))
    p.add_argument("--node1-ssh", default=os.environ.get("UDTCA_NODE1_HOST", DEFAULT_NODE1["ssh"]))
    p.add_argument("--node1-ip", default=os.environ.get("UDTCA_NODE1_IP", DEFAULT_NODE1["ip"]))
    p.add_argument("--remote-dir", default=os.environ.get("UDTCA_REMOTE_BASE_DIR", DEFAULT_REMOTE_DIR))
    p.add_argument("--case-dir", default=None,
                   help="远端 case 目录（默认 <remote-dir>/run_log/xnode_<mode>_<时间戳>）")
    p.add_argument("--local-log-root", default=str(REPO_ROOT / "run_log"))
    p.add_argument("--warmup-s", type=int, default=20,
                   help="node1 起来后等多少秒再起 node0（默认 20）")
    p.add_argument("--timeout-s", type=int, default=None,
                   help="node0 前台运行的最长等待（默认不限；到点只是停止等待，不发信号）")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="透传给训练脚本的环境变量，可重复")
    p.add_argument("--dry-run", action="store_true", help="只打印两边的命令，不执行")
    return p


def main() -> int:
    args = make_arg_parser().parse_args()
    args.env = dict(item.split("=", 1) for item in args.env)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_case = args.case_dir or f"{args.remote_dir}/run_log/xnode_{args.mode}_{ts}"
    local_case = Path(args.local_log_root).expanduser().resolve() / f"xnode_{args.mode}_{ts}"
    local_case.mkdir(parents=True, exist_ok=True)

    node0 = {"ssh": args.node0_ssh, "ip": args.node0_ip, "entry": DEFAULT_NODE0["entry"]}
    node1 = {"ssh": args.node1_ssh, "ip": args.node1_ip, "entry": DEFAULT_NODE1["entry"]}

    script0 = _launch_script(0, node0, args, remote_case)
    script1 = _launch_script(1, node1, args, remote_case)
    cmd0 = _ssh_cmd(args.node0_ssh, args.container, script0)
    cmd1 = _ssh_cmd(args.node1_ssh, args.container, script1)

    total_gpus = args.gpus * 2
    dp = total_gpus // (args.pp * args.tp)

    print("=" * 72)
    print(f"跨机训练  mode={args.mode}  nodes=2×{args.gpus}卡  "
          f"DP={dp} PP={args.pp} TP={args.tp}  steps={args.max_steps}")
    print(f"  node0 {args.node0_ssh} ({args.node0_ip}) → {node0['entry']}")
    print(f"  node1 {args.node1_ssh} ({args.node1_ip}) → {node1['entry']}")
    print(f"  master {args.node0_ip}:{args.master_port}")
    print(f"  case_dir(远端) {remote_case}")
    print(f"  case_dir(本地) {local_case}")
    print("=" * 72)

    if args.dry_run:
        print("\n[node1 后台]\n" + cmd1)
        print("\n[node0 前台]\n" + cmd0)
        return 0

    (local_case / "PRE_RUN.md").write_text(
        "\n".join([
            f"# PRE_RUN {ts}",
            f"- mode={args.mode} gpus={args.gpus}x2 DP={dp} PP={args.pp} TP={args.tp}",
            f"- steps={args.max_steps} micro_batches={args.micro_batches} "
            f"seq_len={args.seq_len} seed={args.seed}",
            f"- master={args.node0_ip}:{args.master_port}",
            f"- container={args.container}",
            f"- remote_case={remote_case}",
            "",
            "## node0",
            "```",
            script0,
            "```",
            "",
            "## node1",
            "```",
            script1,
            "```",
            "",
        ]), encoding="utf-8")

    node1_log = local_case / "node1.ssh.log"
    node0_log = local_case / "node0.ssh.log"

    print("\n[1/3] 起 node1（后台）…")
    proc1 = _run_remote(node1["ssh"], cmd1, node1_log, tee=False)
    print(f"      输出: {node1_log}")
    print(f"[2/3] 等 {args.warmup_s}s 让 node1 进入 rendezvous …")
    import time
    time.sleep(args.warmup_s)

    print("[3/3] 起 node0（前台，实时回显）…")
    proc0 = _run_remote(node0["ssh"], cmd0, node0_log, tee=True)
    rc0, node0_timed_out = _wait_node0(proc0, args.timeout_s)
    if node0_timed_out:
        print(f"\nnode0 超过 {args.timeout_s}s 仍未结束，停止等待（不发任何信号）。")
        print(f"看日志: {node0_log}；要收尾等它自己的看门狗 revoke。")

    # node1 此时大概率也已经退出；不等待超时就只是查一下状态，绝不发信号。
    node1_running = proc1.poll() is None
    if node1_running:
        time.sleep(5)
        node1_running = proc1.poll() is None

    node1_rc = _read_remote(f"{remote_case}/node1/EXIT_CODE", node1["ssh"])
    if node1_rc.lower().startswith("exit_code="):
        node1_rc = node1_rc.split("=", 1)[1]

    print("\n拉取两边日志 …")
    _pull(node0["ssh"], f"{remote_case}/node0", local_case / "node0")
    _pull(node1["ssh"], f"{remote_case}/node1", local_case / "node1")

    summary = _summary(local_case, rc0, node1_rc, node1_running)
    print(summary)
    (local_case / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    ok = (rc0 == 0 and node1_rc in ("", "0") and not node1_running)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
