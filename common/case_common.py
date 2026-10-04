#!/usr/bin/env python3
"""udtca Qwen 实验编排的公共部分。

generate_and_run.py（POLAR / bitscom）和 generate_and_run_baseline.py（baseline）
共用这里的常量、两节点命令封装、case 编号与 case.json 读写、日志收集。
"""

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# 目录约定：
#   <repo>/udtca-config/              本目录（CONFIG_ROOT）
#   <repo>/udtca-config/common/       公共库（COMMON_DIR，这个文件所在处）
#   <repo>/udtca-config/<exp>/        每个实验自己的 default.json / test.json / 编排脚本
#   <repo>/udtca-config/utils/        与具体实验无关的辅助脚本
#   <repo>/experiments/<exp>/         训练代码
COMMON_DIR = Path(__file__).resolve().parent
CONFIG_ROOT = COMMON_DIR.parent
REPO_ROOT = CONFIG_ROOT.parent
UTILS_DIR = CONFIG_ROOT / "utils"

# 本机是 u62 (10.31.10.62)，节点 0 在本地跑；节点 1 通过 ssh 别名 u210 在远程跑。
LOCAL_HOST = "62"
LOCAL_IP = "10.31.10.62"
REMOTE_HOST = "u210"  # ~/.ssh/config 里的别名，等价于 root@10.31.10.210
REMOTE_IP = "10.31.10.210"
REMOTE_BASE_DIR = "/data1/tangruijing/udtca"
CONDA_ENV = "trj-test"
HF_ENDPOINT = "https://hf-mirror.com"


@dataclass(frozen=True)
class ExperimentSpec:
    """一个实验在编排器眼里要交代的全部信息。

    各实验目录里的 generate_and_run*.py 在 main() 开头调 use_experiment() 绑定自己的
    spec，下面那几个模块级路径常量就会被重绑，其余逻辑完全共用。
    """

    name: str                    # "qwen14b" / "qwenvl8b"
    experiment_dir: Path         # <repo>/experiments/<name>
    entry_script: str            # 训练入口文件名
    config_dir: Path             # <repo>/udtca-config/<name>
    remote_experiment_dir: str   # u210 上的对应目录
    master_port: str = "29500"   # 生成的启动脚本里 MASTER_PORT 的默认值

    @property
    def entrypoint(self) -> Path:
        return self.experiment_dir / self.entry_script

    @property
    def default_config_path(self) -> Path:
        return self.config_dir / "default.json"

    @property
    def test_config_path(self) -> Path:
        return self.config_dir / "test.json"


SPEC: Optional[ExperimentSpec] = None

# 默认按 qwen14b 绑定，保证老用法（直接 import 本模块）行为不变；
# 其它实验在 main() 里调 use_experiment() 覆盖。
QWEN_DIR = REPO_ROOT / "experiments" / "qwen14b"
DEFAULT_CONFIG_PATH = CONFIG_ROOT / "qwen14b" / "default.json"
TEST_CONFIG_PATH = CONFIG_ROOT / "qwen14b" / "test.json"
REMOTE_QWEN_DIR = f"{REMOTE_BASE_DIR}/experiments/qwen14b"


def use_experiment(spec: ExperimentSpec) -> ExperimentSpec:
    """绑定当前实验，重绑下面前缀为 QWEN_ 的路径常量。

    注意要在用到这些常量的代码之前调用；`from case_common import QWEN_DIR` 这种写法
    拿的是导入那一刻的值，所以调用方一律用 `case_common.QWEN_DIR` 形式访问。
    """
    global SPEC, QWEN_DIR, DEFAULT_CONFIG_PATH, TEST_CONFIG_PATH, REMOTE_QWEN_DIR
    global ENTRYPOINT
    SPEC = spec
    QWEN_DIR = spec.experiment_dir
    DEFAULT_CONFIG_PATH = spec.default_config_path
    TEST_CONFIG_PATH = spec.test_config_path
    REMOTE_QWEN_DIR = spec.remote_experiment_dir
    ENTRYPOINT = spec.entrypoint
    return spec

# 本地和远程共用的运行时环境准备，四步：
#   1) 跑一遍机器自己的 shell 初始化
#   2) 显式对齐 Corex / CUDA 相关变量
#   3) 进 conda 环境
#   4) HF 镜像 + 清代理
#
# 第 1 步的原因：ssh 非交互连过去不会 source ~/.bashrc，而本地跑时进程是从交互 shell
# 继承环境的，两边会不一致。注意 ~/.bashrc 开头通常有 `[ -z "$PS1" ] && return` 这种
# 非交互保护，直接 source 是空操作，所以先塞一个 PS1 再 source。
#
# 第 2 步的原因：两台的 ~/.bashrc 内容并不一样——u62 设了 COREX_PATH / CPATH /
# LIBRARY_PATH / TRITON_CUDA_SYSROOT / CUDA_DEVICE_MAX_CONNECTIONS=1，
# 而 u210 没有。这些会影响编译和 CUDA/NCCL 行为，必须显式补齐，不能指望 .bashrc。
# 用 ${VAR:-} 是因为这段也会被写进 launch.sh，那里开着 set -u。
SETUP_CMDS = " && ".join(
    [
        # 1) 机器自己的 shell 初始化（含非交互保护绕过）
        #    ~/.bashrc 是给交互 shell 写的，直接 source 会炸：
        #      - `set -e` 下里面任何返回非零的命令都会让整个脚本退出
        #      - `set -u` 下引用未定义变量（这台是 line 28 的 debian_chroot）
        #        会让非交互 shell **整个退出**，set +e 拦不住
        #    launch.sh 就同时开着 -e 和 -u，所以在 source 前后把这两个选项都关掉，
        #    之后再按原样恢复——runner 那条路径本来就没开，不能给它开上。
        'if [ -f ~/.bashrc ]; then PS1="${PS1:-ustca-init}"; _flags="$-"; set +eu;'
        " . ~/.bashrc >/dev/null 2>&1 || true;"
        ' case "$_flags" in *e*) set -e;; esac; case "$_flags" in *u*) set -u;; esac;'
        " unset _flags PS1; fi",
        # 2) Corex / CUDA 变量，两台强制一致
        "export COREX_PATH=/usr/local/corex-4.4.0",
        'export CUDA_HOME="$COREX_PATH"',
        'export CPATH="$COREX_PATH/include:${CPATH:-}"',
        'export LIBRARY_PATH="$COREX_PATH/lib64:${LIBRARY_PATH:-}"',
        'export TRITON_CUDA_SYSROOT="$COREX_PATH"',
        'export BITSCOM_CUDA_COMPILER="$COREX_PATH/bin/clang++"',
        "export CUDA_DEVICE_MAX_CONNECTIONS=1",
        "export PATH=/usr/local/corex/bin:/usr/local/corex/lib64/python3/dist-packages/bin:$PATH",
        "export LD_LIBRARY_PATH=/usr/local/corex/lib64",
        "export PATH=/usr/local/corex/bin:$PATH",
        "export PYTHONPATH=/usr/local/corex/lib64/python3/dist-package",
        "export PATH=/usr/local/corex/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        # 3) conda 环境
        "source /root/miniconda3/etc/profile.d/conda.sh",
        # 注意必须是双引号：单引号下 $PATH 不会展开，PATH 会变成字面量字符串，
        # /usr/bin 和 /bin 就没了，连 bash 都找不到
        'export PATH="/root/miniconda3/bin:$PATH"',
        "unset __conda_setup",
        "(conda deactivate >/dev/null 2>&1 || true)",
        f"conda activate {CONDA_ENV}",
        # 4) rendezvous 地址
        #    node_rank=0 的 torchrun 要**绑定** MASTER_ADDR:MASTER_PORT 当 rendezvous
        #    服务端，node_rank=1 是客户端去连它。所以 MASTER_ADDR 必须是跑
        #    node_rank=0 的那台机器的地址——node_rank=0 在本机 u62 上跑，
        #    写成 u210 的地址 u62 绑不上（EADDRNOTAVAIL），两边会一直干等。
        #    训练脚本里都是 ${MASTER_ADDR:-...} 的可覆盖写法，export 出去就能生效，
        #    不用去改 experiments/qwen14b 下那些脚本。
        f"export MASTER_ADDR={LOCAL_IP}",
        # 5) HF 镜像 + 清代理，不依赖任何代理工具（clash 等）
        f"export HF_ENDPOINT={HF_ENDPOINT}",
        "unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY NO_PROXY no_proxy",
    ]
)

# 每次实验的产物（tensorboard / profiler trace / step csv）统一收到 <repo>/runtime_log/<case>。
# 不叫 log/ 是为了跟仓库里既有的那个 log/ 区分开——那是早期实验留下的旧结构。
# case 目录名按实际运行顺序递增（四位编号），runtime_log/case.json 记录每个 case 的启动信息。
LOG_DIR = REPO_ROOT / "runtime_log"
CASE_JSON_PATH = LOG_DIR / "case.json"

# torchrun 自己的参数，不算训练脚本的启动参数
TORCHRUN_ONLY_KEYS = {"nproc_per_node", "nnodes", "node_rank", "master_addr", "master_port"}

# 训练入口与并行拓扑（两个节点一致，只有 node_rank 不同）
ENTRYPOINT = QWEN_DIR / "run_qwen14b_polar_dp_pp_tp.py"
MASTER_ADDR = LOCAL_IP  # 必须指向跑 node_rank=0 的机器（本机 u62）
MASTER_PORT = "29500"
NNODES = 2
NPROC_PER_NODE = 16

# 机器之间的互联情况，用于交数据时交代环境。
NETWORK = {
    "interface": "ens1f0",
    "switch": "8 口 100Gb 交换机",
    "theoretical_bandwidth": "12.5 Gb/s",
    "note": "两个节点都通过 ens1f0 接到 8 口 100Gb 交换机，理论带宽 12.5 Gb/s；"
            "test.json 的 rate 档位（1/2/5/10 gbit）就是用 tc 在这条链路上做的限速",
}

GPU_CHECK_TIMEOUT = 60  # GPU 检查命令的超时时间（秒）

QUIET_SSH = True  # 设置为 True 可屏蔽 SSH 远程输出
CMD_DELAY_SECONDS = 3  # 每条命令执行完后的等待时间（秒）

# runtime_log 总开关。
#   True  —— 正常跑：占用 case 编号，脚本/日志归集到 runtime_log/<case>/
#   False —— 冒烟测试：只跑训练看环境通不通，不建 case 目录、不写 case.json、
#            不归集脚本和日志（训练产物落在训练进程自己的 cwd 下）
# 优先级：命令行 --runtime-log on|off  >  环境变量 UDTCA_RUNTIME_LOG  >  默认 True
RUNTIME_LOG_ENABLED = str(os.environ.get("UDTCA_RUNTIME_LOG", "1")).strip().lower() not in {
    "0", "false", "no", "n", "off",
}


def runtime_log_enabled() -> bool:
    """每次调用都读当前值，这样命令行开关能生效（直接 import 变量会拿到旧值）。"""
    return RUNTIME_LOG_ENABLED


def set_runtime_log_enabled(value: bool) -> None:
    global RUNTIME_LOG_ENABLED
    RUNTIME_LOG_ENABLED = bool(value)

sys.path.insert(0, str(UTILS_DIR))
from collect_case_logs import collect as collect_case_logs  # noqa: E402


# ---------------------------------------------------------------- GPU 占用检查

# 空载时每张卡会占住约 68 MiB（驱动本身），超过这个阈值就认为有人在用
GPU_MEMORY_THRESHOLD_MIB = 256

# 在某一台上跑的检查脚本。用 base64 传给远程，免去层层引号转义。
#   退出码 0 = 空闲，1 = 被占用，2 = 找不到 SMI 工具
GPU_CHECK_SNIPPET = r"""
THRESHOLD=__THRESHOLD__

SMI=""
for c in ixsmi /usr/local/bin/ixsmi /usr/local/corex/bin/ixsmi \
         /usr/local/corex-4.4.0/bin/ixsmi /usr/local/corex-4.5.0/bin/ixsmi nvidia-smi; do
  if command -v "$c" >/dev/null 2>&1; then SMI="$c"; break; fi
done
if [ -z "$SMI" ]; then
  echo "NO_SMI"
  exit 2
fi

BUSY=$("$SMI" --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
  | awk -F, -v t="$THRESHOLD" '{
      gsub(/[ \t]/,"",$1); gsub(/[ \t]/,"",$2);
      if ($1 != "" && $2+0 > t) printf "GPU%s(已用%sMiB) ", $1, $2
    }')
APPS=$("$SMI" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
  | sed '/^[[:space:]]*$/d')

if [ -n "$BUSY" ] || [ -n "$APPS" ]; then
  echo "BUSY $SMI"
  [ -n "$BUSY" ] && echo "  超过 ${THRESHOLD}MiB 的卡: $BUSY"
  [ -n "$APPS" ] && echo "  正在跑的进程:" && echo "$APPS" | sed 's/^/    /'
  exit 1
fi

echo "FREE $SMI"
exit 0
"""


def _gpu_check_local() -> tuple:
    """在本机跑 GPU 检查，返回 (ok: bool, 可用: bool, 输出文本)。"""
    snippet = GPU_CHECK_SNIPPET.replace("__THRESHOLD__", str(GPU_MEMORY_THRESHOLD_MIB))
    try:
        proc = subprocess.run(f"/bin/bash -c {_shq(snippet)}",
                              shell=True, capture_output=True, text=True,
                              timeout=GPU_CHECK_TIMEOUT)
    except Exception as e:
        return False, False, f"检查失败: {e}"
    return proc.returncode == 0, proc.returncode != 2, proc.stdout.strip()


def _gpu_check_remote() -> tuple:
    """在远程节点跑 GPU 检查（脚本 base64 过去，避免引号转义）。"""
    snippet = GPU_CHECK_SNIPPET.replace("__THRESHOLD__", str(GPU_MEMORY_THRESHOLD_MIB))
    payload = base64.b64encode(snippet.encode()).decode()
    cmd = f"ssh {REMOTE_HOST} 'echo {payload} | base64 -d | bash'"
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=GPU_CHECK_TIMEOUT)
    except Exception as e:
        return False, False, f"检查失败: {e}"
    return proc.returncode == 0, proc.returncode != 2, proc.stdout.strip()


def _shq(text: str) -> str:
    """把一段文本安全地塞进单引号 shell 字符串。"""
    return "'" + text.replace("'", "'\"'\"'") + "'"


def check_gpus_free() -> bool:
    """运行前检查两个节点的 GPU 有没有被占用；被占用就报错返回 False。

    找不到 SMI 工具时只在告警，不阻塞（这种情况本身也说明环境有问题，会被后续步骤暴露）。
    """
    print("\nChecking GPU availability on both nodes...")
    checks = {LOCAL_HOST: _gpu_check_local(), REMOTE_HOST: _gpu_check_remote()}

    occupied = []
    for node, (ok, available, output) in checks.items():
        if not available:
            print(f"  [警告] {node}: 没找到 ixsmi / nvidia-smi，跳过检查（{output}）")
            continue
        first = output.splitlines()[0] if output else ""
        if ok:
            print(f"  [OK] {node}: GPU 空闲（{first}）")
        else:
            print(f"  [占用] {node}: {first}")
            for line in output.splitlines()[1:]:
                print(f"         {line}")
            occupied.append(node)

    if occupied:
        print(f"\n[ERROR] 以下节点的 GPU 正在被占用: {'、'.join(occupied)}")
        return False
    return True


# ---------------------------------------------------------------- 配置文件


def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def merge_config(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = default.copy()
    merged.update(override)
    return merged


def as_bool(value: Any) -> bool:
    """配置里的布尔值是字符串（"True"/"False"），而 "False" 也是真值，必须显式转换。"""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_only(value: Optional[str]) -> Optional[set]:
    """把 --only "1,2,3" 解析成 {1, 2, 3}；没给就返回 None（表示不筛选）。"""
    if not value:
        return None
    items = {int(token) for token in re.split(r"[,\s]+", value.strip()) if token}
    return items or None


def is_baseline(config: Dict[str, Any], default_config: Optional[Dict[str, Any]] = None) -> bool:
    """判断一条 test.json 配置该走 baseline 还是 POLAR 路径。

    只看这条配置自己有没有写 baseline；没写就退回 default.json 的设定。
    """
    default_config = default_config or {}
    return as_bool(config.get("baseline", default_config.get("baseline", False)))


# ---------------------------------------------------------------- case 编号与 case.json


def load_cases() -> list:
    """读取 runtime_log/case.json 里已有的 case 记录（文件不存在或损坏时返回空列表）。"""
    if not CASE_JSON_PATH.exists():
        return []
    try:
        data = json.loads(CASE_JSON_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    cases = data.get("cases", []) if isinstance(data, dict) else []
    return cases if isinstance(cases, list) else []


def next_case_id(cases: list) -> str:
    """下一个 case 编号：在已有编号上递增，保证跨多次运行也连续。"""
    numbers = [int(c["case"]) for c in cases if str(c.get("case", "")).isdigit()]
    return f"{(max(numbers) + 1) if numbers else 1:04d}"


def save_cases(cases: list) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CASE_JSON_PATH.write_text(
        json.dumps({"cases": cases}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_case_record(record: Dict[str, Any], cases: list) -> None:
    """写入/更新一个 case 记录，每次调用都落盘，避免中途崩溃丢记录。"""
    for i, existing in enumerate(cases):
        if existing.get("case") == record["case"]:
            cases[i] = record
            break
    else:
        cases.append(record)
    save_cases(cases)


def allocate_case() -> tuple:
    """占用下一个 case 编号，返回 (case, case_dir, remote_case_dir, cases)。"""
    cases = load_cases()
    case = next_case_id(cases)
    case_dir = LOG_DIR / case
    case_dir.mkdir(parents=True, exist_ok=True)
    return case, case_dir, f"{REMOTE_BASE_DIR}/runtime_log/{case}", cases


def parse_launch_params_text(text: str) -> Dict[str, str]:
    """从启动脚本文本里解析出 --key value 形式的训练参数。"""
    params = OrderedDict()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        for match in re.finditer(r"--([A-Za-z0-9][A-Za-z0-9-]*)\s+([^\s\\]+)", line):
            key, value = match.group(1), match.group(2).strip('"')
            if key in TORCHRUN_ONLY_KEYS or value.startswith("$"):
                continue
            params[key] = value
    return params


def parse_launch_params(script_path: Path) -> Dict[str, str]:
    """解析本机上已有启动脚本的训练参数。

    baseline 的 0/1_train_*_baseline_*.sh 是仓库里已有的文件，不由 runner 生成，
    所以直接把它们的启动参数读出来记进 case.json，免得手写一份对不上。
    """
    if not script_path.exists():
        return {}
    return parse_launch_params_text(script_path.read_text(encoding="utf-8"))


def fetch_remote_file(remote_path: str, dest: Path) -> bool:
    """把远程节点上的文件拷到本地（只读，失败返回 False）。"""
    try:
        out = subprocess.run(f"scp -q {REMOTE_HOST}:{remote_path} {dest}",
                             shell=True, capture_output=True, text=True, timeout=120)
        if out.returncode == 0:
            dest.chmod(0o755)
            return True
        print(f"   [警告] 拉取远程文件失败 {remote_path}: {out.stderr.strip()}")
        return False
    except Exception as e:
        print(f"   [警告] 拉取远程文件失败 {remote_path}: {e}")
        return False


def remote_rm(remote_path: str) -> None:
    """删掉远程节点上的临时文件（只用于 runner 自己下发过去的那种）。"""
    try:
        subprocess.run(f"ssh {REMOTE_HOST} rm -f {remote_path}",
                       shell=True, capture_output=True, text=True, timeout=30)
    except Exception as e:
        print(f"   [警告] 删除远程文件失败 {remote_path}: {e}")


def fetch_remote_text(remote_path: str) -> str:
    """读远程节点上的文件内容（只读，失败返回空串）。

    node1 实际跑的是 u210 上那份脚本，本地那份可能被改过，所以 node1 的参数
    必须从远程读，不能拿本地文件凑数。
    """
    try:
        out = subprocess.run(f"ssh {REMOTE_HOST} cat {remote_path}",
                             shell=True, capture_output=True, text=True, timeout=30)
        return out.stdout if out.returncode == 0 else ""
    except Exception:
        return ""


# ---------------------------------------------------------------- 命令执行


def execute_command(cmd: str, cwd: Optional[Path] = None, background: bool = False,
                    log_path: Optional[Path] = None) -> subprocess.Popen:
    """在本地执行命令。log_path 非空时该命令的输出既落盘、也实时打到终端。"""
    print(f"[LOCAL {LOCAL_HOST}] Executing: {cmd[:160]}{'...' if len(cmd) > 160 else ''}")

    if log_path is not None:
        # 用 tee 而不是纯重定向：日志照样落盘，同时终端能实时看到训练进度。
        # -o pipefail 保证管道退出码还是训练进程的，不会被 tee 的成功掩盖。
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        bash_cmd = (f"/bin/bash -o pipefail -c {_shq(cmd)}"
                    f" 2>&1 | tee -a {_shq(str(log_path))}")
        print(f"[LOCAL {LOCAL_HOST}] 输出同时写入: {log_path}")
    else:
        bash_cmd = f"/bin/bash -c '{cmd}'"

    process = subprocess.Popen(bash_cmd, shell=True, cwd=cwd)
    process._log_file = None

    if not background:
        process.wait()
        stdout, stderr = process.communicate()
        if stdout:
            for line in stdout.decode("utf-8").splitlines():
                print(f"[LOCAL {LOCAL_HOST}] {line}")
        if stderr:
            for line in stderr.decode("utf-8").splitlines():
                print(f"[LOCAL {LOCAL_HOST}] {line}")
        if CMD_DELAY_SECONDS > 0:
            time.sleep(CMD_DELAY_SECONDS)
    return process


def finalize_process_log(process) -> None:
    """进程结束后关掉重定向用的文件句柄（不关的话日志会留着不落盘）。"""
    log_file = getattr(process, "_log_file", None)
    if log_file is not None:
        try:
            log_file.close()
        except Exception:
            pass
        process._log_file = None


def ssh_execute(
    cmd: str,
    host: str = REMOTE_HOST,
    base_dir: str = REMOTE_BASE_DIR,
    setup_cmds: str = SETUP_CMDS,
    background: bool = False,
    log_path: Optional[str] = None,
) -> subprocess.Popen:
    """在远程节点执行命令。log_path 非空时把该命令的 stdout/stderr 重定向到远程文件。

    脚本用 base64 编码后送过去，不要拼进 `ssh host '... /bin/bash -c "..."'` 这种
    两层引号里：外层双引号会让远程的登录 shell 先把 `$VAR` 展开掉，而这些变量正是
    我们要在脚本里 export 的（比如 $COREX_PATH），展开出来全是空串。
    base64 之后整条命令里没有任何特殊字符，绕开这一整类引号问题。
    """
    script = f"cd {base_dir} && {setup_cmds} && {cmd}"
    if log_path:
        # 远程只写自己的文件、不 tee 回终端：两个节点的 torchrun 输出交织在一起太乱，
        # 终端只看 u62 的（PP 最后一级在 u62 上有 rank 14/15，loss 照样看得到）。
        # 跑完 rsync 会把远程 case 目录整个拉回本地，日志不会丢。
        script = f"mkdir -p $(dirname {log_path}) && ({script}) >> {log_path} 2>&1"
        print(f"[REMOTE {host}] 输出写入远程文件（不回显）: {log_path}")
    payload = base64.b64encode(script.encode()).decode()
    full_cmd = f"ssh {host} 'echo {payload} | base64 -d | bash'"
    print(f"[REMOTE {host}] Executing:\n{script}")

    process = subprocess.Popen(full_cmd, shell=True, start_new_session=True)
    if not background:
        process.wait()
        if not QUIET_SSH:
            stdout, stderr = process.communicate()
            if stdout:
                for line in stdout.decode("utf-8").splitlines():
                    print(f"[REMOTE {host}] {line}")
            if stderr:
                for line in stderr.decode("utf-8").splitlines():
                    print(f"[REMOTE {host}] {line}")
        if CMD_DELAY_SECONDS > 0:
            time.sleep(CMD_DELAY_SECONDS)
    return process


# ---------------------------------------------------------------- traffic control


def generate_traffic_control_script(output_path: Path, rate: str) -> None:
    """生成 tc 网络限速脚本。

    只用了 RATE；BURST / LATENCY / DELAY 是保留的可调项，当前 DELAY=0 表示不模拟时延。
    """
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


# ---------------------------------------------------------------- 中断清理


# 只匹配本项目自己的训练进程：解释器必须是 trj-test 的 python3.10，
# 且命令行里带训练入口脚本或 torchrun。
# 绝不能按进程名模糊匹配（pkill -f torchrun）——这是共享机器，会误杀别人的任务。
_TRAIN_KILL_SNIPPET = r'''
killed=0
for pid in $(ls /proc 2>/dev/null); do
  case "$pid" in ''|*[!0-9]*) continue;; esac
  [ "$pid" = "$$" ] && continue
  cmd=$(tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null) || continue
  case "$cmd" in
    /root/miniconda3/envs/trj-test/bin/python3.10*)
      case "$cmd" in
        *run_qwen14b_polar_dp_pp_tp.py*|*/bin/torchrun*)
          kill "$pid" 2>/dev/null && killed=$((killed + 1));;
      esac;;
  esac
done
echo "$killed"
'''


def kill_local_training(timeout: int = 60) -> None:
    """杀掉本机残留的训练进程（本地 torchrun 的 child 不一定随 SIGTERM 一起退）。"""
    try:
        proc = subprocess.run(f"/bin/bash -c {_shq(_TRAIN_KILL_SNIPPET)}",
                              shell=True, capture_output=True, text=True, timeout=timeout)
        print(f"  [LOCAL] 清理训练进程: {proc.stdout.strip()} 个")
    except Exception as e:
        print(f"  [LOCAL] 清理训练进程失败: {e}")


def kill_remote_training(timeout: int = 60) -> None:
    """杀掉远程节点上的训练进程。

    中断 runner 时 ssh 链路断了，但 u210 上的 torchrun 不会跟着死，会一直占着卡。
    这里 ssh 过去按「解释器路径 + 脚本名」精确匹配后 kill。
    """
    payload = base64.b64encode(_TRAIN_KILL_SNIPPET.encode()).decode()
    cmd = f"ssh {REMOTE_HOST} 'echo {payload} | base64 -d | bash'"
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        print(f"  [REMOTE {REMOTE_HOST}] 清理训练进程: {proc.stdout.strip()} 个")
    except Exception as e:
        print(f"  [REMOTE {REMOTE_HOST}] 清理训练进程失败: {e}")


def stop_traffic_control(tc_script_path) -> None:
    """撤掉限速。中断时如果不撤，会给共享机器留下一个限速规则。"""
    if not tc_script_path:
        return
    tc_script_path = Path(tc_script_path)
    if not tc_script_path.exists():
        return
    try:
        subprocess.run(f"/bin/bash -c {_shq(f'bash {tc_script_path} stop')}",
                       shell=True, capture_output=True, text=True, timeout=30)
        print("  [LOCAL] 已撤销 tc 限速")
    except Exception as e:
        print(f"  [LOCAL] 撤销 tc 限速失败: {e}")
    try:
        subprocess.run(
            f"ssh {REMOTE_HOST} 'bash {REMOTE_QWEN_DIR}/{tc_script_path.name} stop'",
            shell=True, capture_output=True, text=True, timeout=30)
        print(f"  [REMOTE {REMOTE_HOST}] 已撤销 tc 限速")
    except Exception as e:
        print(f"  [REMOTE {REMOTE_HOST}] 撤销 tc 限速失败: {e}")


# ---------------------------------------------------------------- 完整启动脚本 / 环境说明


def _render_torchrun_block(params, node_rank: int, indent: str = "  ") -> str:
    """把一条 node 的 torchrun 命令渲染成可独立执行的完整命令。

    输出 tee 到 case 目录里的 train_node{rank}.log：既落盘、也能在终端看实时进度
    （脚本开头已经 `set -euo pipefail`，所以管道退出码还是 torchrun 的）。
    """
    lines = [
        f"{indent}torchrun \\",
        f"{indent}  --nproc_per_node={NPROC_PER_NODE} \\",
        f"{indent}  --nnodes={NNODES} \\",
        f"{indent}  --node_rank={node_rank} \\",
        f'{indent}  --master_addr="{MASTER_ADDR}" \\',
        f'{indent}  --master_port="{MASTER_PORT}" \\',
        f'{indent}  "${{ENTRYPOINT}}" \\',
    ]
    items = list(params.items())
    for i, (key, value) in enumerate(items):
        rendered = str(value).lower() if isinstance(value, bool) else value
        if i < len(items) - 1:
            tail = " \\"
        else:
            # 重定向必须接在同一行末尾：单独起一行的话上一条命令已经结束了，
            # 会变成 torchrun 不重定向 + 多出一条空的纯重定向命令
            tail = (f' 2>&1 | tee -a "${{CASE_DIR}}/train_node{node_rank}.log"')
        lines.append(f"{indent}  --{key} {rendered}{tail}")
    return "\n".join(lines)


def write_launch_script(case_dir: Path, case: str, kind: str, params_by_node: Dict[int, Any],
                        tc_script_name: str, rate: str, started_at: str) -> Path:
    """在 case 目录里写一份完整、自包含、可独立复现的启动脚本。

    不依赖 udtca-config 的 runner，也不依赖生成出来的 0/1_train_*.sh：
    环境准备、限速说明、两个节点的完整 torchrun 命令都在里面。
    """
    env_block = "\n".join(SETUP_CMDS.split(" && "))

    branches = []
    for i, rank in enumerate(sorted(params_by_node)):
        label = f"node {rank}"
        keyword = "if" if i == 0 else "elif"
        branches.append(
            f'{keyword} [ "${{NODE_RANK}}" = "{rank}" ]; then\n'
            f"  # {label}\n"
            f"{_render_torchrun_block(params_by_node[rank], rank)}\n"
        )
    branches.append('else\n  echo "NODE_RANK 只能是 0 或 1" >&2\n  exit 1\nfi')

    content = f"""\
#!/usr/bin/env bash
# =============================================================================
# case {case}   （{kind}）
# 生成时间: {started_at}
#
# 这是一份完整、自包含的启动脚本，用于独立复现本次实验，不依赖 udtca-config。
#
# 用法:
#   node 0 (u62, 10.31.10.62) : bash launch.sh 0
#   node 1 (u210, 10.31.10.210): bash launch.sh 1
#
# 网络限速（可选，需要 root，两个节点都要执行）:
#   sudo bash {tc_script_name} start     # RATE="{rate}"
#   sudo bash {tc_script_name} stop
#
# tensorboard / profiler trace / step csv 都会落在本文件所在目录：
#   {case_dir}
# =============================================================================
set -euo pipefail

NODE_RANK="${{1:?用法: bash launch.sh <node_rank>  (0 在 u62, 1 在 u210)}}"
CASE_DIR="{case_dir}"
ENTRYPOINT="{ENTRYPOINT}"

# ---- 运行环境：shell 初始化 -> Corex/CUDA 变量 -> conda 环境 -> HF 镜像 ----
{env_block}

# 训练侧把 tb/trace 写成相对路径 ./log/...，切到 case 目录就落在本目录下
mkdir -p "${{CASE_DIR}}"
cd "${{CASE_DIR}}"

{chr(10).join(branches)}
"""
    case_dir.mkdir(parents=True, exist_ok=True)
    output_path = case_dir / "launch.sh"
    output_path.write_text(content, encoding="utf-8")
    output_path.chmod(0o755)
    return output_path


def _probe(cmd: str) -> str:
    """跑一条只读探测命令，失败就返回空串。"""
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def write_environment_info() -> Path:
    """写 runtime_log/environment.json，交代跑这批实验的机器与网络环境。"""
    info = {
        "generated_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "nodes": {
            "node0": {"hostname": "u62", "ip": "10.31.10.62", "role": "本地节点，node_rank=0"},
            "node1": {"hostname": "u210", "ip": "10.31.10.210",
                      "role": "远程节点，node_rank=1", "ssh": REMOTE_HOST},
        },
        "repo_path": str(REPO_ROOT),          # 两个节点上的路径相同
        "conda_env": CONDA_ENV,
        "python": _probe(f"/root/miniconda3/envs/{CONDA_ENV}/bin/python -V"),
        "torch": _probe(f"/root/miniconda3/envs/{CONDA_ENV}/bin/python -c "
                        "'import torch;print(torch.__version__)'"),
        "network": NETWORK,
        "parallelism": {
            "nnodes": NNODES,
            "nproc_per_node": NPROC_PER_NODE,
            "dp": 2, "pp": 8, "tp": 2,
        },
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "dataset": "HuggingFaceFW/fineweb",
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / "environment.json"
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------- 日志收集


def finish_case(record: Dict[str, Any], cases: list, case_dir: Path,
                remote_case_dir: str, local_scripts: list, remote_scripts: list,
                exit_code: Optional[int]) -> None:
    """跑完之后收尾：留档两节点各自的启动脚本、拉远程日志、拍平、回写 case.json。

    local_scripts : [(Path, move)]             本机 node0 实际用的文件
    remote_scripts: [(文件名, 远程路径, move)]  远程 node1 实际用的文件，从远程拉回来

    move=True 用于 runner 自己生成/下发的一次性脚本（归集后原地不留）；
    move=False 用于仓库里已有的受版本管理的脚本（必须原地保留，移动会让下次跑不了）。
    script_log 不再留任何东西，脚本只存在于 case 目录里。

    node1 的脚本必须从远程拉：node1 跑的是 u210 上那份，本地同名文件可能是改过的旧版
    （baseline 那两份本地/远程内容确实不一样）。
    """
    print(f"\nCollecting logs into {case_dir} ...")

    print("   Archiving node 0 scripts from local...")
    for src, move in local_scripts:
        src = Path(src)
        dest = case_dir / src.name
        try:
            if move:
                shutil.move(str(src), str(dest))
            else:
                shutil.copy2(str(src), str(dest))
            print(f"     {src.name} ({'moved' if move else 'copied'})")
        except OSError as e:
            print(f"   [警告] 归档 {src.name} 到 case 目录失败: {e}")

    print("   Archiving node 1 scripts from remote...")
    for name, remote_path, move in remote_scripts:
        if fetch_remote_file(remote_path, case_dir / name):
            print(f"     {name} (copied from remote)")
            if move:
                remote_rm(remote_path)

    print("   Pulling node 1 logs from remote...")
    rsync_cmd = f"rsync -a --ignore-existing {REMOTE_HOST}:{remote_case_dir}/ {case_dir}/"
    try:
        result = subprocess.run(rsync_cmd, shell=True)
        if result.returncode != 0:
            print(f"   [警告] 远程日志同步失败 (rsync 退出码 {result.returncode})，"
                  f"case {record['case']} 只包含本地节点的日志")
    except Exception as e:
        print(f"   [警告] 远程日志同步失败: {e}")

    print("   Flattening nested log directories...")
    moved = collect_case_logs(case_dir)
    print("   Collected: " + ", ".join(f"{k}={v}" for k, v in moved.items()))

    record["finished_at"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    record["exit_code"] = exit_code
    write_case_record(record, cases)
