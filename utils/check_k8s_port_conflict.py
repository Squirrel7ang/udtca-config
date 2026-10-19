#!/usr/bin/env python3
"""检查 kube-proxy 的 REJECT 规则会不会挡住 NCCL 随机分配的端口。

背景
----
这两台训练机同时是 k8s 节点，kube-proxy 会给"没有后端的服务端口"生成一条：

    -A KUBE-EXTERNAL-SERVICES -p tcp -m addrtype --dst-type LOCAL \
       -m tcp --dport 40009 -j REJECT --reject-with icmp-port-unreachable

`--dst-type LOCAL` 表示**目标是本机任意地址**都命中，所以即使真有进程在
这个端口上 listen，对端连过来也会被内核回 RST（实测：监听 socket 存在，
connect 仍然 `Connection refused`）。

NCCL 的 bootstrap / P2P 监听口是 `bind(port=0)` 由内核在
`net.ipv4.ip_local_port_range`（本机 32768–60999）里随机分配的。一旦落进被
REJECT 的端口，对端连不进来，`socketPollConnect` 会一直重试（第 n 次睡 n 秒），
训练就卡在初始化、既不报错也不往下走——这就是之前 4-case 偶发卡死的根因。

本脚本做的事
------------
1. 读 `iptables-save`，找出所有「REJECT + dst-type LOCAL」的 dport；
2. 只保留落在临时端口范围里的那些（范围外的端口内核本来就不会分配给我们）；
3. 再读 k8s 的 NodePort 池（kube-apiserver 的 `--service-node-port-range`），
   把与临时端口范围重叠的部分也算进来——现在没被 REJECT 的 NodePort 之后随时
   可能因为新 service 变成 REJECT；
4. 和 `net.ipv4.ip_local_reserved_ports` 对比，列出还没被排除的端口；
5. `--fix` 时把冲突端口并进 `ip_local_reserved_ports`（内核就不会再自动分配给
   任何 socket，不影响别人显式 bind 这些端口，也不动任何 kube 规则）。

用法
----
    python3 check_k8s_port_conflict.py            # 只检查；有冲突退出码 1
    python3 check_k8s_port_conflict.py --fix      # 检查并修复（需要 root）
    python3 check_k8s_port_conflict.py --json     # 机器可读输出
    python3 check_k8s_port_conflict.py --probe    # 额外采样 4000 次 bind(0) 验证

返回码：0 = 没有冲突（或已修复）；1 = 仍有冲突；2 = 环境问题（非 root / 没 iptables）
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PORT_RANGE_FILE = "/proc/sys/net/ipv4/ip_local_port_range"
RESERVED_FILE = "/proc/sys/net/ipv4/ip_local_reserved_ports"


# ------------------------------------------------------------------ 基础读取

def _read(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return None


def read_port_range() -> Tuple[int, int]:
    txt = _read(PORT_RANGE_FILE) or "32768\t60999"
    parts = txt.split()
    return int(parts[0]), int(parts[1])


def parse_ports(spec: str) -> List[int]:
    """把 `1-5,10` / `1:5,10` / `40001-40016` 这种写法展开成端口列表。"""
    out: List[int] = []
    for chunk in spec.replace(":", "-").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return out


def compact_ports(ports: List[int]) -> str:
    """把端口列表压缩成 sysctl 认的区间写法，例：40001-40016,50007。"""
    ports = sorted(set(ports))
    if not ports:
        return ""
    out, start, prev = [], ports[0], ports[0]
    for p in ports[1:]:
        if p == prev + 1:
            prev = p
            continue
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = p
    out.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(out)


def compact_ranges(ranges: List[Tuple[int, int]]) -> str:
    return ",".join(f"{a}-{b}" for a, b in ranges)


def read_reserved() -> List[int]:
    txt = _read(RESERVED_FILE) or ""
    return parse_ports(txt)


# ------------------------------------------------------------------ NodePort 范围
#
# kube-apiserver 的 `--service-node-port-range=40000-45000` 决定了 k8s 会占用
# 哪一段端口。只要这段和临时端口范围（32768-60999）有交集，k8s 随时可能给新
# service 分配一个落在交集里的 NodePort，并让 kube-proxy 下发 REJECT 规则。
# 所以除了"当前已经被 REJECT 的端口"，整段重叠区间都应该预留。

_APISERVER_HINTS = (
    "/etc/kubernetes/manifests",
    "/etc/kubernetes",
    "/etc/default",
)


def find_nodeport_ranges() -> Tuple[List[Tuple[int, int]], List[str]]:
    """返回 (声明的 NodePort 区间列表, 来源文件列表)。"""
    ranges: List[Tuple[int, int]] = []
    sources: List[str] = []
    seen = set()
    for hint in _APISERVER_HINTS:
        root = Path(hint)
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else list(root.glob("*.yaml"))
        candidates += list(root.glob("*apiserver*"))
        for path in candidates:
            if not path.is_file() or str(path) in seen:
                continue
            seen.add(str(path))
            try:
                if path.stat().st_size > 2 * 1024 * 1024:
                    continue
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            found = re.findall(r"service-node-port-range[=\s\"']+(\d+)-(\d+)", text)
            if found:
                sources.append(str(path))
                ranges.extend((int(a), int(b)) for a, b in found)
    return sorted(set(ranges)), sorted(set(sources))


# ------------------------------------------------------------------ iptables

def find_rejected_ports() -> Tuple[List[int], Optional[str]]:
    """返回 (被 REJECT 且 dst-type LOCAL 的端口列表, 错误信息)。"""
    try:
        proc = subprocess.run(["iptables-save"], capture_output=True,
                              text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"执行 iptables-save 失败: {exc}"
    if proc.returncode != 0:
        return [], (proc.stderr.strip() or "iptables-save 返回非 0（需要 root？）")

    ports: List[int] = []
    for line in proc.stdout.splitlines():
        if not line.startswith("-A "):
            continue
        if "-j REJECT" not in line:
            continue
        # 只关心"目标是本机任意地址"的那种；限定具体虚 IP 的规则不影响我们
        if "dst-type LOCAL" not in line and "dst-type local" not in line.lower():
            continue
        m = re.search(r"--dport\s+(\d+)(?:[:-](\d+))?", line)
        if not m:
            continue
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        ports.extend(range(lo, hi + 1))
    return sorted(set(ports)), None


def probe_allocation(rejected: List[int], n: int = 4000) -> Tuple[Optional[List[int]], Optional[str]]:
    """真实回归测试：连续做 n 次 bind(0)，看内核会不会把被 REJECT 的端口发出来。

    这才是我们关心的性质——`ip_local_reserved_ports` 生效后，命中数应该是 0。
    （直接连这些端口一定不通：iptables 的 REJECT 仍然在，我们只是让内核不分配它们。）
    """
    try:                                  # 一次开 n 个 socket，先抬高 fd 上限
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = n + 200
        if soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(want, hard), hard))
    except Exception:                     # noqa: BLE001 - 抬不动就算了，下面会报错
        pass

    banned = set(rejected)
    socks: List[socket.socket] = []
    hits: List[int] = []
    err: Optional[str] = None
    try:
        for _ in range(n):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("0.0.0.0", 0))
            port = sock.getsockname()[1]
            socks.append(sock)
            if port in banned:
                hits.append(port)
    except OSError as exc:
        err = f"{exc}（fd 上限不够？试 ulimit -n {n + 500}）"
    finally:
        for sock in socks:
            sock.close()
    return sorted(set(hits)), err


# ------------------------------------------------------------------ 修复

def apply_reserved(ports: List[int]) -> Tuple[bool, str]:
    spec = compact_ports(ports)
    try:
        proc = subprocess.run(
            ["sysctl", "-w", f"net.ipv4.ip_local_reserved_ports={spec}"],
            capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    if proc.returncode != 0:
        return False, (proc.stderr.strip() or proc.stdout.strip())
    return True, proc.stdout.strip()


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description="检测 kube-proxy REJECT 规则与 NCCL 临时端口的冲突")
    ap.add_argument("--fix", action="store_true",
                    help="把冲突端口写进 net.ipv4.ip_local_reserved_ports（需要 root）")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--probe", action="store_true", help="对冲突端口做真实 bind+connect 验证")
    ap.add_argument("--quiet", action="store_true", help="只在有问题时打印")
    args = ap.parse_args()

    lo, hi = read_port_range()
    rejected, err = find_rejected_ports()
    in_range = [p for p in rejected if lo <= p <= hi]

    # k8s 的 NodePort 池和临时端口范围的重叠部分也要预留：现在没被 REJECT 不代表
    # 以后不会——新 service 一分配就多出一条 REJECT 规则。
    nodeport_ranges, nodeport_sources = find_nodeport_ranges()
    nodeport_in_range: List[int] = []
    for nlo, nhi in nodeport_ranges:
        a, b = max(nlo, lo), min(nhi, hi)
        if a <= b:
            nodeport_in_range.extend(range(a, b + 1))
    nodeport_in_range = sorted(set(nodeport_in_range))

    risky = sorted(set(in_range) | set(nodeport_in_range))
    reserved_before = read_reserved()
    reserved_set = set(reserved_before)
    unresolved = [p for p in risky if p not in reserved_set]

    fixed = None
    if args.fix and unresolved:
        ok, msg = apply_reserved(reserved_before + unresolved)
        fixed = {"ok": ok, "message": msg}
        if ok:
            unresolved = []

    probes: Dict[str, object] = {}
    if args.probe:
        hits, probe_err = probe_allocation(risky)
        probes = {"bind_zero_tries": 4000, "hits": hits, "error": probe_err}

    result = {
        "host": socket.gethostname(),
        "local_port_range": [lo, hi],
        "rejected_ports_total": len(rejected),
        "rejected_ports_in_range": in_range,
        "nodeport_ranges": [list(r) for r in nodeport_ranges],
        "nodeport_sources": nodeport_sources,
        "nodeport_overlap": nodeport_in_range,
        "reserved_ports": compact_ports(read_reserved() if not args.fix or fixed is None
                                        else reserved_before + risky),
        "unresolved": unresolved,
        "fixed": fixed,
        "probe": probes,
        "error": err,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    elif not args.quiet or unresolved or err:
        host = result["host"]
        print(f"[port-check] {host}: 临时端口范围 {lo}-{hi}")
        if err:
            print(f"[port-check] {host}: 无法读取 iptables（{err}），跳过检查")
            return 2
        print(f"[port-check] {host}: 被 kube-proxy REJECT 的端口 {len(rejected)} 个，"
              f"落在临时端口范围里的 {len(in_range)} 个"
              + (f": {compact_ports(in_range)}" if in_range else ""))
        if nodeport_ranges:
            print(f"[port-check] {host}: k8s NodePort 池 {compact_ranges(nodeport_ranges)}"
                  + (f"（来自 {'/'.join(nodeport_sources)}）" if nodeport_sources else "")
                  + (f"，与临时端口重叠 {compact_ports(nodeport_in_range)}"
                     if nodeport_in_range else "，与临时端口不重叠"))
        if fixed is not None and fixed["ok"]:
            print(f"[port-check] {host}: 已写入 ip_local_reserved_ports -> "
                  f"{compact_ports(read_reserved())}")
        if unresolved:
            print(f"[port-check] {host}: 仍有冲突端口未被排除: {compact_ports(unresolved)}\n"
                  f"             修复: sudo sysctl -w "
                  f"net.ipv4.ip_local_reserved_ports="
                  f"{compact_ports(read_reserved() + unresolved)}")
        elif risky:
            print(f"[port-check] {host}: OK（风险端口都已从自动分配里排除）")
        else:
            print(f"[port-check] {host}: OK（没有落在临时端口范围内的 REJECT 端口）")
        if probes:
            if probes.get("error"):
                print(f"[port-check] {host}: bind(0) 采样失败: {probes['error']}")
            else:
                hits = probes.get("hits") or []
                print(f"[port-check] {host}: bind(0) × {probes['bind_zero_tries']}，"
                      f"落到被 REJECT 端口的次数 = {len(hits)}"
                      + (f"（{compact_ports(list(hits))}）" if hits else ""))

    return 1 if unresolved else 0


if __name__ == "__main__":
    sys.exit(main())
