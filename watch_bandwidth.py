#!/usr/bin/env python3
"""实时监测 u62 / u210 两机之间的网络带宽。

读两个节点上 ens1f0 的 rx/tx 字节计数器，按间隔取差值算速率。只读，不干扰训练。

用法：
    python watch_bandwidth.py                 # 每秒刷一行
    python watch_bandwidth.py -i 5            # 5 秒一个采样
    python watch_bandwidth.py --once          # 只出一行（方便塞进别的脚本）
    python watch_bandwidth.py --iface ens1f0

注意：读的是网卡总流量，不只是训练那部分（比如构建数据集时下 HF 也算进去）。

停：Ctrl-C
"""

import argparse
import re
import subprocess
import sys
import threading
import time
from datetime import datetime

LOCAL_HOST = "u62"
REMOTE_HOST = "u210"          # ~/.ssh/config 里的别名
DEFAULT_IFACE = "ens1f0"
# 交换机侧给每台机器的理论带宽
THEORETICAL_GBPS = 12.5


def hr(bits_per_sec: float) -> str:
    """把 bit/s 格式化成人看的。"""
    if bits_per_sec >= 1e9:
        return f"{bits_per_sec / 1e9:.2f} Gbps"
    if bits_per_sec >= 1e6:
        return f"{bits_per_sec / 1e6:.1f} Mbps"
    if bits_per_sec >= 1e3:
        return f"{bits_per_sec / 1e3:.1f} kbps"
    return f"{bits_per_sec:.0f} bps"


def read_counters_local(iface: str):
    """本机网卡的 (rx_bytes, tx_bytes)。"""
    base = f"/sys/class/net/{iface}/statistics"
    try:
        with open(f"{base}/rx_bytes") as f:
            rx = int(f.read().strip())
        with open(f"{base}/tx_bytes") as f:
            tx = int(f.read().strip())
        return rx, tx
    except OSError:
        return None


class RemoteSampler:
    """起一条常驻 ssh，让远端自己按秒吐计数器，避免每秒都新建连接。"""

    def __init__(self, host: str, iface: str):
        self.host, self.iface = host, iface
        self.latest = None
        self.proc = None
        self._stop = threading.Event()

    def start(self) -> bool:
        cmd = (
            f"ssh {self.host} '"
            f"while :; do "
            f"printf \"%s %s\\n\" "
            f"\"$(cat /sys/class/net/{self.iface}/statistics/rx_bytes)\" "
            f"\"$(cat /sys/class/net/{self.iface}/statistics/tx_bytes)\"; "
            f"sleep 1; "
            f"done'"
        )
        try:
            self.proc = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
            )
        except Exception as e:
            print(f"[警告] 连不上 {self.host}: {e}")
            return False
        threading.Thread(target=self._pump, daemon=True).start()
        # 等第一条数据
        for _ in range(50):
            if self.latest is not None:
                return True
            if self.proc.poll() is not None:
                print(f"[警告] {self.host} 上的采样进程退出了")
                return False
            time.sleep(0.1)
        print(f"[警告] {self.host} 没有返回数据")
        return False

    def _pump(self):
        for line in self.proc.stdout:
            parts = line.split()
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                self.latest = (int(parts[0]), int(parts[1]))

    def stop(self):
        self._stop.set()
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()


def current_rate_limit(iface: str) -> str:
    """看 ens1f0 当前有没有被 tc 限速，返回限速值或 '不限速'。"""
    def parse(out: str) -> str:
        m = re.search(r"htb rate (\S+)", out)
        return m.group(1) if m else "不限速"

    try:
        out = subprocess.run(f"tc qdisc show dev {iface}", shell=True,
                             capture_output=True, text=True, timeout=10).stdout
        local = parse(out)
    except Exception:
        local = "?"
    try:
        out = subprocess.run(
            f"ssh {REMOTE_HOST} 'tc qdisc show dev {iface}'", shell=True,
            capture_output=True, text=True, timeout=10).stdout
        remote = parse(out)
    except Exception:
        remote = "?"
    if local == remote:
        return local
    return f"u62={local} u210={remote}"


def parse_args():
    p = argparse.ArgumentParser(description="实时监测两机之间的带宽")
    p.add_argument("-i", "--interval", type=float, default=1.0, help="采样间隔秒（默认 1）")
    p.add_argument("--iface", default=DEFAULT_IFACE, help=f"网卡名（默认 {DEFAULT_IFACE}）")
    p.add_argument("--once", action="store_true", help="只出一行就退出")
    return p.parse_args()


def fmt_line(prevs, nows, interval, limit):
    """prevs/nows: {'62': (rx,tx), 'u210': (rx,tx)}"""
    cells = []
    total = 0.0
    for node in (LOCAL_HOST, REMOTE_HOST):
        p, n = prevs.get(node), nows.get(node)
        if p is None or n is None:
            cells += ["—", "—"]
            continue
        rx = (n[0] - p[0]) * 8 / interval
        tx = (n[1] - p[1]) * 8 / interval
        total += rx + tx
        cells += [hr(rx), hr(tx)]
    ts = datetime.now().strftime("%H:%M:%S")
    return (f"{ts}  {LOCAL_HOST:>4} ↓{cells[0]:>11} ↑{cells[1]:>11}   "
            f"{REMOTE_HOST:>4} ↓{cells[2]:>11} ↑{cells[3]:>11}   "
            f"双向合计 {hr(total):>11}   限速 {limit}")


def main() -> None:
    args = parse_args()
    iface = args.iface

    if read_counters_local(iface) is None:
        print(f"[ERROR] 本机没有网卡 {iface}")
        sys.exit(1)

    remote = RemoteSampler(REMOTE_HOST, iface)
    if not remote.start():
        print(f"[ERROR] 无法从 {REMOTE_HOST} 采集数据，退出。")
        sys.exit(1)

    limit = current_rate_limit(iface)

    if not args.once:
        print(f"监测 {LOCAL_HOST}(本机) <-> {REMOTE_HOST} 的 {iface}，"
              f"每 {args.interval:g} 秒一次，Ctrl-C 退出")
        print(f"交换机侧理论带宽 {THEORETICAL_GBPS} Gb/s，当前限速 {limit}")
        print(f"{'时间':>8}  {'u62 收/发':>26}   {'u210 收/发':>26}   {'':>13}   ")
        print("-" * 108)

    prevs = {LOCAL_HOST: read_counters_local(iface), REMOTE_HOST: remote.latest}
    try:
        while True:
            time.sleep(args.interval)
            nows = {LOCAL_HOST: read_counters_local(iface), REMOTE_HOST: remote.latest}
            if nows[LOCAL_HOST] is None or nows[REMOTE_HOST] is None:
                continue
            print(fmt_line(prevs, nows, args.interval, limit), flush=True)
            prevs = nows
            if args.once:
                break
    except KeyboardInterrupt:
        print("\n停止监测。")
    finally:
        remote.stop()


if __name__ == "__main__":
    main()
