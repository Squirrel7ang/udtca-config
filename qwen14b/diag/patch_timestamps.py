#!/usr/bin/env python3
"""给自研诊断输出加"墙上时钟"，便于和 dmesg / NCCL 日志按时间对齐。

只改打印前缀，不动任何逻辑：
  hooks.py    _dbg():            `[dbg  694.936s ...]` -> `[dbg 10-31 08:36:35.123  694.936s ...]`
              _polar_hook_debug(): 加同样的时钟
  wrapper.py  [comm-watchdog]   各条打印带上时钟

用法: python3 patch_timestamps.py <hooks.py> <wrapper.py>
可重复执行（已打过的锚点不再匹配，脚本会报 already patched）。
"""
import sys

CLOCK_HELPER = '''
def _wall_clock() -> str:
    """墙上时钟（本地时区，毫秒），用来和 dmesg / NCCL 日志对齐。"""
    now = time.time()
    return time.strftime("%m-%d %H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 1000):03d}"

'''


def patch(path, pairs, helper_anchor=None, helper=CLOCK_HELPER):
    src = open(path, encoding="utf-8").read()
    before = len(src)
    changed = 0
    for old, new in pairs:
        if src.count(old) == 1:
            src = src.replace(old, new)
            changed += 1
        elif src.count(old) == 0:
            print(f"  skip (already patched or missing): {old.splitlines()[0][:70]}")
        else:
            raise SystemExit(f"anchor not unique in {path}: {old[:60]!r}")
    # 注意：判据必须是"有没有定义"，不能是"有没有出现 _wall_clock"——上面刚把
    # 调用点替换进去，src 里早就有这个名字了，写成后者会永远跳过定义、
    # 第一次 _dbg() 就 NameError（2026-10-31 polar_dense 的崩溃就是它）。
    if helper_anchor and "def _wall_clock" not in src:
        assert src.count(helper_anchor) == 1, f"helper anchor bad in {path}"
        src = src.replace(helper_anchor, helper + helper_anchor)
    if changed:
        open(path, "w", encoding="utf-8").write(src)
    print(f"{path}: {changed} site(s) patched, {before} -> {len(src)} bytes")


def main():
    hooks, wrapper = sys.argv[1], sys.argv[2]

    patch(
        hooks,
        [
            (
                '''def _dbg(msg: str, site: str = "") -> None:
    print(
        f"[dbg {time.monotonic() - _DBG_T0:9.3f}s rank={_dbg_rank_()} "''',
                '''def _dbg(msg: str, site: str = "") -> None:
    print(
        f"[dbg {_wall_clock()} {time.monotonic() - _DBG_T0:9.3f}s rank={_dbg_rank_()} "''',
            ),
            (
                'f"[polar-hook-debug rank={rank} t={time.time():.6f}] {message}",',
                'f"[polar-hook-debug {_wall_clock()} rank={rank} t={time.time():.6f}] {message}",',
            ),
        ],
        helper_anchor='def _polar_hook_debug(message: str) -> None:',
    )

    patch(
        wrapper,
        [
            (
                'print(f"[comm-watchdog] comm capture failed: {exc!r}", flush=True)',
                'print(f"[comm-watchdog] {_wall_clock()} comm capture failed: {exc!r}", flush=True)',
            ),
            (
                'print("[comm-watchdog] nccl_revoke unavailable; not exiting", flush=True)',
                'print(f"[comm-watchdog] {_wall_clock()} nccl_revoke unavailable; not exiting", flush=True)',
            ),
            (
                'print(f"[comm-watchdog] shutdown failed: {exc!r}", flush=True)',
                'print(f"[comm-watchdog] {_wall_clock()} shutdown failed: {exc!r}", flush=True)',
            ),
        ],
        helper_anchor="class _CommWatchdog",
    )


if __name__ == "__main__":
    main()
