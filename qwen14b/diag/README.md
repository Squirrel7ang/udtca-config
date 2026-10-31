# diag —— 跨机 Qwen14B 排查用的小工具

都是**只读或可回滚**的现场工具，跑之前把源文件备份好（脚本自己会说明备份方式）。

## patch_timestamps.py —— 给自研诊断输出加墙上时钟

```bash
# 在容器内、仓库根执行（两机都要打，打完 md5 应一致）
python3 udtca-config/qwen14b/diag/patch_timestamps.py \
    polar-sgd/src/psgd/parallelism/polar/hooks.py \
    polar-sgd/src/psgd/parallelism/polar/wrapper.py
```

改的是什么：只改打印前缀，不动逻辑。

- `hooks.py` 的 `_dbg()`：`[dbg  694.936s ...]` → `[dbg 10-31 08:36:35.123  694.936s ...]`
- `hooks.py` 的 `_polar_hook_debug()`：同样加时钟

为什么：原来的 `[dbg ...]` 只有进程内的相对秒数，机器卡死之后没法跟 `dmesg` / NCCL 日志 /
内核 XID 对上时间。加了墙上时钟以后，内核里冒出来的错误能直接对应"当时代码走到哪一行"。

可重复执行，已打过的地方会打印 `skip`。

## patch_mask_noitem.py —— 彻底去掉 `layer_attention_mask` 里的 `.item()`

```bash
python3 udtca-config/qwen14b/diag/patch_mask_noitem.py \
    experiments/qwen14b/run_qwen14b_polar_dp_pp_tp.py
```

`run_qwen14b_polar_dp_pp_tp.py:699` 每个 forward chunk 都会做一次
`bool(attention_mask.all().item())`，而数据侧的 mask 是 `torch.ones_like(input_ids)` 造出来的
（同文件 :223），2D mask 恒为全 1。这个补丁按 `dim` 判定，**运行时零 D2H 读回**，语义不变。

用途：把"`.item()` 全流同步"这个变量单独隔离掉做对照。

## patch_mask_nosync.py —— 保守版：按 shape 只判定一次

同上，但保留 `.item()`，只对同一 shape 首次出现时读回一次、结果缓存。
用来区分"**每个 chunk 都同步**"和"**完全不读回**"两种情况。

回滚：两个补丁都只改这一处函数，`git checkout -- experiments/qwen14b/run_qwen14b_polar_dp_pp_tp.py`
即可（或用自己的备份）。

## 已有的现场工具

- `capture_stall.sh`（上一级目录）：卡死现场抓取（py-spy 栈 / NCCL 日志尾 / ixsmi / ps），只读。
- `diag_health_16.sh`、`diag_comm_16.sh`：16 卡健康与通信自检。
- `diag_nccl_teardown.py`、`diag_lowbit_teardown.py`：通信域拆解路径复现。
- `probe_isfinite*.py`、`probe_ar_sync.py`：梯度有限性 / all-reduce 后同步读的探针。
