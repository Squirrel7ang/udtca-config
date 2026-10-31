#!/usr/bin/env python3
"""对照组 B：彻底去掉 layer_attention_mask 里的 .item()，运行时零 D2H 读回。

依据：数据侧 mask 由 `torch.ones_like(input_ids)` 构造（run_qwen14b_polar_dp_pp_tp.py:223），
2D mask 恒为全 1，原分支恒成立。这里直接按 dim 判定，不再读回设备值。

目的是把"是不是 .item() 这个动作导致的"单独隔离出来：
  A 组（当前跑着的）：原样，每 chunk 一次 attention_mask.all().item()
  B 组（本补丁）：    零读回
若 A 挂 / B 不挂 -> .item() 是触发点；若都挂 -> .item() 无关，问题在别处。
"""
import sys

P = sys.argv[1]
src = open(P, encoding="utf-8").read()
orig = src

old = '''    def layer_attention_mask(attention_mask):
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2 and bool(attention_mask.all().item()):
            return None
        return attention_mask
'''

new = '''    def layer_attention_mask(attention_mask):
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            # [no-item] 数据侧 mask 由 torch.ones_like 构造（见 _QwenDataset 的
            # attention_mask = torch.ones_like(input_ids)），2D mask 恒为全 1，
            # 原来那句 attention_mask.all().item() 是对常量的每 chunk 一次主机同步。
            # 这里直接按 dim 判定：运行时零 D2H 读回，语义不变。
            if not getattr(layer_attention_mask, "_announced", False):
                layer_attention_mask._announced = True
                print("[no-item] layer_attention_mask: 2D mask 直接视为全 1，"
                      "不再做 .item() 读回", flush=True)
            return None
        return attention_mask
'''

assert src.count(old) == 1, "anchor not found / already patched"
src = src.replace(old, new)
open(P, "w", encoding="utf-8").write(src)
print(f"patched {P} (+{len(src) - len(orig)} bytes)")
