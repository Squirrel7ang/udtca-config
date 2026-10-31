#!/usr/bin/env python3
"""去掉 layer_attention_mask 里每个 forward chunk 一次的 .item() 全流同步。

数据侧 mask 由 torch.ones_like(input_ids) 构造（第 223 行），同一形状下内容恒定，
所以判定结果按 shape 缓存，只有首次见到该形状时做一次 D2H 读回。
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
            # [nosync-mask] 数据侧 mask 由 torch.ones_like 构造，同一 shape 下内容恒定。
            # 原来每个 forward chunk 都做一次 attention_mask.all().item()，这是一次
            # 主机侧全流同步，会把 PP 流水线拽成锁步；改为按 shape 只判定一次。
            _cache = layer_attention_mask.__dict__.setdefault("_verdict", {})
            _key = tuple(attention_mask.shape)
            _verdict = _cache.get(_key)
            if _verdict is None:
                _verdict = bool(attention_mask.all().item())
                _cache[_key] = _verdict
                print(f"[nosync-mask] first verdict shape={_key} all_ones={_verdict}",
                      flush=True)
            if _verdict:
                return None
        return attention_mask
'''

assert src.count(old) == 1, "anchor not found / already patched"
src = src.replace(old, new)
open(P, "w", encoding="utf-8").write(src)
print(f"patched {P} (+{len(src) - len(orig)} bytes)")
