"""Phase 4C: Triton FlashAttention configs at long sequences (B=1, H=8, D=64), against SDPA and current autotune.

Every config is verified before timing. Usage: python results/v2/phase4_attention.py
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F
import triton

sys.path.insert(0, ".")
from benchmarks.bench_utils import check_close, run_bench
from triton_kernels.flash_attention.flash_attention import _fwd_kernel, flash_attention_triton

# (BLOCK_M, BLOCK_N, num_warps, num_stages). (64, 64, 4, 2) and (128, 64, 8, 2) are in the current autotune space.
CONFIGS = [(64, 64, 4, 2), (128, 64, 8, 2), (128, 64, 8, 3), (64, 128, 4, 2), (64, 128, 8, 3), (128, 128, 8, 2), (128, 128, 8, 3)]


def launch(q, k, v, bm, bn, warps, stages):
    b, h, s, d = q.shape
    out = torch.empty_like(q)
    grid = (triton.cdiv(s, bm), b * h)
    _fwd_kernel[grid](q, k, v, out, 1.0 / d**0.5, q.stride(1), q.stride(2), k.stride(1), k.stride(2), v.stride(1), v.stride(2),
                      out.stride(1), out.stride(2), s, s, d, BLOCK_M=bm, BLOCK_N=bn, BLOCK_D=triton.next_power_of_2(d),
                      IS_CAUSAL=False, num_warps=warps, num_stages=stages)
    return out


torch.manual_seed(0)
for dtype in (torch.float32, torch.float16):
    for S in (2048, 4096):
        q, k, v = (torch.randn(1, 8, S, 64, device="cuda", dtype=dtype) for _ in range(3))
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
        cases = [("SDPA", lambda: F.scaled_dot_product_attention(q, k, v)),
                 ("autotune", lambda: flash_attention_triton(q, k, v))]
        cases += [(f"M{c[0]} N{c[1]} w{c[2]} s{c[3]}", lambda c=c: launch(q, k, v, *c)) for c in CONFIGS]
        for name, fn in cases:
            try:
                check_close(name, fn(), ref)
                print(f"{dtype} S={S} {name:20s} {run_bench(fn):9.4f}", flush=True)
            except Exception as e:  # noqa: BLE001 - a config that fails is a result, keep going
                print(f"{dtype} S={S} {name:20s} FAILED {str(e)[:100]}", flush=True)
