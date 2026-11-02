"""FlashAttention at long sequences (B=1, H=8, D=64, FP32): latency and extra VRAM, SDPA vs Triton vs CUDA.

Every implementation is verified against an SDPA reference before it is timed.
Usage: python results/v2/longseq_attention.py <run_id>
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from benchmarks.bench_utils import check_close, run_bench
from flash_attention_simple.attention_cuda import attention_cuda
from triton_kernels.flash_attention import flash_attention_triton

run_id = sys.argv[1]
torch.manual_seed(0)
B, H, D = 1, 8, 64
IMPLS = {
    "SDPA": lambda q, k, v: F.scaled_dot_product_attention(q, k, v),
    "Triton": lambda q, k, v: flash_attention_triton(q, k, v, causal=False),
    "CUDA": lambda q, k, v: attention_cuda(q, k, v, causal=False),
}
rows = []
for S in (1024, 2048, 4096):
    q, k, v = (torch.randn(B, H, S, D, device="cuda") for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v)
    for name, fn in IMPLS.items():
        try:
            check_close(name, fn(q, k, v), ref)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            fn(q, k, v)
            extra_mb = (torch.cuda.max_memory_allocated() - base) / 2**20
            lat = run_bench(lambda: fn(q, k, v))
            rows.append({"S": S, "impl": name, "verified": "PASS", "latency_ms": round(lat, 4), "peak_extra_mib": round(extra_mb, 1)})
        except Exception as e:  # noqa: BLE001 - record the failure instead of dropping the row
            rows.append({"S": S, "impl": name, "verified": f"FAIL: {str(e)[:120]}", "latency_ms": "", "peak_extra_mib": ""})
        print(rows[-1], flush=True)
out = Path(f"results/v2/csv/run{run_id}/longseq_attention.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
