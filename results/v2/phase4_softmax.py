"""Phase 4B: Triton softmax, multi-pass autotuned kernel vs single-load kernel, with PyTorch for reference.

Both Triton kernels are verified before timing. Larger shapes lift the kernel above Triton's launch floor.
Usage: python results/v2/phase4_softmax.py
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, ".")
from benchmarks.bench_utils import check_close, run_bench
from triton_kernels.softmax.softmax_triton import _softmax_autotuned_kernel, softmax_triton


def multi_pass(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    _softmax_autotuned_kernel[(x.shape[0],)](x, y, stride=x.stride(0), N=x.shape[1])
    return y


torch.manual_seed(0)
print(f"{'shape':>12s} {'impl':>16s} {'latency_ms':>11s}")
for M, N in ((2048, 768), (8192, 768), (32768, 768), (8192, 1024)):
    x = torch.randn(M, N, device="cuda", dtype=torch.float16) * 3
    ref = torch.softmax(x.float(), -1)
    for name, fn in (("PyTorch", lambda: torch.softmax(x, -1)),
                     ("Triton multi", lambda: multi_pass(x)),
                     ("Triton single", lambda: softmax_triton(x))):
        check_close(name, fn(), ref, atol=2e-2, rtol=2e-2)
        print(f"{M}x{N:<7d} {name:>16s} {run_bench(fn):11.4f}", flush=True)
