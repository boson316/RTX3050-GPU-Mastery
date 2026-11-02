"""Diagnose benchmark variance: does a time-based warm-up remove it, and what is each path's launch floor?"""
from __future__ import annotations

import statistics as st
import sys
import time

import torch

sys.path.insert(0, ".")
from benchmarks.bench_utils import run_bench
from gpu_kernels.transformer import transformer_cuda as tc
from triton_kernels.softmax import softmax_triton

N = 8
big = torch.randn(2048, 768, device="cuda", dtype=torch.half)
tiny = torch.randn(1, 8, device="cuda", dtype=torch.half)
impls = {"PyTorch": lambda x: torch.softmax(x, -1), "Triton": softmax_triton, "CUDA": tc.softmax_cuda}
load_a = torch.randn(1024, 1024, device="cuda", dtype=torch.half)


def preheat(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        for _ in range(20):
            load_a @ load_a
        torch.cuda.synchronize()


def series(fn, heat: float) -> list[float]:
    out = []
    for _ in range(N):
        time.sleep(1.5)  # let the GPU drop back toward idle clocks, as between benchmark configs
        if heat:
            preheat(heat)
        out.append(run_bench(fn))
    return out


print(f"{'impl':8s} {'mode':10s} {'median':>8s} {'min':>8s} {'max':>8s} {'spread%':>8s}  samples(ms)")
for name, f in impls.items():
    for mode, heat in (("as-is", 0.0), ("preheat1s", 1.0)):
        s = series(lambda: f(big), heat)
        print(f"{name:8s} {mode:10s} {st.median(s):8.4f} {min(s):8.4f} {max(s):8.4f} {(max(s) - min(s)) / min(s) * 100:8.1f}  "
              + " ".join(f"{v:.4f}" for v in s), flush=True)
print("launch floor (1x8 input, preheated):")
for name, f in impls.items():
    preheat(1.0)
    s = [run_bench(lambda: f(tiny)) for _ in range(5)]
    print(f"{name:8s} median {st.median(s):.4f} ms  samples " + " ".join(f"{v:.4f}" for v in s), flush=True)
