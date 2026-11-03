"""What the probe CSVs can and cannot say about the WMMA clusters. Reads only; runs nothing on the GPU.

Usage: python results/v2_1/analyze_probe.py
"""
from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name: str) -> list[dict]:
    with open(ROOT / f"gpu_state_probe_{name}.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pearson(a: list[float], b: list[float]) -> float:
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return num / (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5


for name in ("fp16_matmul_bench", "fp16_2048"):
    rows = read(name)
    wmma = [float(r["WMMA_TILED_GFLOPS"]) for r in rows]
    print(f"=== {name}: {len(rows)} runs")
    print("sorted WMMA_TILED_GFLOPS:", " ".join(f"{v:.0f}" for v in sorted(wmma)))
    print("WMMA tiled ms: median", st.median(float(r["WMMA_TILED_MS"]) for r in rows))
    print("samples per run: median", st.median(int(r["samples"]) for r in rows), "wall_s median", st.median(float(r["wall_s"]) for r in rows))
    for key in ("CUBLAS_GFLOPS", "WMMA_NAIVE_GFLOPS", "CUDA_CORE_GFLOPS", "sm_mhz_median", "mem_mhz_median", "power_w_median", "temp_c_median"):
        print(f"r(WMMA_TILED_GFLOPS, {key}) = {pearson(wmma, [float(r[key]) for r in rows]):+.2f}")
    verified = all(r[k] == "PASS" for r in rows for k in ("VERIFY_CUBLAS", "VERIFY_CUDA_CORE", "VERIFY_WMMA_NAIVE", "VERIFY_WMMA_TILED"))
    print("all four kernels verified in every run:", verified)

# The sampler's median SM clock covers the whole process, not the kernel: does the clock it reports even allow the rate measured?
rows = read("fp16_matmul_bench")
GFLOPS_PER_MHZ = 21500 / 2100  # peak fp16_tc_fp32acc at the NVML max clock, results/v2/environment.md
fastest = max(float(r["CUBLAS_GFLOPS"]) for r in rows)
med_mhz = st.median(float(r["sm_mhz_median"]) for r in rows)
print(f"\n1024: median sampled SM clock {med_mhz:.0f} MHz allows {med_mhz * GFLOPS_PER_MHZ:.0f} GFLOPS; fastest cuBLAS run measured {fastest:.1f}, needing at least {fastest / GFLOPS_PER_MHZ:.0f} MHz")
slow = [r for r in rows if float(r["WMMA_TILED_GFLOPS"]) < 8058]
fast = [r for r in rows if float(r["WMMA_TILED_GFLOPS"]) >= 8058]
print(f"1024: cuBLAS median in slow-WMMA runs {st.median(float(r['CUBLAS_GFLOPS']) for r in slow):.1f} (n={len(slow)}), in fast-WMMA runs {st.median(float(r['CUBLAS_GFLOPS']) for r in fast):.1f} (n={len(fast)})")
print(f"1024: WMMA naive median in slow runs {st.median(float(r['WMMA_NAIVE_GFLOPS']) for r in slow):.1f}, in fast runs {st.median(float(r['WMMA_NAIVE_GFLOPS']) for r in fast):.1f}")
