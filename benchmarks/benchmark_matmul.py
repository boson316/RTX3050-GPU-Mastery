"""
Matrix multiply benchmark: PyTorch (cuBLAS) vs Triton vs CUDA kernel.
Measures: latency (ms), throughput (GFLOPS), memory bandwidth (GB/s), optional GPU utilization.
Outputs: CSV, benchmark table, matplotlib charts in benchmarks/plots/.

Run from repo root: python benchmarks/benchmark_matmul.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import torch

from benchmarks.bench_utils import (
    BENCHMARKS_DIR,
    PLOTS_DIR,
    run_bench,
    sample_gpu_utilization,
    compute_metrics,
    save_csv,
    print_benchmark_table,
    plot_bar_chart,
    plot_grouped_bars,
    ensure_plots_dir,
)

WARMUP = 10
REPEAT = 50
SIZES = [512, 1024, 2048]
DTYPE = torch.float16


def _bytes(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def run_pytorch(M: int, K: int, N: int) -> tuple[float, float, float]:
    A = torch.randn(M, K, device="cuda", dtype=DTYPE)
    B = torch.randn(K, N, device="cuda", dtype=DTYPE)
    bytes_r = _bytes(A, B)
    bytes_w = M * N * 2
    flops = 2 * M * N * K

    def fn():
        torch.matmul(A, B)

    lat = run_bench(fn, warmup=WARMUP, repeat=REPEAT)
    m = compute_metrics(lat, bytes_r, bytes_w, flops)
    return lat, m.get("gflops", 0.0), m["memory_bandwidth_gbs"]


def run_triton(M: int, K: int, N: int) -> tuple[float | None, float | None, float | None]:
    try:
        from triton_kernels.matmul import matmul_triton
    except Exception:
        return None, None, None
    A = torch.randn(M, K, device="cuda", dtype=DTYPE)
    B = torch.randn(K, N, device="cuda", dtype=DTYPE)
    bytes_r = _bytes(A, B)
    bytes_w = M * N * 2
    flops = 2 * M * N * K

    def fn():
        matmul_triton(A, B)

    lat = run_bench(fn, warmup=WARMUP, repeat=REPEAT)
    m = compute_metrics(lat, bytes_r, bytes_w, flops)
    return lat, m.get("gflops"), m["memory_bandwidth_gbs"]


def run_cuda_standalone() -> tuple[float | None, float | None, float | None]:
    """Run cuda_roadmap fp16_matmul_bench if available (fixed 1024x1024)."""
    exe = root / "cuda_roadmap" / "level4_tensor_core" / "fp16_tensor_core_matmul" / "fp16_matmul_bench"
    if sys.platform == "win32":
        exe = exe.with_suffix(".exe")
    if not exe.is_file():
        return None, None, None
    try:
        out = subprocess.run(
            [str(exe)],
            cwd=exe.parent,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if out.returncode != 0:
            return None, None, None
        opt_ms = None
        for line in out.stdout.splitlines():
            m = re.match(r"CUDA_OPTIMIZED_MS=([\d.]+)", line)
            if m:
                opt_ms = float(m.group(1))
                break
        if opt_ms is None:
            return None, None, None
        M, N, K = 1024, 1024, 1024
        flops = 2 * M * N * K
        gflops = flops / 1e9 / (opt_ms / 1000.0)
        bytes_r = M * K * 2 + K * N * 2
        bytes_w = M * N * 2
        bw = (bytes_r + bytes_w) / 1e9 / (opt_ms / 1000.0)
        return opt_ms, gflops, bw
    except Exception:
        return None, None, None


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; run on GPU for meaningful results.")
        return

    ensure_plots_dir()
    device_name = torch.cuda.get_device_name(0)
    print(f"Matmul benchmark (FP16) — {device_name}")
    print(f"Sizes: {SIZES}; warmup={WARMUP}, repeat={REPEAT}")

    rows: list[dict] = []
    for N in SIZES:
        M, K = N, N
        # PyTorch
        lat_pt, gflops_pt, bw_pt = run_pytorch(M, K, N)
        rows.append({
            "implementation": "PyTorch",
            "size": f"{M}x{K}x{N}",
            "latency_ms": round(lat_pt, 4),
            "gflops": round(gflops_pt, 2),
            "memory_bandwidth_gbs": round(bw_pt, 2),
        })
        # Triton
        lat_tr, gflops_tr, bw_tr = run_triton(M, K, N)
        if lat_tr is not None:
            rows.append({
                "implementation": "Triton",
                "size": f"{M}x{K}x{N}",
                "latency_ms": round(lat_tr, 4),
                "gflops": round(gflops_tr or 0, 2),
                "memory_bandwidth_gbs": round(bw_tr or 0, 2),
            })
        # CUDA (standalone only for 1024)
        if N == 1024:
            lat_cuda, gflops_cuda, bw_cuda = run_cuda_standalone()
            if lat_cuda is not None:
                rows.append({
                    "implementation": "CUDA",
                    "size": f"{M}x{K}x{N}",
                    "latency_ms": round(lat_cuda, 4),
                    "gflops": round(gflops_cuda or 0, 2),
                    "memory_bandwidth_gbs": round(bw_cuda or 0, 2),
                })

    # Add GPU utilization for one size (PyTorch vs Triton)
    try:
        from triton_kernels.matmul import matmul_triton
        A = torch.randn(1024, 1024, device="cuda", dtype=DTYPE)
        B = torch.randn(1024, 1024, device="cuda", dtype=DTYPE)
        util_pt = sample_gpu_utilization(lambda: torch.matmul(A, B), repeat=30)
        util_tr = sample_gpu_utilization(lambda: matmul_triton(A, B), repeat=30)
        for r in rows:
            if r["implementation"] == "PyTorch" and r["size"] == "1024x1024x1024":
                r["gpu_utilization_pct"] = round(util_pt, 1) if util_pt is not None else None
            elif r["implementation"] == "Triton" and r["size"] == "1024x1024x1024":
                r["gpu_utilization_pct"] = round(util_tr, 1) if util_tr is not None else None
    except Exception:
        pass

    fieldnames = ["implementation", "size", "latency_ms", "gflops", "memory_bandwidth_gbs", "gpu_utilization_pct"]
    save_csv(rows, BENCHMARKS_DIR / "benchmark_matmul_results.csv", fieldnames=fieldnames)
    print(f"CSV saved: benchmarks/benchmark_matmul_results.csv")

    print_benchmark_table(rows, title="Matmul benchmark", columns=fieldnames)

    # Charts: latency and GFLOPS by implementation for each size
    for size in [f"{s}x{s}x{s}" for s in SIZES]:
        sub = [r for r in rows if r["size"] == size]
        if not sub:
            continue
        labels = [r["implementation"] for r in sub]
        latencies = [r["latency_ms"] for r in sub]
        plot_bar_chart(
            labels, latencies,
            ylabel="Latency (ms)", title=f"Matmul {size} — Latency",
            filename=f"matmul_latency_{size.replace('x', '_')}.png",
        )
        gflops_vals = [r.get("gflops", 0) for r in sub]
        plot_bar_chart(
            labels, gflops_vals,
            ylabel="GFLOPS", title=f"Matmul {size} — Throughput",
            filename=f"matmul_gflops_{size.replace('x', '_')}.png",
        )

    # Grouped: latency by size for PyTorch vs Triton
    impls = ["PyTorch", "Triton"]
    data = {impl: [] for impl in impls}
    for s in SIZES:
        sz = f"{s}x{s}x{s}"
        for impl in impls:
            r = next((x for x in rows if x["implementation"] == impl and x["size"] == sz), None)
            data[impl].append(r["latency_ms"] if r else 0.0)
    plot_grouped_bars(
        [f"{s}³" for s in SIZES], impls, data,
        ylabel="Latency (ms)", title="Matmul latency by size",
        filename="matmul_latency_by_size.png",
    )
    print(f"Charts saved to {PLOTS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
