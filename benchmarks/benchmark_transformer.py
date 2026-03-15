"""
Transformer kernels benchmark: PyTorch vs Triton vs CUDA (QKV, Softmax, LayerNorm, GELU, MLP).
Measures: latency (ms), throughput (GFLOPS where applicable), memory bandwidth (GB/s), optional GPU utilization.
Outputs: CSV, benchmark table, matplotlib charts in benchmarks/plots/.

Run from repo root: python benchmarks/benchmark_transformer.py
Env:
  TRANSFORMER_BENCHMARK_CUDA=1   — enable CUDA for QKV and MLP only (avoids hang).
  TRANSFORMER_BENCHMARK_CUDA_ALL=1 — also run CUDA for Softmax/LayerNorm/GELU (may hang on some GPUs).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import torch

from benchmarks.transformer_reference import (
    fused_qkv_pytorch,
    softmax_pytorch,
    layernorm_pytorch,
    gelu_pytorch,
    fused_mlp_pytorch,
)
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

_enable_cuda = os.environ.get("TRANSFORMER_BENCHMARK_CUDA", "").strip().lower() in ("1", "true", "yes")
# Softmax/LayerNorm/GELU CUDA can hang on some devices — only run when explicitly requested
_run_cuda_softmax_ln_gelu = os.environ.get("TRANSFORMER_BENCHMARK_CUDA_ALL", "").strip().lower() in ("1", "true", "yes")
SKIP_CUDA = not _enable_cuda

B, S, H = 8, 256, 768
H3 = H * 3
H4 = H * 4
WARMUP, REPEAT = 5, 15
CUDA_WARMUP, CUDA_REPEAT = 2, 5
DTYPE = torch.float16


def _bytes(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def bench_qkv() -> list[dict]:
    M, K, N = B * S, H, H3
    x = torch.randn(M, K, device="cuda", dtype=DTYPE)
    w = torch.randn(K, N, device="cuda", dtype=DTYPE)
    b = torch.randn(N, device="cuda", dtype=DTYPE)
    bytes_r = _bytes(x, w, b)
    bytes_w = M * N * 2
    flops = 2 * M * N * K
    rows = []

    lat = run_bench(lambda: fused_qkv_pytorch(x, w, b), warmup=WARMUP, repeat=REPEAT)
    m = compute_metrics(lat, bytes_r, bytes_w, flops)
    rows.append({"kernel": "QKV", "implementation": "PyTorch", "latency_ms": round(lat, 4), "gflops": round(m.get("gflops", 0), 2), "memory_bandwidth_gbs": round(m["memory_bandwidth_gbs"], 2)})

    try:
        from triton_kernels.qkv import fused_qkv_triton
        lat = run_bench(lambda: fused_qkv_triton(x, w, b), warmup=WARMUP, repeat=REPEAT)
        m = compute_metrics(lat, bytes_r, bytes_w, flops)
        rows.append({"kernel": "QKV", "implementation": "Triton", "latency_ms": round(lat, 4), "gflops": round(m.get("gflops", 0), 2), "memory_bandwidth_gbs": round(m["memory_bandwidth_gbs"], 2)})
    except Exception:
        pass

    if not SKIP_CUDA:
        try:
            from gpu_kernels.transformer.transformer_cuda import fused_qkv_cuda
            out = fused_qkv_cuda(x, w, b)
            if out is not None:
                lat = run_bench(lambda: fused_qkv_cuda(x, w, b), warmup=CUDA_WARMUP, repeat=CUDA_REPEAT)
                m = compute_metrics(lat, bytes_r, bytes_w, flops)
                rows.append({"kernel": "QKV", "implementation": "CUDA", "latency_ms": round(lat, 4), "gflops": round(m.get("gflops", 0), 2), "memory_bandwidth_gbs": round(m["memory_bandwidth_gbs"], 2)})
        except Exception:
            pass
    return rows


def bench_softmax() -> list[dict]:
    M, N = B * S, H
    x = torch.randn(M, N, device="cuda", dtype=DTYPE)
    bytes_r = _bytes(x)
    bytes_w = x.numel() * 2
    rows = []

    lat = run_bench(lambda: softmax_pytorch(x, dim=-1), warmup=WARMUP, repeat=REPEAT)
    m = compute_metrics(lat, bytes_r, bytes_w, None)
    rows.append({"kernel": "Softmax", "implementation": "PyTorch", "latency_ms": round(lat, 4), "gflops": None, "memory_bandwidth_gbs": round(m["memory_bandwidth_gbs"], 2)})

    try:
        from triton_kernels.softmax import softmax_triton
        lat = run_bench(lambda: softmax_triton(x, dim=-1), warmup=WARMUP, repeat=REPEAT)
        m = compute_metrics(lat, bytes_r, bytes_w, None)
        rows.append({"kernel": "Softmax", "implementation": "Triton", "latency_ms": round(lat, 4), "gflops": None, "memory_bandwidth_gbs": round(m["memory_bandwidth_gbs"], 2)})
    except Exception:
        pass

    if not SKIP_CUDA and _run_cuda_softmax_ln_gelu:
        try:
            from gpu_kernels.transformer.transformer_cuda import softmax_cuda
            if softmax_cuda(x, -1) is not None:
                lat = run_bench(lambda: softmax_cuda(x, -1), warmup=CUDA_WARMUP, repeat=CUDA_REPEAT)
                m = compute_metrics(lat, bytes_r, bytes_w, None)
                rows.append({"kernel": "Softmax", "implementation": "CUDA", "latency_ms": round(lat, 4), "gflops": None, "memory_bandwidth_gbs": round(m["memory_bandwidth_gbs"], 2)})
        except Exception:
            pass
    return rows


def bench_layernorm() -> list[dict]:
    M, N = B * S, H
    x = torch.randn(M, N, device="cuda", dtype=DTYPE)
    weight = torch.ones(N, device="cuda", dtype=DTYPE)
    bias = torch.zeros(N, device="cuda", dtype=DTYPE)
    bytes_r = _bytes(x, weight, bias)
    bytes_w = x.numel() * 2
    rows = []

    lat = run_bench(lambda: layernorm_pytorch(x, weight, bias), warmup=WARMUP, repeat=REPEAT)
    m = compute_metrics(lat, bytes_r, bytes_w, None)
    rows.append({"kernel": "LayerNorm", "implementation": "PyTorch", "latency_ms": round(lat, 4), "gflops": None, "memory_bandwidth_gbs": round(m["memory_bandwidth_gbs"], 2)})

    try:
        from triton_kernels.layernorm import layernorm_triton
        lat = run_bench(lambda: layernorm_triton(x, (N,), weight, bias), warmup=WARMUP, repeat=REPEAT)
        m = compute_metrics(lat, bytes_r, bytes_w, None)
        rows.append({"kernel": "LayerNorm", "implementation": "Triton", "latency_ms": round(lat, 4), "gflops": None, "memory_bandwidth_gbs": round(m["memory_bandwidth_gbs"], 2)})
    except Exception:
        pass

    if not SKIP_CUDA and _run_cuda_softmax_ln_gelu:
        try:
            from gpu_kernels.transformer.transformer_cuda import layernorm_cuda
            if layernorm_cuda(x, weight, bias, 1e-5) is not None:
                lat = run_bench(lambda: layernorm_cuda(x, weight, bias, 1e-5), warmup=CUDA_WARMUP, repeat=CUDA_REPEAT)
                m = compute_metrics(lat, bytes_r, bytes_w, None)
                rows.append({"kernel": "LayerNorm", "implementation": "CUDA", "latency_ms": round(lat, 4), "gflops": None, "memory_bandwidth_gbs": round(m["memory_bandwidth_gbs"], 2)})
        except Exception:
            pass
    return rows


def bench_gelu() -> list[dict]:
    n = B * S * H
    x = torch.randn(n, device="cuda", dtype=DTYPE)
    bytes_r = _bytes(x)
    bytes_w = x.numel() * 2
    rows = []

    lat = run_bench(lambda: gelu_pytorch(x), warmup=WARMUP, repeat=REPEAT)
    m = compute_metrics(lat, bytes_r, bytes_w, None)
    rows.append({"kernel": "GELU", "implementation": "PyTorch", "latency_ms": round(lat, 4), "gflops": None, "memory_bandwidth_gbs": round(m["memory_bandwidth_gbs"], 2)})

    try:
        from triton_kernels.gelu import gelu_triton
        lat = run_bench(lambda: gelu_triton(x), warmup=WARMUP, repeat=REPEAT)
        m = compute_metrics(lat, bytes_r, bytes_w, None)
        rows.append({"kernel": "GELU", "implementation": "Triton", "latency_ms": round(lat, 4), "gflops": None, "memory_bandwidth_gbs": round(m["memory_bandwidth_gbs"], 2)})
    except Exception:
        pass

    if not SKIP_CUDA and _run_cuda_softmax_ln_gelu:
        try:
            from gpu_kernels.transformer.transformer_cuda import gelu_cuda
            if gelu_cuda(x) is not None:
                lat = run_bench(lambda: gelu_cuda(x), warmup=CUDA_WARMUP, repeat=CUDA_REPEAT)
                m = compute_metrics(lat, bytes_r, bytes_w, None)
                rows.append({"kernel": "GELU", "implementation": "CUDA", "latency_ms": round(lat, 4), "gflops": None, "memory_bandwidth_gbs": round(m["memory_bandwidth_gbs"], 2)})
        except Exception:
            pass
    return rows


def bench_mlp() -> list[dict]:
    M, K, N1, N2 = B * S, H, H4, H
    x = torch.randn(M, K, device="cuda", dtype=DTYPE)
    w1 = torch.randn(K, N1, device="cuda", dtype=DTYPE)
    b1 = torch.randn(N1, device="cuda", dtype=DTYPE)
    w2 = torch.randn(N1, N2, device="cuda", dtype=DTYPE)
    b2 = torch.randn(N2, device="cuda", dtype=DTYPE)
    bytes_r = _bytes(x, w1, b1, w2, b2) + M * N1 * 2
    bytes_w = M * N2 * 2
    flops = 2 * M * K * N1 + 2 * M * N1 * N2
    rows = []

    lat = run_bench(lambda: fused_mlp_pytorch(x, w1, b1, w2, b2), warmup=WARMUP, repeat=REPEAT)
    m = compute_metrics(lat, bytes_r, bytes_w, flops)
    rows.append({"kernel": "MLP", "implementation": "PyTorch", "latency_ms": round(lat, 4), "gflops": round(m.get("gflops", 0), 2), "memory_bandwidth_gbs": round(m["memory_bandwidth_gbs"], 2)})

    try:
        from triton_kernels.mlp import fused_mlp_triton
        lat = run_bench(lambda: fused_mlp_triton(x, w1, b1, w2, b2), warmup=WARMUP, repeat=REPEAT)
        m = compute_metrics(lat, bytes_r, bytes_w, flops)
        rows.append({"kernel": "MLP", "implementation": "Triton", "latency_ms": round(lat, 4), "gflops": round(m.get("gflops", 0), 2), "memory_bandwidth_gbs": round(m["memory_bandwidth_gbs"], 2)})
    except Exception:
        pass

    if not SKIP_CUDA:
        try:
            from gpu_kernels.transformer.transformer_cuda import fused_mlp_cuda
            if fused_mlp_cuda(x, w1, b1, w2, b2) is not None:
                lat = run_bench(lambda: fused_mlp_cuda(x, w1, b1, w2, b2), warmup=CUDA_WARMUP, repeat=CUDA_REPEAT)
                m = compute_metrics(lat, bytes_r, bytes_w, flops)
                rows.append({"kernel": "MLP", "implementation": "CUDA", "latency_ms": round(lat, 4), "gflops": round(m.get("gflops", 0), 2), "memory_bandwidth_gbs": round(m["memory_bandwidth_gbs"], 2)})
        except Exception:
            pass
    return rows


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; run on GPU for meaningful results.")
        return

    ensure_plots_dir()
    device_name = torch.cuda.get_device_name(0)
    print(f"Transformer kernels benchmark (FP16) — {device_name}")
    print(f"B={B}, S={S}, H={H}; CUDA (QKV+MLP): {'on' if _enable_cuda else 'off'}; CUDA Softmax/LN/GELU: {'on' if _run_cuda_softmax_ln_gelu else 'off (set CUDA_ALL=1 to enable)'}")

    all_rows: list[dict] = []
    for bench_fn in [bench_qkv, bench_softmax, bench_layernorm, bench_gelu, bench_mlp]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time.sleep(0.3)
        all_rows.extend(bench_fn())

    fieldnames = ["kernel", "implementation", "latency_ms", "gflops", "memory_bandwidth_gbs"]
    save_csv(all_rows, BENCHMARKS_DIR / "benchmark_transformer_results.csv", fieldnames=fieldnames)
    print(f"CSV saved: benchmarks/benchmark_transformer_results.csv")

    print_benchmark_table(all_rows, title="Transformer kernels", columns=fieldnames)

    for kernel in ["QKV", "Softmax", "LayerNorm", "GELU", "MLP"]:
        sub = [r for r in all_rows if r["kernel"] == kernel]
        if not sub:
            continue
        labels = [r["implementation"] for r in sub]
        latencies = [r["latency_ms"] for r in sub]
        plot_bar_chart(
            labels, latencies,
            ylabel="Latency (ms)", title=f"Transformer {kernel} — Latency",
            filename=f"transformer_{kernel.lower()}_latency.png",
        )
        if any(r.get("gflops") is not None for r in sub):
            gflops_vals = [r.get("gflops") or 0 for r in sub]
            plot_bar_chart(
                labels, gflops_vals,
                ylabel="GFLOPS", title=f"Transformer {kernel} — Throughput",
                filename=f"transformer_{kernel.lower()}_gflops.png",
            )

    impls = ["PyTorch", "Triton", "CUDA"]
    kernels_ordered = ["QKV", "Softmax", "LayerNorm", "GELU", "MLP"]
    data = {impl: [] for impl in impls}
    for kernel in kernels_ordered:
        for impl in impls:
            r = next((x for x in all_rows if x["kernel"] == kernel and x["implementation"] == impl), None)
            data[impl].append(r["latency_ms"] if r else 0.0)
    plot_grouped_bars(
        kernels_ordered, impls, data,
        ylabel="Latency (ms)", title="Transformer kernels latency",
        filename="transformer_latency_by_kernel.png",
    )
    print(f"Charts saved to {PLOTS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
