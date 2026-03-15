"""
Conv2D (3×3) benchmark: PyTorch vs Triton vs CUDA kernel.
Measures: latency (ms), throughput (GFLOPS), memory bandwidth (GB/s), optional GPU utilization.
Outputs: CSV, benchmark table, matplotlib charts in benchmarks/plots/.

Run from repo root: python benchmarks/benchmark_conv.py
"""
from __future__ import annotations

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import torch
import torch.nn as nn

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
# (B, C_in, H, W), C_out
CONFIGS = [
    (32, 1, 28, 28, 32),
    (128, 1, 28, 28, 32),
    (128, 1, 64, 64, 64),
]
DTYPE = torch.float16


def _bytes(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def conv_flops(B: int, C_in: int, H: int, W: int, C_out: int, K: int = 3) -> float:
    out_h, out_w = H - K + 1, W - K + 1
    return 2 * B * C_out * out_h * out_w * C_in * K * K


def run_pytorch(B: int, C_in: int, H: int, W: int, C_out: int) -> tuple[float, float, float]:
    x = torch.randn(B, C_in, H, W, device="cuda", dtype=DTYPE)
    conv = nn.Conv2d(C_in, C_out, 3, padding=0).to("cuda", dtype=DTYPE)
    w, b = conv.weight, conv.bias
    bytes_r = _bytes(x, w, b)
    out_h, out_w = H - 2, W - 2
    bytes_w = B * C_out * out_h * out_w * 2

    def fn():
        torch.nn.functional.conv2d(x, w, b, padding=0)

    lat = run_bench(fn, warmup=WARMUP, repeat=REPEAT)
    flops = conv_flops(B, C_in, H, W, C_out)
    m = compute_metrics(lat, bytes_r, bytes_w, flops)
    return lat, m.get("gflops", 0.0), m["memory_bandwidth_gbs"]


def run_triton(B: int, C_in: int, H: int, W: int, C_out: int) -> tuple[float | None, float | None, float | None]:
    try:
        from triton_kernels.conv import conv2d_triton
    except Exception:
        return None, None, None
    x = torch.randn(B, C_in, H, W, device="cuda", dtype=DTYPE)
    w = torch.randn(C_out, C_in, 3, 3, device="cuda", dtype=DTYPE)
    b = torch.zeros(C_out, device="cuda", dtype=DTYPE)
    bytes_r = _bytes(x, w, b)
    out_h, out_w = H - 2, W - 2
    bytes_w = B * C_out * out_h * out_w * 2

    def fn():
        conv2d_triton(x, w, b, use_autotune=True)

    lat = run_bench(fn, warmup=WARMUP, repeat=REPEAT)
    flops = conv_flops(B, C_in, H, W, C_out)
    m = compute_metrics(lat, bytes_r, bytes_w, flops)
    return lat, m.get("gflops"), m["memory_bandwidth_gbs"]


def run_cuda(B: int, C_in: int, H: int, W: int, C_out: int) -> tuple[float | None, float | None, float | None]:
    """CUDA extension (custom_conv) only supports float32; we benchmark in FP32."""
    try:
        import custom_conv
    except ImportError:
        return None, None, None
    if C_in != 1 or C_out != 32:
        # extension is built for C_in=1, C_out=32
        return None, None, None
    # custom_conv2d is not implemented for Half — use float32
    x = torch.randn(B, C_in, H, W, device="cuda", dtype=torch.float32)
    conv = nn.Conv2d(C_in, C_out, 3, padding=0).to("cuda", dtype=torch.float32)
    w, b = conv.weight, conv.bias
    bytes_r = _bytes(x, w, b)
    out_h, out_w = H - 2, W - 2
    bytes_w = B * C_out * out_h * out_w * 4  # float32

    def fn():
        custom_conv.custom_conv2d(x, w, b)[0]

    lat = run_bench(fn, warmup=WARMUP, repeat=REPEAT)
    flops = conv_flops(B, C_in, H, W, C_out)
    m = compute_metrics(lat, bytes_r, bytes_w, flops)
    return lat, m.get("gflops"), m["memory_bandwidth_gbs"]


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; run on GPU for meaningful results.")
        return

    ensure_plots_dir()
    device_name = torch.cuda.get_device_name(0)
    print(f"Conv2D (3×3) benchmark (FP16) — {device_name}")
    print(f"Configs: {CONFIGS}; warmup={WARMUP}, repeat={REPEAT}")

    rows: list[dict] = []
    for cfg in CONFIGS:
        B, C_in, H, W, C_out = cfg
        config_name = f"B{B}_C{C_in}x{C_out}_{H}x{W}"

        lat_pt, gflops_pt, bw_pt = run_pytorch(B, C_in, H, W, C_out)
        rows.append({
            "implementation": "PyTorch",
            "config": config_name,
            "latency_ms": round(lat_pt, 4),
            "gflops": round(gflops_pt, 2),
            "memory_bandwidth_gbs": round(bw_pt, 2),
        })

        lat_tr, gflops_tr, bw_tr = run_triton(B, C_in, H, W, C_out)
        if lat_tr is not None:
            rows.append({
                "implementation": "Triton",
                "config": config_name,
                "latency_ms": round(lat_tr, 4),
                "gflops": round(gflops_tr or 0, 2),
                "memory_bandwidth_gbs": round(bw_tr or 0, 2),
            })

        lat_cuda, gflops_cuda, bw_cuda = run_cuda(B, C_in, H, W, C_out)
        if lat_cuda is not None:
            rows.append({
                "implementation": "CUDA",
                "config": config_name,
                "latency_ms": round(lat_cuda, 4),
                "gflops": round(gflops_cuda or 0, 2),
                "memory_bandwidth_gbs": round(bw_cuda or 0, 2),
            })

    # GPU utilization for first config
    B, C_in, H, W, C_out = CONFIGS[0]
    try:
        x = torch.randn(B, C_in, H, W, device="cuda", dtype=DTYPE)
        conv = nn.Conv2d(C_in, C_out, 3, padding=0).to("cuda", dtype=DTYPE)
        util_pt = sample_gpu_utilization(
            lambda: torch.nn.functional.conv2d(x, conv.weight, conv.bias, padding=0), repeat=25
        )
        from triton_kernels.conv import conv2d_triton
        w, b = conv.weight, conv.bias
        util_tr = sample_gpu_utilization(lambda: conv2d_triton(x, w, b), repeat=25)
        for r in rows:
            if r["implementation"] == "PyTorch" and r["config"] == f"B{B}_C{C_in}x{C_out}_{H}x{W}":
                r["gpu_utilization_pct"] = round(util_pt, 1) if util_pt is not None else None
            elif r["implementation"] == "Triton" and r["config"] == f"B{B}_C{C_in}x{C_out}_{H}x{W}":
                r["gpu_utilization_pct"] = round(util_tr, 1) if util_tr is not None else None
    except Exception:
        pass

    fieldnames = ["implementation", "config", "latency_ms", "gflops", "memory_bandwidth_gbs", "gpu_utilization_pct"]
    save_csv(rows, BENCHMARKS_DIR / "benchmark_conv_results.csv", fieldnames=fieldnames)
    print(f"CSV saved: benchmarks/benchmark_conv_results.csv")

    print_benchmark_table(rows, title="Conv2D benchmark", columns=fieldnames)

    for config_name in set(r["config"] for r in rows):
        sub = [r for r in rows if r["config"] == config_name]
        if not sub:
            continue
        labels = [r["implementation"] for r in sub]
        latencies = [r["latency_ms"] for r in sub]
        plot_bar_chart(
            labels, latencies,
            ylabel="Latency (ms)", title=f"Conv2D {config_name} — Latency",
            filename=f"conv_latency_{config_name}.png",
        )
        gflops_vals = [r.get("gflops", 0) for r in sub]
        plot_bar_chart(
            labels, gflops_vals,
            ylabel="GFLOPS", title=f"Conv2D {config_name} — Throughput",
            filename=f"conv_gflops_{config_name}.png",
        )

    impls = ["PyTorch", "Triton", "CUDA"]
    data = {impl: [] for impl in impls}
    configs_ordered = [f"B{B}_C{C_in}x{C_out}_{H}x{W}" for B, C_in, H, W, C_out in CONFIGS]
    for config_name in configs_ordered:
        for impl in impls:
            r = next((x for x in rows if x["implementation"] == impl and x["config"] == config_name), None)
            data[impl].append(r["latency_ms"] if r else 0.0)
    plot_grouped_bars(
        configs_ordered, impls, data,
        ylabel="Latency (ms)", title="Conv2D latency by config",
        filename="conv_latency_by_config.png",
    )
    print(f"Charts saved to {PLOTS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
