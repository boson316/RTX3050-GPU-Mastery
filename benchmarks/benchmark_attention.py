"""
Attention (scaled dot-product / Flash) benchmark: PyTorch SDPA vs Triton vs CUDA kernel.
Measures: latency (ms), throughput (GFLOPS), memory bandwidth (GB/s), optional GPU utilization.
Outputs: CSV, benchmark table, matplotlib charts in benchmarks/plots/.

Run from repo root: python benchmarks/benchmark_attention.py
"""
from __future__ import annotations

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import torch

from benchmarks.bench_utils import (
    BENCHMARKS_DIR,
    PLOTS_DIR,
    ensure_plots_dir,
    measure,
    plot_bar_chart,
    plot_grouped_bars,
    print_benchmark_table,
    sample_gpu_utilization,
    save_csv,
)

WARMUP = 8
REPEAT = 30
# (B, H, S, D)
CONFIGS = [
    (2, 8, 128, 64),
    (4, 8, 256, 64),
    (2, 8, 512, 64),
]
DTYPE = torch.float16


def _bytes(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def attention_flops(B: int, H: int, S: int, D: int) -> float:
    # Q@K^T: 2*S*S*D per head; P@V: 2*S*S*D per head (softmax FLOPs ignored, as is conventional)
    qk = 2 * B * H * S * D * S
    attn_v = 2 * B * H * S * S * D
    return qk + attn_v


def _inputs(B: int, H: int, S: int, D: int, dtype: torch.dtype):
    g = torch.Generator(device="cuda").manual_seed(B * 1000 + S)
    return [torch.randn(B, H, S, D, device="cuda", dtype=dtype, generator=g) for _ in range(3)]


def _reference(q, k, v) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())


def _measure(name, fn, q, k, v, elem_bytes, B, H, S, D):
    m = measure(name, fn, _reference(q, k, v), 3 * B * H * S * D * elem_bytes, B * H * S * D * elem_bytes,
                attention_flops(B, H, S, D), warmup=WARMUP, repeat=REPEAT)
    if m is None:
        return None, None, None
    return m["latency_ms"], m.get("gflops"), m["memory_bandwidth_gbs"]


def run_pytorch(B: int, H: int, S: int, D: int):
    q, k, v = _inputs(B, H, S, D, DTYPE)
    return _measure("SDPA", lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v), q, k, v, 2, B, H, S, D)


def run_triton(B: int, H: int, S: int, D: int):
    try:
        from triton_kernels.flash_attention import flash_attention_triton
    except Exception:
        return None, None, None
    q, k, v = _inputs(B, H, S, D, DTYPE)
    return _measure("Triton flash", lambda: flash_attention_triton(q, k, v, causal=False), q, k, v, 2, B, H, S, D)


def run_cuda(B: int, H: int, S: int, D: int):
    """flash_attention_simple CUDA kernel: FP32, D <= 64. FLOPs are the same, bytes are 2x (FP32)."""
    if D > 64:
        return None, None, None
    try:
        from flash_attention_simple.attention_cuda import attention_cuda
    except Exception:
        return None, None, None
    q, k, v = _inputs(B, H, S, D, torch.float32)
    return _measure("CUDA flash (FP32)", lambda: attention_cuda(q, k, v), q, k, v, 4, B, H, S, D)


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; run on GPU for meaningful results.")
        return

    ensure_plots_dir()
    device_name = torch.cuda.get_device_name(0)
    print(f"Attention benchmark (FP16) — {device_name}")
    print(f"Configs (B,H,S,D): {CONFIGS}; warmup={WARMUP}, repeat={REPEAT}")

    rows: list[dict] = []
    for (B, H, S, D) in CONFIGS:
        config_name = f"B{B}_H{H}_S{S}_D{D}"

        lat_pt, gflops_pt, bw_pt = run_pytorch(B, H, S, D)
        rows.append({
            "implementation": "PyTorch",
            "config": config_name,
            "latency_ms": round(lat_pt, 4),
            "gflops": round(gflops_pt, 2),
            "memory_bandwidth_gbs": round(bw_pt, 2),
        })

        lat_tr, gflops_tr, bw_tr = run_triton(B, H, S, D)
        if lat_tr is not None:
            rows.append({
                "implementation": "Triton",
                "config": config_name,
                "latency_ms": round(lat_tr, 4),
                "gflops": round(gflops_tr or 0, 2),
                "memory_bandwidth_gbs": round(bw_tr or 0, 2),
            })

        lat_cuda, gflops_cuda, bw_cuda = run_cuda(B, H, S, D)
        if lat_cuda is not None:
            rows.append({
                "implementation": "CUDA",
                "config": config_name,
                "latency_ms": round(lat_cuda, 4),
                "gflops": round(gflops_cuda or 0, 2),
                "memory_bandwidth_gbs": round(bw_cuda or 0, 2),
            })

    try:
        B, H, S, D = CONFIGS[0]
        q = torch.randn(B, H, S, D, device="cuda", dtype=DTYPE)
        k = torch.randn(B, H, S, D, device="cuda", dtype=DTYPE)
        v = torch.randn(B, H, S, D, device="cuda", dtype=DTYPE)
        util_pt = sample_gpu_utilization(
            lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False), repeat=20
        )
        from triton_kernels.flash_attention import flash_attention_triton
        util_tr = sample_gpu_utilization(lambda: flash_attention_triton(q, k, v, causal=False), repeat=20)
        for r in rows:
            c0 = f"B{B}_H{H}_S{S}_D{D}"
            if r["implementation"] == "PyTorch" and r["config"] == c0:
                r["gpu_utilization_pct"] = round(util_pt, 1) if util_pt is not None else None
            elif r["implementation"] == "Triton" and r["config"] == c0:
                r["gpu_utilization_pct"] = round(util_tr, 1) if util_tr is not None else None
    except Exception:
        pass

    fieldnames = ["implementation", "config", "latency_ms", "gflops", "memory_bandwidth_gbs", "gpu_utilization_pct"]
    save_csv(rows, BENCHMARKS_DIR / "benchmark_attention_results.csv", fieldnames=fieldnames)
    print("CSV saved: benchmarks/benchmark_attention_results.csv")

    print_benchmark_table(rows, title="Attention benchmark", columns=fieldnames)

    for config_name in set(r["config"] for r in rows):
        sub = [r for r in rows if r["config"] == config_name]
        if not sub:
            continue
        labels = [r["implementation"] for r in sub]
        latencies = [r["latency_ms"] for r in sub]
        plot_bar_chart(
            labels, latencies,
            ylabel="Latency (ms)", title=f"Attention {config_name} — Latency",
            filename=f"attention_latency_{config_name}.png",
        )
        gflops_vals = [r.get("gflops", 0) for r in sub]
        plot_bar_chart(
            labels, gflops_vals,
            ylabel="GFLOPS", title=f"Attention {config_name} — Throughput",
            filename=f"attention_gflops_{config_name}.png",
        )

    impls = ["PyTorch", "Triton", "CUDA"]
    data = {impl: [] for impl in impls}
    configs_ordered = [f"B{B}_H{H}_S{S}_D{D}" for B, H, S, D in CONFIGS]
    for config_name in configs_ordered:
        for impl in impls:
            r = next((x for x in rows if x["implementation"] == impl and x["config"] == config_name), None)
            data[impl].append(r["latency_ms"] if r else 0.0)
    plot_grouped_bars(
        configs_ordered, impls, data,
        ylabel="Latency (ms)", title="Attention latency by config",
        filename="attention_latency_by_config.png",
    )
    print(f"Charts saved to {PLOTS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
