"""
Roofline analysis for CUDA and Triton kernels.

1. Estimates arithmetic intensity (FLOPs/byte) for each kernel.
2. Measures achieved FLOPS and memory bandwidth by running kernels.
3. Generates roofline plots: memory-bound vs compute-bound.

Results and plots are stored in profiling/roofline/.

Run from repo root:
  python profiling/roofline_analysis.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # profiling/roofline_analysis.py -> repo root
sys.path.insert(0, str(REPO_ROOT))

import torch

# RTX 3050 Laptop: FP16 peak ~9 TFLOPS, memory BW ~192 GB/s
PEAK_GFLOPS = float(__import__("os").environ.get("ROOFLINE_PEAK_GFLOPS", "9000"))
PEAK_GBPS = float(__import__("os").environ.get("ROOFLINE_PEAK_GBPS", "192"))
RIDGE_AI = PEAK_GFLOPS / PEAK_GBPS  # FLOPs/byte

OUT_DIR = REPO_ROOT / "profiling" / "roofline"
WARMUP = 5
REPEAT = 25
DTYPE = torch.float16


def run_bench(fn, warmup=WARMUP, repeat=REPEAT):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeat


def compute_metrics(latency_ms, bytes_read, bytes_written, flops):
    time_sec = latency_ms / 1000.0
    total_bytes = bytes_read + bytes_written
    gbps = total_bytes / 1e9 / time_sec if time_sec > 0 else 0.0
    gflops = (flops / 1e9 / time_sec) if (flops and flops > 0 and time_sec > 0) else None
    return gflops, gbps


# ---------------------------------------------------------------------------
# Matmul: C = A @ B, M×K @ K×N
# ---------------------------------------------------------------------------
def matmul_flops_bytes(M, K, N, elem_bytes=2):
    flops = 2 * M * N * K
    bytes_total = (M * K + K * N + M * N) * elem_bytes
    return flops, bytes_total


def run_roofline_matmul():
    results = []
    for N in [512, 1024, 2048]:
        M, K = N, N
        flops, bytes_total = matmul_flops_bytes(M, K, N)
        ai = flops / bytes_total if bytes_total > 0 else 0

        # PyTorch
        A = torch.randn(M, K, device="cuda", dtype=DTYPE)
        B = torch.randn(K, N, device="cuda", dtype=DTYPE)
        lat = run_bench(lambda: torch.matmul(A, B))
        gflops, gbps = compute_metrics(lat, (M * K + K * N) * 2, M * N * 2, flops)
        results.append({
            "kernel": "matmul",
            "config": f"{M}x{K}x{N}",
            "implementation": "PyTorch",
            "flops": flops, "bytes": bytes_total,
            "arithmetic_intensity": ai,
            "achieved_gflops": gflops, "achieved_gbps": gbps,
            "latency_ms": lat,
            "is_memory_bound": ai < RIDGE_AI,
        })

        # Triton
        try:
            from triton_kernels.matmul import matmul_triton
            A = torch.randn(M, K, device="cuda", dtype=DTYPE)
            B = torch.randn(K, N, device="cuda", dtype=DTYPE)
            lat = run_bench(lambda: matmul_triton(A, B))
            gflops, gbps = compute_metrics(lat, (M * K + K * N) * 2, M * N * 2, flops)
            results.append({
                "kernel": "matmul",
                "config": f"{M}x{K}x{N}",
                "implementation": "Triton",
                "flops": flops, "bytes": bytes_total,
                "arithmetic_intensity": ai,
                "achieved_gflops": gflops, "achieved_gbps": gbps,
                "latency_ms": lat,
                "is_memory_bound": ai < RIDGE_AI,
            })
        except Exception:
            pass
    return results


# ---------------------------------------------------------------------------
# Conv2D 3×3
# ---------------------------------------------------------------------------
def conv_flops_bytes(B, C_in, H, W, C_out, elem_bytes=2):
    out_h, out_w = H - 2, W - 2
    flops = 2 * B * C_out * out_h * out_w * C_in * 9
    bytes_total = B * C_in * H * W * elem_bytes + C_out * C_in * 9 * elem_bytes + C_out * elem_bytes + B * C_out * out_h * out_w * elem_bytes
    return flops, bytes_total


def run_roofline_conv():
    results = []
    configs = [(32, 1, 28, 28, 32), (128, 1, 28, 28, 32), (128, 1, 64, 64, 64)]
    for B, C_in, H, W, C_out in configs:
        flops, bytes_total = conv_flops_bytes(B, C_in, H, W, C_out)
        ai = flops / bytes_total if bytes_total > 0 else 0

        # PyTorch
        x = torch.randn(B, C_in, H, W, device="cuda", dtype=DTYPE)
        conv = torch.nn.Conv2d(C_in, C_out, 3, padding=0).to("cuda", dtype=DTYPE)
        w, b = conv.weight, conv.bias
        out_h, out_w = H - 2, W - 2
        lat = run_bench(lambda: torch.nn.functional.conv2d(x, w, b, padding=0))
        br = x.numel() * 2 + w.numel() * 2 + b.numel() * 2
        bw = B * C_out * out_h * out_w * 2
        gflops, gbps = compute_metrics(lat, br, bw, flops)
        results.append({
            "kernel": "conv2d",
            "config": f"B{B}_{C_in}x{C_out}_{H}x{W}",
            "implementation": "PyTorch",
            "flops": flops, "bytes": bytes_total,
            "arithmetic_intensity": ai,
            "achieved_gflops": gflops, "achieved_gbps": gbps,
            "latency_ms": lat,
            "is_memory_bound": ai < RIDGE_AI,
        })

        # Triton
        try:
            from triton_kernels.conv import conv2d_triton
            x = torch.randn(B, C_in, H, W, device="cuda", dtype=DTYPE)
            w = torch.randn(C_out, C_in, 3, 3, device="cuda", dtype=DTYPE)
            b = torch.zeros(C_out, device="cuda", dtype=DTYPE)
            br_tr = x.numel() * 2 + w.numel() * 2 + b.numel() * 2
            bw_tr = B * C_out * out_h * out_w * 2
            lat = run_bench(lambda: conv2d_triton(x, w, b, use_autotune=True))
            gflops, gbps = compute_metrics(lat, br_tr, bw_tr, flops)
            results.append({
                "kernel": "conv2d",
                "config": f"B{B}_{C_in}x{C_out}_{H}x{W}",
                "implementation": "Triton",
                "flops": flops, "bytes": bytes_total,
                "arithmetic_intensity": ai,
                "achieved_gflops": gflops, "achieved_gbps": gbps,
                "latency_ms": lat,
                "is_memory_bound": ai < RIDGE_AI,
            })
        except Exception:
            pass

        # CUDA (only C_in=1, C_out=32)
        if C_in == 1 and C_out == 32:
            try:
                import custom_conv
                x = torch.randn(B, C_in, H, W, device="cuda", dtype=torch.float32)
                conv = torch.nn.Conv2d(C_in, C_out, 3, padding=0).to("cuda", dtype=torch.float32)
                w, b = conv.weight, conv.bias
                br_f32 = x.numel() * 4 + w.numel() * 4 + b.numel() * 4
                bw_f32 = B * C_out * out_h * out_w * 4
                lat = run_bench(lambda: custom_conv.custom_conv2d(x, w, b)[0])
                gflops, gbps = compute_metrics(lat, br_f32, bw_f32, flops)
                bytes_cuda = br_f32 + bw_f32
                ai_cuda = flops / bytes_cuda if bytes_cuda > 0 else 0
                results.append({
                    "kernel": "conv2d",
                    "config": f"B{B}_{C_in}x{C_out}_{H}x{W}",
                    "implementation": "CUDA",
                    "flops": flops, "bytes": bytes_cuda,
                    "arithmetic_intensity": ai_cuda,
                    "achieved_gflops": gflops, "achieved_gbps": gbps,
                    "latency_ms": lat,
                    "is_memory_bound": ai_cuda < RIDGE_AI,
                })
            except Exception:
                pass
    return results


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
def attention_flops_bytes(B, H, S, D, elem_bytes=2):
    # Q@K^T: 2*B*H*S*D*S, attn@V: 2*B*H*S*S*D
    flops = 2 * B * H * S * D * S + 2 * B * H * S * S * D
    bytes_total = (B * H * S * D * 3 + B * H * S * D) * elem_bytes  # Q,K,V read + O write
    return flops, bytes_total


def run_roofline_attention():
    results = []
    configs = [(2, 8, 128, 64), (4, 8, 256, 64), (2, 8, 512, 64)]
    for B, H, S, D in configs:
        flops, bytes_total = attention_flops_bytes(B, H, S, D)
        ai = flops / bytes_total if bytes_total > 0 else 0

        for impl_name, run_fn in [("PyTorch", _run_attn_pytorch), ("Triton", _run_attn_triton)]:
            lat, gflops, gbps = run_fn(B, H, S, D)
            if lat is None:
                continue
            results.append({
                "kernel": "attention",
                "config": f"B{B}_H{H}_S{S}_D{D}",
                "implementation": impl_name,
                "flops": flops, "bytes": bytes_total,
                "arithmetic_intensity": ai,
                "achieved_gflops": gflops, "achieved_gbps": gbps,
                "latency_ms": lat,
                "is_memory_bound": ai < RIDGE_AI,
            })
    return results


def _run_attn_pytorch(B, H, S, D):
    q = torch.randn(B, H, S, D, device="cuda", dtype=DTYPE)
    k = torch.randn(B, H, S, D, device="cuda", dtype=DTYPE)
    v = torch.randn(B, H, S, D, device="cuda", dtype=DTYPE)
    lat = run_bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False))
    flops, bytes_total = attention_flops_bytes(B, H, S, D)
    br, bw = (B * H * S * D * 3) * 2, B * H * S * D * 2
    gflops, gbps = compute_metrics(lat, br, bw, flops)
    return lat, gflops, gbps


def _run_attn_triton(B, H, S, D):
    try:
        from triton_kernels.flash_attention import flash_attention_triton
        q = torch.randn(B, H, S, D, device="cuda", dtype=DTYPE)
        k = torch.randn(B, H, S, D, device="cuda", dtype=DTYPE)
        v = torch.randn(B, H, S, D, device="cuda", dtype=DTYPE)
        lat = run_bench(lambda: flash_attention_triton(q, k, v, causal=False))
        flops, bytes_total = attention_flops_bytes(B, H, S, D)
        br, bw = (B * H * S * D * 3) * 2, B * H * S * D * 2
        gflops, gbps = compute_metrics(lat, br, bw, flops)
        return lat, gflops, gbps
    except Exception:
        return None, None, None


# ---------------------------------------------------------------------------
# Transformer kernels (QKV, Softmax, LayerNorm, GELU, MLP)
# ---------------------------------------------------------------------------
def transformer_flops_bytes(B=8, S=256, H=768):
    M = B * S
    H3, H4 = H * 3, H * 4
    elem = 2
    out = []
    # QKV
    flops = 2 * M * H * H3
    bytes_qkv = M * H * elem + H * H3 * elem + H3 * elem + M * H3 * elem
    out.append(("QKV", flops, bytes_qkv))
    # Softmax
    flops_sm = 5 * M * H
    bytes_sm = M * H * elem * 2
    out.append(("Softmax", flops_sm, bytes_sm))
    # LayerNorm
    flops_ln = 4 * M * H
    bytes_ln = M * H * elem * 2 + H * elem * 2
    out.append(("LayerNorm", flops_ln, bytes_ln))
    # GELU
    n = M * H
    out.append(("GELU", 8 * n, n * elem * 2))
    # MLP
    flops_mlp = 2 * M * H * H4 + 2 * M * H4 * H + 8 * M * H4
    bytes_mlp = (M * H + H * H4 + H4 + M * H4) * elem + (M * H4 + H4 * H + H) * elem + M * H4 * elem
    out.append(("MLP", flops_mlp, bytes_mlp))
    return out


def run_roofline_transformer():
    from benchmarks.transformer_reference import (
        fused_qkv_pytorch,
        softmax_pytorch,
        layernorm_pytorch,
        gelu_pytorch,
        fused_mlp_pytorch,
    )
    B, S, H = 8, 256, 768
    M = B * S
    H3, H4 = H * 3, H * 4
    results = []
    theory = transformer_flops_bytes(B, S, H)

    runners_pytorch = [
        ("QKV", lambda: fused_qkv_pytorch(
            torch.randn(M, H, device="cuda", dtype=DTYPE),
            torch.randn(H, H3, device="cuda", dtype=DTYPE),
            torch.randn(H3, device="cuda", dtype=DTYPE),
        )),
        ("Softmax", lambda: softmax_pytorch(torch.randn(M, H, device="cuda", dtype=DTYPE), dim=-1)),
        ("LayerNorm", lambda: layernorm_pytorch(
            torch.randn(M, H, device="cuda", dtype=DTYPE),
            torch.ones(H, device="cuda", dtype=DTYPE),
            torch.zeros(H, device="cuda", dtype=DTYPE),
        )),
        ("GELU", lambda: gelu_pytorch(torch.randn(M * H, device="cuda", dtype=DTYPE))),
        ("MLP", lambda: fused_mlp_pytorch(
            torch.randn(M, H, device="cuda", dtype=DTYPE),
            torch.randn(H, H4, device="cuda", dtype=DTYPE),
            torch.randn(H4, device="cuda", dtype=DTYPE),
            torch.randn(H4, H, device="cuda", dtype=DTYPE),
            torch.randn(H, device="cuda", dtype=DTYPE),
        )),
    ]

    for (name, flops, bytes_total), (_, run_fn) in zip(theory, runners_pytorch):
        ai = flops / bytes_total if bytes_total > 0 else 0
        lat = run_bench(run_fn)
        gflops, gbps = compute_metrics(lat, bytes_total // 2, bytes_total // 2, flops if "GELU" not in name else None)
        if gflops is None:
            gflops = 0.0
        results.append({
            "kernel": "transformer",
            "config": name,
            "implementation": "PyTorch",
            "flops": flops, "bytes": bytes_total,
            "arithmetic_intensity": ai,
            "achieved_gflops": gflops, "achieved_gbps": gbps,
            "latency_ms": lat,
            "is_memory_bound": ai < RIDGE_AI,
        })

    # Triton
    try:
        from triton_kernels.qkv import fused_qkv_triton
        from triton_kernels.softmax import softmax_triton
        from triton_kernels.layernorm import layernorm_triton
        from triton_kernels.gelu import gelu_triton
        from triton_kernels.mlp import fused_mlp_triton
    except Exception:
        return results

    x_qkv = torch.randn(M, H, device="cuda", dtype=DTYPE)
    w_qkv = torch.randn(H, H3, device="cuda", dtype=DTYPE)
    b_qkv = torch.randn(H3, device="cuda", dtype=DTYPE)
    x_ln = torch.randn(M, H, device="cuda", dtype=DTYPE)
    w_ln = torch.ones(H, device="cuda", dtype=DTYPE)
    b_ln = torch.zeros(H, device="cuda", dtype=DTYPE)
    x_mlp1 = torch.randn(M, H, device="cuda", dtype=DTYPE)
    w1 = torch.randn(H, H4, device="cuda", dtype=DTYPE)
    b1 = torch.randn(H4, device="cuda", dtype=DTYPE)
    w2 = torch.randn(H4, H, device="cuda", dtype=DTYPE)
    b2 = torch.randn(H, device="cuda", dtype=DTYPE)

    triton_runners = [
        ("QKV", lambda: fused_qkv_triton(x_qkv, w_qkv, b_qkv)),
        ("Softmax", lambda: softmax_triton(torch.randn(M, H, device="cuda", dtype=DTYPE), dim=-1)),
        ("LayerNorm", lambda: layernorm_triton(x_ln, (H,), w_ln, b_ln)),
        ("GELU", lambda: gelu_triton(torch.randn(M * H, device="cuda", dtype=DTYPE))),
        ("MLP", lambda: fused_mlp_triton(x_mlp1, w1, b1, w2, b2)),
    ]
    for (name, flops, bytes_total), (_, run_fn) in zip(theory, triton_runners):
        ai = flops / bytes_total if bytes_total > 0 else 0
        try:
            lat = run_bench(run_fn)
            gflops, gbps = compute_metrics(lat, bytes_total // 2, bytes_total // 2, flops if "GELU" not in name else None)
            if gflops is None:
                gflops = 0.0
            results.append({
                "kernel": "transformer",
                "config": name,
                "implementation": "Triton",
                "flops": flops, "bytes": bytes_total,
                "arithmetic_intensity": ai,
                "achieved_gflops": gflops, "achieved_gbps": gbps,
                "latency_ms": lat,
                "is_memory_bound": ai < RIDGE_AI,
            })
        except Exception:
            pass
    return results


# ---------------------------------------------------------------------------
# Save results and plot
# ---------------------------------------------------------------------------
def save_results(all_results):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # CSV
    import csv
    fieldnames = ["kernel", "config", "implementation", "flops", "bytes", "arithmetic_intensity",
                  "achieved_gflops", "achieved_gbps", "latency_ms", "is_memory_bound"]
    csv_path = OUT_DIR / "roofline_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)
    print(f"CSV saved: {csv_path}")

    # JSON (serializable)
    json_path = OUT_DIR / "roofline_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON saved: {json_path}")


def plot_roofline(all_results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plots; pip install matplotlib")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Roofline curve: perf = min(PEAK_GBPS * AI, PEAK_GFLOPS)
    ridge = RIDGE_AI
    xs_curve = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    ys_curve = [min(PEAK_GBPS * x, PEAK_GFLOPS) for x in xs_curve]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(xs_curve, ys_curve, "k-", linewidth=2, label="Roofline")
    ax.axvline(ridge, color="gray", linestyle="--", alpha=0.7, label=f"Ridge (AI={ridge:.1f})")

    # Points: use achieved_gflops for y; arithmetic_intensity for x
    memory_bound = [r for r in all_results if r.get("is_memory_bound", True)]
    compute_bound = [r for r in all_results if not r.get("is_memory_bound", False)]

    for r in memory_bound:
        ai = max(r["arithmetic_intensity"], 0.01)
        perf = r.get("achieved_gflops") or 0.01
        label = f"{r['kernel']} {r['config']} ({r['implementation']})"
        ax.scatter([ai], [perf], c="C0", s=60, marker="o", alpha=0.8, edgecolors="black", linewidths=0.5)

    for r in compute_bound:
        ai = max(r["arithmetic_intensity"], 0.01)
        perf = r.get("achieved_gflops") or 0.01
        ax.scatter([ai], [perf], c="C1", s=60, marker="s", alpha=0.8, edgecolors="black", linewidths=0.5)

    # Legend: single markers for memory vs compute
    ax.scatter([], [], c="C0", s=80, marker="o", label="Memory-bound", edgecolors="black")
    ax.scatter([], [], c="C1", s=80, marker="s", label="Compute-bound", edgecolors="black")

    ax.set_xlabel("Arithmetic intensity (FLOPs/byte)")
    ax.set_ylabel("Performance (GFLOPS)")
    ax.set_title(f"Roofline model — RTX 3050 (peak {PEAK_GFLOPS:.0f} GFLOPS, {PEAK_GBPS:.0f} GB/s)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0.008)
    fig.tight_layout()
    out_path = OUT_DIR / "roofline_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_path}")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; run on GPU for roofline analysis.")
        return

    print(f"Roofline analysis — {torch.cuda.get_device_name(0)}")
    print(f"Peak: {PEAK_GFLOPS:.0f} GFLOPS, {PEAK_GBPS:.0f} GB/s, Ridge AI = {RIDGE_AI:.1f} FLOP/byte")
    print("Running kernels...")

    all_results = []
    all_results.extend(run_roofline_matmul())
    time.sleep(0.5)
    all_results.extend(run_roofline_conv())
    time.sleep(0.5)
    all_results.extend(run_roofline_attention())
    time.sleep(0.5)
    all_results.extend(run_roofline_transformer())

    save_results(all_results)
    plot_roofline(all_results)
    print("Done.")


if __name__ == "__main__":
    main()
