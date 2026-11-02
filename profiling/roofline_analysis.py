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

import os

import torch

from benchmarks.bench_utils import check_close, run_bench
from benchmarks.gpu_specs import query as _query_specs

# Roof = FP16 Tensor Core with FP32 accumulation (what cuBLAS / tl.dot use for FP16 inputs),
# derived from the device's SM count and clocks (benchmarks/gpu_specs.py). An earlier version
# hard-coded 9 TFLOPS, which put the measured cuBLAS matmul points above the roof.
_specs = _query_specs()
PEAK_GFLOPS = float(os.environ.get("ROOFLINE_PEAK_GFLOPS", 0)) or (
    (_specs.peak_tflops("fp16_tc_fp32acc") or 0) * 1e3 if _specs else 0) or 21504.0
PEAK_GBPS = float(os.environ.get("ROOFLINE_PEAK_GBPS", 0)) or ((_specs.peak_dram_gbs() or 0) if _specs else 0) or 132.0
RIDGE_AI = PEAK_GFLOPS / PEAK_GBPS  # FLOPs/byte

OUT_DIR = REPO_ROOT / "profiling" / "roofline"
DTYPE = torch.float16


def _verified_bench(name, fn, ref, atol=2e-2, rtol=2e-2):
    """Verify -> time, with the same harness as benchmarks/: a wrong kernel raises BenchmarkError."""
    check_close(name, fn(), ref, atol=atol, rtol=rtol)
    return run_bench(fn)


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
        lat = _verified_bench("matmul/PyTorch", lambda: torch.matmul(A, B), A.float() @ B.float(), atol=0.5)
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
            lat = _verified_bench("matmul/Triton", lambda: matmul_triton(A, B), A.float() @ B.float(), atol=0.5)
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
        except Exception as e:  # noqa: BLE001 - report the failure, keep the other kernels
            print(f"  [skip] matmul/Triton {M}x{K}x{N}: {e}", file=sys.stderr)
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
        w, b = conv.weight.detach(), conv.bias.detach()
        out_h, out_w = H - 2, W - 2
        ref = torch.nn.functional.conv2d(x.float(), w.float(), b.float())
        lat = _verified_bench("conv/PyTorch", lambda: torch.nn.functional.conv2d(x, w, b, padding=0), ref, atol=5e-2, rtol=1e-2)
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
            ref = torch.nn.functional.conv2d(x.float(), w.float(), b.float())
            lat = _verified_bench("conv/Triton", lambda: conv2d_triton(x, w, b, use_autotune=True), ref, atol=5e-2, rtol=1e-2)
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
        except Exception as e:  # noqa: BLE001 - report the failure, keep the other kernels
            print(f"  [skip] conv/Triton B{B}: {e}", file=sys.stderr)

        # CUDA (only C_in=1, C_out=32)
        if C_in == 1 and C_out == 32:
            try:
                import custom_conv
                x = torch.randn(B, C_in, H, W, device="cuda", dtype=torch.float32)
                conv = torch.nn.Conv2d(C_in, C_out, 3, padding=0).to("cuda", dtype=torch.float32)
                w, b = conv.weight.detach(), conv.bias.detach()
                br_f32 = x.numel() * 4 + w.numel() * 4 + b.numel() * 4
                bw_f32 = B * C_out * out_h * out_w * 4
                ref = torch.nn.functional.conv2d(x, w, b)
                lat = _verified_bench("conv/CUDA", lambda: custom_conv.custom_conv2d(x, w, b)[0], ref, atol=5e-2, rtol=1e-2)
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
            except Exception as e:  # noqa: BLE001 - report the failure, keep the other kernels
                print(f"  [skip] conv/CUDA B{B}: {e}", file=sys.stderr)
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
        for impl_name, run_fn, elem_bytes in [("PyTorch", _run_attn_pytorch, 2), ("Triton", _run_attn_triton, 2),
                                              ("CUDA", _run_attn_cuda, 4)]:
            flops, bytes_total = attention_flops_bytes(B, H, S, D, elem_bytes)
            ai = flops / bytes_total if bytes_total > 0 else 0
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
    ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
    lat = _verified_bench("attention/PyTorch", lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False), ref)
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
        ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
        lat = _verified_bench("attention/Triton", lambda: flash_attention_triton(q, k, v, causal=False), ref)
        flops, bytes_total = attention_flops_bytes(B, H, S, D)
        br, bw = (B * H * S * D * 3) * 2, B * H * S * D * 2
        gflops, gbps = compute_metrics(lat, br, bw, flops)
        return lat, gflops, gbps
    except Exception as e:  # noqa: BLE001 - report the failure, keep the other kernels
        print(f"  [skip] attention/Triton S={S}: {e}", file=sys.stderr)
        return None, None, None


def _run_attn_cuda(B, H, S, D):
    """FP32 kernel, so its bytes (and arithmetic intensity) differ from the FP16 rows."""
    try:
        from flash_attention_simple.attention_cuda import attention_cuda
        q, k, v = (torch.randn(B, H, S, D, device="cuda", dtype=torch.float32) for _ in range(3))
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        lat = _verified_bench("attention/CUDA", lambda: attention_cuda(q, k, v), ref)
        flops, _ = attention_flops_bytes(B, H, S, D)
        gflops, gbps = compute_metrics(lat, (B * H * S * D * 3) * 4, B * H * S * D * 4, flops)
        return lat, gflops, gbps
    except Exception as e:  # noqa: BLE001 - report the failure, keep the other kernels
        print(f"  [skip] attention/CUDA S={S}: {e}", file=sys.stderr)
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
        fused_mlp_pytorch,
        fused_qkv_pytorch,
        gelu_pytorch,
        layernorm_pytorch,
        softmax_pytorch,
    )
    from gpu_kernels.transformer import transformer_cuda as tc
    from triton_kernels.gelu import gelu_triton
    from triton_kernels.layernorm import layernorm_triton
    from triton_kernels.mlp import fused_mlp_triton
    from triton_kernels.qkv import fused_qkv_triton
    from triton_kernels.softmax import softmax_triton

    B, S, H = 8, 256, 768
    M = B * S
    H3, H4 = H * 3, H * 4
    results = []
    theory = {name: (flops, nbytes) for name, flops, nbytes in transformer_flops_bytes(B, S, H)}

    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=DTYPE)

    # Inputs are created once, outside the timed region.
    x, w, b = rand(M, H), rand(H, H3) / H**0.5, rand(H3) * 0.1
    xs = rand(M, H) * 3
    xl, g, be = rand(M, H) * 2 + 1, rand(H), rand(H)
    xg = rand(M * H) * 2
    xm, w1, b1 = rand(M, H), rand(H, H4) / H**0.5, rand(H4) * 0.1
    w2, b2 = rand(H4, H) / H4**0.5, rand(H) * 0.1

    qkv_ref = x.float() @ w.float() + b.float()
    softmax_ref = torch.softmax(xs.float(), -1)
    ln_ref = torch.nn.functional.layer_norm(xl.float(), (H,), g.float(), be.float(), 1e-5)
    gelu_ref = torch.nn.functional.gelu(xg.float())
    gelu_tanh_ref = torch.nn.functional.gelu(xg.float(), approximate="tanh")
    mlp_ref = fused_mlp_pytorch(xm.float(), w1.float(), b1.float(), w2.float(), b2.float())

    cases = [
        ("QKV", "PyTorch", lambda: fused_qkv_pytorch(x, w, b), qkv_ref),
        ("Softmax", "PyTorch", lambda: softmax_pytorch(xs, dim=-1), softmax_ref),
        ("LayerNorm", "PyTorch", lambda: layernorm_pytorch(xl, g, be), ln_ref),
        ("GELU", "PyTorch", lambda: gelu_pytorch(xg), gelu_ref),
        ("MLP", "PyTorch", lambda: fused_mlp_pytorch(xm, w1, b1, w2, b2), mlp_ref),
        ("QKV", "Triton", lambda: fused_qkv_triton(x, w, b), qkv_ref),
        ("Softmax", "Triton", lambda: softmax_triton(xs, dim=-1), softmax_ref),
        ("LayerNorm", "Triton", lambda: layernorm_triton(xl, (H,), g, be), ln_ref),
        ("GELU", "Triton", lambda: gelu_triton(xg), gelu_ref),
        ("MLP", "Triton", lambda: fused_mlp_triton(xm, w1, b1, w2, b2), mlp_ref),
        ("QKV", "CUDA", lambda: tc.fused_qkv_cuda(x, w, b), qkv_ref),
        ("Softmax", "CUDA", lambda: tc.softmax_cuda(xs, -1), softmax_ref),
        ("LayerNorm", "CUDA", lambda: tc.layernorm_cuda(xl, g, be, 1e-5), ln_ref),
        ("GELU", "CUDA", lambda: tc.gelu_cuda(xg), gelu_tanh_ref),  # the CUDA kernel implements the tanh form
        ("MLP", "CUDA", lambda: tc.fused_mlp_cuda(xm, w1, b1, w2, b2), mlp_ref),
    ]
    for name, impl, run_fn, ref in cases:
        flops, bytes_total = theory[name]
        ai = flops / bytes_total if bytes_total > 0 else 0
        try:
            lat = _verified_bench(f"{name}/{impl}", run_fn, ref)
        except Exception as e:  # noqa: BLE001 - report the failure, keep the other kernels
            print(f"  [skip] {name}/{impl}: {e}", file=sys.stderr)
            continue
        gflops, gbps = compute_metrics(lat, bytes_total // 2, bytes_total // 2, flops)
        results.append({
            "kernel": "transformer",
            "config": name,
            "implementation": impl,
            "flops": flops, "bytes": bytes_total,
            "arithmetic_intensity": ai,
            "achieved_gflops": gflops, "achieved_gbps": gbps,
            "latency_ms": lat,
            "is_memory_bound": ai < RIDGE_AI,
        })
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


LABEL_OFFSETS = {
    "attention": ((-40, -75), "right"),
    "MLP": ((-15, -100), "center"),
    "QKV": ((-30, 45), "right"),
    "matmul": ((10, 32), "center"),
}


def plot_roofline(all_results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plots; pip install matplotlib")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ridge = RIDGE_AI
    xs_curve = sorted([10 ** (e / 10) for e in range(-20, 36)] + [ridge])  # 0.01 .. ~3000 FLOP/byte
    ys_curve = [min(PEAK_GBPS * x, PEAK_GFLOPS) for x in xs_curve]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(xs_curve, ys_curve, color="black", linewidth=2, label="Roofline (FP16 Tensor Core, FP32 acc)")
    ax.axvline(ridge, color="gray", linestyle="--", alpha=0.7, label=f"Ridge point (AI = {ridge:.0f} FLOP/byte)")

    styles = {"PyTorch": ("o", "#1f77b4"), "Triton": ("s", "#ff7f0e"), "CUDA": ("^", "#2ca02c")}
    seen = set()
    # Label one point per kernel (largest config, best implementation) so the labels stay readable.
    labeled = {}
    for r in all_results:
        category = r["config"] if r["kernel"] == "transformer" else r["kernel"]
        roof = min(PEAK_GBPS * r["arithmetic_intensity"], PEAK_GFLOPS)
        rank = (r["flops"], r["achieved_gflops"] / roof)
        if category not in labeled or rank > labeled[category][0]:
            labeled[category] = (rank, id(r))
    labeled_ids = {v[1] for v in labeled.values()}
    for r in all_results:
        ai = r["arithmetic_intensity"]
        # Recompute from FLOPs and latency (elementwise kernels were stored with 0 GFLOPS before).
        perf = r["flops"] / (r["latency_ms"] * 1e-3) / 1e9 if r.get("latency_ms") else r.get("achieved_gflops")
        if not ai or not perf:
            continue
        marker, color = styles.get(r["implementation"], ("D", "gray"))
        label = r["implementation"] if r["implementation"] not in seen else None
        seen.add(r["implementation"])
        ax.scatter([ai], [perf], marker=marker, c=color, s=60, edgecolors="black", linewidths=0.5, label=label, zorder=3)
        roof = min(PEAK_GBPS * ai, PEAK_GFLOPS)
        name = r["config"] if r["kernel"] == "transformer" else f"{r['kernel']} {r['config']}"
        if id(r) in labeled_ids:
            # The compute-bound points cluster near the roof, so their labels get leader lines.
            offset, ha = LABEL_OFFSETS.get(r["config"] if r["kernel"] == "transformer" else r["kernel"], ((-6, 8), "right"))
            ax.annotate(f"{name} {r['implementation']} ({100 * perf / roof:.0f}%)", (ai, perf),
                        textcoords="offset points", xytext=offset, ha=ha, fontsize=8,
                        arrowprops={"arrowstyle": "-", "color": "gray", "linewidth": 0.6})

    ax.text(ridge * 0.6, PEAK_GFLOPS * 0.02, "memory-bound", ha="right", fontsize=10, color="gray")
    ax.text(ridge * 1.6, PEAK_GFLOPS * 0.02, "compute-bound", ha="left", fontsize=10, color="gray")
    ax.set_xlabel("Arithmetic intensity (FLOPs / DRAM byte, compulsory traffic)")
    ax.set_ylabel("Achieved GFLOPS")
    ax.set_title(f"Roofline - {PEAK_GFLOPS / 1e3:.1f} TFLOPS FP16 TC, {PEAK_GBPS:.0f} GB/s  (labels: % of roof)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(0.3, 3000)
    ax.set_ylim(10, PEAK_GFLOPS * 2)
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
