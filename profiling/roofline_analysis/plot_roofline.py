"""
Roofline model plot for transformer kernel benchmarks.

Shows:
  - Memory-bound kernels (below ridge point)
  - Compute-bound kernels (above ridge point)
  - Roofline curve: performance = min(peak_BW * intensity, peak_FLOPS)

Uses profiling/nsight_reports/<prefix>_kernels_metrics.json if present,
and/or theoretical FLOPs/bytes from benchmark config (B, S, H).

Run from repo root:
  python profiling/roofline_analysis/plot_roofline.py

Output: profiling/nsight_reports/roofline_model.png
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NSIGHT_REPORTS = REPO_ROOT / "profiling" / "nsight_reports"
DEFAULT_PREFIX = "transformer"

# RTX 3050 (Laptop) typical specs - override with env or pass if needed
# FP16 Tensor Core peak ~9 TFLOPS, memory BW ~192 GB/s
PEAK_GFLOPS_FP16 = float(__import__("os").environ.get("ROOFLINE_PEAK_GFLOPS", "9000"))
PEAK_GBPS = float(__import__("os").environ.get("ROOFLINE_PEAK_GBPS", "192"))


def theoretical_flops_bytes(B: int, S: int, H: int) -> list[tuple[str, float, float]]:
    """Return (kernel_label, FLOPs, bytes) for transformer kernels (FP16)."""
    M = B * S
    H3 = H * 3
    H4 = H * 4
    # Element size FP16 = 2 bytes
    results = []
    # Fused QKV: y = x @ W + b; x (M, H), W (H, 3*H)
    flops_qkv = 2 * M * H * H3
    bytes_qkv = M * H * 2 + H * H3 * 2 + H3 * 2 + M * H3 * 2
    results.append(("QKV (fused)", flops_qkv, bytes_qkv))
    # Softmax: read M*H, write M*H
    flops_sm = 5 * M * H  # approx (max, exp, sum, div)
    bytes_sm = M * H * 2 * 2
    results.append(("Softmax", flops_sm, bytes_sm))
    # LayerNorm: read M*H, write M*H
    flops_ln = 4 * M * H
    bytes_ln = M * H * 2 * 2 + H * 2 * 2
    results.append(("LayerNorm", flops_ln, bytes_ln))
    # GELU: read+write n
    n_gelu = M * H
    flops_gelu = 8 * n_gelu
    bytes_gelu = n_gelu * 2 * 2
    results.append(("GELU", flops_gelu, bytes_gelu))
    # MLP: linear1 + GELU + linear2
    flops_mlp = 2 * M * H * H4 + 2 * M * H4 * H + (8 * M * H4)
    bytes_mlp = M * H * 2 + H * H4 * 2 + H4 * 2 + M * H4 * 2 + M * H4 * 2 + H4 * H * 2 + H * 2
    results.append(("MLP (fused)", flops_mlp, bytes_mlp))
    return results


def load_nsight_metrics(prefix: str) -> list[dict]:
    """Load kernels metrics from nsight_reports JSON."""
    path = NSIGHT_REPORTS / f"{prefix}_kernels_metrics.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("kernels", [])


def kernel_to_roofline_point(
    name: str,
    flops: float,
    bytes_total: float,
    duration_ns: float | None = None,
    achieved_gflops: float | None = None,
    achieved_gbps: float | None = None,
) -> tuple[float, float, str, bool]:
    """
    Return (arithmetic_intensity, performance_GFLOPS, label, is_memory_bound).
    Ridge point = PEAK_GFLOPS_FP16 / PEAK_GBPS.
    """
    if flops <= 0 or bytes_total <= 0:
        return (0.0, 0.0, name, True)
    intensity = flops / bytes_total
    ridge = PEAK_GFLOPS_FP16 / PEAK_GBPS
    is_memory_bound = intensity < ridge
    if duration_ns and duration_ns > 0:
        perf = (flops / 1e9) / (duration_ns / 1e9)
    elif achieved_gflops is not None and achieved_gflops > 0:
        perf = achieved_gflops
    else:
        # Theoretical ceiling: min(memory ceiling, compute ceiling)
        perf = min(PEAK_GBPS * intensity, PEAK_GFLOPS_FP16)
    return (intensity, perf, name, is_memory_bound)


def build_roofline_data(
    prefix: str,
    B: int = 8,
    S: int = 256,
    H: int = 768,
) -> tuple[list[tuple[float, float, str, bool]], list[tuple[float, float]]]:
    """
    Build (points, roof_curve).
    points = [(intensity, perf_gflops, label, is_memory_bound), ...]
    roof_curve = [(intensity, perf), ...] for the roofline line.
    """
    theory = theoretical_flops_bytes(B, S, H)
    metrics = load_nsight_metrics(prefix)
    # 只依「理論 kernel 名稱」匹配 ncu 數據，避免 cuBLAS 內建名稱（如 OKAL RF (Best)）汙染圖
    def get_duration_and_achieved(kernel_label: str) -> tuple[float | None, float | None]:
        key = kernel_label.split()[0].lower()  # qkv, softmax, layernorm, gelu, mlp
        for k in metrics:
            kn = (k.get("kernel") or "").lower()
            # 只接受名稱與我們 kernel 明顯相關的（不含 gemm/addmm 等泛匹配，避免單一 matmul 套到全部）
            if key in kn:
                m = k.get("metrics") or {}
                dur = None
                gflops = None
                for mk, mv in m.items():
                    if "gpu__time_duration" in mk and isinstance(mv, dict):
                        v = mv.get("value")
                        if isinstance(v, (int, float)):
                            dur = float(v)
                    if "sm__sass_throughput" in mk and isinstance(mv, dict):
                        v = mv.get("value")
                        if isinstance(v, (int, float)) and v > 0:
                            gflops = (v / 100.0) * PEAK_GFLOPS_FP16
                return (dur, gflops)
        return (None, None)

    # 只畫我們定義的 5 個 transformer kernel，不從 metrics 加其他 kernel（避免 OKAL RF 等怪點）
    points = []
    for label, flops, bytes_total in theory:
        dur, achieved = get_duration_and_achieved(label)
        intensity, perf, _, is_mem = kernel_to_roofline_point(
            label, flops, bytes_total, duration_ns=dur, achieved_gflops=achieved
        )
        points.append((intensity, perf, label, is_mem))

    # Roofline curve: x = intensity, y = min(PEAK_GBPS * x, PEAK_GFLOPS_FP16)
    ridge = PEAK_GFLOPS_FP16 / PEAK_GBPS
    max_int = max((p[0] for p in points), default=ridge * 2) * 1.2
    curve = []
    for x in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]:
        if x > max_int:
            break
        y = min(PEAK_GBPS * x, PEAK_GFLOPS_FP16)
        curve.append((x, y))
    curve.append((ridge, PEAK_GFLOPS_FP16))
    curve.sort(key=lambda t: t[0])
    return points, curve


def plot_roofline(
    prefix: str = DEFAULT_PREFIX,
    B: int = 8,
    S: int = 256,
    H: int = 768,
    out_path: Path | None = None,
) -> Path:
    """Generate roofline model plot; save to out_path or nsight_reports/roofline_model.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib required: pip install matplotlib")

    points, curve = build_roofline_data(prefix, B=B, S=S, H=H)
    out_path = out_path or NSIGHT_REPORTS / "roofline_model.png"
    NSIGHT_REPORTS.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Roofline curve
    xs, ys = zip(*curve)
    ax.plot(xs, ys, "k-", linewidth=2, label="Roofline")

    # Ridge point
    ridge = PEAK_GFLOPS_FP16 / PEAK_GBPS
    ax.axvline(ridge, color="gray", linestyle="--", alpha=0.7, label=f"Ridge (AI={ridge:.1f})")

    # 每個 kernel 單獨畫點並用圖例標名，避免點旁註解重疊（尤其右上角）
    for intensity, perf, label, is_mem in points:
        color, marker = ("C0", "o") if is_mem else ("C1", "s")
        ax.scatter([intensity], [perf], c=color, s=80, marker=marker, label=label, zorder=5)

    ax.set_xlabel("Arithmetic intensity (FLOPs/byte)")
    ax.set_ylabel("Performance (GFLOPS)")
    ax.set_title(f"Roofline model (peak FP16 {PEAK_GFLOPS_FP16:.0f} GFLOPS, BW {PEAK_GBPS:.0f} GB/s)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Roofline plot saved: {out_path}")
    return out_path


def main():
    import argparse
    p = argparse.ArgumentParser(description="Plot roofline model for transformer kernels")
    p.add_argument("--prefix", default=DEFAULT_PREFIX, help="Report prefix (e.g. transformer)")
    p.add_argument("--B", type=int, default=8, help="Batch size")
    p.add_argument("--S", type=int, default=256, help="Sequence length")
    p.add_argument("--H", type=int, default=768, help="Hidden size")
    p.add_argument("-o", "--output", type=Path, default=None, help="Output PNG path")
    args = p.parse_args()
    plot_roofline(
        prefix=args.prefix,
        B=args.B,
        S=args.S,
        H=args.H,
        out_path=args.output,
    )


if __name__ == "__main__":
    main()
