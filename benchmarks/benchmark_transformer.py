"""
Transformer building blocks: PyTorch vs Triton vs custom CUDA (FP16).
Kernels: QKV projection, Softmax, LayerNorm, GELU, MLP (GEMM -> bias+GELU -> GEMM -> bias).

Every implementation is first checked against the PyTorch reference (FP32 math) and is only
timed if it is correct; results above the hardware ceiling are rejected (bench_utils.measure).

Run from repo root:  python benchmarks/benchmark_transformer.py
Env:
  TRANSFORMER_BENCHMARK_CUDA=0   skip the custom CUDA extension (it is JIT-built on first use)
  TRANSFORMER_BENCHMARK_FULL=1   B=32, S=512 instead of B=8, S=256
"""
from __future__ import annotations

import os
import sys
from collections.abc import Callable
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
    save_csv,
)
from benchmarks.transformer_reference import (
    fused_mlp_pytorch,
    fused_qkv_pytorch,
    gelu_pytorch,
    layernorm_pytorch,
    softmax_pytorch,
)

USE_CUDA_EXT = os.environ.get("TRANSFORMER_BENCHMARK_CUDA", "1").strip().lower() not in ("0", "false", "no")
FULL = os.environ.get("TRANSFORMER_BENCHMARK_FULL", "").strip().lower() in ("1", "true", "yes")
B, S, H = (32, 512, 768) if FULL else (8, 256, 768)
M_LARGE = 8192  # rows for the second, launch-floor-free pass of the row-wise kernels
DTYPE = torch.float16
WARMUP, REPEAT = 10, 50
KERNELS = ["QKV", "Softmax", "LayerNorm", "GELU", "MLP"]
IMPLS = ["PyTorch", "Triton", "CUDA"]


def nbytes(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def _optional(import_fn: Callable[[], Callable]) -> Callable | None:
    try:
        return import_fn()
    except Exception as e:  # noqa: BLE001 - report and continue with the other implementations
        print(f"  [skip] import failed: {e}", file=sys.stderr)
        return None


def _triton(name: str):
    import importlib

    module, fn = {
        "qkv": ("triton_kernels.qkv", "fused_qkv_triton"),
        "softmax": ("triton_kernels.softmax", "softmax_triton"),
        "layernorm": ("triton_kernels.layernorm", "layernorm_triton"),
        "gelu": ("triton_kernels.gelu", "gelu_triton"),
        "mlp": ("triton_kernels.mlp", "fused_mlp_triton"),
    }[name]
    return _optional(lambda: getattr(importlib.import_module(module), fn))


def _cuda(name: str):
    if not USE_CUDA_EXT:
        return None
    from gpu_kernels.transformer import transformer_cuda as tc

    if tc._load_extension() is None:
        return None
    return {
        "qkv": tc.fused_qkv_cuda,
        "softmax": tc.softmax_cuda,
        "layernorm": tc.layernorm_cuda,
        "gelu": tc.gelu_cuda,
        "mlp": tc.fused_mlp_cuda,
    }[name]


def _row(kernel: str, impl: str, m: dict | None) -> dict | None:
    if m is None:
        return None
    return {
        "kernel": kernel,
        "implementation": impl,
        "latency_ms": round(m["latency_ms"], 4),
        "gflops": round(m["gflops"], 1) if "gflops" in m else None,
        "memory_bandwidth_gbs": round(m["memory_bandwidth_gbs"], 1),
    }


def run_case(kernel: str, candidates: list[tuple[str, Callable | None]], ref: torch.Tensor,
             bytes_r: int, bytes_w: int, flops: float | None) -> list[dict]:
    rows = []
    for impl, fn in candidates:
        if fn is None:
            continue
        r = _row(kernel, impl, measure(f"{kernel}/{impl}", fn, ref, bytes_r, bytes_w, flops,
                                       warmup=WARMUP, repeat=REPEAT))
        if r:
            rows.append(r)
    return rows


def bench_rowwise(M: int, tag: str) -> list[dict]:
    """Softmax, LayerNorm and GELU on an M x H FP16 tensor; `tag` is appended to the kernel name."""
    rows: list[dict] = []
    # Softmax over the last dim (attention-score-like rows)
    xs = torch.randn(M, H, device="cuda", dtype=DTYPE) * 3
    ref = torch.softmax(xs.float(), -1)
    t, c = _triton("softmax"), _cuda("softmax")
    rows += run_case("Softmax" + tag, [
        ("PyTorch", lambda: softmax_pytorch(xs, -1)),
        ("Triton", t and (lambda: t(xs, dim=-1))),
        ("CUDA", c and (lambda: c(xs, -1))),
    ], ref, nbytes(xs), nbytes(xs), None)

    # LayerNorm (non-trivial affine params so a broken gamma/beta path would be caught)
    xl = torch.randn(M, H, device="cuda", dtype=DTYPE) * 2 + 1
    g = torch.randn(H, device="cuda", dtype=DTYPE)
    be = torch.randn(H, device="cuda", dtype=DTYPE)
    ref = torch.nn.functional.layer_norm(xl.float(), (H,), g.float(), be.float(), 1e-5)
    t, c = _triton("layernorm"), _cuda("layernorm")
    rows += run_case("LayerNorm" + tag, [
        ("PyTorch", lambda: layernorm_pytorch(xl, g, be)),
        ("Triton", t and (lambda: t(xl, (H,), g, be))),
        ("CUDA", c and (lambda: c(xl, g, be, 1e-5))),
    ], ref, nbytes(xl, g, be), nbytes(xl), None)

    # GELU. Triton implements the exact (erf) form, the CUDA kernel the tanh form: each is
    # verified against its own definition, and both are timed on the same tensor.
    xg = torch.randn(M * H, device="cuda", dtype=DTYPE) * 2
    t, c = _triton("gelu"), _cuda("gelu")
    rows += run_case("GELU" + tag, [("PyTorch", lambda: gelu_pytorch(xg)), ("Triton", t and (lambda: t(xg)))],
                     torch.nn.functional.gelu(xg.float()), nbytes(xg), nbytes(xg), None)
    rows += run_case("GELU" + tag, [("CUDA", c and (lambda: c(xg)))],
                     torch.nn.functional.gelu(xg.float(), approximate="tanh"), nbytes(xg), nbytes(xg), None)
    return rows


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available; run on a GPU.")
        return
    torch.manual_seed(0)
    ensure_plots_dir()
    print(f"Transformer kernels (FP16) on {torch.cuda.get_device_name(0)}: B={B} S={S} H={H}, CUDA ext: {USE_CUDA_EXT}")
    M = B * S
    dev = "cuda"
    rows: list[dict] = []

    # QKV: y = x @ W_qkv + b
    x = torch.randn(M, H, device=dev, dtype=DTYPE)
    w = torch.randn(H, 3 * H, device=dev, dtype=DTYPE) / H**0.5
    b = torch.randn(3 * H, device=dev, dtype=DTYPE) * 0.1
    ref = (x.float() @ w.float() + b.float())
    t, c = _triton("qkv"), _cuda("qkv")
    rows += run_case("QKV", [
        ("PyTorch", lambda: fused_qkv_pytorch(x, w, b)),
        ("Triton", t and (lambda: t(x, w, b))),
        ("CUDA", c and (lambda: c(x, w, b))),
    ], ref, nbytes(x, w, b), M * 3 * H * 2, 2.0 * M * H * 3 * H)

    rows += bench_rowwise(M, "")
    # At 2048 x 768 a row-wise kernel takes ~0.05 ms, right at Triton's ~0.055 ms Python launch
    # floor, so those rows measure launch overhead more than the kernel. 8192 x 768 (~0.2 ms)
    # measures the kernels themselves.
    if M != M_LARGE:
        rows += bench_rowwise(M_LARGE, f"@{M_LARGE}x{H}")

    # MLP: H -> 4H -> H
    xm = torch.randn(M, H, device=dev, dtype=DTYPE)
    w1 = torch.randn(H, 4 * H, device=dev, dtype=DTYPE) / H**0.5
    b1 = torch.randn(4 * H, device=dev, dtype=DTYPE) * 0.1
    w2 = torch.randn(4 * H, H, device=dev, dtype=DTYPE) / (4 * H) ** 0.5
    b2 = torch.randn(H, device=dev, dtype=DTYPE) * 0.1
    ref = fused_mlp_pytorch(xm.float(), w1.float(), b1.float(), w2.float(), b2.float())
    t, c = _triton("mlp"), _cuda("mlp")
    rows += run_case("MLP", [
        ("PyTorch", lambda: fused_mlp_pytorch(xm, w1, b1, w2, b2)),
        ("Triton", t and (lambda: t(xm, w1, b1, w2, b2))),
        ("CUDA", c and (lambda: c(xm, w1, b1, w2, b2))),
    ], ref, nbytes(xm, w1, b1, w2, b2), M * H * 2, 2.0 * M * H * 4 * H * 2)

    fields = ["kernel", "implementation", "latency_ms", "gflops", "memory_bandwidth_gbs"]
    save_csv(rows, BENCHMARKS_DIR / "benchmark_transformer_results.csv", fieldnames=fields)
    print_benchmark_table(rows, title="Transformer kernels (verified)", columns=fields)

    for kernel in KERNELS:
        sub = [r for r in rows if r["kernel"] == kernel]
        if sub:
            plot_bar_chart([r["implementation"] for r in sub], [r["latency_ms"] for r in sub],
                           ylabel="Latency (ms)", title=f"Transformer {kernel} - latency",
                           filename=f"transformer_{kernel.lower()}_latency.png")
    data = {impl: [next((r["latency_ms"] for r in rows if r["kernel"] == k and r["implementation"] == impl), 0.0)
                   for k in KERNELS] for impl in IMPLS}
    plot_grouped_bars(KERNELS, IMPLS, data, ylabel="Latency (ms)", title="Transformer kernels latency",
                      filename="transformer_latency_by_kernel.png")
    print(f"CSV: benchmarks/benchmark_transformer_results.csv, charts: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
