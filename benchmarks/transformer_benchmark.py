"""
Transformer kernels benchmark: PyTorch reference vs Triton vs CUDA (FP16).
Target: RTX 3050. Reports latency (ms), throughput (GFLOPS / GB/s), memory bandwidth (GB/s).
Run from repo root: python benchmarks/transformer_benchmark.py

Default: 較小規模與較少迭代以降低 GPU 負荷，且不跑自訂 CUDA kernel（避免當機）。
Env:
  TRANSFORMER_BENCHMARK_FULL=1   → 完整規模 B=32 S=512、較多迭代（高負載）
  TRANSFORMER_BENCHMARK_CUDA=1   → 啟用自訂 CUDA kernels（會明顯提高 GPU 使用率）
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

# 預設：較小規模 + 少迭代，降低 GPU 負荷與當機風險
_full = os.environ.get("TRANSFORMER_BENCHMARK_FULL", "").strip().lower() in ("1", "true", "yes")
_enable_cuda = os.environ.get("TRANSFORMER_BENCHMARK_CUDA", "").strip().lower() in ("1", "true", "yes")
SKIP_CUDA = not _enable_cuda

if _full:
    B, S, H = 32, 512, 768
    WARMUP, REPEAT = 15, 100
else:
    B, S, H = 8, 256, 768
    # 啟用 CUDA 時少跑幾次，縮短整體 100% 時間
    WARMUP, REPEAT = (5, 8) if _enable_cuda else (5, 15)

# 每個 benchmark 段落之間休息（秒）；啟用 CUDA 時加長休息避免長時間 100%
REST_BETWEEN_SECTIONS = 1.2 if _enable_cuda else 0.4

# 非 FULL 時只跑 Fused QKV 的 CUDA；FULL 時也只跑 QKV + MLP 的 CUDA
# Softmax/LayerNorm/GELU 自訂 CUDA 在部分環境會卡住，預設不跑（設 TRANSFORMER_BENCHMARK_CUDA_ALL=1 可嘗試）
RUN_CUDA_OTHER = _enable_cuda and _full
RUN_CUDA_SOFTMAX_LAYERNORM_GELU = os.environ.get("TRANSFORMER_BENCHMARK_CUDA_ALL", "0").strip().lower() in ("1", "true", "yes")

# 自訂 CUDA kernel 較慢，只測少次以降低 100% GPU 時間（仍可得到延遲數據）
CUDA_WARMUP, CUDA_REPEAT = 2, 5

H3 = H * 3   # QKV
H4 = H * 4   # MLP intermediate
DTYPE = torch.float16


def _bytes(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def _bench(fn, warmup=WARMUP, repeat=REPEAT):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repeat  # ms


def _report(name: str, latency_ms: float, bytes_read: int, bytes_written: int, flops: float | None = None):
    time_sec = latency_ms / 1000.0
    bw_gbs = (bytes_read + bytes_written) / 1e9 / time_sec  # GB/s
    line = f"  {name}: latency={latency_ms:.4f} ms, mem_bandwidth={bw_gbs:.2f} GB/s"
    if flops is not None and flops > 0:
        gflops = flops / 1e9 / time_sec  # GFLOPS
        line += f", throughput={gflops:.1f} GFLOPS"
    print(line)


def bench_qkv():
    M, K, N = B * S, H, H3
    x = torch.randn(M, K, device="cuda", dtype=DTYPE)
    w = torch.randn(K, N, device="cuda", dtype=DTYPE)
    b = torch.randn(N, device="cuda", dtype=DTYPE)
    bytes_read = _bytes(x, w, b)
    bytes_written = M * N * 2
    flops = 2 * M * N * K

    print("\n--- Fused QKV projection (FP16) ---")
    ref = _bench(lambda: fused_qkv_pytorch(x, w, b))
    _report("PyTorch", ref, bytes_read, bytes_written, flops)

    try:
        from triton_kernels.qkv import fused_qkv_triton
        t_tr = _bench(lambda: fused_qkv_triton(x, w, b))
        _report("Triton", t_tr, bytes_read, bytes_written, flops)
    except Exception as e:
        print(f"  Triton: skipped ({e})")

    if not SKIP_CUDA:
        try:
            from gpu_kernels.transformer.transformer_cuda import fused_qkv_cuda
            out = fused_qkv_cuda(x, w, b)
            if out is not None:
                t_cuda = _bench(lambda: fused_qkv_cuda(x, w, b), warmup=CUDA_WARMUP, repeat=CUDA_REPEAT)
                _report("CUDA", t_cuda, bytes_read, bytes_written, flops)
            else:
                print("  CUDA: extension not built")
        except Exception as e:
            print(f"  CUDA: skipped ({e})")
    else:
        print("  CUDA: skipped (default low load; set TRANSFORMER_BENCHMARK_CUDA=1 to enable)")


def bench_softmax():
    M, N = B * S, H
    x = torch.randn(M, N, device="cuda", dtype=DTYPE)
    bytes_read = _bytes(x)
    bytes_written = x.numel() * 2

    print("\n--- Softmax (last dim, FP16) ---")
    ref = _bench(lambda: softmax_pytorch(x, dim=-1))
    _report("PyTorch", ref, bytes_read, bytes_written)

    try:
        from triton_kernels.softmax import softmax_triton
        t_tr = _bench(lambda: softmax_triton(x, dim=-1))
        _report("Triton", t_tr, bytes_read, bytes_written)
    except Exception as e:
        print(f"  Triton: skipped ({e})")

    if not SKIP_CUDA and RUN_CUDA_SOFTMAX_LAYERNORM_GELU:
        try:
            from gpu_kernels.transformer.transformer_cuda import softmax_cuda
            out = softmax_cuda(x, -1)
            if out is not None:
                t_cuda = _bench(lambda: softmax_cuda(x, -1), warmup=CUDA_WARMUP, repeat=CUDA_REPEAT)
                _report("CUDA", t_cuda, bytes_read, bytes_written)
            else:
                print("  CUDA: extension not built")
        except Exception as e:
            print(f"  CUDA: skipped ({e})")
    elif not SKIP_CUDA:
        print("  CUDA: skipped (may hang on this device; set TRANSFORMER_BENCHMARK_CUDA_ALL=1 to try)")
    else:
        print("  CUDA: skipped (default low load; set TRANSFORMER_BENCHMARK_CUDA=1 to enable)")


def bench_layernorm():
    M, N = B * S, H
    x = torch.randn(M, N, device="cuda", dtype=DTYPE)
    weight = torch.ones(N, device="cuda", dtype=DTYPE)
    bias = torch.zeros(N, device="cuda", dtype=DTYPE)
    bytes_read = _bytes(x, weight, bias)
    bytes_written = x.numel() * 2

    print("\n--- LayerNorm (FP16) ---")
    ref = _bench(lambda: layernorm_pytorch(x, weight, bias))
    _report("PyTorch", ref, bytes_read, bytes_written)

    try:
        from triton_kernels.layernorm import layernorm_triton
        t_tr = _bench(lambda: layernorm_triton(x, (N,), weight, bias))
        _report("Triton", t_tr, bytes_read, bytes_written)
    except Exception as e:
        print(f"  Triton: skipped ({e})")

    if not SKIP_CUDA and RUN_CUDA_SOFTMAX_LAYERNORM_GELU:
        try:
            from gpu_kernels.transformer.transformer_cuda import layernorm_cuda
            out = layernorm_cuda(x, weight, bias, 1e-5)
            if out is not None:
                t_cuda = _bench(lambda: layernorm_cuda(x, weight, bias, 1e-5), warmup=CUDA_WARMUP, repeat=CUDA_REPEAT)
                _report("CUDA", t_cuda, bytes_read, bytes_written)
            else:
                print("  CUDA: extension not built")
        except Exception as e:
            print(f"  CUDA: skipped ({e})")
    elif not SKIP_CUDA:
        print("  CUDA: skipped (may hang; set TRANSFORMER_BENCHMARK_CUDA_ALL=1 to try)")
    else:
        print("  CUDA: skipped (default low load; set TRANSFORMER_BENCHMARK_CUDA=1 to enable)")


def bench_gelu():
    n = B * S * H
    x = torch.randn(n, device="cuda", dtype=DTYPE)
    bytes_read = _bytes(x)
    bytes_written = x.numel() * 2

    print("\n--- GELU (FP16) ---")
    ref = _bench(lambda: gelu_pytorch(x))
    _report("PyTorch", ref, bytes_read, bytes_written)

    try:
        from triton_kernels.gelu import gelu_triton
        t_tr = _bench(lambda: gelu_triton(x))
        _report("Triton", t_tr, bytes_read, bytes_written)
    except Exception as e:
        print(f"  Triton: skipped ({e})")

    if not SKIP_CUDA and RUN_CUDA_SOFTMAX_LAYERNORM_GELU:
        try:
            from gpu_kernels.transformer.transformer_cuda import gelu_cuda
            out = gelu_cuda(x)
            if out is not None:
                t_cuda = _bench(lambda: gelu_cuda(x), warmup=CUDA_WARMUP, repeat=CUDA_REPEAT)
                _report("CUDA", t_cuda, bytes_read, bytes_written)
            else:
                print("  CUDA: extension not built")
        except Exception as e:
            print(f"  CUDA: skipped ({e})")
    elif not SKIP_CUDA:
        print("  CUDA: skipped (may hang; set TRANSFORMER_BENCHMARK_CUDA_ALL=1 to try)")
    else:
        print("  CUDA: skipped (default low load; set TRANSFORMER_BENCHMARK_CUDA=1 to enable)")


def bench_mlp():
    M, K, N1, N2 = B * S, H, H4, H
    x = torch.randn(M, K, device="cuda", dtype=DTYPE)
    w1 = torch.randn(K, N1, device="cuda", dtype=DTYPE)
    b1 = torch.randn(N1, device="cuda", dtype=DTYPE)
    w2 = torch.randn(N1, N2, device="cuda", dtype=DTYPE)
    b2 = torch.randn(N2, device="cuda", dtype=DTYPE)
    bytes_read = _bytes(x, w1, b1, w2, b2) + M * N1 * 2  # + intermediate
    bytes_written = M * N2 * 2
    flops = 2 * M * K * N1 + 2 * M * N1 * N2

    print("\n--- Fused MLP block (FP16) ---")
    ref = _bench(lambda: fused_mlp_pytorch(x, w1, b1, w2, b2))
    _report("PyTorch", ref, bytes_read, bytes_written, flops)

    try:
        from triton_kernels.mlp import fused_mlp_triton
        t_tr = _bench(lambda: fused_mlp_triton(x, w1, b1, w2, b2))
        _report("Triton", t_tr, bytes_read, bytes_written, flops)
    except Exception as e:
        print(f"  Triton: skipped ({e})")

    if not SKIP_CUDA:
        # Fused MLP 仍為舊 16x16 tiled kernel，非常慢，非 FULL 模式不跑以免卡住
        if not _full:
            print("  CUDA: skipped (Fused MLP heavy; set TRANSFORMER_BENCHMARK_FULL=1 to enable)")
        else:
            try:
                from gpu_kernels.transformer.transformer_cuda import fused_mlp_cuda
                out = fused_mlp_cuda(x, w1, b1, w2, b2)
                if out is not None:
                    t_cuda = _bench(lambda: fused_mlp_cuda(x, w1, b1, w2, b2), warmup=CUDA_WARMUP, repeat=CUDA_REPEAT)
                    _report("CUDA", t_cuda, bytes_read, bytes_written, flops)
                else:
                    print("  CUDA: extension not built")
            except Exception as e:
                print(f"  CUDA: skipped ({e})")
    else:
        print("  CUDA: skipped (default low load; set TRANSFORMER_BENCHMARK_CUDA=1 to enable)")


def _rest():
    """段落間短暫休息，降低 GPU 持續滿載。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if REST_BETWEEN_SECTIONS > 0:
        time.sleep(REST_BETWEEN_SECTIONS)


def main():
    print("Transformer kernels benchmark (FP16)")
    print(f"  B={B}, S={S}, H={H}  (batch*seq, hidden)")
    if _full:
        print("  [Full mode: high GPU load]")
    else:
        print("  [Default: reduced load to avoid crash]")
    if SKIP_CUDA:
        print("  [Custom CUDA kernels off; PyTorch + Triton only]")
    elif not RUN_CUDA_SOFTMAX_LAYERNORM_GELU:
        print("  [CUDA: only QKV + MLP; Softmax/LayerNorm/GELU skipped to avoid hang]")
    if not torch.cuda.is_available():
        print("  No CUDA; run on GPU for meaningful results.")
        return
    print(f"  Device: {torch.cuda.get_device_name(0)}")

    bench_qkv()
    _rest()
    bench_softmax()
    _rest()
    bench_layernorm()
    _rest()
    bench_gelu()
    _rest()
    bench_mlp()

    print("\nDone.")


if __name__ == "__main__":
    main()
