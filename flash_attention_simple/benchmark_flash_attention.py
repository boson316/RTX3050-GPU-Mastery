"""
Benchmark: PyTorch attention vs CUDA attention vs Triton attention.

Sequence lengths: 128, 256, 512, 1024.
Fixed: B=2, H=8, D=64 (CUDA kernel requires D<=64). All use float32 for fair comparison.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

# Head dim 64 for Triton/PyTorch; CUDA kernel supports up to 64
B, H, D = 2, 8, 64
SEQ_LENGTHS = [128, 256, 512, 1024]
WARMUP = 15
REPEAT = 50


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
    return (time.perf_counter() - start) * 1000 / repeat


def main():
    if not torch.cuda.is_available():
        print("CUDA not available. Skip benchmark.")
        return

    from flash_attention_simple.reference_pytorch import attention_pytorch
    from flash_attention_simple.attention_triton import attention_triton
    from flash_attention_simple.attention_cuda import attention_cuda

    print("=" * 70)
    print("FlashAttention benchmark: PyTorch vs CUDA vs Triton")
    print("  B=%d, H=%d, D=%d, seq lengths: %s" % (B, H, D, SEQ_LENGTHS))
    print("  float32, attention(Q,K,V) = softmax(Q K^T / sqrt(d)) V")
    print("=" * 70)

    results = []
    for S in SEQ_LENGTHS:
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
        scale = 1.0 / (D ** 0.5)

        row = {"S": S}

        # PyTorch reference (full attention matrix)
        try:
            def run_pytorch():
                attention_pytorch(q, k, v, scale=scale)
            row["pytorch_ms"] = _bench(run_pytorch)
        except Exception as e:
            row["pytorch_ms"] = None
            row["pytorch_err"] = str(e)

        # Triton (tiled, no full matrix)
        try:
            def run_triton():
                attention_triton(q, k, v, scale=scale)
            row["triton_ms"] = _bench(run_triton)
        except Exception as e:
            row["triton_ms"] = None
            row["triton_err"] = str(e)

        # CUDA (tiled, shared memory)
        try:
            def run_cuda():
                out = attention_cuda(q, k, v, scale=scale)
                if out is None:
                    raise RuntimeError("CUDA extension not available")
            row["cuda_ms"] = _bench(run_cuda)
        except Exception as e:
            row["cuda_ms"] = None
            row["cuda_err"] = str(e)

        results.append(row)

    # Print table
    print("\n  S    | PyTorch (ms) | CUDA (ms)   | Triton (ms)")
    print("-------|---------------|-------------|-------------")
    for r in results:
        S = r["S"]
        pt = r.get("pytorch_ms")
        cu = r.get("cuda_ms")
        tr = r.get("triton_ms")
        pt_s = "%.4f" % pt if pt is not None else ("err: %s" % r.get("pytorch_err", "?")[:20])
        cu_s = "%.4f" % cu if cu is not None else ("err: %s" % r.get("cuda_err", "?")[:20])
        tr_s = "%.4f" % tr if tr is not None else ("err: %s" % r.get("triton_err", "?")[:20])
        print(" %4d  | %13s | %11s | %11s" % (S, pt_s, cu_s, tr_s))

    print("\n" + "=" * 70)
    print("Done. CUDA requires building the extension (first run may JIT compile).")
    print("=" * 70)


if __name__ == "__main__":
    main()
