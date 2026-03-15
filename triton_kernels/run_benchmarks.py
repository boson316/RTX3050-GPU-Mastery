"""
Run all Triton kernel benchmarks vs PyTorch.

Usage (from repo root):
  python -m triton_kernels.run_benchmarks

Or from triton_kernels/:
  python run_benchmarks.py

Requires: CUDA, PyTorch, Triton (or triton-windows on Windows).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root or from triton_kernels/
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    import torch
    if not torch.cuda.is_available():
        print("CUDA not available; skipping Triton benchmarks.")
        return

    print("=" * 60)
    print("Triton Kernels — Performance vs PyTorch")
    print("=" * 60)

    # 1. Matmul
    print("\n--- 1. Matrix multiplication (M=N=K=1024, FP16) ---")
    try:
        from triton_kernels.matmul import benchmark_matmul
        r = benchmark_matmul(1024, 1024, 1024, dtype=torch.float16)
        for k, v in r.items():
            print(f"  {k}: {v:.4f} ms")
    except Exception as e:
        print(f"  Error: {e}")

    # 2. Conv2D
    print("\n--- 2. Conv2D 3x3 (B=4, H=W=64, C_out=64, FP16) ---")
    try:
        from triton_kernels.conv import benchmark_conv
        r = benchmark_conv(B=4, H=64, W=64, C_out=64, dtype=torch.float16)
        for k, v in r.items():
            print(f"  {k}: {v:.4f} ms")
    except Exception as e:
        print(f"  Error: {e}")

    # 3. LayerNorm
    print("\n--- 3. LayerNorm (M=4096, N=1024, FP16) ---")
    try:
        from triton_kernels.layernorm import benchmark_layernorm
        r = benchmark_layernorm(M=4096, N=1024, dtype=torch.float16)
        for k, v in r.items():
            print(f"  {k}: {v:.4f} ms")
    except Exception as e:
        print(f"  Error: {e}")

    # 4. Softmax
    print("\n--- 4. Softmax (M=4096, N=1024, FP16) ---")
    try:
        from triton_kernels.softmax import benchmark_softmax
        r = benchmark_softmax(M=4096, N=1024, dtype=torch.float16)
        for k, v in r.items():
            print(f"  {k}: {v:.4f} ms")
    except Exception as e:
        print(f"  Error: {e}")

    # 5. Flash Attention
    print("\n--- 5. Flash Attention (B=2, H=8, S=512, D=64, FP16) ---")
    try:
        from triton_kernels.flash_attention import benchmark_flash_attention
        r = benchmark_flash_attention(B=2, H=8, S=512, D=64, dtype=torch.float16)
        for k, v in r.items():
            print(f"  {k}: {v:.4f} ms")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("Done. See module docstrings for block tiling and memory comments.")
    print("=" * 60)


if __name__ == "__main__":
    main()
