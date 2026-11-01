"""
Triton fused MLP block: out = GELU(x @ W1 + b1) @ W2 + b2.

Kernel 1: GEMM with bias + GELU fused into the epilogue (the activation never round-trips
          through DRAM on its own).
Kernel 2: GEMM with bias epilogue.
So the whole block is 2 launches instead of 4 (gemm, bias+gelu, gemm, bias) or 5 unfused.
"""
from __future__ import annotations

import torch

from triton_kernels.qkv.qkv_triton import linear_triton


def fused_mlp_triton(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
) -> torch.Tensor:
    """x (M, H), w1 (H, 4H), w2 (4H, H) -> (M, H). FP16/BF16 (FP32 accumulate). GELU = tanh approximation."""
    mid = linear_triton(x, w1, b1, activation="gelu")
    return linear_triton(mid, w2, b2, activation="none")


__all__ = ["fused_mlp_triton"]
