"""
Triton fused MLP block: out = linear2(GELU(linear1(x))).
Composes: qkv-style matmul (linear1) -> GELU -> matmul (linear2). Optional bias on both linears.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# Compose QKV (matmul+bias) and GELU from same repo
try:
    from triton_kernels.qkv.qkv_triton import fused_qkv_triton
    from triton_kernels.gelu.gelu_triton import gelu_triton
except ImportError:
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(_root))
    from triton_kernels.qkv.qkv_triton import fused_qkv_triton
    from triton_kernels.gelu.gelu_triton import gelu_triton


def _linear_bias_triton(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
    """x @ w + b using fused_qkv_triton (same op)."""
    if b is None:
        b = torch.zeros(w.size(1), device=x.device, dtype=x.dtype)
    return fused_qkv_triton(x, w, b)


def fused_mlp_triton(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
) -> torch.Tensor:
    """
    Fused MLP: out = linear2(GELU(linear1(x))).
    x (M, H), w1 (H, 4*H), w2 (4*H, H) -> (M, H). FP16/BF16.
    """
    mid = _linear_bias_triton(x, w1, b1)
    mid = gelu_triton(mid)
    out = _linear_bias_triton(mid, w2, b2)
    return out


__all__ = ["fused_mlp_triton"]
