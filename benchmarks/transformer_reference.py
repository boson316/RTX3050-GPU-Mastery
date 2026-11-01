"""
PyTorch reference implementations for transformer building blocks (FP16).
Used for correctness checks and baseline benchmarking.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def fused_qkv_pytorch(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused QKV: y = x @ w + b. x (M, H), w (H, 3*H)."""
    y = F.linear(x, w.t(), b)
    return y


def softmax_pytorch(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax over dim (default last)."""
    return F.softmax(x, dim=dim)


def layernorm_pytorch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """LayerNorm over last dimension."""
    return F.layer_norm(x, (x.size(-1),), weight, bias, eps)


def gelu_pytorch(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    """GELU activation (exact erf form by default; approximate="tanh" for the GPT-2 form)."""
    return F.gelu(x, approximate=approximate)


def fused_mlp_pytorch(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
) -> torch.Tensor:
    """MLP: out = linear2(GELU_tanh(linear1(x))) - tanh GELU, as in GPT-2 and the Triton/CUDA kernels."""
    mid = F.linear(x, w1.t(), b1)
    mid = F.gelu(mid, approximate="tanh")
    out = F.linear(mid, w2.t(), b2)
    return out


__all__ = [
    "fused_qkv_pytorch",
    "softmax_pytorch",
    "layernorm_pytorch",
    "gelu_pytorch",
    "fused_mlp_pytorch",
]
