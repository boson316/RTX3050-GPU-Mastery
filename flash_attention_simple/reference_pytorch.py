"""
PyTorch reference: attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V.

Stores the full attention matrix in memory (O(S^2) per head). Use for correctness
check only; not memory-optimized.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def attention_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """
    attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V.

    Args:
        q, k, v: (batch, n_heads, seq_len, head_dim). float32 or float16.
        scale: softmax scale (default 1/sqrt(head_dim)).

    Returns:
        out: (batch, n_heads, seq_len, head_dim), same dtype as q.
    """
    *_, seq_len, head_dim = q.shape
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)
    # (B, H, S, D) @ (B, H, D, S) -> (B, H, S, S)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = F.softmax(scores, dim=-1)
    # (B, H, S, S) @ (B, H, S, D) -> (B, H, S, D)
    out = torch.matmul(attn, v)
    return out
