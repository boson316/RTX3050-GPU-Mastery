"""
CUDA FlashAttention: JIT-built PyTorch extension.

attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V with tiling and shared memory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_cuda_attention = None  # module when loaded; False when load attempted and failed

def _load_cuda_extension():
    global _cuda_attention
    if _cuda_attention is not None:
        return _cuda_attention if _cuda_attention is not False else None
    folder = Path(__file__).resolve().parent
    build_dir = folder / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    cu_path = folder / "flash_attention_cuda.cu"
    cpp_path = folder / "flash_attention_cuda.cpp"
    if not cu_path.is_file() or not cpp_path.is_file():
        _cuda_attention = False
        return None
    try:
        from torch.utils.cpp_extension import load
        _cuda_attention = load(
            name="flash_attention_cuda",
            sources=[str(cu_path), str(cpp_path)],
            extra_cuda_cflags=["-O3"],
            build_directory=str(build_dir),
            with_cuda=True,
        )
        return _cuda_attention
    except Exception as e:
        sys.stderr.write(f"CUDA extension load failed: {e}\n")
        _cuda_attention = False
        return None


def attention_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor | None:
    """
    attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V. CUDA tiled kernel.

    Requires JIT build (first call may compile). Returns None if build fails.
    q, k, v: (B, H, S, D) float32 on CUDA. D must be <= 64.
    """
    if not q.is_cuda or q.dtype != torch.float32:
        return None
    B, H, S, D = q.shape
    if D > 64:
        return None
    ext = _load_cuda_extension()
    if ext is None:
        return None
    scale_val = scale if scale is not None else (1.0 / (D ** 0.5))
    return ext.forward(q, k, v, scale_val)
