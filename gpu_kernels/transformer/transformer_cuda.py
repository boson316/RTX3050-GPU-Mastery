"""
Transformer CUDA kernels (FP16): JIT-loaded PyTorch extension.

Provides: fused_qkv, softmax, layernorm, gelu, fused_mlp.
Build on first use (requires Ninja, VS, CUDA) or pre-build with setup.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_ext = None  # module when loaded; False when load attempted and failed

def _load_extension():
    global _ext
    if _ext is not None:
        return _ext if _ext is not False else None
    # Pre-built extension (pip install -e .) is named _transformer_cuda_native so it does not shadow this .py
    try:
        from . import _transformer_cuda_native as _ext
        return _ext
    except ImportError:
        pass
    folder = Path(__file__).resolve().parent
    build_dir = folder / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    cu = folder / "transformer_kernels.cu"
    cpp = folder / "transformer_cuda.cpp"
    if not cu.is_file() or not cpp.is_file():
        _ext = False
        return None
    try:
        from torch.utils.cpp_extension import load
        _ext = load(
            name="_transformer_cuda_native",
            sources=[str(cu), str(cpp)],
            extra_cuda_cflags=["-O3", "--use_fast_math", "--allow-unsupported-compiler"],
            build_directory=str(build_dir),
            with_cuda=True,
        )
        return _ext
    except Exception as e:
        sys.stderr.write(f"Transformer CUDA extension load failed: {e}\n")
        _ext = False
        return None


def fused_qkv_cuda(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Fused QKV: y = x @ w + b. x (M, H), w (H, 3*H) -> (M, 3*H). FP16."""
    ext = _load_extension()
    if ext is None:
        return None
    return ext.fused_qkv(x, w, b)


def softmax_cuda(x: torch.Tensor, dim: int = -1) -> torch.Tensor | None:
    """Softmax over last dimension. FP16."""
    ext = _load_extension()
    if ext is None:
        return None
    return ext.softmax(x, dim)


def layernorm_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor | None:
    """LayerNorm over last dim. FP16."""
    ext = _load_extension()
    if ext is None:
        return None
    return ext.layernorm(x, weight, bias, eps)


def gelu_cuda(x: torch.Tensor) -> torch.Tensor | None:
    """GELU activation. FP16."""
    ext = _load_extension()
    if ext is None:
        return None
    return ext.gelu(x)


def fused_mlp_cuda(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor | None:
    """Fused MLP: out = linear2(GELU(linear1(x))). FP16."""
    ext = _load_extension()
    if ext is None:
        return None
    return ext.fused_mlp(x, w1, b1, w2, b2)


__all__ = [
    "fused_qkv_cuda",
    "softmax_cuda",
    "layernorm_cuda",
    "gelu_cuda",
    "fused_mlp_cuda",
    "_load_extension",
]
