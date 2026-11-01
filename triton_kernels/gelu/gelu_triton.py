"""
Triton GELU (exact, erf form): y = 0.5 * x * (1 + erf(x / sqrt(2)))  == torch.nn.functional.gelu(x).
Elementwise, memory-bound: one coalesced read + one write per element, FP32 math in registers.
(The fused MLP uses the tanh approximation in its GEMM epilogue instead, see triton_kernels/qkv.)
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    inv_sqrt2 = 0.70710678118654752440
    y = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    tl.store(y_ptr + offs, y.to(x_ptr.dtype.element_ty), mask=mask)


def gelu_triton(
    x: torch.Tensor,
    BLOCK: int = 1024,
) -> torch.Tensor:
    """GELU activation. FP16/BF16; elementwise."""
    x = x.contiguous()
    n = x.numel()
    y = torch.empty_like(x)
    grid = (triton.cdiv(n, BLOCK),)
    _gelu_kernel[grid](x, y, n_elements=n, BLOCK=BLOCK)
    return y


__all__ = ["gelu_triton"]
