"""
Triton GELU: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
Elementwise; coalesced reads/writes.
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
    # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    inv_sqrt2 = 0.70710678118654752440
    y = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    tl.store(y_ptr + offs, y.to(x_ptr.dtype.element_ty), mask=mask)


def gelu_triton(
    x: torch.Tensor,
    BLOCK: int = 1024,
) -> torch.Tensor:
    """GELU activation. FP16/BF16; elementwise."""
    assert x.is_cuda and x.is_contiguous()
    n = x.numel()
    y = torch.empty_like(x)
    grid = (triton.cdiv(n, BLOCK),)
    _gelu_kernel[grid](x, y, n_elements=n, BLOCK=BLOCK)
    return y


__all__ = ["gelu_triton"]
