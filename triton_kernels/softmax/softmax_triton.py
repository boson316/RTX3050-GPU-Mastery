"""
Triton Softmax: y = exp(x - max(x)) / sum(exp(x - max(x))) over the last dimension.

Provides:
- Baseline: two-pass; first pass compute max over N in BLOCK_N chunks, second pass
  exp(x-max) and sum, third pass write exp(x-max)/sum.
- Optimized: same with larger BLOCK_N and fused second/third where possible; one program per row.
- Autotuning over BLOCK_N.
- Benchmark vs torch.softmax.

Block tiling:
- Each program handles one row. Dimension N is tiled into chunks of BLOCK_N. Pass 1: load chunks,
  compute block max, scalar reduce to row max. Pass 2: load chunks, compute exp(x - max), accumulate
  sum. Pass 3: load chunks, compute exp(x - max) / sum, store. Consecutive elements in a row are
  loaded together -> coalesced.

Memory access patterns:
- Read X row by row in BLOCK_N-sized chunks; write Y the same. All coalesced along the last dim.
- Max and sum are scalars per row (or small buffers); kept in registers.

Register usage:
- Accumulators: row_max (scalar), row_sum (scalar). Block data BLOCK_N elements. Autotune
  balances BLOCK_N (fewer loops vs more SRAM).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Baseline: three-pass softmax (max, then sum(exp), then normalize)
# -----------------------------------------------------------------------------


@triton.jit
def _softmax_baseline_kernel(
    X_ptr,
    Y_ptr,
    stride,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    X_ptr += row * stride
    Y_ptr += row * stride

    row_max = tl.zeros([1], dtype=tl.float32) - float("inf")
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    row_sum = tl.zeros([1], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.exp(x - row_max)
        x = tl.where(mask, x, 0.0)
        row_sum += tl.sum(x, axis=0)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.exp(x - row_max) / row_sum
        tl.store(Y_ptr + cols, x.to(X_ptr.dtype.element_ty), mask=mask)


# -----------------------------------------------------------------------------
# Optimized: same three-pass with larger default BLOCK_N; optionally fuse last two
# passes (compute exp/sum and exp/sum in same loop then second loop write - or keep
# three for clarity). Here we keep three passes but use larger block.
# -----------------------------------------------------------------------------


@triton.jit
def _softmax_optimized_kernel(
    X_ptr,
    Y_ptr,
    stride,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    X_ptr += row * stride
    Y_ptr += row * stride

    row_max = tl.zeros([1], dtype=tl.float32) - float("inf")
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    row_sum = tl.zeros([1], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.exp(x - row_max)
        x = tl.where(mask, x, 0.0)
        row_sum += tl.sum(x, axis=0)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.exp(x - row_max) / row_sum
        tl.store(Y_ptr + cols, x.to(X_ptr.dtype.element_ty), mask=mask)


def softmax_baseline(
    x: torch.Tensor,
    dim: int = -1,
    BLOCK_N: int = 256,
) -> torch.Tensor:
    """Baseline softmax over last dimension (dim=-1)."""
    assert dim == -1, "Only dim=-1 supported"
    x = x.contiguous()
    x_flat = x.reshape(-1, x.shape[-1])
    M, N = x_flat.shape
    y = torch.empty_like(x_flat)
    grid = (M,)
    _softmax_baseline_kernel[grid](x_flat, y, stride=x_flat.stride(0), N=N, BLOCK_N=BLOCK_N)
    return y.reshape(x.shape)


def softmax_optimized(
    x: torch.Tensor,
    dim: int = -1,
    BLOCK_N: int = 1024,
) -> torch.Tensor:
    """Optimized softmax with larger block size."""
    assert dim == -1, "Only dim=-1 supported"
    x = x.contiguous()
    x_flat = x.reshape(-1, x.shape[-1])
    M, N = x_flat.shape
    y = torch.empty_like(x_flat)
    grid = (M,)
    _softmax_optimized_kernel[grid](x_flat, y, stride=x_flat.stride(0), N=N, BLOCK_N=BLOCK_N)
    return y.reshape(x.shape)


# -----------------------------------------------------------------------------
# Autotuned
# -----------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 256}, num_warps=2),
        triton.Config({"BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_N": 2048}, num_warps=8),
        triton.Config({"BLOCK_N": 4096}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def _softmax_autotuned_kernel(
    X_ptr,
    Y_ptr,
    stride,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    X_ptr += row * stride
    Y_ptr += row * stride

    row_max = tl.zeros([1], dtype=tl.float32) - float("inf")
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=float("-inf")).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    row_sum = tl.zeros([1], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.exp(x - row_max)
        x = tl.where(mask, x, 0.0)
        row_sum += tl.sum(x, axis=0)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.exp(x - row_max) / row_sum
        tl.store(Y_ptr + cols, x.to(X_ptr.dtype.element_ty), mask=mask)


def softmax_triton(
    x: torch.Tensor,
    dim: int = -1,
    BLOCK_N: int | None = None,
    use_autotune: bool = True,
) -> torch.Tensor:
    """Softmax over last dimension. FP16/BF16. use_autotune picks BLOCK_N."""
    assert dim == -1, "Only dim=-1 supported"
    x = x.contiguous()
    x_flat = x.reshape(-1, x.shape[-1])
    M, N = x_flat.shape
    y = torch.empty_like(x_flat)

    if use_autotune and BLOCK_N is None:
        grid = (M,)
        _softmax_autotuned_kernel[grid](x_flat, y, stride=x_flat.stride(0), N=N)
        return y.reshape(x.shape)

    blk = BLOCK_N or 1024
    grid = (M,)
    _softmax_optimized_kernel[grid](x_flat, y, stride=x_flat.stride(0), N=N, BLOCK_N=blk)
    return y.reshape(x.shape)


# -----------------------------------------------------------------------------
# Benchmark vs PyTorch
# -----------------------------------------------------------------------------


def benchmark_softmax(
    M: int = 4096,
    N: int = 1024,
    dtype: torch.dtype = torch.float16,
    warmup: int = 20,
    repeat: int = 100,
) -> dict[str, float]:
    x = torch.randn(M, N, device="cuda", dtype=dtype)

    def run(fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = __import__("time").perf_counter()
        for _ in range(repeat):
            fn()
        torch.cuda.synchronize()
        return (__import__("time").perf_counter() - start) * 1000 / repeat

    out = {}
    out["pytorch_ms"] = run(lambda: torch.softmax(x, dim=-1))
    out["triton_baseline_ms"] = run(lambda: softmax_baseline(x))
    out["triton_optimized_ms"] = run(lambda: softmax_optimized(x))
    out["triton_autotune_ms"] = run(lambda: softmax_triton(x, use_autotune=True))
    return out


if __name__ == "__main__":
    M, N = 1024, 512
    x = torch.randn(M, N, device="cuda", dtype=torch.float16)
    y = softmax_triton(x, use_autotune=True)
    ref = torch.softmax(x, dim=-1)
    print("softmax max diff:", (y - ref).abs().max().item())
    print("Benchmark:", benchmark_softmax(M=4096, N=1024))
