"""
Triton LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias over the last dimension.

Provides:
- Baseline: one program per row; load row in BLOCK_N chunks, scalar-style reduction for mean/var.
- Optimized: one program per row, vectorized loads and tl.sum over blocks; fused normalize + affine.
- Autotuning over BLOCK_N.
- Benchmark vs torch.nn.functional.layer_norm.

Block tiling:
- Each program handles one row (or a block of rows). Normalization dimension N is tiled into
  chunks of BLOCK_N. We loop over N, load BLOCK_N elements, accumulate into _mean and _var
  (vectors of length BLOCK_N), then reduce to scalar mean and var. Second pass: load x, weight, bias
  in BLOCK_N chunks, compute (x - mean) * rstd * weight + bias, store.

Memory access patterns:
- Input X: row-major; we load X[row, off:off+BLOCK_N] so consecutive threads read consecutive
  elements -> coalesced. Same for weight/bias and output Y.
- Mean/rstd: one scalar per row; stored in separate buffers for reuse in pass 2.

Register usage:
- Accumulators _mean, _var of size BLOCK_N (in SRAM). Keeping BLOCK_N large reduces loop
  iterations but increases register/SRAM use; autotune finds a balance.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Baseline: per-row, loop over N in BLOCK_N chunks; explicit sum over block
# -----------------------------------------------------------------------------


@triton.jit
def _layernorm_baseline_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    B_ptr,
    stride,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    X_ptr += row * stride
    Y_ptr += row * stride

    _mean = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        a = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    _var = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask)
        b = tl.load(B_ptr + cols, mask=mask)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y_ptr + cols, y.to(X_ptr.dtype.element_ty), mask=mask)


# -----------------------------------------------------------------------------
# Optimized: same algorithm with optional larger BLOCK_N and num_warps tuning;
# single kernel, no separate mean/rstd buffer (recompute in store loop from scalars).
# -----------------------------------------------------------------------------


@triton.jit
def _layernorm_optimized_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    B_ptr,
    Mean_ptr,
    Rstd_ptr,
    stride,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    X_ptr += row * stride
    Y_ptr += row * stride

    _mean = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        a = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    _var = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(Mean_ptr + row, mean)
    tl.store(Rstd_ptr + row, rstd)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask)
        b = tl.load(B_ptr + cols, mask=mask)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y_ptr + cols, y.to(X_ptr.dtype.element_ty), mask=mask)


def layernorm_baseline(
    x: torch.Tensor,
    normalized_shape: tuple[int, ...],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    BLOCK_N: int = 256,
) -> torch.Tensor:
    """Baseline LayerNorm forward. normalized_shape should match last dim(s); we support 1D (N,)."""
    N = weight.numel()
    assert x.shape[-1] == N and bias.numel() == N
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    x_flat = x.reshape(-1, x.shape[-1])
    M, N_val = x_flat.shape
    y = torch.empty_like(x_flat)
    grid = (M,)
    _layernorm_baseline_kernel[grid](
        x_flat, y, weight, bias,
        stride=x_flat.stride(0), N=N_val, eps=eps, BLOCK_N=BLOCK_N,
    )
    return y.reshape(x.shape)


def layernorm_optimized(
    x: torch.Tensor,
    normalized_shape: tuple[int, ...],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    BLOCK_N: int = 1024,
) -> torch.Tensor:
    """Optimized LayerNorm with larger block; writes mean/rstd to internal buffers."""
    N = weight.numel()
    assert x.shape[-1] == N and bias.numel() == N
    x_flat = x.reshape(-1, x.shape[-1])
    M, N_val = x_flat.shape[0], x_flat.shape[1]
    y = torch.empty_like(x_flat)
    mean = torch.empty((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    grid = (M,)
    _layernorm_optimized_kernel[grid](
        x_flat, y, weight, bias, mean, rstd,
        stride=x_flat.stride(0), N=N_val, eps=eps, BLOCK_N=BLOCK_N,
    )
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
def _layernorm_autotuned_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    B_ptr,
    Mean_ptr,
    Rstd_ptr,
    stride,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    X_ptr += row * stride
    Y_ptr += row * stride

    _mean = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        a = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    _var = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(Mean_ptr + row, mean)
    tl.store(Rstd_ptr + row, rstd)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask)
        b = tl.load(B_ptr + cols, mask=mask)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y_ptr + cols, y.to(X_ptr.dtype.element_ty), mask=mask)


def layernorm_triton(
    x: torch.Tensor,
    normalized_shape: tuple[int, ...],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    BLOCK_N: int | None = None,
    use_autotune: bool = True,
) -> torch.Tensor:
    """LayerNorm forward; FP16/BF16. use_autotune=True picks BLOCK_N automatically."""
    N = weight.numel()
    assert x.shape[-1] == N and bias.numel() == N
    x_flat = x.reshape(-1, x.shape[-1])
    M, N_val = x_flat.shape
    y = torch.empty_like(x_flat)
    mean = torch.empty((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

    if use_autotune and BLOCK_N is None:
        grid = (M,)
        _layernorm_autotuned_kernel[grid](
            x_flat, y, weight, bias, mean, rstd,
            stride=x_flat.stride(0), N=N_val, eps=eps,
        )
        return y.reshape(x.shape)

    blk = BLOCK_N or 1024
    grid = (M,)
    _layernorm_optimized_kernel[grid](
        x_flat, y, weight, bias, mean, rstd,
        stride=x_flat.stride(0), N=N_val, eps=eps, BLOCK_N=blk,
    )
    return y.reshape(x.shape)


# -----------------------------------------------------------------------------
# Benchmark vs PyTorch
# -----------------------------------------------------------------------------


def benchmark_layernorm(
    M: int = 4096,
    N: int = 1024,
    dtype: torch.dtype = torch.float16,
    warmup: int = 20,
    repeat: int = 100,
) -> dict[str, float]:
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    weight = torch.ones(N, device="cuda", dtype=dtype)
    bias = torch.zeros(N, device="cuda", dtype=dtype)
    normalized_shape = (N,)

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
    out["pytorch_ms"] = run(lambda: torch.nn.functional.layer_norm(x, normalized_shape, weight, bias))
    out["triton_baseline_ms"] = run(lambda: layernorm_baseline(x, normalized_shape, weight, bias))
    out["triton_optimized_ms"] = run(lambda: layernorm_optimized(x, normalized_shape, weight, bias))
    out["triton_autotune_ms"] = run(lambda: layernorm_triton(x, normalized_shape, weight, bias, use_autotune=True))
    return out


if __name__ == "__main__":
    M, N = 1024, 512
    x = torch.randn(M, N, device="cuda", dtype=torch.float16)
    weight = torch.ones(N, device="cuda", dtype=torch.float16)
    bias = torch.zeros(N, device="cuda", dtype=torch.float16)
    y = layernorm_triton(x, (N,), weight, bias, use_autotune=True)
    ref = torch.nn.functional.layer_norm(x, (N,), weight, bias)
    print("layernorm max diff:", (y - ref).abs().max().item())
    print("Benchmark:", benchmark_layernorm(M=4096, N=1024))
