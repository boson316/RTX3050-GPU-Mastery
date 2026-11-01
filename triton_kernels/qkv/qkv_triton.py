"""
Triton linear layer with fused epilogue: y = act(x @ W + b).

Used for the fused QKV projection (act = identity, W = [W_q | W_k | W_v], N = 3H) and for
the MLP (act = GELU fused into the first GEMM, see triton_kernels/mlp).

Epilogue fusion: bias and activation are applied to the FP32 accumulator while the
output tile is still in registers, so the (M x N) result is written to DRAM exactly once.
Without fusion, `gelu(x @ W1 + b1)` costs two extra full read+write passes over the
(M x 4H) intermediate - for M=2048, H=768 that is 2 x 2 x 12.6 MB = ~50 MB of traffic.

Fixed: the K-dimension mask was `offs_k < K` (missing the loop offset), so whenever
K % BLOCK_K != 0 the last iteration read past the end of each row of x and W.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_tanh(x):
    # tanh-approximation GELU (same as the CUDA kernel / torch gelu(approximate="tanh")).
    # tanh(u) = 2*sigmoid(2u) - 1, written with exp so it works on every Triton version.
    u = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    return x * tl.sigmoid(2.0 * u)  # 0.5*x*(1+tanh(u)) == x*sigmoid(2u)


@triton.jit
def _linear_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,  # "none" | "gelu"
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # grouped ordering for L2 reuse (see triton_kernels/matmul/matmul_triton.py)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None].to(tl.int64) * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None].to(tl.int64) * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + offs_k) < K
        a = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(w_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # ---- fused epilogue on the FP32 accumulator ----
    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)[None, :]
    if ACTIVATION == "gelu":
        acc = _gelu_tanh(acc)

    y_ptrs = y_ptr + offs_m[:, None].to(tl.int64) * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def linear_triton(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None = None,
    activation: str = "none",
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 32,
) -> torch.Tensor:
    """y = act(x @ w + b). x (M, K), w (K, N), b (N,) or None; activation in {"none", "gelu"}."""
    assert activation in ("none", "gelu")
    assert x.dim() == 2 and w.dim() == 2 and x.shape[1] == w.shape[0], f"{tuple(x.shape)} @ {tuple(w.shape)}"
    assert x.device == w.device and x.dtype == w.dtype
    if b is not None:
        assert b.numel() == w.shape[1] and b.device == x.device
        b = b.contiguous()
    M, K = x.shape
    N = w.shape[1]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _linear_kernel[grid](
        x, w, b if b is not None else y, y,  # dummy pointer when HAS_BIAS=False (never read)
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        HAS_BIAS=b is not None,
        ACTIVATION=activation,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=8,
    )
    return y


def fused_qkv_triton(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None, **block_sizes) -> torch.Tensor:
    """Fused QKV: y = x @ W_qkv + b. x (M, H), w (H, 3H) -> (M, 3H). One GEMM instead of three."""
    return linear_triton(x, w, b, activation="none", **block_sizes)


__all__ = ["fused_qkv_triton", "linear_triton"]
