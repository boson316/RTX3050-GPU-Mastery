"""
Triton fused QKV projection: y = x @ W_qkv + b.
Single matmul + bias; same as one linear layer. Uses tl.dot for (batch*seq, H) @ (H, 3*H).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qkv_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, K, BLOCK_K):
        a = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(w_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias (broadcast over rows); call with zeros when no bias
    b_ptrs = b_ptr + offs_n
    bias = tl.load(b_ptrs, mask=offs_n < N, other=0.0)
    acc += bias

    offs_ym = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_yn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    y_ptrs = y_ptr + offs_ym[:, None] * stride_ym + offs_yn[None, :] * stride_yn
    mask = (offs_ym[:, None] < M) & (offs_yn[None, :] < N)
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask)


def fused_qkv_triton(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None = None,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 32,
) -> torch.Tensor:
    """Fused QKV: y = x @ w + b. x (M, H), w (H, 3*H) -> (M, 3*H). FP16/BF16."""
    assert x.is_cuda and w.is_cuda
    M, K = x.shape
    _, N = w.shape
    assert w.shape[0] == K
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    ACC_TYPE = tl.float32 if x.dtype in (torch.float16, torch.bfloat16) else tl.float32

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    if b is None or not b.is_cuda:
        b = torch.zeros(N, device=x.device, dtype=x.dtype)
    _qkv_kernel[grid](
        x, w, b, y,
        M=M, N=N, K=K,
        stride_xm=x.stride(0), stride_xk=x.stride(1),
        stride_wn=w.stride(1), stride_wk=w.stride(0),
        stride_ym=y.stride(0), stride_yn=y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        ACC_TYPE=ACC_TYPE,
    )
    return y


__all__ = ["fused_qkv_triton"]
