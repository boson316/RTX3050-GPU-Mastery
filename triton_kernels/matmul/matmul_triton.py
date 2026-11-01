"""
Triton matrix multiplication: C = A @ B (FP16 / BF16 inputs, FP32 accumulation).

One kernel, three entry points:
- matmul_baseline:  32x32x16 tiles, row-major program order (the "textbook" version).
- matmul_optimized: 64x64x32 tiles + grouped (swizzled) program order for L2 reuse.
- matmul_triton:    autotuned over BLOCK_M/N/K, GROUP_M, num_warps and num_stages
                    (software pipelining depth for the async-copy K loop on Ampere).

Grouped ordering (GROUP_M):
  With plain row-major ordering, consecutive programs walk along one row of C tiles,
  so each program re-reads a *different* B column panel and the B panels fall out of
  L2 before the next row of tiles reuses them. Launching programs in groups of
  GROUP_M rows makes GROUP_M consecutive programs share the same B panel, and the
  A panels of the group stay hot in L2 as well. For 2048^3 FP16 on a 2 MB L2 this
  is the difference between L2-resident and DRAM-bound panels.

Boundary handling:
  M, N and K need not be multiples of the tile size; all three dimensions are
  masked (an earlier version masked only M and N and read past the end of A/B
  whenever K % BLOCK_K != 0 - see tests/test_triton_kernels.py::test_matmul_odd_shapes).

Memory access patterns:
- A (M, K) row-major: BLOCK_M x BLOCK_K tiles, contiguous along K -> coalesced.
- B (K, N) row-major: BLOCK_K x BLOCK_N tiles, contiguous along N -> coalesced.
- C (M, N): BLOCK_M x BLOCK_N tile written once, contiguous along N.

Register usage:
  The FP32 accumulator is BLOCK_M*BLOCK_N / (32*num_warps) registers per thread
  (128x128 with 8 warps -> 64 regs), plus the A/B fragments; autotuning keeps this
  below the 255-register limit without spilling.
"""
from __future__ import annotations

import torch
import triton.language as tl

import triton


@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Grouped ordering: programs [g*GROUP_M*num_pid_n, (g+1)*GROUP_M*num_pid_n) cover
    # GROUP_M tile-rows, walking down a column of tiles before moving right.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None].to(tl.int64) * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None].to(tl.int64) * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + offs_k) < K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=c_mask)


_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8}, num_warps=w, num_stages=s)
    for bm, bn, bk, w, s in [
        (128, 128, 32, 8, 3),
        (128, 64, 32, 4, 4),
        (64, 128, 32, 4, 4),
        (128, 128, 64, 8, 2),
        (64, 64, 64, 4, 3),
        (64, 64, 32, 4, 4),
        (128, 256, 32, 8, 3),
        (256, 128, 32, 8, 3),
    ]
]

_matmul_kernel_autotuned = triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["M", "N", "K"])(_matmul_kernel)


def _prepare(A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
    assert A.dim() == 2 and B.dim() == 2, "expected 2D matrices"
    assert A.shape[1] == B.shape[0], f"shape mismatch: {tuple(A.shape)} @ {tuple(B.shape)}"
    assert A.device == B.device and A.dtype == B.dtype
    M, K = A.shape
    N = B.shape[1]
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    return C, M, N, K


def _launch_fixed(A, B, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M):
    C, M, N, K = _prepare(A, B)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _matmul_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
    )
    return C


def matmul_baseline(A: torch.Tensor, B: torch.Tensor, BLOCK_M: int = 32, BLOCK_N: int = 32,
                    BLOCK_K: int = 16) -> torch.Tensor:
    """Small tiles, row-major program order (GROUP_M=1). Reference point for the optimizations."""
    return _launch_fixed(A, B, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M=1)


def matmul_optimized(A: torch.Tensor, B: torch.Tensor, BLOCK_M: int = 64, BLOCK_N: int = 64,
                     BLOCK_K: int = 32) -> torch.Tensor:
    """Larger tiles + grouped program order for L2 reuse."""
    return _launch_fixed(A, B, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M=8)


def matmul_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    BLOCK_M: int | None = None,
    BLOCK_N: int | None = None,
    BLOCK_K: int | None = None,
    use_autotune: bool = True,
) -> torch.Tensor:
    """C = A @ B. FP16/BF16 (FP32 accumulate). Autotuned unless explicit block sizes are given."""
    if use_autotune and BLOCK_M is None and BLOCK_N is None and BLOCK_K is None:
        C, M, N, K = _prepare(A, B)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)  # noqa: E731
        _matmul_kernel_autotuned[grid](
            A, B, C, M, N, K,
            A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
        )
        return C
    return _launch_fixed(A, B, BLOCK_M or 64, BLOCK_N or 64, BLOCK_K or 32, GROUP_M=8)


# -----------------------------------------------------------------------------
# Benchmark vs PyTorch
# -----------------------------------------------------------------------------


def benchmark_matmul(
    M: int = 1024,
    N: int = 1024,
    K: int = 1024,
    dtype: torch.dtype = torch.float16,
    warmup: int = 20,
    repeat: int = 100,
) -> dict[str, float]:
    """Returns dict with keys: pytorch_ms, triton_baseline_ms, triton_optimized_ms, triton_autotune_ms."""
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)

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
    out["pytorch_ms"] = run(lambda: torch.matmul(A, B))
    out["triton_baseline_ms"] = run(lambda: matmul_baseline(A, B))
    out["triton_optimized_ms"] = run(lambda: matmul_optimized(A, B))
    out["triton_autotune_ms"] = run(lambda: matmul_triton(A, B, use_autotune=True))
    return out


if __name__ == "__main__":
    M, K, N = 1024, 1024, 1024
    for dtype in (torch.float16, torch.bfloat16):
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            continue
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(K, N, device="cuda", dtype=dtype)
        C = matmul_triton(A, B, use_autotune=True)
        C_ref = torch.matmul(A.float(), B.float()).to(dtype)
        err = (C - C_ref).abs().max().item()
        print(f"Triton matmul {dtype} max diff: {err}")

    print("\nBenchmark (FP16 1024x1024x1024):")
    for k, v in benchmark_matmul(1024, 1024, 1024).items():
        print(f"  {k}: {v:.4f} ms")
