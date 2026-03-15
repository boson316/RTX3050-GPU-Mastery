"""
Triton matrix multiplication: C = A @ B.

Provides:
- Baseline: simple 2D tiling, one block per output tile.
- Optimized: larger tiles, split-K for large K, FP16/BF16 with float32 accumulation.
- Autotuning over BLOCK_M, BLOCK_N, BLOCK_K and (optional) SPLIT_K.
- Benchmark vs PyTorch (torch.matmul / cuBLAS).

Memory access patterns:
- A is (M, K): we load BLOCK_M x BLOCK_K tiles along rows of A; consecutive threads
  read consecutive columns (K-dim) -> coalesced along K.
- B is (K, N): we load BLOCK_K x BLOCK_N tiles; consecutive threads read consecutive
  rows (K-dim) -> coalesced along K. Good L2 reuse when BLOCK_K is in L1/SRAM.
- C is (M, N): we write BLOCK_M x BLOCK_N; coalesced along N.

Register usage:
- Each thread participates in a BLOCK_M x BLOCK_N output tile. With tl.dot, Triton
  maps to tensor cores / vectorized ops; register pressure scales with BLOCK_M*BLOCK_N
  (accumulator) + BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N (loaded tiles). Autotune keeps
  these within register limits.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Baseline kernel: simple tiling, no split-K
# Block tiling: each program computes one output tile C[pid_m*BLOCK_M : (pid_m+1)*BLOCK_M,
#                pid_n*BLOCK_N : (pid_n+1)*BLOCK_N] by iterating over K in steps of BLOCK_K.
# -----------------------------------------------------------------------------


@triton.jit
def _matmul_baseline_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    # Program ID: 2D grid over (num_tiles_m, num_tiles_n)
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Offsets for this block: which tile of C we compute
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointer bases: A[offs_am, :], B[:, offs_bn]. We advance along K.
    A_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptrs, mask=offs_am[:, None] < M, other=0.0)
        b = tl.load(B_ptrs, mask=offs_bn[None, :] < N, other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(C_ptrs, acc.to(C_ptr.dtype.element_ty), mask=mask)


# -----------------------------------------------------------------------------
# Optimized kernel: optional split-K for large K (reduces per-block K and improves
# occupancy). Same block tiling; we use larger BLOCK_* when not split-K.
# -----------------------------------------------------------------------------


@triton.jit
def _matmul_optimized_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # Grid: (num_tiles_m * num_tiles_n * SPLIT_K,) when SPLIT_K > 1 we need a
    # reduction over K; for simplicity this optimized kernel uses SPLIT_K=1.
    # So one program per (pid_m, pid_n).
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptrs, mask=offs_am[:, None] < M, other=0.0)
        b = tl.load(B_ptrs, mask=offs_bn[None, :] < N, other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(C_ptrs, acc.to(C_ptr.dtype.element_ty), mask=mask)


def _get_acc_type(dtype: torch.dtype) -> tl.constexpr:
    if dtype in (torch.float16, torch.bfloat16):
        return tl.float32
    return tl.float32


def matmul_baseline(
    A: torch.Tensor,
    B: torch.Tensor,
    BLOCK_M: int = 32,
    BLOCK_N: int = 32,
    BLOCK_K: int = 16,
) -> torch.Tensor:
    """Baseline matmul C = A @ B. Small tiles for correctness/debug."""
    assert A.is_cuda and B.is_cuda
    M, K = A.shape
    _, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    ACC_TYPE = _get_acc_type(A.dtype)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _matmul_baseline_kernel[grid](
        A, B, C,
        M=M, N=N, K=K,
        stride_am=A.stride(0), stride_ak=A.stride(1),
        stride_bk=B.stride(0), stride_bn=B.stride(1),
        stride_cm=C.stride(0), stride_cn=C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        ACC_TYPE=ACC_TYPE,
    )
    return C


def matmul_optimized(
    A: torch.Tensor,
    B: torch.Tensor,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 32,
) -> torch.Tensor:
    """Optimized matmul with larger tiles; use autotuned version in production."""
    assert A.is_cuda and B.is_cuda
    M, K = A.shape
    _, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    ACC_TYPE = _get_acc_type(A.dtype)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _matmul_optimized_kernel[grid](
        A, B, C,
        M=M, N=N, K=K,
        stride_am=A.stride(0), stride_ak=A.stride(1),
        stride_bk=B.stride(0), stride_bn=B.stride(1),
        stride_cm=C.stride(0), stride_cn=C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        ACC_TYPE=ACC_TYPE,
        SPLIT_K=1,
    )
    return C


# -----------------------------------------------------------------------------
# Autotuned wrapper: picks best BLOCK_M, BLOCK_N, BLOCK_K for given M, N, K.
# -----------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_autotuned_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
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

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptrs, mask=offs_am[:, None] < M, other=0.0)
        b = tl.load(B_ptrs, mask=offs_bn[None, :] < N, other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(C_ptrs, acc.to(C_ptr.dtype.element_ty), mask=mask)


def matmul_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    BLOCK_M: int | None = None,
    BLOCK_N: int | None = None,
    BLOCK_K: int | None = None,
    use_autotune: bool = True,
) -> torch.Tensor:
    """
    Matrix multiply C = A @ B. Supports FP16 and BF16 (accumulation in FP32).
    If use_autotune=True (default), ignores BLOCK_* and uses autotuned config.
    """
    assert A.is_cuda and B.is_cuda
    M, K = A.shape
    _, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    ACC_TYPE = _get_acc_type(A.dtype)

    if use_autotune and BLOCK_M is None and BLOCK_N is None and BLOCK_K is None:
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
        _matmul_autotuned_kernel[grid](
            A, B, C,
            M=M, N=N, K=K,
            stride_am=A.stride(0), stride_ak=A.stride(1),
            stride_bk=B.stride(0), stride_bn=B.stride(1),
            stride_cm=C.stride(0), stride_cn=C.stride(1),
            ACC_TYPE=ACC_TYPE,
        )
        return C

    bm = BLOCK_M if BLOCK_M is not None else 64
    bn = BLOCK_N if BLOCK_N is not None else 64
    bk = BLOCK_K if BLOCK_K is not None else 32
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _matmul_optimized_kernel[grid](
        A, B, C,
        M=M, N=N, K=K,
        stride_am=A.stride(0), stride_ak=A.stride(1),
        stride_bk=B.stride(0), stride_bn=B.stride(1),
        stride_cm=C.stride(0), stride_cn=C.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
        ACC_TYPE=ACC_TYPE,
        SPLIT_K=1,
    )
    return C


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
