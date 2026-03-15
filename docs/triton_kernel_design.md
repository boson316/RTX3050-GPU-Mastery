# Triton Kernel Design

This document describes the design of Triton kernels in this repository: **matmul**, **conv**, **layernorm**, **softmax**, and **Flash Attention**. Audience: GPU engineers and advanced students.

---

## 1. Algorithm overview

| Module | Algorithm | Grid / block semantics |
|--------|-----------|------------------------|
| **matmul** | C = A @ B | One program per output tile (BLOCK_M×BLOCK_N); loop over K in steps of BLOCK_K |
| **conv** | 3×3 Conv2D | One program per output tile (BLOCK_OH×BLOCK_OW×BLOCK_C); load 3×3 input region per tile |
| **layernorm** | (x−μ)*rstd*γ+β over last dim | One program per row block |
| **softmax** | Row-wise exp(x−max)/sum(exp) | One program per row block |
| **flash_attention** | softmax(QK^T/√d)V with tiling | One program per Q block; loop over K/V blocks (see [flashattention_algorithm.md](flashattention_algorithm.md)) |

Triton compiles each *program* to GPU kernels; the **grid** is over program IDs (e.g. over output tiles or rows).

---

## 2. GPU memory access patterns

### Matmul (`triton_kernels/matmul/matmul_triton.py`)

- **A (M×K):** Load BLOCK_M×BLOCK_K tiles; `offs_am` varies along M, `offs_k` along K. Consecutive threads along K → coalesced.
- **B (K×N):** Load BLOCK_K×BLOCK_N tiles; consecutive threads along N → coalesced.
- **C (M×N):** Write BLOCK_M×BLOCK_N; coalesced along N.

```python
# triton_kernels/matmul/matmul_triton.py
A_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
# ...
acc += tl.dot(a, b, allow_tf32=True)
tl.store(C_ptrs, acc.to(C_ptr.dtype.element_ty), mask=mask)
```

- **Register usage:** Accumulator is BLOCK_M×BLOCK_N; tiles a, b are BLOCK_M×BLOCK_K and BLOCK_K×BLOCK_N. Autotune picks block sizes that fit registers and map well to tensor cores.

### Conv (`triton_kernels/conv/conv_triton.py`)

- Load 9 (BLOCK_OH, BLOCK_OW) input blocks for the 3×3 kernel; reuse across BLOCK_C (unrolled).
- Weight loaded per output channel (or channel block); output tile written coalesced.

### Layernorm / Softmax

- Row-wise: each program handles a block of rows; loads/stores are contiguous along the last dimension (coalesced).
- Reductions (mean, max, sum) use Triton’s primitives or explicit tree reduction in SRAM/registers.

### Flash Attention

- Q: one BLOCK_M×BLOCK_D per program; loaded once.
- K, V: streamed in BLOCK_N×BLOCK_D chunks; coalesced along D.
- Output: BLOCK_M×BLOCK_D written once. No full S×S matrix in global memory (see [flashattention_algorithm.md](flashattention_algorithm.md)).

---

## 3. Kernel launch configuration

Triton exposes **grid** as the number of *programs*; each program is mapped to one or more GPU thread blocks internally.

### Matmul

```python
# triton_kernels/matmul/matmul_triton.py
grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
_matmul_baseline_kernel[grid](
    A, B, C, M=M, N=N, K=K,
    stride_am=A.stride(0), stride_ak=A.stride(1),
    stride_bk=B.stride(0), stride_bn=B.stride(1),
    stride_cm=C.stride(0), stride_cn=C.stride(1),
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    ACC_TYPE=acc_type,
)
```

- **Grid:** 1D over (num_tiles_m × num_tiles_n). BLOCK_M, BLOCK_N, BLOCK_K are compile-time constants (e.g. 64, 64, 32).

### Flash Attention

```python
# triton_kernels/flash_attention/flash_attention.py
grid = (triton.cdiv(seq_len_q, BLOCK_M), batch * num_heads)
_fwd_kernel[grid](q, k, v, o, ...)
```

- **Grid:** (number of Q blocks, batch×heads). Each program handles one BLOCK_M chunk of the Q sequence and loops over K/V blocks.

### Conv

- Grid is 2D or 3D over output spatial and channel blocks; see `conv_triton.py` for exact `grid` and block constants.

---

## 4. Optimization techniques used

1. **Block tiling:** All kernels tile over the output and (where relevant) over the reduction dimension (K for matmul, sequence for attention) to keep working set in SRAM/registers.
2. **tl.dot:** Uses tensor cores / vectorized ops; FP16/BF16 with float32 accumulation.
3. **Autotuning:** Matmul, conv, layernorm, softmax, and flash_attention expose BLOCK_* and other constants for autotune; Triton picks configurations that improve occupancy and reuse.
4. **Masking:** All loads/stores use boundary masks (`mask=...`) to handle non-multiples of block size without branches.
5. **allow_tf32:** Set to True for matmul where precision allows, for higher throughput on Ampere+.
6. **Online softmax (Flash Attention):** Running max and sum; no materialization of full attention matrix (see [flashattention_algorithm.md](flashattention_algorithm.md)).

---

## 5. Benchmark results (RTX 3050)

| Kernel | Config | Result |
|--------|--------|--------|
| Matmul 1024×1024 FP16 | Triton vs PyTorch | Comparable or better GFLOPS (run `python benchmarks/matmul_benchmark.py`) |
| Conv 3×3 B=128 FP16 | Triton vs torch | **~1.27×** (see docs/benchmarks.md) |
| Flash Attention | Triton vs SDPA | See `triton_kernels/flash_attention` and `benchmarks/attention_benchmark.py` |

Run from repo root: `python -m triton_kernels.run_benchmarks`.

---

## 6. Diagrams / tables

### Matmul block tiling

| Stage | A tile | B tile | C tile |
|-------|--------|--------|--------|
| Load | BLOCK_M×BLOCK_K | BLOCK_K×BLOCK_N | — |
| Compute | in SRAM | in SRAM | accumulator |
| Store | — | — | BLOCK_M×BLOCK_N |

### Triton vs CUDA (conceptual)

| Aspect | Triton | CUDA |
|--------|--------|------|
| Grid | Program IDs (logical) | blockIdx / gridDim |
| Block | Implicit from BLOCK_* | threadIdx / blockDim |
| Shared memory | Implicit (SRAM) | Explicit `__shared__` |
| Reduction | Primitives / loops | Warp shuffle, shared |

---

## 7. Code snippets (from this repo)

### Matmul baseline (Triton)

```python
# triton_kernels/matmul/matmul_triton.py
@triton.jit
def _matmul_baseline_kernel(A_ptr, B_ptr, C_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ACC_TYPE: tl.constexpr,
):
    pid = tl.program_id(0)
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
    tl.store(C_ptrs, acc.to(C_ptr.dtype.element_ty), mask=mask)
```

### Grid launch

```python
grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
_matmul_baseline_kernel[grid](A, B, C, ...)
```

---

## References

- [flashattention_algorithm.md](flashattention_algorithm.md) — Flash Attention tiling and online softmax.
- [gpu_memory_hierarchy.md](gpu_memory_hierarchy.md) — memory hierarchy and coalescing.
- [optimization_guide.md](optimization_guide.md) — general optimization steps.
