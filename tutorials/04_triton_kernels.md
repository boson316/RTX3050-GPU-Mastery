# Tutorial 04: Triton Kernels

**Triton** is a language and compiler for writing GPU kernels at a higher level than raw CUDA. You describe **block-level** programs (tiles, loops, loads/stores), and Triton generates efficient GPU code and handles many optimization details. This tutorial introduces the programming model and links to matmul, softmax, and related kernels in this repo.

---

## Concepts

### 1. Block-level programming

- In CUDA you think in **threads** (threadIdx, blockIdx) and often manage shared memory by hand.
- In Triton you think in **programs**: each **program** handles a block of the output (e.g. a tile of a matrix). You use **ranges** like `tl.arange(0, BLOCK_M)` and **pointers** that move over the data; Triton maps this to threads and memory.

### 2. Key abstractions

- **`tl.program_id(axis)`** — which block (program instance) this is along a given grid axis.
- **`tl.arange(...)`** — ranges used for indexing; e.g. `offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)`.
- **`tl.load(ptr, mask=..., other=...)`** / **`tl.store(ptr, value, mask=...)`** — block-level load/store; the compiler produces coalesced access when possible.
- **`tl.dot(a, b)`** — matrix multiply of blocks; can map to Tensor Cores.
- **`tl.sum`, `tl.max`, etc.** — reductions over dimensions.

### 3. Decorator and launch

- Define a kernel with `@triton.jit` and **constexpr** tile sizes (e.g. `BLOCK_M: tl.constexpr`).
- Launch with `kernel_name[grid](args, BLOCK_M=64, BLOCK_N=64, ...)` where `grid` is a tuple of program counts (e.g. `(num_tiles_m * num_tiles_n,)` for 1D grid over tiles).

### 4. Why Triton?

- **Productivity:** less boilerplate than CUDA; no explicit shared memory or thread indexing for simple cases.
- **Portability:** same kernel can target different GPU architectures; autotuning over block sizes is common.
- **Performance:** often close to hand-tuned CUDA for many dense ops (matmul, softmax, attention).

---

## Simplified code: Triton matmul

**Goal:** C = A @ B. Each program computes one tile of C of size `BLOCK_M×BLOCK_N`, iterating over K in steps of `BLOCK_K`.

```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptrs, mask=offs_am[:, None] < M, other=0.0)
        b = tl.load(B_ptrs, mask=offs_bn[None, :] < N, other=0.0)
        acc += tl.dot(a, b)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))
```

- **Grid:** 1D over tiles: `grid = (tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N),)`.
- **Loop over K:** load tiles of A and B, `tl.dot`, accumulate; pointers advance by `BLOCK_K`.

---

## Simplified code: Triton softmax (per row)

**Goal:** softmax over the last dimension: `y = exp(x - max(x)) / sum(exp(x - max(x)))`. One program per row; pass 1: compute row max in blocks; pass 2: compute sum(exp); pass 3: write normalized values.

```python
@triton.jit
def softmax_kernel(X_ptr, Y_ptr, stride, N, BLOCK_N: tl.constexpr):
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
        row_sum += tl.sum(tl.where(mask, x, 0.0), axis=0)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.exp(x - row_max) / row_sum
        tl.store(Y_ptr + cols, x, mask=mask)
```

---

## Implementations in this repository

| Kernel | Location | Description |
|--------|----------|-------------|
| **Matmul** | [triton_kernels/matmul/matmul_triton.py](../triton_kernels/matmul/matmul_triton.py) | Baseline + optimized + autotuned; benchmark vs PyTorch. |
| **Softmax** | [triton_kernels/softmax/softmax_triton.py](../triton_kernels/softmax/softmax_triton.py) | Row-wise softmax; 3-pass (max, sum, normalize). |
| **LayerNorm** | [triton_kernels/layernorm/layernorm_triton.py](../triton_kernels/layernorm/layernorm_triton.py) | Fused normalize + affine over last dimension. |
| **Conv2D** | [triton_kernels/conv/conv_triton.py](../triton_kernels/conv/conv_triton.py) | 3×3 Conv2D FP16; tile reuse, autotune. |
| **GELU** | [triton_kernels/gelu/gelu_triton.py](../triton_kernels/gelu/gelu_triton.py) | GELU activation. |
| **QKV** | [triton_kernels/qkv/qkv_triton.py](../triton_kernels/qkv/qkv_triton.py) | Fused QKV projection. |
| **Flash Attention** | [triton_kernels/flash_attention/flash_attention.py](../triton_kernels/flash_attention/flash_attention.py) | Fused tiled attention with online softmax (see [05_flashattention.md](05_flashattention.md)). |

**Run all Triton benchmarks:**

```bash
python -m triton_kernels.run_benchmarks
```

**Docs:** [docs/triton_kernel_design.md](../docs/triton_kernel_design.md).
