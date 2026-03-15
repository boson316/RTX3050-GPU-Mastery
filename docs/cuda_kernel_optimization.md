# CUDA Kernel Optimization

This document covers the CUDA kernel optimizations used in this repository: tiled matmul, custom conv2d, and the transformer kernels (QKV, softmax, layernorm, GELU, MLP). Target device: **RTX 3050 (Ampere sm_86)**.

**Thread hierarchy diagram:** [images/cuda_thread_hierarchy.png](images/cuda_thread_hierarchy.png) (grid → block → warp → thread; from `python docs/scripts/generate_diagrams.py`)

---

## 1. Algorithm overview

| Kernel | Algorithm | Optimization focus |
|--------|-----------|---------------------|
| **Tiled matmul** | C = A×B; each block computes a tile of C | Shared-memory tiling, coalesced loads |
| **Conv 3×3** | Output[b,oc,oh,ow] = sum over 3×3×in_ch | Tiled input + weight in shared, __half2 for 2 channels |
| **Fused QKV** | y = x@W + b | cuBLAS matmul + small bias kernel (avoid slow custom matmul) |
| **Softmax** | Row-wise exp(x-max)/sum(exp) | Warp shuffle reduction, multi-row per block |
| **LayerNorm** | (x-μ)*rstd*γ+β | Three passes: mean, var, normalize; warp shuffle |
| **GELU** | 0.5*x*(1+tanh(…)) | Elementwise, coalesced |
| **Fused MLP** | linear2(GELU(linear1(x))) | Two 16×16 tiled matmuls + GELU |

---

## 2. GPU memory access patterns

### Tiled matmul (Level 2 roadmap)

- **A** is row-major (M×K): each block loads a **TILE×TILE** tile of A along the row; `threadIdx.y` is row, `threadIdx.x` is column within tile → column-wise load is coalesced (adjacent threads = adjacent K indices).
- **B** is row-major (K×N): tile of B loaded so that `threadIdx.y` corresponds to K dimension → coalesced along K.
- **C**: one output element per thread; write is coalesced along N.

```cuda
// cuda_roadmap/level2_memory/tiled_matmul/tiled_matmul_bench.cu
As[threadIdx.y][threadIdx.x] = (row < n && colA < n) ? A[row*n+colA] : 0.f;
Bs[threadIdx.y][threadIdx.x] = (rowB < n && col < n) ? B[rowB*n+col] : 0.f;
```

### Conv2d (extension)

- **Input tile:** 18×18 (16×16 output tile + 2 for 3×3 kernel) loaded cooperatively into `__shared__ __half tile_in[SHARD_H][SHARD_W]`.
- **Weight:** 32 output channels × 9 weights; stored as `__half2` (2 channels per entry) in shared memory for vectorized use.
- **Output:** Each thread writes one (oh, ow) for one output channel; writes are coalesced.

```cuda
// extension/conv_kernel.cu
__shared__ __half tile_in[SHARD_H][SHARD_W];
__shared__ __half2 tile_w[C_OUT_H2][9];  // 16 groups x 9, 2 channels per group
```

### Softmax / LayerNorm

- **Read:** Each thread reads elements of one row (stride = row length); multiple threads per row → contiguous segments are coalesced.
- **Reduction:** Warp shuffle (`__shfl_xor_sync`) for max/sum; no shared memory for the reduction itself in the warp.

---

## 3. Kernel launch configuration

| Kernel | Block | Grid | Notes |
|--------|--------|------|--------|
| **matmul_tiled** | dim3(16,16) | (ceil(N/16), ceil(N/16)) | One block per 16×16 output tile |
| **optimized_conv2d_fp16** | (16, 16, 1) | (out_w/16, out_h/16, batch*C_out) | blockIdx.z = batch*C_out + oc |
| **fused_qkv** (bias) | 256 | (M*N+255)/256 | 1D over full output |
| **softmax_fp16** | 256 | ceil(num_rows/8) | 8 rows per block |
| **layernorm_fp16** | 256 | num_rows | One block per row |
| **gelu_fp16** | 256 | (n+255)/256 | 1D |
| **mlp_linear_gelu / mlp_linear2** | 16×16 | (ceil(N1/16), ceil(M/16)) etc. | Same as tiled matmul |

From the repo:

```cuda
// gpu_kernels/transformer/transformer_kernels.cu
softmax_fp16_kernel<<<num_blocks, 256, 0, stream>>>(x, y, stride, N, num_rows);
layernorm_fp16_kernel<<<num_rows, 256, 0, stream>>>(x, weight, bias, y, stride, N, eps);
gelu_fp16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, y, n);
dim3 block(TILE_QKV, TILE_QKV);
dim3 grid1((N1 + TILE_QKV - 1) / TILE_QKV, (M + TILE_QKV - 1) / TILE_QKV);
mlp_linear_gelu_fp16_kernel<<<grid1, block, 0, stream>>>(x, w1, b1, mid, M, K, N1);
```

---

## 4. Optimization techniques used

1. **Shared memory tiling:** Matmul and conv load tiles into `__shared__` and reuse them; reduces global traffic (see [gpu_memory_hierarchy.md](gpu_memory_hierarchy.md)).
2. **Coalescing:** All global loads/stores are arranged so that adjacent threads access adjacent addresses (row/column mapping in matmul and conv).
3. **Warp shuffle reductions:** Softmax and LayerNorm use `__shfl_xor_sync` for max and sum to avoid extra shared memory and reduce warp divergence.
4. **FP16 storage, float accumulation:** Transformer kernels use `__half` in global/shared but accumulate in `float` in registers.
5. **cuBLAS for large matmul:** Fused QKV uses `cublasGemmEx` for the matmul and a small custom kernel only for bias; avoids slow custom 16×16 tiled path for large M,K,N.
6. **Vectorized loads (conv):** `__half2` for two output channels at a time in the conv extension.
7. **Loop unrolling:** `#pragma unroll` in conv inner loops (see extension).

---

## 5. Benchmark results (RTX 3050)

| Kernel | Configuration | Result |
|--------|----------------|--------|
| Matmul tiled (N=1024) | 16×16, FP32 | **~521×** speedup vs CPU; tiled much faster than naive |
| Conv 3×3 FP16 (B=1024) | Extension vs torch | **~1.49×** (0.81 ms vs 1.20 ms) |
| Conv 3×3 FP16 (B=128) | Extension vs Triton | Extension **~1.27×** vs Triton (0.11 ms vs 0.14 ms) |
| Transformer QKV (B=8,S=256,H=768) | cuBLAS + bias | Latency in ms range (see transformer_benchmark.py) |

(Exact numbers from `docs/benchmarks.md` and `python benchmarks/transformer_benchmark.py`.)

---

## 6. GPU architecture (RTX 3050)

| Item | Value |
|------|--------|
| Architecture | Ampere (sm_86) |
| Shared memory per block | Configurable with L1; 48 KB typical |
| Max threads per block | 1024 |
| Warp size | 32 |
| Registers per SM | Limited; high register use lowers occupancy |

Tiling choices (e.g. 16×16) balance shared memory size and occupancy. Use Nsight Compute to check occupancy and memory throughput (see [nsight_profiling_guide.md](nsight_profiling_guide.md)).

---

## 7. Code snippets (from this repo)

### Tiled matmul (shared tiles)

```cuda
// cuda_roadmap/level2_memory/tiled_matmul/tiled_matmul_bench.cu
#define TILE 16
__global__ void matmul_tiled(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    // ... row, col from blockIdx, threadIdx
    for (int t = 0; t < numTiles; t++) {
        As[threadIdx.y][threadIdx.x] = (row < n && colA < n) ? A[row*n+colA] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (rowB < n && col < n) ? B[rowB*n+col] : 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; k++) sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < n && col < n) C[row*n+col] = sum;
}
```

### Warp shuffle reduction (softmax)

```cuda
// gpu_kernels/transformer/transformer_kernels.cu
__device__ __forceinline__ float blockReduceMax(float val) {
    __shared__ float smem[8];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    // ... then reduce across warps
    return result;
}
```

---

## References

- [gpu_memory_hierarchy.md](gpu_memory_hierarchy.md) — memory levels and coalescing.
- [optimization_guide.md](optimization_guide.md) — high-level tuning steps.
- [transformer_gpu_kernels.md](transformer_gpu_kernels.md) — transformer-specific kernels.
