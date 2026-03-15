# Transformer GPU Kernels

This document describes the **transformer building-block kernels** implemented in this repository: Fused QKV, Softmax, LayerNorm, GELU, and Fused MLP. Implementations: **PyTorch** (reference), **Triton** (where available), and **CUDA** (`gpu_kernels/transformer/`). Target: **RTX 3050 (sm_86)**, FP16.

---

## 1. Algorithm overview

| Kernel | Formula | Role in transformer |
|--------|---------|---------------------|
| **Fused QKV** | y = x @ W_qkv + b | Single matmul for Q, K, V projections |
| **Softmax** | y_i = exp(x_i - max(x)) / sum(exp(x - max)) | Attention weights (row-wise) |
| **LayerNorm** | y = (x - μ) * rstd * γ + β | Pre/post attention and MLP |
| **GELU** | 0.5·x·(1 + tanh(√(2/π)(x + 0.044715·x³))) | Activation in MLP |
| **Fused MLP** | out = linear2(GELU(linear1(x))) | Two matmuls + activation |

All use **FP16** storage with **float32 accumulation** where needed for stability.

---

## 2. GPU memory access patterns

### Fused QKV

- **Current production path:** cuBLAS `cublasGemmEx` for matmul (x @ W); then a small kernel to copy cuBLAS output (col-major) to row-major and add bias.
- **Bias kernel:** One thread per output element; `y[idx] = temp[col*M+row] + b[col]`. Writes are coalesced (idx = row*N+col).
- **Legacy tiled path:** 16×16 shared tiles for A and B; coalesced loads along K dimension; one output element per thread (coalesced along N).

```cuda
// gpu_kernels/transformer/transformer_kernels.cu
qkv_colmajor_to_rowmajor_add_bias<<<(total + 255) / 256, 256, 0, stream>>>(temp, b, y, M, N);
```

### Softmax

- **Read:** Each block handles 8 rows (SOFMAX_ROWS_PER_BLOCK). Threads stride along the row (stride = blockDim.x); segments are coalesced.
- **Reduction:** Warp shuffle for max and sum; results broadcast, then each thread writes exp(x-max)/sum for its elements. Writes coalesced.

### LayerNorm

- **One block per row.** Threads stride along the row for sum (mean), then sum of squares (variance), then normalize and affine. All loads/stores contiguous along the row → coalesced.

### GELU

- **Elementwise:** One thread per element; index = blockIdx.x*blockDim.x + threadIdx.x. Fully coalesced read and write.

### Fused MLP

- **Two tiled matmuls:** Same pattern as tiled matmul — 16×16 shared tiles, coalesced along K. First kernel: mid = GELU(x@W1+b1); second: out = mid@W2+b2. Intermediate `mid` is in global memory between the two launches.

---

## 3. Kernel launch configuration

From `gpu_kernels/transformer/transformer_kernels.cu`:

| Kernel | Block | Grid |
|--------|--------|------|
| qkv_colmajor_to_rowmajor_add_bias | 256 | ceil(M*N/256) |
| softmax_fp16_kernel | 256 | ceil(num_rows/8) |
| layernorm_fp16_kernel | 256 | num_rows |
| gelu_fp16_kernel | 256 | ceil(n/256) |
| mlp_linear_gelu_fp16_kernel | 16×16 | (ceil(N1/16), ceil(M/16)) |
| mlp_linear2_fp16_kernel | 16×16 | (ceil(N2/16), ceil(M/16)) |

```cuda
// Launch wrappers
void softmax_fp16_launch(..., int num_rows, ...) {
    int num_blocks = (num_rows + SOFMAX_ROWS_PER_BLOCK - 1) / SOFMAX_ROWS_PER_BLOCK;
    softmax_fp16_kernel<<<num_blocks, 256, 0, stream>>>(x, y, stride, N, num_rows);
}
void layernorm_fp16_launch(..., int num_rows, ...) {
    layernorm_fp16_kernel<<<num_rows, 256, 0, stream>>>(x, weight, bias, y, stride, N, eps);
}
void gelu_fp16_launch(..., int n, ...) {
    gelu_fp16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, y, n);
}
dim3 block(TILE_QKV, TILE_QKV);
dim3 grid1((N1 + TILE_QKV - 1) / TILE_QKV, (M + TILE_QKV - 1) / TILE_QKV);
mlp_linear_gelu_fp16_kernel<<<grid1, block, 0, stream>>>(x, w1, b1, mid, M, K, N1);
mlp_linear2_fp16_kernel<<<grid2, block, 0, stream>>>(mid, w2, b2, out, M, N1, N2);
```

---

## 4. Optimization techniques used

1. **cuBLAS for QKV matmul:** Avoids slow custom 16×16 tiled path for large M,K,N; keeps latency and GPU load reasonable (see `docs/transformer_benchmark_gpu_load.md`).
2. **Warp shuffle reductions:** Softmax and LayerNorm use `__shfl_xor_sync` for max and sum to avoid extra shared memory and limit warp divergence.
3. **Multi-row per block (softmax):** 8 rows per block (SOFMAX_ROWS_PER_BLOCK) to reduce block count and improve occupancy.
4. **FP16 storage, float accumulation:** All kernels accumulate in float in registers; store __half to global.
5. **Tiled matmul for MLP:** 16×16 tiles in shared memory; coalesced loads and stores.
6. **Single GELU in MLP:** First MLP kernel computes linear1 + bias then GELU in the same kernel (fused).

---

## 5. Benchmark results (RTX 3050)

Run from repo root: `python benchmarks/transformer_benchmark.py`.

Typical output (B=8, S=256, H=768, FP16) includes:

- **Latency (ms)** per kernel for PyTorch, Triton, and CUDA.
- **Memory bandwidth (GB/s)** and **throughput (GFLOPS)** where applicable.

Example (format only; actual numbers from your run):

| Kernel | PyTorch (ms) | Triton (ms) | CUDA (ms) |
|--------|--------------|-------------|-----------|
| QKV (fused) | — | — | — |
| Softmax | — | — | — |
| LayerNorm | — | — | — |
| GELU | — | — | — |
| MLP (fused) | — | — | — |

Use env: `TRANSFORMER_BENCHMARK_FULL=1` for larger B,S; `TRANSFORMER_BENCHMARK_CUDA=1` to enable custom CUDA kernels.

---

## 6. GPU architecture (RTX 3050)

| Item | Value |
|------|--------|
| SM | Ampere sm_86 |
| Block size (softmax/ln/gelu) | 256 (8 warps) |
| Tiled matmul block | 16×16 = 256 threads |
| Shared memory per block | 16×16×2 (half) × 2 tiles = 1 KB per tile; two tiles ~2 KB |

Larger tile sizes (e.g. 32×32) would increase shared memory and can reduce occupancy; 16×16 is a safe default for sm_86.

---

## 7. Code snippets (from this repo)

### Softmax: warp shuffle max

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
    if (threadIdx.x < 8) {
        val = smem[threadIdx.x];
        for (int offset = 4; offset > 0; offset /= 2)
            val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
        if (threadIdx.x == 0) result = val;
    }
    __syncthreads();
    return result;
}
```

### LayerNorm: three passes

```cuda
// Pass 1: mean
float sum = 0.f;
for (int i = threadIdx.x; i < N; i += blockDim.x)
    sum += __half2float(row_in[i]);
sum = blockReduceSum(sum);
float mean = sum / (float)N;
// Pass 2: variance
// Pass 3: normalize and affine
for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float v = (__half2float(row_in[i]) - mean) * rstd;
    row_out[i] = __float2half(v * w + b);
}
```

### GELU

```cuda
__device__ __forceinline__ float gelu_f(float x) {
    const float sqrt_2_over_pi = 0.79788456080286535588f;
    return 0.5f * x * (1.f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}
__global__ void gelu_fp16_kernel(const __half* x, __half* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(gelu_f(__half2float(x[i])));
}
```

---

## References

- [cuda_kernel_optimization.md](cuda_kernel_optimization.md) — tiling and launch config.
- [transformer_benchmark_gpu_load.md](transformer_benchmark_gpu_load.md) — why QKV uses cuBLAS and load control.
- [roofline_analysis.md](roofline_analysis.md) — memory-bound vs compute-bound for these kernels.
