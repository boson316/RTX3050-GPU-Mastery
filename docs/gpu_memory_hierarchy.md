# GPU Memory Hierarchy

Understanding the memory hierarchy is essential for kernel optimization (e.g. for ML infrastructure roles at NVIDIA, OpenAI, DeepMind). This document explains each level, access patterns, and how they appear in kernels in this repository.

---

## 1. Algorithm: Why hierarchy matters

GPUs hide **memory latency** by running many threads. To keep them busy, data must be supplied fast enough. The hierarchy is organized by **speed and scope**:

- **Fast, private** (registers) → **fast, shared** (shared memory) → **slower, shared** (L1/L2) → **slow, global** (VRAM).

Kernel optimization often means: **move working set from global into shared/registers** so that each byte fetched from global memory is reused many times (high arithmetic intensity).

---

## 2. Levels (fast → slow)

| Level | Scope | Latency (approx) | Bandwidth | Use in kernels |
|-------|--------|------------------|-----------|----------------|
| **Registers** | Per-thread | 0 cycles | Highest | Locals, loop indices, accumulators |
| **Shared memory** | Per-block | ~20–30 cycles | Very high | Tiles, reductions, cooperation |
| **L1 / L2 cache** | Per-SM / GPU | Variable | High | Implicit from global access patterns |
| **Global (VRAM)** | GPU | 200–400 cycles | Lower | Large tensors, inputs/outputs |

### RTX 3050 (Ampere sm_86) reference

| Spec | Value |
|------|--------|
| FP16 peak | ~9 TFLOPS |
| Memory bandwidth | ~192 GB/s |
| Ridge point (FP16) | ~47 FLOP/byte |

---

## 3. GPU memory access patterns

### Coalescing (global memory)

- **Idea:** One warp (32 threads) should access **consecutive addresses** so the memory system can issue one or few transactions.
- **Good:** `threadIdx.x` → `base + threadIdx.x` (contiguous).
- **Bad:** Strided access (e.g. `base + threadIdx.x * N`) or random access increases transactions per warp.

Example from this repo (bias copy — contiguous):

```cuda
// gpu_kernels/transformer/transformer_kernels.cu
__global__ void qkv_fill_bias_kernel(__half* y, const __half* b, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int col = idx % N;
        y[idx] = b ? b[col] : __float2half(0.f);  // coalesced write
    }
}
```

### Shared memory: tiling

- Load a **tile** from global in a coalesced way into `__shared__`; then have threads read from shared memory (reuse).
- **Bank conflicts:** 32 banks; stride that is a multiple of 32 can serialize access. Use padding or different layouts.

Tiled matmul in this repo (each thread loads one element of A and B tiles):

```cuda
// cuda_roadmap/level2_memory/tiled_matmul/tiled_matmul_bench.cu
__global__ void matmul_tiled(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0;
    for (int t = 0; t < numTiles; t++) {
        int colA = t * TILE + threadIdx.x, rowB = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < n && colA < n) ? A[row*n+colA] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (rowB < n && col < n) ? B[rowB*n+col] : 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; k++) sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < n && col < n) C[row*n+col] = sum;
}
```

---

## 4. Kernel launch configuration (typical in this repo)

| Kernel type | Block shape | Grid | Shared memory |
|-------------|-------------|------|----------------|
| Tiled matmul (16×16) | 16×16 | (N/16, N/16) | 2×16×16 floats |
| Softmax (row-wise) | 256 1D | (num_rows / 8, 1) | warp shuffle + 8 floats |
| LayerNorm | 256 1D | (num_rows, 1) | warp shuffle |
| GELU (elementwise) | 256 1D | (n+255)/256 | — |
| Conv 3×3 (extension) | 16×16, blockIdx.z = batch*ch | 2D×batch*C_out | tile_in 18×18, tile_w |

---

## 5. Optimization techniques used

1. **Coalescing:** All global loads/stores in the kernels above are arranged so that adjacent threads touch adjacent addresses where possible.
2. **Shared memory tiling:** Matmul, conv, and FlashAttention use tiles to reduce global traffic.
3. **Warp-level primitives:** Softmax/LayerNorm use `__shfl_xor_sync` for reduction (see `transformer_kernels.cu`) to avoid extra shared memory and reduce divergence.
4. **Float32 accumulation:** FP16 kernels accumulate in float in registers to preserve precision (e.g. transformer kernels, Triton matmul).

---

## 6. Roofline (where hierarchy meets performance)

- **Arithmetic intensity** = FLOPs / (bytes read + written from global memory).
- If intensity **&lt; ridge point** (peak_FLOPS / peak_BW), the kernel is **memory-bound** (e.g. vector add, small matmul).
- If intensity **&gt; ridge point**, **compute-bound** (e.g. large tiled matmul). Tiling and shared memory move you toward the compute roof.

See [roofline_analysis.md](roofline_analysis.md) and [memory_hierarchy_diagrams.md](memory_hierarchy_diagrams.md) for plots and diagrams.

---

## 7. Benchmark results (RTX 3050)

| Kernel | Configuration | Effect of hierarchy |
|--------|----------------|---------------------|
| Matmul naive vs tiled (N=1024) | 16×16 tile, shared | **~521×** vs CPU; tiled much faster than naive (less global traffic) |
| Conv torch vs extension | 16×16 tile, shared tile_in/tile_w | **~1.5×** over torch (better reuse) |
| Transformer QKV | cuBLAS matmul + bias kernel | Matmul uses library-optimized access; bias is trivial coalesced write |

---

## 8. Diagrams

See **[memory_hierarchy_diagrams.md](memory_hierarchy_diagrams.md)** for Mermaid and ASCII diagrams (Host/GPU, tiled matmul data flow, roofline).

---

## References

- NVIDIA "Maximizing Memory Throughput" and "CUDA C++ Best Practices Guide".
- [optimization_guide.md](optimization_guide.md) in this repo for concrete tuning steps.
