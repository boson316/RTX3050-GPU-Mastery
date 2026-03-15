# Kernel Explanations

Short descriptions of each kernel and the optimizations used.

---

## Vector Add (`gpu_kernels/vector_add`)

**Operation**: `C[i] = A[i] + B[i]` for each index `i`.

- **Parallelism**: One thread per element; grid of blocks, 256 threads per block typical.
- **Memory**: Coalesced global reads and writes (contiguous 32-bit elements per warp).
- **Use case**: Bandwidth baseline; no shared memory, so useful for roofline “ridge point” and transfer benchmarks.

---

## Reduction (`gpu_kernels/reduction`)

**Operation**: Sum (or max/min) over a large array.

- **Algorithm**: Tree reduction inside each block using `__shared__`; multiple blocks produce partial sums, then combined on host (or second kernel).
- **Optimizations**: Sequential addressing to avoid bank conflicts; optional warp-level `__shfl_down_sync` for last warp to reduce shared memory.
- **Use case**: Demonstrates shared memory and synchronization; building block for softmax, norm, etc.

---

## Matrix Multiply (`gpu_kernels/matrix_mul`)

**Operation**: `C = A × B` (N×N).

- **Naive kernel**: Each thread computes one element of C by reading a row of A and a column of B from global memory → high global traffic.
- **Tiled kernel**: Block computes a TILE×TILE block of C; loads A and B tiles into `__shared__`; each thread cooperates in loading and then computes dot products from shared memory. Reduces global reads from O(N) to O(N/TILE) per output element.
- **Use case**: Classic compute-bound example; approaching peak FLOPS with good tiling and block size (e.g. 16×16 or 32×32).

---

## Conv2D (`gpu_kernels/conv2d`, `extension/conv_kernel.cu`)

**Operation**: 3×3 convolution, no padding, single input channel, multiple output channels.

- **Tiling**: Output tile (e.g. 16×16) per block; input tile includes halo (18×18 for 3×3). Weights 3×3 per output channel in shared memory.
- **FP16**: Extension uses `__half2` and 16×16 tiles for better throughput on Ampere (sm_86).
- **Use case**: CNN building block; compare with cuDNN and Triton in `benchmarks/conv_benchmark.py`.

---

## Attention (`gpu_kernels/attention`, `triton_kernels/flash_attention`)

**Operation**: `softmax(Q K^T / sqrt(d)) V`.

- **Naive CUDA**: One block per (batch, head, query row); computes scores, softmax, then weighted sum over V. Not optimized for HBM traffic.
- **Flash Attention (Triton)**: Tiled computation with online softmax and recomputation to keep working set in SRAM; much less HBM traffic and better for long sequences.
- **Use case**: Transformer kernel optimization; see `docs/optimization_guide.md` for memory complexity.

---

## Triton Matmul (`triton_kernels/matmul`)

**Operation**: `C = A @ B` with configurable block sizes.

- **Implementation**: `tl.dot` on tiles; block-wise traversal over K dimension; supports FP16 and TF32.
- **Use case**: Portable matmul for benchmarking and as reference for custom tuning.

---

## Triton Conv (`triton_kernels/conv`)

**Operation**: Same 3×3 conv as above; implemented in Triton for quick iteration and portability.

- **Grid**: One program per (batch, tile_h, tile_w, output-channel block).
- **Use case**: Compare with CUDA extension and cuDNN in `benchmarks/conv_benchmark.py`.
