# Optimization Guide

Practical steps for optimizing CUDA and Triton kernels, suitable for ML infrastructure and GPU programming roles.

---

## 1. Establish a baseline

- **Benchmark** with fixed input sizes and warmup; report mean time and optionally variance.
- Compare **vs a reference**: e.g. PyTorch/cuBLAS, cuDNN, or a simple CPU implementation.
- Use **CUDA events** for GPU timing (`cudaEventRecord` / `cudaEventSynchronize` / `cudaEventElapsedTime`) to avoid host sync bias.

---

## 2. Memory-bound kernels

- **Coalescing**: Ensure adjacent threads access adjacent addresses (e.g. row-major with threadIdx.x along columns).
- **Shared memory**: Load global data in coalesced fashion into `__shared__`; then have threads read from shared memory (with attention to bank conflicts).
- **Vectorized loads**: Use float4 / int4 where applicable to increase effective bandwidth.
- **Roofline**: Measure arithmetic intensity; if below ridge point, focus on reducing bytes per FLOP (tiling, recomputation).

---

## 3. Compute-bound kernels

- **Tiling**: For matmul/conv, tile so that each block’s working set fits in shared memory; iterate over K (or equivalent) dimension.
- **Block size**: Try 8×8, 16×16, 32×32 (matmul); balance occupancy and register pressure (Nsight Compute).
- **FP16 / TF32**: Use half precision or TF32 where accuracy allows to increase throughput on modern GPUs.
- **Loop unrolling**: `#pragma unroll` on inner loops; Triton often unrolls by default.

---

## 4. Reduction and softmax

- **Tree reduction**: Logarithmic steps in shared memory; avoid warp divergence in the last warp (e.g. use `__shfl_down_sync`).
- **Online softmax**: For attention, use max-subtraction and running sum to avoid two passes (see Flash Attention).

---

## 5. Profiling (Nsight)

- **Nsight Systems**: Timeline, kernel launch overhead, memory copy; identify bottlenecks.
- **Nsight Compute**: Per-kernel metrics: occupancy, memory throughput (L1/L2/DRAM), warp stall reasons, FLOPs.
- **Roofline**: Compute theoretical intensity; compare achieved GFLOPS vs peak; determine if memory- or compute-bound.

---

## 6. Reproducibility

- **Fixed seeds** and input sizes in benchmarks.
- **Document** GPU model, driver, CUDA version, and (if applicable) Triton version.
- **CI**: Run benchmarks on CPU or small inputs when GPU not available; full benchmarks in a separate workflow or locally.

---

## 7. Transformer / attention

- **Flash Attention**: Fused kernel with tiling and online softmax; O(N) memory for attention matrix vs O(N²) materialization. Use for long context.
- **Causal masking**: Implement with block-level bounds or explicit mask in Triton/CUDA.

See `triton_kernels/flash_attention` and `gpu_kernels/attention` in this repo.
