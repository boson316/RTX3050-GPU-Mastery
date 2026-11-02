# Changelog

## v2.0.0

First release verified on the real GPU (RTX 3050 6GB Laptop). Measurements are in the README; raw logs and CSVs are in `results/v2/`.

### Correctness
- Fixed 12 bugs found without a GPU, including a data race, deadlocks, an out-of-bounds cuBLAS call, a transposed WMMA operand, masking bugs and unfair baselines. See `docs/ENGINEERING_LOG.md`, #1 to #12.
- Fixed 5 more found on the GPU:
  - #13: the FlashAttention JIT build failed on current MSVC.
  - #14: the documented install replaced CUDA PyTorch with a CPU build.
  - #15: sub-millisecond benchmarks were timed before the clocks left idle, varying by up to 16x.
  - #16: the roofline had no correctness check, dropped failures silently and timed `torch.randn` with some kernels.
  - #17: the dashboard overwrote a chart with hard-coded pre-fix numbers.

### Verification
- Every timing path now verifies output before timing it and preheats the GPU for 1 s.
- `pytest` on the GPU covers compiled Triton kernels in FP32 and FP16, the autotuned variants, all three PyTorch CUDA extensions and the timing harness (116 cases).
- The CUDA CPU emulator builds and passes with MSVC (27 cases).
- Charts are drawn only from verified CSVs.

### Measured results
- cuBLAS at 2048³: 14.36 TFLOPS, which is 67% of the roof at the NVML maximum clock and 81% at the sustained clock of about 1747 MHz.
- WMMA tiled GEMM at 1024³: median 81% of cuBLAS, range 65.6-109.6% over 12 runs. The cause of the spread is unknown.
- Long-sequence FlashAttention at S = 4096 (FP32): Triton 5.1 ms, CUDA 17.3 ms, SDPA 20.2 ms. Triton's error is about 1000x larger because `tl.dot` uses TF32.
- 20 SMs confirmed. Peaks are 21.5 TFLOPS (FP16 Tensor Core, FP32 accumulate) and 132 GB/s, replacing assumed values of 15.9 TFLOPS and 168 GB/s.

### Known limits
- Kernels under about 0.1 ms remain noisy, and Triton has a launch floor of about 0.055 ms.
- MSVC 14.50 is outside CUDA 12.4's supported range and is used with `-allow-unsupported-compiler`.
- Nsight Compute was not run.
