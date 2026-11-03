# Changelog

## v2.1.0

Follow-ups from analysing the v2.0.0 measurements, measured on the same RTX 3050 (3 rounds; raw logs and CSVs in `results/v2_1/`, summary in `results/v2_1/SUMMARY.md`).

Conditions differ from v2.0: background apps held 532-551 MiB on the GPU at idle. Unchanged PyTorch rows are the control: softmax, LayerNorm and GELU moved by at most 0.4 %, QKV by +6.3 %, MLP by +1.8 %.

### Performance
- **Softmax / LayerNorm (CUDA):** new warp-per-row path for n_cols % 8 == 0 and n_cols <= 1024: 128-bit loads, row held in registers (one DRAM read), shuffle-only reductions, 4 rows per block. 67 / 57 GB/s to 118 / 118 GB/s; target of 100 GB/s met. Against PyTorch in the same round: softmax 1.1-1.8 % faster, LayerNorm 5.0-5.9 % faster.
- **QKV / MLP (CUDA):** bias and bias+GELU fused into the GEMM epilogue with cublasLt. MLP went from 1.1-6.0 % slower than PyTorch to 1.4-2.3 % faster. QKV's gap fell from 23.8-27.4 % to 1.1-9.9 %, 0.609 ms median vs 0.675 ms in v2.0: much closer, but the 0.545 ms target was not reached and CUDA was slower in all 3 rounds.

### Measurement
- **Attention precision made explicit:** `flash_attention_triton(precision="tf32" | "ieee")`, and `benchmarks/benchmark_attention_longseq.py` reports FP16, IEEE FP32 and TF32 separately with the error vs FP64. Triton IEEE error is 4.2e-07 to 9.7e-07. In FP16, Triton is 18-26 % faster than SDPA at equal error. In IEEE FP32, CUDA is 13.7 % faster than SDPA at S = 4096 (17.6 vs 20.4 ms). v2.0's "Triton 4x faster" was TF32 vs IEEE: the same kernel is 4.1-4.2x slower in IEEE mode. TF32 is 2.1-2.2x slower than FP16 with a larger error.
- **Row-wise kernels at two sizes:** `benchmark_transformer.py` also runs softmax / LayerNorm / GELU at 8192x768, because at 2048x768 Triton sits on its ~0.055 ms launch floor. There Triton is within 2.6 % of CUDA on all three, at 117-122 GB/s.
- **`tools/gpu_state_probe.py`:** records P-state, clocks, power and NVML throttle reasons per run. On the 1024³ WMMA spread: no recorded variable separates slow from fast runs, and cuBLAS in the same process does not slow down with WMMA (r = -0.36), against v2.0's whole-run GPU state theory. The cause is still unknown; 20 ms sampling cannot see a 0.2 ms kernel. No clusters at 2048³.

### Tests
- 12 new CPU-emulator cases for the warp-per-row kernels; GPU tests for both softmax/LayerNorm paths, odd-shaped and bias-free cublasLt GEMMs, and Triton attention in IEEE vs TF32. pytest on the GPU: 129 / 129. CPU emulator: 39 / 39.

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
- WMMA tiled GEMM: 79.1-86.5% of cuBLAS at 2048³ (6 runs). At 1024³, median 81%, range 65.6-109.6% over 12 runs; the cause of that spread is unknown.
- Long-sequence FlashAttention at S = 4096 (FP32): Triton 5.1 ms, CUDA 17.3 ms, SDPA 20.2 ms. Triton's error is about 1000x larger because `tl.dot` uses TF32.
- 20 SMs confirmed. Peaks are 21.5 TFLOPS (FP16 Tensor Core, FP32 accumulate) and 132 GB/s, replacing assumed values of 15.9 TFLOPS and 168 GB/s.

### Optimization attempts (no kernel changed)
Each candidate was measured before and after, twice, with verification. None justified a change, so none was kept. Logs are `results/v2/logs/phase4_*`.
- **WMMA tiled GEMM:** the 60% of cuBLAS target at 2048³ was already met (79.1-86.5%), so `cp.async` and 128x128 tiles were not tried. Added a `GEMM_SIZE` compile option; the default is unchanged.
- **Triton softmax:** a single-load kernel for rows that fit in one block gained at most about 1% and still trailed PyTorch slightly, so it was reverted. The kernel is bandwidth-bound. The earlier 69 vs 98 GB/s gap does not appear with the corrected timing (110 vs 116 GB/s).
- **Triton FlashAttention:** no candidate clearly beat the current autotune at S = 2048 and 4096. Larger tiles such as 128x128 exceed shared memory in FP32, and in FP16 they ran but were not faster.

### Known limits
- Kernels under about 0.1 ms remain noisy, and Triton has a launch floor of about 0.055 ms.
- MSVC 14.50 is outside CUDA 12.4's supported range and is used with `-allow-unsupported-compiler`.
- Nsight Compute was not run.
