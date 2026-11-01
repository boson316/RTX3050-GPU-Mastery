# Engineering log: what was wrong, how it was found, how it was fixed

The first version of this repo had fast-looking numbers and almost no verification. Some of
those numbers turned out to be wrong: a few kernels were "fast" only because they skipped part
of the work. This log records each bug with its symptom, root cause, the tool that exposed
it, the fix, and the regression test that now guards it.

The single biggest lesson: **a benchmark without a correctness check measures nothing.**
Every timing path in the repo now verifies the output against a reference *before* timing it
and rejects throughput above the hardware ceiling.

---

## How bugs were found without a GPU

All of this was done on a machine with no GPU, using three tools that now also run in CI:

| Tool | What it does | Where |
|---|---|---|
| **Triton interpreter** | `TRITON_INTERPRET=1` executes the real `@triton.jit` source with NumPy on the CPU, so indexing, masking and math are checked exactly as written | `tests/test_triton_kernels.py` |
| **CUDA execution-model emulator** | Runs the real `__global__` source on the CPU: each CUDA thread is a `std::thread`, `__syncthreads()` is a `std::barrier`, `__shfl_xor_sync` uses a per-warp exchange buffer and its own barrier. A barrier or full-mask shuffle that some threads skip **deadlocks**, so the bug can't hide | `tests/cuda_emulation/` |
| **nvcc + ptxas** | Every `.cu` compiled for `sm_86` with `-Werror`; `--resource-usage` for registers/spills | `.github/workflows/ci.yml` |

What the emulator cannot do: model timing, warps-as-SIMD, memory coalescing or Tensor Cores
(WMMA kernels are covered by self-verifying binaries that run on the GPU instead).

---

## Bugs

### 1. Triton FlashAttention only used part of the head dimension
- **Symptom:** Triton attention reported 13.6 TFLOPS vs 7.2 TFLOPS for PyTorch SDPA (B4 H8 S256 D64), i.e. "2x faster than PyTorch".
- **Root cause:** `BLOCK_D` was an autotuning parameter in {16, 32} while `head_dim = 64`. The kernel loaded `Q[:, :BLOCK_D]` and `K[:, :BLOCK_D]`, so `QK^T` was a partial dot product, and output columns `>= BLOCK_D` were never written (`torch.empty` garbage). The autotuner picked the config that did the *least* work. The file's own self-test used `D = 32`, the only size where some configs happened to be correct.
- **Found by:** interpreter tests. 13 of 14 attention cases failed, then the interpreter segfaulted on an out-of-bounds access.
- **Fix:** `BLOCK_D = next_power_of_2(head_dim)` with a `d < head_dim` mask; it is no longer tunable. Q is loaded once outside the K/V loop, causal attention skips fully-masked key blocks (about half the FLOPs), and the softmax uses `exp2` with `log2(e)` folded into the scale.
- **Test:** `test_flash_attention[*]` (D = 64, 40, 16; S = 37, 100, 130; causal and non-causal).

### 2. CUDA FlashAttention: data race, deadlock, 64x redundant work, impossible benchmark numbers
- **Symptom:** The benchmark CSV reported **0.0005 ms / 1.9 PFLOPS** for B2 H8 S512 D64. That is about 100x the GPU's peak and below the cost of a single kernel launch.
- **Root causes (old `flash_attention_cuda.cu`):**
  1. A block had `16 rows x 64 threads`, and all 64 threads of a row computed the *same* 32 dot products: 64x redundant FLOPs.
  2. Those 64 threads then ran `s_scores[row][col] = expf(s_scores[row][col] - m)` **in place, concurrently**, across two warps. A thread could read a value another warp had already exponentiated. That is a **data race**: in emulation the output was wrong (max error 0.69) even for S = 64.
  3. Threads with `row >= S` (or `d >= D`) `return`ed before the cooperative K/V tile load and the `__syncthreads()`s. For `S % 16 != 0`, some K/V rows were never loaded, and the barrier was skipped by part of the block (undefined behaviour). In emulation, S = 100 **deadlocks**.
  4. The launch was never error-checked, and the benchmark had no correctness gate and no plausibility check. Whatever produced the 0.5 µs (no kernel ran in the timed region), nothing in the harness could notice.
- **Fix:** new kernel `flash_attention_kernel.cuh`, shared by the extension and the standalone binary. Each thread owns one query row; `q[D]` and `acc[D]` live in registers; K/V tiles go through shared memory (broadcast reads, no bank conflicts); online softmax uses `exp2f`; causal tile skipping; `D ≤ 64` via template dispatch on `D_PAD ∈ {16, 32, 64}`. Every thread reaches every barrier. The launch status is returned and raised in Python. The register budget is documented: D=64 uses 215 regs and 0 spills.
- **Test:** `tests/cuda_emulation/test_flash_attention_emu.cpp` (S = 1, 17, 64, 100, 130, 257; D = 8…64; causal). Max error is about 1e-6.

### 3. QKV projection: cuBLAS read out of bounds
- **Root cause:** `cublasGemmEx(OP_T, OP_N, M, N, K, x, K, w, N, …)` treats `w` (row-major K x N) as a **column-major K x N matrix with ld = N**. Element (k, n) is then read at `w[k + n*N]`. For the QKV shape (N = 3K) the largest index is about 9K² in a 3K² buffer. A col-to-row transpose kernel and a global temp buffer that was never freed came on top of that.
- **Found by:** code review, then confirmed with a NumPy model of cuBLAS's column-major indexing. For M=3, K=4, N=12 the old call reads index 48 of a 48-element buffer.
- **Fix:** the standard row-major trick `Y^T = W^T X^T`, i.e. `cublasGemmEx(N, N, m=N, n=M, k=K, A=w, lda=N, B=x, ldb=K, C=y, ldc=N)`. It writes row-major Y directly through ATen's per-stream handle. The bias goes in a separate epilogue kernel.

### 4. Block reduction could hang the GPU
- **Symptom:** the transformer benchmark needed `TRANSFORMER_BENCHMARK_CUDA_ALL=1  # may hang` to run the custom softmax/LayerNorm/GELU kernels.
- **Root cause:** `if (threadIdx.x < 8) { … __shfl_xor_sync(0xffffffff, v, off) … }`. The full mask promises that all 32 lanes participate, but only 8 were active. That is undefined behaviour, and with Volta+ independent thread scheduling it can deadlock.
- **Found by:** reading the code; confirmed by the emulator, where the old softmax kernel deadlocks (timeout).
- **Fix:** the second reduction stage runs on the **whole** first warp, and lanes ≥ n_warps contribute the identity. While rewriting: softmax is now *online* (max and sum in one pass, so 2 passes over the row instead of 3), and LayerNorm uses Welford/Chan merging (2 passes instead of 3, no catastrophic cancellation).
- **Test:** `test_transformer_emu.cpp` (cols = 7, 33, 100, 768, 1500; block sizes 32…1024).

### 5. Triton matmul read past the end of A and B
- **Root cause:** loads masked M and N but not K: `mask=offs_am[:, None] < M`. Every K that is not a multiple of `BLOCK_K` (up to 128 in the autotune space) read out of bounds.
- **Found by:** `test_matmul_odd_shapes` (K = 50, 97, 300): 5 failures on the old code.
- **Fix:** mask `(k0 + offs_k) < K`. The three copy-pasted kernels were merged into one, with **grouped program ordering** (`GROUP_M = 8`, for L2 reuse of the B panels) and `num_stages` in the autotune space (software pipelining of the async-copy K loop).

### 6. Triton QKV / MLP: K mask without the loop offset, and a silently dropped bias
- **Root cause:** `mask=(offs_k[None, :] < K)` is always true once `k > 0` (it is missing `+ k0`). Also, `if b is None or not b.is_cuda: b = zeros(...)` replaced any non-CUDA bias with zeros.
- **Fix:** one `linear_triton` kernel with a fused epilogue (`HAS_BIAS`, `ACTIVATION = none | gelu`). The MLP is now 2 launches: GEMM+bias+GELU, then GEMM+bias. The `(M x 4H)` activation never takes an extra round trip through DRAM.

### 7. Triton conv2d wrote every column tile to the same place
- **Root cause:** the "optimized" and "autotuned" kernels stored to `oh*out_w + off_h*out_w + off_w`, missing `+ ow`. Every benchmark shape has `out_w > BLOCK_OW`, so every result was wrong (max error 22.6 on random inputs). Revealingly, `benchmark_conv()` computed `ref = F.conv2d(...)` and **never used it**.
- **Fix:** one kernel computes a `[BLOCK_C channels] x [BLOCK_OH*BLOCK_OW pixels]` tile as **9 vectorised outer-product updates** (`acc += w[:, tap] ⊗ patch_tap`). Each input patch is loaded once for all BLOCK_C channels. The autotuned variant wraps the same kernel, and the benchmark now asserts correctness.
- **Test:** `test_conv3x3[*]` (30x27 input, 20 channels).

### 8. WMMA GEMM computed A·Bᵀ
- **Root cause:** the B fragment was declared `wmma::col_major` while B is stored row-major. The inputs were the constant 0.01, for which `A @ B == A @ B^T`, so the bug was invisible. The kernel was also timed once, cold (CUDA 12 loads modules lazily on first launch).
- **Fix:** `row_major` fragments; random inputs; CPU reference; median of 20 warm runs. A new **shared-memory tiled WMMA kernel** (64x64x32 block tile, 4 warps x 2x2 fragments, `int4` global→smem copies, +8-half row padding) is reported **as a percentage of cuBLAS** rather than as a speedup over a naive kernel.

### 9. Unfair comparisons in the CUDA roadmap
| Benchmark | Problem | Fix |
|---|---|---|
| Coalescing | the "strided" kernel copied only `N/256` elements, so it did 256x less work than the coalesced one | both copy all N elements (the strided one via a permutation); report GB/s |
| Bank conflicts | contrived `sdata[tid*32]` layout that used 32 KB of smem | canonical tiled transpose, `[32][32]` vs `[32][33]` |
| Reduction | "naive" computed the full sum, "optimized" only per-block partials | both compute the full sum and are verified |
| Speedup vs CPU | "521x vs CPU" used an i-j-k triple loop as the baseline | CPU reference uses i-k-j; GPU kernels are compared with each other and with cuBLAS |

### 10. The custom conv "1.5x faster than cuDNN" compared FP16 math with FP32
- **Root cause:** the extension's float32 path converted inputs to FP16 inside the kernel and accumulated with `__hadd2`, entirely in half precision. It was then compared against cuDNN running in FP32. The output was also allocated with `torch::zeros` (an extra memset kernel), launches went to the legacy default stream, and nothing was error-checked.
- **Fix:** `conv3x3_kernel.cuh` uses FP32 accumulation for both FP32 and FP16 storage, `torch::empty`, PyTorch's current stream, and `C10_CUDA_KERNEL_LAUNCH_CHECK()`.
- **Test:** `test_conv_emu.cpp` (MNIST 28x28 plus non-tile-aligned shapes).

### 11. The roofline put measured points above the roof
- **Root cause:** peak hard-coded as "FP16 Tensor Core ~9 TFLOPS", while cuBLAS measured 14.9 TFLOPS. The device is the **RTX 3050 6GB Laptop** refresh, whose SM count and bus width differ from the original RTX 3050 Laptop, and GeForce Ampere runs FP16 MMA with FP32 accumulation at half the FP16-accumulate rate.
- **Fix:** `benchmarks/gpu_specs.py` derives peaks as `SMs x FLOPs/clk/SM x clock` (SM count from torch; clocks and bus width from NVML). The roofline and the plausibility ceiling use it. Elementwise kernels were stored with 0 GFLOPS and are now recomputed from FLOPs/latency.

### 12. Timing methodology and CI
- `time.perf_counter()` around a Python loop became **CUDA events**, `rounds x repeat` launches, median.
- The CUDA roadmap used a single cold launch; it now does warm-up plus the median of 20 event-timed runs.
- The CI ran `ruff check . || true` and `echo "file present"`, so it could never fail. It now runs ruff, the interpreter tests, an `nvcc -Werror` compile of every `.cu` file, and the emulator tests.

---

## Lessons that are now enforced in code

1. **Verify before you time.** `bench_utils.measure()` runs `check_close()` before `run_bench()`. A wrong kernel is reported as `[skip] WRONG RESULT - not timed`, never as a fast kernel.
2. **Check against physics.** Throughput above the device ceiling raises `BenchmarkError`. Bandwidth above DRAM peak is flagged as L2-resident.
3. **Test the edges.** Shapes that are not multiples of the block size, `head_dim` that is not a power of two, `S = 1`, and causal masking. Almost every bug above lived in a remainder tile.
4. **Random inputs, never constants.** Constant inputs hid the transposed WMMA operand.
5. **No hardware is no excuse.** Interpreters and emulators found every logic bug in this log without a GPU.

## Re-running everything on the GPU

```bash
pytest -q                                   # Triton (compiled) + autotuned FP16 variants
tests/cuda_emulation/run_all.sh             # CUDA kernel logic on the CPU
cd cuda_roadmap && ./build.sh && cd .. && python cuda_roadmap/run_benchmarks.py   # self-verifying CUDA benchmarks
python benchmarks/gpu_specs.py              # peaks for this exact GPU
python tools/performance_dashboard.py --skip-mnist   # verified benchmarks + report + charts
```
