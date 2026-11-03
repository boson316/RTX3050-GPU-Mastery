# Transformer CUDA Kernels (FP16)

GPU kernels used in transformer architectures, targeting RTX 3050 (sm_86).

## Kernels

| Kernel | Description |
|--------|-------------|
| **Fused QKV** | `y = x @ W_qkv + b`: one row-major cublasLt GEMM (`Yᵀ = WᵀXᵀ`) with the bias fused into the epilogue |
| **Softmax** | Warp-per-row path (n_cols % 8 == 0, ≤ 1024): 128-bit loads, row in registers, one DRAM read. Fallback: block-per-row **online** softmax (2 reads) |
| **LayerNorm** | Warp-per-row path as above with exact two-pass statistics from registers. Fallback: block-per-row Welford (2 reads) |
| **GELU** | tanh form `0.5·x·(1 + tanh(√(2/π)(x + 0.044715x³)))`, grid-stride loop |
| **Fused MLP** | cublasLt GEMM with bias+GELU epilogue → cublasLt GEMM with bias epilogue (2 launches) |

FP16 storage, FP32 math. Block reductions always run the second stage on a full warp; the earlier
version used `__shfl_xor_sync(0xffffffff)` with 8 active lanes, which could hang (see
[docs/ENGINEERING_LOG.md](../../docs/ENGINEERING_LOG.md#4-block-reduction-could-hang-the-gpu)).
Device code lives in `transformer_kernels.cuh` and is tested on the CPU by
`tests/cuda_emulation/test_transformer_emu.cpp`.

## Build

**Requirements:** Ninja (`pip install ninja`), **Visual Studio with "Desktop development with C++"** (or Build Tools), CUDA toolkit.

**Windows:** The C++ compiler `cl.exe` must be in PATH. Either:

1. Open **"x64 Native Tools Command Prompt for VS 2022"** (or 2019) from the Start Menu, then:
   ```cmd
   cd path\to\RTX3050-GPU-Mastery
   python benchmarks/benchmark_transformer.py
   ```
   On first run the CUDA extension will JIT-compile (may take 1–2 minutes).

2. Or run `run_transformer_benchmark.bat` from the repo root; it tries to load the VS environment automatically if `cl.exe` is not in PATH (requires VS in a standard install path).

**JIT (first use):**  
Call any of the Python wrappers in `transformer_cuda.py`; the extension builds on first import when `cl.exe` and Ninja are available.

**Manual (optional):**  
From repo root, in a shell where `cl` and `nvcc` are in PATH:

```bash
pip install --no-build-isolation -e gpu_kernels/transformer/
```

## Benchmark

From repo root (FP16, latency + throughput + memory bandwidth):

```bash
python benchmarks/benchmark_transformer.py
```

Compares PyTorch reference, Triton, and these CUDA kernels on RTX 3050.
