# Transformer CUDA Kernels (FP16)

GPU kernels used in transformer architectures, targeting RTX 3050 (sm_86).

## Kernels

| Kernel | Description |
|--------|-------------|
| **Fused QKV** | `y = x @ W_qkv + b` — single matmul for Q,K,V projection |
| **Softmax** | Row-wise softmax over last dimension (3-pass: max, sum, normalize) |
| **LayerNorm** | `(x - mean) * rstd * weight + bias` over last dimension |
| **GELU** | `0.5 * x * (1 + tanh(sqrt(2/π)(x + 0.044715 x³)))` |
| **Fused MLP** | `out = linear2(GELU(linear1(x)))` — two matmuls + GELU |

All kernels use FP16 (`__half` / `torch.float16`) with float32 accumulation where needed.

## Build

**Requirements:** Ninja (`pip install ninja`), **Visual Studio with "Desktop development with C++"** (or Build Tools), CUDA toolkit.

**Windows:** The C++ compiler `cl.exe` must be in PATH. Either:

1. Open **"x64 Native Tools Command Prompt for VS 2022"** (or 2019) from the Start Menu, then:
   ```cmd
   cd path\to\RTX3050-GPU-Mastery
   python benchmarks/transformer_benchmark.py
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
python benchmarks/transformer_benchmark.py
```

Compares PyTorch reference, Triton, and these CUDA kernels on RTX 3050.
