# Simplified FlashAttention

Implements **attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V** with:

1. **Memory optimization**: no full S×S attention matrix; tiled computation with online softmax.
2. **Tiling**: Q rows in blocks of BLOCK_M, K/V in BLOCK_N; intermediate results in shared memory (CUDA) / SRAM (Triton).

## Implementations

| Implementation | File | Notes |
|----------------|------|--------|
| **PyTorch reference** | `reference_pytorch.py` | Full attention matrix (O(S²) memory). For correctness only. |
| **CUDA** | `flash_attention_cuda.cu` | Tiled kernel, shared memory, online softmax. D ≤ 64. |
| **Triton** | `attention_triton.py` | Wraps `triton_kernels.flash_attention`. Tiled, no full matrix. |

## Build (CUDA)

**Option A – JIT (Python):** On first use, the CUDA extension is built via `torch.utils.cpp_extension.load` (needs Ninja, Visual Studio, CUDA). If JIT fails (e.g. missing Ninja or DLL), use Option B.

**Option B – Standalone executable (Windows):** nvcc needs `cl.exe` in PATH. Use the provided build script:

```bat
cd flash_attention_simple
build.bat
```

If you see "Cannot find compiler 'cl.exe'", open **x64 Native Tools Command Prompt for VS** from the Start Menu, then run `build.bat` again. After a successful build, run (Windows adds `.exe`):

```bat
flash_attention_cuda_standalone.exe 128
flash_attention_cuda_standalone.exe 256
flash_attention_cuda_standalone.exe 512
flash_attention_cuda_standalone.exe 1024
```

Each run prints `CUDA_MS=...` so you can compare with the PyTorch/Triton table from `python flash_attention_simple/benchmark_flash_attention.py`.

## Benchmark

From repo root:

```bash
python flash_attention_simple/benchmark_flash_attention.py
```

Compares **PyTorch**, **CUDA**, and **Triton** for sequence lengths **128, 256, 512, 1024** (B=2, H=8, D=64, float32).

## Usage

```python
from flash_attention_simple import attention_pytorch, attention_triton, attention_cuda

# q, k, v: (B, H, S, D) on CUDA
out_pt = attention_pytorch(q, k, v)
out_tr = attention_triton(q, k, v)
out_cu = attention_cuda(q, k, v)  # None if extension unavailable or D>64
```
