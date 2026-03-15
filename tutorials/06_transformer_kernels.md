# Tutorial 06: Transformer Kernels

Modern transformer blocks are built from a small set of operations: **QKV projection**, **attention** (e.g. FlashAttention), **output projection**, **LayerNorm**, **GELU**, and **MLP** (two linear layers with GELU in between). This tutorial summarizes each building block and points to the **fused or dedicated kernels** in this repository (PyTorch reference, Triton, and CUDA).

---

## Concepts

### 1. Transformer block layout

A typical block (e.g. decoder-only):

1. **LayerNorm** on input → **QKV projection** → **Attention** (e.g. FlashAttention) → **Output projection** → residual + **LayerNorm**
2. **MLP:** `x = LayerNorm(x); x = x + MLP(x)` where `MLP(x) = linear2(GELU(linear1(x)))`

So the **kernels** we care about are: **LayerNorm**, **QKV matmul (+ bias)**, **attention**, **output matmul**, **GELU**, **linear1**, **linear2**.

### 2. Why fuse?

- Each **kernel launch** has overhead; reading/writing the same tensor from global memory multiple times is costly.
- **Fusing** means doing several steps in one kernel (e.g. linear + bias, or LayerNorm + residual) so we read/write global memory fewer times and use registers/shared memory for intermediates.

### 3. Kernels used in this repo

| Kernel | Formula / role | Notes |
|--------|----------------|--------|
| **Fused QKV** | \( y = x W_{qkv} + b \) | One matmul for Q, K, V; then split or use in attention. |
| **Softmax** | Row-wise softmax over last dim | 3-pass: max, sum(exp), normalize. |
| **LayerNorm** | \( (x - \mu) \cdot \mathrm{rstd} \cdot \gamma + \beta \) | Over last dimension; fused with affine. |
| **GELU** | \( 0.5 \cdot x \cdot (1 + \tanh(\sqrt{2/\pi}(x + 0.044715 x^3))) \) | Used in MLP; FP16 with float32 accumulation where needed. |
| **Fused MLP** | `linear2(GELU(linear1(x)))` | Two matmuls + GELU in the middle. |

All are implemented in **FP16** with **float32 accumulation** where necessary for stability.

---

## Simplified code ideas

### LayerNorm (last dimension)

Per row: compute mean and variance over the last dim, then `y = (x - mean) * rstd * weight + bias`. In Triton/CUDA we tile the last dimension and use reductions (sum for mean, sum of squares for var).

### GELU

Elementwise: `out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`. Often kept in float32 internally then cast back to FP16.

### QKV

Single matmul: `Y = X @ W_qkv + b` where `W_qkv` is `[hidden, 3*hidden]`; then split the last dimension into Q, K, V for the attention kernel.

### Softmax

See [04_triton_kernels.md](04_triton_kernels.md): row max → row sum(exp) → write `exp(x-max)/sum`.

---

## Implementations in this repository

| Kernel | Location | Description |
|--------|----------|-------------|
| **Transformer CUDA (QKV, Softmax, LayerNorm, GELU, MLP)** | [gpu_kernels/transformer/](../gpu_kernels/transformer/) | CUDA kernels and PyTorch bindings; used in transformer benchmark. |
| **transformer_kernels.cu** | [gpu_kernels/transformer/transformer_kernels.cu](../gpu_kernels/transformer/transformer_kernels.cu) | Core CUDA implementations. |
| **transformer_cuda.py** | [gpu_kernels/transformer/transformer_cuda.py](../gpu_kernels/transformer/transformer_cuda.py) | Python API and JIT build. |
| **Triton QKV** | [triton_kernels/qkv/qkv_triton.py](../triton_kernels/qkv/qkv_triton.py) | Fused QKV projection. |
| **Triton Softmax** | [triton_kernels/softmax/softmax_triton.py](../triton_kernels/softmax/softmax_triton.py) | Row-wise softmax. |
| **Triton LayerNorm** | [triton_kernels/layernorm/layernorm_triton.py](../triton_kernels/layernorm/layernorm_triton.py) | Fused LayerNorm. |
| **Triton GELU** | [triton_kernels/gelu/gelu_triton.py](../triton_kernels/gelu/gelu_triton.py) | GELU activation. |
| **Triton MLP** | [triton_kernels/mlp/mlp_triton.py](../triton_kernels/mlp/mlp_triton.py) | Fused MLP block. |
| **Transformer benchmark** | [benchmarks/transformer_benchmark.py](../benchmarks/transformer_benchmark.py) | Compares PyTorch, Triton, and CUDA kernels (latency, throughput, bandwidth). |

**Build (CUDA extension):** from repo root, in a shell with `cl.exe` and `nvcc` in PATH (e.g. x64 Native Tools Command Prompt for VS):

```bash
pip install --no-build-isolation -e gpu_kernels/transformer/
```

Or run the benchmark once; it will JIT-build the extension:

```bash
python benchmarks/transformer_benchmark.py
```

**Docs:** [gpu_kernels/transformer/README.md](../gpu_kernels/transformer/README.md), [docs/transformer_gpu_kernels.md](../docs/transformer_gpu_kernels.md), [docs/transformer_benchmark_gpu_load.md](../docs/transformer_benchmark_gpu_load.md).
