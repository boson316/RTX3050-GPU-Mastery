# Tutorial Notebooks — RTX 3050 GPU Lab

Beginner-friendly notebooks with **runnable examples** on RTX 3050 (or any CUDA GPU). Run from **repository root** so imports work.

## List

### Kernel demos (algorithm + usage + benchmark + plots)

| Notebook | Content |
|----------|---------|
| **cuda_kernel_demo.ipynb** | CUDA kernels: transformer (QKV, softmax, layernorm, GELU, MLP), optional custom_conv; example usage, benchmark, bar chart |
| **triton_kernel_demo.ipynb** | Triton kernels: matmul, conv2d, layernorm; algorithm, example usage, benchmark, latency/GFLOPS plots |
| **flashattention_demo.ipynb** | FlashAttention: tiling, online softmax; Triton usage, correctness vs SDPA, benchmark, performance chart |
| **transformer_kernel_demo.ipynb** | Transformer building blocks (QKV, Softmax, LayerNorm, GELU, MLP): PyTorch vs Triton vs CUDA; algorithm, usage, benchmark, bar chart |

### Tutorials

| Notebook | Content |
|----------|---------|
| **01_getting_started_rtx3050.ipynb** | Check GPU, first PyTorch CUDA ops (vector add, matmul) |
| **02_gpu_memory_and_roofline.ipynb** | Memory hierarchy, arithmetic intensity, roofline plot |
| **03_benchmarks_matmul_conv.ipynb** | Matmul (PyTorch vs Triton), Conv (torch vs extension vs Triton) |
| **04_transformer_and_profiling.ipynb** | Transformer kernel benchmark, profiling intro, roofline |

## Run

From repo root:

```bash
pip install jupyter  # if needed
jupyter notebook notebooks/
```

Or open each `.ipynb` in VS Code / Cursor with the Jupyter extension. Set kernel to the environment where `torch` and (optionally) `triton` are installed.

## Requirements

- PyTorch with CUDA
- matplotlib (for roofline in 02, 04)
- triton / triton-windows (for Triton cells in 03)
- Optional: custom_conv extension (pip install in `extension/`) for 03

All examples use small-to-medium sizes suitable for **RTX 3050 6GB**.
