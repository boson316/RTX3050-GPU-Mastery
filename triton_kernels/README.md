# Triton Kernels

High-performance Triton kernels for deep learning. Each module provides **baseline**, **optimized**, **autotuned** variants and **benchmarks vs PyTorch**. FP16/BF16 where applicable.

| Module           | Description |
|------------------|-------------|
| **matmul**       | Matrix multiply C = A @ B; block tiling, optional autotune |
| **conv**         | 3×3 Conv2D; input tile reuse across output channels |
| **layernorm**    | LayerNorm over last dim; fused normalize + affine |
| **softmax**      | Softmax over last dim; three-pass (max, sum, normalize) |
| **flash_attention** | Fused softmax(QK^T/√d)V with online softmax; causal option |

**Run all benchmarks** (from repo root):

```bash
python -m triton_kernels.run_benchmarks
```

**Install**: `pip install triton` (Linux) or `pip install triton-windows` (Windows). Run from repo root so `triton_kernels` is on path.
