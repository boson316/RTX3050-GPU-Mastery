# Flash Attention (Triton)

Fused scaled dot-product attention with tiling and online softmax to reduce HBM reads/writes. Supports causal masking.

## Usage

```python
from triton_kernels.flash_attention.flash_attention import flash_attention_triton
out = flash_attention_triton(q, k, v, causal=False)  # q,k,v (B,H,S,D) fp16
```

## Reference

- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- Compare with `gpu_kernels/attention` (naive CUDA) and `benchmarks/attention_benchmark.py`.
