# Custom Attention

Thin wrapper around PyTorch `scaled_dot_product_attention` with optional Triton Flash backend for benchmarking.

## Usage

```python
from pytorch_extensions.custom_attention import custom_attention
out = custom_attention(q, k, v, causal=False, use_triton=True)
```

See `benchmarks/attention_benchmark.py` for timing vs PyTorch SDPA and Triton Flash.
