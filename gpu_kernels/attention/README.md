# Attention (Scaled Dot-Product)

Reference CUDA implementation of **softmax(Q K^T / sqrt(d)) V**. One block per (batch, head, query position); each block computes one output row.

## Build

```bash
nvcc -o attention attention.cu
```

## Limitations

- This kernel is **not** memory-bandwidth or compute optimized; it is for teaching and correctness reference.
- For transformer-scale workloads use **Flash Attention** (see `triton_kernels/flash_attention/` and `docs/optimization_guide.md`).

## Flash Attention

Flash Attention reduces HBM traffic by tiling and keeping softmax state in SRAM, enabling longer context and higher throughput. The Triton flash attention kernel in this repo demonstrates that pattern.
