# Triton LayerNorm

LayerNorm over the last dimension: `y = (x - mean) / sqrt(var + eps) * weight + bias`.

- **Baseline**: Small BLOCK_N (256); per-row, loop over N in chunks.
- **Optimized**: Larger BLOCK_N (1024); fused normalize + affine, optional mean/rstd output.
- **Autotune**: Picks BLOCK_N from [256, 512, 1024, 2048, 4096].

Supports FP16 and BF16. Benchmark vs `torch.nn.functional.layer_norm`.
