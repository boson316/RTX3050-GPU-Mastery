# Triton Softmax

Softmax over the last dimension: `y = exp(x - max(x)) / sum(exp(x - max(x)))`.

- **Baseline**: Three-pass (max, sum(exp), normalize) with BLOCK_N=256.
- **Optimized**: Same algorithm with BLOCK_N=1024.
- **Autotune**: Picks BLOCK_N from [256, 512, 1024, 2048, 4096].

Supports FP16 and BF16. Benchmark vs `torch.softmax`.
