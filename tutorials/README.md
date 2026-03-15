# GPU Programming Tutorials

Step-by-step tutorials for GPU programming in this repository. Each tutorial explains the concept, shows simplified code, and links to the actual implementation.

---

## Tutorial index

| # | Tutorial | Topic |
|---|----------|--------|
| 01 | [01_cuda_basics.md](01_cuda_basics.md) | CUDA basics: thread hierarchy, kernel launch, vector add, reduction |
| 02 | [02_shared_memory.md](02_shared_memory.md) | Shared memory, coalescing, bank conflicts |
| 03 | [03_matrix_tiling.md](03_matrix_tiling.md) | Matrix tiling for matmul, shared-memory tiles |
| 04 | [04_triton_kernels.md](04_triton_kernels.md) | Triton: block-level kernels, matmul, softmax |
| 05 | [05_flashattention.md](05_flashattention.md) | FlashAttention: tiling, online softmax, no full S×S matrix |
| 06 | [06_transformer_kernels.md](06_transformer_kernels.md) | Transformer building blocks: QKV, Softmax, LayerNorm, GELU, MLP |

---

## Suggested order

1. **01_cuda_basics** — Get familiar with grids, blocks, and a simple kernel.
2. **02_shared_memory** — Use shared memory and understand coalescing and bank conflicts.
3. **03_matrix_tiling** — Apply tiling to matrix multiply.
4. **04_triton_kernels** — Write the same ideas in Triton (matmul, softmax).
5. **05_flashattention** — See how tiling and online softmax give memory-efficient attention.
6. **06_transformer_kernels** — Tie everything together for transformer blocks.

---

## Running the code

- **CUDA (Level 1–4):** `cd cuda_roadmap && ./build.sh` (or `build.bat` on Windows), then `python cuda_roadmap/run_benchmarks.py`.
- **Triton:** `python -m triton_kernels.run_benchmarks`.
- **FlashAttention:** `python flash_attention_simple/benchmark_flash_attention.py`.
- **Transformer:** `python benchmarks/transformer_benchmark.py`.

See [getting_started.md](../docs/getting_started.md) and the main [README.md](../README.md) for full setup.
