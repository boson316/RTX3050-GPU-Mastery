# Documentation Index

Central index for **RTX 3050 GPU Optimization Lab** documentation. Beginner-friendly intro first; deeper technical docs follow.

---

## Technical articles (research lab)

In-depth technical docs for GPU engineers and advanced students. Each includes algorithm, memory access patterns, launch config, optimizations, benchmarks, and code from this repo.

| Doc | Description |
|-----|-------------|
| **[GPU Memory Hierarchy](gpu_memory_hierarchy.md)** | Levels, coalescing, tiling, roofline; RTX 3050; code snippets |
| **[CUDA Kernel Optimization](cuda_kernel_optimization.md)** | Tiled matmul, conv, transformer CUDA kernels; launch config; benchmarks |
| **[Triton Kernel Design](triton_kernel_design.md)** | Matmul, conv, layernorm, softmax, Flash Attention; grid/block; code |
| **[FlashAttention Algorithm](flashattention_algorithm.md)** | Tiled attention, online softmax; CUDA & Triton; memory O(S) vs O(S²) |
| **[Transformer GPU Kernels](transformer_gpu_kernels.md)** | QKV, Softmax, LayerNorm, GELU, MLP; warp shuffle; launch config |
| **[Nsight Profiling Guide](nsight_profiling_guide.md)** | nsys/ncu workflow; metrics; scripts; report layout |
| **[Roofline Analysis](roofline_analysis.md)** | Roofline model; ridge point; memory- vs compute-bound; plot script |

---

## Start here

| Doc | Description |
|-----|-------------|
| **[Getting Started](getting_started.md)** | Environment setup, first GPU run, RTX 3050 checklist |
| **[GPU Memory Hierarchy](gpu_memory_hierarchy.md)** | Registers → shared → L2 → global; why it matters |
| **[Memory Hierarchy Diagrams](memory_hierarchy_diagrams.md)** | Visual diagrams (Mermaid + [generated PNGs](images/README.md)) for memory, thread hierarchy, FlashAttention tiling |

---

## CUDA kernel learning

| Doc | Description |
|-----|-------------|
| **[CUDA Roadmap](cuda_roadmap.md)** | Level 1–4: vector add, reduction, matmul, tiling, Tensor Cores |
| **[Level 1: Basic Kernels](level1_kernels.md)** | Vector add, reduction, naive matmul |
| **[Level 2: Memory](level2_memory.md)** | Tiled matmul, coalescing, bank conflict |
| **[Level 3: Advanced](level3_advanced.md)** | Warp shuffle, fused ops, persistent kernel |
| **[Level 4: Tensor Core](level4_tensor_core.md)** | FP16 WMMA, Tensor Core matmul |
| **[Kernel Explanations](kernel_explanations.md)** | Conceptual notes on key kernels |

---

## Optimization & profiling

| Doc | Description |
|-----|-------------|
| **[Optimization Guide](optimization_guide.md)** | Memory-bound vs compute-bound, Nsight, roofline |
| **[Benchmarks](benchmarks.md)** | Conv, matmul, MNIST, transformer — numbers and tables |
| **[Transformer Benchmark (GPU load)](transformer_benchmark_gpu_load.md)** | Why GPU hits 100%, mitigations |

---

## Reproducibility

- **Charts:** Run `python benchmarks/generate_charts.py` (see [Getting Started](getting_started.md#generate-benchmark-charts)).
- **Full pipeline:** `scripts/reproduce_all.bat` (Windows) or `scripts/reproduce_all.sh` (Linux/macOS).

---

## Other

| Doc | Description |
|-----|-------------|
| [DCARD_POST.md](DCARD_POST.md) | Social post template (Dcard) |
| [LINKEDIN_POST.md](LINKEDIN_POST.md) | Social post template (LinkedIn) |
