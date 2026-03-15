# CUDA Kernel Learning Roadmap

Progressive learning path: Level 1 (basics) → Level 2 (memory) → Level 3 (advanced) → Level 4 (Tensor Core).

Each kernel includes **naive** and **optimized** implementations; benchmarks compare vs **CPU** and **PyTorch**.

## Build (from this directory)

```bash
# Linux / WSL
./build.sh

# Windows (x64 Native Tools)
build.bat
```

## Run benchmarks

From repo root (so PyTorch and paths work):

```bash
python cuda_roadmap/run_benchmarks.py
```

Or run individual level binaries after building, e.g.:

```bash
./level1_basics/vector_add/vector_add_bench
```

Output format for parsing: lines like `CUDA_NAIVE_MS=0.12`, `CUDA_OPTIMIZED_MS=0.05`, `CPU_MS=1.2`, `PYTORCH_MS=0.01`.

## Docs

- [Roadmap overview](../docs/cuda_roadmap.md)
- [Level 1: Basics](../docs/level1_kernels.md)
- [Level 2: Memory](../docs/level2_memory.md)
- [Level 3: Advanced](../docs/level3_advanced.md)
- [Level 4: Tensor Core](../docs/level4_tensor_core.md)
