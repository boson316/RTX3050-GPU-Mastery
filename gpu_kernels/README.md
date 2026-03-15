# GPU Kernels (Pure CUDA)

Standalone CUDA kernels for learning and benchmarking. Each subdirectory contains a single kernel and a README.

| Kernel      | Description                    | Build / Run              |
|------------|---------------------------------|--------------------------|
| vector_add | C = A + B (bandwidth baseline)  | `nvcc vector_add.cu -o vector_add` |
| reduction  | Parallel sum (shared-memory)    | `nvcc reduction.cu -o reduction`   |
| matrix_mul | C = A×B tiled (shared memory)   | `nvcc matrix_mul.cu -o matrix_mul` |
| conv2d     | 3×3 conv2d, no padding          | `nvcc -o conv2d conv2d.cu`         |
| attention  | Scaled dot-product attention    | `nvcc -o attention attention.cu`   |

**Requirements**: CUDA Toolkit (e.g. 12.x), `nvcc` in PATH. On Windows use **x64 Native Tools** or equivalent.

See `docs/kernel_explanations.md` and `docs/gpu_memory_hierarchy.md` for deeper explanations.
