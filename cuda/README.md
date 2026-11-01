# CUDA Kernels

- **vectorAdd.cu** — First kernel; 1 thread per element.
- **matrixMul.cu** — Naive + shared-memory tiled; ~521x vs a naive single-threaded i-j-k CPU loop at N=1024 (a weak baseline; [`cuda_roadmap/`](../cuda_roadmap) compares GPU kernels with each other and with cuBLAS instead).
- **reduction.cu** — Tree reduction in shared memory; 1M elements in &lt;1 ms.

Build with nvcc（Windows 需 VS x64 Native Tools）。
