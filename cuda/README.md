# CUDA Kernels

- **vectorAdd.cu** — First kernel; 1 thread per element.
- **matrixMul.cu** — Naive + shared-memory tiled; **~521x** speedup vs CPU at N=1024.
- **reduction.cu** — Tree reduction in shared memory; 1M elements in &lt;1 ms.

Build with nvcc（Windows 需 VS x64 Native Tools）。
