# Tutorial 01: CUDA Basics

Learn the fundamental concepts of GPU programming with CUDA: **thread hierarchy**, **kernel launch**, and **memory management**.

---

## Concepts

### 1. Thread hierarchy

A CUDA kernel runs many **threads** in parallel. Threads are grouped into:

- **Block** — a group of threads that can cooperate via shared memory and synchronize (`__syncthreads()`).  
  Identified by `blockIdx` (which block) and `threadIdx` (which thread within the block).
- **Grid** — a collection of blocks.  
  The kernel is launched with a **grid size** (number of blocks) and **block size** (number of threads per block).

Each thread has a unique **global index** you compute from these:

```text
global_index = blockIdx.x * blockDim.x + threadIdx.x
```

For 2D (e.g. matrices): use `blockIdx.y`, `blockDim.y`, `threadIdx.y` and form `row`, `col` accordingly.

### 2. Kernel launch

On the host (CPU), you allocate device memory (`cudaMalloc`), copy data (`cudaMemcpy`), then launch the kernel:

```cpp
kernel_name<<<grid_size, block_size>>>(arg1, arg2, ...);
```

After the kernel, you typically `cudaDeviceSynchronize()` (or use events) and copy results back.

### 3. Memory

- **Global memory** — visible to all threads; allocated with `cudaMalloc`. High latency, so we minimize accesses and prefer **coalesced** access (consecutive threads reading/writing consecutive addresses).
- **Registers** — per-thread, fastest. Used for local variables and loop indices.

---

## Simplified code: Vector Add

**Goal:** `C[i] = A[i] + B[i]` for every index `i`.

```cuda
__global__ void vector_add_kernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// Host launch (e.g. N = 1M, 256 threads per block):
int block = 256;
int grid = (N + block - 1) / block;
vector_add_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
```

- Each thread computes one output element.  
- Access is **coalesced**: thread `i` reads `A[i]`, `B[i]` and writes `C[i]` (consecutive addresses).

**Optimization (optional):** use **vectorized load/store** (e.g. `float4`) so each thread handles 4 elements, reducing the number of threads and memory transactions:

```cuda
__global__ void vector_add_float4(const float* A, const float* B, float* C, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        float4 a = *reinterpret_cast<const float4*>(A + i);
        float4 b = *reinterpret_cast<const float4*>(B + i);
        float4 c;
        c.x = a.x + b.x; c.y = a.y + b.y; c.z = a.z + b.z; c.w = a.w + b.w;
        *reinterpret_cast<float4*>(C + i) = c;
    }
    // else: scalar tail for boundary
}
```

---

## Implementations in this repository

| Topic | Location | Description |
|-------|----------|-------------|
| **Vector add (naive + float4)** | [cuda_roadmap/level1_basics/vector_add/vector_add_bench.cu](../cuda_roadmap/level1_basics/vector_add/vector_add_bench.cu) | Level 1 benchmark: naive vs optimized (vectorized). |
| **Standalone vector add** | [gpu_kernels/vector_add/vector_add.cu](../gpu_kernels/vector_add/vector_add.cu) | Simple kernel + host code; build with `nvcc vector_add.cu -o vector_add`. |
| **Reduction (sum)** | [cuda_roadmap/level1_basics/reduction/reduction_bench.cu](../cuda_roadmap/level1_basics/reduction/reduction_bench.cu) | Level 1: naive (atomicAdd) vs shared-memory tree reduction. |
| **Naive matmul** | [cuda_roadmap/level1_basics/naive_matmul/naive_matmul_bench.cu](../cuda_roadmap/level1_basics/naive_matmul/naive_matmul_bench.cu) | Level 1: matrix multiply with global memory only. |

**Run benchmarks:**

```bash
cd cuda_roadmap && ./build.sh   # or build.bat on Windows
python cuda_roadmap/run_benchmarks.py
```

**Docs:** [docs/level1_kernels.md](../docs/level1_kernels.md), [docs/cuda_roadmap.md](../docs/cuda_roadmap.md).
