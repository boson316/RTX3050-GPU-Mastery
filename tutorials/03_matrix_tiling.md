# Tutorial 03: Matrix Tiling

**Tiling** (blocking) is the key technique for fast matrix multiply on the GPU: load small **tiles** of A and B into **shared memory**, then compute the corresponding tile of C with much less global memory traffic.

---

## Concepts

### 1. Why tiling?

In naive matmul, each thread reads a full row of A and a full column of B from global memory — **O(N)** accesses per thread for one output element. That is memory-bound and slow.

**Idea:** partition the output matrix C into **tiles** (e.g. 16×16). Each **block** is responsible for one tile of C. To compute that tile, the block needs the corresponding row tiles of A and column tiles of B. Load each such tile into **shared memory** once; all threads in the block can reuse it. Then iterate over the K dimension: load a tile of A and a tile of B → multiply → accumulate into the output tile. Global reads drop to **O(N / TILE)** per element.

### 2. Roles of threads

- **Block:** one tile of C, e.g. `TILE×TILE` (e.g. 16×16). So `blockDim = (TILE, TILE)` and `gridDim = (ceil(N/TILE), ceil(N/TILE))`.
- **Thread:** each thread can compute one (or a few) elements of the output tile. Simplest: one thread per output element in the tile, so `threadIdx.y`, `threadIdx.x` are the row/col inside the tile.
- **Global indices:**  
  `row = blockIdx.y * TILE + threadIdx.y`  
  `col = blockIdx.x * TILE + threadIdx.x`

### 3. Loop over K

For each tile of C we need to sum over all K. So we loop in steps of TILE along K:

- Load tile of A: `A[row, k_start : k_start+TILE]` into `As[threadIdx.y][:]`.
- Load tile of B: `B[k_start : k_start+TILE, col]` into `Bs[:][threadIdx.x]`.
- `__syncthreads()` so all threads see the loaded tiles.
- Compute partial dot product: `sum += As[threadIdx.y][k] * Bs[k][threadIdx.x]` for `k = 0..TILE-1`.
- `__syncthreads()` before loading the next tiles (avoid overwriting before everyone has read).

---

## Simplified code: Tiled matmul

**Goal:** C = A × B (N×N). Tiles of size `TILE×TILE` (e.g. 16×16).

```cuda
#define TILE 16

__global__ void matmul_tiled(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0;
    int numTiles = (n + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; t++) {
        int colA = t * TILE + threadIdx.x;
        int rowB = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < n && colA < n) ? A[row * n + colA] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (rowB < n && col < n) ? B[rowB * n + col] : 0.f;
        __syncthreads();

        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < n && col < n)
        C[row * n + col] = sum;
}

// Launch:
dim3 block(TILE, TILE);
dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
```

- **Loading A:** each thread loads one element of the current A-tile into `As`; the tile is shared by the block.
- **Loading B:** same for B into `Bs`. Layout is chosen so that consecutive threads access consecutive addresses (coalesced).
- **Compute:** each thread holds one accumulator `sum` and updates it along K; finally writes one element of C.

Larger tiles (e.g. 32×32) improve reuse but use more shared memory and registers; often we progress to **register blocking** (each thread computes a small block of C) and **Tensor Cores** (WMMA) for FP16.

---

## Implementations in this repository

| Topic | Location | Description |
|-------|----------|-------------|
| **Tiled matmul (naive vs tiled)** | [cuda_roadmap/level2_memory/tiled_matmul/tiled_matmul_bench.cu](../cuda_roadmap/level2_memory/tiled_matmul/tiled_matmul_bench.cu) | Level 2: global-only matmul vs 16×16 tiled shared-memory matmul. |
| **Naive matmul** | [cuda_roadmap/level1_basics/naive_matmul/naive_matmul_bench.cu](../cuda_roadmap/level1_basics/naive_matmul/naive_matmul_bench.cu) | Baseline: no tiling. |
| **Pure CUDA matmul** | [gpu_kernels/matrix_mul/matrix_mul.cu](../gpu_kernels/matrix_mul/matrix_mul.cu) | Matrix multiply kernel used in benchmarks. |
| **FP16 Tensor Core (WMMA)** | [cuda_roadmap/level4_tensor_core/](../cuda_roadmap/level4_tensor_core/) | Level 4: tiling with Tensor Cores for FP16. |

**Docs:** [docs/level2_memory.md](../docs/level2_memory.md), [docs/level4_tensor_core.md](../docs/level4_tensor_core.md).

**Run:** build `cuda_roadmap`, then run `python cuda_roadmap/run_benchmarks.py` (includes tiled matmul and CPU comparison).
