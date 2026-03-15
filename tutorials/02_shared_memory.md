# Tutorial 02: Shared Memory

Use **shared memory** to make threads within a block cooperate: load data once from global memory, then reuse it from fast on-chip memory. This tutorial also covers **coalescing** and **bank conflicts**.

---

## Concepts

### 1. What is shared memory?

- **Shared memory** is a small, fast memory **per block**. All threads in the block can read and write it.
- It is much faster than global memory (~20–30 cycles vs hundreds). Use it for **tiles** of data that multiple threads need, or for **reductions** within a block.
- Declare with `__shared__`; size can be fixed or dynamic (passed as the third argument to the kernel launch).

### 2. Synchronization

After writing to shared memory, threads must **synchronize** before others read it:

```cuda
__syncthreads();  // all threads in the block wait here
```

Use `__syncthreads()` between phases: e.g. after loading a tile, before using it; after one step of a reduction, before the next.

### 3. Coalesced access (global memory)

Even when using shared memory, we still load from global memory first. **Coalescing** means: consecutive threads access **consecutive addresses**. The GPU then merges these into fewer memory transactions.

- **Good:** thread `i` reads `in[i]` (or `in[i * 4]` with float4).
- **Bad:** thread `i` reads `in[i * stride]` with large `stride` — each warp touches many cache lines.

### 4. Bank conflicts (shared memory)

Shared memory is divided into **32 banks** (for 4-byte words). If multiple threads in the same warp access different addresses that map to the **same bank**, the access is **serialized** (bank conflict).

- **Sequential indexing** `sdata[tid]`: threads 0–31 access different banks → no conflict.
- **Stride-32** e.g. `sdata[tid * 32]`: all threads in a warp hit bank 0 → conflict.
- **Fix:** pad the shared array so that “logical” indices map to different banks, e.g. `sdata[tid + tid/32]` with total size `BLOCK + 32`.

---

## Simplified code: Reduction in shared memory

**Goal:** Sum all elements of an array. Each block sums a chunk into one value; the host (or a second kernel) can sum the per-block results.

**Naive (slow):** every thread does `atomicAdd(output, input[i])` — serialized at one global address.

**Optimized:** each block loads its chunk into **shared memory**, then does a **tree reduction** (half the threads add two values, then half again, until one value remains). Only one thread writes the block result to global memory.

```cuda
__global__ void reduce_shared(const float* input, float* output, int n) {
    extern __shared__ float sdata[];  // dynamic size: blockDim.x floats
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? input[i] : 0.f;
    __syncthreads();

    // Tree reduction: in each step, half the threads add two elements
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

// Launch with dynamic shared memory size:
reduce_shared<<<numBlocks, 256, 256 * sizeof(float)>>>(d_in, d_out, N);
```

### Avoiding bank conflicts in reduction

With sequential indexing `sdata[tid]` and the usual tree (thread `tid` adds `sdata[tid]` and `sdata[tid + s]`), bank conflicts can occur for some step sizes. Padding avoids this:

```cuda
__shared__ float sdata[BLOCK + 32];  // pad
sdata[tid + tid / 32] = (i < n) ? input[i] : 0.f;
// ... then use same indexing in the loop: sdata[tid + tid/32] += sdata[(tid+s) + (tid+s)/32]
```

---

## Simplified code: Coalesced vs strided copy

To **feel** the effect of coalescing, compare:

- **Strided (bad):** thread `i` reads `in[i * stride]`, so consecutive threads touch far-apart addresses.
- **Coalesced (good):** thread `i` reads `in[i]`.

```cuda
// Bad: strided
__global__ void copy_strided(const float* in, float* out, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * stride;
    if (idx < n) out[idx] = in[idx];
}

// Good: coalesced
__global__ void copy_coalesced(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}
```

---

## Implementations in this repository

| Topic | Location | Description |
|-------|----------|-------------|
| **Reduction (atomic vs shared)** | [cuda_roadmap/level1_basics/reduction/reduction_bench.cu](../cuda_roadmap/level1_basics/reduction/reduction_bench.cu) | Naive atomicAdd vs shared-memory tree reduction. |
| **Coalescing** | [cuda_roadmap/level2_memory/coalescing/coalescing_bench.cu](../cuda_roadmap/level2_memory/coalescing/coalescing_bench.cu) | Strided vs coalesced copy; measures bandwidth. |
| **Bank conflict** | [cuda_roadmap/level2_memory/bank_conflict/bank_conflict_bench.cu](../cuda_roadmap/level2_memory/bank_conflict/bank_conflict_bench.cu) | Reduction with bad layout (conflict) vs padded layout (no conflict). |

**Docs:** [docs/level2_memory.md](../docs/level2_memory.md).

**Run:** from repo root, build the CUDA roadmap then run the Level 2 benchmarks (coalescing, bank_conflict are in `level2_memory`).
