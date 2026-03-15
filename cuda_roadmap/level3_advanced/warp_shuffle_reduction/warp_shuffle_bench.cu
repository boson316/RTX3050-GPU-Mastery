/*
 * Level 3: Warp shuffle reduction — Naive (shared memory tree) vs Optimized (__shfl_down_sync)
 * Outputs: CUDA_NAIVE_MS=... CUDA_OPTIMIZED_MS=...
 */
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(1); } \
} while(0)

const int N = 1 << 20;
#define BLOCK 256

// Naive: full tree reduction in shared memory
__global__ void reduce_shared(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK + tid;
    sdata[tid] = (i < n) ? input[i] : 0.f;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Optimized: warp shuffle for the last 32 elements (no shared memory for warp level)
__device__ __forceinline__ float warp_reduce(float val) {
    for (int offset = 32 / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void reduce_warp_shuffle(const float* input, float* output, int n) {
    __shared__ float sdata[32];  // one value per warp
    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK + tid;
    float val = (i < n) ? input[i] : 0.f;
    int lane = tid % 32;
    int wid = tid / 32;
    val = warp_reduce(val);
    if (lane == 0) sdata[wid] = val;
    __syncthreads();
    if (tid < 32) val = (tid < (BLOCK + 31) / 32) ? sdata[tid] : 0.f;
    if (tid < 32) val = warp_reduce(val);
    if (tid == 0) output[blockIdx.x] = val;
}

int main() {
    int nBlocks = (N + BLOCK - 1) / BLOCK;
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)nBlocks * sizeof(float)));
    float *h_in = (float*)malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_in, h_in, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    reduce_shared<<<nBlocks, BLOCK, BLOCK*sizeof(float)>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop, 0)); CUDA_CHECK(cudaEventSynchronize(stop));
    float naive_ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start, stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    reduce_warp_shuffle<<<nBlocks, BLOCK>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop, 0)); CUDA_CHECK(cudaEventSynchronize(stop));
    float opt_ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&opt_ms, start, stop));

    printf("CUDA_NAIVE_MS=%.6f\n", naive_ms);
    printf("CUDA_OPTIMIZED_MS=%.6f\n", opt_ms);
    printf("N=%d\n", N);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_in); cudaFree(d_out); free(h_in);
    return 0;
}
