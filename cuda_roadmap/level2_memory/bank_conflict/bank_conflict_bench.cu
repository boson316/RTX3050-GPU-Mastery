/*
 * Level 2: Bank conflict — Naive (same bank) vs Optimized (padding to avoid conflict)
 * 32 banks, 4-byte words: indices 0,32,64,... share bank. Stride-32 access = conflict.
 * We do a reduction-style sum in shared memory: naive sdata[tid], optimized sdata[tid + tid/32].
 */
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(1); } \
} while(0)

const int N = 1 << 20;
#define BLOCK 256

// Naive: sdata[tid] — sequential addressing, but when we do sdata[tid + s] with s=128,64,32...
// actually sequential addressing in tree reduction avoids conflict. So "naive" = bad layout:
// store by (tid % 32) * 32 + tid/32 to force conflicts when all threads in a warp read same bank.
#define BANKS 32
// Bad: sdata[tid] then sdata[tid + 1], sdata[tid + 2] ... different banks. Good is sequential.
// To show conflict: use sdata[tid * BANKS] so all threads in warp access same bank (bank 0).
__global__ void reduce_bank_conflict(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK * BANKS];  // padded: index as sdata[tid * BANKS] for conflict
    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK + tid;
    sdata[tid * BANKS] = (i < n) ? input[i] : 0.f;  // naive: no padding, would be sdata[tid]; here we simulate conflict by using same bank
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid * BANKS] += sdata[(tid + s) * BANKS];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Optimized: use padding — sdata[tid + tid/32] so adjacent threads go to different banks
__global__ void reduce_no_conflict(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK + BANKS];  // pad to avoid bank conflict
    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK + tid;
    sdata[tid + tid / BANKS] = (i < n) ? input[i] : 0.f;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int other = tid + s;
            sdata[tid + tid/BANKS] += sdata[other + other/BANKS];
        }
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
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
    reduce_bank_conflict<<<nBlocks, BLOCK>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop, 0)); CUDA_CHECK(cudaEventSynchronize(stop));
    float naive_ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start, stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    reduce_no_conflict<<<nBlocks, BLOCK>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop, 0)); CUDA_CHECK(cudaEventSynchronize(stop));
    float opt_ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&opt_ms, start, stop));

    printf("CUDA_NAIVE_MS=%.6f\n", naive_ms);
    printf("CUDA_OPTIMIZED_MS=%.6f\n", opt_ms);
    printf("N=%d\n", N);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_in); cudaFree(d_out); free(h_in);
    return 0;
}
