/*
 * Level 2: Memory Coalescing — Naive (strided read) vs Optimized (coalesced read)
 * Copies input to output; we measure bandwidth. Strided = thread i reads [i*stride].
 * Outputs: CUDA_NAIVE_MS=... CUDA_OPTIMIZED_MS=...
 */
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(1); } \
} while(0)

const int N = 1 << 20;  // 1M floats
const int STRIDE = 256; // bad: each warp touches 256 separate cache lines

// Naive (bad): strided access — thread i reads in[i*STRIDE], writes out[i*STRIDE]
__global__ void copy_strided(const float* __restrict__ in, float* __restrict__ out, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * stride;
    if (idx < n) out[idx] = in[idx];
}

// Optimized (good): coalesced — thread i reads in[i], writes out[i]
__global__ void copy_coalesced(const float* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

int main() {
    size_t size = (size_t)N * sizeof(float);
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, size));
    CUDA_CHECK(cudaMalloc(&d_out, size));
    float *h = (float*)malloc(size);
    for (int i = 0; i < N; i++) h[i] = (float)i;
    CUDA_CHECK(cudaMemcpy(d_in, h, size, cudaMemcpyHostToDevice));

    int n_strided = N / STRIDE;  // number of elements we actually touch in strided
    int block = 256, grid_std = (n_strided + block - 1) / block;
    int grid_co = (N + block - 1) / block;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int r = 0; r < 20; r++) copy_strided<<<grid_std, block>>>(d_in, d_out, N, STRIDE);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop, 0)); CUDA_CHECK(cudaEventSynchronize(stop));
    float naive_ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start, stop)); naive_ms /= 20.f;

    CUDA_CHECK(cudaEventRecord(start, 0));
    for (int r = 0; r < 20; r++) copy_coalesced<<<grid_co, block>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop, 0)); CUDA_CHECK(cudaEventSynchronize(stop));
    float opt_ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&opt_ms, start, stop)); opt_ms /= 20.f;

    printf("CUDA_NAIVE_MS=%.6f\n", naive_ms);
    printf("CUDA_OPTIMIZED_MS=%.6f\n", opt_ms);
    printf("N=%d STRIDE=%d\n", N, STRIDE);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_in); cudaFree(d_out); free(h);
    return 0;
}
