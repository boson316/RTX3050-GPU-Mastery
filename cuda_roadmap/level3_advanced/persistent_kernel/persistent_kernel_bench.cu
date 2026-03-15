/*
 * Level 3: Persistent kernel — Naive (many blocks, each does one chunk) vs
 * Optimized (few blocks, each loop over many chunks to reduce launch overhead / improve occupancy)
 * We do vector add: each "chunk" is one block's work. Persistent = one grid launch, blocks loop.
 */
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(1); } \
} while(0)

const int N = 1 << 20;
const int BLOCK = 256;

// Naive: one grid launch, one block per chunk of BLOCK elements
__global__ void vector_add_naive(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

// Persistent: use only 4 blocks (or num SMs), each block processes multiple chunks in a loop
__global__ void vector_add_persistent(const float* A, const float* B, float* C, int n) {
    int total_threads = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += total_threads)
        C[i] = A[i] + B[i];
}

int main() {
    size_t size = (size_t)N * sizeof(float);
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size)); CUDA_CHECK(cudaMalloc(&d_B, size)); CUDA_CHECK(cudaMalloc(&d_C, size));
    float *h_A = (float*)malloc(size), *h_B = (float*)malloc(size);
    for (int i = 0; i < N; i++) { h_A[i] = (float)i; h_B[i] = (float)(2*i); }
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    int grid_naive = (N + BLOCK - 1) / BLOCK;
    int num_blocks_persistent = 128;  // fewer blocks, each does more work

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    vector_add_naive<<<grid_naive, BLOCK>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop, 0)); CUDA_CHECK(cudaEventSynchronize(stop));
    float naive_ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start, stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    vector_add_persistent<<<num_blocks_persistent, BLOCK>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop, 0)); CUDA_CHECK(cudaEventSynchronize(stop));
    float opt_ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&opt_ms, start, stop));

    printf("CUDA_NAIVE_MS=%.6f\n", naive_ms);
    printf("CUDA_OPTIMIZED_MS=%.6f\n", opt_ms);
    printf("N=%d\n", N);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); free(h_A); free(h_B);
    return 0;
}
