/*
 * Level 1: Vector Add — Naive vs Optimized (vectorized float4)
 * Outputs: CUDA_NAIVE_MS=... CUDA_OPTIMIZED_MS=... for run_benchmarks.py
 */
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(1); } \
} while(0)

const int N = 1 << 20;  // 1M

// ---- Naive: one float per thread ----
__global__ void vector_add_naive(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

// ---- Optimized: float4 vectorized load/store (4x fewer threads, coalesced) ----
__global__ void vector_add_optimized(const float* A, const float* B, float* C, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        float4 a = *reinterpret_cast<const float4*>(A + i);
        float4 b = *reinterpret_cast<const float4*>(B + i);
        float4 c;
        c.x = a.x + b.x; c.y = a.y + b.y; c.z = a.z + b.z; c.w = a.w + b.w;
        *reinterpret_cast<float4*>(C + i) = c;
    } else {
        for (int j = 0; j < 4 && i + j < n; j++)
            C[i + j] = A[i + j] + B[i + j];
    }
}

static float run_kernel_naive(float* d_A, float* d_B, float* d_C) {
    int block = 256;
    int grid = (N + block - 1) / block;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    vector_add_naive<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

static float run_kernel_optimized(float* d_A, float* d_B, float* d_C) {
    int block = 256;
    int grid = (N / 4 + block - 1) / block;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    vector_add_optimized<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main() {
    size_t size = (size_t)N * sizeof(float);
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    for (int i = 0; i < N; i++) { h_A[i] = (float)i; h_B[i] = (float)(2*i); }
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    float naive_ms = run_kernel_naive(d_A, d_B, d_C);
    float opt_ms  = run_kernel_optimized(d_A, d_B, d_C);

    printf("CUDA_NAIVE_MS=%.6f\n", naive_ms);
    printf("CUDA_OPTIMIZED_MS=%.6f\n", opt_ms);
    printf("N=%d\n", N);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B);
    return 0;
}
