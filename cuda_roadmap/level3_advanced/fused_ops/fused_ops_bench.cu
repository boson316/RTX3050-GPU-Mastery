/*
 * Level 3: Fused operations — Naive (two kernels: add then relu) vs Optimized (one kernel: add+relu)
 * C = relu(A + B). Outputs: CUDA_NAIVE_MS=... CUDA_OPTIMIZED_MS=...
 */
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(1); } \
} while(0)

const int N = 1 << 20;

__global__ void kernel_add(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}
__global__ void kernel_relu(float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && C[i] < 0.f) C[i] = 0.f;
}

__global__ void kernel_add_relu_fused(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = A[i] + B[i];
        C[i] = (s > 0.f) ? s : 0.f;
    }
}

int main() {
    size_t size = (size_t)N * sizeof(float);
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size)); CUDA_CHECK(cudaMalloc(&d_B, size)); CUDA_CHECK(cudaMalloc(&d_C, size));
    float *h_A = (float*)malloc(size), *h_B = (float*)malloc(size);
    for (int i = 0; i < N; i++) { h_A[i] = (float)(i % 10 - 5); h_B[i] = (float)(i % 7 - 3); }
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    int block = 256, grid = (N + block - 1) / block;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    kernel_add<<<grid, block>>>(d_A, d_B, d_C, N);
    kernel_relu<<<grid, block>>>(d_C, N);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop, 0)); CUDA_CHECK(cudaEventSynchronize(stop));
    float naive_ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start, stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    kernel_add_relu_fused<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop, 0)); CUDA_CHECK(cudaEventSynchronize(stop));
    float opt_ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&opt_ms, start, stop));

    printf("CUDA_NAIVE_MS=%.6f\n", naive_ms);
    printf("CUDA_OPTIMIZED_MS=%.6f\n", opt_ms);
    printf("N=%d\n", N);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); free(h_A); free(h_B);
    return 0;
}
