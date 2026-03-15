/*
 * Level 1: Naive Matrix Multiply — Naive only (Level 2 has tiled optimized)
 * C = A*B, each thread one output, all reads from global memory.
 * Outputs: CUDA_NAIVE_MS=... (OPTIMIZED same as naive for level1; level2 has tiled)
 */
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(1); } \
} while(0)

static double get_time_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

const int N = 1024;

void matmul_cpu(const float* A, const float* B, float* C, int n) {
    for (int row = 0; row < n; row++)
        for (int col = 0; col < n; col++) {
            float sum = 0;
            for (int k = 0; k < n; k++) sum += A[row*n+k] * B[k*n+col];
            C[row*n+col] = sum;
        }
}

__global__ void matmul_naive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (row < n && col < n) {
        for (int k = 0; k < n; k++) sum += A[row*n+k] * B[k*n+col];
        C[row*n+col] = sum;
    }
}

int main() {
    size_t size = (size_t)N * N * sizeof(float);
    float *h_A = (float*)malloc(size), *h_B = (float*)malloc(size);
    float *h_C_cpu = (float*)malloc(size), *h_C_gpu = (float*)malloc(size);
    srand(12345);
    for (int i = 0; i < N*N; i++) { h_A[i] = rand()/(float)RAND_MAX; h_B[i] = rand()/(float)RAND_MAX; }

    double t0 = get_time_ms();
    matmul_cpu(h_A, h_B, h_C_cpu, N);
    float cpu_ms = (float)(get_time_ms() - t0);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size)); CUDA_CHECK(cudaMalloc(&d_B, size)); CUDA_CHECK(cudaMalloc(&d_C, size));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    dim3 block(16, 16), grid((N+15)/16, (N+15)/16);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float gpu_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));

    printf("CPU_MS=%.6f\n", cpu_ms);
    printf("CUDA_NAIVE_MS=%.6f\n", gpu_ms);
    printf("CUDA_OPTIMIZED_MS=%.6f\n", gpu_ms);  // Level1: no optimized; Level2 has tiled
    printf("N=%d\n", N);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    return 0;
}
