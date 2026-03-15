/*
 * CUDA Matrix Multiply — Tiled shared-memory optimization.
 * C = A * B (NxN). Compares naive vs tiled kernel; reports GPU time via CUDA events.
 * Build: nvcc matrix_mul.cu -o matrix_mul
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>

static double get_time_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

#define N 1024
#define TILE 16

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

void matrix_mul_cpu(const float* A, const float* B, float* C, int n) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            float sum = 0;
            for (int k = 0; k < n; k++)
                sum += A[row * n + k] * B[k * n + col];
            C[row * n + col] = sum;
        }
    }
}

__global__ void matrix_mul_naive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (row < n && col < n) {
        for (int k = 0; k < n; k++)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

__global__ void matrix_mul_tiled(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

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

int main() {
    size_t size = (size_t)N * N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C_cpu = (float*)malloc(size);
    float *h_C_gpu = (float*)malloc(size);
    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    srand(12345);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    double t0 = get_time_ms();
    matrix_mul_cpu(h_A, h_B, h_C_cpu, N);
    float cpu_ms = (float)(get_time_ms() - t0);
    printf("CPU time: %.3f ms\n", cpu_ms);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));
    CUDA_CHECK(cudaEventRecord(evStart, 0));
    matrix_mul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evStop, 0));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float gpu_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, evStart, evStop));
    printf("GPU (tiled) time: %.3f ms\n", gpu_ms);
    if (cpu_ms > 0)
        printf("Speedup vs CPU: %.1fx\n", cpu_ms / gpu_ms);

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));
    float err = 0;
    for (int i = 0; i < 10; i++)
        err += fabsf(h_C_cpu[i] - h_C_gpu[i]);
    printf("Error (first 10): %f\n", err);

    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    return 0;
}
