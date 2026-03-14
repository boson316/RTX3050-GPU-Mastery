/*
 * CUDA Matrix Multiply：Shared Memory 優化 + 註解版
 *
 * 功能：C = A * B，比較 CPU / GPU naive / GPU shared 三種時間
 * 矩陣大小：N x N（可改 #define N，建議 1024 或 2048）
 * 優化：用 __shared__ 把 A、B 的 tile 搬進 block 內共用，減少 global memory 讀取
 *
 * 編譯：nvcc matrixMul.cu -o matrixMul.exe -allow-unsupported-compiler
 * 執行：matrixMul 或 build_matrixMul.bat
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>

#define N 1024
#define TILE 16   // 每個 block 負責 16x16 的 C；shared memory 也是 16x16，可改 32 試更快（需 GPU 支援）

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ----- CPU 版本（基準，單執行緒）-----
// C[row,col] = sum_k A[row,k]*B[k,col]，三重迴圈
void matrixMulCPU(float *A, float *B, float *C, int n) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            float sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[row * n + k] * B[k * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}

// ----- GPU Kernel（naive）：每個 thread 算 C 的一個元素 -----
// 從 global memory 讀 A[row,*]、B[*,col]，重複讀取多、較慢
__global__ void matrixMulKernelNaive(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (row < n && col < n) {
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// ----- GPU Kernel（Shared Memory 優化）：同一個 block 共用 tile，減少 global 讀取 -----
// 概念：C 的一塊 16x16 由一個 block 負責；算這塊時需要 A 的幾條 row、B 的幾條 col
// 每次把 A 的一塊 16x16、B 的一塊 16x16 載入 __shared__，算完再載下一對 tile
__global__ void matrixMulKernelShared(float *A, float *B, float *C, int n) {
    __shared__ float As[TILE][TILE];  // block 內共用，同一 block 的 thread 都可讀
    __shared__ float Bs[TILE][TILE];

    // 這個 thread 負責的 C 的 (row, col)
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0;

    // 沿 k 方向切成 numTiles 段，每段長 TILE
    int numTiles = (n + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; t++) {
        // 每個 thread 負責把 A、B 各一個元素載入 shared memory
        int colA = t * TILE + threadIdx.x;
        int rowB = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < n && colA < n) ? A[row * n + colA] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (rowB < n && col < n) ? B[rowB * n + col] : 0.f;
        __syncthreads();  // 等全 block 都載完再往下

        // 用這塊 tile 做內積：sum += A[row, k] * B[k, col]，k 在這段 [t*TILE, t*TILE+TILE)
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();  // 下一輪要覆寫 shared，先同步
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

static double getTime() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
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

    printf("Matrix size: %d x %d\n", N, N);

    srand(12345);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // ---- CPU 計算 ----
    double start = getTime();
    matrixMulCPU(h_A, h_B, h_C_cpu, N);
    printf("CPU time:     %.3f ms\n", (getTime() - start) * 1000);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // ---- GPU naive（對照用）----
    start = getTime();
    matrixMulKernelNaive<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("GPU (naive):   %.3f ms\n", (getTime() - start) * 1000);

    // ---- GPU Shared Memory：用 CUDA Event 量 kernel 時間（主要 GPU time）----
    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));
    CUDA_CHECK(cudaEventRecord(evStart, 0));
    matrixMulKernelShared<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evStop, 0));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float gpuMs = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuMs, evStart, evStop));
    printf("GPU time:      %.3f ms\n", gpuMs);
    printf("GPU shared:    %.3f ms (CUDA event)\n", gpuMs);

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost));

    float err = 0;
    for (int i = 0; i < 10; i++) {
        err += fabs(h_C_cpu[i] - h_C_gpu[i]);
    }
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
