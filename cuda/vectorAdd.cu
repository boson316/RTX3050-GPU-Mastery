/*
 * CUDA 第一個程式：Vector Add
 * C[i] = A[i] + B[i]，每個 thread 處理一個元素
 * 編譯：nvcc vectorAdd.cu -o vectorAdd
 * 執行：./vectorAdd  或  vectorAdd.exe (Windows)
 */

#include <stdio.h>
#include <cuda_runtime.h>

// GPU kernel：由每個 thread 負責一個索引 i
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main() {
    const int N = 1000;
    size_t size = N * sizeof(float);

    // Host (CPU) 陣列
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // 初始化：A[i]=i, B[i]=2*i → C[i]=3*i，故 C[0]=0
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    // GPU 記憶體配置
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // CPU → GPU 複製
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 啟動 kernel（每 block 256 threads）+ GPU 計時
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpuMs = 0.f;
    cudaEventElapsedTime(&gpuMs, start, stop);
    printf("GPU time: %.3f ms\n", gpuMs);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // GPU → CPU 複製結果
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 檢查：C[0] 應為 0（0 + 2*0）
    printf("Result C[0]: %f\n", h_C[0]);

    // 釋放記憶體
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
