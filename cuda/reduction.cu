/*
 * CUDA Reduction：平行求和（shared memory）
 * 修正：kernel 內用 BLOCK_SIZE，launch 用 nBlocks 覆蓋全部 N（1M）
 * 編譯：nvcc reduction.cu -o reduction.exe -allow-unsupported-compiler
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void reduceSum(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK_SIZE + tid;

    sdata[tid] = (i < n) ? input[i] : 0.f;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

int main() {
    int N = 1 << 20;  // 1M
    int nBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 覆蓋全部 1M 元素

    float *h_in = (float*)malloc(N * sizeof(float));
    float *h_out = (float*)malloc(nBlocks * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, nBlocks * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduceSum<<<nBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_out, d_out, nBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    float result = 0.f;
    for (int i = 0; i < nBlocks; i++) result += h_out[i];
    printf("GPU sum 1M ones: %.0f (correct: 1048576) in %.3f ms\n", result, ms);

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
