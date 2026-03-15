/*
 * CUDA Reduction — Parallel sum with shared memory.
 * Tree reduction within block; multi-block output then summed on host (or second kernel).
 * Build: nvcc reduction.cu -o reduction
 */
#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

__global__ void reduce_sum_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK_SIZE + tid;

    sdata[tid] = (i < n) ? input[i] : 0.f;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

int main() {
    int N = 1 << 20;  // 1M
    int nBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float* h_in = (float*)malloc((size_t)N * sizeof(float));
    float* h_out = (float*)malloc((size_t)nBlocks * sizeof(float));
    for (int i = 0; i < N; i++)
        h_in[i] = 1.0f;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)nBlocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    reduce_sum_kernel<<<nBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_out, d_out, (size_t)nBlocks * sizeof(float), cudaMemcpyDeviceToHost));
    float sum = 0.f;
    for (int i = 0; i < nBlocks; i++)
        sum += h_out[i];
    printf("reduction N=%d: sum=%.0f (expected %d) in %.4f ms\n", N, sum, N, ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}
