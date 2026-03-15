/*
 * CUDA Scaled Dot-Product Attention: softmax(Q K^T / sqrt(d)) V
 * Simplified reference implementation — one block per (batch, head, query row).
 * For production use Flash Attention (triton_kernels/flash_attention).
 * Build: nvcc -o attention attention.cu
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define BLOCK_D 32

__global__ void attention_fwd_kernel(const float* __restrict__ Q,
                                      const float* __restrict__ K,
                                      const float* __restrict__ V,
                                      float* __restrict__ out,
                                      int B, int H, int S, int D,
                                      float scale) {
    int b = blockIdx.x / (H * S);
    int h = (blockIdx.x / S) % H;
    int sq = blockIdx.x % S;

    int q_row = ((b * H + h) * S + sq) * D;
    int d = threadIdx.x;
    if (d >= D) return;

    float max_s = -1e30f;
    for (int k = 0; k < S; k++) {
        float dot = 0.f;
        int k_row = ((b * H + h) * S + k) * D;
        for (int dd = 0; dd < D; dd++)
            dot += Q[q_row + dd] * K[k_row + dd];
        float s = dot * scale;
        max_s = fmaxf(max_s, s);
    }
    __shared__ float smax;
    if (threadIdx.x == 0)
        smax = max_s;
    __syncthreads();
    max_s = smax;

    float sum_exp = 0.f;
    for (int k = 0; k < S; k++) {
        float dot = 0.f;
        int k_row = ((b * H + h) * S + k) * D;
        for (int dd = 0; dd < D; dd++)
            dot += Q[q_row + dd] * K[k_row + dd];
        sum_exp += expf(dot * scale - max_s);
    }
    __shared__ float ssum;
    if (threadIdx.x == 0)
        ssum = sum_exp;
    __syncthreads();
    float inv_sum = 1.f / (ssum + 1e-6f);

    float acc = 0.f;
    for (int k = 0; k < S; k++) {
        float dot = 0.f;
        int k_row = ((b * H + h) * S + k) * D;
        for (int dd = 0; dd < D; dd++)
            dot += Q[q_row + dd] * K[k_row + dd];
        float p = expf(dot * scale - max_s) * inv_sum;
        acc += p * V[k_row + d];
    }
    out[((b * H + h) * S + sq) * D + d] = acc;
}

int main() {
    int B = 2, H = 4, S = 64, D = 32;
    float scale = 1.f / sqrtf((float)D);
    size_t qkv_size = (size_t)B * H * S * D * sizeof(float);

    float *h_Q = (float*)malloc(qkv_size);
    float *h_K = (float*)malloc(qkv_size);
    float *h_V = (float*)malloc(qkv_size);
    float *h_out = (float*)malloc(qkv_size);
    for (size_t i = 0; i < (size_t)B * H * S * D; i++) {
        h_Q[i] = 0.01f;
        h_K[i] = 0.01f;
        h_V[i] = 0.01f;
    }

    float *d_Q, *d_K, *d_V, *d_out;
    CUDA_CHECK(cudaMalloc(&d_Q, qkv_size));
    CUDA_CHECK(cudaMalloc(&d_K, qkv_size));
    CUDA_CHECK(cudaMalloc(&d_V, qkv_size));
    CUDA_CHECK(cudaMalloc(&d_out, qkv_size));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, qkv_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, qkv_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, qkv_size, cudaMemcpyHostToDevice));

    int num_blocks = B * H * S;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    attention_fwd_kernel<<<num_blocks, BLOCK_D>>>(
        d_Q, d_K, d_V, d_out, B, H, S, D, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("attention B=%d H=%d S=%d D=%d: %.4f ms\n", B, H, S, D, ms);

    CUDA_CHECK(cudaMemcpy(h_out, d_out, qkv_size, cudaMemcpyDeviceToHost));
    printf("Sample out[0,0,0,0]=%.6f\n", h_out[0]);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_out);
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_out);
    return 0;
}
