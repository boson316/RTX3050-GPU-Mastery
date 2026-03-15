/*
 * Standalone CUDA 3x3 Conv2D — No padding, single input channel, multiple output channels.
 * For PyTorch-integrated version see pytorch_extensions/custom_attention (or extension/).
 * Build: nvcc -o conv2d conv2d.cu
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE 16
#define K 3
#define SHARD_H (TILE + K - 1)
#define SHARD_W (TILE + K - 1)
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

__global__ void conv2d_kernel(const float* __restrict__ input,
                             const float* __restrict__ weight,
                             const float* __restrict__ bias,
                             float* __restrict__ output,
                             int batch, int in_h, int in_w,
                             int out_ch, int out_h, int out_w) {
    __shared__ float tile[SHARD_H][SHARD_W];
    __shared__ float wbuf[9];

    int tx = threadIdx.x, ty = threadIdx.y;
    int oh = blockIdx.y * TILE;
    int ow = blockIdx.x * TILE;
    int b = blockIdx.z % batch;
    int oc = blockIdx.z / batch;
    int ih_base = oh, iw_base = ow;

    for (int idx = ty * blockDim.x + tx; idx < SHARD_H * SHARD_W; idx += blockDim.x * blockDim.y) {
        int i = idx / SHARD_W, j = idx % SHARD_W;
        int ih = ih_base + i, iw = iw_base + j;
        tile[i][j] = (ih < in_h && iw < in_w)
            ? input[(b * in_h + ih) * in_w + iw] : 0.f;
    }
    if (ty * blockDim.x + tx < 9)
        wbuf[ty * blockDim.x + tx] = weight[oc * 9 + (ty * blockDim.x + tx)];
    __syncthreads();

    int oy = oh + ty, ox = ow + tx;
    if (oy < out_h && ox < out_w) {
        float sum = bias[oc];
#pragma unroll
        for (int kh = 0; kh < K; kh++)
#pragma unroll
            for (int kw = 0; kw < K; kw++)
                sum += tile[ty + kh][tx + kw] * wbuf[kh * K + kw];
        output[(b * out_ch + oc) * (out_h * out_w) + oy * out_w + ox] = sum;
    }
}

int main() {
    int batch = 4, in_h = 28, in_w = 28, out_ch = 32;
    int out_h = in_h - 2, out_w = in_w - 2;

    size_t in_size = (size_t)batch * in_h * in_w * sizeof(float);
    size_t w_size = (size_t)out_ch * 9 * sizeof(float);
    size_t b_size = (size_t)out_ch * sizeof(float);
    size_t out_size = (size_t)batch * out_ch * out_h * out_w * sizeof(float);

    float *h_in = (float*)malloc(in_size);
    float *h_w = (float*)malloc(w_size);
    float *h_b = (float*)malloc(b_size);
    float *h_out = (float*)malloc(out_size);
    for (size_t i = 0; i < in_size / sizeof(float); i++) h_in[i] = 0.1f;
    for (size_t i = 0; i < w_size / sizeof(float); i++) h_w[i] = 0.01f;
    for (size_t i = 0; i < b_size / sizeof(float); i++) h_b[i] = 0.f;

    float *d_in, *d_w, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, in_size));
    CUDA_CHECK(cudaMalloc(&d_w, w_size));
    CUDA_CHECK(cudaMalloc(&d_b, b_size));
    CUDA_CHECK(cudaMalloc(&d_out, out_size));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, in_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w, w_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, b_size, cudaMemcpyHostToDevice));

    dim3 threads(TILE, TILE);
    dim3 blocks((out_w + TILE - 1) / TILE, (out_h + TILE - 1) / TILE, batch * out_ch);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    conv2d_kernel<<<blocks, threads>>>(d_in, d_w, d_b, d_out,
        batch, in_h, in_w, out_ch, out_h, out_w);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("conv2d batch=%d %dx%d -> %d ch: %.4f ms\n", batch, in_h, in_w, out_ch, ms);

    CUDA_CHECK(cudaMemcpy(h_out, d_out, out_size, cudaMemcpyDeviceToHost));
    printf("Sample output[0,0,0,0]=%.6f\n", h_out[0]);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_out);
    free(h_in);
    free(h_w);
    free(h_b);
    free(h_out);
    return 0;
}
