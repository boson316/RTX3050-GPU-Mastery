/*
 * Standalone FlashAttention benchmark: compile with nvcc and run for a given S.
 * Usage: flash_attention_cuda_standalone.exe [S]
 * Default S=128. Prints "CUDA_MS=<time_ms>" for parsing.
 *
 * Build (Windows, from this directory):
 *   nvcc -o flash_attention_cuda_standalone flash_attention_cuda_standalone.cu -O3
 * Or use the shared kernel from flash_attention_cuda.cu by including it.
 *
 * This file is self-contained and includes a copy of the tiled kernel for standalone use.
 */
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define BLOCK_M 16
#define BLOCK_N 32
#define BLOCK_D 64

__global__ void flash_attention_fwd_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ out,
    const int S,
    const int D,
    const float scale,
    const long long stride_bh
) {
    const int start_m = blockIdx.x * BLOCK_M;
    const long long base = (long long)blockIdx.y * stride_bh;

    __shared__ float s_Q[BLOCK_M][BLOCK_D];
    __shared__ float s_K[BLOCK_N][BLOCK_D];
    __shared__ float s_V[BLOCK_N][BLOCK_D];
    __shared__ float s_scores[BLOCK_M][BLOCK_N];
    __shared__ float s_mi[BLOCK_M];
    __shared__ float s_li[BLOCK_M];
    __shared__ float s_alpha[BLOCK_M];

    const int tid = threadIdx.x;
    const int tid_m = tid / BLOCK_D;
    const int tid_d = tid % BLOCK_D;

    float acc = 0.f;
    float mi = -1e30f;
    float li = 0.f;

    if (start_m + tid_m >= S || tid_d >= D) {
        if (tid_m < BLOCK_M && tid_d == 0) { s_mi[tid_m] = -1e30f; s_li[tid_m] = 0.f; }
        __syncthreads();
        if (start_m + tid_m < S && tid_d < D)
            out[base + (start_m + tid_m) * D + tid_d] = 0.f;
        return;
    }
    if (tid_d == 0) { s_mi[tid_m] = -1e30f; s_li[tid_m] = 0.f; }
    __syncthreads();

    const int num_n_blocks = (S + BLOCK_N - 1) / BLOCK_N;
    for (int n_block = 0; n_block < num_n_blocks; n_block++) {
        const int start_n = n_block * BLOCK_N;
        for (int i = tid; i < BLOCK_M * BLOCK_D; i += blockDim.x) {
            int row = i / BLOCK_D, col = i % BLOCK_D;
            int gr = start_m + row;
            s_Q[row][col] = (gr < S && col < D) ? Q[base + gr * D + col] : 0.f;
        }
        for (int i = tid; i < BLOCK_N * BLOCK_D; i += blockDim.x) {
            int row = i / BLOCK_D, col = i % BLOCK_D;
            int gr = start_n + row;
            s_K[row][col] = (gr < S && col < D) ? K[base + gr * D + col] : 0.f;
            s_V[row][col] = (gr < S && col < D) ? V[base + gr * D + col] : 0.f;
        }
        __syncthreads();

        if (start_m + tid_m < S) {
            for (int col = 0; col < BLOCK_N; col++) {
                float dot = 0.f;
                for (int d = 0; d < D && d < BLOCK_D; d++) dot += s_Q[tid_m][d] * s_K[col][d];
                int gc = start_n + col;
                s_scores[tid_m][col] = (gc < S) ? (dot * scale) : -1e30f;
            }
        }
        __syncthreads();

        if (start_m + tid_m < S && tid_d == 0) {
            float row_max = -1e30f;
            for (int col = 0; col < BLOCK_N && start_n + col < S; col++)
                row_max = fmaxf(row_max, s_scores[tid_m][col]);
            float mi_new = fmaxf(mi, row_max);
            s_alpha[tid_m] = expf(mi - mi_new);
            s_mi[tid_m] = mi_new;
            mi = mi_new;
        }
        __syncthreads();

        if (start_m + tid_m < S) {
            for (int col = 0; col < BLOCK_N; col++) {
                if (start_n + col < S) s_scores[tid_m][col] = expf(s_scores[tid_m][col] - s_mi[tid_m]);
                else s_scores[tid_m][col] = 0.f;
            }
        }
        __syncthreads();

        if (start_m + tid_m < S && tid_d == 0) {
            float l_ij = 0.f;
            for (int col = 0; col < BLOCK_N && start_n + col < S; col++) l_ij += s_scores[tid_m][col];
            li = li * s_alpha[tid_m] + l_ij;
            s_li[tid_m] = li;
        }
        __syncthreads();
        li = s_li[tid_m];

        if (start_m + tid_m < S && tid_d < D) {
            acc *= s_alpha[tid_m];
            float sum_v = 0.f;
            for (int n = 0; n < BLOCK_N && start_n + n < S; n++) sum_v += s_scores[tid_m][n] * s_V[n][tid_d];
            acc += sum_v;
        }
        __syncthreads();
    }

    if (start_m + tid_m < S && tid_d < D) {
        float inv_li = (li > 1e-10f) ? (1.f / li) : 0.f;
        out[base + (start_m + tid_m) * D + tid_d] = acc * inv_li;
    }
}

int main(int argc, char** argv) {
    int S = 128;
    if (argc >= 2) S = atoi(argv[1]);
    const int B = 2, H = 8, D = 64;
    const float scale = 1.f / sqrtf((float)D);
    const long long stride_bh = (long long)S * D;

    size_t qkv_size = (size_t)B * H * S * D * sizeof(float);
    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_out = nullptr;
    cudaMalloc(&d_Q, qkv_size);
    cudaMalloc(&d_K, qkv_size);
    cudaMalloc(&d_V, qkv_size);
    cudaMalloc(&d_out, qkv_size);
    cudaMemset(d_Q, 0, qkv_size);
    cudaMemset(d_K, 0, qkv_size);
    cudaMemset(d_V, 0, qkv_size);

    int total_blocks_m = (S + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(total_blocks_m, B * H);
    int block = BLOCK_M * BLOCK_D;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int r = 0; r < 50; r++) {
        flash_attention_fwd_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_out, S, D, scale, stride_bh);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 50.f;
    printf("CUDA_MS=%.4f\n", ms);
    printf("S=%d B=%d H=%d D=%d\n", S, B, H, D);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_out);
    return 0;
}
