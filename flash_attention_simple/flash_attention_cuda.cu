/*
 * Simplified FlashAttention CUDA: attention(Q,K,V) = softmax(Q K^T / sqrt(d)) V.
 *
 * - Does NOT store the full SxS attention matrix; uses online softmax over K tiles.
 * - Tiling: Q rows in blocks of BLOCK_M, K/V rows in blocks of BLOCK_N.
 * - Shared memory: Q_tile [BLOCK_M, BLOCK_D], K_tile [BLOCK_N, BLOCK_D], V_tile [BLOCK_N, BLOCK_D],
 *   scores [BLOCK_M, BLOCK_N], and per-row m_i, l_i for online softmax.
 *
 * Layout: Q, K, V, out are (B, H, S, D) in row-major; stride for (b,h) = S*D.
 */
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    } \
} while(0)

#define BLOCK_M 16
#define BLOCK_N 32
#define BLOCK_D 64

__global__ void flash_attention_fwd_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ out,
    const int B,
    const int H,
    const int S,
    const int D,
    const float scale,
    const long long stride_bh
) {
    const int total_blocks_m = (S + BLOCK_M - 1) / BLOCK_M;
    const int block_linear = blockIdx.x;
    const int bh = block_linear / total_blocks_m;
    const int start_m = (block_linear % total_blocks_m) * BLOCK_M;

    const long long base = (long long)bh * stride_bh;

    __shared__ float s_Q[BLOCK_M][BLOCK_D];
    __shared__ float s_K[BLOCK_N][BLOCK_D];
    __shared__ float s_V[BLOCK_N][BLOCK_D];
    __shared__ float s_scores[BLOCK_M][BLOCK_N];
    __shared__ float s_mi[BLOCK_M];
    __shared__ float s_li[BLOCK_M];

    const int tid = threadIdx.x;
    const int tid_m = tid / BLOCK_D;
    const int tid_d = tid % BLOCK_D;

    // Initialize: thread (tid_m, tid_d) holds output for row start_m+tid_m, dim tid_d
    float acc = 0.f;
    float mi = -1e30f;
    float li = 0.f;

    if (start_m + tid_m >= S || tid_d >= D) {
        if (tid_m < BLOCK_M && tid_d == 0) {
            s_mi[tid_m] = -1e30f;
            s_li[tid_m] = 0.f;
        }
        __syncthreads();
        if (start_m + tid_m < S && tid_d < D)
            out[base + (start_m + tid_m) * D + tid_d] = 0.f;
        return;
    }

    if (tid_d == 0) {
        s_mi[tid_m] = -1e30f;
        s_li[tid_m] = 0.f;
    }
    __syncthreads();

    const int num_n_blocks = (S + BLOCK_N - 1) / BLOCK_N;
    for (int n_block = 0; n_block < num_n_blocks; n_block++) {
        const int start_n = n_block * BLOCK_N;

        // Load Q tile
        for (int i = tid; i < BLOCK_M * BLOCK_D; i += blockDim.x) {
            int row = i / BLOCK_D;
            int col = i % BLOCK_D;
            int gr = start_m + row;
            s_Q[row][col] = (gr < S && col < D) ? Q[base + gr * D + col] : 0.f;
        }
        for (int i = tid; i < BLOCK_N * BLOCK_D; i += blockDim.x) {
            int row = i / BLOCK_D;
            int col = i % BLOCK_D;
            int gr = start_n + row;
            s_K[row][col] = (gr < S && col < D) ? K[base + gr * D + col] : 0.f;
            s_V[row][col] = (gr < S && col < D) ? V[base + gr * D + col] : 0.f;
        }
        __syncthreads();

        // Scores for my row: s_scores[tid_m, :] = Q[tid_m, :] @ K^T
        if (start_m + tid_m < S) {
            for (int col = 0; col < BLOCK_N; col++) {
                float dot = 0.f;
                for (int d = 0; d < D && d < BLOCK_D; d++)
                    dot += s_Q[tid_m][d] * s_K[col][d];
                int gc = start_n + col;
                s_scores[tid_m][col] = (gc < S) ? (dot * scale) : -1e30f;
            }
        }
        __syncthreads();

        // Row max for online softmax; compute alpha = exp(mi_old - mi_new) for rescaling
        __shared__ float s_alpha[BLOCK_M];
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

        // exp(s - mi) and sum
        if (start_m + tid_m < S) {
            for (int col = 0; col < BLOCK_N; col++) {
                if (start_n + col < S)
                    s_scores[tid_m][col] = expf(s_scores[tid_m][col] - s_mi[tid_m]);
                else
                    s_scores[tid_m][col] = 0.f;
            }
        }
        __syncthreads();

        float l_ij = 0.f;
        if (start_m + tid_m < S && tid_d == 0) {
            for (int col = 0; col < BLOCK_N && start_n + col < S; col++)
                l_ij += s_scores[tid_m][col];
            li = li * s_alpha[tid_m] + l_ij;
            s_li[tid_m] = li;
        }
        __syncthreads();
        li = s_li[tid_m];

        // acc = acc * alpha + P[tid_m, :] @ V[:, tid_d]
        if (start_m + tid_m < S && tid_d < D) {
            acc *= s_alpha[tid_m];
            float sum_v = 0.f;
            for (int n = 0; n < BLOCK_N && start_n + n < S; n++)
                sum_v += s_scores[tid_m][n] * s_V[n][tid_d];
            acc += sum_v;
        }
        __syncthreads();
    }

    if (start_m + tid_m < S && tid_d < D) {
        float inv_li = (li > 1e-10f) ? (1.f / li) : 0.f;
        out[base + (start_m + tid_m) * D + tid_d] = acc * inv_li;
    }
}

extern "C" {
void flash_attention_cuda_launch(
    const float* Q,
    const float* K,
    const float* V,
    float* out,
    int B,
    int H,
    int S,
    int D,
    float scale,
    cudaStream_t stream
) {
    long long stride_bh = (long long)S * D;
    int total_blocks_m = (S + BLOCK_M - 1) / BLOCK_M;
    int grid = B * H * total_blocks_m;
    int block = BLOCK_M * BLOCK_D;  // 16*64 = 1024
    flash_attention_fwd_kernel<<<grid, block, 0, stream>>>(
        Q, K, V, out, B, H, S, D, scale, stride_bh
    );
}
}
