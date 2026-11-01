/*
 * FlashAttention forward (FP32), shared by the PyTorch extension (flash_attention_cuda.cu)
 * and the standalone benchmark (flash_attention_cuda_standalone.cu).
 *
 *   O = softmax(Q K^T * scale) V        Q, K, V, O: (B, H, S, D) contiguous, D <= 64
 *
 * Work decomposition ("one query row per thread"):
 *   grid  = (ceil(S / BLOCK_M), B * H)
 *   block = BLOCK_M threads; thread t owns query row  start_m + t.
 *   Each thread keeps q[D_PAD] and acc[D_PAD] in registers for the whole kernel, so the
 *   S x S score matrix is never materialised - only a BLOCK_N-wide slice of one row.
 *
 * Per K/V tile (BLOCK_N keys):
 *   1. All threads cooperatively copy K_tile, V_tile (BLOCK_N x D) into shared memory,
 *      coalesced: consecutive threads read consecutive addresses.
 *   2. Each thread computes its BLOCK_N scores. Every thread reads the same s_K[j][d]
 *      at the same time -> shared-memory broadcast, no bank conflicts.
 *   3. Online softmax (Milakov & Gimelshein 2018; Dao et al. 2022):
 *        m_new = max(m, max_j s_j)
 *        acc   = acc * exp(m - m_new) + sum_j exp(s_j - m_new) * V_j
 *        l     = l   * exp(m - m_new) + sum_j exp(s_j - m_new)
 *   4. After the last tile: O = acc / l.
 *
 * Why this layout (vs. the previous version of this file):
 *   - The old kernel used BLOCK_M*64 = 1024 threads where all 64 threads of a row
 *     recomputed the *same* 32 dot products (64x redundant FLOPs) and serialised the
 *     row max / row sum on one thread.
 *   - Threads past the end of the sequence returned before the cooperative tile load,
 *     so for S % 16 != 0 some K/V rows were never copied to shared memory -> wrong output.
 *   Here every thread participates in every load and __syncthreads(); out-of-range rows
 *   just skip the math and the final store.
 *
 * Causal masking skips whole K/V tiles above the diagonal (about 2x fewer FLOPs).
 */
#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <math_constants.h>
#endif

namespace flash_attn {

constexpr int BLOCK_M = 64;   // query rows per block (= threads per block)
constexpr int BLOCK_N = 16;   // keys per shared-memory tile

// Register budget (ptxas -O3, sm_86, --resource-usage):
//   D_PAD=64: ~215 regs/thread, 0 spills (q[64] + acc[64] + s[BLOCK_N] dominate)
//   D_PAD=32: ~154 regs,  D_PAD=16: ~80 regs
// With 64 threads/block, D=64 is register-limited to 4 blocks = 8 warps per SM (of 48).
// BLOCK_N=32 would push D=64 to 254 regs; keeping q in shared memory instead (padded
// to D_PAD+1 to avoid 32-way bank conflicts on per-row access) frees 64 regs but adds a
// second shared-memory load per FMA. Low occupancy is acceptable here because each
// thread has 64 independent FMA chains per key (plenty of ILP).

template <int D_PAD>
__global__ void __launch_bounds__(BLOCK_M)
flash_attention_fwd_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int S, int D, float scale, bool causal)
{
    __shared__ float s_K[BLOCK_N][D_PAD];
    __shared__ float s_V[BLOCK_N][D_PAD];

    const int tid = threadIdx.x;
    const int start_m = blockIdx.x * BLOCK_M;
    const int row = start_m + tid;
    const bool row_valid = row < S;
    const size_t bh_offset = static_cast<size_t>(blockIdx.y) * S * D;

    Q += bh_offset; K += bh_offset; V += bh_offset; O += bh_offset;

    // Fold log2(e) into the scale so the inner loop can use exp2f (a single MUFU.EX2).
    const float scale_log2 = scale * 1.4426950408889634f;

    float q[D_PAD];
    float acc[D_PAD];
#pragma unroll
    for (int d = 0; d < D_PAD; ++d) {
        q[d] = (row_valid && d < D) ? Q[static_cast<size_t>(row) * D + d] * scale_log2 : 0.f;
        acc[d] = 0.f;
    }
    float m_i = -CUDART_INF_F;  // running max (log2 domain)
    float l_i = 0.f;            // running sum of exp2(s - m_i)

    // Causal: keys > last row of this block are never needed by any thread in the block.
    const int n_end = causal ? min(S, start_m + BLOCK_M) : S;

    for (int start_n = 0; start_n < n_end; start_n += BLOCK_N) {
        // ---- 1. cooperative, coalesced tile load (zero-padded) ----
        for (int i = tid; i < BLOCK_N * D_PAD; i += BLOCK_M) {
            const int r = i / D_PAD, c = i % D_PAD;
            const int key = start_n + r;
            const bool ok = key < S && c < D;
            const size_t g = static_cast<size_t>(key) * D + c;
            s_K[r][c] = ok ? K[g] : 0.f;
            s_V[r][c] = ok ? V[g] : 0.f;
        }
        __syncthreads();

        if (row_valid) {
            // ---- 2. scores for this thread's row ----
            float s[BLOCK_N];
            float tile_max = -CUDART_INF_F;
#pragma unroll
            for (int j = 0; j < BLOCK_N; ++j) {
                const int key = start_n + j;
                float dot = 0.f;
#pragma unroll
                for (int d = 0; d < D_PAD; ++d) dot = fmaf(q[d], s_K[j][d], dot);
                const bool visible = key < S && (!causal || key <= row);
                s[j] = visible ? dot : -CUDART_INF_F;
                tile_max = fmaxf(tile_max, s[j]);
            }

            // ---- 3. online softmax update ----
            const float m_new = fmaxf(m_i, tile_max);
            if (m_new != -CUDART_INF_F) {           // tile fully masked for this row -> nothing to add
                const float alpha = exp2f(m_i - m_new);  // exp2(-inf) = 0 on the first tile
                l_i *= alpha;
#pragma unroll
                for (int d = 0; d < D_PAD; ++d) acc[d] *= alpha;
#pragma unroll
                for (int j = 0; j < BLOCK_N; ++j) {
                    const float p = exp2f(s[j] - m_new);  // masked keys: exp2(-inf) = 0
                    l_i += p;
#pragma unroll
                    for (int d = 0; d < D_PAD; ++d) acc[d] = fmaf(p, s_V[j][d], acc[d]);
                }
                m_i = m_new;
            }
        }
        __syncthreads();  // tile is overwritten in the next iteration
    }

    // ---- 4. normalise and store ----
    if (row_valid) {
        const float inv_l = l_i > 0.f ? 1.f / l_i : 0.f;
#pragma unroll
        for (int d = 0; d < D_PAD; ++d)
            if (d < D) O[static_cast<size_t>(row) * D + d] = acc[d] * inv_l;
    }
}

#ifdef __CUDACC__
// Host-side launcher. Returns the launch error (cudaSuccess on success) instead of
// silently ignoring it - a failed launch must never be timed as a "fast" kernel.
inline cudaError_t launch_flash_attention_fwd(
    const float* Q, const float* K, const float* V, float* O,
    int B, int H, int S, int D, float scale, bool causal, cudaStream_t stream)
{
    if (D <= 0 || D > 64 || S <= 0 || B * H <= 0) return cudaErrorInvalidValue;
    const dim3 grid((S + BLOCK_M - 1) / BLOCK_M, B * H);
    const dim3 block(BLOCK_M);
    if (D <= 16)
        flash_attention_fwd_kernel<16><<<grid, block, 0, stream>>>(Q, K, V, O, S, D, scale, causal);
    else if (D <= 32)
        flash_attention_fwd_kernel<32><<<grid, block, 0, stream>>>(Q, K, V, O, S, D, scale, causal);
    else
        flash_attention_fwd_kernel<64><<<grid, block, 0, stream>>>(Q, K, V, O, S, D, scale, causal);
    return cudaGetLastError();
}
#endif  // __CUDACC__

}  // namespace flash_attn
