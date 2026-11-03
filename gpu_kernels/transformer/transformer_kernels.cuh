/*
 * Transformer building-block kernels (FP16 storage, FP32 math) for sm_86.
 *
 * Device code only; launchers are in transformer_kernels.cu and the GEMMs (QKV / MLP)
 * go through cuBLAS in transformer_cuda.cpp. Kept header-only so the same code can be
 * checked on the CPU emulator in tests/cuda_emulation/.
 *
 * Kernels
 *   softmax_warp      : 1 warp per row, row cached in registers, 128-bit loads (n_cols % 8 == 0, <= 1024)
 *   layernorm_warp    : same layout, exact two-pass statistics from registers
 *   softmax_rows      : fallback, 1 block per row, ONLINE softmax -> 2 passes over the row instead of 3
 *   layernorm_rows    : fallback, 1 block per row, Welford mean/variance -> 2 passes instead of 3
 *   gelu              : elementwise tanh-GELU, grid-stride loop
 *   bias_gelu_rows    : fused epilogue y = GELU(y + b)  (after the first MLP GEMM)
 *   bias_rows         : fused epilogue y = y + b
 *
 * Bugs fixed vs. the previous version (see docs/ENGINEERING_LOG.md):
 *   - blockReduce{Max,Sum} called __shfl_xor_sync(0xffffffff, ...) from inside
 *     `if (threadIdx.x < 8)`. The full mask promises all 32 lanes participate; with only
 *     8 active lanes the behaviour is undefined and on Volta+ (independent thread
 *     scheduling) it can hang - which is why the benchmark needed a "may hang" env flag.
 *     The reductions below run the second stage on a *full* warp.
 *   - The smem staging array was hard-coded to 8 warps (256 threads); now sized for 1024.
 */
#pragma once

#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

namespace tk {

constexpr unsigned FULL_MASK = 0xffffffffu;
constexpr int MAX_WARPS = 32;  // 1024 threads / 32

// ---------------------------------------------------------------------------
// Warp / block reductions. Every lane of every warp must call these.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(FULL_MASK, v, off);
    return v;
}

// (max, sum) pair for online softmax: merge two partial results.
__device__ __forceinline__ void softmax_merge(float& m, float& s, float m_o, float s_o) {
    const float m_new = fmaxf(m, m_o);
    // guard -inf - -inf = NaN when both partials are empty
    s = (m == m_new ? s : s * __expf(m - m_new)) + (m_o == m_new ? s_o : s_o * __expf(m_o - m_new));
    m = m_new;
}

__device__ __forceinline__ void warp_softmax_reduce(float& m, float& s) {
    for (int off = 16; off > 0; off >>= 1) {
        const float m_o = __shfl_xor_sync(FULL_MASK, m, off);
        const float s_o = __shfl_xor_sync(FULL_MASK, s, off);
        softmax_merge(m, s, m_o, s_o);
    }
}

// Welford / Chan parallel merge of (count, mean, M2).
__device__ __forceinline__ void welford_merge(float& n, float& mean, float& m2, float n_o, float mean_o, float m2_o) {
    const float n_new = n + n_o;
    if (n_new == 0.f) return;
    const float delta = mean_o - mean;
    const float w = n_o / n_new;
    mean += delta * w;
    m2 += m2_o + delta * delta * n * w;
    n = n_new;
}

__device__ __forceinline__ void warp_welford_reduce(float& n, float& mean, float& m2) {
    for (int off = 16; off > 0; off >>= 1) {
        const float n_o = __shfl_xor_sync(FULL_MASK, n, off);
        const float mean_o = __shfl_xor_sync(FULL_MASK, mean, off);
        const float m2_o = __shfl_xor_sync(FULL_MASK, m2, off);
        welford_merge(n, mean, m2, n_o, mean_o, m2_o);
    }
}

__device__ __forceinline__ float gelu_tanh(float x) {
    const float k0 = 0.7978845608028654f;  // sqrt(2/pi)
    const float k1 = 0.044715f;
    return 0.5f * x * (1.f + tanhf(k0 * (x + k1 * x * x * x)));
}

// ---------------------------------------------------------------------------
// Softmax over the last dim. blockDim.x must be a multiple of 32.
//   pass 1: each thread folds its strided elements into a running (max, sum)
//   reduce : warp shuffle, then warp 0 merges the per-warp partials
//   pass 2: y = exp(x - max) / sum
// ---------------------------------------------------------------------------
__global__ void softmax_rows_kernel(const __half* __restrict__ x, __half* __restrict__ y, int n_cols) {
    __shared__ float s_m[MAX_WARPS], s_s[MAX_WARPS];
    __shared__ float s_row_max, s_row_inv_sum;

    const size_t row_off = static_cast<size_t>(blockIdx.x) * n_cols;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n_warps = blockDim.x >> 5;

    float m = -INFINITY, s = 0.f;
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
        const float v = __half2float(x[row_off + i]);
        if (v > m) { s = s * __expf(m - v) + 1.f; m = v; }
        else       { s += __expf(v - m); }
    }
    warp_softmax_reduce(m, s);
    if (lane == 0) { s_m[warp] = m; s_s[warp] = s; }
    __syncthreads();
    if (warp == 0) {  // whole warp 0 participates; lanes >= n_warps contribute the identity
        m = lane < n_warps ? s_m[lane] : -INFINITY;
        s = lane < n_warps ? s_s[lane] : 0.f;
        warp_softmax_reduce(m, s);
        if (lane == 0) { s_row_max = m; s_row_inv_sum = 1.f / s; }
    }
    __syncthreads();

    const float row_max = s_row_max, inv_sum = s_row_inv_sum;
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x)
        y[row_off + i] = __float2half(__expf(__half2float(x[row_off + i]) - row_max) * inv_sum);
}

// ---------------------------------------------------------------------------
// LayerNorm over the last dim with Welford statistics (one read for mean+var).
// ---------------------------------------------------------------------------
__global__ void layernorm_rows_kernel(const __half* __restrict__ x, const __half* __restrict__ gamma,
                                      const __half* __restrict__ beta, __half* __restrict__ y,
                                      int n_cols, float eps) {
    __shared__ float s_n[MAX_WARPS], s_mean[MAX_WARPS], s_m2[MAX_WARPS];
    __shared__ float s_row_mean, s_row_rstd;

    const size_t row_off = static_cast<size_t>(blockIdx.x) * n_cols;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n_warps = blockDim.x >> 5;

    float n = 0.f, mean = 0.f, m2 = 0.f;
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
        const float v = __half2float(x[row_off + i]);
        n += 1.f;
        const float d = v - mean;
        mean += d / n;
        m2 += d * (v - mean);
    }
    warp_welford_reduce(n, mean, m2);
    if (lane == 0) { s_n[warp] = n; s_mean[warp] = mean; s_m2[warp] = m2; }
    __syncthreads();
    if (warp == 0) {
        n = lane < n_warps ? s_n[lane] : 0.f;
        mean = lane < n_warps ? s_mean[lane] : 0.f;
        m2 = lane < n_warps ? s_m2[lane] : 0.f;
        warp_welford_reduce(n, mean, m2);
        if (lane == 0) { s_row_mean = mean; s_row_rstd = rsqrtf(m2 / n + eps); }
    }
    __syncthreads();

    const float mu = s_row_mean, rstd = s_row_rstd;
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
        const float g = gamma ? __half2float(gamma[i]) : 1.f;
        const float b = beta ? __half2float(beta[i]) : 0.f;
        y[row_off + i] = __float2half((__half2float(x[row_off + i]) - mu) * rstd * g + b);
    }
}

// ---------------------------------------------------------------------------
// Warp-per-row softmax / LayerNorm for rows of <= 1024 elements with n_cols % 8 == 0
// (every hidden size used in practice: 768, 1024, ...).
//
// Why (v2 measurements, 2048 x 768 FP16): the block-per-row kernels above reached 67 GB/s
// (softmax) and 57 GB/s (LayerNorm) vs ~115 GB/s for PyTorch on a 132 GB/s part. With one
// 2-byte load per thread and 768-thread blocks, each SM has only ~2 rows in flight, so there
// are too few outstanding memory requests to cover DRAM latency, and the row is read twice.
//
// Here one warp owns one row:
//   - 128-bit loads (8 halves per lane per chunk); chunk c of lane l covers columns
//     (c*32 + l)*8 .. +7, so each chunk is one fully coalesced 512-byte warp access;
//   - the whole row stays in registers (CHUNKS*8 floats per lane), so DRAM is read once;
//   - reductions are pure warp shuffles: no shared memory, no __syncthreads;
//   - 4 warps (4 rows) per 128-thread block -> many independent rows in flight per SM.
// A warp whose row is out of range returns as a whole, which is safe because the kernel
// has no block-level barrier.
// ---------------------------------------------------------------------------
constexpr int WARP_ROWS_PER_BLOCK = 4;
constexpr int VEC = 8;  // halves per 128-bit access

__device__ __forceinline__ void load8(const __half* p, float v[VEC]) {
#ifdef __CUDACC__
    const int4 raw = *reinterpret_cast<const int4*>(p);
    const __half2* h = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __half22float2(h[i]);
        v[2 * i] = f.x;
        v[2 * i + 1] = f.y;
    }
#else
    for (int i = 0; i < VEC; ++i) v[i] = __half2float(p[i]);
#endif
}

__device__ __forceinline__ void store8(__half* p, const float v[VEC]) {
#ifdef __CUDACC__
    int4 raw;
    __half2* h = reinterpret_cast<__half2*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = __floats2half2_rn(v[2 * i], v[2 * i + 1]);
    *reinterpret_cast<int4*>(p) = raw;
#else
    for (int i = 0; i < VEC; ++i) p[i] = __float2half(v[i]);
#endif
}

__device__ __forceinline__ float warp_max(float v) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, off));
    return v;
}

template <int CHUNKS>
__global__ void __launch_bounds__(WARP_ROWS_PER_BLOCK * 32)
softmax_warp_kernel(const __half* __restrict__ x, __half* __restrict__ y, int n_rows, int n_cols) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * WARP_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n_rows) return;
    const size_t off = static_cast<size_t>(row) * n_cols;

    float v[CHUNKS][VEC];
    float m = -INFINITY;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int col = (c * 32 + lane) * VEC;
        if (col < n_cols) {
            load8(x + off + col, v[c]);
#pragma unroll
            for (int e = 0; e < VEC; ++e) m = fmaxf(m, v[c][e]);
        } else {
#pragma unroll
            for (int e = 0; e < VEC; ++e) v[c][e] = -INFINITY;  // exp -> 0, never stored
        }
    }
    m = warp_max(m);
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c)
#pragma unroll
        for (int e = 0; e < VEC; ++e) {
            v[c][e] = __expf(v[c][e] - m);
            s += v[c][e];
        }
    const float inv = 1.f / warp_sum(s);
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int col = (c * 32 + lane) * VEC;
        if (col < n_cols) {
#pragma unroll
            for (int e = 0; e < VEC; ++e) v[c][e] *= inv;
            store8(y + off + col, v[c]);
        }
    }
}

template <int CHUNKS>
__global__ void __launch_bounds__(WARP_ROWS_PER_BLOCK * 32)
layernorm_warp_kernel(const __half* __restrict__ x, const __half* __restrict__ gamma,
                      const __half* __restrict__ beta, __half* __restrict__ y, int n_rows, int n_cols, float eps) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * WARP_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= n_rows) return;
    const size_t off = static_cast<size_t>(row) * n_cols;

    float v[CHUNKS][VEC];
    float sum = 0.f;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int col = (c * 32 + lane) * VEC;
        if (col < n_cols) {
            load8(x + off + col, v[c]);
#pragma unroll
            for (int e = 0; e < VEC; ++e) sum += v[c][e];
        }
    }
    const float mean = warp_sum(sum) / n_cols;
    // Exact two-pass variance from registers: no extra DRAM traffic, no cancellation.
    float sq = 0.f;
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int col = (c * 32 + lane) * VEC;
        if (col < n_cols) {
#pragma unroll
            for (int e = 0; e < VEC; ++e) {
                const float d = v[c][e] - mean;
                sq += d * d;
            }
        }
    }
    const float rstd = rsqrtf(warp_sum(sq) / n_cols + eps);
#pragma unroll
    for (int c = 0; c < CHUNKS; ++c) {
        const int col = (c * 32 + lane) * VEC;
        if (col < n_cols) {
            float g[VEC], b[VEC];
            if (gamma) load8(gamma + col, g); else for (int e = 0; e < VEC; ++e) g[e] = 1.f;
            if (beta) load8(beta + col, b); else for (int e = 0; e < VEC; ++e) b[e] = 0.f;
#pragma unroll
            for (int e = 0; e < VEC; ++e) v[c][e] = (v[c][e] - mean) * rstd * g[e] + b[e];
            store8(y + off + col, v[c]);
        }
    }
}

// ---------------------------------------------------------------------------
// Elementwise kernels (grid-stride loops: any grid size covers any n).
// ---------------------------------------------------------------------------
__global__ void gelu_kernel(const __half* __restrict__ x, __half* __restrict__ y, size_t n) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
        y[i] = __float2half(gelu_tanh(__half2float(x[i])));
}

// In-place epilogues on a row-major (rows x n_cols) GEMM output.
template <bool APPLY_GELU>
__global__ void bias_act_kernel(__half* __restrict__ y, const __half* __restrict__ bias, size_t total, int n_cols) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < total;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        float v = __half2float(y[i]);
        if (bias) v += __half2float(bias[i % n_cols]);
        if (APPLY_GELU) v = gelu_tanh(v);
        y[i] = __float2half(v);
    }
}

}  // namespace tk
