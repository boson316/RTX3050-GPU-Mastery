/*
 * 3x3 valid convolution, C_in = 1, C_out = 32 (the first layer of the MNIST CNN in this repo).
 *
 * Block = 16x16 threads = one 16x16 output tile, ALL 32 output channels.
 *   - The 18x18 input halo tile is loaded into shared memory once and reused by
 *     32 channels x 9 taps = 288 MACs per input load.
 *   - The 32x9 weights + 32 biases are staged in shared memory once per block; every thread
 *     reads the same weight at the same time -> broadcast, no bank conflicts.
 *   - Output writes for a given channel are coalesced along the row (tx).
 * Storage type T is float or __half; accumulation is always FP32.
 *
 * Fixed vs. the previous version (see docs/ENGINEERING_LOG.md):
 *   - The "float32" path converted inputs to FP16 and accumulated with __hadd2 in FP16, so the
 *     "1.5x faster than cuDNN" comparison pitted half-precision math against cuDNN's FP32.
 *   - Output was allocated with torch::zeros (an extra full-size memset kernel) -> torch::empty.
 *   - No launch-error check, and launches went to the legacy default stream instead of
 *     PyTorch's current stream.
 */
#pragma once

#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

namespace conv3x3 {

constexpr int TILE = 16;
constexpr int HALO = TILE + 2;
constexpr int C_OUT = 32;

__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ void store(float* p, float v) { *p = v; }
#ifdef __CUDACC__
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ void store(__half* p, float v) { *p = __float2half_rn(v); }
#endif

template <typename T>
__global__ void __launch_bounds__(TILE * TILE)
conv3x3_c32_kernel(const T* __restrict__ input,   // (B, 1, H, W)
                   const T* __restrict__ weight,  // (32, 1, 3, 3)
                   const T* __restrict__ bias,    // (32)
                   T* __restrict__ output,        // (B, 32, H-2, W-2)
                   int in_h, int in_w) {
    __shared__ float s_in[HALO][HALO];
    __shared__ float s_w[C_OUT][9];
    __shared__ float s_b[C_OUT];

    const int out_h = in_h - 2, out_w = in_w - 2;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TILE + tx;
    const int oh = blockIdx.y * TILE, ow = blockIdx.x * TILE;
    const size_t b = blockIdx.z;
    const T* in_b = input + b * in_h * in_w;

    for (int i = tid; i < HALO * HALO; i += TILE * TILE) {
        const int r = i / HALO, c = i % HALO;
        const int ih = oh + r, iw = ow + c;
        s_in[r][c] = (ih < in_h && iw < in_w) ? to_float(in_b[(size_t)ih * in_w + iw]) : 0.f;
    }
    for (int i = tid; i < C_OUT * 9; i += TILE * TILE) s_w[i / 9][i % 9] = to_float(weight[i]);
    if (tid < C_OUT) s_b[tid] = bias ? to_float(bias[tid]) : 0.f;
    __syncthreads();

    const int oy = oh + ty, ox = ow + tx;
    if (oy >= out_h || ox >= out_w) return;  // after the only barrier: safe

    float patch[9];
#pragma unroll
    for (int k = 0; k < 9; ++k) patch[k] = s_in[ty + k / 3][tx + k % 3];

    T* out = output + b * C_OUT * out_h * out_w + (size_t)oy * out_w + ox;
#pragma unroll 4
    for (int c = 0; c < C_OUT; ++c) {
        float acc = s_b[c];
#pragma unroll
        for (int k = 0; k < 9; ++k) acc = fmaf(patch[k], s_w[c][k], acc);
        store(out + (size_t)c * out_h * out_w, acc);
    }
}

}  // namespace conv3x3
