/*
 * Launchers for the transformer kernels (device code in transformer_kernels.cuh).
 * Every launcher returns the launch status so the PyTorch binding can raise on failure
 * instead of timing a kernel that never ran.
 */
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "transformer_kernels.cuh"

namespace {
// Threads per row for the row-wise kernels: enough to cover the row once, capped at 1024,
// rounded to a whole number of warps (the block reductions need full warps).
int row_block_size(int n_cols) {
    int t = std::min(1024, std::max(32, n_cols));
    return (t + 31) / 32 * 32;
}
// Grid for grid-stride elementwise kernels: a few waves over the SMs is enough.
unsigned elementwise_grid(size_t n, int block) {
    const size_t blocks = (n + block - 1) / block;
    return static_cast<unsigned>(std::min<size_t>(blocks, 16 * 1024));
}
bool aligned16(const void* p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; }

// Warp-per-row path: n_cols % 8 == 0 (whole 128-bit chunks), n_cols <= 1024 (row fits in
// 4 chunks x 8 halves x 32 lanes of registers) and 16-byte aligned rows.
bool use_warp_path(int n_cols, std::initializer_list<const void*> ptrs) {
    if (n_cols <= 0 || n_cols > 4 * 32 * tk::VEC || n_cols % tk::VEC != 0) return false;
    for (const void* p : ptrs)
        if (p && !aligned16(p)) return false;
    return true;
}

unsigned warp_grid(int n_rows) {
    return static_cast<unsigned>((n_rows + tk::WARP_ROWS_PER_BLOCK - 1) / tk::WARP_ROWS_PER_BLOCK);
}
}  // namespace

extern "C" {

cudaError_t softmax_fp16_launch(const __half* x, __half* y, int n_cols, int n_rows, cudaStream_t stream) {
    if (n_rows == 0) return cudaSuccess;
    if (use_warp_path(n_cols, {x, y})) {
        const unsigned grid = warp_grid(n_rows), block = tk::WARP_ROWS_PER_BLOCK * 32;
        switch ((n_cols + 255) / 256) {  // 8-element chunks per lane
            case 1: tk::softmax_warp_kernel<1><<<grid, block, 0, stream>>>(x, y, n_rows, n_cols); break;
            case 2: tk::softmax_warp_kernel<2><<<grid, block, 0, stream>>>(x, y, n_rows, n_cols); break;
            case 3: tk::softmax_warp_kernel<3><<<grid, block, 0, stream>>>(x, y, n_rows, n_cols); break;
            default: tk::softmax_warp_kernel<4><<<grid, block, 0, stream>>>(x, y, n_rows, n_cols); break;
        }
        return cudaGetLastError();
    }
    tk::softmax_rows_kernel<<<n_rows, row_block_size(n_cols), 0, stream>>>(x, y, n_cols);
    return cudaGetLastError();
}

cudaError_t layernorm_fp16_launch(const __half* x, const __half* gamma, const __half* beta, __half* y,
                                  int n_cols, int n_rows, float eps, cudaStream_t stream) {
    if (n_rows == 0) return cudaSuccess;
    if (use_warp_path(n_cols, {x, gamma, beta, y})) {
        const unsigned grid = warp_grid(n_rows), block = tk::WARP_ROWS_PER_BLOCK * 32;
        switch ((n_cols + 255) / 256) {
            case 1: tk::layernorm_warp_kernel<1><<<grid, block, 0, stream>>>(x, gamma, beta, y, n_rows, n_cols, eps); break;
            case 2: tk::layernorm_warp_kernel<2><<<grid, block, 0, stream>>>(x, gamma, beta, y, n_rows, n_cols, eps); break;
            case 3: tk::layernorm_warp_kernel<3><<<grid, block, 0, stream>>>(x, gamma, beta, y, n_rows, n_cols, eps); break;
            default: tk::layernorm_warp_kernel<4><<<grid, block, 0, stream>>>(x, gamma, beta, y, n_rows, n_cols, eps); break;
        }
        return cudaGetLastError();
    }
    tk::layernorm_rows_kernel<<<n_rows, row_block_size(n_cols), 0, stream>>>(x, gamma, beta, y, n_cols, eps);
    return cudaGetLastError();
}

cudaError_t gelu_fp16_launch(const __half* x, __half* y, size_t n, cudaStream_t stream) {
    if (n == 0) return cudaSuccess;
    tk::gelu_kernel<<<elementwise_grid(n, 256), 256, 0, stream>>>(x, y, n);
    return cudaGetLastError();
}

cudaError_t bias_act_fp16_launch(__half* y, const __half* bias, size_t rows, int n_cols, bool gelu,
                                 cudaStream_t stream) {
    const size_t total = rows * static_cast<size_t>(n_cols);
    if (total == 0) return cudaSuccess;
    if (gelu)
        tk::bias_act_kernel<true><<<elementwise_grid(total, 256), 256, 0, stream>>>(y, bias, total, n_cols);
    else
        tk::bias_act_kernel<false><<<elementwise_grid(total, 256), 256, 0, stream>>>(y, bias, total, n_cols);
    return cudaGetLastError();
}

}  // extern "C"
