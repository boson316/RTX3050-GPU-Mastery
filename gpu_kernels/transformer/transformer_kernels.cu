/*
 * Launchers for the transformer kernels (device code in transformer_kernels.cuh).
 * Every launcher returns the launch status so the PyTorch binding can raise on failure
 * instead of timing a kernel that never ran.
 */
#include <algorithm>
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
}  // namespace

extern "C" {

cudaError_t softmax_fp16_launch(const __half* x, __half* y, int n_cols, int n_rows, cudaStream_t stream) {
    if (n_rows == 0) return cudaSuccess;
    tk::softmax_rows_kernel<<<n_rows, row_block_size(n_cols), 0, stream>>>(x, y, n_cols);
    return cudaGetLastError();
}

cudaError_t layernorm_fp16_launch(const __half* x, const __half* gamma, const __half* beta, __half* y,
                                  int n_cols, int n_rows, float eps, cudaStream_t stream) {
    if (n_rows == 0) return cudaSuccess;
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
