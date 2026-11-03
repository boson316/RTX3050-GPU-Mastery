// Checks the softmax / layernorm / epilogue kernels of transformer_kernels.cuh on the CPU emulator.
// Build & run:  g++ -std=c++20 -O2 -pthread -Wno-unknown-pragmas test_transformer_emu.cpp -o t && ./t
#include <algorithm>
#include <cstdio>
#include <random>
#include <type_traits>
#include <vector>

#include "cuda_emu.h"
#include "../../gpu_kernels/transformer/transformer_kernels.cuh"

static int failures = 0;

static void report(const char* name, int rows, int cols, int threads, float err, float tol) {
    const bool ok = err < tol;
    failures += !ok;
    std::printf("%s %-10s rows=%3d cols=%5d threads=%4d  max|err|=%.2e\n", ok ? "PASS" : "FAIL", name, rows, cols,
                threads, err);
}

static int block_for(int cols) { return (std::min(1024, std::max(32, cols)) + 31) / 32 * 32; }

static void test_softmax(int rows, int cols, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.f, 3.f);
    std::vector<float> x(size_t(rows) * cols), y(x.size());
    for (auto& v : x) v = dist(rng);
    const int threads = block_for(cols);
    emu_launch(Dim3{unsigned(rows), 1, 1}, Dim3{unsigned(threads), 1, 1},
               [&] { tk::softmax_rows_kernel(x.data(), y.data(), cols); });
    float err = 0.f;
    for (int r = 0; r < rows; ++r) {
        const float* xr = &x[size_t(r) * cols];
        double mx = *std::max_element(xr, xr + cols), sum = 0;
        for (int c = 0; c < cols; ++c) sum += std::exp(xr[c] - mx);
        for (int c = 0; c < cols; ++c)
            err = std::max(err, float(std::fabs(std::exp(xr[c] - mx) / sum - y[size_t(r) * cols + c])));
    }
    report("softmax", rows, cols, threads, err, 1e-5f);
}

static void test_layernorm(int rows, int cols, std::mt19937& rng) {
    std::normal_distribution<float> dist(2.f, 5.f);  // non-zero mean stresses the variance computation
    std::vector<float> x(size_t(rows) * cols), y(x.size()), g(cols), b(cols);
    for (auto& v : x) v = dist(rng);
    for (int c = 0; c < cols; ++c) { g[c] = 1.f + 0.01f * c; b[c] = -0.5f + 0.001f * c; }
    const int threads = block_for(cols);
    const float eps = 1e-5f;
    emu_launch(Dim3{unsigned(rows), 1, 1}, Dim3{unsigned(threads), 1, 1},
               [&] { tk::layernorm_rows_kernel(x.data(), g.data(), b.data(), y.data(), cols, eps); });
    float err = 0.f;
    for (int r = 0; r < rows; ++r) {
        const float* xr = &x[size_t(r) * cols];
        double mean = 0, var = 0;
        for (int c = 0; c < cols; ++c) mean += xr[c];
        mean /= cols;
        for (int c = 0; c < cols; ++c) var += (xr[c] - mean) * (xr[c] - mean);
        var /= cols;
        for (int c = 0; c < cols; ++c) {
            const double ref = (xr[c] - mean) / std::sqrt(var + eps) * g[c] + b[c];
            err = std::max(err, float(std::fabs(ref - y[size_t(r) * cols + c])));
        }
    }
    report("layernorm", rows, cols, threads, err, 1e-4f);
}

static void test_bias_gelu(int rows, int cols, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.f, 2.f);
    std::vector<float> y(size_t(rows) * cols), b(cols);
    for (auto& v : y) v = dist(rng);
    for (auto& v : b) v = dist(rng);
    const auto orig = y;
    // deliberately small grid: the grid-stride loop must still cover everything
    emu_launch(Dim3{3, 1, 1}, Dim3{64, 1, 1},
               [&] { tk::bias_act_kernel<true>(y.data(), b.data(), y.size(), cols); });
    float err = 0.f;
    for (size_t i = 0; i < y.size(); ++i) {
        const double v = orig[i] + b[i % cols];
        const double ref = 0.5 * v * (1 + std::tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));
        err = std::max(err, float(std::fabs(ref - y[i])));
    }
    report("bias+gelu", rows, cols, 64, err, 1e-5f);
}

// Warp-per-row kernels: rows not a multiple of WARP_ROWS_PER_BLOCK (a whole warp exits early),
// and every chunk count 1..4 including a partial last chunk (n_cols = 1000).
static void test_warp_kernels(int rows, int cols, std::mt19937& rng) {
    std::normal_distribution<float> dist(1.f, 3.f);
    std::vector<float> x(size_t(rows) * cols), ys(x.size(), -7.f), yl(x.size(), -7.f), g(cols), b(cols);
    for (auto& v : x) v = dist(rng);
    for (int c = 0; c < cols; ++c) { g[c] = 0.5f + 0.001f * c; b[c] = 0.1f - 0.0005f * c; }
    const int chunks = (cols + 255) / 256;
    const Dim3 grid{unsigned((rows + tk::WARP_ROWS_PER_BLOCK - 1) / tk::WARP_ROWS_PER_BLOCK), 1, 1};
    const Dim3 block{unsigned(tk::WARP_ROWS_PER_BLOCK * 32), 1, 1};
    auto launch = [&](auto kernel_tag) {
        constexpr int C = decltype(kernel_tag)::value;
        emu_launch(grid, block, [&] { tk::softmax_warp_kernel<C>(x.data(), ys.data(), rows, cols); });
        emu_launch(grid, block, [&] { tk::layernorm_warp_kernel<C>(x.data(), g.data(), b.data(), yl.data(), rows, cols, 1e-5f); });
    };
    switch (chunks) {
        case 1: launch(std::integral_constant<int, 1>{}); break;
        case 2: launch(std::integral_constant<int, 2>{}); break;
        case 3: launch(std::integral_constant<int, 3>{}); break;
        default: launch(std::integral_constant<int, 4>{}); break;
    }
    float es = 0.f, el = 0.f;
    for (int r = 0; r < rows; ++r) {
        const float* xr = &x[size_t(r) * cols];
        double mx = *std::max_element(xr, xr + cols), sum = 0, mean = 0, var = 0;
        for (int c = 0; c < cols; ++c) { sum += std::exp(xr[c] - mx); mean += xr[c]; }
        mean /= cols;
        for (int c = 0; c < cols; ++c) var += (xr[c] - mean) * (xr[c] - mean);
        var /= cols;
        for (int c = 0; c < cols; ++c) {
            es = std::max(es, float(std::fabs(std::exp(xr[c] - mx) / sum - ys[size_t(r) * cols + c])));
            const double ref = (xr[c] - mean) / std::sqrt(var + 1e-5) * g[c] + b[c];
            el = std::max(el, float(std::fabs(ref - yl[size_t(r) * cols + c])));
        }
    }
    report("softmax_w", rows, cols, int(block.x), es, 1e-5f);
    report("layernrm_w", rows, cols, int(block.x), el, 1e-4f);
}

int main() {
    std::mt19937 rng(42);
    // cols < 32, not a multiple of 32, > 1024 (multiple elements per thread), typical d_model
    for (int cols : {7, 33, 100, 768, 1500}) {
        test_softmax(4, cols, rng);
        test_layernorm(4, cols, rng);
    }
    test_bias_gelu(5, 37, rng);
    for (int cols : {8, 256, 264, 768, 1000, 1024}) test_warp_kernels(7, cols, rng);
    std::printf(failures ? "\n%d case(s) FAILED\n" : "\nall cases passed\n", failures);
    return failures ? 1 : 0;
}
