// Checks flash_attention_kernel.cuh against a naive O(S^2) reference on the CPU emulator.
// Build & run:  g++ -std=c++20 -O2 -pthread -Wno-unknown-pragmas test_flash_attention_emu.cpp -o t && ./t
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

#include "cuda_emu.h"
#include "../../flash_attention_simple/flash_attention_kernel.cuh"

static void reference(const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V,
                      std::vector<float>& O, int BH, int S, int D, float scale, bool causal) {
    std::vector<double> p(S);
    for (int bh = 0; bh < BH; ++bh)
        for (int i = 0; i < S; ++i) {
            const size_t base = static_cast<size_t>(bh) * S * D;
            double mx = -1e300;
            for (int j = 0; j < S; ++j) {
                double dot = 0;
                for (int d = 0; d < D; ++d) dot += double(Q[base + i * D + d]) * K[base + j * D + d];
                p[j] = (causal && j > i) ? -1e300 : dot * scale;
                mx = std::max(mx, p[j]);
            }
            double sum = 0;
            for (int j = 0; j < S; ++j) { p[j] = (causal && j > i) ? 0 : std::exp(p[j] - mx); sum += p[j]; }
            for (int d = 0; d < D; ++d) {
                double acc = 0;
                for (int j = 0; j < S; ++j) acc += p[j] * V[base + j * D + d];
                O[base + i * D + d] = static_cast<float>(acc / sum);
            }
        }
}

template <int D_PAD>
static void run_kernel(const float* Q, const float* K, const float* V, float* O,
                       int BH, int S, int D, float scale, bool causal) {
    using namespace flash_attn;
    Dim3 grid{static_cast<unsigned>((S + BLOCK_M - 1) / BLOCK_M), static_cast<unsigned>(BH), 1};
    emu_launch(grid, Dim3{BLOCK_M, 1, 1}, [&] {
        flash_attention_fwd_kernel<D_PAD>(Q, K, V, O, S, D, scale, causal);
    });
}

static int check(int BH, int S, int D, bool causal, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.f, 1.f);
    const size_t n = static_cast<size_t>(BH) * S * D;
    std::vector<float> Q(n), K(n), V(n), O(n, -12345.f), R(n);
    for (auto* v : {&Q, &K, &V}) for (auto& x : *v) x = dist(rng);
    const float scale = 1.f / std::sqrt(static_cast<float>(D));

    if (D <= 16) run_kernel<16>(Q.data(), K.data(), V.data(), O.data(), BH, S, D, scale, causal);
    else if (D <= 32) run_kernel<32>(Q.data(), K.data(), V.data(), O.data(), BH, S, D, scale, causal);
    else run_kernel<64>(Q.data(), K.data(), V.data(), O.data(), BH, S, D, scale, causal);
    reference(Q, K, V, R, BH, S, D, scale, causal);

    float max_err = 0.f;
    for (size_t i = 0; i < n; ++i) max_err = std::max(max_err, std::fabs(O[i] - R[i]));
    const bool ok = max_err < 1e-4f;
    std::printf("%s BH=%d S=%4d D=%2d causal=%d  max|err|=%.2e\n", ok ? "PASS" : "FAIL", BH, S, D, causal, max_err);
    return ok ? 0 : 1;
}

int main() {
    int failures = 0;
    // Includes S not divisible by BLOCK_M / BLOCK_N and D not a power of two:
    // exactly the shapes the previous kernel got wrong.
    for (bool causal : {false, true}) {
        failures += check(2, 64, 64, causal, 1);
        failures += check(2, 100, 64, causal, 2);
        failures += check(1, 17, 32, causal, 3);
        failures += check(3, 130, 40, causal, 4);
        failures += check(1, 1, 8, causal, 5);
        failures += check(1, 257, 16, causal, 6);
    }
    std::printf(failures ? "\n%d case(s) FAILED\n" : "\nall cases passed\n", failures);
    return failures ? 1 : 0;
}
