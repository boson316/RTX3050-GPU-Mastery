// Checks extension/conv3x3_kernel.cuh against a direct CPU convolution on the emulator.
#include <cstdio>
#include <random>
#include <vector>

#include "cuda_emu.h"
#include "../../extension/conv3x3_kernel.cuh"

static int check(int B, int H, int W, unsigned seed) {
    using namespace conv3x3;
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<float> x(size_t(B) * H * W), w(C_OUT * 9), b(C_OUT);
    for (auto* v : {&x, &w, &b}) for (auto& e : *v) e = dist(rng);
    const int oh = H - 2, ow = W - 2;
    std::vector<float> y(size_t(B) * C_OUT * oh * ow, -999.f);

    Dim3 grid{unsigned((ow + TILE - 1) / TILE), unsigned((oh + TILE - 1) / TILE), unsigned(B)};
    emu_launch(grid, Dim3{TILE, TILE, 1}, [&] { conv3x3_c32_kernel<float>(x.data(), w.data(), b.data(), y.data(), H, W); });

    float err = 0.f;
    for (int n = 0; n < B; ++n)
        for (int c = 0; c < C_OUT; ++c)
            for (int i = 0; i < oh; ++i)
                for (int j = 0; j < ow; ++j) {
                    double ref = b[c];
                    for (int k = 0; k < 9; ++k) ref += double(x[(size_t(n) * H + i + k / 3) * W + j + k % 3]) * w[c * 9 + k];
                    err = std::max(err, float(std::fabs(ref - y[((size_t(n) * C_OUT + c) * oh + i) * ow + j])));
                }
    const bool ok = err < 1e-5f;
    std::printf("%s conv3x3 B=%d H=%3d W=%3d  max|err|=%.2e\n", ok ? "PASS" : "FAIL", B, H, W, err);
    return ok ? 0 : 1;
}

int main() {
    int f = 0;
    f += check(2, 28, 28, 1);   // MNIST: 26x26 output, not a multiple of the 16x16 tile
    f += check(1, 18, 18, 2);   // exactly one tile
    f += check(1, 3, 40, 3);    // single output row, several column tiles
    f += check(3, 35, 19, 4);
    std::printf(f ? "\n%d case(s) FAILED\n" : "\nall cases passed\n", f);
    return f ? 1 : 0;
}
