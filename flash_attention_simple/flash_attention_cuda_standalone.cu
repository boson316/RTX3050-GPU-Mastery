/*
 * Standalone FlashAttention benchmark (no PyTorch needed).
 *   Usage: flash_attention_cuda_standalone [S] [causal]      default S=512, causal=0
 * Uses the same kernel as the PyTorch extension (flash_attention_kernel.cuh), verifies a
 * slice of the output against a double-precision CPU reference, then times it.
 * Prints KEY=VALUE lines; exit code 1 if verification fails.
 *
 * Build: nvcc -O3 --use_fast_math -arch=sm_86 -std=c++17 flash_attention_cuda_standalone.cu -o flash_attention_cuda_standalone
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../cuda_roadmap/common/bench_common.cuh"
#include "flash_attention_kernel.cuh"

int main(int argc, char** argv) {
    const int S = argc > 1 ? std::atoi(argv[1]) : 512;
    const bool causal = argc > 2 && std::atoi(argv[2]) != 0;
    const int B = 2, H = 8, D = 64;
    const size_t n = (size_t)B * H * S * D;
    const float scale = 1.f / std::sqrt((float)D);

    auto hQ = bench::random_vector(n, 1), hK = bench::random_vector(n, 2), hV = bench::random_vector(n, 3);
    float *dQ, *dK, *dV, *dO;
    for (float** p : {&dQ, &dK, &dV, &dO}) CUDA_CHECK(cudaMalloc(p, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    auto run = [&] {
        CUDA_CHECK(flash_attn::launch_flash_attention_fwd(dQ, dK, dV, dO, B, H, S, D, scale, causal, 0));
    };
    run();

    // Reference for the first (b, h) slice only (full reference is O(B*H*S^2*D) on the CPU).
    std::vector<double> ref((size_t)S * D);
    std::vector<double> p(S);
    for (int i = 0; i < S; i++) {
        double mx = -1e300;
        for (int j = 0; j < S; j++) {
            double dot = 0;
            for (int d = 0; d < D; d++) dot += (double)hQ[(size_t)i * D + d] * hK[(size_t)j * D + d];
            p[j] = (causal && j > i) ? -1e300 : dot * scale;
            mx = std::max(mx, p[j]);
        }
        double sum = 0;
        for (int j = 0; j < S; j++) { p[j] = (causal && j > i) ? 0.0 : std::exp(p[j] - mx); sum += p[j]; }
        for (int d = 0; d < D; d++) {
            double acc = 0;
            for (int j = 0; j < S; j++) acc += p[j] * hV[(size_t)j * D + d];
            ref[(size_t)i * D + d] = acc / sum;
        }
    }
    const bool ok = bench::verify("FLASH_ATTENTION", dO, ref, 1e-4, 1e-3);

    const float ms = bench::time_ms(run, 3, 20);
    const double flops = 4.0 * B * H * S * S * D * (causal ? 0.5 : 1.0);
    std::printf("CUDA_MS=%.4f\n", ms);
    bench::print_gflops("GFLOPS", flops, ms);
    std::printf("S=%d B=%d H=%d D=%d CAUSAL=%d\n", S, B, H, D, causal ? 1 : 0);
    for (float* p2 : {dQ, dK, dV, dO}) cudaFree(p2);
    return ok ? 0 : 1;
}
