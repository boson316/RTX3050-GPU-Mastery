/*
 * Shared helpers for the cuda_roadmap benchmarks.
 *
 * Methodology (applies to every benchmark in this directory):
 *   1. Warm-up launches first. CUDA 12 loads modules lazily, so the very first launch of
 *      a kernel includes module loading and can be 10-100x slower than steady state.
 *   2. Each iteration is timed with its own pair of CUDA events; the MEDIAN is reported
 *      (robust to clock boost / OS jitter on a laptop GPU).
 *   3. Every kernel's output is checked against a CPU reference before its time is
 *      printed. A wrong-but-fast kernel prints VERIFY_...=FAIL and the program exits 1.
 *
 * Output format (parsed by run_benchmarks.py): KEY=VALUE lines.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),    \
                         __FILE__, __LINE__);                                             \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

namespace bench {

// Median time (ms) of `launch()` over `iters` runs after `warmup` runs.
template <typename F>
float time_ms(F&& launch, int warmup = 3, int iters = 20) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> samples(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&samples[i], start, stop));
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::nth_element(samples.begin(), samples.begin() + iters / 2, samples.end());
    return samples[iters / 2];
}

// Compare a device buffer against a host reference: |got - ref| <= atol + rtol * |ref|.
template <typename T>
bool verify(const char* name, const T* d_got, const std::vector<double>& ref, double atol, double rtol) {
    std::vector<T> got(ref.size());
    CUDA_CHECK(cudaMemcpy(got.data(), d_got, ref.size() * sizeof(T), cudaMemcpyDeviceToHost));
    double max_err = 0.0;
    size_t bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double g = static_cast<double>(got[i]);
        const double e = std::fabs(g - ref[i]);
        max_err = std::max(max_err, e);
        if (!(e <= atol + rtol * std::fabs(ref[i]))) ++bad;  // also catches NaN
    }
    std::printf("VERIFY_%s=%s\n", name, bad == 0 ? "PASS" : "FAIL");
    std::printf("MAXERR_%s=%.3e\n", name, max_err);
    if (bad) std::fprintf(stderr, "%s: %zu / %zu elements out of tolerance\n", name, bad, ref.size());
    return bad == 0;
}

inline std::vector<float> random_vector(size_t n, unsigned seed, float lo = -1.f, float hi = 1.f) {
    std::vector<float> v(n);
    unsigned s = seed;
    for (auto& x : v) {  // xorshift32: deterministic, no <random> needed
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        x = lo + (hi - lo) * (s / 4294967296.0f);
    }
    return v;
}

inline void print_gbps(const char* key, double bytes, float ms) {
    std::printf("%s=%.1f\n", key, bytes / (ms * 1e-3) / 1e9);
}

inline void print_gflops(const char* key, double flops, float ms) {
    std::printf("%s=%.1f\n", key, flops / (ms * 1e-3) / 1e9);
}

}  // namespace bench
