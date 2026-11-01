/*
 * Level 2: Memory coalescing. Both kernels copy ALL N floats exactly once; only the
 * mapping thread -> element differs, so the GB/s numbers are directly comparable.
 *   coalesced: thread i copies element i -> a warp touches 32 consecutive floats = 1 x 128 B
 *   strided:   thread i copies element (i % R) * STRIDE + i / R   (R = N / STRIDE)
 *              -> the 32 lanes of a warp are STRIDE*4 bytes apart = 32 separate 32 B sectors,
 *              so each 4-byte access drags a whole sector through the memory system (8x waste).
 * (The previous version had the strided kernel copy only N/STRIDE elements, so it did 256x
 *  less work than the coalesced one and the comparison was meaningless.)
 */
#include "../../common/bench_common.cuh"

const int N = 1 << 24;   // 64 MB per buffer: far larger than the 2 MB L2
const int STRIDE = 32;

__global__ void copy_strided(const float* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int rows = n / STRIDE;
    if (i < n) {
        int idx = (i % rows) * STRIDE + i / rows;  // a permutation of [0, n)
        out[idx] = in[idx];
    }
}

__global__ void copy_coalesced(const float* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

int main() {
    static_assert(N % STRIDE == 0, "N must be a multiple of STRIDE");
    const size_t bytes = (size_t)N * sizeof(float);
    auto h = bench::random_vector(N, 3);
    std::vector<double> ref(h.begin(), h.end());

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes)); CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice));

    const int block = 256, grid = (N + block - 1) / block;
    auto strided = [&] { copy_strided<<<grid, block>>>(d_in, d_out, N); };
    auto coalesced = [&] { copy_coalesced<<<grid, block>>>(d_in, d_out, N); };

    bool ok = true;
    CUDA_CHECK(cudaMemset(d_out, 0, bytes)); strided();   ok &= bench::verify("NAIVE", d_out, ref, 0, 0);
    CUDA_CHECK(cudaMemset(d_out, 0, bytes)); coalesced(); ok &= bench::verify("OPTIMIZED", d_out, ref, 0, 0);

    const float naive_ms = bench::time_ms(strided), opt_ms = bench::time_ms(coalesced);
    std::printf("CUDA_NAIVE_MS=%.6f\nCUDA_OPTIMIZED_MS=%.6f\n", naive_ms, opt_ms);
    bench::print_gbps("NAIVE_GBPS", 2.0 * bytes, naive_ms);
    bench::print_gbps("OPTIMIZED_GBPS", 2.0 * bytes, opt_ms);
    std::printf("N=%d STRIDE=%d\n", N, STRIDE);

    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
