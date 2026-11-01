/*
 * Level 1: Sum reduction of N = 2^20 floats.
 *   naive:     every thread does atomicAdd on ONE global address -> 2^20 serialised atomics
 *   optimized: shared-memory tree per block (sequential addressing, no bank conflicts),
 *              then one atomicAdd per block (4096 atomics instead of 1M)
 * Both compute the full sum, so the results are directly comparable and verified.
 */
#include "../../common/bench_common.cuh"

const int N = 1 << 20;
const int BLOCK = 256;

__global__ void reduce_naive(const float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(output, input[i]);
}

__global__ void reduce_optimized(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK + tid;
    sdata[tid] = (i < n) ? input[i] : 0.f;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {  // sequential addressing: active threads stay contiguous
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(output, sdata[0]);
}

int main() {
    auto h_in = bench::random_vector(N, 7, 0.f, 1.f);
    double sum = 0; for (float v : h_in) sum += v;
    const std::vector<double> ref{sum};

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    const int grid = (N + BLOCK - 1) / BLOCK;
    // The memset is part of the operation (the kernels accumulate into *d_out).
    auto naive = [&] { cudaMemsetAsync(d_out, 0, sizeof(float)); reduce_naive<<<grid, BLOCK>>>(d_in, d_out, N); };
    auto opt = [&] { cudaMemsetAsync(d_out, 0, sizeof(float)); reduce_optimized<<<grid, BLOCK>>>(d_in, d_out, N); };

    // FP32 accumulation of 1M values in arbitrary (atomic) order: allow ~1e-4 relative error.
    bool ok = true;
    naive(); ok &= bench::verify("NAIVE", d_out, ref, 0.0, 1e-4);
    opt();   ok &= bench::verify("OPTIMIZED", d_out, ref, 0.0, 1e-4);

    const float naive_ms = bench::time_ms(naive), opt_ms = bench::time_ms(opt);
    std::printf("CUDA_NAIVE_MS=%.6f\nCUDA_OPTIMIZED_MS=%.6f\n", naive_ms, opt_ms);
    bench::print_gbps("OPTIMIZED_GBPS", (double)N * sizeof(float), opt_ms);
    std::printf("N=%d\n", N);

    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
