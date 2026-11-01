/*
 * Level 1: Vector add  C = A + B  (N = 2^20 floats)
 *   naive:     one float per thread (32-bit loads)
 *   optimized: one float4 per thread (128-bit loads -> 4x fewer load instructions)
 * Both are bandwidth-bound (12 bytes moved per FLOP), so the figure of merit is GB/s vs
 * the ~192 GB/s DRAM peak of the RTX 3050 Laptop, not FLOPS.
 */
#include "../../common/bench_common.cuh"

const int N = 1 << 20;

__global__ void vector_add_naive(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

// cudaMalloc returns 256-byte aligned pointers, so the float4 casts are aligned.
__global__ void vector_add_optimized(const float* A, const float* B, float* C, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        float4 a = *reinterpret_cast<const float4*>(A + i);
        float4 b = *reinterpret_cast<const float4*>(B + i);
        *reinterpret_cast<float4*>(C + i) = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    } else {
        for (int j = i; j < n; j++) C[j] = A[j] + B[j];  // tail (n % 4 != 0)
    }
}

int main() {
    const size_t bytes = (size_t)N * sizeof(float);
    auto h_A = bench::random_vector(N, 1), h_B = bench::random_vector(N, 2);
    std::vector<double> ref(N);
    for (int i = 0; i < N; i++) ref[i] = (double)h_A[i] + h_B[i];

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes)); CUDA_CHECK(cudaMalloc(&d_B, bytes)); CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    const int block = 256;
    auto naive = [&] { vector_add_naive<<<(N + block - 1) / block, block>>>(d_A, d_B, d_C, N); };
    auto opt = [&] { vector_add_optimized<<<(N / 4 + block - 1) / block, block>>>(d_A, d_B, d_C, N); };

    bool ok = true;
    CUDA_CHECK(cudaMemset(d_C, 0, bytes)); naive(); ok &= bench::verify("NAIVE", d_C, ref, 1e-6, 1e-6);
    CUDA_CHECK(cudaMemset(d_C, 0, bytes)); opt();   ok &= bench::verify("OPTIMIZED", d_C, ref, 1e-6, 1e-6);

    const float naive_ms = bench::time_ms(naive), opt_ms = bench::time_ms(opt);
    std::printf("CUDA_NAIVE_MS=%.6f\nCUDA_OPTIMIZED_MS=%.6f\n", naive_ms, opt_ms);
    bench::print_gbps("NAIVE_GBPS", 3.0 * bytes, naive_ms);
    bench::print_gbps("OPTIMIZED_GBPS", 3.0 * bytes, opt_ms);
    std::printf("N=%d\n", N);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return ok ? 0 : 1;
}
