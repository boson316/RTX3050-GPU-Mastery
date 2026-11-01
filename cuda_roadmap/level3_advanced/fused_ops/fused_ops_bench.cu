/*
 * Level 3: Kernel fusion, C = relu(A + B), N = 2^24 floats.
 *   naive: add kernel (read A, B, write C) + relu kernel (read C, write C) = 5 x 4N bytes
 *   fused: one kernel (read A, B, write C)                                  = 3 x 4N bytes
 * Both are bandwidth-bound, so the expected speedup is ~5/3 = 1.67x - a prediction you
 * can check against the measured numbers.
 */
#include "../../common/bench_common.cuh"

const int N = 1 << 24;

__global__ void kernel_add(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}
__global__ void kernel_relu(float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = fmaxf(C[i], 0.f);
}
__global__ void kernel_add_relu_fused(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = fmaxf(A[i] + B[i], 0.f);
}

int main() {
    const size_t bytes = (size_t)N * sizeof(float);
    auto h_A = bench::random_vector(N, 31), h_B = bench::random_vector(N, 32);
    std::vector<double> ref(N);
    for (int i = 0; i < N; i++) ref[i] = std::max(0.f, h_A[i] + h_B[i]);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes)); CUDA_CHECK(cudaMalloc(&d_B, bytes)); CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    const int block = 256, grid = (N + block - 1) / block;
    auto naive = [&] { kernel_add<<<grid, block>>>(d_A, d_B, d_C, N); kernel_relu<<<grid, block>>>(d_C, N); };
    auto fused = [&] { kernel_add_relu_fused<<<grid, block>>>(d_A, d_B, d_C, N); };

    bool ok = true;
    CUDA_CHECK(cudaMemset(d_C, 0, bytes)); naive(); ok &= bench::verify("NAIVE", d_C, ref, 0, 0);
    CUDA_CHECK(cudaMemset(d_C, 0, bytes)); fused(); ok &= bench::verify("OPTIMIZED", d_C, ref, 0, 0);

    const float naive_ms = bench::time_ms(naive), opt_ms = bench::time_ms(fused);
    std::printf("CUDA_NAIVE_MS=%.6f\nCUDA_OPTIMIZED_MS=%.6f\n", naive_ms, opt_ms);
    bench::print_gbps("NAIVE_EFFECTIVE_GBPS", 5.0 * bytes, naive_ms);
    bench::print_gbps("OPTIMIZED_EFFECTIVE_GBPS", 3.0 * bytes, opt_ms);
    std::printf("EXPECTED_SPEEDUP=%.2f\nN=%d\n", 5.0 / 3.0, N);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return ok ? 0 : 1;
}
