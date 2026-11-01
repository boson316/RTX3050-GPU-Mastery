/*
 * Level 1: Naive SGEMM  C = A @ B  (N x N, FP32), one thread per output element.
 * Every thread streams a full row of A and column of B from global memory: 2*N loads per
 * 2*N FLOPs -> arithmetic intensity ~0.25 FLOP/byte before caching, far left of the
 * RTX 3050's ridge point (~23 FLOP/byte), i.e. memory-bound. Level 2 fixes this with tiling.
 *
 * The CPU reference uses the cache-friendly i-k-j loop order so the "speedup vs CPU" number
 * is not inflated by a deliberately slow baseline.
 */
#include <chrono>
#include "../../common/bench_common.cuh"

const int N = 1024;

__global__ void matmul_naive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.f;
        for (int k = 0; k < n; k++) sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

static void matmul_cpu(const std::vector<float>& A, const std::vector<float>& B, std::vector<double>& C, int n) {
    std::fill(C.begin(), C.end(), 0.0);
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            const double a = A[(size_t)i * n + k];
            for (int j = 0; j < n; j++) C[(size_t)i * n + j] += a * B[(size_t)k * n + j];
        }
}

int main() {
    const size_t bytes = (size_t)N * N * sizeof(float);
    auto h_A = bench::random_vector((size_t)N * N, 11), h_B = bench::random_vector((size_t)N * N, 12);
    std::vector<double> ref((size_t)N * N);
    auto t0 = std::chrono::steady_clock::now();
    matmul_cpu(h_A, h_B, ref, N);
    const double cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes)); CUDA_CHECK(cudaMalloc(&d_B, bytes)); CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16), grid((N + 15) / 16, (N + 15) / 16);
    auto naive = [&] { matmul_naive<<<grid, block>>>(d_A, d_B, d_C, N); };
    naive();
    const bool ok = bench::verify("NAIVE", d_C, ref, 1e-3, 1e-4);

    const float gpu_ms = bench::time_ms(naive);
    std::printf("CPU_MS=%.6f\nCUDA_NAIVE_MS=%.6f\n", cpu_ms, gpu_ms);
    bench::print_gflops("NAIVE_GFLOPS", 2.0 * N * N * N, gpu_ms);
    std::printf("N=%d\n", N);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return ok ? 0 : 1;
}
