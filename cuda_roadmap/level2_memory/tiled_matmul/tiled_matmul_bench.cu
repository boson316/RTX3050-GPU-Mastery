/*
 * Level 2: Shared-memory tiled SGEMM vs naive (N x N, FP32).
 * Each TILE x TILE block stages a tile of A and of B in shared memory, so every global
 * load is reused TILE times: global traffic drops by TILE (16x), arithmetic intensity
 * rises from ~0.25 to ~4 FLOP/byte.
 */
#include "../../common/bench_common.cuh"

const int N = 1024;
#define TILE 16

__global__ void matmul_naive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.f;
        for (int k = 0; k < n; k++) sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

__global__ void matmul_tiled(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.f;
    for (int t = 0; t < (n + TILE - 1) / TILE; t++) {
        int colA = t * TILE + threadIdx.x, rowB = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < n && colA < n) ? A[row * n + colA] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (rowB < n && col < n) ? B[rowB * n + col] : 0.f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; k++) sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < n && col < n) C[row * n + col] = sum;
}

int main() {
    const size_t bytes = (size_t)N * N * sizeof(float);
    auto h_A = bench::random_vector((size_t)N * N, 21), h_B = bench::random_vector((size_t)N * N, 22);
    std::vector<double> ref((size_t)N * N, 0.0);
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            const double a = h_A[(size_t)i * N + k];
            for (int j = 0; j < N; j++) ref[(size_t)i * N + j] += a * h_B[(size_t)k * N + j];
        }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes)); CUDA_CHECK(cudaMalloc(&d_B, bytes)); CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    auto naive = [&] { matmul_naive<<<grid, block>>>(d_A, d_B, d_C, N); };
    auto tiled = [&] { matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, N); };

    bool ok = true;
    CUDA_CHECK(cudaMemset(d_C, 0, bytes)); naive(); ok &= bench::verify("NAIVE", d_C, ref, 1e-3, 1e-4);
    CUDA_CHECK(cudaMemset(d_C, 0, bytes)); tiled(); ok &= bench::verify("OPTIMIZED", d_C, ref, 1e-3, 1e-4);

    const float naive_ms = bench::time_ms(naive), opt_ms = bench::time_ms(tiled);
    std::printf("CUDA_NAIVE_MS=%.6f\nCUDA_OPTIMIZED_MS=%.6f\n", naive_ms, opt_ms);
    bench::print_gflops("NAIVE_GFLOPS", 2.0 * N * N * N, naive_ms);
    bench::print_gflops("OPTIMIZED_GFLOPS", 2.0 * N * N * N, opt_ms);
    std::printf("N=%d\n", N);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return ok ? 0 : 1;
}
