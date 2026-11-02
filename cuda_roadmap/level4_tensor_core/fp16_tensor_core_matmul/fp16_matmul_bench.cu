/*
 * Level 4: FP16 GEMM on Tensor Cores.  C (fp32) = A (fp16, MxK) @ B (fp16, KxN), all row-major.
 *
 *   1. cuda_core  : one thread per output, FP32 FMA on CUDA cores (no Tensor Cores)
 *   2. wmma_naive : one warp per 16x16 output tile, fragments loaded straight from global memory
 *   3. wmma_tiled : 64x64x32 block tile staged in shared memory, 4 warps x (2x2) fragments;
 *                   each A/B fragment loaded from smem is reused by 2 MMAs, each global load
 *                   by 2 warps; 128-bit (int4) global->smem copies; +8 half padding per smem
 *                   row to spread fragment loads across banks
 *   4. cuBLAS     : cublasGemmEx with Tensor Cores - the reference we compare against
 *
 * Bug fixed (see docs/ENGINEERING_LOG.md): the previous WMMA kernel declared the B fragment
 * as wmma::col_major while B is stored row-major, so it computed A @ B^T. The benchmark
 * filled A and B with the constant 0.01, for which A @ B == A @ B^T, so the bug was invisible.
 * Inputs are now random and every kernel is verified against a CPU reference.
 *
 * Build: nvcc -O3 -arch=sm_86 -lcublas fp16_matmul_bench.cu -o fp16_matmul_bench
 */
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "../../common/bench_common.cuh"

using namespace nvcuda;

#ifndef GEMM_SIZE
#define GEMM_SIZE 1024
#endif
const int M = GEMM_SIZE, N = GEMM_SIZE, K = GEMM_SIZE;

// ---------------------------------------------------------------- 1. CUDA cores
__global__ void matmul_fp16_cuda_core(const half* A, const half* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    float sum = 0.f;
    for (int i = 0; i < k; i++) sum += __half2float(A[row * k + i]) * __half2float(B[i * n + col]);
    C[row * n + col] = sum;
}

// ---------------------------------------------------------------- 2. WMMA, no shared memory
__global__ void matmul_wmma_naive(const half* A, const half* B, float* C, int m, int n, int k) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;  // B is row-major!
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    const int row = blockIdx.y * 16, col = blockIdx.x * 16;
    for (int i = 0; i < k; i += 16) {
        wmma::load_matrix_sync(a_frag, A + row * k + i, k);
        wmma::load_matrix_sync(b_frag, B + i * n + col, n);
        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }
    wmma::store_matrix_sync(C + row * n + col, acc, n, wmma::mem_row_major);
}

// ---------------------------------------------------------------- 3. WMMA, shared-memory tiled
constexpr int BM = 64, BN = 64, BK = 32;
constexpr int APAD = 8, BPAD = 8;      // halves; keeps rows 16-byte aligned (80 B / 144 B)
constexpr int TILED_THREADS = 128;     // 4 warps in a 2x2 grid, each owns a 32x32 sub-tile

__global__ void __launch_bounds__(TILED_THREADS)
matmul_wmma_tiled(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C,
                  int m, int n, int k) {
    __shared__ __align__(32) half As[BM][BK + APAD];
    __shared__ __align__(32) half Bs[BK][BN + BPAD];

    const int warp = threadIdx.x / 32;
    const int wm = warp / 2, wn = warp % 2;
    const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) wmma::fill_fragment(acc[i][j], 0.f);

    for (int k0 = 0; k0 < k; k0 += BK) {
        // 64x32 A tile and 32x64 B tile = 256 int4 (8 halves) each -> 2 per thread, coalesced
        for (int i = threadIdx.x; i < BM * BK / 8; i += TILED_THREADS) {
            const int r = i / (BK / 8), c = (i % (BK / 8)) * 8;
            *reinterpret_cast<int4*>(&As[r][c]) = *reinterpret_cast<const int4*>(&A[(size_t)(bm + r) * k + k0 + c]);
        }
        for (int i = threadIdx.x; i < BK * BN / 8; i += TILED_THREADS) {
            const int r = i / (BN / 8), c = (i % (BN / 8)) * 8;
            *reinterpret_cast<int4*>(&Bs[r][c]) = *reinterpret_cast<const int4*>(&B[(size_t)(k0 + r) * n + bn + c]);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[2];
#pragma unroll
            for (int i = 0; i < 2; i++) wmma::load_matrix_sync(a_frag[i], &As[wm * 32 + i * 16][kk], BK + APAD);
#pragma unroll
            for (int j = 0; j < 2; j++) wmma::load_matrix_sync(b_frag[j], &Bs[kk][wn * 32 + j * 16], BN + BPAD);
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 2; j++) wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::store_matrix_sync(C + (size_t)(bm + wm * 32 + i * 16) * n + bn + wn * 32 + j * 16, acc[i][j], n,
                                    wmma::mem_row_major);
}

int main() {
    static_assert(M % BM == 0 && N % BN == 0 && K % BK == 0, "sizes must be multiples of the tile");
    auto fa = bench::random_vector((size_t)M * K, 101), fb = bench::random_vector((size_t)K * N, 102);
    std::vector<half> h_A(fa.size()), h_B(fb.size());
    std::vector<float> a32(fa.size()), b32(fb.size());  // the exact fp16 values, for the reference
    for (size_t i = 0; i < fa.size(); i++) { h_A[i] = __float2half(fa[i]); a32[i] = __half2float(h_A[i]); }
    for (size_t i = 0; i < fb.size(); i++) { h_B[i] = __float2half(fb[i]); b32[i] = __half2float(h_B[i]); }
    std::vector<double> ref((size_t)M * N, 0.0);
    for (int i = 0; i < M; i++)
        for (int p = 0; p < K; p++) {
            const double a = a32[(size_t)i * K + p];
            for (int j = 0; j < N; j++) ref[(size_t)i * N + j] += a * b32[(size_t)p * N + j];
        }

    half *d_A, *d_B;
    float* d_C;
    CUDA_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, h_B.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(half), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "cublasCreate failed\n"); return 1; }
    const float alpha = 1.f, beta = 0.f;

    auto cuda_core = [&] { matmul_fp16_cuda_core<<<dim3(N / 16, M / 16), dim3(16, 16)>>>(d_A, d_B, d_C, M, N, K); };
    auto wmma_naive = [&] { matmul_wmma_naive<<<dim3(N / 16, M / 16), 32>>>(d_A, d_B, d_C, M, N, K); };
    auto wmma_tiled = [&] { matmul_wmma_tiled<<<dim3(N / BN, M / BM), TILED_THREADS>>>(d_A, d_B, d_C, M, N, K); };
    // Row-major C = A @ B  <=>  column-major C^T = B^T @ A^T  (see gpu_kernels/transformer/transformer_cuda.cpp)
    auto cublas = [&] {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, CUDA_R_16F, N, d_A, CUDA_R_16F, K,
                     &beta, d_C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    };

    const double flops = 2.0 * M * N * K, atol = 1e-2, rtol = 1e-3;
    bool ok = true;
    float ms[4];
    const char* names[4] = {"CUDA_CORE", "WMMA_NAIVE", "WMMA_TILED", "CUBLAS"};
    for (int v = 0; v < 4; v++) {
        CUDA_CHECK(cudaMemset(d_C, 0, (size_t)M * N * sizeof(float)));
        auto run = [&] { if (v == 0) cuda_core(); else if (v == 1) wmma_naive(); else if (v == 2) wmma_tiled(); else cublas(); };
        run();
        ok &= bench::verify(names[v], d_C, ref, atol, rtol);
        ms[v] = bench::time_ms(run, 3, 20);
        std::printf("%s_MS=%.6f\n", names[v], ms[v]);
        char key[64]; std::snprintf(key, sizeof key, "%s_GFLOPS", names[v]);
        bench::print_gflops(key, flops, ms[v]);
    }
    std::printf("WMMA_TILED_PCT_OF_CUBLAS=%.1f\n", 100.0 * ms[3] / ms[2]);
    // keys understood by run_benchmarks.py
    std::printf("CUDA_NAIVE_MS=%.6f\nCUDA_OPTIMIZED_MS=%.6f\n", ms[0], ms[2]);
    std::printf("M=%d N=%d K=%d\n", M, N, K);

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return ok ? 0 : 1;
}
