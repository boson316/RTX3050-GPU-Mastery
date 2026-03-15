/*
 * Level 4: FP16 Matrix Multiply — Naive (FP32 CUDA cores) vs Tensor Core (FP16)
 * Naive = FP16 elements but scalar/tiled on CUDA cores; Optimized = use WMMA (tensor cores).
 * Build: nvcc -arch=sm_70 (or sm_86 for Ampere) -o fp16_matmul_bench fp16_matmul_bench.cu
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(1); } \
} while(0)

#define M 1024
#define N 1024
#define K 1024
#define TILE 16

// Naive: FP16 tiled matmul on CUDA cores (no tensor core)
__global__ void matmul_fp16_naive(const __half* A, const __half* B, __half* C, int m, int n, int k) {
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    if (row >= m || col >= n) return;
    float sum = 0.f;
    for (int i = 0; i < k; i++)
        sum += __half2float(A[row * k + i]) * __half2float(B[i * n + col]);
    C[row * n + col] = __float2half(sum);
}

// Optimized: use Tensor Cores via WMMA (16x16x16)
#include <mma.h>
using namespace nvcuda;
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void matmul_fp16_wmma(const half* A, const half* B, float* C, int m, int n, int k) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.f);

    int row = blockIdx.y * WMMA_M;
    int col = blockIdx.x * WMMA_N;
    for (int i = 0; i < k; i += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + row * k + i, k);
        wmma::load_matrix_sync(b_frag, B + i * n + col, n);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    wmma::store_matrix_sync(C + row * n + col, acc_frag, n, wmma::mem_row_major);
}

int main() {
    size_t szA = (size_t)M * K * sizeof(__half);
    size_t szB = (size_t)K * N * sizeof(__half);
    size_t szC = (size_t)M * N * sizeof(__half);
    size_t szCf = (size_t)M * N * sizeof(float);
    __half *d_A, *d_B, *d_C;
    float *d_Cf;
    CUDA_CHECK(cudaMalloc(&d_A, szA));
    CUDA_CHECK(cudaMalloc(&d_B, szB));
    CUDA_CHECK(cudaMalloc(&d_C, szC));
    CUDA_CHECK(cudaMalloc(&d_Cf, szCf));
    __half *h_A = (__half*)malloc(szA), *h_B = (__half*)malloc(szB);
    for (int i = 0; i < M*K; i++) h_A[i] = __float2half(0.01f);
    for (int i = 0; i < K*N; i++) h_B[i] = __float2half(0.01f);
    CUDA_CHECK(cudaMemcpy(d_A, h_A, szA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, szB, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE), grid((N+TILE-1)/TILE, (M+TILE-1)/TILE);
    dim3 blockW(32, 1), gridW((N+WMMA_N-1)/WMMA_N, (M+WMMA_M-1)/WMMA_M);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    matmul_fp16_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop, 0)); CUDA_CHECK(cudaEventSynchronize(stop));
    float naive_ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start, stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    matmul_fp16_wmma<<<gridW, blockW>>>(d_A, d_B, d_Cf, M, N, K);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaEventRecord(stop, 0)); CUDA_CHECK(cudaEventSynchronize(stop));
    float opt_ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&opt_ms, start, stop));

    printf("CUDA_NAIVE_MS=%.6f\n", naive_ms);
    printf("CUDA_OPTIMIZED_MS=%.6f\n", opt_ms);
    printf("M=%d N=%d K=%d\n", M, N, K);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_Cf); free(h_A); free(h_B);
    return 0;
}
