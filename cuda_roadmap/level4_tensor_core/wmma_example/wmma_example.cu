/*
 * Level 4: WMMA Example — 16x16x16 matrix multiply using Tensor Cores.
 * Single 16x16 C block = A(16x16) * B(16x16). Demonstrates load_matrix_sync, mma_sync, store_matrix_sync.
 * Build: nvcc -arch=sm_70 -o wmma_example wmma_example.cu
 */
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(1); } \
} while(0)

using namespace nvcuda;
const int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

__global__ void wmma_matmul_16x16(const half* A, const half* B, float* C) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.f);
    wmma::load_matrix_sync(a_frag, A, WMMA_K);
    wmma::load_matrix_sync(b_frag, B, WMMA_N);
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    wmma::store_matrix_sync(C, acc_frag, WMMA_N, wmma::mem_row_major);
}

int main() {
    const int M = 16, N = 16, K = 16;
    size_t sz = M * K * sizeof(half);
    half *d_A, *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sz)); CUDA_CHECK(cudaMalloc(&d_B, sz)); CUDA_CHECK(cudaMalloc(&d_C, M*N*sizeof(float)));
    half *h_A = (half*)malloc(sz), *h_B = (half*)malloc(sz);
    float *h_C = (float*)malloc(M*N*sizeof(float));
    for (int i = 0; i < M*K; i++) h_A[i] = __float2half(1.f/(i+1));
    for (int i = 0; i < K*N; i++) h_B[i] = __float2half(1.f/(i+1));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice));

    wmma_matmul_16x16<<<1, 32>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost));
    printf("WMMA 16x16x16 sample C[0][0]=%.6f\n", h_C[0]);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
