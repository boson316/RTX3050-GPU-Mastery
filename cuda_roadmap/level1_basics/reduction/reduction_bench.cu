/*
 * Level 1: Reduction — Naive (atomicAdd) vs Optimized (shared memory tree)
 * Outputs: CUDA_NAIVE_MS=... CUDA_OPTIMIZED_MS=...
 */
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(1); } \
} while(0)

const int N = 1 << 20;
const int BLOCK = 256;

// ---- Naive: every thread atomicAdd to a single global location (very slow) ----
__global__ void reduce_naive(const float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        atomicAdd(output, input[i]);
}

// ---- Optimized: tree reduction in shared memory, one value per block ----
__global__ void reduce_optimized(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK + tid;
    sdata[tid] = (i < n) ? input[i] : 0.f;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

static float run_naive(float* d_in, float* d_out) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    CUDA_CHECK(cudaEventRecord(start, 0));
    reduce_naive<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms;
}

static float run_optimized(float* d_in, float* d_out) {
    int nBlocks = (N + BLOCK - 1) / BLOCK;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    reduce_optimized<<<nBlocks, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms;
}

int main() {
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)((N + BLOCK - 1) / BLOCK) * sizeof(float)));
    float *h_in = (float*)malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_in, h_in, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    float naive_ms = run_naive(d_in, d_out);
    float opt_ms  = run_optimized(d_in, d_out);

    printf("CUDA_NAIVE_MS=%.6f\n", naive_ms);
    printf("CUDA_OPTIMIZED_MS=%.6f\n", opt_ms);
    printf("N=%d\n", N);

    cudaFree(d_in); cudaFree(d_out); free(h_in);
    return 0;
}
