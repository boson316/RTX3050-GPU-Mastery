/*
 * Level 3: Grid-stride ("persistent-style") kernel vs one-thread-per-element.
 *   naive:      N/256 blocks, each thread handles one element
 *   persistent: exactly as many blocks as can be resident at once
 *               (SM count x max active blocks per SM, from the occupancy API);
 *               each thread loops over the array with stride gridDim.x * blockDim.x.
 * For a memory-bound op the win is small (fewer block launches / index computations);
 * the pattern matters more for kernels with per-block setup cost, and it makes the
 * kernel correct for any N with a fixed grid.
 */
#include "../../common/bench_common.cuh"

const int N = 1 << 24;
const int BLOCK = 256;

__global__ void vector_add_naive(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

__global__ void vector_add_persistent(const float* A, const float* B, float* C, int n) {
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) C[i] = A[i] + B[i];
}

int main() {
    const size_t bytes = (size_t)N * sizeof(float);
    auto h_A = bench::random_vector(N, 41), h_B = bench::random_vector(N, 42);
    std::vector<double> ref(N);
    for (int i = 0; i < N; i++) ref[i] = h_A[i] + h_B[i];

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes)); CUDA_CHECK(cudaMalloc(&d_B, bytes)); CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    int num_sms = 0, blocks_per_sm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, vector_add_persistent, BLOCK, 0));
    const int grid_persistent = num_sms * blocks_per_sm;

    auto naive = [&] { vector_add_naive<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(d_A, d_B, d_C, N); };
    auto persistent = [&] { vector_add_persistent<<<grid_persistent, BLOCK>>>(d_A, d_B, d_C, N); };

    bool ok = true;
    CUDA_CHECK(cudaMemset(d_C, 0, bytes)); naive();      ok &= bench::verify("NAIVE", d_C, ref, 0, 0);
    CUDA_CHECK(cudaMemset(d_C, 0, bytes)); persistent(); ok &= bench::verify("OPTIMIZED", d_C, ref, 0, 0);

    const float naive_ms = bench::time_ms(naive), opt_ms = bench::time_ms(persistent);
    std::printf("CUDA_NAIVE_MS=%.6f\nCUDA_OPTIMIZED_MS=%.6f\n", naive_ms, opt_ms);
    bench::print_gbps("NAIVE_GBPS", 3.0 * bytes, naive_ms);
    bench::print_gbps("OPTIMIZED_GBPS", 3.0 * bytes, opt_ms);
    std::printf("N=%d GRID_PERSISTENT=%d SMS=%d\n", N, grid_persistent, num_sms);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return ok ? 0 : 1;
}
