/*
 * Level 3: Block reduction - shared-memory tree vs warp shuffle (N = 2^24 floats).
 *   naive:     8-step tree in shared memory, __syncthreads() after every step
 *   optimized: grid-stride accumulation in registers (each thread sums many elements),
 *              then __shfl_down_sync inside each warp (no shared memory, no block barrier),
 *              one shared-memory exchange between warps, one atomicAdd per block.
 * The optimized kernel launches only enough blocks to fill the GPU, which removes most
 * of the per-block overhead; it should run close to DRAM bandwidth.
 */
#include "../../common/bench_common.cuh"

const int N = 1 << 24;
#define BLOCK 256

__global__ void reduce_shared(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK + tid;
    sdata[tid] = (i < n) ? input[i] : 0.f;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(output, sdata[0]);
}

__device__ __forceinline__ float warp_reduce(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void reduce_warp_shuffle(const float* input, float* output, int n) {
    __shared__ float warp_sums[BLOCK / 32];
    float val = 0.f;
    for (int i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += gridDim.x * BLOCK) val += input[i];
    const int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    val = warp_reduce(val);  // all 32 lanes of every warp participate
    if (lane == 0) warp_sums[wid] = val;
    __syncthreads();
    if (wid == 0) {  // the WHOLE first warp runs the second stage (full mask is valid)
        val = lane < BLOCK / 32 ? warp_sums[lane] : 0.f;
        val = warp_reduce(val);
        if (lane == 0) atomicAdd(output, val);
    }
}

int main() {
    auto h_in = bench::random_vector(N, 9, 0.f, 1.f);
    double sum = 0; for (float v : h_in) sum += v;
    const std::vector<double> ref{sum};

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    int num_sms = 0, blocks_per_sm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, reduce_warp_shuffle, BLOCK, 0));
    const int grid_shfl = num_sms * blocks_per_sm;

    auto naive = [&] { cudaMemsetAsync(d_out, 0, sizeof(float)); reduce_shared<<<(N + BLOCK - 1) / BLOCK, BLOCK>>>(d_in, d_out, N); };
    auto opt = [&] { cudaMemsetAsync(d_out, 0, sizeof(float)); reduce_warp_shuffle<<<grid_shfl, BLOCK>>>(d_in, d_out, N); };

    bool ok = true;
    naive(); ok &= bench::verify("NAIVE", d_out, ref, 0.0, 1e-4);
    opt();   ok &= bench::verify("OPTIMIZED", d_out, ref, 0.0, 1e-4);

    const float naive_ms = bench::time_ms(naive), opt_ms = bench::time_ms(opt);
    std::printf("CUDA_NAIVE_MS=%.6f\nCUDA_OPTIMIZED_MS=%.6f\n", naive_ms, opt_ms);
    bench::print_gbps("NAIVE_GBPS", (double)N * sizeof(float), naive_ms);
    bench::print_gbps("OPTIMIZED_GBPS", (double)N * sizeof(float), opt_ms);
    std::printf("N=%d GRID_OPTIMIZED=%d\n", N, grid_shfl);

    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
