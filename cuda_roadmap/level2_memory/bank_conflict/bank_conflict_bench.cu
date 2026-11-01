/*
 * Level 2: Shared-memory bank conflicts, shown with the classic tiled matrix transpose.
 *
 * Shared memory has 32 banks of 4 bytes; addresses addr and addr + 128 B hit the same bank.
 * A 32x32 float tile stores row r at offset 32*r, so a warp reading one COLUMN
 * (tile[0][c], tile[1][c], ..., tile[31][c]) hits the same bank 32 times -> 32-way conflict,
 * serialised into 32 shared-memory transactions.
 *
 *   naive:     __shared__ float tile[32][32]   -> column reads are 32-way conflicted
 *   optimized: __shared__ float tile[32][33]   -> one padding word shifts each row by one bank,
 *                                                 column reads hit 32 distinct banks
 *
 * Both kernels read and write global memory fully coalesced (that is why the transpose goes
 * through shared memory at all), so the only difference is the bank layout.
 * Verify with Nsight Compute: l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
 */
#include "../../common/bench_common.cuh"

const int N = 4096;  // N x N matrix
const int TILE = 32;
const int ROWS = 8;  // block = 32 x 8 threads, each thread moves 4 elements

template <int PAD>
__global__ void transpose_tiled(const float* __restrict__ in, float* __restrict__ out, int n) {
    __shared__ float tile[TILE][TILE + PAD];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += ROWS)  // coalesced read of a row of the input tile
        if (x < n && y + j < n) tile[threadIdx.y + j][threadIdx.x] = in[(size_t)(y + j) * n + x];
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;  // swap block coordinates for the output
    y = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += ROWS)  // column read from shared memory, coalesced global write
        if (x < n && y + j < n) out[(size_t)(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];
}

int main() {
    const size_t bytes = (size_t)N * N * sizeof(float);
    auto h = bench::random_vector((size_t)N * N, 5);
    std::vector<double> ref((size_t)N * N);
    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++) ref[(size_t)c * N + r] = h[(size_t)r * N + c];

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes)); CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, ROWS), grid(N / TILE, N / TILE);
    auto conflicted = [&] { transpose_tiled<0><<<grid, block>>>(d_in, d_out, N); };
    auto padded = [&] { transpose_tiled<1><<<grid, block>>>(d_in, d_out, N); };

    bool ok = true;
    CUDA_CHECK(cudaMemset(d_out, 0, bytes)); conflicted(); ok &= bench::verify("NAIVE", d_out, ref, 0, 0);
    CUDA_CHECK(cudaMemset(d_out, 0, bytes)); padded();     ok &= bench::verify("OPTIMIZED", d_out, ref, 0, 0);

    const float naive_ms = bench::time_ms(conflicted), opt_ms = bench::time_ms(padded);
    std::printf("CUDA_NAIVE_MS=%.6f\nCUDA_OPTIMIZED_MS=%.6f\n", naive_ms, opt_ms);
    bench::print_gbps("NAIVE_GBPS", 2.0 * bytes, naive_ms);
    bench::print_gbps("OPTIMIZED_GBPS", 2.0 * bytes, opt_ms);
    std::printf("N=%d\n", N);

    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
