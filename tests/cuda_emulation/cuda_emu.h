// Minimal CPU emulation of the CUDA execution model, for checking kernel *logic*
// (indexing, masking, barrier placement, online-softmax math) without a GPU.
//
// - Every CUDA thread of a block runs as a std::thread.
// - __syncthreads() is a std::barrier over the block's threads, so a kernel that
//   lets some threads skip a barrier deadlocks here instead of "usually working".
// - __shared__ becomes a function-local `static`, i.e. one instance shared by all
//   emulated threads; blocks run one at a time, so that matches per-block smem.
//
// It does not model warps, memory coalescing or timing - it is a correctness tool only.
#pragma once
#include <barrier>
#include <math.h>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <algorithm>
#include <thread>
#include <vector>

struct Dim3 { unsigned x = 1, y = 1, z = 1; };
inline thread_local Dim3 threadIdx, blockIdx;
inline Dim3 blockDim, gridDim;
inline std::barrier<>* g_block_barrier = nullptr;

// Warp shuffles: each warp gets an exchange buffer and its own barrier. A shuffle with
// FULL_MASK requires all 32 lanes to arrive - if only some lanes call it (the classic
// `if (threadIdx.x < 8) __shfl_xor_sync(0xffffffff, ...)` bug) the emulator deadlocks,
// which the test driver reports as a failure via a timeout.
struct WarpState {
    std::barrier<>* bar;
    float buf[32];
};
inline std::vector<WarpState>* g_warps = nullptr;

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __restrict__
#define __launch_bounds__(...)
#define __shared__ static
#define CUDART_INF_F std::numeric_limits<float>::infinity()

inline void __syncthreads() { g_block_barrier->arrive_and_wait(); }

inline float __shfl_xor_sync(unsigned /*mask*/, float v, int lane_mask) {
    const unsigned tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    WarpState& w = (*g_warps)[tid / 32];
    const unsigned lane = tid % 32;
    w.buf[lane] = v;
    w.bar->arrive_and_wait();
    const float out = w.buf[lane ^ static_cast<unsigned>(lane_mask)];
    w.bar->arrive_and_wait();  // nobody overwrites buf before everyone has read
    return out;
}

// FP16 storage is modelled as float: the tests check algorithmic correctness, not rounding.
using __half = float;
inline float __half2float(__half h) { return h; }
inline __half __float2half(float f) { return f; }
inline float __expf(float x) { return std::exp(x); }
inline float rsqrtf(float x) { return 1.f / std::sqrt(x); }
// exp2f, tanhf, fmaf, fmaxf come from <math.h> (defining them here breaks MSVC, where the
// UCRT already provides inline definitions).
using std::min;

// Launch `kernel(args...)` over grid x block, one block at a time.
template <typename F>
void emu_launch(Dim3 grid, Dim3 block, F&& kernel) {
    gridDim = grid;
    blockDim = block;
    const unsigned nthreads = block.x * block.y * block.z;
    for (unsigned bz = 0; bz < grid.z; ++bz)
        for (unsigned by = 0; by < grid.y; ++by)
            for (unsigned bx = 0; bx < grid.x; ++bx) {
                std::barrier<> bar(nthreads);
                g_block_barrier = &bar;
                const unsigned nwarps = (nthreads + 31) / 32;
                std::vector<std::unique_ptr<std::barrier<>>> warp_bars;
                std::vector<WarpState> warps(nwarps);
                for (unsigned w = 0; w < nwarps; ++w) {
                    const unsigned lanes = std::min(32u, nthreads - w * 32);
                    warp_bars.emplace_back(std::make_unique<std::barrier<>>(lanes));
                    warps[w].bar = warp_bars.back().get();
                }
                g_warps = &warps;
                std::vector<std::thread> threads;
                threads.reserve(nthreads);
                for (unsigned t = 0; t < nthreads; ++t) {
                    threads.emplace_back([&, t] {
                        blockIdx = {bx, by, bz};
                        threadIdx = {t % block.x, (t / block.x) % block.y, t / (block.x * block.y)};
                        kernel();
                    });
                }
                for (auto& th : threads) th.join();
                g_warps = nullptr;  // the block's barriers die at the end of this iteration
                g_block_barrier = nullptr;
            }
}
