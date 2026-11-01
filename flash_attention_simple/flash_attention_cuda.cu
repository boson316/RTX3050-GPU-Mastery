/*
 * PyTorch-facing launcher for the FlashAttention forward kernel.
 * The kernel itself lives in flash_attention_kernel.cuh (shared with the standalone benchmark).
 */
#include <cuda_runtime.h>
#include "flash_attention_kernel.cuh"

extern "C" cudaError_t flash_attention_cuda_launch(
    const float* Q, const float* K, const float* V, float* out,
    int B, int H, int S, int D, float scale, bool causal, cudaStream_t stream)
{
    return flash_attn::launch_flash_attention_fwd(Q, K, V, out, B, H, S, D, scale, causal, stream);
}
