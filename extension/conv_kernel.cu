/*
 * PyTorch CUDA Extension: custom conv2d 3x3 (in_ch=1, out_ch=32, no padding)
 * 優化：FP16 + 16x16 tile、shared tile_in[18][18]、tile_w[32][9]、
 *      一 block 產出 32 通道，減少 global 讀取與 block 數。
 * 目標：B=1024 約 2x torch（RTX 3050 sm_86）
 * Profile: nsight-sys profile python mnist_custom_conv.py → 看 occupancy / warp stall
 */

#include <torch/extension.h>
#include <cuda_fp16.h>

#define TILE_SIZE 16
#define K 3
#define SHARD_H (TILE_SIZE + K - 1)  // 18
#define SHARD_W (TILE_SIZE + K - 1)  // 18
#define C_OUT 32

// FP32 path（保留做對照 / fallback）
template <typename scalar_t>
__global__ void optimized_conv2d_fp32_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch,
    const int in_h,
    const int in_w,
    const int out_ch,
    const int out_h,
    const int out_w) {
    __shared__ scalar_t tile[SHARD_H * SHARD_W];
    __shared__ scalar_t sh_weight[9];

    int tx = threadIdx.x, ty = threadIdx.y;
    int oh = blockIdx.y * TILE_SIZE;
    int ow = blockIdx.x * TILE_SIZE;
    int b = blockIdx.z % batch;
    int oc = blockIdx.z / batch;
    int ih_base = oh, iw_base = ow;

    for (int idx = (ty * blockDim.x + tx); idx < SHARD_H * SHARD_W; idx += blockDim.x * blockDim.y) {
        int i = idx / SHARD_W, j = idx % SHARD_W;
        int ih = ih_base + i, iw = iw_base + j;
        tile[idx] = (ih < in_h && iw < in_w)
            ? input[(b * in_h + ih) * in_w + iw]
            : 0;
    }
    if (ty * blockDim.x + tx < 9)
        sh_weight[ty * blockDim.x + tx] = weight[oc * 9 + (ty * blockDim.x + tx)];
    __syncthreads();

    int oy = oh + ty, ox = ow + tx;
    if (oy < out_h && ox < out_w) {
        scalar_t sum = bias[oc];
#pragma unroll
        for (int kh = 0; kh < K; ++kh)
#pragma unroll
            for (int kw = 0; kw < K; ++kw)
                sum += tile[(ty + kh) * SHARD_W + (tx + kw)] * sh_weight[kh * K + kw];
        output[(b * out_ch + oc) * (out_h * out_w) + oy * out_w + ox] = sum;
    }
}

// 權重以 __half2 存 2 通道一組，共 16 組
#define C_OUT_H2 (C_OUT / 2)

// FP16 path（float 介面）：kernel 內轉 half，用 __half2 一次算 2 通道
__global__ void optimized_conv2d_fp16_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w) {
    __shared__ __half tile_in[SHARD_H][SHARD_W];
    __shared__ __half2 tile_w[C_OUT_H2][9];  // 16 組 x 9，每組 2 通道

    int tx = threadIdx.x, ty = threadIdx.y;
    int oh = blockIdx.y * TILE_SIZE;
    int ow = blockIdx.x * TILE_SIZE;
    int b = blockIdx.z;
    int ih_base = oh, iw_base = ow;

    for (int idx = ty * blockDim.x + tx; idx < SHARD_H * SHARD_W; idx += blockDim.x * blockDim.y) {
        int i = idx / SHARD_W, j = idx % SHARD_W;
        int ih = ih_base + i, iw = iw_base + j;
        float v = (ih < in_h && iw < in_w)
            ? input[(b * in_h + ih) * in_w + iw]
            : 0.f;
        tile_in[i][j] = __float2half_rn(v);
    }
    for (int idx = ty * blockDim.x + tx; idx < C_OUT_H2 * 9; idx += blockDim.x * blockDim.y) {
        int g = idx / 9, k = idx % 9;
        __half2 w2 = __halves2half2(
            __float2half_rn(weight[(g * 2 + 0) * 9 + k]),
            __float2half_rn(weight[(g * 2 + 1) * 9 + k]));
        tile_w[g][k] = w2;
    }
    __syncthreads();

    int oy = oh + ty, ox = ow + tx;
    if (oy < out_h && ox < out_w) {
        const int out_st = out_h * out_w;
        const int base = (b * C_OUT) * out_st + oy * out_w + ox;
        for (int g = 0; g < C_OUT_H2; ++g) {
            __half2 sum2 = __halves2half2(
                __float2half_rn(bias[g * 2]),
                __float2half_rn(bias[g * 2 + 1]));
#pragma unroll
            for (int kh = 0; kh < K; ++kh)
#pragma unroll
                for (int kw = 0; kw < K; ++kw) {
                    __half in_h = tile_in[ty + kh][tx + kw];
                    sum2 = __hadd2(sum2, __hmul2(__halves2half2(in_h, in_h), tile_w[g][kh * K + kw]));
                }
            output[base + (g * 2 + 0) * out_st] = __half2float(__low2half(sum2));
            output[base + (g * 2 + 1) * out_st] = __half2float(__high2half(sum2));
        }
    }
}

// 純 FP16 路徑：global 已是 half，無轉換、頻寬減半，適合 .half() 測速
__global__ void optimized_conv2d_fp16_pure_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    const int batch,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w) {
    __shared__ __half tile_in[SHARD_H][SHARD_W];
    __shared__ __half2 tile_w[C_OUT_H2][9];

    int tx = threadIdx.x, ty = threadIdx.y;
    int oh = blockIdx.y * TILE_SIZE;
    int ow = blockIdx.x * TILE_SIZE;
    int b = blockIdx.z;
    int ih_base = oh, iw_base = ow;

    for (int idx = ty * blockDim.x + tx; idx < SHARD_H * SHARD_W; idx += blockDim.x * blockDim.y) {
        int i = idx / SHARD_W, j = idx % SHARD_W;
        int ih = ih_base + i, iw = iw_base + j;
        tile_in[i][j] = (ih < in_h && iw < in_w)
            ? input[(b * in_h + ih) * in_w + iw]
            : __float2half_rn(0.f);
    }
    for (int idx = ty * blockDim.x + tx; idx < C_OUT_H2 * 9; idx += blockDim.x * blockDim.y) {
        int g = idx / 9, k = idx % 9;
        tile_w[g][k] = __halves2half2(
            weight[(g * 2 + 0) * 9 + k],
            weight[(g * 2 + 1) * 9 + k]);
    }
    __syncthreads();

    int oy = oh + ty, ox = ow + tx;
    if (oy < out_h && ox < out_w) {
        const int out_st = out_h * out_w;
        const int base = (b * C_OUT) * out_st + oy * out_w + ox;
        for (int g = 0; g < C_OUT_H2; ++g) {
            __half2 sum2 = __halves2half2(bias[g * 2], bias[g * 2 + 1]);
#pragma unroll
            for (int kh = 0; kh < K; ++kh)
#pragma unroll
                for (int kw = 0; kw < K; ++kw) {
                    __half in_h = tile_in[ty + kh][tx + kw];
                    sum2 = __hadd2(sum2, __hmul2(__halves2half2(in_h, in_h), tile_w[g][kh * K + kw]));
                }
            output[base + (g * 2 + 0) * out_st] = __low2half(sum2);
            output[base + (g * 2 + 1) * out_st] = __high2half(sum2);
        }
    }
}

std::vector<torch::Tensor> custom_conv2d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda());
    TORCH_CHECK(input.dim() == 4 && weight.dim() == 4 && bias.dim() == 1);
    int batch = input.size(0);
    int in_h = input.size(2), in_w = input.size(3);
    int out_ch = weight.size(0);
    int out_h = in_h - 2, out_w = in_w - 2;
    TORCH_CHECK(out_ch == C_OUT, "This kernel expects out_ch=32");

    auto output = torch::zeros({batch, out_ch, out_h, out_w}, input.options());

    const int T = TILE_SIZE;
    dim3 threads(T, T);
    dim3 blocks(
        (out_w + T - 1) / T,
        (out_h + T - 1) / T,
        batch);

    // 先處理 Half，避免落入 AT_DISPATCH_FLOATING_TYPES（僅支援 Float/Double）
    if (input.scalar_type() == torch::kFloat16 || input.scalar_type() == c10::kHalf) {
        optimized_conv2d_fp16_pure_kernel<<<blocks, threads>>>(
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            batch, in_h, in_w, out_h, out_w);
        return {output};
    }

    if (input.scalar_type() == torch::kFloat32) {
        optimized_conv2d_fp16_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            output.data_ptr<float>(),
            batch, in_h, in_w, out_h, out_w);
    } else {
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "custom_conv2d", ([&] {
            optimized_conv2d_fp32_kernel<scalar_t><<<
                dim3(blocks.x, blocks.y, batch * out_ch),
                threads>>>(
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch, in_h, in_w, out_ch, out_h, out_w);
        }));
    }

    return {output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_conv2d", &custom_conv2d, "Custom conv2d 3x3 (CUDA, FP16+16x16 tile, sm_86)");
}
