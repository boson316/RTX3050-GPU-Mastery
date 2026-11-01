/*
 * PyTorch CUDA extension: custom 3x3 conv2d (C_in = 1, C_out = 32, no padding), FP32 or FP16.
 * Kernel: conv3x3_kernel.cuh.
 */
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "conv3x3_kernel.cuh"

std::vector<torch::Tensor> custom_conv2d(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(input.dim() == 4 && input.size(1) == 1, "input must be (B, 1, H, W)");
    TORCH_CHECK(weight.sizes() == torch::IntArrayRef({conv3x3::C_OUT, 1, 3, 3}), "weight must be (32, 1, 3, 3)");
    TORCH_CHECK(bias.numel() == conv3x3::C_OUT, "bias must have 32 elements");
    TORCH_CHECK(input.scalar_type() == weight.scalar_type() && input.scalar_type() == bias.scalar_type(),
                "input/weight/bias dtypes must match");
    TORCH_CHECK(input.size(2) >= 3 && input.size(3) >= 3, "spatial size must be >= 3");

    const at::cuda::CUDAGuard guard(input.device());
    auto x = input.contiguous(), w = weight.contiguous(), b = bias.contiguous();
    const int batch = x.size(0), in_h = x.size(2), in_w = x.size(3);
    auto output = torch::empty({batch, conv3x3::C_OUT, in_h - 2, in_w - 2}, x.options());

    const dim3 threads(conv3x3::TILE, conv3x3::TILE);
    const dim3 blocks((in_w - 2 + conv3x3::TILE - 1) / conv3x3::TILE, (in_h - 2 + conv3x3::TILE - 1) / conv3x3::TILE,
                      batch);
    auto stream = at::cuda::getCurrentCUDAStream();

    if (x.scalar_type() == torch::kFloat16) {
        conv3x3::conv3x3_c32_kernel<__half><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(w.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()), in_h, in_w);
    } else {
        TORCH_CHECK(x.scalar_type() == torch::kFloat32, "only float32 and float16 are supported");
        conv3x3::conv3x3_c32_kernel<float><<<blocks, threads, 0, stream>>>(
            x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), output.data_ptr<float>(), in_h, in_w);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_conv2d", &custom_conv2d, "3x3 conv2d, C_in=1, C_out=32 (CUDA, FP32/FP16 storage, FP32 accumulate)");
}
