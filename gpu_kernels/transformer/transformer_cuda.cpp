/*
 * PyTorch bindings for transformer CUDA kernels (FP16).
 */
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>

extern "C" {
void fused_qkv_fp16_launch(
    const __half* x, const __half* w, const __half* b, __half* y,
    int M, int K, int N, cudaStream_t stream);
void softmax_fp16_launch(
    const __half* x, __half* y, int stride, int N, int num_rows, cudaStream_t stream);
void layernorm_fp16_launch(
    const __half* x, const __half* weight, const __half* bias, __half* y,
    int stride, int N, int num_rows, float eps, cudaStream_t stream);
void gelu_fp16_launch(const __half* x, __half* y, int n, cudaStream_t stream);
void fused_mlp_fp16_launch(
    const __half* x,
    const __half* w1, const __half* b1,
    const __half* w2, const __half* b2,
    __half* mid, __half* out,
    int M, int K, int N1, int N2,
    cudaStream_t stream);
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_HALF(x) TORCH_CHECK(x.dtype() == torch::kFloat16, #x " must be float16")

torch::Tensor fused_qkv_cuda(
    torch::Tensor x,
    torch::Tensor w,
    c10::optional<torch::Tensor> b
) {
    CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_HALF(x);
    CHECK_CUDA(w); CHECK_CONTIGUOUS(w); CHECK_HALF(w);
    int M = x.size(0), K = x.size(1), N = w.size(1);
    TORCH_CHECK(w.size(0) == K);
    auto y = torch::empty({M, N}, x.options());
    const __half* b_ptr = (b.has_value() && b->defined())
        ? reinterpret_cast<const __half*>(b->data_ptr<at::Half>()) : nullptr;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    fused_qkv_fp16_launch(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(w.data_ptr<at::Half>()),
        b_ptr,
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        M, K, N, stream);
    return y;
}

torch::Tensor softmax_cuda(torch::Tensor x, int dim) {
    CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_HALF(x);
    TORCH_CHECK(dim == -1 || dim == x.dim() - 1, "Only last-dim softmax supported");
    auto y = torch::empty_like(x);
    auto x_flat = x.reshape({-1, x.size(-1)});
    auto y_flat = y.reshape({-1, x.size(-1)});
    int num_rows = x_flat.size(0), N = x_flat.size(1);
    int stride = x_flat.stride(0);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softmax_fp16_launch(
        reinterpret_cast<const __half*>(x_flat.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y_flat.data_ptr<at::Half>()),
        stride, N, num_rows, stream);
    return y;
}

torch::Tensor layernorm_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_HALF(x);
    CHECK_CUDA(weight); CHECK_CUDA(bias);
    auto y = torch::empty_like(x);
    auto x_flat = x.reshape({-1, x.size(-1)});
    auto y_flat = y.reshape({-1, x.size(-1)});
    int num_rows = x_flat.size(0), N = x_flat.size(1);
    int stride = x_flat.stride(0);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    layernorm_fp16_launch(
        reinterpret_cast<const __half*>(x_flat.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y_flat.data_ptr<at::Half>()),
        stride, N, num_rows, (float)eps, stream);
    return y;
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_HALF(x);
    auto y = torch::empty_like(x);
    int n = x.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    gelu_fp16_launch(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        n, stream);
    return y;
}

torch::Tensor fused_mlp_cuda(
    torch::Tensor x,
    torch::Tensor w1, torch::Tensor b1,
    torch::Tensor w2, torch::Tensor b2
) {
    CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_HALF(x);
    int M = x.size(0), K = x.size(1), N1 = w1.size(1), N2 = w2.size(1);
    TORCH_CHECK(w1.size(0) == K && w2.size(0) == N1);
    auto mid = torch::empty({M, N1}, x.options());
    auto out = torch::empty({M, N2}, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    fused_mlp_fp16_launch(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(w1.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(b1.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(w2.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(b2.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(mid.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        M, K, N1, N2, stream);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_qkv", &fused_qkv_cuda, "Fused QKV projection (FP16)");
    m.def("softmax", &softmax_cuda, "Softmax last dim (FP16)");
    m.def("layernorm", &layernorm_cuda, "LayerNorm (FP16)");
    m.def("gelu", &gelu_cuda, "GELU (FP16)");
    m.def("fused_mlp", &fused_mlp_cuda, "Fused MLP block (FP16)");
}
