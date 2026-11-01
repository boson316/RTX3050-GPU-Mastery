#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

extern "C" cudaError_t flash_attention_cuda_launch(
    const float* Q, const float* K, const float* V, float* out,
    int B, int H, int S, int D, float scale, bool causal, cudaStream_t stream);

torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                      double scale, bool causal) {
    for (const auto& t : {Q, K, V}) {
        TORCH_CHECK(t.is_cuda(), "inputs must be CUDA tensors");
        TORCH_CHECK(t.scalar_type() == torch::kFloat32, "inputs must be float32");
        TORCH_CHECK(t.dim() == 4, "inputs must be (B, H, S, D)");
    }
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(), "Q, K, V must have the same shape");
    TORCH_CHECK(Q.size(3) <= 64, "head_dim must be <= 64");

    const at::cuda::CUDAGuard guard(Q.device());
    auto q = Q.contiguous(), k = K.contiguous(), v = V.contiguous();
    auto out = torch::empty_like(q);
    const int B = q.size(0), H = q.size(1), S = q.size(2), D = q.size(3);
    cudaError_t err = flash_attention_cuda_launch(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<float>(),
        B, H, S, D, static_cast<float>(scale), causal, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(err == cudaSuccess, "flash_attention kernel launch failed: ", cudaGetErrorString(err));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_forward, "FlashAttention forward (CUDA, FP32)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"), py::arg("causal") = false);
}
