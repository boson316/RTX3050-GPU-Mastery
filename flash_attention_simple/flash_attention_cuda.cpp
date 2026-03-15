#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" void flash_attention_cuda_launch(
    const float* Q,
    const float* K,
    const float* V,
    float* out,
    int B,
    int H,
    int S,
    int D,
    float scale,
    cudaStream_t stream
);

torch::Tensor flash_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    double scale
) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda());
    TORCH_CHECK(Q.dtype() == torch::kFloat32);
    auto out = torch::empty_like(Q);
    int B = Q.size(0), H = Q.size(1), S = Q.size(2), D = Q.size(3);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    flash_attention_cuda_launch(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        out.data_ptr<float>(),
        B, H, S, D,
        (float)scale,
        stream
    );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_forward, "FlashAttention forward (CUDA)");
}
