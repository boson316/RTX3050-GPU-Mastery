/*
 * PyTorch bindings for the transformer CUDA kernels (FP16).
 *
 * GEMMs (QKV projection, MLP) use cuBLAS through ATen's per-stream handle.
 *
 * Row-major GEMM with column-major cuBLAS:
 *   We want Y (M x N) = X (M x K) @ W (K x N), all row-major.
 *   A row-major matrix is the transpose of the same buffer read column-major, so
 *   Y^T (N x M) = W^T (N x K) @ X^T (K x M) where W^T is just `w` with ld = N and
 *   X^T is `x` with ld = K. One cublasGemmEx(N, N, m=N, n=M, k=K) writes row-major Y
 *   directly - no transpose kernel, no temp buffer.
 *
 *   The previous version called cublasGemmEx(T, N, M, N, K, x, K, w, N, ...), i.e. it
 *   read `w` as a column-major K x N matrix with ld = N. For the QKV shape (N = 3K) that
 *   indexes up to w[(K-1) + (N-1)*N] ~ 9K^2 elements into a 3K^2 buffer: an out-of-bounds
 *   read that produced wrong results, plus a col->row transpose kernel and a global temp
 *   buffer that was never freed.
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern "C" {
cudaError_t softmax_fp16_launch(const __half* x, __half* y, int n_cols, int n_rows, cudaStream_t stream);
cudaError_t layernorm_fp16_launch(const __half* x, const __half* gamma, const __half* beta, __half* y,
                                  int n_cols, int n_rows, float eps, cudaStream_t stream);
cudaError_t gelu_fp16_launch(const __half* x, __half* y, size_t n, cudaStream_t stream);
cudaError_t bias_act_fp16_launch(__half* y, const __half* bias, size_t rows, int n_cols, bool gelu,
                                 cudaStream_t stream);
}

#define CHECK_INPUT(x)                                                         \
    TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor");                   \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous");                \
    TORCH_CHECK((x).scalar_type() == torch::kFloat16, #x " must be float16")

#define CHECK_LAUNCH(expr)                                                     \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        TORCH_CHECK(_e == cudaSuccess, #expr " failed: ", cudaGetErrorString(_e)); \
    } while (0)

namespace {

inline const __half* hptr(const torch::Tensor& t) { return reinterpret_cast<const __half*>(t.data_ptr<at::Half>()); }
inline __half* hptr_mut(torch::Tensor& t) { return reinterpret_cast<__half*>(t.data_ptr<at::Half>()); }

// y (M x N, row-major) = x (M x K) @ w (K x N); FP16 in/out, FP32 accumulate, Tensor Cores allowed.
void gemm_rowmajor_fp16(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y) {
    const int M = x.size(0), K = x.size(1), N = w.size(1);
    const float alpha = 1.f, beta = 0.f;
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();  // already bound to the current stream
    const cublasStatus_t st = cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        /*m=*/N, /*n=*/M, /*k=*/K,
        &alpha,
        hptr(w), CUDA_R_16F, /*lda=*/N,
        hptr(x), CUDA_R_16F, /*ldb=*/K,
        &beta,
        hptr_mut(y), CUDA_R_16F, /*ldc=*/N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasGemmEx failed with status ", static_cast<int>(st));
}

void check_bias(const c10::optional<torch::Tensor>& b, int64_t n) {
    if (b.has_value() && b->defined()) {
        CHECK_INPUT(*b);
        TORCH_CHECK(b->numel() == n, "bias must have ", n, " elements");
    }
}

const __half* opt_ptr(const c10::optional<torch::Tensor>& b) {
    return (b.has_value() && b->defined()) ? hptr(*b) : nullptr;
}

}  // namespace

torch::Tensor fused_qkv_cuda(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> b) {
    CHECK_INPUT(x); CHECK_INPUT(w);
    TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && w.size(0) == x.size(1), "expected x (M,K) and w (K,N)");
    check_bias(b, w.size(1));
    const at::cuda::CUDAGuard guard(x.device());
    auto y = torch::empty({x.size(0), w.size(1)}, x.options());
    gemm_rowmajor_fp16(x, w, y);
    if (opt_ptr(b))
        CHECK_LAUNCH(bias_act_fp16_launch(hptr_mut(y), opt_ptr(b), y.size(0), y.size(1), false,
                                          at::cuda::getCurrentCUDAStream()));
    return y;
}

torch::Tensor softmax_cuda(torch::Tensor x, int64_t dim) {
    CHECK_INPUT(x);
    TORCH_CHECK(dim == -1 || dim == x.dim() - 1, "only last-dim softmax is supported");
    const at::cuda::CUDAGuard guard(x.device());
    auto y = torch::empty_like(x);
    const int n_cols = x.size(-1);
    const int n_rows = n_cols ? x.numel() / n_cols : 0;
    CHECK_LAUNCH(softmax_fp16_launch(hptr(x), hptr_mut(y), n_cols, n_rows, at::cuda::getCurrentCUDAStream()));
    return y;
}

torch::Tensor layernorm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
    CHECK_INPUT(x); CHECK_INPUT(weight); CHECK_INPUT(bias);
    const int n_cols = x.size(-1);
    TORCH_CHECK(weight.numel() == n_cols && bias.numel() == n_cols, "weight/bias must match last dim");
    const at::cuda::CUDAGuard guard(x.device());
    auto y = torch::empty_like(x);
    const int n_rows = n_cols ? x.numel() / n_cols : 0;
    CHECK_LAUNCH(layernorm_fp16_launch(hptr(x), hptr(weight), hptr(bias), hptr_mut(y), n_cols, n_rows,
                                       static_cast<float>(eps), at::cuda::getCurrentCUDAStream()));
    return y;
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    CHECK_INPUT(x);
    const at::cuda::CUDAGuard guard(x.device());
    auto y = torch::empty_like(x);
    CHECK_LAUNCH(gelu_fp16_launch(hptr(x), hptr_mut(y), x.numel(), at::cuda::getCurrentCUDAStream()));
    return y;
}

// out = (GELU(x @ w1 + b1)) @ w2 + b2. Bias+GELU is fused into one epilogue pass over the
// intermediate instead of separate bias and activation kernels.
torch::Tensor fused_mlp_cuda(torch::Tensor x, torch::Tensor w1, torch::Tensor b1, torch::Tensor w2,
                             torch::Tensor b2) {
    CHECK_INPUT(x); CHECK_INPUT(w1); CHECK_INPUT(w2);
    TORCH_CHECK(x.dim() == 2 && w1.size(0) == x.size(1) && w2.size(0) == w1.size(1), "shape mismatch");
    check_bias(b1, w1.size(1));
    check_bias(b2, w2.size(1));
    const at::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto mid = torch::empty({x.size(0), w1.size(1)}, x.options());
    gemm_rowmajor_fp16(x, w1, mid);
    CHECK_LAUNCH(bias_act_fp16_launch(hptr_mut(mid), hptr(b1), mid.size(0), mid.size(1), true, stream));
    auto out = torch::empty({x.size(0), w2.size(1)}, x.options());
    gemm_rowmajor_fp16(mid, w2, out);
    CHECK_LAUNCH(bias_act_fp16_launch(hptr_mut(out), hptr(b2), out.size(0), out.size(1), false, stream));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_qkv", &fused_qkv_cuda, "QKV projection y = x @ w + b (cuBLAS + bias epilogue, FP16)",
          py::arg("x"), py::arg("w"), py::arg("b") = py::none());
    m.def("softmax", &softmax_cuda, "Online softmax over last dim (FP16)");
    m.def("layernorm", &layernorm_cuda, "LayerNorm, Welford statistics (FP16)");
    m.def("gelu", &gelu_cuda, "tanh-GELU (FP16)");
    m.def("fused_mlp", &fused_mlp_cuda, "MLP: GEMM -> fused bias+GELU -> GEMM -> bias (FP16)");
}
