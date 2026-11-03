/*
 * PyTorch bindings for the transformer CUDA kernels (FP16).
 *
 * GEMMs (QKV projection, MLP) use cublasLt with bias / bias+GELU fused into the GEMM epilogue.
 *
 * Row-major GEMM with column-major cuBLAS:
 *   We want Y (M x N) = X (M x K) @ W (K x N), all row-major.
 *   A row-major matrix is the transpose of the same buffer read column-major, so
 *   Y^T (N x M) = W^T (N x K) @ X^T (K x M) where W^T is just `w` with ld = N and
 *   X^T is `x` with ld = K. One cublasGemmEx(N, N, m=N, n=M, k=K) writes row-major Y
 *   directly - no transpose kernel, no temp buffer. cublasLt uses the same column-major view.
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
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <mutex>
#include <unordered_map>
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

// ---------------------------------------------------------------------------------------------
// cublasLt GEMM with a fused epilogue: y = act(x @ w + b), all row-major FP16, FP32 accumulate.
//
// Why: v2 measurements showed QKV 0.675 ms vs PyTorch 0.545 ms. The whole 0.13 ms gap is the
// separate bias kernel, which re-reads and re-writes the 2048 x 2304 FP16 output (18.9 MB at
// 132 GB/s = 0.14 ms). cublasLt applies bias (and GELU) in the GEMM epilogue while the tile is
// still in registers, the same thing PyTorch's addmm does - so the extra DRAM pass disappears.
//
// Row-major trick (see the file header): column-major D (N x M) = W (N x K) * X (K x M);
// the bias vector runs along D's rows, i.e. over the N output features, as required.
// CUBLASLT_EPILOGUE_GELU_BIAS applies cuBLAS's GELU (an approximation; it differs from the exact
// erf form by < 1e-3, well inside FP16 tolerance). tests/test_cuda_extensions.py checks it on the GPU.
// ---------------------------------------------------------------------------------------------
#define CHECK_LT(expr)                                                                   \
    do {                                                                                 \
        cublasStatus_t _s = (expr);                                                      \
        TORCH_CHECK(_s == CUBLAS_STATUS_SUCCESS, #expr " failed with status ", static_cast<int>(_s)); \
    } while (0)

cublasLtHandle_t lt_handle() {
    // One handle per device, created lazily. cublasLt handles are not tied to a stream.
    static std::mutex mu;
    static std::unordered_map<int, cublasLtHandle_t> handles;
    int dev = 0;
    cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lock(mu);
    auto it = handles.find(dev);
    if (it != handles.end()) return it->second;
    cublasLtHandle_t h = nullptr;
    CHECK_LT(cublasLtCreate(&h));
    handles[dev] = h;
    return h;
}

struct LtDescs {  // RAII for the cublasLt descriptors of one call
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    ~LtDescs() {
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (d) cublasLtMatrixLayoutDestroy(d);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (op) cublasLtMatmulDescDestroy(op);
    }
};

void gemm_bias_act_lt(const torch::Tensor& x, const torch::Tensor& w, const __half* bias, bool gelu,
                      torch::Tensor& y) {
    const int64_t M = x.size(0), K = x.size(1), N = w.size(1);
    const float alpha = 1.f, beta = 0.f;
    LtDescs d;
    CHECK_LT(cublasLtMatmulDescCreate(&d.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t no_trans = CUBLAS_OP_N;
    CHECK_LT(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_TRANSA, &no_trans, sizeof(no_trans)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_TRANSB, &no_trans, sizeof(no_trans)));
    cublasLtEpilogue_t epi = gelu ? (bias ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_GELU)
                                  : (bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT);
    CHECK_LT(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
    if (bias) {
        const cudaDataType_t bias_type = CUDA_R_16F;
        CHECK_LT(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
        CHECK_LT(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type,
                                                sizeof(bias_type)));
    }
    // Column-major views of the row-major buffers: W -> N x K (ld N), X -> K x M (ld K), Y -> N x M (ld N).
    CHECK_LT(cublasLtMatrixLayoutCreate(&d.a, CUDA_R_16F, N, K, N));
    CHECK_LT(cublasLtMatrixLayoutCreate(&d.b, CUDA_R_16F, K, M, K));
    CHECK_LT(cublasLtMatrixLayoutCreate(&d.d, CUDA_R_16F, N, M, N));

    // Workspace from PyTorch's caching allocator (no cudaMalloc per call).
    size_t ws_size = 4u << 20;
    auto ws = torch::empty({static_cast<int64_t>(ws_size)}, x.options().dtype(torch::kUInt8));
    CHECK_LT(cublasLtMatmulPreferenceCreate(&d.pref));
    CHECK_LT(cublasLtMatmulPreferenceSetAttribute(d.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size,
                                                  sizeof(ws_size)));
    cublasLtMatmulHeuristicResult_t heur{};
    int found = 0;
    cublasLtHandle_t h = lt_handle();
    CHECK_LT(cublasLtMatmulAlgoGetHeuristic(h, d.op, d.a, d.b, d.d, d.d, d.pref, 1, &heur, &found));
    TORCH_CHECK(found > 0, "cublasLt found no algorithm for this GEMM + epilogue");
    CHECK_LT(cublasLtMatmul(h, d.op, &alpha, hptr(w), d.a, hptr(x), d.b, &beta, hptr_mut(y), d.d, hptr_mut(y), d.d,
                            &heur.algo, ws.data_ptr(), ws_size, at::cuda::getCurrentCUDAStream()));
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
    gemm_bias_act_lt(x, w, opt_ptr(b), /*gelu=*/false, y);  // bias fused into the GEMM epilogue
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
    // 2 launches total: GEMM + fused bias+GELU, then GEMM + fused bias.
    auto mid = torch::empty({x.size(0), w1.size(1)}, x.options());
    gemm_bias_act_lt(x, w1, opt_ptr(b1), /*gelu=*/true, mid);
    auto out = torch::empty({x.size(0), w2.size(1)}, x.options());
    gemm_bias_act_lt(mid, w2, opt_ptr(b2), /*gelu=*/false, out);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_qkv", &fused_qkv_cuda, "QKV projection y = x @ w + b (cublasLt, bias fused into the GEMM epilogue, FP16)",
          py::arg("x"), py::arg("w"), py::arg("b") = py::none());
    m.def("softmax", &softmax_cuda, "Online softmax over last dim (FP16)");
    m.def("layernorm", &layernorm_cuda, "LayerNorm, Welford statistics (FP16)");
    m.def("gelu", &gelu_cuda, "tanh-GELU (FP16)");
    m.def("fused_mlp", &fused_mlp_cuda, "MLP: cublasLt GEMM+bias+GELU epilogue -> GEMM+bias epilogue (FP16, 2 launches)");
}
