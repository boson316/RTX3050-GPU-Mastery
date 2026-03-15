/*
 * Transformer GPU kernels (FP16) for RTX 3050 (sm_86).
 * 1. Fused QKV projection (cuBLAS matmul + bias kernel)  2. Softmax  3. Layernorm  4. GELU  5. Fused MLP block.
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        /* avoid fprintf in device code path */ \
    } \
} while(0)

#define TILE_QKV 16

// -----------------------------------------------------------------------------
// 1. Fused QKV: y = x @ W_qkv + b. 使用 cuBLAS 做 matmul（高效），再用小 kernel 加 bias，
//    避免自訂 16x16 tiled kernel 造成長時間 100% GPU 與極慢延遲。
// -----------------------------------------------------------------------------

__global__ void qkv_fill_bias_kernel(__half* __restrict__ y, const __half* __restrict__ b, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx < total) {
        int col = idx % N;
        y[idx] = b ? b[col] : __float2half(0.f);
    }
}

// 將 cuBLAS 的 col-major 結果 temp (ldc=M) 加上 bias 寫入 row-major y
__global__ void qkv_colmajor_to_rowmajor_add_bias(
    const __half* __restrict__ temp,
    const __half* __restrict__ b,
    __half* __restrict__ y,
    int M,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx < total) {
        int row = idx / N, col = idx % N;
        __half v = temp[row + col * M];  // temp 為 col-major MxN
        if (b) v = __float2half(__half2float(v) + __half2float(b[col]));
        y[idx] = v;
    }
}

// 保留舊的 tiled kernel 供參考；launch 改為 cuBLAS + bias kernel
__global__ void fused_qkv_fp16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    const __half* __restrict__ b,
    __half* __restrict__ y,
    int M,
    int K,
    int N
) {
    __shared__ __half As[TILE_QKV][TILE_QKV];
    __shared__ __half Bs[TILE_QKV][TILE_QKV];
    int row = blockIdx.y * TILE_QKV + threadIdx.y;
    int col = blockIdx.x * TILE_QKV + threadIdx.x;
    float sum = 0.f;
    int numTiles = (K + TILE_QKV - 1) / TILE_QKV;
    for (int t = 0; t < numTiles; t++) {
        int colA = t * TILE_QKV + threadIdx.x;
        int rowB = t * TILE_QKV + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && colA < K) ? x[row * K + colA] : __float2half(0.f);
        Bs[threadIdx.y][threadIdx.x] = (rowB < K && col < N) ? w[rowB * N + col] : __float2half(0.f);
        __syncthreads();
        for (int k = 0; k < TILE_QKV; k++)
            sum += __half2float(As[threadIdx.y][k]) * __half2float(Bs[k][threadIdx.x]);
        __syncthreads();
    }
    if (row < M && col < N) {
        float bias_val = b ? __half2float(b[col]) : 0.f;
        y[row * N + col] = __float2half(sum + bias_val);
    }
}

// -----------------------------------------------------------------------------
// 2. Softmax over last dimension (row-wise). Three-pass: max, sum(exp), normalize.
// -----------------------------------------------------------------------------
__device__ __forceinline__ float blockReduceMax(float val) {
    __shared__ float smem[8];
    __shared__ float result;
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    if (threadIdx.x < 8) {
        val = smem[threadIdx.x];
        for (int offset = 4; offset > 0; offset /= 2)
            val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
        if (threadIdx.x == 0) result = val;
    }
    __syncthreads();
    return result;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    __shared__ float smem[8];
    __shared__ float result;
    if (threadIdx.x % 32 == 0) smem[threadIdx.x / 32] = val;
    __syncthreads();
    if (threadIdx.x < 8) {
        val = smem[threadIdx.x];
        for (int offset = 4; offset > 0; offset /= 2)
            val += __shfl_xor_sync(0xffffffff, val, offset);
        if (threadIdx.x == 0) result = val;
    }
    __syncthreads();
    return result;
}

#define SOFMAX_ROWS_PER_BLOCK 8

__global__ void softmax_fp16_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ y,
    int stride,
    int N,
    int num_rows
) {
    // 每個 block 處理多行，減少 block 數（避免 16384 blocks 造成卡住/逾時）
    for (int r = 0; r < SOFMAX_ROWS_PER_BLOCK; r++) {
        int row = blockIdx.x * SOFMAX_ROWS_PER_BLOCK + r;
        if (row >= num_rows) break;
        const __half* row_in = x + (size_t)row * stride;
        __half* row_out = y + (size_t)row * stride;

        float row_max = -1e30f;
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            float v = __half2float(row_in[i]);
            row_max = fmaxf(row_max, v);
        }
        row_max = blockReduceMax(row_max);

        float row_sum = 0.f;
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            float v = __half2float(row_in[i]);
            row_sum += expf(v - row_max);
        }
        row_sum = blockReduceSum(row_sum);
        row_sum = fmaxf(row_sum, 1e-12f);

        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            float v = __half2float(row_in[i]);
            row_out[i] = __float2half(expf(v - row_max) / row_sum);
        }
    }
}

// -----------------------------------------------------------------------------
// 3. LayerNorm over last dimension: y = (x - mean) * rstd * weight + bias
// -----------------------------------------------------------------------------
__global__ void layernorm_fp16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __half* __restrict__ y,
    int stride,
    int N,
    float eps
) {
    int row = blockIdx.x;
    const __half* row_in = x + (size_t)row * stride;
    __half* row_out = y + (size_t)row * stride;

    // Pass 1: mean
    float sum = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        sum += __half2float(row_in[i]);
    sum = blockReduceSum(sum);
    float mean = sum / (float)N;

    // Pass 2: variance
    float var_sum = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = __half2float(row_in[i]) - mean;
        var_sum += v * v;
    }
    var_sum = blockReduceSum(var_sum);
    float rstd = rsqrtf(var_sum / (float)N + eps);

    // Pass 3: normalize and affine
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = (__half2float(row_in[i]) - mean) * rstd;
        float w = weight ? __half2float(weight[i]) : 1.f;
        float b = bias ? __half2float(bias[i]) : 0.f;
        row_out[i] = __float2half(v * w + b);
    }
}

// -----------------------------------------------------------------------------
// 4. GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// -----------------------------------------------------------------------------
__device__ __forceinline__ float gelu_f(float x) {
    const float sqrt_2_over_pi = 0.79788456080286535588f;
    return 0.5f * x * (1.f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}

__global__ void gelu_fp16_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ y,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = __float2half(gelu_f(__half2float(x[i])));
}

// -----------------------------------------------------------------------------
// 5. Fused MLP block: out = linear2(GELU(linear1(x)))
//    Kernel A: mid = GELU(x @ W1 + b1).  Kernel B: out = mid @ W2 + b2.
//    "Fused" = single launch that runs both (intermediate in global).
// -----------------------------------------------------------------------------
__global__ void mlp_linear_gelu_fp16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    const __half* __restrict__ b,
    __half* __restrict__ mid,
    int M,
    int K,
    int N
) {
    __shared__ __half As[TILE_QKV][TILE_QKV];
    __shared__ __half Bs[TILE_QKV][TILE_QKV];

    int row = blockIdx.y * TILE_QKV + threadIdx.y;
    int col = blockIdx.x * TILE_QKV + threadIdx.x;
    float sum = 0.f;

    int numTiles = (K + TILE_QKV - 1) / TILE_QKV;
    for (int t = 0; t < numTiles; t++) {
        int colA = t * TILE_QKV + threadIdx.x;
        int rowB = t * TILE_QKV + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && colA < K) ? x[row * K + colA] : __float2half(0.f);
        Bs[threadIdx.y][threadIdx.x] = (rowB < K && col < N) ? w[rowB * N + col] : __float2half(0.f);
        __syncthreads();
        for (int k = 0; k < TILE_QKV; k++)
            sum += __half2float(As[threadIdx.y][k]) * __half2float(Bs[k][threadIdx.x]);
        __syncthreads();
    }
    if (row < M && col < N) {
        float bias_val = b ? __half2float(b[col]) : 0.f;
        mid[row * N + col] = __float2half(gelu_f(sum + bias_val));
    }
}

__global__ void mlp_linear2_fp16_kernel(
    const __half* __restrict__ mid,
    const __half* __restrict__ w,
    const __half* __restrict__ b,
    __half* __restrict__ out,
    int M,
    int K,
    int N
) {
    __shared__ __half As[TILE_QKV][TILE_QKV];
    __shared__ __half Bs[TILE_QKV][TILE_QKV];

    int row = blockIdx.y * TILE_QKV + threadIdx.y;
    int col = blockIdx.x * TILE_QKV + threadIdx.x;
    float sum = 0.f;

    int numTiles = (K + TILE_QKV - 1) / TILE_QKV;
    for (int t = 0; t < numTiles; t++) {
        int colA = t * TILE_QKV + threadIdx.x;
        int rowB = t * TILE_QKV + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && colA < K) ? mid[row * K + colA] : __float2half(0.f);
        Bs[threadIdx.y][threadIdx.x] = (rowB < K && col < N) ? w[rowB * N + col] : __float2half(0.f);
        __syncthreads();
        for (int k = 0; k < TILE_QKV; k++)
            sum += __half2float(As[threadIdx.y][k]) * __half2float(Bs[k][threadIdx.x]);
        __syncthreads();
    }
    if (row < M && col < N) {
        float bias_val = b ? __half2float(b[col]) : 0.f;
        out[row * N + col] = __float2half(sum + bias_val);
    }
}

// -----------------------------------------------------------------------------
// Launch wrappers (called from C++)
// -----------------------------------------------------------------------------
extern "C" {

static cublasHandle_t g_cublas_handle = nullptr;
static __half* g_qkv_temp = nullptr;
static size_t g_qkv_temp_size = 0;

void fused_qkv_fp16_launch(
    const __half* x, const __half* w, const __half* b, __half* y,
    int M, int K, int N, cudaStream_t stream
) {
    if (g_cublas_handle == nullptr) {
        cublasCreate(&g_cublas_handle);
    }
    cublasSetStream(g_cublas_handle, stream);

    size_t need = (size_t)M * (size_t)N * sizeof(__half);
    if (g_qkv_temp_size < need) {
        if (g_qkv_temp) cudaFree(g_qkv_temp);
        cudaMalloc(&g_qkv_temp, need);
        g_qkv_temp_size = need;
    }
    __half* temp = g_qkv_temp;

    float alpha = 1.f, beta = 0.f;
    cublasGemmEx(
        g_cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        x, CUDA_R_16F, K,
        w, CUDA_R_16F, N,
        &beta,
        temp, CUDA_R_16F, M,
        CUBLAS_COMPUTE_32F,
        (cublasGemmAlgo_t)0
    );

    int total = M * N;
    qkv_colmajor_to_rowmajor_add_bias<<<(total + 255) / 256, 256, 0, stream>>>(temp, b, y, M, N);
}

void softmax_fp16_launch(
    const __half* x, __half* y, int stride, int N, int num_rows, cudaStream_t stream
) {
    int num_blocks = (num_rows + SOFMAX_ROWS_PER_BLOCK - 1) / SOFMAX_ROWS_PER_BLOCK;
    softmax_fp16_kernel<<<num_blocks, 256, 0, stream>>>(x, y, stride, N, num_rows);
}

void layernorm_fp16_launch(
    const __half* x, const __half* weight, const __half* bias, __half* y,
    int stride, int N, int num_rows, float eps, cudaStream_t stream
) {
    layernorm_fp16_kernel<<<num_rows, 256, 0, stream>>>(x, weight, bias, y, stride, N, eps);
}

void gelu_fp16_launch(const __half* x, __half* y, int n, cudaStream_t stream) {
    gelu_fp16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, y, n);
}

void fused_mlp_fp16_launch(
    const __half* x,
    const __half* w1, const __half* b1,
    const __half* w2, const __half* b2,
    __half* mid, __half* out,
    int M, int K, int N1, int N2,
    cudaStream_t stream
) {
    dim3 block(TILE_QKV, TILE_QKV);
    dim3 grid1((N1 + TILE_QKV - 1) / TILE_QKV, (M + TILE_QKV - 1) / TILE_QKV);
    dim3 grid2((N2 + TILE_QKV - 1) / TILE_QKV, (M + TILE_QKV - 1) / TILE_QKV);
    mlp_linear_gelu_fp16_kernel<<<grid1, block, 0, stream>>>(x, w1, b1, mid, M, K, N1);
    mlp_linear2_fp16_kernel<<<grid2, block, 0, stream>>>(mid, w2, b2, out, M, N1, N2);
}

}  // extern "C"
