"""
Correctness tests for every Triton kernel against a PyTorch reference.

Shapes deliberately include sizes that are NOT multiples of the block sizes and head_dim
values that are not powers of two: those are exactly the cases where the bugs listed in
docs/ENGINEERING_LOG.md (missing K mask, partial head_dim) produced wrong results.

CPU (interpreter) runs use FP32 inputs because NumPy has no fast FP16 matmul; on a GPU the
FP16 path is tested as well.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from conftest import HAS_CUDA, requires_cuda

torch.manual_seed(0)

DTYPES = [torch.float32] + ([torch.float16] if HAS_CUDA else [])


def tol(dtype: torch.dtype) -> dict:
    # Triton's tl.dot uses TF32 for FP32 inputs on Ampere; FP16 has a 10-bit mantissa.
    return {"atol": 2e-2, "rtol": 2e-2} if dtype == torch.float16 or HAS_CUDA else {"atol": 1e-4, "rtol": 1e-4}


# ----------------------------------------------------------------------------- matmul
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M,N,K", [(64, 64, 64), (100, 72, 50), (33, 130, 97), (1, 16, 300)])
@pytest.mark.parametrize("variant", ["baseline", "optimized"])
def test_matmul_odd_shapes(device, dtype, M, N, K, variant):
    from triton_kernels.matmul import matmul_baseline, matmul_optimized

    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)
    fn = matmul_baseline if variant == "baseline" else matmul_optimized
    out = fn(a, b)
    ref = (a.float() @ b.float()).to(dtype)
    torch.testing.assert_close(out, ref, **tol(dtype))


@requires_cuda
def test_matmul_autotuned_fp16():
    from triton_kernels.matmul import matmul_triton

    a = torch.randn(512, 384, device="cuda", dtype=torch.float16)
    b = torch.randn(384, 257, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(matmul_triton(a, b), (a.float() @ b.float()).half(), atol=5e-2, rtol=2e-2)


# ----------------------------------------------------------------------------- flash attention
def _sdpa_ref(q, k, v, causal):
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal).to(q.dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("B,H,S,D", [(1, 2, 64, 64), (2, 1, 100, 64), (1, 1, 37, 40), (1, 2, 130, 16)])
@pytest.mark.parametrize("variant", ["baseline", "optimized"])
def test_flash_attention(device, dtype, causal, B, H, S, D, variant):
    """head_dim=64 is the configuration the old kernel got wrong (it only computed BLOCK_D=16/32 dims)."""
    from triton_kernels.flash_attention import flash_attention_baseline, flash_attention_optimized

    q, k, v = (torch.randn(B, H, S, D, device=device, dtype=dtype) for _ in range(3))
    fn = flash_attention_baseline if variant == "baseline" else flash_attention_optimized
    out = fn(q, k, v, causal=causal)
    torch.testing.assert_close(out, _sdpa_ref(q, k, v, causal), **tol(dtype))


@requires_cuda
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_autotuned_fp16(causal):
    from triton_kernels.flash_attention import flash_attention_triton

    q, k, v = (torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    out = flash_attention_triton(q, k, v, causal=causal)
    torch.testing.assert_close(out, _sdpa_ref(q, k, v, causal), atol=2e-2, rtol=2e-2)


# ----------------------------------------------------------------------------- softmax / layernorm
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(4, 7), (3, 100), (2, 3, 1000), (5, 2049)])
@pytest.mark.parametrize("variant", ["baseline", "optimized"])
def test_softmax(device, dtype, shape, variant):
    from triton_kernels.softmax import softmax_baseline, softmax_optimized

    x = torch.randn(*shape, device=device, dtype=dtype) * 4
    out = (softmax_baseline if variant == "baseline" else softmax_optimized)(x)
    torch.testing.assert_close(out, torch.softmax(x.float(), -1).to(dtype), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(4, 7), (3, 100), (2, 3, 768), (5, 2049)])
@pytest.mark.parametrize("variant", ["baseline", "optimized"])
def test_layernorm(device, dtype, shape, variant):
    from triton_kernels.layernorm import layernorm_baseline, layernorm_optimized

    n = shape[-1]
    x = torch.randn(*shape, device=device, dtype=dtype) * 3 + 2  # non-zero mean
    w = torch.randn(n, device=device, dtype=dtype)
    b = torch.randn(n, device=device, dtype=dtype)
    fn = layernorm_baseline if variant == "baseline" else layernorm_optimized
    out = fn(x, (n,), w, b, eps=1e-5)
    ref = F.layer_norm(x.float(), (n,), w.float(), b.float(), eps=1e-5).to(dtype)
    torch.testing.assert_close(out, ref, atol=2e-2 if dtype == torch.float16 else 1e-4, rtol=1e-3)


# ----------------------------------------------------------------------------- elementwise / linear
@pytest.mark.parametrize("dtype", DTYPES)
def test_gelu(device, dtype):
    from triton_kernels.gelu import gelu_triton

    x = torch.randn(3, 1001, device=device, dtype=dtype) * 3
    torch.testing.assert_close(gelu_triton(x), F.gelu(x.float()).to(dtype), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M,K,N", [(64, 64, 192), (50, 70, 33)])
def test_fused_qkv(device, dtype, M, K, N):
    """K=70 is not a multiple of BLOCK_K=32: regression test for the missing K-offset mask."""
    from triton_kernels.qkv import fused_qkv_triton

    x = torch.randn(M, K, device=device, dtype=dtype)
    w = torch.randn(K, N, device=device, dtype=dtype)
    b = torch.randn(N, device=device, dtype=dtype)
    ref = (x.float() @ w.float() + b.float()).to(dtype)
    torch.testing.assert_close(fused_qkv_triton(x, w, b), ref, **tol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_mlp(device, dtype):
    from triton_kernels.mlp import fused_mlp_triton

    M, H = 48, 40
    x = torch.randn(M, H, device=device, dtype=dtype) * 0.5
    w1 = torch.randn(H, 4 * H, device=device, dtype=dtype) / math.sqrt(H)
    b1 = torch.randn(4 * H, device=device, dtype=dtype) * 0.1
    w2 = torch.randn(4 * H, H, device=device, dtype=dtype) / math.sqrt(4 * H)
    b2 = torch.randn(H, device=device, dtype=dtype) * 0.1
    ref = F.gelu(x.float() @ w1.float() + b1.float(), approximate="tanh") @ w2.float() + b2.float()
    torch.testing.assert_close(fused_mlp_triton(x, w1, b1, w2, b2), ref.to(dtype), **tol(dtype))


# ----------------------------------------------------------------------------- conv
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("variant", ["baseline", "optimized"])
def test_conv3x3(device, dtype, variant):
    from triton_kernels.conv import conv2d_baseline, conv2d_optimized

    x = torch.randn(2, 1, 30, 27, device=device, dtype=dtype)
    w = torch.randn(20, 1, 3, 3, device=device, dtype=dtype)
    b = torch.randn(20, device=device, dtype=dtype)
    fn = conv2d_baseline if variant == "baseline" else conv2d_optimized
    ref = F.conv2d(x.float(), w.float(), b.float()).to(dtype)
    torch.testing.assert_close(fn(x, w, b), ref, atol=2e-2 if dtype == torch.float16 else 1e-4, rtol=1e-3)
