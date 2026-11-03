"""
Correctness tests for the three PyTorch CUDA extensions against a PyTorch reference, on the GPU.

The first run JIT-compiles the FlashAttention and Transformer extensions (needs nvcc and a C++
compiler; on Windows, run from a VS x64 prompt). custom_conv must be installed first:
    cd extension && pip install --no-build-isolation .   (Windows: set DISTUTILS_USE_SDK=1)

A missing custom_conv on a GPU machine is a failure, not a skip: skipping would hide a broken build.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from conftest import requires_cuda

REPO = Path(__file__).resolve().parent.parent
JIT_MODULES = ["flash_attention_simple/attention_cuda.py", "gpu_kernels/transformer/transformer_cuda.py"]
M, H = 2048, 768


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


def _randn(*shape: int, dtype: torch.dtype = torch.half) -> torch.Tensor:
    return torch.randn(*shape, device="cuda", dtype=dtype)


# Runs without a GPU: JIT builds fail on MSVC newer than the CUDA toolkit supports without this flag.
@pytest.mark.parametrize("path", JIT_MODULES)
def test_jit_build_allows_newer_msvc(path: str) -> None:
    flags = re.search(r"extra_cuda_cflags=\[(.*?)\]", (REPO / path).read_text(encoding="utf-8"), re.S)
    assert flags is not None
    assert "allow-unsupported-compiler" in flags.group(1)


@requires_cuda
@pytest.mark.parametrize("S,causal", [(100, False), (100, True), (512, False), (1024, True)])
def test_flash_attention_cuda(S: int, causal: bool) -> None:
    from flash_attention_simple.attention_cuda import attention_cuda

    q, k, v = (_randn(2, 8, S, 64, dtype=torch.float32) for _ in range(3))
    out = attention_cuda(q, k, v, causal=causal)
    assert out is not None, "extension failed to build - see stderr"
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-3)


@requires_cuda
def test_fused_qkv_cuda() -> None:
    from gpu_kernels.transformer import transformer_cuda as tc

    x, w, b = _randn(M, H), _randn(H, 3 * H) / H**0.5, _randn(3 * H) * 0.1
    ref = x.float() @ w.float() + b.float()
    torch.testing.assert_close(tc.fused_qkv_cuda(x, w, b).float(), ref, atol=2e-2, rtol=2e-2)


@requires_cuda
@pytest.mark.parametrize("m,k,n,with_bias", [(100, 72, 40, True), (64, 768, 2304, False), (33, 64, 8, True)])
def test_fused_qkv_cuda_shapes(m: int, k: int, n: int, with_bias: bool) -> None:
    """cublasLt epilogue path: odd shapes and the no-bias epilogue (CUBLASLT_EPILOGUE_DEFAULT)."""
    from gpu_kernels.transformer import transformer_cuda as tc

    x, w = _randn(m, k), _randn(k, n) / k**0.5
    b = _randn(n) * 0.1 if with_bias else None
    ref = x.float() @ w.float() + (b.float() if b is not None else 0)
    torch.testing.assert_close(tc.fused_qkv_cuda(x, w, b).float(), ref, atol=2e-2, rtol=2e-2)


# 768 / 1000 / 1024 take the warp-per-row path (n_cols % 8 == 0, <= 1024);
# 1001 and 2048 take the block-per-row fallback.
ROW_WIDTHS = [768, 1000, 1024, 1001, 2048]


@requires_cuda
@pytest.mark.parametrize("cols", ROW_WIDTHS)
def test_softmax_cuda(cols: int) -> None:
    from gpu_kernels.transformer import transformer_cuda as tc

    x = _randn(M, cols) * 3
    torch.testing.assert_close(tc.softmax_cuda(x).float(), torch.softmax(x.float(), -1), atol=1e-3, rtol=1e-2)


@requires_cuda
@pytest.mark.parametrize("cols", ROW_WIDTHS)
def test_layernorm_cuda(cols: int) -> None:
    from gpu_kernels.transformer import transformer_cuda as tc

    x, g, b = _randn(M - 3, cols) * 2 + 1, _randn(cols), _randn(cols)  # M - 3: rows not a multiple of 4
    ref = F.layer_norm(x.float(), (cols,), g.float(), b.float())
    torch.testing.assert_close(tc.layernorm_cuda(x, g, b).float(), ref, atol=2e-2, rtol=2e-2)


@requires_cuda
def test_gelu_cuda() -> None:
    from gpu_kernels.transformer import transformer_cuda as tc

    x = _randn(M, 1000) * 3
    ref = F.gelu(x.float(), approximate="tanh")
    torch.testing.assert_close(tc.gelu_cuda(x).float(), ref, atol=1e-2, rtol=1e-2)


@requires_cuda
def test_fused_mlp_cuda() -> None:
    from gpu_kernels.transformer import transformer_cuda as tc

    x = _randn(M, H)
    w1, b1 = _randn(H, 4 * H) / H**0.5, _randn(4 * H) * 0.1
    w2, b2 = _randn(4 * H, H) / (4 * H) ** 0.5, _randn(H) * 0.1
    ref = F.gelu(x.float() @ w1.float() + b1.float(), approximate="tanh") @ w2.float() + b2.float()
    torch.testing.assert_close(tc.fused_mlp_cuda(x, w1, b1, w2, b2).float(), ref, atol=3e-2, rtol=3e-2)


@requires_cuda
@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-4), (torch.half, 1e-2)])
def test_custom_conv(dtype: torch.dtype, atol: float) -> None:
    import custom_conv

    x, w, b = _randn(64, 1, 28, 28, dtype=dtype), _randn(32, 1, 3, 3, dtype=dtype) / 3, _randn(32, dtype=dtype) * 0.1
    ref = F.conv2d(x.float(), w.float(), b.float())
    torch.testing.assert_close(custom_conv.custom_conv2d(x, w, b)[0].float(), ref, atol=atol, rtol=1e-3)
