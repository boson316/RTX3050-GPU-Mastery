"""
Flash Attention — Triton implementation of fused softmax(QK^T/sqrt(d))V with tiling.

Provides:
- Baseline: same online softmax algorithm with smaller blocks (BLOCK_M=32, BLOCK_N=32, BLOCK_D=16)
  to reduce register pressure; more blocks, more kernel launches.
- Optimized: larger blocks (BLOCK_M=64, BLOCK_N=64, BLOCK_D=32); fewer blocks, better reuse.
- Autotuning over BLOCK_M, BLOCK_N, BLOCK_D.
- Benchmark vs torch.nn.functional.scaled_dot_product_attention.

Block tiling:
- Each program computes a BLOCK_M x head_dim output tile of O. It corresponds to a block of rows
  of the attention matrix. We iterate over K/V in steps of BLOCK_N. For each K/V block we load
  Q[BLOCK_M, BLOCK_D], K[BLOCK_N, BLOCK_D], V[BLOCK_N, BLOCK_D]; compute scores BLOCK_M x BLOCK_N,
  apply online softmax (update running max and sum), then accumulate into output BLOCK_M x BLOCK_D.
  So we never materialize the full QK^T matrix in HBM — only blocks in SRAM.

Memory access patterns:
- Q: loaded once per program (BLOCK_M x BLOCK_D); coalesced along D.
- K, V: loaded in a loop over key sequence; each load BLOCK_N x BLOCK_D; coalesced.
- O: written once BLOCK_M x BLOCK_D; coalesced. Online softmax keeps m_i, l_i, acc in registers/SRAM.

Register usage:
- m_i, l_i: BLOCK_M scalars (running max and sum for online softmax).
- acc: BLOCK_M x BLOCK_D (output accumulator).
- One block of q, k, p, v in flight. Larger blocks improve reuse but increase register pressure.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Shared forward kernel: online softmax over K dimension, output O = softmax(QK^T/sqrt(d)) @ V
# -----------------------------------------------------------------------------


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Out,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    seq_len_q,
    seq_len_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # Program (0): block index along Q sequence; (1): batch*head index
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qb

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    k_ptrs = K + qvk_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
    v_ptrs = V + qvk_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd

    # Online softmax state: per-row max and sum of exp
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    seq_len_k_padded = tl.cdiv(seq_len_k, BLOCK_N) * BLOCK_N
    for start_n in range(0, seq_len_k_padded, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len_q, other=0.0)
        k = tl.load(k_ptrs, mask=offs_n[None, :] < seq_len_k, other=0.0)
        qk = tl.dot(q, k, allow_tf32=False) * softmax_scale
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        qk = tl.where(offs_m[:, None] < seq_len_q, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        p = p.to(Q.dtype.element_ty)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < seq_len_k, other=0.0)
        acc = acc * alpha[:, None]
        acc += tl.dot(p, v, allow_tf32=False)
        m_i = m_ij
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    acc = acc / l_i[:, None]
    o_ptrs = Out + qvk_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Q.dtype.element_ty), mask=offs_m[:, None] < seq_len_q)


def _flash_attention_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_D: int,
) -> torch.Tensor:
    batch, n_heads, seq_len_q, head_dim = q.shape
    _, _, seq_len_k, _ = k.shape
    out = torch.empty_like(q)
    softmax_scale = 1.0 / (head_dim ** 0.5)
    stride_bh = q.stride(1)
    grid = (triton.cdiv(seq_len_q, BLOCK_M), batch * n_heads)
    _fwd_kernel[grid](
        q, k, v, out, softmax_scale,
        stride_qb=stride_bh, stride_qh=q.stride(1), stride_qm=q.stride(2), stride_qd=q.stride(3),
        stride_kb=stride_bh, stride_kh=k.stride(1), stride_kn=k.stride(2), stride_kd=k.stride(3),
        stride_vb=stride_bh, stride_vh=v.stride(1), stride_vn=v.stride(2), stride_vd=v.stride(3),
        stride_ob=stride_bh, stride_oh=out.stride(1), stride_om=out.stride(2), stride_od=out.stride(3),
        seq_len_q=seq_len_q, seq_len_k=seq_len_k,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        IS_CAUSAL=causal,
    )
    return out


def flash_attention_baseline(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Baseline flash attention with small blocks (more kernel launches, lower register use)."""
    return _flash_attention_impl(q, k, v, causal, BLOCK_M=32, BLOCK_N=32, BLOCK_D=16)


def flash_attention_optimized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Optimized flash attention with larger blocks (fewer launches, better reuse)."""
    return _flash_attention_impl(q, k, v, causal, BLOCK_M=64, BLOCK_N=64, BLOCK_D=32)


# -----------------------------------------------------------------------------
# Autotuned
# -----------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_D": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_D": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_D": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_D": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_D": 32}, num_warps=8),
    ],
    key=["seq_len_q", "seq_len_k"],
)
@triton.jit
def _fwd_kernel_autotuned(
    Q,
    K,
    V,
    Out,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    seq_len_q,
    seq_len_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qb

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    k_ptrs = K + qvk_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
    v_ptrs = V + qvk_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    seq_len_k_padded = tl.cdiv(seq_len_k, BLOCK_N) * BLOCK_N
    for start_n in range(0, seq_len_k_padded, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len_q, other=0.0)
        k = tl.load(k_ptrs, mask=offs_n[None, :] < seq_len_k, other=0.0)
        qk = tl.dot(q, k, allow_tf32=False) * softmax_scale
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        qk = tl.where(offs_m[:, None] < seq_len_q, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        p = p.to(Q.dtype.element_ty)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < seq_len_k, other=0.0)
        acc = acc * alpha[:, None]
        acc += tl.dot(p, v, allow_tf32=False)
        m_i = m_ij
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    acc = acc / l_i[:, None]
    o_ptrs = Out + qvk_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Q.dtype.element_ty), mask=offs_m[:, None] < seq_len_q)


def flash_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    BLOCK_M: int | None = None,
    BLOCK_N: int | None = None,
    BLOCK_D: int | None = None,
    use_autotune: bool = True,
) -> torch.Tensor:
    """
    Flash attention: O = softmax(QK^T / sqrt(d)) V. q,k,v: (batch, n_heads, seq_len, head_dim).
    FP16/BF16. use_autotune=True picks block sizes automatically.
    """
    batch, n_heads, seq_len_q, head_dim = q.shape
    _, _, seq_len_k, _ = k.shape
    assert head_dim == k.shape[-1] == v.shape[-1]
    out = torch.empty_like(q)
    softmax_scale = 1.0 / (head_dim ** 0.5)
    stride_bh = q.stride(1)

    if use_autotune and BLOCK_M is None and BLOCK_N is None and BLOCK_D is None:
        grid = lambda meta: (triton.cdiv(seq_len_q, meta["BLOCK_M"]), batch * n_heads)
        _fwd_kernel_autotuned[grid](
            q, k, v, out, softmax_scale,
            stride_qb=stride_bh, stride_qh=q.stride(1), stride_qm=q.stride(2), stride_qd=q.stride(3),
            stride_kb=stride_bh, stride_kh=k.stride(1), stride_kn=k.stride(2), stride_kd=k.stride(3),
            stride_vb=stride_bh, stride_vh=v.stride(1), stride_vn=v.stride(2), stride_vd=v.stride(3),
            stride_ob=stride_bh, stride_oh=out.stride(1), stride_om=out.stride(2), stride_od=out.stride(3),
            seq_len_q=seq_len_q, seq_len_k=seq_len_k,
            IS_CAUSAL=causal,
        )
        return out

    bm = BLOCK_M or 64
    bn = BLOCK_N or 64
    bd = BLOCK_D or 32
    return _flash_attention_impl(q, k, v, causal, bm, bn, bd)


# -----------------------------------------------------------------------------
# Benchmark vs PyTorch SDPA
# -----------------------------------------------------------------------------


def benchmark_flash_attention(
    B: int = 2,
    H: int = 8,
    S: int = 512,
    D: int = 64,
    dtype: torch.dtype = torch.float16,
    causal: bool = False,
    warmup: int = 10,
    repeat: int = 50,
) -> dict[str, float]:
    q = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
    v = torch.randn(B, H, S, D, device="cuda", dtype=dtype)
    scale = 1.0 / (D ** 0.5)

    def run(fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = __import__("time").perf_counter()
        for _ in range(repeat):
            fn()
        torch.cuda.synchronize()
        return (__import__("time").perf_counter() - start) * 1000 / repeat

    out = {}
    out["pytorch_sdpa_ms"] = run(
        lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=causal)
    )
    out["triton_baseline_ms"] = run(lambda: flash_attention_baseline(q, k, v, causal=causal))
    out["triton_optimized_ms"] = run(lambda: flash_attention_optimized(q, k, v, causal=causal))
    out["triton_autotune_ms"] = run(lambda: flash_attention_triton(q, k, v, causal=causal, use_autotune=True))
    return out


if __name__ == "__main__":
    B, H, S, D = 2, 4, 128, 32
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    out = flash_attention_triton(q, k, v, causal=False, use_autotune=True)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), scale=1.0 / (D ** 0.5)
    ).half()
    print("Flash attention max diff vs PyTorch SDPA:", (out - ref).abs().max().item())
    print("Benchmark:", benchmark_flash_attention(B=2, H=8, S=512, D=64))
