"""
Flash Attention — Triton implementation of fused softmax(QK^T/sqrt(d))V with tiling.

Provides:
- Baseline: online softmax with small tiles (BLOCK_M=32, BLOCK_N=32).
- Optimized: larger tiles (BLOCK_M=64, BLOCK_N=64): fewer programs, more K/V reuse.
- Autotuning over BLOCK_M, BLOCK_N, num_warps, num_stages (BLOCK_D is fixed to cover head_dim).
- Causal masking skips fully-masked key blocks (~2x fewer FLOPs) and masks only the diagonal band.
- exp2 with log2(e) folded into the softmax scale.
- Benchmark vs torch.nn.functional.scaled_dot_product_attention.

Block tiling:
- Each program computes a BLOCK_M x head_dim output tile of O. It corresponds to a block of rows
  of the attention matrix. We iterate over K/V in steps of BLOCK_N. For each K/V block we load
  Q[BLOCK_M, BLOCK_D], K[BLOCK_N, BLOCK_D], V[BLOCK_N, BLOCK_D]; compute scores BLOCK_M x BLOCK_N,
  apply online softmax (update running max and sum), then accumulate into output BLOCK_M x BLOCK_D.
  So we never materialize the full QK^T matrix in HBM — only blocks in SRAM.

Memory access patterns:
- Q: loaded once per program (BLOCK_M x BLOCK_D), kept in registers; coalesced along D.
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
# Forward kernel: online softmax over the key dimension.
#
# BLOCK_D is NOT a tuning knob: it is next_power_of_2(head_dim) and must cover the
# whole head dimension, otherwise QK^T would be a partial dot product and output
# columns >= BLOCK_D would never be written. (An earlier version of this file tuned
# BLOCK_D in {16, 32} with head_dim=64, which silently computed wrong results and
# made the kernel look ~2x faster than PyTorch SDPA. tests/test_triton_kernels.py
# now guards against this.)
# -----------------------------------------------------------------------------


@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    K_block_ptr_base, V_block_ptr_base,
    stride_kn, stride_vn,
    offs_m, offs_n, offs_d,
    start_n_lo, start_n_hi,
    seq_len_k, head_dim,
    qk_scale,
    BLOCK_N: tl.constexpr,
    MASK_DIAG: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    d_mask = offs_d < head_dim
    for start_n in range(start_n_lo, start_n_hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        cols = start_n + offs_n
        k = tl.load(
            K_block_ptr_base + cols[None, :] * stride_kn + offs_d[:, None],
            mask=(cols[None, :] < seq_len_k) & d_mask[:, None],
            other=0.0,
        )
        # qk_scale already folds in log2(e) so we can use exp2 (cheaper than exp on GPU).
        qk = tl.dot(q, k, input_precision=INPUT_PRECISION) * qk_scale
        qk = tl.where(cols[None, :] < seq_len_k, qk, float("-inf"))
        if MASK_DIAG:
            qk = tl.where(offs_m[:, None] >= cols[None, :], qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(
            V_block_ptr_base + cols[:, None] * stride_vn + offs_d[None, :],
            mask=(cols[:, None] < seq_len_k) & d_mask[None, :],
            other=0.0,
        )
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v, input_precision=INPUT_PRECISION)
        m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    softmax_scale,
    stride_qbh, stride_qm,
    stride_kbh, stride_kn,
    stride_vbh, stride_vn,
    stride_obh, stride_om,
    seq_len_q, seq_len_k, head_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    # program_id(0): tile of BLOCK_M query rows; program_id(1): flattened (batch, head).
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1).to(tl.int64)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    m_mask = offs_m < seq_len_q
    d_mask = offs_d < head_dim

    Q_base = Q + off_bh * stride_qbh
    K_base = K + off_bh * stride_kbh
    V_base = V + off_bh * stride_vbh
    O_base = Out + off_bh * stride_obh

    # Q tile is loaded ONCE and stays in registers for the whole K/V sweep.
    q = tl.load(
        Q_base + offs_m[:, None] * stride_qm + offs_d[None, :],
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    qk_scale = softmax_scale * 1.4426950408889634  # log2(e)

    if IS_CAUSAL:
        # Causal: key blocks entirely below the diagonal need no mask; blocks entirely
        # above it are skipped (about half the FLOPs), only the diagonal band is masked.
        lo_full = 0
        hi_full = (start_m * BLOCK_M) // BLOCK_N * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_base, V_base, stride_kn, stride_vn,
            offs_m, offs_n, offs_d, lo_full, hi_full, seq_len_k, head_dim, qk_scale,
            BLOCK_N=BLOCK_N, MASK_DIAG=False, INPUT_PRECISION=INPUT_PRECISION,
        )
        hi_diag = tl.minimum((start_m + 1) * BLOCK_M, seq_len_k)
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_base, V_base, stride_kn, stride_vn,
            offs_m, offs_n, offs_d, hi_full, hi_diag, seq_len_k, head_dim, qk_scale,
            BLOCK_N=BLOCK_N, MASK_DIAG=True, INPUT_PRECISION=INPUT_PRECISION,
        )
    else:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_base, V_base, stride_kn, stride_vn,
            offs_m, offs_n, offs_d, 0, seq_len_k, seq_len_k, head_dim, qk_scale,
            BLOCK_N=BLOCK_N, MASK_DIAG=False, INPUT_PRECISION=INPUT_PRECISION,
        )

    # Padded rows (offs_m >= seq_len_q) have l_i == 0; guard the division, they are masked on store.
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]
    tl.store(
        O_base + offs_m[:, None] * stride_om + offs_d[None, :],
        acc.to(Out.dtype.element_ty),
        mask=m_mask[:, None] & d_mask[None, :],
    )


_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
    for bm, bn, w, s in [
        (32, 32, 4, 2),
        (64, 32, 4, 2),
        (64, 64, 4, 2),
        (64, 64, 4, 3),
        (128, 32, 4, 2),
        (128, 64, 8, 2),
    ]
]

_fwd_kernel_autotuned = triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["seq_len_q", "seq_len_k", "head_dim", "IS_CAUSAL", "INPUT_PRECISION"],
)(_fwd_kernel)


def _check_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> None:
    assert q.dim() == k.dim() == v.dim() == 4, "expected (batch, heads, seq, head_dim)"
    assert q.shape[-1] == k.shape[-1] == v.shape[-1], "head_dim mismatch"
    assert k.shape == v.shape and q.shape[:2] == k.shape[:2], "batch/heads mismatch"
    assert q.shape[-1] <= 256, "head_dim > 256 not supported"
    if causal:
        assert q.shape[2] == k.shape[2], "causal attention assumes seq_len_q == seq_len_k"


PRECISIONS = ("tf32", "ieee")


def _launch(q, k, v, causal, block_m=None, block_n=None, precision="tf32"):
    _check_inputs(q, k, v, causal)
    assert precision in PRECISIONS, f"precision must be one of {PRECISIONS}"
    # The kernel assumes unit stride on the last dim and a flattenable (batch, head) dim.
    q, k, v = (t.contiguous() for t in (q, k, v))
    batch, n_heads, seq_len_q, head_dim = q.shape
    seq_len_k = k.shape[2]
    out = torch.empty_like(q)
    softmax_scale = 1.0 / (head_dim ** 0.5)
    block_d = max(16, triton.next_power_of_2(head_dim))  # tl.dot needs K >= 16
    args = (
        q, k, v, out, softmax_scale,
        q.stride(1), q.stride(2),
        k.stride(1), k.stride(2),
        v.stride(1), v.stride(2),
        out.stride(1), out.stride(2),
        seq_len_q, seq_len_k, head_dim,
    )
    if block_m is None:
        grid = lambda meta: (triton.cdiv(seq_len_q, meta["BLOCK_M"]), batch * n_heads)  # noqa: E731
        _fwd_kernel_autotuned[grid](*args, BLOCK_D=block_d, IS_CAUSAL=causal, INPUT_PRECISION=precision)
    else:
        grid = (triton.cdiv(seq_len_q, block_m), batch * n_heads)
        _fwd_kernel[grid](*args, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=block_d, IS_CAUSAL=causal,
                          INPUT_PRECISION=precision)
    return out


def flash_attention_baseline(q, k, v, causal: bool = False) -> torch.Tensor:
    """Small tiles (32x32): lower register pressure, less K/V reuse per program."""
    return _launch(q, k, v, causal, block_m=32, block_n=32)


def flash_attention_optimized(q, k, v, causal: bool = False) -> torch.Tensor:
    """Larger tiles (64x64): each K/V tile loaded from DRAM is reused by 64 query rows."""
    return _launch(q, k, v, causal, block_m=64, block_n=64)


def flash_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    BLOCK_M: int | None = None,
    BLOCK_N: int | None = None,
    use_autotune: bool = True,
    precision: str = "tf32",
) -> torch.Tensor:
    """
    O = softmax(Q K^T / sqrt(d)) V for q, k, v of shape (batch, heads, seq, head_dim), FP16/BF16/FP32.

    use_autotune=True (and no explicit block sizes) autotunes BLOCK_M / BLOCK_N / num_warps / num_stages,
    keyed on (seq_len_q, seq_len_k, head_dim, causal, precision).

    precision only matters for FP32 inputs: "tf32" (Triton's default) runs both matmuls on Tensor
    Cores with a 10-bit mantissa (relative error ~5e-4: fast, but not FP32-accurate); "ieee" uses
    FP32 FMA on CUDA cores. v2 measured Triton FP32 attention 4x faster than SDPA at S=4096 with
    ~1000x larger error - that was TF32 vs IEEE, so benchmarks now report the precision explicitly.
    FP16/BF16 inputs always use Tensor Cores with FP32 accumulation.
    """
    if use_autotune and BLOCK_M is None and BLOCK_N is None:
        return _launch(q, k, v, causal, precision=precision)
    return _launch(q, k, v, causal, block_m=BLOCK_M or 64, block_n=BLOCK_N or 64, precision=precision)


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
