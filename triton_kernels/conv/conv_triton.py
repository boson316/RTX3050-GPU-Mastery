"""
Triton 3x3 Conv2D for deep learning.

Provides:
- Baseline: one program per (batch, tile_h, tile_w, output_channel); each loads
  input on demand (no shared input tile reuse).
- Optimized: each program loads the 9 shifted input patches of a spatial tile once and
  reuses them for BLOCK_C output channels via 9 vectorised outer-product updates.
- Autotuning over BLOCK_OH, BLOCK_OW, BLOCK_C, num_warps (same kernel as "optimized").
- Scope: C_in = 1, stride 1, no padding (the MNIST first layer this repo benchmarks).
- Benchmark vs torch.nn.functional.conv2d.

Block tiling:
- Output is tiled into blocks of size BLOCK_H x BLOCK_W (spatial) and BLOCK_C (output channels).
- Each program writes one output tile; it may iterate over the 3x3 kernel and (for baseline)
  load input pixels per (oh+kh, ow+kw). Optimized version loads a full (BLOCK_H+2)x(BLOCK_W+2)
  input patch into SRAM and reuses for all BLOCK_C output channels.

Memory access patterns:
- Input: coalesced when reading rows (W-stride); avoid strided access along H when possible.
- Weight: (C_out, 9) flattened; each program loads 9 floats per output channel.
- Output: write BLOCK_H x BLOCK_W per channel; coalesced along W.

Register usage:
- Baseline: accumulators BLOCK_H*BLOCK_W per output channel, plus loaded input tile.
- Optimized: larger input tile in SRAM (BLOCK_H+2)*(BLOCK_W+2); fewer global loads.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Baseline: straightforward 3x3 conv, load input per (kh,kw) and per oc
# -----------------------------------------------------------------------------


@triton.jit
def _conv_baseline_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    B,
    H,
    W,
    C_out,
    out_h,
    out_w,
    stride_b: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    n_th = tl.cdiv(out_h, BLOCK_H)
    n_tw = tl.cdiv(out_w, BLOCK_W)
    n_oc = tl.cdiv(C_out, BLOCK_C)
    pid_b = pid // (n_th * n_tw * n_oc)
    pid_rest = pid % (n_th * n_tw * n_oc)
    pid_th = pid_rest // (n_tw * n_oc)
    pid_tw = (pid_rest // n_oc) % n_tw
    pid_oc = pid_rest % n_oc

    oh = pid_th * BLOCK_H
    ow = pid_tw * BLOCK_W
    oc_base = pid_oc * BLOCK_C

    off_h = tl.arange(0, BLOCK_H)
    off_w = tl.arange(0, BLOCK_W)
    mask_sp = (oh + off_h[:, None] < out_h) & (ow + off_w[None, :] < out_w)

    for c in range(BLOCK_C):
        oc = oc_base + c
        if oc < C_out:
            acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
            for kh in range(3):
                for kw in range(3):
                    in_mask = (oh + kh + off_h[:, None] < H) & (ow + kw + off_w[None, :] < W)
                    inp = tl.load(
                        x_ptr + pid_b * stride_b + (oh + kh) * stride_h + (ow + kw) * stride_w
                        + off_h[:, None] * stride_h + off_w[None, :] * stride_w,
                        mask=in_mask,
                        other=0.0,
                    ).to(tl.float32)
                    w = tl.load(w_ptr + oc * 9 + kh * 3 + kw).to(tl.float32)
                    acc += inp * w
            acc += tl.load(b_ptr + oc).to(tl.float32)
            tl.store(
                y_ptr + pid_b * C_out * out_h * out_w + oc * out_h * out_w
                + oh * out_w
                + ow
                + off_h[:, None] * out_w
                + off_w[None, :],
                acc.to(x_ptr.dtype.element_ty),
                mask=mask_sp,
            )


# -----------------------------------------------------------------------------
# Optimized: one program computes a [BLOCK_C output channels] x [BLOCK_OH x BLOCK_OW pixels]
# tile. The 9 shifted input patches are loaded ONCE and reused by all BLOCK_C channels; the
# 9 weight taps are loaded as BLOCK_C-wide vectors. The convolution becomes 9 rank-1
# (outer-product) updates  acc[c, p] += w[c, tap] * patch_tap[p]  - fully vectorised, no
# per-channel scalar loop.
#
# Fixed: the previous optimized/autotuned kernels stored to `oh * out_w + off_h * out_w + off_w`
# without the `+ ow` column offset, so every column tile overwrote columns [0, BLOCK_OW) and the
# output was wrong whenever out_w > BLOCK_OW (i.e. for every benchmark configuration).
# -----------------------------------------------------------------------------


@triton.jit
def _conv_optimized_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, H, W, C_out, out_h, out_w,
    stride_b, stride_h, stride_w,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    n_th = tl.cdiv(out_h, BLOCK_OH)
    n_tw = tl.cdiv(out_w, BLOCK_OW)
    n_oc = tl.cdiv(C_out, BLOCK_C)
    # channel-block is the fastest-varying index: consecutive programs reuse the same input
    # patch from L1/L2 for different output channels
    pid_oc = pid % n_oc
    pid_tw = (pid // n_oc) % n_tw
    pid_th = (pid // (n_oc * n_tw)) % n_th
    pid_b = pid // (n_oc * n_tw * n_th)

    offs_c = pid_oc * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_p = tl.arange(0, BLOCK_OH * BLOCK_OW)
    ph = pid_th * BLOCK_OH + offs_p // BLOCK_OW   # output row of each pixel in the tile
    pw = pid_tw * BLOCK_OW + offs_p % BLOCK_OW    # output col
    c_mask = offs_c < C_out
    p_mask = (ph < out_h) & (pw < out_w)          # valid output => all 9 taps are in bounds

    x_base = x_ptr + pid_b.to(tl.int64) * stride_b
    acc = tl.zeros((BLOCK_C, BLOCK_OH * BLOCK_OW), dtype=tl.float32)
    for tap in tl.static_range(9):
        kh = tap // 3
        kw = tap % 3
        patch = tl.load(x_base + (ph + kh) * stride_h + (pw + kw) * stride_w, mask=p_mask, other=0.0)
        wt = tl.load(w_ptr + offs_c * 9 + tap, mask=c_mask, other=0.0)
        acc += wt.to(tl.float32)[:, None] * patch.to(tl.float32)[None, :]
    acc += tl.load(b_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)[:, None]

    y_base = y_ptr + pid_b.to(tl.int64) * C_out * out_h * out_w
    y_ptrs = y_base + offs_c[:, None] * (out_h * out_w) + (ph * out_w + pw)[None, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=c_mask[:, None] & p_mask[None, :])


def _conv_prepare(x, w, b):
    assert x.dim() == 4 and x.shape[1] == 1, "this kernel implements C_in = 1 (MNIST-style first layer)"
    assert w.shape[1:] == (1, 3, 3), "expected 3x3 weights of shape (C_out, 1, 3, 3)"
    B, _, H, W = x.shape
    C_out = w.shape[0]
    x = x.contiguous()
    w_flat = w.contiguous().view(C_out, 9)
    b = b.contiguous()
    y = torch.empty((B, C_out, H - 2, W - 2), device=x.device, dtype=x.dtype)
    return x, w_flat, b, y, B, H, W, C_out


def _conv_impl(x, w, b, BLOCK_H, BLOCK_W, BLOCK_C, kernel_fn):
    x, w_flat, b, y, B, H, W, C_out = _conv_prepare(x, w, b)
    out_h, out_w = H - 2, W - 2
    grid = (B * triton.cdiv(out_h, BLOCK_H) * triton.cdiv(out_w, BLOCK_W) * triton.cdiv(C_out, BLOCK_C),)
    kernel_fn[grid](
        x, w_flat, b, y,
        B=B, H=H, W=W, C_out=C_out, out_h=out_h, out_w=out_w,
        stride_b=x.stride(0), stride_h=x.stride(2), stride_w=x.stride(3),
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C,
    )
    return y


def _conv_optimized_impl(x, w, b, BLOCK_OH, BLOCK_OW, BLOCK_C):
    x, w_flat, b, y, B, H, W, C_out = _conv_prepare(x, w, b)
    out_h, out_w = H - 2, W - 2
    grid = (B * triton.cdiv(out_h, BLOCK_OH) * triton.cdiv(out_w, BLOCK_OW) * triton.cdiv(C_out, BLOCK_C),)
    _conv_optimized_kernel[grid](
        x, w_flat, b, y,
        B, H, W, C_out, out_h, out_w,
        x.stride(0), x.stride(2), x.stride(3),
        BLOCK_OH=BLOCK_OH, BLOCK_OW=BLOCK_OW, BLOCK_C=BLOCK_C,
    )
    return y


def conv2d_baseline(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    BLOCK_H: int = 8,
    BLOCK_W: int = 8,
    BLOCK_C: int = 16,
) -> torch.Tensor:
    """Baseline 3x3 conv2d (no padding). x: (B, 1, H, W), w: (C_out, 1, 3, 3), b: (C_out,)."""
    return _conv_impl(x, w, b, BLOCK_H, BLOCK_W, BLOCK_C, _conv_baseline_kernel)


def conv2d_optimized(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    BLOCK_OH: int = 8,
    BLOCK_OW: int = 8,
    BLOCK_C: int = 32,
) -> torch.Tensor:
    """Optimized 3x3 conv2d: load 9 (BLOCK_OH, BLOCK_OW) blocks once, reuse across BLOCK_C output channels."""
    return _conv_optimized_impl(x, w, b, BLOCK_OH, BLOCK_OW, BLOCK_C)


# -----------------------------------------------------------------------------
# Autotuned
# -----------------------------------------------------------------------------


_conv_autotuned_kernel = triton.autotune(
    configs=[
        triton.Config({"BLOCK_OH": oh, "BLOCK_OW": ow, "BLOCK_C": c}, num_warps=w)
        for oh, ow, c, w in [
            (8, 8, 16, 4), (8, 16, 16, 4), (16, 16, 16, 4),
            (8, 8, 32, 4), (8, 16, 32, 4), (16, 16, 32, 8),
        ]
    ],
    key=["B", "out_h", "out_w", "C_out"],
)(_conv_optimized_kernel)


def conv2d_triton(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    BLOCK_H: int | None = None,
    BLOCK_W: int | None = None,
    BLOCK_C: int | None = None,
    use_autotune: bool = True,
) -> torch.Tensor:
    """3x3 valid Conv2D, C_in = 1. FP16/BF16/FP32. Autotuned unless explicit block sizes are given."""
    if use_autotune and BLOCK_H is None and BLOCK_W is None and BLOCK_C is None:
        x, w_flat, b, y, B, H, W, C_out = _conv_prepare(x, w, b)
        out_h, out_w = H - 2, W - 2
        grid = lambda meta: (  # noqa: E731
            B * triton.cdiv(out_h, meta["BLOCK_OH"]) * triton.cdiv(out_w, meta["BLOCK_OW"])
            * triton.cdiv(C_out, meta["BLOCK_C"]),
        )
        _conv_autotuned_kernel[grid](
            x, w_flat, b, y, B, H, W, C_out, out_h, out_w,
            x.stride(0), x.stride(2), x.stride(3),
        )
        return y
    return _conv_optimized_impl(x, w, b, BLOCK_H or 8, BLOCK_W or 16, BLOCK_C or 32)


# -----------------------------------------------------------------------------
# Benchmark vs PyTorch
# -----------------------------------------------------------------------------


def benchmark_conv(
    B: int = 4,
    H: int = 64,
    W: int = 64,
    C_out: int = 64,
    dtype: torch.dtype = torch.float16,
    warmup: int = 10,
    repeat: int = 50,
) -> dict[str, float]:
    """Returns pytorch_ms, triton_baseline_ms, triton_optimized_ms, triton_autotune_ms."""
    x = torch.randn(B, 1, H, W, device="cuda", dtype=dtype)
    w = torch.randn(C_out, 1, 3, 3, device="cuda", dtype=dtype)
    b = torch.zeros(C_out, device="cuda", dtype=dtype)

    def run(fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = __import__("time").perf_counter()
        for _ in range(repeat):
            fn()
        torch.cuda.synchronize()
        return (__import__("time").perf_counter() - start) * 1000 / repeat

    ref = torch.nn.functional.conv2d(x.float(), w.float(), b.float(), padding=0).to(dtype)
    # Correctness gate: never time a kernel that produces the wrong answer.
    for name, fn in [("baseline", conv2d_baseline), ("optimized", conv2d_optimized), ("autotune", conv2d_triton)]:
        torch.testing.assert_close(fn(x, w, b), ref, atol=5e-2, rtol=1e-2, msg=lambda m, n=name: f"conv2d {n}: {m}")

    out = {}
    out["pytorch_ms"] = run(lambda: torch.nn.functional.conv2d(x, w, b, padding=0))
    out["triton_baseline_ms"] = run(lambda: conv2d_baseline(x, w, b))
    out["triton_optimized_ms"] = run(lambda: conv2d_optimized(x, w, b))
    out["triton_autotune_ms"] = run(lambda: conv2d_triton(x, w, b, use_autotune=True))
    return out


# Legacy name
def conv2d_triton_fp16(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, **kwargs) -> torch.Tensor:
    return conv2d_triton(x, w, b, use_autotune=False, BLOCK_H=16, BLOCK_W=16, BLOCK_C=32)


if __name__ == "__main__":
    B, H, W, C_out = 4, 28, 28, 32
    x = torch.randn(B, 1, H, W, device="cuda", dtype=torch.float16)
    w = torch.randn(C_out, 1, 3, 3, device="cuda", dtype=torch.float16)
    b = torch.zeros(C_out, device="cuda", dtype=torch.float16)
    y = conv2d_triton(x, w, b, use_autotune=True)
    ref = torch.nn.functional.conv2d(x.float(), w.float(), b.float(), padding=0).half()
    print("conv2d_triton shape:", y.shape, "max diff:", (y - ref).abs().max().item())
    print("Benchmark:", benchmark_conv(B=4, H=64, W=64, C_out=64))
