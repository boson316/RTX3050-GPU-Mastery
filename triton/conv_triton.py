"""
Triton 3x3 Conv2d FP16：10x 生產力、code 少 80%。
單一 pid，每 program 一 tile × BLOCK_C channels，auto compile → PTX，sm_86 友善。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def triton_conv(
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
    """1D grid：pid → (batch, tile_h, tile_w, oc_block)。一 program 產出 BLOCK_H×BLOCK_W×BLOCK_C。"""
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
                + oh * out_w + ow + off_h[:, None] * out_w + off_w[None, :],
                acc.to(tl.float16),
                mask=mask_sp,
            )


def conv2d_triton_fp16(x, w, b, BLOCK_H=16, BLOCK_W=16, BLOCK_C=32):
    """3x3 conv FP16, no padding. x:(B,1,H,W), w:(C_out,1,3,3), b:(C_out,)."""
    B, _, H, W = x.shape
    C_out = w.shape[0]
    out_h, out_w = H - 2, W - 2
    x = x.contiguous().to(torch.float16)
    w = w.contiguous().to(torch.float16)
    b = b.contiguous().to(torch.float16)
    y = torch.empty((B, C_out, out_h, out_w), device=x.device, dtype=torch.float16)

    grid = (B * triton.cdiv(out_h, BLOCK_H) * triton.cdiv(out_w, BLOCK_W) * triton.cdiv(C_out, BLOCK_C),)
    triton_conv[grid](
        x, w, b, y,
        B=B, H=H, W=W, C_out=C_out, out_h=out_h, out_w=out_w,
        stride_b=x.stride(0), stride_h=x.stride(2), stride_w=x.stride(3),
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C,
    )
    return y
