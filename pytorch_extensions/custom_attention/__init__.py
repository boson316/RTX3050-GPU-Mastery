"""
Custom attention module: PyTorch SDPA backend with optional Triton Flash.
Use for benchmarking and drop-in in transformer blocks.
"""
import torch


def custom_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     causal: bool = False, use_triton: bool = False) -> torch.Tensor:
    """
    Scaled dot-product attention. q, k, v: (B, H, S, D).
    If use_triton=True and Triton available, use flash_attention_triton; else torch SDPA.
    """
    if use_triton:
        try:
            from triton_kernels.flash_attention import flash_attention_triton
            return flash_attention_triton(q, k, v, causal=causal)
        except Exception:
            pass
    scale = 1.0 / (q.size(-1) ** 0.5)
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal, scale=scale
    )


__all__ = ["custom_attention"]
