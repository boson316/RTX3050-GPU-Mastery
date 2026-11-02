"""Which SDPA backend runs here, what would a matrix-materializing attention allocate, and how accurate is each path?"""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from flash_attention_simple.attention_cuda import attention_cuda
from triton_kernels.flash_attention import flash_attention_triton

print("cudnn", torch.backends.cudnn.version(), "flash_sdp", torch.backends.cuda.flash_sdp_enabled(),
      "mem_eff_sdp", torch.backends.cuda.mem_efficient_sdp_enabled(), "math_sdp", torch.backends.cuda.math_sdp_enabled())
torch.manual_seed(0)
for S in (1024, 2048, 4096):
    q, k, v = (torch.randn(1, 8, S, 64, device="cuda") for _ in range(3))
    print(f"S={S} _fused_sdp_choice={torch._fused_sdp_choice(q, k, v)}")

    def naive():
        return torch.softmax(q @ k.transpose(-1, -2) / 64**0.5, -1) @ v

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = naive()
    print(f"  naive q@k^T peak_extra_mib={(torch.cuda.max_memory_allocated() - base) / 2**20:.1f}")
    del out
    ref = torch.softmax(q.double() @ k.double().transpose(-1, -2) / 64**0.5, -1) @ v.double()
    for name, fn in (("SDPA", lambda: F.scaled_dot_product_attention(q, k, v)),
                     ("Triton", lambda: flash_attention_triton(q, k, v, causal=False)),
                     ("CUDA", lambda: attention_cuda(q, k, v, causal=False))):
        print(f"  {name:7s} max_abs_err_vs_FP64={(fn().double() - ref).abs().max().item():.3e}")
