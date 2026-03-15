"""Minimal GPU workload for ncu quick mode: one matmul, ~1 min to profile."""
import torch
if torch.cuda.is_available():
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    torch.mm(a, b)
    torch.cuda.synchronize()
