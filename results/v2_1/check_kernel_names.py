# ruff: noqa
# v2.1 task check: confirm the warp paths and cublasLt kernels actually run.
# Changed from the task's snippet: CPU activity added, because torch 2.6 on Windows raises
# "Legacy CUDA profiling requires use_cpu=True" with CUDA alone.
import torch, sys; sys.path.insert(0, ".")
import torch.nn.functional as F
from gpu_kernels.transformer import transformer_cuda as tc
print("kineto_available:", torch.profiler.kineto_available())
x = torch.randn(2048, 768, device="cuda", dtype=torch.half)
g = torch.randn(768, device="cuda", dtype=torch.half); b = torch.randn(768, device="cuda", dtype=torch.half)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
    tc.softmax_cuda(x); tc.layernorm_cuda(x, g, b); tc.fused_qkv_cuda(x, torch.randn(768, 2304, device="cuda", dtype=torch.half), None)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15, max_name_column_width=60))
print("event names:", sorted({e.key for e in prof.key_averages()}))

# Which GELU form does the cublasLt epilogue follow? Max error of the MLP against each reference.
torch.manual_seed(0)
M, H = 2048, 768
xm = torch.randn(M, H, device="cuda", dtype=torch.half)
w1 = torch.randn(H, 4 * H, device="cuda", dtype=torch.half) / H**0.5; b1 = torch.randn(4 * H, device="cuda", dtype=torch.half) * 0.1
w2 = torch.randn(4 * H, H, device="cuda", dtype=torch.half) / (4 * H)**0.5; b2 = torch.randn(H, device="cuda", dtype=torch.half) * 0.1
out = tc.fused_mlp_cuda(xm, w1, b1, w2, b2).double()
for form in ("tanh", "none"):
    ref = F.gelu(xm.double() @ w1.double() + b1.double(), approximate=form) @ w2.double() + b2.double()
    print(f"MLP max_abs_err vs GELU approximate={form!r}: {(out - ref).abs().max().item():.3e}")
