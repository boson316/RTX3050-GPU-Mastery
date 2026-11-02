# ruff: noqa
# Verbatim check script from the v2 task; its pytest form is tests/test_cuda_extensions.py.
import sys, torch, torch.nn.functional as F
sys.path.insert(0, ".")
torch.manual_seed(0)
dev = "cuda"
# 1) CUDA FlashAttention (FP32, D<=64)
from flash_attention_simple.attention_cuda import attention_cuda
for S, causal in [(100, False), (100, True), (512, False), (1024, True)]:
    q, k, v = (torch.randn(2, 8, S, 64, device=dev) for _ in range(3))
    out = attention_cuda(q, k, v, causal=causal)
    assert out is not None, "extension failed to build - see stderr"
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v, is_causal=causal), atol=1e-4, rtol=1e-3)
print("flash_attention_cuda OK")
# 2) Transformer kernels (FP16)
from gpu_kernels.transformer import transformer_cuda as tc
M, H = 2048, 768
x = torch.randn(M, H, device=dev, dtype=torch.half)
w = torch.randn(H, 3 * H, device=dev, dtype=torch.half) / H**0.5
b = torch.randn(3 * H, device=dev, dtype=torch.half) * 0.1
torch.testing.assert_close(tc.fused_qkv_cuda(x, w, b).float(), x.float() @ w.float() + b.float(), atol=2e-2, rtol=2e-2)
xs = torch.randn(M, 1000, device=dev, dtype=torch.half) * 3
torch.testing.assert_close(tc.softmax_cuda(xs).float(), torch.softmax(xs.float(), -1), atol=1e-3, rtol=1e-2)
g, be = torch.randn(H, device=dev, dtype=torch.half), torch.randn(H, device=dev, dtype=torch.half)
xl = torch.randn(M, H, device=dev, dtype=torch.half) * 2 + 1
torch.testing.assert_close(tc.layernorm_cuda(xl, g, be).float(), F.layer_norm(xl.float(), (H,), g.float(), be.float()), atol=2e-2, rtol=2e-2)
torch.testing.assert_close(tc.gelu_cuda(xs).float(), F.gelu(xs.float(), approximate="tanh"), atol=1e-2, rtol=1e-2)
w1 = torch.randn(H, 4 * H, device=dev, dtype=torch.half) / H**0.5; b1 = torch.randn(4 * H, device=dev, dtype=torch.half) * 0.1
w2 = torch.randn(4 * H, H, device=dev, dtype=torch.half) / (4 * H)**0.5; b2 = torch.randn(H, device=dev, dtype=torch.half) * 0.1
ref = F.gelu(x.float() @ w1.float() + b1.float(), approximate="tanh") @ w2.float() + b2.float()
torch.testing.assert_close(tc.fused_mlp_cuda(x, w1, b1, w2, b2).float(), ref, atol=3e-2, rtol=3e-2)
print("transformer_cuda OK")
# 3) custom_conv (先執行：cd extension && pip install --no-build-isolation . && cd ..)
import custom_conv
for dt in (torch.float32, torch.half):
    xc = torch.randn(64, 1, 28, 28, device=dev, dtype=dt)
    wc = torch.randn(32, 1, 3, 3, device=dev, dtype=dt) / 3; bc = torch.randn(32, device=dev, dtype=dt) * 0.1
    torch.testing.assert_close(custom_conv.custom_conv2d(xc, wc, bc)[0].float(), F.conv2d(xc.float(), wc.float(), bc.float()), atol=1e-2 if dt == torch.half else 1e-4, rtol=1e-3)
print("custom_conv OK")
