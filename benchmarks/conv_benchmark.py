"""
Conv benchmark: torch (and optional extension/triton) on 3×3 conv.
CI (no GPU): runs torch on CPU and prints timing.
Local (CUDA): run from repo root after pip install in extension/; can compare extension & Triton.
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Allow extension/mnist_custom_conv to find triton when run from repo root
_root = Path(__file__).resolve().parent.parent
if (_root / "triton").is_dir():
    sys.path.insert(0, str(_root / "triton"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B, C_in, H, W = 128, 1, 28, 28
C_out = 32
n_iters = 50

def main():
    torch_conv = nn.Conv2d(C_in, C_out, 3, padding=0).to(device)
    x = torch.randn(B, C_in, H, W, device=device)

    # Torch baseline
    for _ in range(5):
        _ = torch_conv(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = torch_conv(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_torch = (time.perf_counter() - start) * 1000 / n_iters
    print(f"conv_benchmark | device={device.type} B={B} | torch.nn.Conv2d: {t_torch:.4f} ms")

    # Optional: custom_conv (requires pip install in extension/)
    try:
        import custom_conv
        w, b = torch_conv.weight, torch_conv.bias
        for _ in range(5):
            _ = custom_conv.custom_conv2d(x, w, b)[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = custom_conv.custom_conv2d(x, w, b)[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_custom = (time.perf_counter() - start) * 1000 / n_iters
        print(f"conv_benchmark | custom_conv2d: {t_custom:.4f} ms (speedup vs torch: {t_torch/max(t_custom,1e-9):.2f}x)")
    except ImportError:
        print("conv_benchmark | custom_conv not installed (OK in CI)")

    # Optional: Triton
    try:
        from conv_triton import conv2d_triton_fp16
        x_h = x.half()
        w_h = torch_conv.weight.half()
        b_h = torch_conv.bias.half()
        for _ in range(5):
            _ = conv2d_triton_fp16(x_h, w_h, b_h)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = conv2d_triton_fp16(x_h, w_h, b_h)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_tr = (time.perf_counter() - start) * 1000 / n_iters
        print(f"conv_benchmark | triton: {t_tr:.4f} ms (speedup vs torch: {t_torch/max(t_tr,1e-9):.2f}x)")
    except Exception as e:
        print("conv_benchmark | triton skipped:", e)

    print("conv_benchmark done.")

if __name__ == "__main__":
    main()
