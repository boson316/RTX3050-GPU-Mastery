"""
產出 Custom Conv2d benchmark 長條圖：benchmark_custom_conv.png
需先 pip install . 且能 import custom_conv。
執行：python plot_benchmark.py
"""

import sys
import time

try:
    import torch
    import torch.nn as nn
    import custom_conv
except ImportError as e:
    print("請先編譯 extension: pip install --no-build-isolation .", file=sys.stderr)
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cuda")
B, C_in, H, W = 128, 1, 28, 28
C_out = 32
torch_conv = nn.Conv2d(C_in, C_out, 3, padding=0).to(device)
x = torch.randn(B, C_in, H, W, device=device)


def run_torch(n_iters=100):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = torch_conv(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / n_iters


def run_custom(n_iters=100):
    w, b = torch_conv.weight, torch_conv.bias
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = custom_conv.custom_conv2d(x, w, b)[0]
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / n_iters


if __name__ == "__main__":
    run_torch(10)
    run_custom(10)
    t_torch = run_torch(100)
    t_custom = run_custom(100)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    labels = ["torch.nn.Conv2d", "custom_conv2d"]
    times = [t_torch, t_custom]
    colors = ["#7fbf7f", "#7f7fff"]
    bars = ax.bar(labels, times, color=colors, edgecolor="gray")
    ax.set_ylabel("Time (ms) per forward")
    ax.set_title(f"Custom 3×3 Conv2d Benchmark (B={B}, {C_in}→{C_out}, 100 iters)")
    for b, t in zip(bars, times):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002, f"{t:.4f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig("benchmark_custom_conv.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("已存檔: benchmark_custom_conv.png")
