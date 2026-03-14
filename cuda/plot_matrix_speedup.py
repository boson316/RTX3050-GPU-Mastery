"""
Matrix Multiply Speedup 圖（RTX 3050）
用你實際跑 build_matrixMul.bat 的數據：N=1024，CPU vs GPU time
執行：python plot_matrix_speedup.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 你的實際數據（N=1024）
N = 1024
cpu_ms = 2738.25   # CPU time (ms)
gpu_naive_ms = 7.89
gpu_shared_ms = 5.26

speedup = cpu_ms / gpu_shared_ms  # ~520x

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# 左：Time (ms) 長條圖
labels = ["CPU", "GPU (naive)", "GPU (shared)"]
times = [cpu_ms, gpu_naive_ms, gpu_shared_ms]
colors = ["#ff7f7f", "#7fbf7f", "#7f7fff"]
bars = ax1.bar(labels, times, color=colors, edgecolor="gray")
ax1.set_ylabel("Time (ms)")
ax1.set_title(f"Matrix Multiply N={N}×{N} (RTX 3050)")
ax1.set_yscale("log")
for b, t in zip(bars, times):
    ax1.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.05, f"{t:.1f}", ha="center", fontsize=10)

# 右：Speedup 標示
ax2.text(0.5, 0.6, f"Speedup\n{speedup:.0f}×", ha="center", va="center", fontsize=28, fontweight="bold")
ax2.text(0.5, 0.25, f"CPU {cpu_ms:.0f} ms  →  GPU {gpu_shared_ms:.2f} ms", ha="center", va="center", fontsize=11)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis("off")
ax2.set_title("GPU vs CPU")

plt.suptitle("GPU ML Demo: Matrix Multiply (CUDA shared memory)", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("matrix_mul_speedup.png", dpi=120, bbox_inches="tight")
plt.close()
print("已存檔: matrix_mul_speedup.png")
