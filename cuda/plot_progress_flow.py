"""
畫出 CUDA + PyTorch 學習進度流程圖，存成 progress_flow.png
對應 progress_flow.mmd 的 Mermaid 內容（用 matplotlib 畫，不需安裝 mermaid-cli）
執行：python plot_progress_flow.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 流程節點（文字縮短，避免超出框）
nodes = [
    ("Vector Add\n500x", "#90EE90"),
    ("Matrix Mul\nN=1024 tiled", "#90EE90"),
    ("Reduction\n1M <1ms", "#90EE90"),
    ("MNIST CNN\n99%+", "#FFD700"),
    ("Custom CUDA\nPyTorch ext", "#87CEEB"),
    ("GitHub ready", "#FFB6C1"),
]

fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 3)
ax.set_aspect("equal")
ax.axis("off")

box_w, box_h = 1.35, 0.65
y_center = 1.5
n = len(nodes)
dx = 11.0 / (n + 1)
x_positions = [dx * (i + 1) for i in range(n)]

for i, (text, color) in enumerate(nodes):
    x = x_positions[i]
    rect = mpatches.FancyBboxPatch(
        (x - box_w / 2, y_center - box_h / 2), box_w, box_h,
        boxstyle="round,pad=0.02", facecolor=color, edgecolor="gray", linewidth=1
    )
    ax.add_patch(rect)
    ax.text(x, y_center, text, ha="center", va="center", fontsize=8, multialignment="center")

for i in range(n - 1):
    x1, x2 = x_positions[i] + box_w / 2, x_positions[i + 1] - box_w / 2
    ax.annotate("", xy=(x2, y_center), xytext=(x1, y_center),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

plt.tight_layout()
plt.savefig("progress_flow.png", dpi=120, bbox_inches="tight")
plt.close()
print("已存檔: progress_flow.png")
