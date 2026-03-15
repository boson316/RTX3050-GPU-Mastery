"""
Generate GPU architecture and kernel execution diagrams for the docs.

Outputs (PNG) to docs/images/:
  1. gpu_memory_hierarchy.png  — Global, L2, Shared, Registers
  2. cuda_thread_hierarchy.png — Grid, Block, Warp, Thread
  3. flashattention_tiling.png — FlashAttention tiling strategy

Run from repo root: python docs/scripts/generate_diagrams.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IMAGES_DIR = REPO_ROOT / "docs" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    return plt, mpatches


def draw_memory_hierarchy():
    plt, mpatches = _setup_mpl()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # Stack from bottom (slow) to top (fast) — or left to right
    layers = [
        ("Global memory\n(VRAM)", "200–400 cycles", "Lower BW", "Full GPU", "#4a90d9"),
        ("L2 cache", "Variable", "High BW", "GPU-wide", "#7eb8da"),
        ("Shared memory", "~20–30 cycles", "Very high BW", "Per block", "#a8d4e6"),
        ("Registers", "~0 cycles", "Highest BW", "Per thread", "#c5e7f2"),
    ]
    y0 = 1.5
    h = 1.8
    for i, (name, latency, bw, scope, color) in enumerate(layers):
        y = y0 + i * h
        rect = mpatches.FancyBboxPatch((1.5, y), 7, h - 0.15, boxstyle="round,pad=0.02",
                                        facecolor=color, edgecolor="black", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(5, y + (h - 0.15) / 2, name, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(8.2, y + (h - 0.15) / 2, f"{latency}\n{bw}\n{scope}", ha="left", va="center", fontsize=8)

    ax.text(5, 9.2, "GPU memory hierarchy\n(fast ← → slow)", ha="center", fontsize=14, fontweight="bold")
    ax.text(5, 0.5, "↑ Faster, smaller scope", ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    out = IMAGES_DIR / "gpu_memory_hierarchy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    return out


def draw_cuda_thread_hierarchy():
    plt, mpatches = _setup_mpl()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Grid (outer)
    grid = mpatches.FancyBboxPatch((0.5, 5.5), 9, 4, boxstyle="round,pad=0.05",
                                    facecolor="#e8f4f8", edgecolor="#2c5aa0", linewidth=2)
    ax.add_patch(grid)
    ax.text(5, 9.1, "Grid (e.g. 1D or 2D array of blocks)", ha="center", fontsize=12, fontweight="bold")

    # Blocks inside grid
    for i in range(3):
        bx = 1.2 + i * 2.9
        block = mpatches.FancyBboxPatch((bx, 5.9), 2.5, 3.3, boxstyle="round,pad=0.03",
                                         facecolor="#b8d4e8", edgecolor="#1e5a8c", linewidth=1.5)
        ax.add_patch(block)
        ax.text(bx + 1.25, 8.9, f"Block {i}", ha="center", fontsize=10, fontweight="bold")
        # Warps inside one block (expand middle block)
        if i == 1:
            for w in range(2):
                wx = bx + 0.15
                wy = 7.4 - w * 0.7
                warp = mpatches.Rectangle((wx, wy), 2.2, 0.55, facecolor="#7eb8da", edgecolor="#0d3d6b")
                ax.add_patch(warp)
                ax.text(wx + 1.1, wy + 0.27, f"Warp (32 threads)", ha="center", fontsize=8)
            ax.text(bx + 1.25, 6.6, "… more warps", ha="center", fontsize=8, style="italic")
        else:
            ax.text(bx + 1.25, 7.5, "Block\n(warps)", ha="center", fontsize=9)

    # Arrow and Thread detail (below)
    ax.annotate("", xy=(5, 4.8), xytext=(5, 5.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(5.4, 5.15, "contains", fontsize=9)

    # One warp expanded: threads
    warp_box = mpatches.FancyBboxPatch((2.5, 2.2), 5, 2.2, boxstyle="round,pad=0.03",
                                        facecolor="#a8d4e6", edgecolor="#1e5a8c", linewidth=1.5)
    ax.add_patch(warp_box)
    ax.text(5, 4.2, "Warp = 32 threads (execute in lockstep)", ha="center", fontsize=11, fontweight="bold")
    for t in range(8):
        tx = 3.0 + t * 0.55
        ax.add_patch(mpatches.Rectangle((tx, 2.6), 0.45, 0.5, facecolor="#c5e7f2", edgecolor="gray"))
        ax.text(tx + 0.22, 2.85, str(t), ha="center", fontsize=7)
    ax.text(7.5, 2.85, "… 32 threads", ha="center", fontsize=9)
    ax.text(5, 1.6, "Thread (smallest unit of execution)", ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    out = IMAGES_DIR / "cuda_thread_hierarchy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    return out


def draw_flashattention_tiling():
    plt, mpatches = _setup_mpl()
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Title
    ax.text(5.5, 7.5, "FlashAttention tiling strategy", ha="center", fontsize=14, fontweight="bold")
    ax.text(5.5, 7.0, "Avoid materializing full S×S attention matrix; work in blocks in SRAM/shared memory", ha="center", fontsize=9)

    # Q matrix (S × D) — split by rows into BLOCK_M
    ax.add_patch(mpatches.Rectangle((0.5, 4.5), 2.5, 2.2, facecolor="#d4e8f7", edgecolor="#1e5a8c", linewidth=1.5))
    ax.text(1.75, 5.6, "Q", ha="center", fontsize=12, fontweight="bold")
    ax.text(1.75, 5.1, "S × D", ha="center", fontsize=10)
    for i in range(3):
        y = 4.7 + i * 0.6
        ax.axhline(y, xmin=0.5/11, xmax=3/11, color="gray", linewidth=0.8)
    ax.text(1.75, 4.3, "BLOCK_M rows", ha="center", fontsize=9, style="italic")

    # K, V (S × D) — split by rows into BLOCK_N
    ax.add_patch(mpatches.Rectangle((4.0, 4.5), 2.5, 2.2, facecolor="#e8f4d4", edgecolor="#5a8c1e", linewidth=1.5))
    ax.text(5.25, 5.6, "K, V", ha="center", fontsize=12, fontweight="bold")
    ax.text(5.25, 5.1, "S × D", ha="center", fontsize=10)
    ax.text(5.25, 4.3, "BLOCK_N rows", ha="center", fontsize=9, style="italic")

    # Arrow: load to SRAM
    ax.annotate("", xy=(3.8, 5.6), xytext=(3.2, 5.6), arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"))
    ax.annotate("", xy=(6.4, 5.6), xytext=(6.6, 5.6), arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"))
    ax.text(5.1, 5.9, "Load blocks to SRAM / shared memory", ha="center", fontsize=9)

    # SRAM box: Q_tile, K_tile, V_tile → scores → output
    sram = mpatches.FancyBboxPatch((6.8, 3.8), 3.5, 2.8, boxstyle="round,pad=0.04",
                                   facecolor="#f0f0f0", edgecolor="black", linewidth=1.2)
    ax.add_patch(sram)
    ax.text(8.55, 6.4, "SRAM / shared memory", ha="center", fontsize=10, fontweight="bold")
    ax.text(8.55, 5.95, "Q_tile [BLOCK_M × D]", ha="center", fontsize=9)
    ax.text(8.55, 5.5, "K_tile, V_tile [BLOCK_N × D]", ha="center", fontsize=9)
    ax.text(8.55, 5.0, "scores = Q_tile @ K_tileᵀ  (BLOCK_M × BLOCK_N)", ha="center", fontsize=8)
    ax.text(8.55, 4.5, "Online softmax → acc += P @ V_tile", ha="center", fontsize=8)

    # Output O (S × D) — one block at a time
    ax.add_patch(mpatches.Rectangle((6.8, 1.2), 3.5, 1.8, facecolor="#f7e8d4", edgecolor="#8c5a1e", linewidth=1.5))
    ax.text(8.55, 2.1, "Output block [BLOCK_M × D]", ha="center", fontsize=10)
    ax.text(8.55, 1.5, "Write to global (no full S×S stored)", ha="center", fontsize=9)
    ax.annotate("", xy=(8.55, 2.95), xytext=(8.55, 3.8), arrowprops=dict(arrowstyle="->", lw=1.2))

    # Loop over K/V blocks
    ax.text(5.25, 3.2, "Loop over K/V blocks\n(BLOCK_N)", ha="center", fontsize=9, style="italic")
    ax.annotate("", xy=(5.25, 4.0), xytext=(5.25, 4.5), arrowprops=dict(arrowstyle="->", lw=1, color="green"))
    ax.text(1.75, 3.2, "One Q block\n(BLOCK_M)", ha="center", fontsize=9, style="italic")
    ax.annotate("", xy=(1.75, 4.0), xytext=(1.75, 4.5), arrowprops=dict(arrowstyle="->", lw=1, color="green"))

    # Key idea
    ax.text(5.5, 0.6, "Key: For each Q block, iterate over all K/V blocks; online softmax keeps O(1) state per row.", ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    out = IMAGES_DIR / "flashattention_tiling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    return out


def main():
    try:
        import matplotlib
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        sys.exit(1)
    draw_memory_hierarchy()
    draw_cuda_thread_hierarchy()
    draw_flashattention_tiling()
    print(f"All diagrams saved to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
