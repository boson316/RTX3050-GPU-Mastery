"""
Generate benchmark charts for the repo (RTX 3050–friendly).

Saves PNGs to benchmarks/:
  - matrix_mul_speedup.png  (from matmul benchmark or cuda plot)
  - conv_benchmark.png      (torch vs extension vs Triton)
  - mnist_acc_loss.png      (optional, from pytorch/mnist_gpu.py)

Run from repo root:
  python benchmarks/generate_charts.py
  python benchmarks/generate_charts.py --skip-mnist   # skip MNIST training (~few min)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"


def run(cmd: list[str], cwd: Path, env=None) -> bool:
    try:
        subprocess.run(cmd, cwd=cwd, check=True, env=env)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  Skip: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description="Generate benchmark charts")
    ap.add_argument("--skip-mnist", action="store_true", help="Do not run MNIST training (saves time)")
    args = ap.parse_args()

    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Matrix multiply speedup: use cuda/plot_matrix_speedup.py if present, else plot from matmul_benchmark data
    print("1. Matrix multiply chart...")
    cuda_plot = REPO_ROOT / "cuda" / "plot_matrix_speedup.py"
    if cuda_plot.exists():
        if run([sys.executable, str(cuda_plot)], REPO_ROOT / "cuda"):
            src = REPO_ROOT / "cuda" / "matrix_mul_speedup.png"
            if src.exists():
                dst = BENCHMARKS_DIR / "matrix_mul_speedup.png"
                dst.write_bytes(src.read_bytes())
                print(f"   -> {dst}")
    else:
        _plot_matmul_fallback()

    # 2. Conv benchmark chart: run conv_benchmark and plot
    print("2. Conv benchmark chart...")
    _plot_conv()

    # 3. MNIST acc/loss (optional)
    if not args.skip_mnist:
        print("3. MNIST acc/loss chart (training a few epochs)...")
        mnist = REPO_ROOT / "pytorch" / "mnist_gpu.py"
        if mnist.exists():
            if run([sys.executable, str(mnist)], REPO_ROOT / "pytorch"):
                src = REPO_ROOT / "pytorch" / "mnist_acc_loss.png"
                if src.exists():
                    dst = BENCHMARKS_DIR / "mnist_acc_loss.png"
                    dst.write_bytes(src.read_bytes())
                    print(f"   -> {dst}")
        else:
            print("   (pytorch/mnist_gpu.py not found)")
    else:
        print("3. MNIST skipped (use without --skip-mnist to generate mnist_acc_loss.png)")

    print("Done. Charts in", BENCHMARKS_DIR)


def _plot_matmul_fallback():
    """Plot matmul from live benchmark when cuda/plot_matrix_speedup.py is not used."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("   matplotlib required for fallback plot")
        return
    sys.path.insert(0, str(REPO_ROOT))
    import torch
    times_torch, times_triton = [], []
    sizes = [512, 1024, 2048]
    for N in sizes:
        M, K = N, N
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        for _ in range(10):
            torch.matmul(A, B)
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        for _ in range(50):
            torch.matmul(A, B)
        torch.cuda.synchronize()
        times_torch.append((time.perf_counter() - t0) * 1000 / 50)
        try:
            from triton_kernels.matmul import matmul_triton
            t0 = time.perf_counter()
            for _ in range(50):
                matmul_triton(A, B)
            torch.cuda.synchronize()
            times_triton.append((time.perf_counter() - t0) * 1000 / 50)
        except Exception:
            times_triton.append(None)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(sizes))
    ax.bar([i - 0.2 for i in x], times_torch, 0.4, label="PyTorch", color="C0")
    if any(t for t in times_triton if t is not None):
        ax.bar([i + 0.2 for i in x], [t or 0 for t in times_triton], 0.4, label="Triton", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Matrix size N×N")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Matmul benchmark (RTX 3050)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(BENCHMARKS_DIR / "matrix_mul_speedup.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"   -> {BENCHMARKS_DIR / 'matrix_mul_speedup.png'}")


def _plot_conv():
    """Run conv benchmark and plot torch vs extension vs Triton."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torch
        import torch.nn as nn
    except ImportError:
        print("   torch/matplotlib required")
        return
    sys.path.insert(0, str(REPO_ROOT))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C_in, H, W = 128, 1, 28, 28
    C_out = 32
    n = 50
    torch_conv = nn.Conv2d(C_in, C_out, 3, padding=0).to(device)
    x = torch.randn(B, C_in, H, W, device=device)
    labels, times = ["torch.nn.Conv2d"], []
    for _ in range(5):
        _ = torch_conv(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    for _ in range(n):
        _ = torch_conv(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1000 / n)

    try:
        import custom_conv
        w, b = torch_conv.weight, torch_conv.bias
        for _ in range(5):
            _ = custom_conv.custom_conv2d(x, w, b)[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            _ = custom_conv.custom_conv2d(x, w, b)[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        labels.append("custom_conv2d")
        times.append((time.perf_counter() - t0) * 1000 / n)
    except ImportError:
        pass

    try:
        from triton_kernels.conv import conv2d_triton_fp16
        x_h, w_h = x.half(), torch_conv.weight.half()
        b_h = torch_conv.bias.half()
        for _ in range(5):
            _ = conv2d_triton_fp16(x_h, w_h, b_h)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            _ = conv2d_triton_fp16(x_h, w_h, b_h)
        if device.type == "cuda":
            torch.cuda.synchronize()
        labels.append("Triton")
        times.append((time.perf_counter() - t0) * 1000 / n)
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2ecc71", "#3498db", "#9b59b6"][: len(labels)]
    ax.bar(labels, times, color=colors, edgecolor="gray")
    ax.set_ylabel("Time (ms) per forward")
    ax.set_title(f"3×3 Conv2d benchmark (B={B}, {C_in}→{C_out}, RTX 3050)")
    for i, t in enumerate(times):
        ax.text(i, t + 0.005, f"{t:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(BENCHMARKS_DIR / "conv_benchmark.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"   -> {BENCHMARKS_DIR / 'conv_benchmark.png'}")


if __name__ == "__main__":
    main()
