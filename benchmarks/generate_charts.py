"""
Generate benchmark charts for the repo (RTX 3050–friendly).

Saves PNGs to benchmarks/:
  - matrix_mul_speedup.png  (from benchmark_matmul_results.csv)
  - conv_benchmark.png      (from benchmark_conv_results.csv)
  - mnist_acc_loss.png      (optional, from pytorch/mnist_gpu.py)

Run from repo root:
  python benchmarks/generate_charts.py
  python benchmarks/generate_charts.py --skip-mnist   # skip MNIST training (~few min)
"""
from __future__ import annotations

import argparse
import csv
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

    # Charts come only from the verified CSVs written by the benchmark scripts: this file must
    # never time anything itself or carry numbers of its own (docs/ENGINEERING_LOG.md #17).
    print("1. Matrix multiply chart...")
    _plot_from_csv("benchmark_matmul_results.csv", "size", "matrix_mul_speedup.png", "Matmul latency (FP16, RTX 3050)")

    print("2. Conv benchmark chart...")
    _plot_from_csv("benchmark_conv_results.csv", "config", "conv_benchmark.png", "3x3 conv latency (RTX 3050)")

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


def _plot_from_csv(csv_name: str, group_key: str, png_name: str, title: str) -> None:
    """Grouped latency bars from a benchmark CSV. Missing implementations simply have no bar."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("   matplotlib required")
        return
    csv_path = BENCHMARKS_DIR / csv_name
    if not csv_path.exists():
        print(f"   ({csv_name} not found: run the benchmark first)")
        return
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups = list(dict.fromkeys(r[group_key] for r in rows))
    impls = list(dict.fromkeys(r["implementation"] for r in rows))
    width = 0.8 / len(impls)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, impl in enumerate(impls):
        points = [(groups.index(r[group_key]), float(r["latency_ms"])) for r in rows if r["implementation"] == impl]
        xs = [p[0] + i * width - 0.4 + width / 2 for p in points]
        bars = ax.bar(xs, [p[1] for p in points], width, label=impl, color=f"C{i}")
        ax.bar_label(bars, fmt="%.3f", fontsize=7)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(BENCHMARKS_DIR / png_name, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"   -> {BENCHMARKS_DIR / png_name}")


if __name__ == "__main__":
    main()
