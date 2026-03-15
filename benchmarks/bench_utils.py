"""
Shared utilities for benchmark suite: timing, CSV, tables, matplotlib charts, GPU utilization.
All charts are saved under benchmarks/plots/.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import torch

# Default output dir for plots
REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BENCHMARKS_DIR / "plots"

# Optional: pynvml for GPU utilization (pip install nvidia-ml-py)
_pynvml = None
try:
    import pynvml
    _pynvml = pynvml
except ImportError:
    pass


def ensure_plots_dir() -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return PLOTS_DIR


def _bytes_tensors(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def run_bench(
    fn: Callable[[], Any],
    warmup: int = 10,
    repeat: int = 50,
    sync_cuda: bool = True,
) -> float:
    """Run fn() warmup + repeat times; return mean latency in ms."""
    for _ in range(warmup):
        fn()
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeat


def sample_gpu_utilization(
    fn: Callable[[], Any],
    repeat: int = 20,
    sample_interval_ms: float = 50,
) -> float | None:
    """
    Run fn() in a loop while sampling GPU utilization in a background thread.
    Returns average GPU utilization (0–100) or None if pynvml not available.
    """
    if not torch.cuda.is_available() or _pynvml is None:
        return None
    try:
        _pynvml.nvmlInit()
        h = _pynvml.nvmlDeviceGetHandleByIndex(0)
        samples: list[float] = []
        stop = threading.Event()

        def sampler():
            while not stop.is_set():
                try:
                    util = _pynvml.nvmlDeviceGetUtilizationRates(h)
                    samples.append(util.gpu)
                except Exception:
                    pass
                stop.wait(timeout=sample_interval_ms / 1000.0)

        t = threading.Thread(target=sampler, daemon=True)
        t.start()
        for _ in range(repeat):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stop.set()
        t.join(timeout=2.0)
        _pynvml.nvmlShutdown()
        return sum(samples) / len(samples) if samples else None
    except Exception:
        return None


def compute_metrics(
    latency_ms: float,
    bytes_read: int = 0,
    bytes_written: int = 0,
    flops: float | None = None,
) -> dict[str, float]:
    """Compute throughput (GFLOPS), memory bandwidth (GB/s) from latency and bytes/flops."""
    time_sec = latency_ms / 1000.0
    total_bytes = bytes_read + bytes_written
    bandwidth_gbs = total_bytes / 1e9 / time_sec if time_sec > 0 else 0.0
    gflops = (flops / 1e9 / time_sec) if (flops is not None and flops > 0 and time_sec > 0) else None
    out = {"latency_ms": latency_ms, "memory_bandwidth_gbs": bandwidth_gbs}
    if gflops is not None:
        out["gflops"] = gflops
    return out


def save_csv(
    rows: list[dict[str, Any]],
    filepath: Path | str,
    fieldnames: list[str] | None = None,
) -> Path:
    """Write list of dicts to CSV. fieldnames default to union of all keys."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return filepath
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return filepath


def print_benchmark_table(
    rows: list[dict[str, Any]],
    title: str = "Benchmark",
    columns: list[str] | None = None,
) -> None:
    """Print a simple text table. columns default to all keys in first row."""
    if not rows:
        return
    if columns is None:
        columns = list(rows[0].keys())
    widths = [max(len(str(c)), 4) for c in columns]
    for r in rows:
        for i, c in enumerate(columns):
            if c in r:
                widths[i] = max(widths[i], len(str(r[c])))
    header = " | ".join(c.ljust(widths[i]) for i, c in enumerate(columns))
    sep = "-+-".join("-" * w for w in widths)
    print(f"\n--- {title} ---")
    print(header)
    print(sep)
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(widths[i]) for i, c in enumerate(columns)))
    print()


def plot_bar_chart(
    labels: list[str],
    values: list[float],
    ylabel: str,
    title: str,
    filename: str,
    value_fmt: str = ".3f",
) -> Path:
    """Save a bar chart to benchmarks/plots/<filename>."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skip chart", filename, file=sys.stderr)
        return PLOTS_DIR / filename
    ensure_plots_dir()
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(labels))
    bars = ax.bar(x, values, color=["#2ecc71", "#3498db", "#9b59b6"][: len(labels)], edgecolor="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{v:{value_fmt}}", ha="center", fontsize=9)
    fig.tight_layout()
    out = PLOTS_DIR / filename
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def plot_grouped_bars(
    x_labels: list[str],
    implementations: list[str],
    data: dict[str, list[float]],
    ylabel: str,
    title: str,
    filename: str,
) -> Path:
    """Grouped bar chart: x = x_labels, groups = implementations, data[impl] = list of values."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return PLOTS_DIR / filename
    ensure_plots_dir()
    fig, ax = plt.subplots(figsize=(10, 5))
    n = len(x_labels)
    width = 0.8 / len(implementations)
    for i, impl in enumerate(implementations):
        vals = data.get(impl, [0.0] * n)
        offset = (i - len(implementations) / 2 + 0.5) * width
        ax.bar([j + offset for j in range(n)], vals, width, label=impl)
    ax.set_xticks(range(n))
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out = PLOTS_DIR / filename
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out


def plot_latency_throughput(
    rows: list[dict[str, Any]],
    impl_column: str,
    latency_col: str,
    gflops_col: str | None,
    title: str,
    filename: str,
) -> Path:
    """One subplot: latency by impl; optional second: GFLOPS by impl."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return PLOTS_DIR / filename
    ensure_plots_dir()
    impls = [r[impl_column] for r in rows]
    latencies = [r[latency_col] for r in rows]
    n = len(impls)
    if gflops_col and rows and gflops_col in rows[0]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        gflops = [r.get(gflops_col) for r in rows]
        gflops = [g if g is not None else 0.0 for g in gflops]
        ax2.bar(range(n), gflops, color=["#2ecc71", "#3498db", "#9b59b6"][:n])
        ax2.set_xticks(range(n))
        ax2.set_xticklabels(impls)
        ax2.set_ylabel("GFLOPS")
        ax2.set_title("Throughput")
    else:
        fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(range(n), latencies, color=["#2ecc71", "#3498db", "#9b59b6"][:n])
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(impls)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Latency")
    fig.suptitle(title)
    fig.tight_layout()
    out = PLOTS_DIR / filename
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    return out
