"""
Performance dashboard: run all benchmarks, generate tables/charts, and write a markdown report.

Usage (from repo root):
  python tools/performance_dashboard.py
  python tools/performance_dashboard.py --skip-benchmarks   # only regenerate report from existing CSVs
  python tools/performance_dashboard.py --skip-charts       # do not run generate_charts.py
  python tools/performance_dashboard.py --skip-mnist       # pass to generate_charts (skip MNIST training)

Output:
  - benchmarks/performance_report.md
  - benchmarks/benchmark_*_results.csv (from benchmark scripts)
  - benchmarks/plots/*.png (from benchmark scripts)
  - benchmarks/*.png (from generate_charts: matrix_mul_speedup, conv_benchmark, etc.)
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Repo root: parent of tools/
REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
PLOTS_DIR = BENCHMARKS_DIR / "plots"
REPORT_PATH = BENCHMARKS_DIR / "performance_report.md"

# Benchmark scripts to run (from repo root)
BENCHMARK_SCRIPTS = [
    "benchmarks/benchmark_matmul.py",
    "benchmarks/benchmark_conv.py",
    "benchmarks/benchmark_attention.py",
    "benchmarks/benchmark_transformer.py",
]

# CSV files produced by the above (same order for labeling)
BENCHMARK_CSVS = [
    BENCHMARKS_DIR / "benchmark_matmul_results.csv",
    BENCHMARKS_DIR / "benchmark_conv_results.csv",
    BENCHMARKS_DIR / "benchmark_attention_results.csv",
    BENCHMARKS_DIR / "benchmark_transformer_results.csv",
]

BENCHMARK_TITLES = {
    "benchmark_matmul_results.csv": "Matrix Multiply (FP16)",
    "benchmark_conv_results.csv": "Conv2D (3×3, FP16)",
    "benchmark_attention_results.csv": "Attention (SDPA / Flash)",
    "benchmark_transformer_results.csv": "Transformer Kernels (QKV, Softmax, LayerNorm, GELU, MLP)",
}


def run_cmd(cmd: list[str], cwd: Path, timeout: int | None = 300) -> tuple[bool, str]:
    """Run command; return (success, stderr+stdout)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (result.stderr or "").strip() + "\n" + (result.stdout or "").strip()
        return result.returncode == 0, out
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except FileNotFoundError as e:
        return False, str(e)


def get_gpu_name() -> str:
    """Return GPU device name if CUDA available."""
    try:
        sys.path.insert(0, str(REPO_ROOT))
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0) or "NVIDIA GPU"
    except Exception:
        pass
    return "N/A (no CUDA)"


def load_csv(filepath: Path) -> list[dict[str, str]]:
    """Load CSV into list of dicts."""
    if not filepath.is_file():
        return []
    rows = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def csv_to_markdown_table(rows: list[dict], columns: list[str] | None = None) -> str:
    """Convert CSV rows to a markdown table."""
    if not rows:
        return "*No data*"
    if columns is None:
        columns = list(rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def collect_all_csvs() -> dict[str, list[dict]]:
    """Load all benchmark CSVs; key = filename."""
    data = {}
    for p in BENCHMARKS_DIR.glob("benchmark_*_results.csv"):
        data[p.name] = load_csv(p)
    return data


def collect_chart_paths() -> list[tuple[str, Path]]:
    """List charts in benchmarks/plots/ and benchmarks/*.png for the report."""
    chart_list: list[tuple[str, Path]] = []
    if PLOTS_DIR.is_dir():
        for f in sorted(PLOTS_DIR.glob("*.png")):
            chart_list.append((f"plots/{f.name}", f))
    for p in ["matrix_mul_speedup.png", "conv_benchmark.png", "mnist_acc_loss.png"]:
        fp = BENCHMARKS_DIR / p
        if fp.is_file():
            chart_list.append((p, fp))
    return chart_list


def write_report(
    csv_data: dict[str, list[dict]],
    chart_list: list[tuple[str, Path]],
    gpu_name: str,
    ran_benchmarks: bool,
    ran_charts: bool,
) -> None:
    """Write benchmarks/performance_report.md."""
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Performance Report",
        "",
        f"**Generated:** {now}  ",
        f"**Device:** {gpu_name}",
        "",
        "---",
        "",
        "## Summary",
        "",
    ]
    if ran_benchmarks:
        lines.append("- Benchmarks were run by the dashboard; results below.")
    else:
        lines.append("- Report built from existing CSV files (benchmarks were not run).")
    if ran_charts:
        lines.append("- Chart generation was run (e.g. `generate_charts.py`).")
    else:
        lines.append("- Chart generation was skipped.")
    lines.append("")
    # One-line highlights per benchmark
    lines.append("### Highlights")
    lines.append("")
    for csv_name, rows in csv_data.items():
        title = BENCHMARK_TITLES.get(csv_name, csv_name)
        if not rows:
            lines.append(f"- **{title}:** No data.")
            continue
        # Pick best by latency if column exists
        if "latency_ms" in rows[0]:
            try:
                best = min(rows, key=lambda r: float(r.get("latency_ms", float("inf"))))
                impl = best.get("implementation", "?")
                kernel = best.get("kernel", "")
                lat = best.get("latency_ms", "?")
                who = f"{kernel} ({impl})" if kernel else impl
                lines.append(f"- **{title}:** Best latency **{lat} ms** ({who}).")
            except (ValueError, TypeError):
                lines.append(f"- **{title}:** See table below.")
        else:
            lines.append(f"- **{title}:** See table below.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Performance Tables")
    lines.append("")
    for csv_name in ["benchmark_matmul_results.csv", "benchmark_conv_results.csv",
                     "benchmark_attention_results.csv", "benchmark_transformer_results.csv"]:
        rows = csv_data.get(csv_name, [])
        title = BENCHMARK_TITLES.get(csv_name, csv_name)
        lines.append(f"### {title}")
        lines.append("")
        if rows:
            lines.append(csv_to_markdown_table(rows))
        else:
            lines.append("*No data. Run the corresponding benchmark script.*")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Benchmark Charts")
    lines.append("")
    if chart_list:
        lines.append("Charts are saved in `benchmarks/plots/` and `benchmarks/*.png`.")
        lines.append("")
        for label, path in chart_list:
            lines.append(f"- `{label}`")
        lines.append("")
        lines.append("Example embedding (paths relative to this report in `benchmarks/`):")
        lines.append("")
        for label, _ in chart_list[:6]:  # limit to 6 in doc
            lines.append(f"![{label}]({label})")
        if len(chart_list) > 6:
            lines.append(f"*... and {len(chart_list) - 6} more in `benchmarks/plots/`*")
    else:
        lines.append("*No charts found. Run benchmarks and/or `python benchmarks/generate_charts.py`.*")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by `tools/performance_dashboard.py`*")
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written: {REPORT_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run benchmarks and generate performance report")
    ap.add_argument("--skip-benchmarks", action="store_true", help="Do not run benchmark scripts; only build report from existing CSVs")
    ap.add_argument("--skip-charts", action="store_true", help="Do not run generate_charts.py")
    ap.add_argument("--skip-mnist", action="store_true", help="Pass --skip-mnist to generate_charts.py")
    args = ap.parse_args()

    gpu_name = get_gpu_name()
    print(f"GPU: {gpu_name}")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Report output: {REPORT_PATH}")

    ran_benchmarks = False
    if not args.skip_benchmarks:
        for script in BENCHMARK_SCRIPTS:
            path = REPO_ROOT / script
            if not path.is_file():
                print(f"Skip (not found): {script}")
                continue
            print(f"Running {script} ...")
            ok, out = run_cmd([sys.executable, str(path)], REPO_ROOT)
            ran_benchmarks = ran_benchmarks or ok
            if not ok:
                print(f"  Warning: exit non-zero. Last lines:\n  " + "\n  ".join(out.strip().splitlines()[-5:]))
            else:
                print("  OK")
        print("Benchmarks done.")
    else:
        print("Skipping benchmark runs (--skip-benchmarks).")

    ran_charts = False
    if not args.skip_charts:
        gen_charts = REPO_ROOT / "benchmarks" / "generate_charts.py"
        if gen_charts.is_file():
            print("Running generate_charts.py ...")
            cmd = [sys.executable, str(gen_charts)]
            if args.skip_mnist:
                cmd.append("--skip-mnist")
            ok, _ = run_cmd(cmd, REPO_ROOT, timeout=600)
            ran_charts = ok
            print("  OK" if ok else "  Failed or skipped")
        else:
            print("generate_charts.py not found; skip.")
    else:
        print("Skipping chart generation (--skip-charts).")

    csv_data = collect_all_csvs()
    chart_list = collect_chart_paths()
    write_report(csv_data, chart_list, gpu_name, ran_benchmarks, ran_charts)
    print("Done.")


if __name__ == "__main__":
    main()
