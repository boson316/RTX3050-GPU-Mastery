"""Median and range of the three v2.1 rounds, the v2.0 control next to them, and GPU state during each round.

Writes results/v2_1/csv/final/ (the row of the median-latency round, so all columns come from one real run)
and prints the comparison. Usage: python results/v2_1/aggregate.py
"""
from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V21 = ROOT / "results" / "v2_1" / "csv"
V20 = ROOT / "results" / "v2" / "csv"
LOGS = ROOT / "results" / "v2_1" / "logs"
TRANSFORMER = "benchmark_transformer_results.csv"
LONGSEQ = "benchmark_attention_longseq_results.csv"


def read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def medians(fname: str, keys: tuple[str, ...]) -> tuple[list[dict], list[dict], dict]:
    runs = [read(V21 / f"run{r}" / fname) for r in (1, 2, 3)]
    out, ranges = [], {}
    for row in runs[0]:
        key = tuple(row[k] for k in keys)
        same = sorted((next(x for x in run if tuple(x[k] for k in keys) == key) for run in runs), key=lambda x: float(x["latency_ms"]))
        out.append(same[1])
        ranges[key] = (float(same[0]["latency_ms"]), float(same[2]["latency_ms"]))
    return runs[0], out, ranges


def write(fname: str, rows: list[dict]) -> None:
    final = V21 / "final"
    final.mkdir(exist_ok=True)
    with open(final / fname, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def v20_values(impl_key: tuple[str, str]) -> tuple[float | None, float | None, float | None]:
    """Return (final latency, min, max) of a v2.0 transformer row, or Nones when v2.0 did not measure it."""
    vals = []
    for r in (1, 2, 3):
        for row in read(V20 / f"run{r}" / TRANSFORMER):
            if (row["kernel"], row["implementation"]) == impl_key:
                vals.append(float(row["latency_ms"]))
    if not vals:
        return None, None, None
    return st.median(vals), min(vals), max(vals)


def transformer() -> None:
    _, rows, ranges = medians(TRANSFORMER, ("kernel", "implementation"))
    write(TRANSFORMER, rows)
    print("kernel | impl | v2.1 median ms [min-max] | GB/s | v2.0 median ms [min-max] | change %")
    for row in rows:
        key = (row["kernel"], row["implementation"])
        lo, hi = ranges[key]
        lat = float(row["latency_ms"])
        med0, lo0, hi0 = v20_values(key)
        old = f"{med0:.4f} [{lo0:.4f}-{hi0:.4f}] | {(lat - med0) / med0 * 100:+.1f}%" if med0 else "- | -"
        print(f"{key[0]:20s} | {key[1]:8s} | {lat:.4f} [{lo:.4f}-{hi:.4f}] | {row['memory_bandwidth_gbs']:>6s} | {old}")


def longseq() -> None:
    _, rows, ranges = medians(LONGSEQ, ("S", "precision", "impl"))
    write(LONGSEQ, rows)
    print("\nS | precision | impl | median ms [min-max] | TFLOPS | max err vs FP64")
    for row in rows:
        key = (row["S"], row["precision"], row["impl"])
        lo, hi = ranges[key]
        print(f"{key[0]:>5s} | {key[1]:5s} | {key[2]:7s} | {float(row['latency_ms']):.4f} [{lo:.4f}-{hi:.4f}] | {row['tflops']:>6s} | {row['max_abs_err_vs_fp64']}")


def clocks() -> None:
    print("\ntrace | busy samples | SM median [min-max] MHz | memory MHz | W median | T max | used MiB median")
    for path in sorted(LOGS.glob("*_run[0-9]_clocks.csv")):
        rows = [[c.strip() for c in r] for r in csv.reader(open(path, newline=""))]
        busy = [r for r in rows if len(r) >= 9 and r[6].split()[0].isdigit() and int(r[6].split()[0]) >= 50]
        if not busy:
            print(f"{path.name}: no samples at 50% utilization or above")
            continue
        sm = [int(r[2].split()[0]) for r in busy]
        mem = sorted({int(r[3].split()[0]) for r in busy})
        watt = [float(r[4].split()[0]) for r in busy if r[4].split()[0].replace(".", "").isdigit()]
        used = [int(r[8].split()[0]) for r in busy]
        print(f"{path.name:32s} | {len(busy):3d} | {st.median(sm):.0f} [{min(sm)}-{max(sm)}] | {mem} | {st.median(watt):.1f} | {max(int(r[5]) for r in busy)} | {st.median(used):.0f}")


transformer()
longseq()
clocks()


def derived() -> None:
    """Every ratio quoted in the docs, computed here rather than by hand."""
    runs = {r: {(x["kernel"], x["implementation"]): float(x["latency_ms"]) for x in read(V21 / f"run{r}" / TRANSFORMER)} for r in (1, 2, 3)}
    rows20 = {r: {(x["kernel"], x["implementation"]): float(x["latency_ms"]) for x in read(V20 / f"run{r}" / TRANSFORMER)} for r in (1, 2, 3)}
    print("\nCUDA vs PyTorch in the same round, ms and %: v2.1 rounds then v2.0 rounds")
    for kernel in ("QKV", "MLP", "Softmax", "LayerNorm", "Softmax@8192x768", "LayerNorm@8192x768"):
        for label, data in (("v2.1", runs), ("v2.0", rows20)):
            if (kernel, "CUDA") not in data[1]:
                continue
            pairs = [(d[(kernel, "CUDA")], d[(kernel, "PyTorch")]) for d in data.values()]
            gaps = " ".join(f"{c - p:+.4f} ({(c - p) / p * 100:+.1f}%)" for c, p in pairs)
            print(f"{kernel:20s} {label}: {gaps}")
    t = {(x["kernel"], x["implementation"]): float(x["latency_ms"]) for x in read(V21 / "final" / TRANSFORMER)}
    print("\nTriton vs CUDA, median, %")
    for kernel in ("Softmax", "LayerNorm", "GELU"):
        small = (t[(kernel, "Triton")] / t[(kernel, "CUDA")] - 1) * 100
        large = (t[(f"{kernel}@8192x768", "Triton")] / t[(f"{kernel}@8192x768", "CUDA")] - 1) * 100
        print(f"{kernel:10s} 2048x768: {small:+.1f}%   8192x768: {large:+.1f}%")
    a = {(x["S"], x["precision"], x["impl"]): x for x in read(V21 / "final" / LONGSEQ)}
    print("\nlong sequence ratios, median")
    for s in ("1024", "2048", "4096"):
        lat = lambda p, i: float(a[(s, p, i)]["latency_ms"])  # noqa: E731
        print(f"S={s}: Triton FP16 vs SDPA FP16 {(lat('fp16', 'Triton') / lat('fp16', 'SDPA') - 1) * 100:+.1f}%; "
              f"CUDA vs SDPA FP32 {(lat('fp32', 'CUDA') / lat('fp32', 'SDPA') - 1) * 100:+.1f}%; "
              f"Triton IEEE vs SDPA FP32 {(lat('fp32', 'Triton') / lat('fp32', 'SDPA') - 1) * 100:+.1f}%; "
              f"Triton TF32 / Triton FP16 {lat('tf32', 'Triton') / lat('fp16', 'Triton'):.2f}x; "
              f"Triton IEEE / Triton TF32 {lat('fp32', 'Triton') / lat('tf32', 'Triton'):.2f}x")


derived()
