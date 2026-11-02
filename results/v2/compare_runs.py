"""Compare latency across kept benchmark runs and summarize the clock traces. Usage: compare_runs.py <run> <run> ..."""
from __future__ import annotations

import csv
import glob
import os
import statistics as st
import sys

BASE = "results/v2"


def load(run: str, name: str) -> dict:
    path = f"{BASE}/csv/run{run}/benchmark_{name}_results.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keyf = [k for k in rows[0] if k not in ("latency_ms", "gflops", "memory_bandwidth_gbs", "gpu_utilization_pct", "throughput_gbs", "tflops")]
    return {tuple(r[k] for k in keyf): float(r["latency_ms"]) for r in rows if r.get("latency_ms")}


runs = sys.argv[1:]
over = 0
for name in ("matmul", "conv", "attention", "transformer"):
    data = [load(r, name) for r in runs]
    print(f"=== {name}")
    for key in data[0]:
        vals = [d[key] for d in data if key in d]
        spread = (max(vals) - min(vals)) / min(vals) * 100
        flag = " >5%" if spread > 5 else ""
        over += spread > 5
        print(f"  {' | '.join(key):44s} {' '.join(f'{v:9.4f}' for v in vals)}  {spread:5.1f}%{flag}")
print(f"entries over 5%: {over}")
print("=== clocks while GPU util >= 50%")
for f in sorted(glob.glob(f"{BASE}/logs/*_run[0-9]_clocks.csv")):
    with open(f, newline="") as fh:
        rows = [r for r in csv.reader(fh) if len(r) >= 8 and r[2].strip().split()[0].isdigit()]
    busy = [r for r in rows if int(r[6].split()[0]) >= 50]
    if not busy:
        print(f"  {os.path.basename(f)}: no samples >= 50% ({len(rows)} samples)")
        continue
    sm = [int(r[2].split()[0]) for r in busy]
    watt = [float(r[4].split()[0]) for r in busy if r[4].split()[0] != "[N/A]"]
    print(f"  {os.path.basename(f):38s} n={len(busy):3d} SM med={st.median(sm):.0f} min={min(sm)} max={max(sm)} MHz  W med={st.median(watt) if watt else 0:.1f}  Tmax={max(int(r[5]) for r in busy)}")
