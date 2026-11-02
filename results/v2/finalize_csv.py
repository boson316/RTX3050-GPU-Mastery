"""For each benchmark row, take the row of the median-latency run (runs 1-3) so all columns come from one real run."""
from __future__ import annotations

import csv
import json
from pathlib import Path

BASE = Path("results/v2/csv")
METRICS = ("latency_ms", "gflops", "memory_bandwidth_gbs", "gpu_utilization_pct")
spreads = {}
for name in ("matmul", "conv", "attention", "transformer"):
    fname = f"benchmark_{name}_results.csv"
    runs = []
    for r in (1, 2, 3):
        with open(BASE / f"run{r}" / fname, newline="", encoding="utf-8") as f:
            runs.append(list(csv.DictReader(f)))
    fields = list(runs[0][0])
    keys = [k for k in fields if k not in METRICS]
    out = []
    for row in runs[0]:
        key = tuple(row[k] for k in keys)
        same = [next(x for x in run if tuple(x[k] for k in keys) == key) for run in runs]
        same.sort(key=lambda x: float(x["latency_ms"]))
        out.append(same[1])
        lo, hi = float(same[0]["latency_ms"]), float(same[2]["latency_ms"])
        spreads[f"{name}|{'|'.join(key)}"] = {"min": lo, "median": float(same[1]["latency_ms"]), "max": hi}
    for path in (Path("benchmarks") / fname, BASE / "final" / fname):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(out)
(BASE / "final" / "latency_ranges.json").write_text(json.dumps(spreads, indent=1), encoding="utf-8")
print(f"{len(spreads)} rows")
