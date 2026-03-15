# Nsight Profiling Guide

This document explains how to use **NVIDIA Nsight Systems** and **Nsight Compute** to profile kernels in this repository, and how the profiling pipeline (scripts, reports, and metrics) is set up. Target: **RTX 3050**; applicable to any CUDA GPU.

---

## 1. Algorithm / workflow

Profiling in this repo is organized as:

1. **Nsight Systems (nsys):** Run the application under `nsys profile` to capture a **timeline** of kernel launches, memory copies, and (optionally) CPU activity.
2. **Nsight Compute (ncu):** Run the application under `ncu` to collect **per-kernel metrics** (occupancy, memory throughput, warp execution efficiency, duration).
3. **Export and parse:** ncu reports are exported to CSV (raw page); a Python script parses them and writes **JSON** for downstream use (e.g. roofline).
4. **Roofline:** A separate script uses theoretical FLOPs/bytes and (optionally) ncu metrics to plot the **roofline model** (see [roofline_analysis.md](roofline_analysis.md)).

---

## 2. GPU “memory access” for profiling

Profiling does not change kernel memory access patterns; it **measures** their effect:

- **Occupancy:** How many threads per SM vs theoretical max; influenced by registers and shared memory per block.
- **Memory throughput:** Fraction of peak DRAM/L2 bandwidth used; reflects coalescing and reuse.
- **Warp execution efficiency:** Fraction of threads in a warp that are active (low → warp divergence).
- **Timeline:** Order and duration of kernel launches; helps see overlap and bottlenecks.

So “memory access” in the sense of **how data is read/written** is optimized in the kernels (see [gpu_memory_hierarchy.md](gpu_memory_hierarchy.md)); Nsight gives you the **observed** utilization and efficiency.

---

## 3. Kernel launch configuration (for profiling)

You profile **whatever the application launches**. In this repo:

- **Default benchmark:** `python benchmarks/transformer_benchmark.py` (light: B=8, S=256, H=768).
- **Quick probe:** `profiling/_ncu_quick_probe.py` (one matmul) for a short ncu run (~1 min).
- **Full transformer:** Optional env `TRANSFORMER_BENCHMARK_FULL=1` and `TRANSFORMER_BENCHMARK_CUDA=1`.

Launch configuration is unchanged; Nsight **wraps** the run and replays kernels (ncu) or traces API calls (nsys).

---

## 4. Optimization metrics (what to look for)

| Metric (Nsight Compute) | Meaning | Optimization hint |
|--------------------------|--------|--------------------|
| **achieved_occupancy** | Threads per SM / max | Low → try fewer registers or less shared memory per block |
| **dram__throughput.avg.pct_of_peak_sustained_elapsed** | DRAM BW utilization | Low on memory-bound kernel → check coalescing, reuse |
| **l1tex__throughput**, **lts__throughput** | L1/L2 utilization | Complement to DRAM |
| **warp_execution_efficiency** | Active threads / 32 | Low → warp divergence; reduce branches or use warp primitives |
| **gpu__time_duration.sum** | Kernel time (ns) | For roofline: achieved GFLOPS = FLOPs / time |
| **sm__sass_throughput.avg.pct_of_peak_sustained_elapsed** | Compute utilization | High on compute-bound kernels |

---

## 5. Benchmark results (reports)

After running the profiling scripts:

| Artifact | Tool | Location |
|----------|------|----------|
| Timeline | Nsight Systems | `profiling/nsight_reports/<prefix>_timeline.nsys-rep` |
| Per-kernel metrics (binary) | Nsight Compute | `profiling/nsight_reports/<prefix>_kernels.ncu-rep` or `.nsight-cuprof` |
| Parsed metrics (JSON) | Python (from ncu) | `profiling/nsight_reports/<prefix>_kernels_metrics.json` |
| Roofline plot | plot_roofline.py | `profiling/nsight_reports/roofline_model.png` |

Open `.nsys-rep` in Nsight Systems GUI and `.ncu-rep` in Nsight Compute GUI for detailed inspection.

---

## 6. Diagrams / tables

### Profiling pipeline

| Step | Command / script | Output |
|------|-------------------|--------|
| 1 | `nsys profile -o ... python benchmarks/transformer_benchmark.py` | .nsys-rep |
| 2 | `ncu -o ... -c 10 --set basic python benchmarks/transformer_benchmark.py` | .ncu-rep |
| 3 | `ncu --import <rep> --page raw --csv` (or script) | stdout → parsed to JSON |
| 4 | `python profiling/roofline_analysis/plot_roofline.py` | roofline_model.png |

### Script usage (from repo root)

```bash
# Optional: skip nsys (timeline); only run ncu
set SKIP_NSYS=1

# Optional: quick ncu run (~1 min, one kernel)
set NCU_QUICK=1

# Run profiling (ncu and optionally nsys)
python profiling/run_nsight_profiling.py
```

---

## 7. Code snippets (from this repo)

### Finding nsys / ncu (Windows and PATH)

```python
# profiling/run_nsight_profiling.py
def find_tool(name: str) -> str | None:
    exe = name + (".exe" if sys.platform == "win32" else "")
    for path in os.environ.get("PATH", "").split(os.pathsep):
        p = Path(path) / exe
        if p.exists():
            return str(p)
    # CUDA_PATH, then Windows Program Files (Nsight Systems*, Nsight Compute*)
    ...
```

### Nsight Systems command

```python
cmd = [
    nsys, "profile",
    "-o", str(out_file),
    "--stats=true",
    "--trace=cuda",
    sys.executable, str(BENCHMARK_SCRIPT),
]
subprocess.run(cmd, cwd=REPO_ROOT, ...)
```

### Nsight Compute command (basic set, limit kernels)

```python
cmd = [
    ncu,
    "-o", str(out_rep.with_suffix("")),
    "-c", str(ncu_kernel_count),  # e.g. 10 or 1 for NCU_QUICK
    "--set", "basic",
    "--page", "raw",
    "--csv",
    "--force-overwrite",
    sys.executable, str(script),
]
```

### Export raw CSV from report (for parsing)

```python
cmd = [ncu, "--import", str(rep_file), "--page", "raw", "--csv"]
result = subprocess.run(cmd, capture_output=True, text=True)
# Parse result.stdout and save to *_kernels_metrics.json
```

---

## References

- [roofline_analysis.md](roofline_analysis.md) — using metrics for roofline.
- [optimization_guide.md](optimization_guide.md) — how to act on profiling results.
- [profiling/nsight_reports/README.md](../profiling/nsight_reports/README.md) — file layout and quick commands.
