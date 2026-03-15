# Nsight Reports

NVIDIA Nsight Systems and Nsight Compute reports are stored here for reproducibility and roofline analysis.

## Generated files

| File | Tool | Description |
|------|------|-------------|
| `<prefix>_timeline.nsys-rep` | Nsight Systems | Kernel execution **timeline** (CUDA/NVTX) |
| `<prefix>_kernels.ncu-rep` | Nsight Compute | Per-kernel **occupancy**, **memory bandwidth**, **warp divergence** |
| `<prefix>_kernels_metrics.json` | Parsed from ncu | Metrics exported for roofline script |
| `roofline_model.png` | `plot_roofline.py` | **Roofline plot**: memory-bound vs compute-bound kernels |

Default prefix: `transformer`.

## Generating reports

From repo root, run only the commands below (on Windows CMD, do **not** type lines starting with `#`—those are comments and CMD will error):

```bash
python profiling/run_nsight_profiling.py
```

This will:

1. **Nsight Systems** – Capture kernel timeline; output `profiling/nsight_reports/transformer_timeline.nsys-rep`.
2. **Nsight Compute** – For each profiled kernel: measure **GPU occupancy**, **memory bandwidth** (DRAM/L2), **warp execution efficiency** (warp divergence), and kernel duration; output `transformer_kernels.ncu-rep` and `transformer_kernels_metrics.json`.

Optional: set `NSIGHT_REPORT_PREFIX` to use a different prefix.

```bash
python profiling/roofline_analysis/plot_roofline.py
```

**If nsys/ncu not found:** install [Nsight Systems](https://developer.nvidia.com/nsight-systems) and [Nsight Compute](https://developer.nvidia.com/nsight-compute), or add their install folders to PATH. The script also looks under `C:\Program Files\NVIDIA Corporation\Nsight Systems*` and `Nsight Compute*` on Windows.

Output: `profiling/nsight_reports/roofline_model.png` showing:

- **Memory-bound kernels** (below ridge point)
- **Compute-bound kernels** (above ridge point)
- Roofline curve and ridge point (peak FLOPS / peak bandwidth)

Plot options: `--prefix`, `--B`, `--S`, `--H`, `-o <path>`. GPU peak can be set via `ROOFLINE_PEAK_GFLOPS` and `ROOFLINE_PEAK_GBPS`.

## Manual one-off profiling

```bash
# Nsight Systems (timeline only)
nsys profile -o nsight_reports/my_timeline --stats=true python benchmarks/transformer_benchmark.py

# Nsight Compute (first 5 kernels, basic set)
ncu -o nsight_reports/my_kernels -c 5 --set basic python benchmarks/transformer_benchmark.py
```

Open `.nsys-rep` in Nsight Systems GUI and `.ncu-rep` in Nsight Compute GUI for detailed inspection.

## Metrics to check

- **Occupancy**: threads per SM vs theoretical max (Nsight Compute → achieved_occupancy).
- **Memory throughput**: L2/DRAM bandwidth utilization (dram__throughput, l1tex__throughput).
- **Warp divergence**: low warp execution efficiency → more divergence (warp_execution_efficiency).

See `docs/optimization_guide.md` for interpretation.
