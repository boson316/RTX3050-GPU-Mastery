# Reproducible Scripts

Run all benchmarks and generate charts from **repository root**. The repo is designed for **easy reproduction** on any CUDA-capable machine.

## One-command reproduce

**Option A — Full dashboard (recommended):** runs all four benchmark suites and writes `benchmarks/performance_report.md`:

```bash
python tools/performance_dashboard.py --skip-mnist
```

**Option B — Scripts (benchmarks + charts + roofline):**

### Windows

```bat
scripts\reproduce_all.bat
```

### Linux / macOS

```bash
chmod +x scripts/reproduce_all.sh
./scripts/reproduce_all.sh
```

## What runs (Option B)

1. **Matmul benchmark** — `python benchmarks/matmul_benchmark.py`
2. **Conv benchmark** — `python benchmarks/conv_benchmark.py`
3. **Transformer benchmark** — `python benchmarks/transformer_benchmark.py` (light mode)
4. **Chart generation** — `python benchmarks/generate_charts.py --skip-mnist` (omit `--skip-mnist` to include MNIST training)
5. **Roofline plot** — `python profiling/roofline_analysis/plot_roofline.py`

**Requirements:** Repo root, Python env with `torch`, `matplotlib`, and optionally `triton` installed. See [docs/getting_started.md](../docs/getting_started.md).
