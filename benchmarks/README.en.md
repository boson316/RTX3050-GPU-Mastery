# Benchmarks — Reproducible performance tests & charts

**Languages:** [中文](README.md) · [English](README.en.md) — repo root: [English](../README.md) · [中文](../README.zh-TW.md)

## Benchmark suites (recommended)

Run from **repository root**. Each script compares **PyTorch / CUDA kernel / Triton kernel** and outputs:

- **Latency (ms)**, **throughput (GFLOPS)**, **memory bandwidth (GB/s)**, **GPU utilization (%)**
- **CSV:** `benchmarks/benchmark_<name>_results.csv`
- **Terminal tables**
- **Matplotlib charts** under `benchmarks/plots/`

| Script | Description |
|--------|-------------|
| `benchmark_matmul.py` | Matmul: PyTorch (cuBLAS) vs Triton vs CUDA (standalone 1024³) |
| `benchmark_conv.py` | 3×3 Conv2D: PyTorch vs Triton vs CUDA (extension) |
| `benchmark_attention.py` | Attention: PyTorch SDPA vs Triton Flash vs CUDA (`flash_attention_simple`) |
| `benchmark_transformer.py` | Transformer sub-kernels: QKV / Softmax / LayerNorm / GELU / MLP |

```bash
python benchmarks/benchmark_matmul.py
python benchmarks/benchmark_conv.py
python benchmarks/benchmark_attention.py
python benchmarks/benchmark_transformer.py
```

**Transformer CUDA:** Off by default to reduce load; set `TRANSFORMER_BENCHMARK_CUDA=1` to enable.

## Plot output

| Directory | Description |
|-----------|-------------|
| `benchmarks/plots/` | PNG charts from `benchmark_*.py` (latency, GFLOPS, by config) |

## Legacy scripts (still supported)

| Script | Description |
|--------|-------------|
| `benchmark_matmul.py` | PyTorch vs Triton matmul, multiple N and dtypes |
| `benchmark_conv.py` | torch.nn.Conv2d vs custom_conv vs Triton |
| `benchmark_attention.py` | PyTorch SDPA vs Triton Flash Attention |
| `benchmark_transformer.py` | Per-kernel latency & bandwidth |
| `generate_charts.py` | Aggregate charts (`--skip-mnist` to skip MNIST training) |

## One-command dashboard

```bash
python tools/performance_dashboard.py --skip-mnist
```

Report: `benchmarks/performance_report.md`.

See [../docs/getting_started.md](../docs/getting_started.md) and [../README.md](../README.md).
