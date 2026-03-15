# Getting Started — RTX 3050 GPU Optimization Lab

This guide gets you from zero to running your first GPU benchmarks and charts on an **RTX 3050** (or any CUDA-capable GPU). No prior CUDA experience required. The repository is designed for **easy reproduction** (clone → install → run dashboard).

---

## Quick reproduce (3 steps)

```bash
git clone https://github.com/boson316/RTX3050-GPU-Mastery.git
cd RTX3050-GPU-Mastery
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install triton matplotlib ninja
python tools/performance_dashboard.py --skip-mnist
```

Report: `benchmarks/performance_report.md`. Charts: `benchmarks/plots/`, `benchmarks/*.png`.

---

## 1. Prerequisites

- **GPU:** NVIDIA GPU with CUDA support (e.g. RTX 3050 Laptop 6GB, sm_86).
- **OS:** Windows 10/11 or Linux.
- **Python:** 3.10+ (3.11 recommended).
- **CUDA Toolkit:** 12.x (optional for PyTorch; required for building CUDA extensions).

---

## 2. Environment setup

### Option A: Conda (recommended)

```bash
conda create -n gpu python=3.11
conda activate gpu
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install matplotlib triton ninja
```

**Windows:** For Triton use `pip install triton-windows` if `triton` is not available.

### Option B: venv + pip

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install matplotlib triton ninja
```

### Verify GPU

```python
import torch
print(torch.cuda.is_available())   # True
print(torch.cuda.get_device_name(0))  # e.g. "NVIDIA GeForce RTX 3050 Laptop GPU"
```

---

## 3. Clone and install (optional)

If you use the transformer CUDA extension or custom conv:

```bash
git clone https://github.com/boson316/RTX3050-GPU-Mastery.git
cd RTX3050-GPU-Mastery
pip install -e .   # optional: project root
```

**Custom conv (PyTorch extension):**

```bash
cd extension
pip install --no-build-isolation .
```

Requires **Visual Studio** (Windows) with "Desktop development with C++" and **CUDA Toolkit** so that `cl.exe` and `nvcc` are available.

---

## 4. First runs (RTX 3050 examples)

All commands from **repository root**.

### Matrix multiply (PyTorch vs Triton)

```bash
python benchmarks/matmul_benchmark.py
```

Example output (RTX 3050): torch and Triton latency (ms) and GFLOPS for N=512, 1024, 2048.

### Conv2d (torch vs extension vs Triton)

```bash
python benchmarks/conv_benchmark.py
```

Example: torch.nn.Conv2d vs custom_conv2d vs Triton 3×3 conv (B=128, 1→32 channels).

### Transformer kernels (QKV, Softmax, LayerNorm, GELU, MLP)

```bash
python benchmarks/transformer_benchmark.py
```

Reports latency (ms), memory bandwidth (GB/s), and GFLOPS per kernel. Use `TRANSFORMER_BENCHMARK_CUDA=1` to include custom CUDA kernels (requires built extension).

### Roofline plot (no Nsight required)

```bash
pip install matplotlib
python profiling/roofline_analysis/plot_roofline.py
```

Output: `profiling/nsight_reports/roofline_model.png` — memory-bound vs compute-bound kernels.

---

## 5. Generate benchmark charts

From repo root:

```bash
python benchmarks/generate_charts.py
```

This will:

- Run matmul and conv benchmarks and save **matrix_mul_speedup.png** and **conv_benchmark.png** under `benchmarks/` (or copy from existing plot scripts if available).
- Optionally run MNIST training and save **mnist_acc_loss.png** (use `--skip-mnist` to skip, as training takes a few minutes).

Charts are written to `benchmarks/` for use in docs and README.

---

## 6. Full reproducibility script

**Windows:**

```bat
scripts\reproduce_all.bat
```

**Linux / macOS:**

```bash
chmod +x scripts/reproduce_all.sh
./scripts/reproduce_all.sh
```

This runs benchmarks and chart generation; see `scripts/README.md` for details.

---

## 7. Tutorial notebooks

Jupyter notebooks with runnable examples (RTX 3050–friendly):

| Notebook | Content |
|----------|---------|
| `notebooks/01_getting_started_rtx3050.ipynb` | Check GPU, first PyTorch CUDA ops |
| `notebooks/02_gpu_memory_and_roofline.ipynb` | Memory hierarchy, roofline concept |
| `notebooks/03_benchmarks_matmul_conv.ipynb` | Matmul & conv benchmarks, interpret results |
| `notebooks/04_transformer_and_profiling.ipynb` | Transformer benchmark, profiling intro |

Run Jupyter from repo root:

```bash
pip install jupyter
jupyter notebook notebooks/
```

---

## 8. Next steps

- **Deeper theory:** [GPU Memory Hierarchy](gpu_memory_hierarchy.md), [Memory Hierarchy Diagrams](memory_hierarchy_diagrams.md), [Optimization Guide](optimization_guide.md).
- **CUDA from scratch:** [CUDA Roadmap](cuda_roadmap.md) and Level 1–4 docs.
- **Profiling:** [profiling/nsight_reports/README.md](../profiling/nsight_reports/README.md) for Nsight Systems/Compute and roofline.

All examples in this repo are designed to run on an **RTX 3050 Laptop 6GB**; larger GPUs will show different absolute numbers but the same relative patterns.
