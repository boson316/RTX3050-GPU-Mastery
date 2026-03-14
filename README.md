# RTX 3050 Laptop GPU Programming Benchmarks

大二資工生使用 **RTX 3050 6GB Laptop GPU (sm_86)** 實作 GPU programming benchmarks。

[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%20%7C%20cuDNN-red?logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📑 Table of Contents

- [技術棧](#-技術棧)
- [主要成果](#-主要成果)
- [Benchmark 圖表](#-benchmark-圖表)
- [資料夾結構](#-資料夾結構)
- [安裝與重現](#-安裝與重現)
- [Citation & Star](#-citation--star)

---

## 🛠️ 技術棧

- **CUDA 12.4**
- **PyTorch 2.4 + cuDNN**
- **C++ / CUDA Extension**（PyTorch custom op）
- **Triton Kernel Language**（Python JIT kernel）

---

## 📊 主要成果

| Task | Implementation | Performance |
|------|----------------|--------------|
| Matrix Multiplication | Pure CUDA (shared memory tiled) | **521x** CPU speedup (N=1024, 5.3ms) |
| Reduction | Pure CUDA shared memory | **0.763ms** (1M elements) |
| MNIST CNN | PyTorch GPU (SmallCNN + AMP) | **99%** test accuracy |
| 3×3 Conv (1→32) FP16 | CUDA Extension | **1.50x** PyTorch (B=1024, 0.81ms) |
| 3×3 Conv FP16 | Triton Python kernel | **1.27x** PyTorch (B=128, 0.14ms) |

*Device: NVIDIA GeForce RTX 3050 6GB Laptop GPU (Ampere sm_86)*

---

## 📈 Benchmark 圖表

| 圖表 | 說明 |
|------|------|
| Matrix Mul 521x | CPU vs GPU 時間與 speedup (N=1024) |
| MNIST 99% | Train/test loss 與 accuracy 曲線 |
| Conv 1.5x | torch vs Extension vs Triton 耗時比較 |

![Matrix Mul 521x](benchmarks/matrix_mul_speedup.png)
*Matrix multiplication: 521x speedup vs CPU (N=1024)*

![MNIST 99%](benchmarks/mnist_acc_loss.png)
*MNIST SmallCNN: 99% test accuracy*

![Conv 1.5x](benchmarks/conv_benchmark.png)
*3×3 Conv FP16: Extension 1.5x、Triton 1.27x vs PyTorch*

---

## 📁 資料夾結構

```
├── cuda/              # Pure CUDA kernels (vector add, matrix mul, reduction)
├── pytorch/           # MNIST 99% CNN (SmallCNN, AMP)
├── extension/        # 1.5x Conv — PyTorch CUDA Extension (FP16, 16×16 tile)
├── triton/            # Triton Python kernels (3×3 conv FP16)
├── benchmarks/        # 詳細數據 + 圖表
├── docs/              # benchmarks.md、說明文件
├── .github/workflows/ # CI (lint / test)
├── README.md
└── LICENSE
```

---

## 🔧 安裝與重現

```bash
# 環境
conda create -n gpu python=3.11
conda activate gpu
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install triton matplotlib

# Pure CUDA（需 CUDA Toolkit + VS x64 Native Tools）
cd cuda && build.bat

# PyTorch CUDA Extension（需 CUDA 12.4 + MSVC）
cd extension && pip install --no-build-isolation .

# Benchmark（從 repo 根目錄執行）
python extension/mnist_custom_conv.py
```

**Windows 注意**：Extension 編譯若路徑含中文，請使用 `build_in_english_path.bat`（在 x64 Native Tools 下執行）。Triton 請用 `pip install triton-windows`。

---

## ⭐ Citation & Star

若此 repo 對你有幫助，歡迎 **Star** ⭐。

```text
RTX 3050 Laptop GPU Programming Benchmarks — 521x matrix, 99% MNIST, 1.5x conv.
https://github.com/boson316/RTX3050-GPU-Mastery
```

## License

[MIT](LICENSE)
