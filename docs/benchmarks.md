# Benchmarks — RTX 3050 6GB Laptop GPU

All timings on **NVIDIA GeForce RTX 3050 6GB Laptop GPU** (Ampere sm_86). PyTorch CUDA 12.4, Windows.

---

## 1. Custom Conv2d (3×3, in=1, out=32, no padding)

### FP32 (interface float; kernel uses FP16)

| Batch | torch (ms) | Extension (ms) | Speedup | max_diff |
|-------|------------|----------------|---------|----------|
| 1024  | 2.25       | 1.63           | **1.37x** | ~0.005 (OK) |
| 512   | 1.13       | 0.82           | **1.37x** | ~0.005 (OK) |
| 128   | 0.30       | 0.22           | **1.36x** | ~0.004 (OK) |

### FP16 (pure half: torch vs extension vs Triton)

| Batch | torch (ms) | Extension (ms) | Triton (ms) | Ext speedup | Triton speedup |
|-------|------------|----------------|-------------|-------------|----------------|
| 1024  | 1.20       | 0.81           | 1.08        | **1.49x**   | 1.11x          |
| 512   | 0.63       | 0.41           | 0.52        | **1.52x**   | 1.22x          |
| 128   | 0.18       | 0.11           | 0.14        | **1.63x**   | 1.27x          |

- **Extension**: FP16 kernel, 16×16 tile, `tile_in[18][18]`, `tile_w[32][9]`, `__half2`; sm_86.
- **Triton**: JIT kernel, 1D grid, BLOCK_C=32; max_diff vs torch ~0.002.

---

## 2. Matrix Multiply (N=1024)

| Implementation | Time (ms) | Speedup vs CPU |
|----------------|-----------|-----------------|
| CPU (reference) | —        | 1x              |
| GPU naive      | —        | —               |
| GPU tiled (16×16 shared) | —  | **~521x**       |

---

## 3. MNIST (SmallCNN + AMP)

| Metric | Value |
|--------|--------|
| Test accuracy | **99%+** |
| Time per epoch | ~1–1.5 s |
| Batch size | 2048 |
| Device | RTX 3050 |

---

## 4. 圖檔

圖檔位於 [../benchmarks/](../benchmarks/)：`matrix_mul_speedup.png`、`mnist_acc_loss.png`、`conv_benchmark.png`。
