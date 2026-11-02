<!-- device -->
| Metric | Value |
|---|---|
| Peak at NVML max clock | 21.5 TFLOPS @ 2100 MHz |
| Median SM clock under load | 1747 MHz |
| Peak at the sustained clock | 17.9 TFLOPS |
| Peak DRAM bandwidth | 132 GB/s |

<!-- bench -->
| Kernel | Shape | PyTorch | Triton | CUDA |
|---|---|---|---|---|
| GEMM | 512×512×512 | 0.053 (0.053–0.066) · 5.03 TFLOPS | 0.069 (0.060–0.069) · 3.92 TFLOPS | — |
| GEMM | 1024×1024×1024 | 0.168 (0.168–0.184) · 12.80 TFLOPS | 0.141 (0.138–0.155) · 15.23 TFLOPS | 0.202 (0.201–0.222) · 10.65 TFLOPS |
| GEMM | 2048×2048×2048 | 1.197 (1.178–1.205) · 14.36 TFLOPS | 1.226 (1.210–1.233) · 14.01 TFLOPS | — |
| QKV | 2048×768→2304 | 0.545 (0.537–0.609) · 13.30 TFLOPS | 0.587 (0.546–0.674) · 12.36 TFLOPS | 0.675 (0.674–0.775) · 10.74 TFLOPS |
| MLP | 2048×768→3072→768 | 1.612 (1.584–1.708) · 11.99 TFLOPS | 1.634 (1.630–1.698) · 11.83 TFLOPS | 1.681 (1.679–1.726) · 11.50 TFLOPS |
| Softmax | 2048×768 | 0.054 (0.054–0.055) · 116 GB/s | 0.057 (0.057–0.060) · 110 GB/s | 0.094 (0.093–0.099) · 67 GB/s |
| LayerNorm | 2048×768 | 0.056 (0.056–0.077) · 112 GB/s | 0.067 (0.065–0.073) · 94 GB/s | 0.111 (0.110–0.118) · 57 GB/s |
| GELU | 1572864 | 0.056 (0.055–0.056) · 113 GB/s | 0.054 (0.054–0.060) · 116 GB/s | 0.056 (0.056–0.057) · 112 GB/s |
| Conv 3×3 | B32_C1x32_28x28 | 0.084 (0.041–0.086) · 0.15 TFLOPS | 0.071 (0.067–0.103) · 0.17 TFLOPS | 0.036 (0.035–0.036) · 0.35 TFLOPS |
| Conv 3×3 | B128_C1x32_28x28 | 0.168 (0.168–0.171) · 0.30 TFLOPS | 0.083 (0.076–0.102) · 0.60 TFLOPS | 0.124 (0.124–0.125) · 0.40 TFLOPS |
| Conv 3×3 | B128_C1x64_64x64 | 1.680 (1.669–1.710) · 0.34 TFLOPS | 1.044 (1.019–1.049) · 0.54 TFLOPS | — |
| Attention | B2_H8_S128_D64 | 0.037 (0.016–0.049) · 1.81 TFLOPS | 0.093 (0.071–0.128) · 0.72 TFLOPS | 0.097 (0.097–0.107) · 0.69 TFLOPS |
| Attention | B4_H8_S256_D64 | 0.059 (0.058–0.060) · 9.02 TFLOPS | 0.064 (0.060–0.141) · 8.42 TFLOPS | 0.386 (0.385–0.405) · 1.39 TFLOPS |
| Attention | B2_H8_S512_D64 | 0.106 (0.105–0.107) · 10.13 TFLOPS | 0.088 (0.087–0.097) · 12.19 TFLOPS | 0.688 (0.678–0.691) · 1.56 TFLOPS |

<!-- roof -->
| Kernel | Bound | % of NVML-max roof (PyTorch / Triton / CUDA) | % of sustained roof (PyTorch / Triton / CUDA) |
|---|---|---|---|
| GEMM 2048³ | compute | 67 / 65 / — | 81 / 78 / — |
| QKV | compute | 62 / 59 / 48 | 74 / 70 / 58 |
| MLP | compute | 56 / 57 / 53 | 68 / 68 / 64 |
| Attention S512 | compute | 48 / 57 / 9 | 57 / 68 / 9 |
| Conv B128 64×64 | memory | 29 / 47 / — | 29 / 47 / — |
| Conv B128 28×28 | memory | 26 / 44 / 70 | 26 / 44 / 70 |
| Softmax | memory | 88 / 78 / 50 | 88 / 78 / 50 |
| LayerNorm | memory | 84 / 65 / 43 | 84 / 65 / 43 |
| GELU | memory | 86 / 78 / 85 | 86 / 78 / 85 |

<!-- longseq -->
| S | Implementation | Latency (ms) | Extra VRAM (MiB) | Max error vs FP64 |
|---|---|---|---|---|
| 1024 | SDPA | 1.257 (1.254–1.271) | 2.0 | 1.104e-06 |
| 1024 | Triton | 0.357 (0.356–0.357) | 2.0 | 1.364e-03 |
| 1024 | CUDA | 1.238 (1.233–1.241) | 2.0 | 8.538e-07 |
| 1024 | Matrix-materializing | — | 72.1 | — |
| 2048 | SDPA | 4.752 (4.750–4.775) | 4.0 | 7.044e-07 |
| 2048 | Triton | 1.291 (1.289–1.301) | 4.0 | 8.341e-04 |
| 2048 | CUDA | 4.834 (4.817–4.836) | 4.0 | 6.183e-07 |
| 2048 | Matrix-materializing | — | 256.0 | — |
| 4096 | SDPA | 20.181 (20.171–20.210) | 8.0 | 3.338e-07 |
| 4096 | Triton | 5.115 (5.098–5.119) | 8.0 | 5.551e-04 |
| 4096 | CUDA | 17.303 (17.283–17.314) | 8.0 | 6.030e-07 |
| 4096 | Matrix-materializing | — | 1024.0 | — |

<!-- wmma -->
| Metric | Median | Min | Max | Runs |
|---|---|---|---|---|
| 1024³ WMMA tiled (GFLOPS) | 7530.6 | 7489.8 | 10979.9 | 12 |
| 1024³ cuBLAS (GFLOPS) | 11366.8 | 6944.2 | 11459.8 | 12 |
| 1024³ WMMA % of cuBLAS (%) | 81.4 | 65.6 | 109.6 | 12 |
| 2048³ WMMA tiled (GFLOPS) | 11648.3 | 11358.3 | 12122.3 | 6 |
| 2048³ cuBLAS (GFLOPS) | 14123.6 | 13969.4 | 14364.1 | 6 |
| 2048³ WMMA % of cuBLAS (%) | 81.9 | 79.1 | 86.5 | 6 |

<!-- floor -->
| Implementation | Launch floor (ms) |
|---|---|
| PyTorch | 0.0112 |
| Triton | 0.0553 |
| CUDA | 0.0092 |
