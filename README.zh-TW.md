<p align="center">
  <a href="https://github.com/boson316/RTX3050-GPU-Mastery/actions/workflows/ci.yml"><img src="https://github.com/boson316/RTX3050-GPU-Mastery/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <img src="https://img.shields.io/badge/CUDA-12.x-76b900?style=flat-square&logo=nvidia" alt="CUDA"/>
  <img src="https://img.shields.io/badge/Triton-3.x-ff6b35?style=flat-square" alt="Triton"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=flat-square&logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/GPU-RTX%203050%20Laptop%20(sm__86)-76b900?style=flat-square" alt="GPU"/>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License"/></a>
</p>

# RTX 3050 GPU Kernel Lab

**在 6GB 筆電 GPU 上手寫 CUDA 與 Triton kernel（GEMM、FlashAttention、Transformer 區塊、卷積），並附上一套不需要 GPU 也能執行的驗證流程。**

**Languages:** [English](README.md) · 中文

大多數 GPU kernel 作品集只放加速比的圖。這個專案也呈現這些數字是怎麼被檢驗的：每個 kernel 計時前都先跟參考答案比對，量到的吞吐量會跟硬體理論峰值做合理性檢查，kernel 的邏輯則在沒有 GPU 的 CI 機器上測試。建立這套流程的過程中，找出了第一版裡十幾個真正的 bug 和方法論問題，其中包含兩個「比 PyTorch 快」的結果，事後證明都是因為 kernel 少算了一部分。完整紀錄在 **[docs/ENGINEERING_LOG.md](docs/ENGINEERING_LOG.md)**。

---

## 亮點

| | 內容 | 位置 |
|---|---|---|
| ⚡ | **FlashAttention forward**：CUDA 版（每個 thread 負責一個 query row、K/V tile 放在 shared memory、用 `exp2` 做 online softmax、causal 時跳過整塊 tile）和 Triton 版（autotune、`tl.dot` 走 Tensor Core） | [`flash_attention_simple/`](flash_attention_simple)、[`triton_kernels/flash_attention/`](triton_kernels/flash_attention) |
| 🧮 | **WMMA Tensor Core GEMM**：64×64×32 shared-memory block tile、4 個 warp 各算 2×2 fragment、128-bit 搬移、shared memory padding。實測：**2048³ 為 cuBLAS 的 79～87%**；1024³ 中位數 81%，但範圍很大（65.6～109.6%） | [`cuda_roadmap/level4_tensor_core/`](cuda_roadmap/level4_tensor_core) |
| 🔗 | **Kernel fusion**：Triton linear 把 bias + GELU 融合進 epilogue（MLP 只需 2 次 launch）、online softmax（讀 2 次而不是 3 次）、Welford LayerNorm、把卷積寫成 9 次向量化 outer product | [`triton_kernels/`](triton_kernels)、[`gpu_kernels/transformer/`](gpu_kernels/transformer) |
| 🧪 | **CI 上不需要 GPU 的驗證**：Triton interpreter 在 CPU 上執行真正的 kernel；自製的 **CUDA 執行模型模擬器**用 `std::thread` 跑每個 thread、用 `std::barrier` 模擬 `__syncthreads`、用 per-warp 交換區模擬 `__shfl_xor_sync`，直接執行真正的 `__global__` 程式碼。如果有 thread 沒走到 barrier，模擬器會直接死鎖，bug 藏不住 | [`tests/`](tests) |
| 📈 | **由裝置推導峰值的 Roofline**：SM 數 × 每 clock FLOPs × NVML 查到的時脈。cuBLAS 在 2048³ 達到 NVML 最大時脈 roof 的 67%，以實際維持的時脈計算則是 81% | [`profiling/`](profiling)、[`benchmarks/gpu_specs.py`](benchmarks/gpu_specs.py) |
| 🐛 | **工程日誌**：data race、兩個死鎖、cuBLAS 越界讀取、WMMA 運算元被轉置、4 個 mask bug、不公平的 baseline，以及在真實 GPU 上新發現的 5 個 bug。每個都附根本原因、修法和回歸測試 | [`docs/ENGINEERING_LOG.md`](docs/ENGINEERING_LOG.md) |

---

## 正確性優先

```mermaid
flowchart LR
    K[kernel] --> V{輸出 == 參考答案?}
    V -- 否 --> X["[skip] WRONG RESULT，不計時"]
    V -- 是 --> T["計時：CUDA events、<br/>預熱 1 秒、warm-up、取中位數"]
    T --> P{"≤ 裝置峰值？<br/>(SM × FLOP/clk × 時脈)"}
    P -- 否 --> Y["BenchmarkError：<br/>物理上不可能"]
    P -- 是 --> R[CSV / 報告 / roofline]
```

| 檢查 | 執行環境 | 涵蓋範圍 |
|---|---|---|
| `pytest` 搭配 `TRITON_INTERPRET=1` | CI，CPU | 所有 Triton kernel 對照 PyTorch：奇數形狀、非 2 次方的 `head_dim`、causal（46 個案例） |
| `tests/cuda_emulation/run_all.sh` | CI，CPU | CUDA FlashAttention、softmax、LayerNorm、bias+GELU、conv3x3 對照 naive 參考實作（27 個案例） |
| `nvcc -arch=sm_86 --Werror all-warnings` | CI | 每一個 `.cu` 檔 |
| `pytest`（GPU） | RTX 3050 | 編譯後的 Triton kernel（FP32 與 FP16）、autotune 版本、三個 PyTorch CUDA extension、計時工具（116 個案例） |
| 自我驗證的 CUDA 執行檔（`VERIFY_*=PASS`） | GPU | 所有 roadmap kernel（含 WMMA）對照 CPU 參考答案 |
| `bench_utils.measure()` | GPU | 每個 benchmark：驗證 → 計時 → 合理性檢查 |

---

## 內容

| 模組 | 內容 | 關鍵技術 |
|---|---|---|
| [`cuda_roadmap/`](cuda_roadmap) | 11 個會自我驗證的 CUDA benchmark，Level 1–4 | coalescing、bank conflict（`[32][33]` 轉置）、warp shuffle reduction、fusion、grid-stride + occupancy API、WMMA |
| [`triton_kernels/`](triton_kernels) | matmul、linear / QKV / MLP、softmax、LayerNorm、GELU、conv3x3、FlashAttention | 為 L2 重用設計的 grouped program order、`num_stages` pipelining、fused epilogue、autotune |
| [`flash_attention_simple/`](flash_attention_simple) | CUDA FlashAttention（PyTorch extension + standalone） | online softmax、累加器常駐 register、有記錄的 register 預算（215 regs、0 spill） |
| [`gpu_kernels/transformer/`](gpu_kernels/transformer) | QKV、softmax、LayerNorm、GELU、MLP 的 PyTorch CUDA extension | 用 `Yᵀ = WᵀXᵀ` 呼叫 row-major cuBLAS、fused bias+GELU epilogue、Welford 統計 |
| [`extension/`](extension) | 3×3 卷積（C_in = 1、C_out = 32）PyTorch extension | 18×18 halo tile 給 32 個 channel × 9 個 tap 重用 |
| [`benchmarks/`](benchmarks) | 有驗證的 benchmark 框架、裝置峰值查詢 | CUDA events、正確性閘門、合理性上限 |
| [`profiling/`](profiling) | Roofline 分析、Nsight Compute / Systems 腳本 | 算術強度 vs 實測吞吐量 |
| [`tests/`](tests) | Triton interpreter 測試、CUDA CPU 模擬器 | 見上方 |

---

## 結果

**GPU：** NVIDIA GeForce RTX 3050 6GB Laptop（Ampere、sm_86），20 個 SM、96-bit 匯流排、43W，插電並使用最佳效能模式。
**軟體：** 驅動程式 610.62、CUDA 12.4、PyTorch 2.6.0、Triton 3.2.0、MSVC 14.50。完整環境見 [`results/v2/environment.md`](results/v2/environment.md)。

表格每一格是**延遲中位數，單位 ms（3 輪的最小值～最大值）**。每個 kernel 都先通過正確性檢查才計時。原始 log 和 CSV 在 [`results/v2/`](results/v2)，摘要見 [`results/v2/SUMMARY.md`](results/v2/SUMMARY.md)。

| 指標 | 數值 |
|---|---|
| NVML 最大時脈的峰值 | 21.5 TFLOPS @ 2100 MHz |
| 負載下 SM 時脈中位數 | 1747 MHz |
| 持續時脈的峰值 | 17.9 TFLOPS |
| DRAM 峰值頻寬 | 132 GB/s |

Roof 用兩種時脈呈現，因為 2100 MHz 是 NVML 回報的最大值，而這台 43W 筆電在這些負載下實際約為 1747 MHz。

### Roofline

![Roofline](profiling/roofline/roofline_plot.png)

圖上使用 NVML 最大時脈的 roof，每類 kernel 只標一個點。下表另外列出以持續時脈計算的 roof。

| Kernel | 受限於 | 佔 NVML 最大時脈屋頂 % (PyTorch / Triton / CUDA) | 佔持續時脈屋頂 % (PyTorch / Triton / CUDA) |
|---|---|---|---|
| GEMM 2048³ | 運算 | 67 / 65 / — | 81 / 78 / — |
| QKV | 運算 | 62 / 59 / 48 | 74 / 70 / 58 |
| MLP | 運算 | 56 / 57 / 53 | 68 / 68 / 64 |
| Attention S512 | 運算 | 48 / 57 / 9 | 57 / 68 / 9 |
| Conv B128 64×64 | 記憶體 | 29 / 47 / — | 29 / 47 / — |
| Conv B128 28×28 | 記憶體 | 26 / 44 / 70 | 26 / 44 / 70 |
| Softmax | 記憶體 | 88 / 78 / 50 | 88 / 78 / 50 |
| LayerNorm | 記憶體 | 84 / 65 / 43 | 84 / 65 / 43 |
| GELU | 記憶體 | 86 / 78 / 85 | 86 / 78 / 85 |

- **GEMM 受運算能力限制，cuBLAS 已接近這張卡能維持的上限。** 自訂 kernel 的空間很小：Triton 在 2048³ 與它打平，CUDA 在 1024³ 也落在相近範圍。
- **Softmax、LayerNorm、GELU 受頻寬限制。** PyTorch 已達 DRAM roof 的 84～88%，自訂 kernel 很難贏太多。CUDA 的 softmax 和 LayerNorm 較慢，分別只有 50% 和 43%。
- **cuDNN 處理這些 C_in = 1 的卷積時只用到記憶體 roof 的 26～29%**，所以專用 kernel 在這裡才有機會勝出。

### 量測結果

精度說明：GEMM 和 Transformer 各列都是 FP16，PyTorch 和 Triton 的卷積與 attention 也是 FP16。**CUDA 的卷積和 attention kernel 使用 FP32 儲存**，資料量是兩倍，attention kernel 也是跑在 CUDA core 而不是 Tensor Core。CUDA 的 GELU 實作 tanh 形式，另外兩者實作精確形式，各自對照自己的定義驗證。

| Kernel | 形狀 | PyTorch | Triton | CUDA |
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

![GEMM latency](benchmarks/matrix_mul_speedup.png)
![Conv latency](benchmarks/conv_benchmark.png)
![Attention latency](benchmarks/attention_benchmark.png)
![Transformer kernel latency](benchmarks/transformer_benchmark.png)

**自訂 kernel 較慢的地方，以及原因：**
- **CUDA 的 QKV 和 MLP 比 PyTorch 慢：** QKV 的差距（0.13 ms）全部來自獨立的 bias kernel，它要把 2048×2304 的輸出再讀寫一次（18.9 MB，在 132 GB/s 下約 0.14 ms）；MLP 多出的 bias pass（6 MB，約 0.05 ms）也剛好對得上差距。PyTorch 把 bias 融進 GEMM 的 epilogue。*v2.1 改用 cublasLt 做同樣的事，待重新量測。*
- **CUDA 的 softmax 和 LayerNorm 比 PyTorch 慢約 2 倍：** 每個 block 處理一列、每個 thread 只讀 2 bytes，同時在途的記憶體請求太少，而且每列要讀兩次。*v2.1 新增 warp-per-row 路徑：128-bit 讀取，整列留在 register 裡，待重新量測。*
- S = 512 時 CUDA attention 比 Triton 慢約 8 倍，比較的是 CUDA core 上的 FP32 和 Tensor Core 上的 FP16。
- PyTorch 在 512³ 的 GEMM 較快，SDPA 在 S = 128、256 的 attention 較快，但這兩個 S 的結果已接近下面所說的計時極限。

**自訂 kernel 較快的地方：** B = 32 的 CUDA 卷積、B = 128 的 Triton 卷積、1024³ 的 Triton GEMM，以及 S = 512 的 Triton attention。

**2048×768 的逐列 kernel 落在 Triton 的 launch 下限上**（約 0.055 ms，見下方「限制」），所以那幾格 Triton 的數字主要反映 Python 的 launch 開銷。改用 8192×768 才量得到 kernel 本身：softmax 的 PyTorch 0.207 ms（122 GB/s，DRAM 峰值的 92%），Triton 0.210 ms（120 GB/s），各跑兩次（[`phase4_softmax_run1.log`](results/v2/logs/phase4_softmax_run1.log)、[`run2`](results/v2/logs/phase4_softmax_run2.log)）。兩者都已經頂到頻寬牆。`benchmark_transformer.py` 現在會在兩種大小下量所有逐列 kernel。

### 長序列 FlashAttention（B=1、H=8、D=64、FP32）

| S | 實作 | 延遲（ms） | 額外顯存（MiB） | 相對 FP64 最大誤差 |
|---|---|---|---|---|
| 1024 | SDPA | 1.257 (1.254–1.271) | 2.0 | 1.104e-06 |
| 1024 | Triton | 0.357 (0.356–0.357) | 2.0 | 1.364e-03 |
| 1024 | CUDA | 1.238 (1.233–1.241) | 2.0 | 8.538e-07 |
| 1024 | 傳統（產生完整矩陣） | — | 72.1 | — |
| 2048 | SDPA | 4.752 (4.750–4.775) | 4.0 | 7.044e-07 |
| 2048 | Triton | 1.291 (1.289–1.301) | 4.0 | 8.341e-04 |
| 2048 | CUDA | 4.834 (4.817–4.836) | 4.0 | 6.183e-07 |
| 2048 | 傳統（產生完整矩陣） | — | 256.0 | — |
| 4096 | SDPA | 20.181 (20.171–20.210) | 8.0 | 3.338e-07 |
| 4096 | Triton | 5.115 (5.098–5.119) | 8.0 | 5.551e-04 |
| 4096 | CUDA | 17.303 (17.283–17.314) | 8.0 | 6.030e-07 |
| 4096 | 傳統（產生完整矩陣） | — | 1024.0 | — |

- **顯存：** 三種實作都只需要輸出緩衝區，傳統寫法在 S = 4096 時要 1024 MiB。但這裡 PyTorch 的 SDPA 本身就避開了完整矩陣（後端選擇是 `EFFICIENT_ATTENTION`），所以省下的是相對傳統寫法，不是相對 SDPA。
- **Triton 這一欄是 TF32，不是 FP32。** `tl.dot` 處理 FP32 輸入時預設走 TF32 Tensor Core（10-bit 尾數，相對精度約 5e-4），正好就是量到的 5.6e-4～1.4e-3 誤差。SDPA（memory-efficient 後端）和 CUDA kernel 則是在 CUDA core 上跑 IEEE FP32。所以 Triton 快 4 倍是「Tensor Core 對 CUDA core」的結果，不代表 kernel 比較好，也不能和另外兩欄直接比較。
- **公平的 IEEE FP32 比較是 CUDA 對 SDPA：** 到 S = 2048 都相當，S = 4096 時 CUDA kernel **快 14%**（17.3 對 20.2 ms），精度相同。
- *v2.1：* `flash_attention_triton(..., precision="ieee" | "tf32")` 和 [`benchmarks/benchmark_attention_longseq.py`](benchmarks/benchmark_attention_longseq.py) 會分開報告 FP16（實際模型用的：SDPA FlashAttention-2 對 Triton）、IEEE FP32（SDPA、Triton、CUDA）和 TF32，並各自附上相對 FP64 的誤差。待重新量測。

### WMMA tiled GEMM 與 cuBLAS（FP16）

| 指標 | 中位數 | 最小 | 最大 | 次數 |
|---|---|---|---|---|
| 1024³ WMMA tiled (GFLOPS) | 7530.6 | 7489.8 | 10979.9 | 12 |
| 1024³ cuBLAS (GFLOPS) | 11366.8 | 6944.2 | 11459.8 | 12 |
| 1024³ WMMA % of cuBLAS (%) | 81.4 | 65.6 | 109.6 | 12 |
| 2048³ WMMA tiled (GFLOPS) | 11648.3 | 11358.3 | 12122.3 | 6 |
| 2048³ cuBLAS (GFLOPS) | 14123.6 | 13969.4 | 14364.1 | 6 |
| 2048³ WMMA % of cuBLAS (%) | 81.9 | 79.1 | 86.5 | 6 |

這些是彼此獨立的行程執行，每次都通過驗證。2048³ 各次相當一致。1024³ 則分成兩群，而且是整次執行落在較慢的那一群。**目前還不知道原因**：1 秒預熱並沒有消除它（[`results/v2/logs/wmma_old_vs_new.log`](results/v2/logs/wmma_old_vs_new.log)）。整次執行一起變慢、而且 cuBLAS 也會受影響，代表原因比較可能是 GPU 的狀態（P-state、功耗上限），而不是 kernel 本身；[`tools/gpu_state_probe.py`](tools/gpu_state_probe.py) 會在每次執行時記錄 P-state、時脈、功耗和 NVML 的降頻原因，用來驗證這個假設。請不要把這個 benchmark 的單次結果當成答案。

CUDA roadmap 的 11 個 benchmark 全部通過驗證，結果見 [`cuda_roadmap/results.md`](cuda_roadmap/results.md)。不是每個「最佳化」都有效：bank conflict 的 padding 量到 1.00x，persistent grid-stride kernel 是 0.97x。

### 這些數字的限制

- **發送下限。** 低於下表時間的量測，反映的是主機端發送成本而不是 kernel 本身。Triton 的 Python 發送路徑下限最高，所以它 0.1 ms 以下的結果大多量到的是這個下限：

| 實作 | 發送下限（ms） |
|---|---|
| PyTorch | 0.0112 |
| Triton | 0.0553 |
| CUDA | 0.0092 |

- **雜訊。** 修正計時方式後，約 0.1 ms 以下的 kernel 在不同輪之間仍有變化，所以表格都附上範圍。
- **只有一台筆電。** 量測受散熱和功耗限制。編譯器也超出 CUDA 12.4 的支援範圍（MSVC 14.50 搭配 `-allow-unsupported-compiler`）。
- **沒有執行 Nsight Compute**，所以瓶頸解釋來自 roofline，不是硬體計數器。

---

## 快速開始

```bash
git clone https://github.com/boson316/RTX3050-GPU-Mastery.git && cd RTX3050-GPU-Mastery
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt "triton==3.2.*" pytest nvidia-ml-py      # Windows：triton-windows==3.2.*

pytest -q                              # kernel 正確性（GPU；沒有 GPU 時需要 CPU 版 torch 和最新版 Triton，與 CI 相同）
tests/cuda_emulation/run_all.sh        # 在 CPU 上驗證 CUDA kernel 邏輯（需要 g++ >= 11）
python benchmarks/gpu_specs.py         # 這張 GPU 的理論峰值
scripts/reproduce_all.sh               # 一次跑完：驗證、benchmark、報告、roofline
```

| 任務 | 指令 |
|---|---|
| CUDA roadmap | `cd cuda_roadmap && ./build.sh && cd .. && python cuda_roadmap/run_benchmarks.py` |
| Triton kernels | `python -m triton_kernels.run_benchmarks` |
| FlashAttention（CUDA vs Triton vs SDPA） | `python benchmarks/benchmark_attention.py` |
| Transformer 區塊 | `python benchmarks/benchmark_transformer.py` |
| Matmul / conv | `python benchmarks/benchmark_matmul.py`、`python benchmarks/benchmark_conv.py` |
| 完整報告 | `python tools/performance_dashboard.py --skip-mnist` → [`benchmarks/performance_report.md`](benchmarks/performance_report.md) |
| Roofline | `python profiling/roofline_analysis.py` |
| Nsight | `python profiling/run_nsight_profiling.py`（[說明](docs/nsight_profiling_guide.md)） |

Windows 上請在 *x64 Native Tools Command Prompt for VS* 裡編譯 CUDA extension。

---

## 學習資源

- [`tutorials/`](tutorials)：CUDA 基礎 → shared memory → tiling → Triton → FlashAttention → Transformer kernels
- [`docs/`](docs/INDEX.md)：記憶體階層、優化指南、roofline、Nsight、各 Level 的 kernel 說明
- [`notebooks/`](notebooks)：互動式範例

## 專案結構

```
cuda_roadmap/            Level 1-4 CUDA benchmark + common/bench_common.cuh（計時 + 驗證）
triton_kernels/          Triton kernels（matmul、linear/qkv/mlp、softmax、layernorm、gelu、conv、flash_attention）
flash_attention_simple/  CUDA FlashAttention：flash_attention_kernel.cuh、PyTorch extension、standalone
gpu_kernels/             transformer CUDA extension + 小型 standalone kernel
extension/               conv3x3 PyTorch extension（conv3x3_kernel.cuh）
benchmarks/              有驗證的 benchmark 框架（bench_utils.py）、gpu_specs.py、結果 CSV
profiling/               roofline 分析、Nsight 腳本與報告
tests/                   Triton interpreter 測試、cuda_emulation/（CUDA kernel 的 CPU 模擬器）
docs/ tutorials/ notebooks/
```

## 授權

[MIT](LICENSE)
