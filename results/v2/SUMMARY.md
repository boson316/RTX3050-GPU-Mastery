# RTX3050-GPU-Mastery v2：實測摘要

## 環境

驅動程式 610.62、CUDA 12.4（nvcc 12.4.99）、PyTorch 2.6.0+cu124、Triton 3.2.0、Python 3.11.0、MSVC 14.50（VS 2026）。完整內容見 [environment.md](environment.md)。

```
NVIDIA GeForce RTX 3050 6GB Laptop GPU: 20 SMs, sm_86, SM clock 2100.0 MHz, mem clock 5501.0 MHz, bus 96-bit, L2 1536 KB
  peak fp32             : 10.8 TFLOPS
  peak fp16_tc_fp32acc  : 21.5 TFLOPS
  peak fp16_tc_fp16acc  : 43.0 TFLOPS
  peak DRAM bandwidth : 132 GB/s
  ridge point (FP16 TC, FP32 acc): 163 FLOP/byte
```

## Phase 1：正確性

最終程式碼上的結果：

| 步驟 | 結果 | Log |
|---|---|---|
| ruff | 通過 | [ruff.log](logs/ruff.log) |
| pytest（GPU，含 Triton、autotune、三個 extension、計時工具） | 116 / 116 | [pytest.log](logs/pytest.log) |
| pytest（模擬 CI：CPU 版 torch、最新 Triton；僅 Windows） | 53 通過、17 skip、0 失敗 | [pytest_ci_like.log](logs/pytest_ci_like.log) |
| CUDA CPU 模擬器（MSVC） | 27 / 27 | [emulator_msvc.log](logs/emulator_msvc.log) |
| CUDA roadmap | 11 / 11 PASS | [roadmap.log](logs/roadmap.log) |
| FlashAttention standalone | 8 / 8 PASS | [fa_standalone.log](logs/fa_standalone.log) |
| 三個 PyTorch extension | 全部 OK | [check_extensions.log](logs/check_extensions.log) |

模擬器修改紀錄：`cuda_emu.h` 這次沒有改，所以沒有重跑 g++ 和 clang。

## 關鍵數字

延遲格式為中位數（3 輪最小值～最大值），單位 ms。來源：`csv/final/benchmark_*_results.csv`、`csv/final/latency_ranges.json`。

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

裝置峰值。來源：`environment.md`、`logs/*_run[123]_clocks.csv`。

| 指標 | 數值 |
|---|---|
| NVML 最大時脈的峰值 | 21.5 TFLOPS @ 2100 MHz |
| 負載下 SM 時脈中位數 | 1747 MHz |
| 持續時脈的峰值 | 17.9 TFLOPS |
| DRAM 峰值頻寬 | 132 GB/s |

Roofline。來源：`csv/run[123]/roofline_results.csv`。

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

長序列 FlashAttention。來源：`csv/run[123]/longseq_attention.csv`、`logs/diag_sdpa.log`。

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

WMMA 與 cuBLAS。來源：`logs/wmma_final.log`、`logs/phase4_wmma_2048_baseline.log`。

| 指標 | 中位數 | 最小 | 最大 | 次數 |
|---|---|---|---|---|
| 1024³ WMMA tiled (GFLOPS) | 7530.6 | 7489.8 | 10979.9 | 12 |
| 1024³ cuBLAS (GFLOPS) | 11366.8 | 6944.2 | 11459.8 | 12 |
| 1024³ WMMA % of cuBLAS (%) | 81.4 | 65.6 | 109.6 | 12 |
| 2048³ WMMA tiled (GFLOPS) | 11648.3 | 11358.3 | 12122.3 | 6 |
| 2048³ cuBLAS (GFLOPS) | 14123.6 | 13969.4 | 14364.1 | 6 |
| 2048³ WMMA % of cuBLAS (%) | 81.9 | 79.1 | 86.5 | 6 |

發送下限。來源：`logs/diag_timing.log`。

| 實作 | 發送下限（ms） |
|---|---|
| PyTorch | 0.0112 |
| Triton | 0.0553 |
| CUDA | 0.0092 |

CUDA roadmap 全表見 [`cuda_roadmap/results.md`](../../cuda_roadmap/results.md)。

## 新發現並修正的 bug

詳見 [`docs/ENGINEERING_LOG.md`](../../docs/ENGINEERING_LOG.md)。

- **#13** FlashAttention JIT 在目前的 MSVC 上編譯失敗（少了 `-allow-unsupported-compiler`）。
- **#14** 照 README 安裝會把 CUDA 版 PyTorch 換成 CPU 版。只修了文件，沒有自動化測試。
- **#15** 計時前沒有拉高時脈，同一個 kernel 最多差 16 倍。修正前的資料隔離在 `csv/pre_fix_run*`，沒有採用。
- **#16** Roofline 沒有驗證、默默丟掉失敗項目，還把 `torch.randn` 算進延遲。
- **#17** Dashboard 用寫死的「~520x」舊數字蓋掉圖；另外兩張圖的計時也沒有驗證。

## Phase 4：優化嘗試

三項都沒有改動 kernel，結論見 [`CHANGELOG.md`](../../CHANGELOG.md)。

- **A. WMMA：** 2048³ 已達 cuBLAS 的 79.1～86.5%，超過 60% 目標，所以沒有嘗試 `cp.async` 和 128×128 tile。
- **B. Triton softmax：** 單次讀取最多快約 1%，已 revert。Log：`logs/phase4_softmax_run*.log`。
- **C. Triton FlashAttention：** 沒有候選設定明顯勝過現有 autotune。Log：`logs/phase4_attention_run*.log`。

## 沒做完或有疑慮的地方

- **WMMA 在 1024³ 的分群現象原因不明。** 整次執行會落在約 7.5 或約 10.9 TFLOPS，cuBLAS 也偶爾偏低。1 秒預熱無法消除。
- **0.1 ms 以下的 kernel 仍有雜訊。** 修正後仍可能相差超過 5%，最終 3 輪比較有 24 筆超過 5%（`logs/compare_runs.log`）。這些項目都有跑第 3 輪並附上範圍。
- **C++ 計時沒有預熱。** CUDA roadmap 和 standalone 的數字仍用原本的 warm-up，因為加了預熱也沒有改善。FlashAttention standalone 的時間也可能因此偏慢。
- **編譯器超出支援範圍。** MSVC 14.50 超出 CUDA 12.4 的支援範圍，靠 `-allow-unsupported-compiler` 編譯。所有驗證都通過，但這不是官方支援的組合。
- **Triton 版本取決於環境。** GPU 環境要用 3.2 才符合 PyTorch 2.6；沒有 GPU 時 3.2 會在 import 階段失敗，需要較新版本。
- **CI 尚未實際跑過。** CI 用 Linux 和不鎖版本的 Triton，這裡只在 Windows 上模擬過 pytest 步驟。
- **Nsight 沒有執行**（`ncu` 不在 PATH）。瓶頸解釋來自 roofline，不是硬體計數器。
- **`mnist_acc_loss.png` 沒有重新產生**（全程使用 `--skip-mnist`），README 也沒有引用。
- **`docs/benchmarks.md` 保留舊數字**，頁首已標明是歷史紀錄。
- **`profiling/roofline_analysis/plot_roofline.py` 只更新了 fallback 值**，這次沒有執行。
- **量測條件。** 插電和最佳效能模式有確認；是否有其他程式佔用 GPU，只在開始時檢查過一次。

## 尚未推送的 commit

`git log --oneline origin/main..HEAD`（不含加入本檔的最後一個 commit）：

```
8b89a07 docs(results): record Phase 4 attempts; no kernel changed
0d4ec7d docs: publish measured v2 results; release v2.0.0
d305e8f chore(results): add verified Phase 2 measurements, logs and charts
d5f8c88 fix(benchmarks): make all timing verified, preheated and CSV-sourced
bbf4a05 chore(results): add Phase 1 logs and check scripts
d503af8 docs(log): record the JIT flag and install bugs found on the GPU
2d5503a docs(install): stop the documented install from replacing CUDA PyTorch
3cde07a fix(attention): pass -allow-unsupported-compiler to the JIT build; test extensions
0cbe8d1 chore(results): record v2 environment and MSVC wrapper script
14c87a0 fix(roofline): correct fallback peaks to the values read from this GPU
873b316 chore: .gitattributes - CRLF for .bat, LF for .sh
db3519a test: make the CUDA emulator build with MSVC; add run_all.bat; emulator step in reproduce_all.bat
df6e8ed style: fix import order (stale triton/ dir made ruff treat triton as first-party)
a705569 docs: rewrite README (EN/zh-TW) around verified results and the verification pipeline
cc8e93d docs: add engineering log of bugs found and fixed; update script references in READMEs
b72e6f7 ci: real checks - ruff, Triton interpreter tests, nvcc -Werror compile, CUDA CPU emulation
258a1d9 fix(benchmarks): derive peaks from the device, drop invalid results, fix roofline plot
adeae17 chore: drop legacy triton/ copy of the conv kernel (triton_kernels/conv is canonical); ruff
ef344ae style: ruff (E, F, I, B) clean
7c94fb4 fix(triton): wrong-result bugs found by new interpreter-based test suite
6ad7a28 fix(cuda_roadmap): verify every kernel, fix WMMA layout bug, fair benchmarks
78869be fix(cuda): correctness bugs in attention, transformer and conv kernels; add CPU emulator tests
c83663b chore: remove build artifacts, duplicate scripts and non-technical files
```
