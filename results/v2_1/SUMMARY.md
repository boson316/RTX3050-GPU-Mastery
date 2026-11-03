# RTX3050-GPU-Mastery v2.1：實測摘要

所有數字都來自腳本輸出。比值由 [`aggregate.py`](aggregate.py) 算出，見 [`aggregate.log`](logs/aggregate.log)；README 表格由 [`make_tables.py`](make_tables.py) 產生。

## 先看這段：量測條件和 v2.0 不同

| 條件 | v2.0 | v2.1 | 來源 |
|---|---|---|---|
| 插電、最佳效能模式 | 是 | 是 | Phase 0 檢查 |
| 開始時 GPU 記憶體用量 | 0 MiB | 532～551 MiB（背景程式，未關閉） | [`idle_background.csv`](logs/idle_background.csv) |
| 每項輪數、預熱 | 3 輪、1 秒 | 3 輪、1 秒 | [`run_phase2.py`](run_phase2.py) |
| Transformer 負載下 SM 時脈中位數（3 輪） | 1755／1665／1732 MHz | 1732／1650／1687 MHz | v2：`results/v2/logs/transformer_run*_clocks.csv`；v2.1：[`aggregate.log`](logs/aggregate.log) |

背景程式包含 Parsec、Discord、ChatGPT／Codex、MSI Center（DCv2）、Nahimic、Edge WebView、Claude。使用者說「繼續」後，我沒有關閉它們，只記錄下來。

**對照組：沒有改動的 PyTorch 列。** Softmax、LayerNorm、GELU（受頻寬限制）最多變動 0.4%；QKV（受運算能力限制）慢 6.3%，MLP 慢 1.8%。所以下面的結論都用**同一輪內**的 CUDA 對 PyTorch，不用 v2.0 的絕對數字。

## v2.0 → v2.1 對照

延遲單位 ms，格式為中位數（最小～最大），各 3 輪。v2.0 來源 `results/v2/csv/run{1,2,3}/benchmark_transformer_results.csv`；v2.1 來源 [`csv/run{1,2,3}/benchmark_transformer_results.csv`](csv/)，中位數列在 [`csv/final/`](csv/final/)。

| Kernel | 實作 | v2.0 | v2.1 | 變化 | v2.0 GB/s | v2.1 GB/s |
|---|---|---|---|---|---|---|
| QKV | CUDA | 0.675（0.674～0.775） | 0.609（0.596～0.668） | -9.8% | 23.9 | 26.5 |
| QKV | PyTorch（對照） | 0.545（0.537～0.609） | 0.579（0.542～0.661） | +6.3% | 29.6 | 27.8 |
| MLP | CUDA | 1.681（1.679～1.726） | 1.603（1.595～1.663） | -4.6% | 9.4 | 9.8 |
| MLP | PyTorch（對照） | 1.612（1.584～1.708） | 1.641（1.617～1.691） | +1.8% | 9.8 | 9.6 |
| Softmax | CUDA | 0.094（0.093～0.099） | 0.053（0.053～0.054） | -43.1% | 67.0 | 117.7 |
| Softmax | PyTorch（對照） | 0.054（0.054～0.055） | 0.054（0.054～0.054） | -0.4% | 115.7 | 116.0 |
| LayerNorm | CUDA | 0.111（0.110～0.118） | 0.053（0.053～0.054） | -51.9% | 56.9 | 118.3 |
| LayerNorm | PyTorch（對照） | 0.056（0.056～0.077） | 0.056（0.056～0.056） | +0.0% | 111.8 | 111.7 |

**CUDA 對同一輪 PyTorch 的差距（3 輪）**

| Kernel | v2.0 | v2.1 |
|---|---|---|
| QKV | +25.4%、+27.4%、+23.8% | +9.9%、+1.1%、+5.1% |
| MLP | +6.0%、+1.1%、+4.3% | -1.7%、-1.4%、-2.3% |
| Softmax | +71.7%、+81.0%、+72.6% | -1.3%、-1.8%、-1.1% |
| LayerNorm | +96.4%、+52.8%、+96.6% | -5.5%、-5.9%、-5.0% |

**8192×768 的逐列 kernel**（只有 v2.1）

| Kernel | PyTorch | Triton | CUDA | Triton 對 CUDA（2048×768 → 8192×768） |
|---|---|---|---|---|
| Softmax | 0.208 ms · 121 GB/s | 0.212 ms · 119 GB/s | 0.207 ms · 122 GB/s | +6.0% → +2.6% |
| LayerNorm | 0.214 ms · 117 GB/s | 0.211 ms · 119 GB/s | 0.207 ms · 122 GB/s | +32.0% → +2.3% |
| GELU | 0.209 ms · 121 GB/s | 0.209 ms · 120 GB/s | 0.214 ms · 117 GB/s | -2.5% → -2.4% |

## 預期與實測

| 項目 | 預期 | 實測 | 結果 | 原因 |
|---|---|---|---|---|
| Softmax（CUDA） | ≥100 GB/s | 117.7 GB/s | **達標** | 每列只讀一次 DRAM、128-bit 讀取；warp 路徑確實有執行（[`check_paths.log`](logs/check_paths.log)） |
| LayerNorm（CUDA） | ≥100 GB/s | 118.3 GB/s | **達標** | 同上，而且比 PyTorch 快 5.0～5.9% |
| MLP（CUDA） | 與 PyTorch 打平（約 1.61 ms） | 1.603 ms，比同輪 PyTorch 快 1.4～2.3% | **達標** | bias 與 GELU 融進 epilogue |
| QKV（CUDA） | 與 PyTorch 打平（約 0.545 ms） | 0.609 ms，比同輪 PyTorch 慢 1.1～9.9% | **未完全達標** | 見下方 |
| Triton IEEE 誤差 | ≤1e-4 | 4.2e-07～9.7e-07 | **達標** | [`csv/final/`](csv/final/benchmark_attention_longseq_results.csv) |
| 8192×768 無 launch 下限影響 | Triton 不受影響 | 三個 kernel 都與 CUDA 相差 2.6% 以內 | **符合** | kernel 時間約為下限（0.055 ms）的 4 倍 |

**QKV 為什麼沒打平：**
- 差距中位數從 0.14 ms 降到 0.03 ms，可見獨立的 bias pass 確實是主因。
- 背景負載讓 0.545 ms 這個絕對目標變得不公平，因為 PyTorch 這次自己也只有 0.579 ms。
- 但這不能完全解釋差距：3 輪都是 CUDA 較慢，第 1 輪差到 9.9%。
- 剩下的差距**沒有量到原因**。MLP 呼叫兩次 cublasLt 仍贏過 PyTorch，所以不太像 cublasLt 每次呼叫的開銷，但這點沒有實際測試。

## 長序列 attention（B=1、H=8、D=64）

來源 [`csv/run{1,2,3}/benchmark_attention_longseq_results.csv`](csv/)。所有列都通過驗證。比值的中位數見 [`aggregate.log`](logs/aggregate.log)。

| 比較 | S=1024 | S=2048 | S=4096 |
|---|---|---|---|
| FP16：Triton 對 SDPA（誤差相同到 4 位有效數字） | -17.8% | -25.9% | -25.8% |
| IEEE FP32：CUDA 對 SDPA | -3.4% | +1.9% | -13.7% |
| IEEE FP32：Triton 對 SDPA | +14.2% | +20.7% | +4.6% |
| Triton IEEE ÷ Triton TF32 | 4.07 倍 | 4.24 倍 | 4.09 倍 |
| Triton TF32 ÷ Triton FP16 | 2.16 倍 | 2.14 倍 | 2.13 倍 |
| TF32 誤差／FP16 誤差 | 1.367e-03／3.642e-04 | 1.767e-03／2.649e-04 | 5.593e-04／1.029e-04 |

- v2.0 的「Triton 快 4 倍」確實是 TF32 對 IEEE 造成的。
- **預期之外：TF32 在這裡每項都輸給 FP16**，慢約 2 倍、顯存兩倍、誤差還更大。TF32 誤差為何比 FP16 大，**沒有量測**。

## WMMA 1024³ 分群的結論

來源 [`gpu_state_probe_fp16_matmul_bench.csv`](gpu_state_probe_fp16_matmul_bench.csv)（20 次）、[`gpu_state_probe_fp16_2048.csv`](gpu_state_probe_fp16_2048.csv)（10 次）、[`analyze_probe.log`](logs/analyze_probe.log)。

| 狀態變數 | 慢群（7 次） | 快群（13 次） |
|---|---|---|
| WMMA tiled GFLOPS（中位數） | 7570.9 | 10295.9 |
| WMMA naive GFLOPS | 2297.0 | 3281.9 |
| cuBLAS GFLOPS | 11155.1 | 10988.8 |
| SM 時脈 MHz | 990.0 | 990.0 |
| 記憶體時脈 MHz | 5501.0 | 5501.0 |
| 功耗 W | 9.0 | 7.7 |
| 溫度 °C | 65.0 | 65.0 |
| P-state | P0、P0/P3、P0/P3/P5、P0/P5 | P0、P0/P3、P0/P5、P3 |
| 降頻原因 | none、sw_power_cap+sw_thermal | none、sw_power_cap+sw_thermal |

**結論：沒有任何記錄到的狀態變數能分開這兩組，原因仍然未知。**
- **不是兩個緊密的群：** 7 次在 7410～7626，3 次在 8490～9362，10 次在 9849～10700。
- **v2.0 的「整次執行 GPU 狀態」假設得不到支持：** WMMA naive 和 tiled 同步變動（r = +0.99），但同一行程內緊接著計時的 cuBLAS 沒有（r = -0.36）。
- **探測工具的極限：** 每 20 ms 取樣一次，kernel 只跑約 0.2 ms。取樣到的時脈中位數 990 MHz 最多只能到 10136 GFLOPS，cuBLAS 卻量到 11397.6，代表樣本沒有反映 kernel 執行當下的時脈。這麼短暫的狀態變化不能排除。
- **2048³ 沒有分群：** 10901～11792 GFLOPS，10 次都碰到 `sw_power_cap`。
- **新數字：** 這次 2048³ WMMA 為 cuBLAS 的 75.6～83.0%（10 次），比 v2.0 的 79.1～86.5%（6 次）低，當時有背景程式。

## 新 bug

**沒有發現新的程式 bug，所以 `docs/ENGINEERING_LOG.md` 沒有新增條目（#18 未使用）。** 有幾件事要交代：

- **任務提供的 profiler 片段無法確認 kernel 名稱。** 只開 CUDA activity 時會報錯 `Legacy CUDA profiling requires use_cpu=True`，加上 CPU activity 後表格只有 `aten::*`（[`kernel_names.log`](logs/kernel_names.log)）。我改用 [`check_paths.py`](check_paths.py) 在暫存副本加追蹤訊息，並開 `CUBLASLT_LOG_LEVEL=5` 確認路徑（[`check_paths.log`](logs/check_paths.log)）。repo 原始碼沒有動。
- **README 舊說法修正。** 「整次執行一起變慢、也影響 cuBLAS，代表是 GPU 狀態」已改為上面的結論。
- **`transformer_cuda.cpp` 的註解尚未驗證。** 它寫 cuBLAS GELU 與 erf 差距小於 1e-3。我量到的是整個 MLP 輸出的誤差（對 tanh 1.552e-03、對 erf 1.643e-03），這不是單獨比 GELU，無法判斷註解對錯，也分不出 cublasLt 用哪種形式。
- **第一次執行 Phase 2 的指令回傳 `Exit code 4`，原因未查明。** 當時只完成閒置取樣，沒有產生量測結果；之後拆開重跑才正常。

## 其他說明

- **檢查：** ruff 通過（[`ruff_final.log`](logs/ruff_final.log)）；pytest 129／129（[`pytest_final.log`](logs/pytest_final.log)）；CPU 模擬器 39／39（[`emulator_final.log`](logs/emulator_final.log)）。容差、skip、測試和參考答案都沒有改動。
- **任務沒列但我多跑的項目：** Roofline 3 輪。README 的 Roofline 表有舊的 CUDA 百分比，不更新會和新數字矛盾。表中只換 QKV、MLP、softmax、LayerNorm、GELU 這幾列，其他維持 v2.0。
- **README 的 transformer 列整列換成 v2.1**，包含 PyTorch 和 Triton 欄，同一列才能同輪比較。v2.0 對照表放在下方。
- **小干擾：** WMMA 探測和 Roofline 量測期間，我各跑過一次純 CPU 的彙整腳本（約 1 秒），另外在探測期間讀過檔案。
- **`benchmarks/plots/transformer_*.png` 和 Roofline 圖來自最後一輪**，不是 3 輪中位數；`benchmarks/transformer_benchmark.png` 則由中位數 CSV 產生。
- **作品集的「79–87%」維持 v2.0 的數字**（標明 6 次，測試也檢查這個字串），沒有改成這次的 75.6～83.0%。

## 尚未推送的 commit

`git log --oneline origin/main..HEAD`，表列內容不含加入本摘要的 commit。

