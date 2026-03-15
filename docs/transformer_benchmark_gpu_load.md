# Transformer Benchmark：為什麼 GPU 會一直 100%、以及改善方式

## 為什麼會一直 100%？

1. **自訂 Fused QKV kernel 太慢**  
   原本用 **16×16 tiled matmul** 自訂 kernel 做 `y = x @ W_qkv + b`。  
   - 對 M=16384, K=768, N=2304 這種規模，要啟動大量 block、每個 block 做很多次 shared memory 讀寫與計算。  
   - 實測單次約 **90ms**，而 PyTorch（底層用 cuBLAS/cuDNN）只要約 **4ms**，約 **20 倍慢**。  
   - Benchmark 又對同一個 kernel 做 100+ 次重複測量 → GPU 長時間滿載，容易出現 100% 使用率、過熱或當機。

2. **滿載時間過長**  
   - 單一 kernel 慢 × 重複次數多 = 連續高負載時間很長。  
   - 筆電 GPU 散熱有限，長時間 100% 容易觸發降頻、驅動逾時或當機。

## 已做的程式碼改善

### 1. Fused QKV 改為 **cuBLAS matmul + 小 kernel 加 bias**（已實作）

- **位置**：`gpu_kernels/transformer/transformer_kernels.cu`  
- **做法**：  
  - 矩陣乘法 `y = x @ w` 改為呼叫 **cuBLAS `cublasGemmEx`**（FP16，與 PyTorch 同類庫），延遲會接近 PyTorch（約數 ms 等級）。  
  - bias 用一個輕量 kernel：`y[row, col] = b[col]`（或 0），再讓 cuBLAS 做 `y = x*w + y`，等價於 `y = x @ w + b`。  
- **效果**：  
  - Fused QKV 的「單次耗時」從約 90ms 降到約數 ms，大幅縮短每次測量時間。  
  - 總體 GPU 滿載時間變短，較不易一直維持 100% 或當機。

### 2. Benchmark 負荷控制（先前已做）

- **預設低負載**：較小 B/S、較少 WARMUP/REPEAT，且預設不跑自訂 CUDA。  
- **跑 CUDA 時減少重複**：自訂 CUDA 只測 2 warmup + 5 repeat，避免慢 kernel 被跑上百次。  
- **段落間休息**：各 benchmark 段落之間 sync + 短暫 sleep，讓 GPU 有間歇。

### 3. 之後可再做的優化（文件中 69–70 行提到的方向）

- **較大 tile（例如 32×32）**：若仍保留自訂 matmul kernel，可試 32×32 tile 並優化 shared memory 存取，減少 bank conflict、提高 occupancy。  
- **長期**：若要自訂 kernel 又接近 PyTorch 效能，可考慮 **WMMA / CUTLASS** 做 FP16 矩陣乘，再與 bias、activation 等 fuse。

## 小結

- **一直 100% 的原因**：主要是自訂 Fused QKV kernel 很慢（約 90ms/次），又被重複測很多次，導致 GPU 長時間滿載。  
- **已直接改的程式碼**：Fused QKV 改為 cuBLAS matmul + bias kernel，延遲與負荷都會明顯下降。  
- 若仍遇到當機，可繼續用「預設低負載」或「不開自訂 CUDA」（不設 `TRANSFORMER_BENCHMARK_CUDA=1`）。
