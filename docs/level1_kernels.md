# Level 1: Basic Kernels — 說明與效能比較

## 1. Vector Add

**操作**: `C[i] = A[i] + B[i]`，N = 2^20。

### Naive 實作
- 每個 thread 負責一個索引 `i`，從 global memory 讀 `A[i]`、`B[i]`，寫回 `C[i]`。
- 存取已具備 coalescing（相鄰 thread 存取相鄰位址），是良好的 baseline。

### Optimized 實作
- 使用 **float4** 向量化：每個 thread 一次處理 4 個 float，載入/儲存時用 `float4`。
- 效果：thread 數變為 1/4、記憶體 transaction 次數減少，頻寬利用更佳。

### 效能比較
- **CPU**：單執行緒迴圈，通常慢兩個數量級以上。
- **PyTorch**：`a + b` 使用高度優化的 kernel，常優於自寫 naive，接近或略優於 float4 版（視 GPU 與驅動而定）。
- **學習重點**：Coalescing、向量化載入、CUDA events 計時。

---

## 2. Parallel Reduction

**操作**：對長度 N 的陣列求和（N = 2^20）。

### Naive 實作
- 每個 thread 讀一個元素，用 **atomicAdd** 加到一個 global 變數。
- 缺點：大量 serialization，global atomic 非常慢，僅作教學對比用。

### Optimized 實作
- **Shared memory 樹狀 reduction**：每個 block 將 BLOCK_SIZE 個元素載入 `__shared__`，在 block 內做 log(BLOCK_SIZE) 步樹狀加法，最後每個 block 寫出一個 partial sum；再於 host 或第二個 kernel 加總。
- 效果：大幅減少 global 存取與 atomic 競爭。

### 效能比較
- **CPU**：單執行緒 sum 迴圈，慢很多。
- **PyTorch**：`x.sum()` 使用優化過的 reduction kernel，通常最快。
- **Naive CUDA**：atomicAdd 版本極慢，僅用於對比。
- **學習重點**：Shared memory、`__syncthreads()`、樹狀 reduction。

---

## 3. Naive Matrix Multiply

**操作**：C = A × B，N×N（N=1024）。

### Naive 實作
- 每個 thread 計算 C 的一個 (row, col)：從 global 讀取 A 的一整 row、B 的一整 column，內積後寫回 C。
- 每個輸出元素需 2N 次 global 讀取 + 1 次寫入，頻寬壓力大、未利用 shared memory。

### Optimized 實作（在 Level 2）
- Level 2 的 **tiled matrix multiply** 使用 shared memory 降低 global 讀取，為本 kernel 的優化版。

### 效能比較
- **CPU**：三重迴圈 O(N³)，時間通常遠大於 GPU。
- **PyTorch**：`torch.mm(a, b)` 使用 cuBLAS，效能最佳。
- **Naive CUDA**：比 CPU 快很多，但遠不如 tiled 與 cuBLAS。
- **學習重點**：Grid/block 對應到二維輸出、global 讀取量分析。
