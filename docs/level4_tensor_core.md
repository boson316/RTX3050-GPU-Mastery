# Level 4: Tensor Core Usage — 說明與效能比較

## 1. FP16 Tensor Core Matrix Multiply

**操作**：C = A × B，M×K × K×N = M×N，使用 FP16；比較 CUDA core 與 Tensor Core。

### Naive 實作
- **FP16 資料**：A、B、C 皆為 `__half`。
- **計算**：用一般 thread 做 16×16 tile 的乘加，每個元素用 `__half2float` 轉成 float 計算後再寫回 half。
- 使用一般 **CUDA cores**，未呼叫 Tensor Core。

### Optimized 實作（WMMA）
- 使用 **WMMA API**（`nvcuda::wmma`）：`wmma::fragment`、`load_matrix_sync`、`mma_sync`、`store_matrix_sync`。
- 以 **16×16×16** 為單位：一個 warp 負責一塊 16×16 的 C，從 A 取 16×16、B 取 16×16，做一次 `mma_sync`，由 **Tensor Core** 執行。
- 效果：在支援的 GPU（Volta、Turing、Ampere）上，矩陣乘的 throughput 可大幅提升。

### 效能比較
- **FP16 naive（CUDA core）**：已比 FP32 省頻寬，但算力未用 Tensor Core。
- **WMMA（Tensor Core）**：通常明顯較快，尤其大矩陣。
- **學習重點**：Tensor Core 適用範圍（矩陣乘）、FP16、WMMA 的 16×16×16 塊。

---

## 2. WMMA Example

**檔案**：`cuda_roadmap/level4_tensor_core/wmma_example/wmma_example.cu`

**操作**：最小範例 — 單一 16×16×16 矩陣乘，一個 warp 完成。

- **A**：16×16 row-major；**B**：16×16 column-major（WMMA 要求）；**C**：16×16。
- 步驟：`fill_fragment(acc, 0)` → `load_matrix_sync(a_frag, A, ...)` → `load_matrix_sync(b_frag, B, ...)` → `mma_sync(acc, a_frag, b_frag, acc)` → `store_matrix_sync(C, acc, ...)`。
- **Build**：需 `-arch=sm_70` 或以上（如 sm_86 for Ampere）。

### 學習重點
- Fragment 型別與維度（matrix_a, matrix_b, accumulator）。
- Row-major / column-major 與 stride。
- 一個 warp（32 threads）對應一塊 16×16 輸出。

---

## 編譯與執行

Level 4 需 **Compute Capability 7.0+**（Volta 起）：

```bash
nvcc -arch=sm_70 -o fp16_matmul_bench fp16_matmul_bench.cu   # or sm_86 for RTX 3050
nvcc -arch=sm_70 -o wmma_example wmma_example.cu
```

若編譯時出現 WMMA 相關錯誤，請確認 GPU 為 Volta/Turing/Ampere 並使用對應 `-arch`（例如 sm_86）。
