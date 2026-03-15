# Level 3: Advanced GPU Techniques — 說明與效能比較

## 1. Warp Shuffle Reduction

**操作**：對陣列做 sum reduction（同 Level 1），比較 shared memory 與 warp shuffle。

### Naive 實作
- 完全在 shared memory 內做樹狀 reduction（同 Level 1 optimized）。
- 最後一個 warp 的 32 個元素仍用 shared memory 做多步。

### Optimized 實作
- **Warp 內**：用 `__shfl_down_sync` 做 reduction，不需 shared memory；每次 thread 與對應 lane 的 thread 相加，log2(32)=5 步得到 warp 內總和。
- **Warp 間**：每個 warp 的結果寫入 shared memory 一個位置，再用一個 warp 做一次 shuffle reduction。
- 效果：減少 shared memory 使用與 bank conflict 機會、降低延遲。

### 效能比較
- **Shared only**：已很快。
- **Warp shuffle**：通常略快或相當，且程式更精簡。
- **學習重點**：`__shfl_down_sync`、lane index、warp 內 cooperation。

---

## 2. Fused Operations

**操作**：C = relu(A + B)，即先相加再做 ReLU。

### Naive 實作
- **兩個 kernel**：第一個 kernel 做 A+B 寫入 C；第二個 kernel 讀 C、做 ReLU、寫回 C。
- 缺點：兩次 global 讀寫、兩次 launch 開銷。

### Optimized 實作
- **單一 kernel**：每個 thread 讀 A[i]、B[i]，算 `s = A[i]+B[i]`，再 `C[i] = s > 0 ? s : 0`。
- 效果：一次 global 讀寫、一次 launch，頻寬與 launch 開銷都減少。

### 效能比較
- **兩次 kernel**：較慢。
- **Fused**：明顯較快。
- **學習重點**：Kernel fusion 對頻寬與 launch 的影響、常見融合模式（add+relu、scale+add 等）。

---

## 3. Persistent Kernel

**操作**：Vector add C = A + B，比較「多 block 各做一塊」與「少 block 迴圈做完全部」。

### Naive 實作
- **傳統**：block 數 = N / blockDim.x，每個 block 只執行一次，處理一塊後結束。
- 若 N 很大，block 數多；若 N 較小或 block 過多，可能 launch 開銷或 scheduling 影響效率。

### Optimized 實作
- **Persistent**：只 launch 少數 block（例如 128 個），每個 thread 用迴圈處理多個索引：`for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < N; i += gridDim.x*blockDim.x)`。
- 效果：單一 launch 即可覆蓋整個問題、可提高 occupancy、減少 small problem 的 block 數過多問題。

### 效能比較
- **Normal**：對大 N 通常已足夠快。
- **Persistent**：在問題較小或希望固定 block 數時有優勢，有時略快。
- **學習重點**：Grid persistence、何時用 persistent、與 occupancy 的關係。
