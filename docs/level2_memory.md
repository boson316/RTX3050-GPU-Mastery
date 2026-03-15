# Level 2: Memory Optimization — 說明與效能比較

## 1. Tiled Matrix Multiply

**操作**：C = A × B，N×N（N=1024），與 Level 1 naive 相同問題，改用 shared memory 優化。

### Naive 實作（同 Level 1）
- 每 thread 從 global 讀 A 的一 row、B 的一 column，計算一個 C 元素。

### Optimized 實作
- **16×16 tile**：每個 block 負責 C 的一塊 16×16；將對應的 A、B 小塊載入 `__shared__`（As[TILE][TILE], Bs[TILE][TILE]）。
- 沿 K 維度迴圈：每次載入一對 tile，block 內同步後做內積，再載入下一對。
- 每個 C 元素從 global 讀取次數由 O(N) 降為 O(N/TILE)，大幅提升 arithmetic intensity。

### 效能比較
- **CPU**：仍為三重迴圈，時間遠大於 GPU。
- **CUDA naive**：僅用 global，較慢。
- **CUDA tiled**：明顯加速，可接近或達數百倍於 CPU（視 N 與 GPU）。
- **學習重點**：Shared memory tiling、`__syncthreads()`、tile 與 block 對應。

---

## 2. Memory Coalescing

**操作**：從陣列 `in` 複製到 `out`，比較兩種存取模式。

### Naive（Bad）— Strided
- Thread `i` 讀寫 `in[i * STRIDE]`、`out[i * STRIDE]`（STRIDE=256）。
- 同一 warp 內 32 個 thread 存取相距 256 個元素的位址 → 多個 cache line、多個 transaction，頻寬浪費。

### Optimized（Good）— Coalesced
- Thread `i` 讀寫 `in[i]`、`out[i]`。
- 同一 warp 存取連續 128 bytes（32 個 float）→ 合併成少數 transaction，頻寬利用最佳。

### 效能比較
- **Strided**：明顯較慢。
- **Coalesced**：較快，愈接近理論頻寬。
- **學習重點**：Warp 內連續存取、coalescing 規則、如何避免 strided access。

---

## 3. Bank Conflict

**背景**：Shared memory 分成多個 bank（常見 32 個）；同一 warp 內多個 thread 存取同一 bank 的不同位址會產生 bank conflict，需 serialized。

### Naive（Bad）
- 使用 `sdata[tid * BANKS]` 等 layout，使同一 warp 的 thread 存取同一 bank 的不同 word → 發生 conflict。

### Optimized（Good）
- **Padding**：使用 `sdata[tid + tid/32]` 等索引，讓相鄰 thread 對應到不同 bank，避免 32-way conflict。
- 或選用不會造成 conflict 的 addressing（例如 sequential）。

### 效能比較
- **Conflict**：shared memory 延遲增加、throughput 下降。
- **No conflict**：較快。
- **學習重點**：32 banks、4-byte stride、padding 技巧。
