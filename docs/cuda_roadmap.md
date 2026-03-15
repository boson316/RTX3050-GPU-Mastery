# Progressive CUDA Kernel Learning & Optimization Roadmap

本路線圖依難度分為四個層級，每個 kernel 均包含 **naive 實作**、**optimized 實作**、**效能比較** 以及 **與 CPU / PyTorch 的對照 benchmark**。

---

## Level 1: Basic Kernels

| Kernel | 說明 | Naive | Optimized | 文件 |
|--------|------|--------|-----------|------|
| **Vector Add** | C[i] = A[i] + B[i] | 每 thread 一元素，基本 coalescing | float4 向量化載入/儲存 | [level1_kernels.md](level1_kernels.md) |
| **Parallel Reduction** | Sum of array | 每 thread 一元素 + atomicAdd | Shared memory 樹狀 reduction | 同上 |
| **Naive Matrix Multiply** | C = A×B | 每 thread 一輸出，全從 global 讀取 | (Level 2 為 tiled 版) | 同上 |

**學習重點**：thread/block/grid 對應、global memory 存取、CUDA events 計時。

---

## Level 2: Memory Optimization

| Kernel | 說明 | Naive | Optimized | 文件 |
|--------|------|--------|-----------|------|
| **Tiled Matrix Multiply** | C = A×B | Level 1 naive | 16×16 shared memory tile | [level2_memory.md](level2_memory.md) |
| **Memory Coalescing** | 連續 vs 跨步存取 | Strided 讀取 (bad) | Coalesced 讀取 (good) | 同上 |
| **Bank Conflict** | Shared memory  bank | 衝突存取 (bad) | Padding 避免衝突 (good) | 同上 |

**學習重點**：shared memory、coalescing、bank 與 padding。

---

## Level 3: Advanced GPU Techniques

| Kernel | 說明 | Naive | Optimized | 文件 |
|--------|------|--------|-----------|------|
| **Warp Shuffle Reduction** | 單 block 內 sum | Shared memory 樹狀 | `__shfl_down_sync` 無 shared | [level3_advanced.md](level3_advanced.md) |
| **Fused Operations** | 多步驟合併 | 兩次 kernel (e.g. add + relu) | 單一 kernel 融合 | 同上 |
| **Persistent Kernel** | 單次 launch 多輪 | 多個 block 各做一塊 | 少數 block 迴圈處理全部 | 同上 |

**學習重點**：warp 階層、kernel 融合、persistent 設計。

---

## Level 4: Tensor Core Usage

| Kernel | 說明 | Naive | Optimized | 文件 |
|--------|------|--------|-----------|------|
| **FP16 Tensor Core Matmul** | 半精度矩陣乘 | FP32 CUDA core | WMMA / Tensor Core FP16 | [level4_tensor_core.md](level4_tensor_core.md) |
| **WMMA Example** | 使用 wmma  API | — | 16×16×16 矩陣塊 | 同上 |

**學習重點**：Tensor Core、WMMA、FP16 精度與效能。

---

## 目錄結構

```
cuda_roadmap/
├── README.md
├── build.bat / build.sh
├── run_benchmarks.py       # 全路線圖 vs CPU/PyTorch
├── level1_basics/
│   ├── vector_add/        # vector_add_bench.cu
│   ├── reduction/         # reduction_bench.cu
│   └── naive_matmul/      # naive_matmul_bench.cu
├── level2_memory/
│   ├── tiled_matmul/      # tiled_matmul_bench.cu
│   ├── coalescing/        # coalescing_bench.cu
│   └── bank_conflict/     # bank_conflict_bench.cu
├── level3_advanced/
│   ├── warp_shuffle_reduction/
│   ├── fused_ops/
│   └── persistent_kernel/
└── level4_tensor_core/
    ├── fp16_tensor_core_matmul/  # fp16_matmul_bench.cu
    └── wmma_example/             # wmma_example.cu
```

## 執行 Benchmark

```bash
# 編譯所有 roadmap kernels（需 nvcc）
cd cuda_roadmap && ./build.sh   # 或 build.bat on Windows

# 執行完整比較（CPU、PyTorch、CUDA naive、CUDA optimized）
python run_benchmarks.py
```

詳細每個 kernel 的實作說明與效能解讀請見 `docs/level1_kernels.md` 等文件。
