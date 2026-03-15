# Benchmarks — 可重現效能測試與圖表

## 綜合 Benchmark 套件（推薦）

從 repo 根目錄執行。每個腳本比較 **PyTorch / CUDA kernel / Triton kernel**，並輸出：

- **延遲 (ms)**、**吞吐量 (GFLOPS)**、**記憶體頻寬 (GB/s)**、**GPU 使用率 (%)**
- **CSV**：`benchmarks/benchmark_<name>_results.csv`
- **終端表格**：即時印出
- **Matplotlib 圖表**：儲存於 `benchmarks/plots/`

| 腳本 | 說明 |
|------|------|
| `benchmark_matmul.py` | 矩陣乘法：PyTorch (cuBLAS) vs Triton vs CUDA (standalone 1024³) |
| `benchmark_conv.py` | 3×3 Conv2D：PyTorch vs Triton vs CUDA (extension) |
| `benchmark_attention.py` | Attention：PyTorch SDPA vs Triton Flash vs CUDA (flash_attention_simple) |
| `benchmark_transformer.py` | Transformer 子 kernel：QKV / Softmax / LayerNorm / GELU / MLP（PyTorch vs Triton vs CUDA） |

```bash
python benchmarks/benchmark_matmul.py
python benchmarks/benchmark_conv.py
python benchmarks/benchmark_attention.py
python benchmarks/benchmark_transformer.py
```

**Transformer CUDA**：預設關閉以降低負載；啟用請設環境變數 `TRANSFORMER_BENCHMARK_CUDA=1`。

## 圖檔輸出目錄

| 目錄 | 說明 |
|------|------|
| `benchmarks/plots/` | 由 `benchmark_*.py` 產出的 PNG 圖表（延遲、GFLOPS、依 config 比較） |

## 舊版腳本（仍可使用）

| 腳本 | 說明 |
|------|------|
| `matmul_benchmark.py` | PyTorch vs Triton matmul，多種 N、dtype |
| `conv_benchmark.py` | torch.nn.Conv2d vs custom_conv vs Triton conv |
| `attention_benchmark.py` | PyTorch SDPA vs Triton Flash Attention |
| `transformer_benchmark.py` | Transformer 子 kernel 終端輸出（無 CSV/圖表） |

```bash
python benchmarks/matmul_benchmark.py
python benchmarks/conv_benchmark.py
python benchmarks/attention_benchmark.py
python benchmarks/transformer_benchmark.py
```

## 其他圖檔

| 檔名 | 說明 |
|------|------|
| `generate_charts.py` | 產生 matrix_mul_speedup、conv_benchmark、mnist_acc_loss 等圖（置於 `benchmarks/`） |
| `matrix_mul_speedup.png` | Matrix multiplication 比較 (N=1024) |
| `mnist_acc_loss.png` | MNIST 訓練 loss / accuracy 曲線 |
| `conv_benchmark.png` | Conv torch vs Extension vs Triton 比較 |
