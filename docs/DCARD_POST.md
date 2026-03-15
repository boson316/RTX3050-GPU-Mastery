# [心得] 7小時 RTX3050 GPU programming → 1.5x cuDNN (Extension+Triton)

大家好，我是資工大二生。

這週末用 **RTX 3050 laptop GPU** 衝了 7 小時，做出這些東西：

---

## 成果總覽

1. **Pure CUDA**：matrix mul **521x speedup**（N=1024, 5ms）
2. **Reduction**：1M elements **0.763ms**
3. **PyTorch MNIST CNN**：**99% accuracy**
4. **CUDA Extension**：FP16 conv **1.50x PyTorch cuDNN**（B=1024）
5. **Triton Python kernel**：**1.27x PyTorch**

---

## Benchmark（B=1024 FP16）

| 方法 | 時間 | 加速比 |
|------|------|--------|
| Torch (cuDNN) | 1.20ms | 1.00x |
| Extension | 0.81ms | **1.49x** |
| Triton | 1.08ms | **1.27x** |

---

完整程式碼都在這裡，歡迎參考或一起玩：

🔗 **https://github.com/boson316/RTX3050-GPU-Mastery**

這次全程用 **Cursor Pro + Claude Sonnet** 輔助開發，從 CUDA kernel 到 Triton 寫起來都滿順的。

目標是考 **NTU 資工碩**，想進 **HPC lab** 或 **NVIDIA intern**，有同路人可以交流一下～

#資工 #GPU #PyTorch #CUDA #CUDAExtension #Triton #深度學習

---

## 純文字版（直接複製到 Dcard）

```
[心得] 7小時 RTX3050 GPU programming → 1.5x cuDNN (Extension+Triton)

大家好，我是資工大二生。

這週末用 RTX 3050 laptop GPU 衝了 7 小時，做出這些東西：

1. Pure CUDA：matrix mul 521x speedup（N=1024, 5ms）
2. Reduction：1M elements 0.763ms
3. PyTorch MNIST CNN：99% accuracy
4. CUDA Extension：FP16 conv 1.50x PyTorch cuDNN（B=1024）
5. Triton Python kernel：1.27x PyTorch

Benchmark（B=1024 FP16）：
- Torch (cuDNN)：1.20ms
- Extension：0.81ms（1.49x）
- Triton：1.08ms（1.27x）

完整程式碼：https://github.com/boson316/RTX3050-GPU-Mastery

用 Cursor Pro + Claude Sonnet 全程輔助，超順。
目標 NTU 資工碩，想進 HPC lab / NVIDIA intern，同路人歡迎交流～

#資工 #GPU #PyTorch #CUDA
```
