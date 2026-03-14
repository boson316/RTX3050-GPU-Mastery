"""
PyTorch GPU 矩陣乘法：與 CUDA matrixMul.cu 對照
需已安裝：pip install torch --index-url https://download.pytorch.org/whl/cu124
"""

import torch
import time

N = 512
# A, B 直接在 GPU 上建立，device='cuda' → tensor.device 為 cuda:0，避免後來再搬移
A = torch.rand(N, N, device="cuda")
B = torch.rand(N, N, device="cuda")

start = time.time()
C = torch.mm(A, B)  # GPU matmul（PyTorch 用 CUDA 實作）
torch.cuda.synchronize()  # 等 GPU 算完再計時
print(f"PyTorch GPU matmul: {(time.time() - start) * 1000:.1f} ms")
print("C[0, :5]:", C[0, :5])  # 檢查前 5 個
