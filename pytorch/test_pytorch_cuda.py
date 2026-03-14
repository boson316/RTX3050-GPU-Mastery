"""
PyTorch CUDA 安裝驗證 + GPU tensor 加法
安裝：pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
執行：python test_pytorch_cuda.py
"""

import torch

# 檢查 CUDA 是否可用（device：目前 PyTorch 看到的 GPU）
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))  # 例如 RTX 3050

# GPU tensor 加法（a, b 的 .device 都是 'cuda:0'，計算在 GPU 上完成，c 也在 cuda:0）
a = torch.tensor([1.0, 2.0, 3.0], device="cuda")
b = torch.tensor([4.0, 5.0, 6.0], device="cuda")
c = a + b
print(c)  # tensor([5., 7., 9.], device='cuda:0')
