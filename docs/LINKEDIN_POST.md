# LinkedIn Post Draft — RTX 3050 GPU Portfolio

**Suggested post (English):**

---

I spent a week pushing my laptop GPU to its limits — and ended up with a custom conv layer that beats cuDNN by 1.5x on my RTX 3050.

What I built:
✅ CUDA from scratch: vector add → matrix multiply (~521x speedup) → reduction  
✅ PyTorch: MNIST to 99%+ in ~1.5 s/epoch with AMP  
✅ PyTorch C++ CUDA Extension: 3×3 conv with FP16, shared-memory tiling, sm_86  
✅ Triton: same conv in a fraction of the code, 1.27x vs torch  

All on one laptop GPU. No cloud, no A100 — just CUDA, PyTorch, and a lot of reading.

Repo: [YOUR_REPO_URL]  
#CUDA #PyTorch #GPU #DeepLearning #NVIDIA #Triton

---

**繁中版本（可選）：**

用一週在筆電 RTX 3050 上從 CUDA 基礎做到自寫 Conv 比 cuDNN 快 1.5x。

做了什麼：
✅ CUDA：vector add → matrix multiply（約 521x 加速）→ reduction  
✅ PyTorch：MNIST 99%+，AMP 約 1.5 s/epoch  
✅ PyTorch C++ Extension：3×3 conv FP16、shared memory tiling、sm_86  
✅ Triton：同一個 conv，少 80% code，1.27x torch  

全部在一張筆電 GPU 上完成。Repo: [YOUR_REPO_URL]  
#CUDA #PyTorch #GPU
