# LinkedIn post (English)

Copy everything between the lines below into LinkedIn. Attach 1–2 charts from the repo (see **Images**).

Suggested images:
- `benchmarks/plots/matmul_latency_by_size.png`
- `profiling/nsight_reports/roofline_model.png`
- or `benchmarks/plots/transformer_latency_by_kernel.png`

---

Built an open-source GPU Optimization Lab on a laptop RTX 3050 (Ampere sm_86, 6GB) — not a datacenter GPU, on purpose.

What I did

• CUDA L1–L4: naive kernels → shared-memory tiled GEMM → FP16 WMMA Tensor Cores
• Triton: matmul, conv, layernorm, softmax, FlashAttention (vs PyTorch/cuBLAS baselines)
• FlashAttention: PyTorch reference + CUDA + Triton (tiling + online softmax, O(S) extra memory)
• Transformer blocks: QKV, softmax, LayerNorm, GELU, MLP — per-kernel latency & bandwidth
• Profiling loop: Nsight Systems/Compute + Roofline (ridge ≈ 23 FLOP/Byte on this card)

Numbers (RTX 3050 Laptop)

• 1024² GEMM (tiled CUDA): ~521× vs CPU
• FP16 matmul 1024³ (Triton): ~14 TFLOPS
• 1M-element reduction (shared mem): ~0.76 ms

What I learned (honest part)

Custom CUDA doesn’t always beat cuBLAS — on some Transformer ops (e.g. QKV/MLP), my kernels were slower than PyTorch. That’s why I benchmark before optimizing, and use profiling to see if a kernel is memory-bound or compute-bound instead of guessing.

📎 Charts in repo: matmul latency, transformer per-kernel breakdown, roofline plot.

🔗 https://github.com/boson316/RTX3050-GPU-Mastery

If you work on ML systems / GPU kernels, I’d love feedback or a conversation. Stars welcome ⭐

#CUDA #Triton #GPU #MLSystems #HPC #FlashAttention #PyTorch

---
