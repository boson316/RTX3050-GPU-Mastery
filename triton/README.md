# Triton — 3×3 Conv2d FP16

- **conv_triton.py** — JIT kernel, 1D grid, BLOCK_C=32; **~1.27x** vs torch on RTX 3050.

Requires `pip install triton`（Windows 請用 `triton-windows`）。與 `mnist_custom_conv.py` FP16 區塊一併執行可跑 benchmark。
