# PyTorch Extensions

| Subdir           | Description |
|------------------|-------------|
| fused_ops        | Points to top-level `extension/` (custom conv2d CUDA extension). |
| custom_attention| Python wrapper: SDPA + optional Triton Flash; for benchmarking.   |

Building the CUDA extension: use **`extension/`** at repo root (`pip install --no-build-isolation .`).
