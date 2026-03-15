# Triton Matmul

Tiled matrix multiply `C = A @ B` using Triton. Uses `tl.dot` with configurable block sizes; supports FP16 and TF32.

## Run

From repo root (with `triton` or `triton-windows` installed):

```bash
python -m triton_kernels.matmul.matmul_triton
```

Or from `triton_kernels/matmul`: `python matmul_triton.py`

## Tuning

Block sizes `BLOCK_M`, `BLOCK_N`, `BLOCK_K` can be tuned per GPU (see `benchmarks/matmul_benchmark.py`).
