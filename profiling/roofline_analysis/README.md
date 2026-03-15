# Roofline Analysis

Roofline model: performance is bounded by either **memory bandwidth** or **peak FLOPS**, depending on arithmetic intensity (FLOPs/byte).

## Gathering data

1. **Peak FLOPS** and **peak memory bandwidth**: from GPU spec (e.g. RTX 3050 Laptop: ~4.5 TFLOPS FP32, ~192 GB/s).
2. **Kernel FLOPS and bytes**: from Nsight Compute or manual calculation.
3. **Arithmetic intensity** = FLOPS / (bytes read + written).

If intensity is below the **ridge point** (peak_FLOPS / peak_BW), the kernel is memory-bound; above, compute-bound.

## Scripts

- Vector add: memory-bound baseline (low intensity).
- Matrix multiply (large N): compute-bound with tiling (high intensity).
- See `docs/optimization_guide.md` for formulas and plots.
