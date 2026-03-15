# Matrix Multiply

C = A × B (N×N). Demonstrates naive global-memory vs tiled shared-memory kernel.

## Build

```bash
nvcc matrix_mul.cu -o matrix_mul
```

## Tiled kernel

- Each block computes a TILE×TILE block of C.
- Load A and B tiles into `__shared__`; each thread cooperates to load; then compute dot products from shared memory.
- Reduces global memory reads from O(N) per output element to O(N/TILE).

## Roofline

Matrix multiply is compute-bound for large N; shared-memory tiling improves arithmetic intensity and approaches peak FLOPS.
