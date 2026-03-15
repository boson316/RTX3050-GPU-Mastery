# Vector Add

Element-wise addition `C = A + B`. Baseline for GPU memory bandwidth and launch overhead.

## Build

```bash
nvcc vector_add.cu -o vector_add
```

## Run

```bash
./vector_add   # Linux
vector_add.exe # Windows
```

## Memory hierarchy

- **Global memory**: coalesced reads/writes (contiguous indices per warp).
- **No shared memory**: bandwidth-bound; useful for roofline baseline.
