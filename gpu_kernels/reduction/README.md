# Reduction

Parallel sum (or max/min) over a large array. Demonstrates shared-memory tree reduction and block-level parallelism.

## Build

```bash
nvcc reduction.cu -o reduction
```

## Optimization notes

- **Shared memory**: per-block partial sums; reduces global memory traffic.
- **Tree reduction**: O(log BLOCK_SIZE) steps within block.
- **Warp-level**: could add `__shfl_down_sync` for last warp to reduce shared memory use (see optimization_guide.md).
