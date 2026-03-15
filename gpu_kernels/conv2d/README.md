# Conv2D (3×3)

Standalone CUDA 3×3 convolution, no padding. Shared-memory tiling for input and weights.

## Build

```bash
nvcc -o conv2d conv2d.cu
```

## Design

- One block per (tile_h, tile_w, output_channel); each block produces TILE×TILE outputs for one channel.
- Input tile loaded into `__shared__` with halo (SHARD_H × SHARD_W = 18×18 for 16×16 output tile).
- Weight 3×3 per channel in shared memory; reduces global traffic for repeated filter use.
