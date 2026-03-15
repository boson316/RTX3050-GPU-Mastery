# Triton Conv (3×3)

FP16 3×3 convolution with no padding. One program per output tile and output-channel block.

## Usage

```python
from triton_kernels.conv.conv_triton import conv2d_triton_fp16
y = conv2d_triton_fp16(x, w, b)  # x (B,1,H,W), w (C_out,1,3,3), b (C_out,)
```

Benchmark: `python benchmarks/conv_benchmark.py` (imports from this module or legacy `triton/`).
