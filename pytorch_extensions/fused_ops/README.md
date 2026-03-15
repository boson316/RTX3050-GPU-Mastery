# Fused Ops (PyTorch CUDA Extension)

Custom fused operators built with PyTorch's `torch.utils.cpp_extension.CUDAExtension`.

The **canonical implementation** for this repo is in the top-level **`extension/`** directory:

- `extension/conv_kernel.cu` — 3×3 Conv2D FP16, 16×16 tile, `__half2`
- `extension/setup.py` — build with `pip install --no-build-isolation .`
- `extension/mnist_custom_conv.py` — benchmark and MNIST integration

## Build from extension/

```bash
cd extension
pip install --no-build-isolation .
```

Then `import custom_conv` and use `custom_conv.custom_conv2d(input, weight, bias)`.

This folder documents the layout for ML infrastructure roles; the actual extension source lives in **`extension/`** to keep a single build and CI path.
