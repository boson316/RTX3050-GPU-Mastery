# PyTorch CUDA Extension — Custom 3×3 Conv2d

16×16 output tile × 32 channels per block, 18×18 halo tile in shared memory, FP32 or FP16 storage with FP32 accumulation (`conv3x3_kernel.cuh`). The earlier "~1.5x vs cuDNN" figure compared an FP16-accumulating kernel against FP32 cuDNN and has been withdrawn, see [docs/ENGINEERING_LOG.md](../docs/ENGINEERING_LOG.md#10-the-custom-conv-15x-faster-than-cudnn-compared-fp16-math-with-fp32).

## Build

- **Requires**: PyTorch (CUDA), CUDA Toolkit 12.4, MSVC (Windows) or gcc (Linux).
- **Windows (x64 Native Tools)**:
  ```cmd
  set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;%PATH%
  set DISTUTILS_USE_SDK=1
  pip install --no-build-isolation .
  ```
- **Files**: `setup.py`, `conv_kernel.cu`（本目錄已含）。

## Run benchmark

```bash
python mnist_custom_conv.py
```

See [../docs/benchmarks.md](../docs/benchmarks.md).
