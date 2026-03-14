# PyTorch CUDA Extension — Custom 3×3 Conv2d

FP16 kernel, 16×16 tile, `tile_in[18][18]`, `tile_w[32][9]`, sm_86. **~1.5x** vs `torch.nn.Conv2d` on RTX 3050.

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
