# Environment (Phase 0)

由指令實際輸出產生，未手動編輯數值。

## Versions
```
$ python -c "import torch, triton; ..."
3.11.0 2.6.0+cu124 12.4 NVIDIA GeForce RTX 3050 6GB Laptop GPU 3.2.0
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Feb_27_16:28:36_Pacific_Standard_Time_2024
Cuda compilation tools, release 12.4, V12.4.99
Build cuda_12.4.r12.4/compiler.33961263_0
$ cl (via results/v2/vsenv.bat)
C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe
Microsoft (R) C/C++ Optimizing Compiler Version 19.50.35726 for x64
$ nvidia-smi --query-gpu=...
name, driver_version, vbios_version, power.max_limit [W], clocks.max.sm [MHz], clocks.max.memory [MHz]
NVIDIA GeForce RTX 3050 6GB Laptop GPU, 610.62, 94.07.82.00.37, 45.00 W, 2100 MHz, 5501 MHz
```

## python benchmarks/gpu_specs.py
```
NVIDIA GeForce RTX 3050 6GB Laptop GPU: 20 SMs, sm_86, SM clock 2100.0 MHz, mem clock 5501.0 MHz, bus 96-bit, L2 1536 KB
  peak fp32             : 10.8 TFLOPS
  peak fp16_tc_fp32acc  : 21.5 TFLOPS
  peak fp16_tc_fp16acc  : 43.0 TFLOPS
  peak DRAM bandwidth : 132 GB/s
  ridge point (FP16 TC, FP32 acc): 163 FLOP/byte
```

## 與任務規格的差異

- **編譯器：** 本機是 Visual Studio Community 2026（MSVC 14.50），沒有 VS 2022。CUDA 12.4 的 `host_config.h` 只接受 `_MSC_VER < 1950`，因此 nvcc 需要 `-allow-unsupported-compiler`。
- **torch 版本：** `pip install -r requirements.txt` 的 `torchvision` 未鎖版本，會把 `torch 2.6.0+cu124` 換成 PyPI 的 CPU 版 `2.15.0`。已改為從 cu124 index 明確安裝 `torch==2.6.0+cu124`、`torchvision==0.21.0+cu124`。
- **Triton 版本：** 依 triton-windows README 對照表（PyTorch 2.6 對應 Triton 3.2）鎖定 `triton-windows==3.2.0.post21`。完整清單見 `requirements-freeze.txt`。
- **峰值假設：** `SM clock 2100 MHz` 是 NVML 回報的最大時脈，不是持續負載時脈；實際負載時脈於 Phase 2 記錄。
