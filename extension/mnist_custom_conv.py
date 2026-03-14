"""
MNIST Custom CUDA Conv2d + Benchmark vs torch.nn.Conv2d
- custom_conv：3x3, in=1, out=32，FP16 + 16x16 tile（CUDA extension）
- conv_triton：Triton 重寫 FP16 conv，目標約 1.5x
"""
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if (_root / "triton").is_dir():
    sys.path.insert(0, str(_root / "triton"))

import torch
import torch.nn as nn
import time

try:
    import custom_conv
    HAS_CUSTOM = True
    _custom_path = getattr(custom_conv, "__file__", None)
except ImportError:
    HAS_CUSTOM = False
    _custom_path = None

try:
    from conv_triton import conv2d_triton_fp16
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

device = torch.device("cuda")
C_in, H, W = 1, 28, 28
C_out = 32


def run_benchmark(B, n_iters=100, use_fp16=False):
    torch_conv = nn.Conv2d(C_in, C_out, 3, padding=0).to(device)
    x = torch.randn(B, C_in, H, W, device=device)
    if use_fp16:
        torch_conv = torch_conv.half()
        x = x.half()

    def run_torch():
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = torch_conv(x)
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000 / n_iters

    def run_custom():
        w, b = torch_conv.weight, torch_conv.bias
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = custom_conv.custom_conv2d(x, w, b)[0]
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000 / n_iters

    with torch.no_grad():
        y_t = torch_conv(x)
        y_c = custom_conv.custom_conv2d(x, torch_conv.weight, torch_conv.bias)[0]
    err = (y_t.float() - y_c.float()).abs().max().item()
    return run_torch(), run_custom(), err


def run_benchmark_fp16_three(B, n_iters=100):
    """FP16：同時測 torch、extension、Triton，回傳 (t_torch, t_ext, t_triton, err_ext, err_triton)。"""
    torch_conv = nn.Conv2d(C_in, C_out, 3, padding=0).to(device).half()
    x = torch.randn(B, C_in, H, W, device=device, dtype=torch.float16)
    w, b = torch_conv.weight, torch_conv.bias

    def run_torch():
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = torch_conv(x)
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000 / n_iters

    def run_ext():
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = custom_conv.custom_conv2d(x, w, b)[0]
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000 / n_iters

    def run_triton():
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = conv2d_triton_fp16(x, w, b)
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000 / n_iters

    with torch.no_grad():
        y_t = torch_conv(x)
        y_c = custom_conv.custom_conv2d(x, w, b)[0]
        y_tr = conv2d_triton_fp16(x, w, b)
    err_ext = (y_t.float() - y_c.float()).abs().max().item()
    err_triton = (y_t.float() - y_tr.float()).abs().max().item()
    return run_torch(), run_ext(), run_triton(), err_ext, err_triton


if __name__ == "__main__":
    if not HAS_CUSTOM:
        print("請先編譯 extension: build_in_english_path.bat")
        exit(1)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Conv: 3x3, in_ch={C_in}, out_ch={C_out}, no padding")
    if _custom_path:
        print(f"Extension: {_custom_path}")
    print(f"Triton: {'OK' if HAS_TRITON else '未安裝 triton'}\n")

    n = 100
    tol_fp32, tol_fp16 = 5e-3, 1e-2

    print("--- FP32（介面 float，kernel 內 FP16 計算）---")
    for B in [1024, 512, 128]:
        t_torch, t_custom, err = run_benchmark(B, n, use_fp16=False)
        ok = " (OK)" if err < tol_fp32 else " (check!)"
        print(f"  B={B} max_diff={err:.6f}{ok}  torch={t_torch:.4f} ms  ext={t_custom:.4f} ms  speedup={t_torch/max(t_custom,1e-9):.2f}x")
    print()

    print("--- FP16：torch vs extension vs Triton（目標 1.5x）---")
    if not HAS_TRITON:
        print("  [略過] 請 pip install triton 後再跑")
    else:
        try:
            for B in [1024, 512, 128]:
                t_torch, t_ext, t_tr, err_ext, err_tr = run_benchmark_fp16_three(B, n)
                ok_ext = " (OK)" if err_ext < tol_fp16 else " (check!)"
                ok_tr = " (OK)" if err_tr < tol_fp16 else " (check!)"
                sp_ext = t_torch / max(t_ext, 1e-9)
                sp_tr = t_torch / max(t_tr, 1e-9)
                print(f"  B={B}  torch={t_torch:.4f} ms  ext={t_ext:.4f} ms ({sp_ext:.2f}x)  triton={t_tr:.4f} ms ({sp_tr:.2f}x)")
                print(f"         max_diff: ext={err_ext:.4f}{ok_ext}  triton={err_tr:.4f}{ok_tr}")
        except RuntimeError as e:
            if "Half" in str(e) or "not implemented" in str(e):
                print("  [略過] extension 未含 FP16，請重新執行 build_in_english_path.bat")
            else:
                raise
