"""
Matrix multiply benchmark: PyTorch (torch.matmul / cuBLAS) vs Triton matmul.
Run from repo root. GPU required for meaningful results.
"""
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import torch

def _bench(fn, *args, warmup=10, repeat=50):
    for _ in range(warmup):
        fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repeat


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtypes = [torch.float32]
    if torch.cuda.is_available():
        dtypes.append(torch.float16)

    for N in [512, 1024, 2048]:
        M, K = N, N
        for dtype in dtypes:
            A = torch.randn(M, K, device=device, dtype=dtype)
            B = torch.randn(K, N, device=device, dtype=dtype)
            t_torch = _bench(lambda: torch.matmul(A, B))
            gflops = 2 * M * N * K / 1e9
            print(f"matmul_benchmark | {M}x{K} @ {K}x{N} {dtype} | torch: {t_torch:.4f} ms ({gflops/t_torch*1e3:.1f} GFLOPS)")

        if torch.cuda.is_available():
            try:
                from triton_kernels.matmul import matmul_triton
                A = torch.randn(M, K, device=device, dtype=torch.float16)
                B = torch.randn(K, N, device=device, dtype=torch.float16)
                t_triton = _bench(matmul_triton, A, B)
                print(f"matmul_benchmark | {M}x{K} @ {K}x{N} fp16 | triton: {t_triton:.4f} ms ({gflops/t_triton*1e3:.1f} GFLOPS)")
            except Exception as e:
                print(f"matmul_benchmark | triton skipped: {e}")

    print("matmul_benchmark done.")


if __name__ == "__main__":
    main()
