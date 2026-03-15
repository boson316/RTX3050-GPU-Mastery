"""
Attention benchmark: PyTorch SDPA vs Triton Flash Attention.
Run from repo root. GPU required.
"""
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import torch


def _bench(fn, warmup=5, repeat=30):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repeat


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = [
        (2, 8, 128, 64),
        (4, 8, 256, 64),
        (2, 8, 512, 64),
    ]
    for B, H, S, D in configs:
        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        def run_sdpa():
            torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False
            )

        t_sdpa = _bench(run_sdpa)
        print(f"attention_benchmark | B={B} H={H} S={S} D={D} | torch SDPA: {t_sdpa:.4f} ms")

        try:
            from triton_kernels.flash_attention import flash_attention_triton
            def run_flash():
                flash_attention_triton(q, k, v, causal=False)
            t_flash = _bench(run_flash)
            print(f"attention_benchmark | B={B} H={H} S={S} D={D} | triton flash: {t_flash:.4f} ms (speedup: {t_sdpa/max(t_flash,1e-9):.2f}x)")
        except Exception as e:
            print(f"attention_benchmark | triton flash skipped: {e}")

    print("attention_benchmark done.")


if __name__ == "__main__":
    main()
