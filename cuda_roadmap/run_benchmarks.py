"""
Run all CUDA roadmap benchmarks and compare with CPU / PyTorch.
Execute from repo root: python cuda_roadmap/run_benchmarks.py
Parses output lines: CUDA_NAIVE_MS=..., CUDA_OPTIMIZED_MS=..., CPU_MS=..., PYTORCH_MS=...
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

def run_exe(path: Path) -> dict:
    exe = path
    if not exe.is_file():
        exe = path.with_suffix(".exe")
    if not exe.is_file():
        return {"error": f"not built: {path.name}"}
    try:
        out = subprocess.run(
            [str(exe)],
            cwd=exe.parent,
            capture_output=True,
            text=True,
            timeout=60,
        )
        text = (out.stdout or "") + (out.stderr or "")
    except Exception as e:
        return {"error": str(e)}
    result = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if "=" in line:
            k, v = line.split("=", 1)
            try:
                result[k.strip()] = float(v.strip())
            except ValueError:
                result[k.strip()] = v
    return result

def bench_pytorch_vector_add(n=1 << 20):
    import torch
    if not torch.cuda.is_available():
        return None
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        c = a + b
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / 20

def bench_cpu_vector_add(n=1 << 20):
    start = time.perf_counter()
    a = [float(i) for i in range(n)]
    b = [float(2*i) for i in range(n)]
    c = [a[i] + b[i] for i in range(n)]
    return (time.perf_counter() - start) * 1000

def bench_pytorch_reduction(n=1 << 20):
    import torch
    if not torch.cuda.is_available():
        return None
    x = torch.ones(n, device="cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        s = x.sum()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / 20

def bench_cpu_reduction(n=1 << 20):
    start = time.perf_counter()
    s = sum(1.0 for _ in range(n))
    return (time.perf_counter() - start) * 1000

def bench_pytorch_matmul(n=1024):
    import torch
    if not torch.cuda.is_available():
        return None
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(5):
        c = torch.mm(a, b)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / 5

def main():
    print("=" * 60)
    print("CUDA Roadmap Benchmarks (vs CPU / PyTorch)")
    print("=" * 60)

    # Level 1: vector add
    print("\n--- Level 1: Vector Add (N=1M) ---")
    r = run_exe(ROOT / "level1_basics/vector_add/vector_add_bench")
    if "error" in r:
        print("  CUDA:", r["error"])
    else:
        print(f"  CUDA naive:    {r.get('CUDA_NAIVE_MS', '?'):.4f} ms")
        print(f"  CUDA optimized:{r.get('CUDA_OPTIMIZED_MS', '?'):.4f} ms")
    try:
        t = bench_cpu_vector_add()
        print(f"  CPU (Python): {t:.4f} ms")
    except Exception as e:
        print(f"  CPU: {e}")
    try:
        t = bench_pytorch_vector_add()
        if t is not None:
            print(f"  PyTorch:       {t:.4f} ms")
    except Exception as e:
        print(f"  PyTorch: {e}")

    # Level 1: reduction
    print("\n--- Level 1: Reduction (N=1M) ---")
    r = run_exe(ROOT / "level1_basics/reduction/reduction_bench")
    if "error" in r:
        print("  CUDA:", r["error"])
    else:
        print(f"  CUDA naive:    {r.get('CUDA_NAIVE_MS', '?'):.4f} ms")
        print(f"  CUDA optimized:{r.get('CUDA_OPTIMIZED_MS', '?'):.4f} ms")
    try:
        t = bench_cpu_reduction()
        print(f"  CPU (Python):  {t:.4f} ms")
    except Exception as e:
        print(f"  CPU: {e}")
    try:
        t = bench_pytorch_reduction()
        if t is not None:
            print(f"  PyTorch:       {t:.4f} ms")
    except Exception as e:
        print(f"  PyTorch: {e}")

    # Level 1: naive matmul
    print("\n--- Level 1: Naive Matrix Multiply (N=1024) ---")
    r = run_exe(ROOT / "level1_basics/naive_matmul/naive_matmul_bench")
    if "error" in r:
        print("  CUDA:", r["error"])
    else:
        print(f"  CPU:           {r.get('CPU_MS', '?'):.4f} ms")
        print(f"  CUDA naive:    {r.get('CUDA_NAIVE_MS', '?'):.4f} ms")
    try:
        t = bench_pytorch_matmul()
        if t is not None:
            print(f"  PyTorch:       {t:.4f} ms")
    except Exception as e:
        print(f"  PyTorch: {e}")

    # Level 2: tiled matmul
    print("\n--- Level 2: Tiled Matrix Multiply (N=1024) ---")
    r = run_exe(ROOT / "level2_memory/tiled_matmul/tiled_matmul_bench")
    if "error" in r:
        print("  CUDA:", r["error"])
    else:
        print(f"  CPU:           {r.get('CPU_MS', '?'):.4f} ms")
        print(f"  CUDA naive:    {r.get('CUDA_NAIVE_MS', '?'):.4f} ms")
        print(f"  CUDA optimized:{r.get('CUDA_OPTIMIZED_MS', '?'):.4f} ms")

    # Level 2: coalescing
    print("\n--- Level 2: Memory Coalescing ---")
    r = run_exe(ROOT / "level2_memory/coalescing/coalescing_bench")
    if "error" in r:
        print("  CUDA:", r["error"])
    else:
        print(f"  CUDA (strided):   {r.get('CUDA_NAIVE_MS', '?'):.4f} ms")
        print(f"  CUDA (coalesced): {r.get('CUDA_OPTIMIZED_MS', '?'):.4f} ms")

    # Level 2: bank conflict
    print("\n--- Level 2: Bank Conflict (reduction) ---")
    r = run_exe(ROOT / "level2_memory/bank_conflict/bank_conflict_bench")
    if "error" in r:
        print("  CUDA:", r["error"])
    else:
        print(f"  CUDA (conflict): {r.get('CUDA_NAIVE_MS', '?'):.4f} ms")
        print(f"  CUDA (no conf):  {r.get('CUDA_OPTIMIZED_MS', '?'):.4f} ms")

    # Level 3
    print("\n--- Level 3: Warp Shuffle Reduction ---")
    r = run_exe(ROOT / "level3_advanced/warp_shuffle_reduction/warp_shuffle_bench")
    if "error" in r:
        print("  CUDA:", r["error"])
    else:
        print(f"  CUDA (shared):  {r.get('CUDA_NAIVE_MS', '?'):.4f} ms")
        print(f"  CUDA (shuffle):{r.get('CUDA_OPTIMIZED_MS', '?'):.4f} ms")

    print("\n--- Level 3: Fused Add+ReLU ---")
    r = run_exe(ROOT / "level3_advanced/fused_ops/fused_ops_bench")
    if "error" in r:
        print("  CUDA:", r["error"])
    else:
        print(f"  CUDA (2 kernels): {r.get('CUDA_NAIVE_MS', '?'):.4f} ms")
        print(f"  CUDA (fused):     {r.get('CUDA_OPTIMIZED_MS', '?'):.4f} ms")

    print("\n--- Level 3: Persistent Kernel (vector add) ---")
    r = run_exe(ROOT / "level3_advanced/persistent_kernel/persistent_kernel_bench")
    if "error" in r:
        print("  CUDA:", r["error"])
    else:
        print(f"  CUDA (normal):   {r.get('CUDA_NAIVE_MS', '?'):.4f} ms")
        print(f"  CUDA (persistent):{r.get('CUDA_OPTIMIZED_MS', '?'):.4f} ms")

    # Level 4
    print("\n--- Level 4: FP16 Tensor Core Matmul (1024x1024) ---")
    r = run_exe(ROOT / "level4_tensor_core/fp16_tensor_core_matmul/fp16_matmul_bench")
    if "error" in r:
        print("  CUDA:", r["error"])
    else:
        print(f"  CUDA (FP16 naive):  {r.get('CUDA_NAIVE_MS', '?'):.4f} ms")
        print(f"  CUDA (WMMA/TC):     {r.get('CUDA_OPTIMIZED_MS', '?'):.4f} ms")

    print("\n" + "=" * 60)
    print("Done. See docs/level1_kernels.md etc. for explanations.")
    print("=" * 60)

if __name__ == "__main__":
    main()
