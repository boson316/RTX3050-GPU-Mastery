"""
Long-sequence attention (B=1, H=8, D=64; S = 1024, 2048, 4096) with the precision of every
implementation made explicit.

Why this script exists: v2 compared everything in FP32 and found Triton 4x faster than SDPA with
~1000x larger error. That was not a bug - in FP32, Triton's tl.dot defaults to TF32 Tensor Cores
(10-bit mantissa) while SDPA (memory-efficient backend) and our CUDA kernel use IEEE FP32 on CUDA
cores. Comparing them as one column was apples to oranges. This script reports:

  FP16  : SDPA (FlashAttention-2 backend)  vs  Triton FP16                       <- what models use
  FP32  : SDPA (IEEE)  vs  Triton IEEE  vs  our CUDA kernel (IEEE)                <- fair FP32 race
  TF32  : Triton with TF32 Tensor Cores, labelled as such, error reported         <- for reference

Each implementation is verified against an FP64 reference (computed head by head so S=4096 fits
in 6 GB) before it is timed; the max absolute error is recorded next to the latency.

Run from repo root:  python benchmarks/benchmark_attention_longseq.py [run_id]
Output: benchmarks/benchmark_attention_longseq_results.csv (or results/v2_1/csv/run<id>/ with run_id)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import torch
import torch.nn.functional as F

from benchmarks.bench_utils import BENCHMARKS_DIR, BenchmarkError, check_plausible, compute_metrics, run_bench

B, H, D = 1, 8, 64
SEQ_LENS = (1024, 2048, 4096)
# max |error| vs the FP64 reference allowed before timing, per precision
TOL = {"fp16": 2e-2, "fp32": 1e-4, "tf32": 5e-3}


def reference_fp64(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    out = torch.empty(q.shape, dtype=torch.float64, device=q.device)
    for h in range(q.shape[1]):  # one head at a time: S x S in FP64 is 134 MB at S=4096
        qh, kh, vh = (t[:, h].double() for t in (q, k, v))
        p = torch.softmax(qh @ kh.transpose(-1, -2) / D**0.5, dim=-1)
        out[:, h] = p @ vh
    return out


def implementations():
    impls = [
        ("fp16", torch.float16, "SDPA", lambda q, k, v: F.scaled_dot_product_attention(q, k, v)),
        ("fp32", torch.float32, "SDPA", lambda q, k, v: F.scaled_dot_product_attention(q, k, v)),
    ]
    try:
        from triton_kernels.flash_attention import flash_attention_triton

        impls += [
            ("fp16", torch.float16, "Triton", lambda q, k, v: flash_attention_triton(q, k, v)),
            ("fp32", torch.float32, "Triton", lambda q, k, v: flash_attention_triton(q, k, v, precision="ieee")),
            ("tf32", torch.float32, "Triton", lambda q, k, v: flash_attention_triton(q, k, v, precision="tf32")),
        ]
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] Triton unavailable: {e}", file=sys.stderr)
    try:
        from flash_attention_simple.attention_cuda import attention_cuda

        impls.append(("fp32", torch.float32, "CUDA", lambda q, k, v: attention_cuda(q, k, v)))
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] CUDA extension unavailable: {e}", file=sys.stderr)
    return impls


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available; run on a GPU.")
        return
    run_id = sys.argv[1] if len(sys.argv) > 1 else None
    out_dir = root / "results" / "v2_1" / "csv" / f"run{run_id}" if run_id else BENCHMARKS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    rows = []
    for S in SEQ_LENS:
        base = [torch.randn(B, H, S, D, device="cuda") for _ in range(3)]
        ref = reference_fp64(*base)
        flops = 4.0 * B * H * S * S * D
        for precision, dtype, name, fn in implementations():
            q, k, v = (t.to(dtype) for t in base)
            row = {"S": S, "precision": precision, "impl": name}
            try:
                out = fn(q, k, v)
                if out is None:
                    raise BenchmarkError(f"{name}: returned None (extension not built?)")
                err = (out.double() - ref).abs().max().item()
                row["max_abs_err_vs_fp64"] = f"{err:.3e}"
                if not err <= TOL[precision]:
                    raise BenchmarkError(f"{name} {precision}: max error {err:.3e} > {TOL[precision]} - not timed")
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before = torch.cuda.memory_allocated()
                fn(q, k, v)
                torch.cuda.synchronize()
                row["peak_extra_mib"] = round((torch.cuda.max_memory_allocated() - before) / 2**20, 1)
                lat = run_bench(lambda: fn(q, k, v), warmup=5, repeat=20)
                m = compute_metrics(lat, 4 * q.numel() * q.element_size(), 0, flops)
                check_plausible(f"{name} {precision} S={S}", m)
                row.update(verified="PASS", latency_ms=round(lat, 4), tflops=round(m["gflops"] / 1e3, 2))
            except BenchmarkError as e:
                row["verified"] = f"FAIL: {str(e)[:150]}"
            except torch.cuda.OutOfMemoryError:
                row["verified"] = "OOM"
                torch.cuda.empty_cache()
            rows.append(row)
            print(row, flush=True)
    fields = ["S", "precision", "impl", "verified", "latency_ms", "tflops", "peak_extra_mib", "max_abs_err_vs_fp64"]
    path = out_dir / "benchmark_attention_longseq_results.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"CSV: {path}")


if __name__ == "__main__":
    main()
