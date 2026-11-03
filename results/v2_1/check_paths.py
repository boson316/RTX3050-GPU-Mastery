"""Confirm which kernel each call runs, without the profiler (it records no kernel names on this Windows setup).

Copies gpu_kernels/transformer to a temporary folder, adds one stderr trace line per branch to the COPY,
builds it, and calls it with the benchmark's shapes. The repo's sources are not touched.
Run with CUBLASLT_LOG_LEVEL=5 to also get cuBLASLt's own record of each matmul.
Usage: python results/v2_1/check_paths.py <temp_dir>
"""
from __future__ import annotations

import importlib.util
import re
import shutil
import sys
from pathlib import Path

import torch

src = Path("gpu_kernels/transformer")
dst = Path(sys.argv[1]) / "transformer_trace"
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(src, dst, ignore=shutil.ignore_patterns("build", "__pycache__"))

cu = dst / "transformer_kernels.cu"
s = cu.read_text(encoding="utf-8")
s = '#include <cstdio>\n' + s
for fn, cond in (("softmax", "use_warp_path(n_cols, {x, y})"), ("layernorm", "use_warp_path(n_cols, {x, gamma, beta, y})")):
    old = f"    if ({cond}) {{\n" if f"    if ({cond}) {{\n" in s else f"    if ({cond}) {{\r\n"
    assert s.count(old) == 1, cond
    s = s.replace(old, old + f'        fprintf(stderr, "TRACE {fn}: warp path, n_cols=%d\\n", n_cols);\n')
    pat = re.compile(rf"(\n)(    tk::{fn}_rows_kernel<<<)")
    s, n = pat.subn(rf'\1    fprintf(stderr, "TRACE {fn}: fallback rows kernel, n_cols=%d\\n", n_cols);\n\2', s)
    assert n == 1, fn
s, n = re.subn(r"(cudaError_t bias_act_fp16_launch\([^)]*\)\s*\{)", r'\1\n    fprintf(stderr, "TRACE bias_act_kernel launched\\n");', s)
assert n >= 1
cu.write_text(s, encoding="utf-8")

spec = importlib.util.spec_from_file_location("transformer_cuda_trace", dst / "transformer_cuda.py")
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)


def rand(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.half)


H = 768
for M in (2048, 8192):
    print(f"--- softmax / LayerNorm {M}x{H}", file=sys.stderr, flush=True)
    x, g, b = rand(M, H), rand(H), rand(H)
    tc.softmax_cuda(x)
    tc.layernorm_cuda(x, g, b)
print("--- softmax / LayerNorm 2048x1001 (must fall back)", file=sys.stderr, flush=True)
x, g, b = rand(2048, 1001), rand(1001), rand(1001)
tc.softmax_cuda(x)
tc.layernorm_cuda(x, g, b)
print("--- QKV with bias", file=sys.stderr, flush=True)
x = rand(2048, H)
print(f"QKV out {tc.fused_qkv_cuda(x, rand(H, 3 * H), rand(3 * H)).shape}", flush=True)
print("--- MLP", file=sys.stderr, flush=True)
print(f"MLP out {tc.fused_mlp_cuda(x, rand(H, 4 * H), rand(4 * H), rand(4 * H, H), rand(H)).shape}", flush=True)
torch.cuda.synchronize()
print("--- done", file=sys.stderr, flush=True)
