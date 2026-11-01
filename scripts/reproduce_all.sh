#!/usr/bin/env bash
# Verify, benchmark and plot everything. Run from repo root on a CUDA machine.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "=== 1. Correctness: Triton kernels (compiled on GPU) ==="
python -m pytest -q
echo "=== 2. Correctness: CUDA kernel logic on the CPU emulator ==="
tests/cuda_emulation/run_all.sh
echo "=== 3. Device peaks ==="
python benchmarks/gpu_specs.py
echo "=== 4. CUDA roadmap (self-verifying binaries) ==="
(cd cuda_roadmap && ./build.sh) && python cuda_roadmap/run_benchmarks.py
echo "=== 5. Verified benchmarks + report + charts ==="
python tools/performance_dashboard.py --skip-mnist
echo "=== 6. Roofline ==="
python profiling/roofline_analysis.py
echo "Done: benchmarks/performance_report.md, benchmarks/plots/, profiling/roofline/, cuda_roadmap/results.md"
