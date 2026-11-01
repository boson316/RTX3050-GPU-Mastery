@echo off
REM Verify, benchmark and plot everything. Run from repo root in an "x64 Native Tools Command Prompt for VS".
cd /d "%~dp0\.."
echo === 1. Correctness: Triton kernels ===
python -m pytest -q || exit /b 1
echo === 2. Correctness: CUDA kernel logic on the CPU emulator (MSVC) ===
call tests\cuda_emulation\run_all.bat || exit /b 1
echo === 3. Device peaks ===
python benchmarks\gpu_specs.py
echo === 4. CUDA roadmap (self-verifying binaries) ===
pushd cuda_roadmap && call build.bat && popd
python cuda_roadmap\run_benchmarks.py || exit /b 1
echo === 5. Verified benchmarks + report + charts ===
python tools\performance_dashboard.py --skip-mnist
echo === 6. Roofline ===
python profiling\roofline_analysis.py
echo Done: benchmarks\performance_report.md, benchmarks\plots\, profiling\roofline\, cuda_roadmap\results.md
