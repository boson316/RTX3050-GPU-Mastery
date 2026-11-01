@echo off
REM Verify, benchmark and plot everything. Run from repo root in an "x64 Native Tools Command Prompt for VS".
cd /d "%~dp0\.."
echo === 1. Correctness: Triton kernels ===
python -m pytest -q || exit /b 1
echo === 2. Device peaks ===
python benchmarks\gpu_specs.py
echo === 3. CUDA roadmap (self-verifying binaries) ===
pushd cuda_roadmap && call build.bat && popd
python cuda_roadmap\run_benchmarks.py || exit /b 1
echo === 4. Verified benchmarks + report + charts ===
python tools\performance_dashboard.py --skip-mnist
echo === 5. Roofline ===
python profiling\roofline_analysis.py
echo Done. (CPU emulator tests need g++ / WSL: tests\cuda_emulation\run_all.sh)
