@echo off
REM Reproduce benchmarks and charts. Run from repo root.
cd /d "%~dp0\.."
echo === RTX 3050 GPU Lab - Reproduce all ===
echo.
echo 1. Matmul benchmark
python benchmarks/matmul_benchmark.py
echo.
echo 2. Conv benchmark
python benchmarks/conv_benchmark.py
echo.
echo 3. Transformer benchmark (light)
python benchmarks/transformer_benchmark.py
echo.
echo 4. Generate charts (skip MNIST by default)
python benchmarks/generate_charts.py --skip-mnist
echo.
echo 5. Roofline plot
python profiling/roofline_analysis/plot_roofline.py
echo.
echo Done. Charts in benchmarks/ and profiling/nsight_reports/
pause
