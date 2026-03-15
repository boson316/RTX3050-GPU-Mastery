@echo off
REM Nsight profiling: timeline + per-kernel occupancy, memory BW, warp divergence.
REM Run from repo root. Do not run lines starting with # in CMD (those are comments).
echo Running Nsight profiling...
python profiling/run_nsight_profiling.py
echo.
echo To generate roofline plot, run:
echo   python profiling/roofline_analysis/plot_roofline.py
pause
