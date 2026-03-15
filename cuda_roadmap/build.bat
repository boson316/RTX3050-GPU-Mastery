@echo off
setlocal enabledelayedexpansion

:: 切換到本 script 所在目錄
cd /d "%~dp0"

:: 若找不到 cl.exe，先載入 Visual Studio 的 x64 編譯環境（在切到短路徑之前載入，確保 vcvars 正常執行）
where cl.exe >nul 2>&1
if errorlevel 1 (
  echo cl.exe not in PATH. Loading Visual Studio x64 environment...
  set "VCVARS="
  if exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
  if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
  if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
  if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
  if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  if exist "C:\Program Files\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
  if not defined VCVARS (
    echo.
    echo ERROR: Could not find vcvars64.bat. Please do ONE of:
    echo   1. Open "x64 Native Tools Command Prompt for VS" from Start Menu, then run build.bat again.
    echo   2. Install Visual Studio with "Desktop development with C++" and run this from that prompt.
    echo.
    exit /b 1
  )
  call "!VCVARS!"
  where cl.exe >nul 2>&1
  if errorlevel 1 (
    echo.
    echo ERROR: After loading VS environment, cl.exe still not in PATH.
    echo Please run build.bat from "x64 Native Tools Command Prompt for VS" from Start Menu.
    exit /b 1
  )
  echo cl.exe found. Proceeding to build...
)

:: 改用 8.3 短路徑，避免路徑中的中文導致 cl.exe 報錯
for %%I in (.) do set "SHORT_CD=%%~sI"
if defined SHORT_CD (
  cd /d "!SHORT_CD!"
  echo Building in short path: !CD!
)

set NVCC=nvcc
set ARCH=sm_86
if defined CUDA_ARCH set ARCH=%CUDA_ARCH%

call "%NVCC%" -arch=%ARCH% -o level1_basics\vector_add\vector_add_bench level1_basics\vector_add\vector_add_bench.cu -allow-unsupported-compiler
call "%NVCC%" -arch=%ARCH% -o level1_basics\reduction\reduction_bench level1_basics\reduction\reduction_bench.cu -allow-unsupported-compiler
call "%NVCC%" -arch=%ARCH% -o level1_basics\naive_matmul\naive_matmul_bench level1_basics\naive_matmul\naive_matmul_bench.cu -allow-unsupported-compiler
call "%NVCC%" -arch=%ARCH% -o level2_memory\tiled_matmul\tiled_matmul_bench level2_memory\tiled_matmul\tiled_matmul_bench.cu -allow-unsupported-compiler
call "%NVCC%" -arch=%ARCH% -o level2_memory\coalescing\coalescing_bench level2_memory\coalescing\coalescing_bench.cu -allow-unsupported-compiler
call "%NVCC%" -arch=%ARCH% -o level2_memory\bank_conflict\bank_conflict_bench level2_memory\bank_conflict\bank_conflict_bench.cu -allow-unsupported-compiler
call "%NVCC%" -arch=%ARCH% -o level3_advanced\warp_shuffle_reduction\warp_shuffle_bench level3_advanced\warp_shuffle_reduction\warp_shuffle_bench.cu -allow-unsupported-compiler
call "%NVCC%" -arch=%ARCH% -o level3_advanced\fused_ops\fused_ops_bench level3_advanced\fused_ops\fused_ops_bench.cu -allow-unsupported-compiler
call "%NVCC%" -arch=%ARCH% -o level3_advanced\persistent_kernel\persistent_kernel_bench level3_advanced\persistent_kernel\persistent_kernel_bench.cu -allow-unsupported-compiler
call "%NVCC%" -arch=%ARCH% -o level4_tensor_core\fp16_tensor_core_matmul\fp16_matmul_bench level4_tensor_core\fp16_tensor_core_matmul\fp16_matmul_bench.cu -allow-unsupported-compiler
call "%NVCC%" -arch=%ARCH% -o level4_tensor_core\wmma_example\wmma_example level4_tensor_core\wmma_example\wmma_example.cu -allow-unsupported-compiler
echo Done. Run: python run_benchmarks.py
