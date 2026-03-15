@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

:: Ninja is required (pip install ninja)
:: Always load Visual Studio x64 environment so NVCC can find cl.exe (required for CUDA JIT/pre-build)
set "VCVARS="
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if exist "C:\Program Files\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if defined VCVARS (
  call "!VCVARS!" >nul 2>&1
  echo [OK] Visual Studio x64 environment loaded.
) else (
  where cl.exe >nul 2>&1
  if errorlevel 1 (
    echo.
    echo ERROR: vcvars64.bat not found and cl.exe not in PATH.
    echo CUDA extension needs Visual Studio or Build Tools ^(Desktop development with C++^).
    echo Install path not detected. Either:
    echo   1. Open "x64 Native Tools Command Prompt for VS" from Start Menu, then run this script again.
    echo   2. Or install VS 2022/2019 and re-run this script from a normal CMD.
    echo.
    exit /b 1
  )
  echo [OK] Using existing cl.exe in PATH.
)

:: Required for PyTorch/setuptools when VC is already active (avoids "DISTUTILS_USE_SDK is not set")
set DISTUTILS_USE_SDK=1

:: Remove old extension .pyd if present (extension was renamed to _transformer_cuda_native so .py is not shadowed)
for %%f in (gpu_kernels\transformer\transformer_cuda.*.pyd) do del "%%f" 2>nul

:: Pre-build CUDA extension so we get a clear build error here (JIT often fails with same cl.exe issue)
echo Building transformer CUDA extension (first time may take a minute)...
pip install --no-build-isolation -e gpu_kernels\transformer
if errorlevel 1 (
  echo.
  echo Pre-build failed. If error was "Cannot find compiler cl.exe", run this script from
  echo "x64 Native Tools Command Prompt for VS" ^(Start Menu^).
  echo Continuing to run benchmark ^(CUDA kernels may show as "extension not built"^)...
  echo.
)

:: 預設為低負載（程式內建）。若要完整 benchmark 且含自訂 CUDA，請設：
::   set TRANSFORMER_BENCHMARK_FULL=1
::   set TRANSFORMER_BENCHMARK_CUDA=1

python benchmarks/transformer_benchmark.py
