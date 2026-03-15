@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

:: Load Visual Studio x64 environment if cl.exe not in PATH (nvcc needs it)
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
    echo ERROR: Could not find vcvars64.bat. Open "x64 Native Tools Command Prompt for VS" and run build.bat again.
    exit /b 1
  )
  call "!VCVARS!"
  where cl.exe >nul 2>&1
  if errorlevel 1 (
    echo ERROR: cl.exe still not in PATH. Use "x64 Native Tools Command Prompt for VS".
    exit /b 1
  )
  echo cl.exe found.
)

set ARCH=sm_86
if defined CUDA_ARCH set ARCH=%CUDA_ARCH%

echo Building flash_attention_cuda_standalone.exe ...
nvcc -arch=%ARCH% -o flash_attention_cuda_standalone flash_attention_cuda_standalone.cu -O3 -allow-unsupported-compiler
if errorlevel 1 (
  echo Build failed.
  exit /b 1
)

echo.
echo Build OK. Run CUDA benchmark (sequence lengths 128, 256, 512, 1024):
echo   flash_attention_cuda_standalone.exe 128
echo   flash_attention_cuda_standalone.exe 256
echo   flash_attention_cuda_standalone.exe 512
echo   flash_attention_cuda_standalone.exe 1024
echo Each prints CUDA_MS=... to compare with PyTorch/Triton.
