@echo off
REM 編譯/重裝 custom_conv extension
REM 請在「x64 Native Tools Command Prompt」執行，且先啟用 .venv
REM 編譯前會把 CUDA 12.4 的 bin 放到 PATH 最前面（與 PyTorch cu124 一致）

set "CUDA124=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
if exist "%CUDA124%\nvcc.exe" (
    set "PATH=%CUDA124%;%PATH%"
    echo [OK] Using CUDA 12.4 for build
) else (
    echo [!!] CUDA 12.4 not found at %CUDA124%
    echo      If PyTorch is cu124, install CUDA 12.4 or set PATH manually and run: pip install --no-build-isolation .
    exit /b 1
)

cd /d "%~dp0"
REM 避免 VC 環境被重複啟動導致編譯失敗（PyTorch 建議）
set DISTUTILS_USE_SDK=1
pip install --no-build-isolation .
if %ERRORLEVEL% neq 0 (
    echo.
    echo If CUDA version mismatch: set PATH to CUDA 12.4 bin then retry.
    echo If cl not found: run this from "x64 Native Tools Command Prompt".
    exit /b 1
)
echo.
echo Done. Run: python mnist_custom_conv.py
