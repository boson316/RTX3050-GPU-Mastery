@echo off
REM 編譯並執行 matrixMul.cu（需先有 VS 的 cl 環境，或從 x64 Native Tools CMD 執行）

set "VCVARS="
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath 2^>nul`) do set "VSINST=%%i"
if defined VSINST if exist "%VSINST%\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%VSINST%\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

if not defined VCVARS (
    echo 找不到 Visual Studio 的 vcvars64.bat，請用 x64 Native Tools CMD 執行：
    echo   nvcc matrixMul.cu -o matrixMul.exe -allow-unsupported-compiler
    echo   matrixMul
    exit /b 1
)

call "%VCVARS%" >nul 2>&1
nvcc matrixMul.cu -o matrixMul.exe -allow-unsupported-compiler
if %ERRORLEVEL% neq 0 (
    echo 編譯失敗
    exit /b 1
)
echo 編譯成功，執行 matrixMul.exe（下方會顯示 CPU time / GPU time）
matrixMul.exe
