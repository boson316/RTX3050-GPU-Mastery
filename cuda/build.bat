@echo off
REM 在一般 CMD 編譯 CUDA：先載入 Visual Studio 的 cl.exe 環境，再跑 nvcc
REM 需要已安裝：CUDA Toolkit + Visual Studio（或 Build Tools）含「使用 C++ 的桌面開發」

set "VCVARS="
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath 2^>nul`) do set "VSINST=%%i"
if defined VSINST if exist "%VSINST%\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%VSINST%\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

if not defined VCVARS (
    echo 找不到 Visual Studio 的 vcvars64.bat
    echo 請安裝 Visual Studio 或 Build Tools，並勾選「使用 C++ 的桌面開發」
    echo 或從開始選單開「x64 Native Tools Command Prompt for VS」再執行 nvcc
    exit /b 1
)

call "%VCVARS%" >nul 2>&1
nvcc vectorAdd.cu -o vectorAdd.exe -allow-unsupported-compiler
if %ERRORLEVEL% neq 0 (
    echo 編譯失敗
    exit /b 1
)
echo 編譯成功，執行 vectorAdd.exe
vectorAdd.exe
