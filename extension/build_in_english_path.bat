@echo off
REM 在「純英文路徑」建置 extension，避免 MSVC 對「程式相關」等中文路徑解析錯誤。
REM 建置完成後會把 custom_conv 複製回你專案用的 .venv。
REM 請在「x64 Native Tools Command Prompt」執行；執行前請先手動啟用你的 .venv 一次（僅為取得 Python 路徑），或本 script 會建立並使用 C:\cuda_build_venv。

set "BUILD_ROOT=C:\cuda_torch_ext_build"
set "BUILD_VENV=C:\cuda_build_venv"
set "CUDA124=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
set "SCRIPT_DIR=%~dp0"
set "ORIGINAL_VENV=%SCRIPT_DIR%..\..\.venv"

echo [1/5] 準備英文路徑建置目錄: %BUILD_ROOT%
if not exist "%BUILD_ROOT%" mkdir "%BUILD_ROOT%"
REM 清除舊 build 與 PyTorch extension 快取，強制重編
if exist "%BUILD_ROOT%\build" rd /s /q "%BUILD_ROOT%\build"
if exist "%USERPROFILE%\.cache\torch_extensions" rd /s /q "%USERPROFILE%\.cache\torch_extensions"
copy /Y "%SCRIPT_DIR%setup.py" "%BUILD_ROOT%\"
copy /Y "%SCRIPT_DIR%conv_kernel.cu" "%BUILD_ROOT%\"
findstr /C:"kFloat16" "%BUILD_ROOT%\conv_kernel.cu" >nul && echo     [OK] 原始碼已含 FP16 分支 || echo     [!!] 原始碼無 kFloat16，請確認 conv_kernel.cu 已儲存

echo.
echo [2/5] 建置用 venv（僅建置時用）: %BUILD_VENV%
if not exist "%BUILD_VENV%\Scripts\activate.bat" (
    echo     首次建立中（使用專案 .venv 的 64 位元 Python）...
    set "PY64=%ORIGINAL_VENV%\Scripts\python.exe"
    if exist "%PY64%" (
        "%PY64%" -m venv "%BUILD_VENV%"
    ) else (
        echo     [!!] 找不到專案 .venv，改用 py -3-64
        py -3-64 -m venv "%BUILD_VENV%"
    )
    call "%BUILD_VENV%\Scripts\activate.bat"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
) else (
    call "%BUILD_VENV%\Scripts\activate.bat"
)

echo.
echo [3/5] 設定 CUDA 與 DISTUTILS_USE_SDK
if exist "%CUDA124%\nvcc.exe" set "PATH=%CUDA124%;%PATH%"
set DISTUTILS_USE_SDK=1

echo [4/5] 在英文路徑下編譯 extension
cd /d "%BUILD_ROOT%"
pip install --no-build-isolation .
if %ERRORLEVEL% neq 0 (
    echo.
    echo 編譯失敗。若為 CUDA 版本不符，請安裝 CUDA 12.4 並確認 PATH。
    exit /b 1
)

echo.
echo [5/5] 複製 custom_conv 回專案 .venv: %ORIGINAL_VENV%
if not exist "%ORIGINAL_VENV%\Lib\site-packages" (
    echo [!!] 找不到專案 .venv，請手動將 %BUILD_VENV%\Lib\site-packages 下的 custom_conv 與 custom_conv-*.dist-info 複製到你的 .venv\Lib\site-packages
    exit /b 1
)
set "SP=%BUILD_VENV%\Lib\site-packages"
set "OV=%ORIGINAL_VENV%\Lib\site-packages"
if exist "%SP%\custom_conv" (
    xcopy /Y /E /I "%SP%\custom_conv" "%OV%\custom_conv\"
) else (
    for %%F in ("%SP%\custom_conv*.pyd") do (
        copy /Y "%%F" "%OV%\"
        echo     已複製: %%~nxF
    )
)
for /d %%D in ("%SP%\custom_conv-*.dist-info") do xcopy /Y /E /I "%%D" "%OV%\%%~nxD\"
echo.
echo 完成。執行時請確認 Python 使用此 .venv：%ORIGINAL_VENV%
echo 可執行: "%ORIGINAL_VENV%\Scripts\python.exe" cursor\cuda_torch_ext\mnist_custom_conv.py
pause
