@echo off
REM Build and run the CUDA-kernel CPU emulation tests with MSVC (no GPU needed).
REM Run from an "x64 Native Tools Command Prompt for VS 2022" (needs cl.exe with C++20 <barrier>).
REM A test that deadlocks (barrier / full-mask shuffle not reached by every thread) never finishes:
REM if a test hangs for more than a minute, that is a failure.
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "OUT=%TEMP%\cuda_emu_tests"
if not exist "%OUT%" mkdir "%OUT%"
set STATUS=0
for %%F in (test_*_emu.cpp) do (
    echo === %%~nF
    cl /nologo /std:c++20 /O2 /EHsc /W3 /wd4068 /Fe"%OUT%\%%~nF.exe" /Fo"%OUT%\\" "%%F" >nul
    if errorlevel 1 (
        echo !!! %%~nF failed to compile - rerun the cl command above without ^>nul to see errors
        set STATUS=1
    ) else (
        "%OUT%\%%~nF.exe"
        if errorlevel 1 ( echo !!! %%~nF failed & set STATUS=1 )
    )
)
exit /b %STATUS%
