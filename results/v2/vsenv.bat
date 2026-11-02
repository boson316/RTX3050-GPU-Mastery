@echo off
rem Load MSVC x64 environment (located via vswhere -latest), then run the given command.
for /f "usebackq delims=" %%I in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath`) do set "VSROOT=%%I"
if not exist "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat" (
  echo ERROR: vcvars64.bat not found under "%VSROOT%"
  exit /b 9
)
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul
set "DISTUTILS_USE_SDK=1"
cd /d "%~dp0..\.."
%*
