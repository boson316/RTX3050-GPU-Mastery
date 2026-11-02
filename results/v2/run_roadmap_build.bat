@echo off
cd /d "%~dp0..\..\cuda_roadmap"
call "%CD%\build.bat"
exit /b %errorlevel%
