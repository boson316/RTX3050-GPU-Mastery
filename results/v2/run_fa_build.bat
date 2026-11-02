@echo off
cd /d "%~dp0..\..\flash_attention_simple"
call "%CD%\build.bat"
exit /b %errorlevel%
