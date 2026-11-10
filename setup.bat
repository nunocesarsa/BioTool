@echo off
REM One-time setup: installs uv (if needed), Python 3.12 and all dependencies into .venv
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo Installing uv ...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

echo Installing BioTool dependencies (first time takes a few minutes, ~1.5 GB) ...
REM optional extras installed before (one per line in .venv\biotool-extras.txt) are kept
set "EXTRAS="
if exist ".venv\biotool-extras.txt" for /f "usebackq tokens=1" %%e in (".venv\biotool-extras.txt") do call :addextra %%e
uv sync %EXTRAS%
if errorlevel 1 (
    echo.
    echo Setup failed - see the messages above.
    pause
    exit /b 1
)
echo.
echo Setup complete. Start the app with BioTool.bat
pause
exit /b 0

:addextra
echo %EXTRAS% | find "--extra %1 " >nul || set "EXTRAS=%EXTRAS% --extra %1 "
exit /b 0
