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
uv sync
if errorlevel 1 (
    echo.
    echo Setup failed - see the messages above.
    pause
    exit /b 1
)
echo.
echo Setup complete. Start the app with BioTool.bat
pause
