@echo off
REM Installs the optional DeepForestSound model (African forest species).
REM Adds PyTorch (~500 MB). The model files (~700 MB) download the first time DFS is used.
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
uv sync --extra dfs
if errorlevel 1 (
    echo Installation failed - see the messages above.
    pause
    exit /b 1
)
echo dfs> ".venv\biotool-extras.txt"
echo.
echo DeepForestSound support installed. Restart BioTool and tick it in the model list.
pause
