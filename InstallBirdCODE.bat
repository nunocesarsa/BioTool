@echo off
REM Installs the optional BirdCODE model (zero-shot detection of bird calls, Earth Species Project).
REM Adds PyTorch (shared with the other optional PyTorch models).
REM The model downloads from huggingface.co the first time BirdCODE is used.
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
REM optional extras installed before (one per line in .venv\biotool-extras.txt) are kept
set "EXTRAS="
if exist ".venv\biotool-extras.txt" for /f "usebackq tokens=1" %%e in (".venv\biotool-extras.txt") do call :addextra %%e
call :addextra birdcode
uv sync%EXTRAS%
if errorlevel 1 (
    echo Installation failed - see the messages above.
    pause
    exit /b 1
)
if exist ".venv\biotool-extras.txt" del ".venv\biotool-extras.txt"
for %%e in (%EXTRAS:--extra =%) do echo %%e>> ".venv\biotool-extras.txt"
echo.
echo BirdCODE support installed. Restart BioTool and tick it in the model list.
pause
exit /b 0

:addextra
echo %EXTRAS% | find "--extra %1 " >nul || set "EXTRAS=%EXTRAS% --extra %1 "
exit /b 0
