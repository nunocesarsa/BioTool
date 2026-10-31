@echo off
REM Starts the BioTool desktop app (run setup.bat once first)
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
if not exist ".venv\Scripts\pythonw.exe" (
    echo BioTool is not installed yet - running setup first ...
    call setup.bat
)
start "" ".venv\Scripts\pythonw.exe" -m biotool.gui.app %*
