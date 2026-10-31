@echo off
REM Same as BioTool.bat but keeps a console window open so error messages are visible
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
uv run python -m biotool.gui.app %*
pause
