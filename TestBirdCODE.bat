@echo off
REM Checks BioTool's BirdCODE against the expected results of the original BirdCODE code
REM (two demo recordings from github.com/earthspecies/sound-event-detection).
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
uv run python -m biotool.check_birdcode
pause
