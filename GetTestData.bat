@echo off
REM Downloads the test data (BirdNET example, ESC-50 sample, Powdermill sample ~110 MB) into test_data\
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
uv run biotool get-test-data
pause
