@echo off
REM High-pass pre-filter experiment: runs the evaluation without a filter and with high-pass
REM filters at 300, 500 and 1000 Hz, then opens the report (comparison table at the top).
REM Takes roughly 10-15 minutes. The no-filter run is reused from Evaluate.bat if available.
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
set "PM=test_data\Powdermill"
set "ARGS=--output evaluation --highpass 0,300,500,1000"
if exist "%PM%\sample_20min" set "ARGS=%ARGS% --powdermill %PM%\sample_20min --powdermill-annotations %PM%\annotations --powdermill-readme %PM%\README.txt"
if exist "test_data\ESC-50_sample\labels.csv" set "ARGS=%ARGS% --esc50 test_data\ESC-50_sample"
uv run biotool evaluate %ARGS%
if exist "evaluation\evaluation_report.html" start "" "evaluation\evaluation_report.html"
pause
