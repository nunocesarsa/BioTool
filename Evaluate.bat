@echo off
REM Scores all models against the annotated test data in test_data\ and opens an HTML report.
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
set "PM=test_data\Powdermill"
set "ARGS=--output evaluation"
if exist "%PM%\sample_20min" set "ARGS=%ARGS% --powdermill %PM%\sample_20min --powdermill-annotations %PM%\annotations --powdermill-readme %PM%\README.txt"
if exist "test_data\ESC-50_sample\labels.csv" set "ARGS=%ARGS% --esc50 test_data\ESC-50_sample"
uv run biotool evaluate %ARGS%
if exist "evaluation\evaluation_report.html" start "" "evaluation\evaluation_report.html"
pause
