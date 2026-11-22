@echo off
chcp 65001 >nul
REM Runs one pipeline step. Change SCRIPT to the step you want to run.
set "SCRIPT=05_compare.py"

cd /d "%~dp0"
set "PYTHONUTF8=1"
where python >nul 2>nul && (set "PY=python") || (set "PY=py -3")

%PY% -c "import numpy, PIL, skimage, sklearn, pandas, matplotlib" 2>nul
if errorlevel 1 (
    echo Installing missing packages ...
    %PY% -m pip install numpy pillow scikit-image scikit-learn pandas matplotlib
)

echo Running %SCRIPT% ...
%PY% -u %SCRIPT% > "%~dp0last_run_log.txt" 2>&1
type "%~dp0last_run_log.txt"
echo.
echo Finished. Log saved to code\last_run_log.txt
pause
