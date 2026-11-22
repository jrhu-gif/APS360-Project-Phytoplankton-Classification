@echo off
chcp 65001 >nul
REM Starts a local Jupyter server that Google Colab can connect to.
REM Keep this window open while you use Colab. Close it to stop the server.

cd /d "%~dp0.."
if not exist "content" mkdir "content"

where python >nul 2>nul && (set "PY=python") || (set "PY=py -3")

%PY% -c "import notebook" 2>nul
if errorlevel 1 (
    echo Jupyter is not installed - installing it now ...
    %PY% -m pip install jupyter
)

echo.
echo ============================================================
echo  Copy the link that starts with  http://localhost:8888/
echo  (remove "/tree" if it is there), then in Colab:
echo  Connect (arrow) - Connect to a local runtime - paste - Connect
echo ============================================================
echo.
%PY% -m notebook --no-browser --port=8888 ^
  --NotebookApp.allow_origin="https://colab.research.google.com" ^
  --NotebookApp.port_retries=0 ^
  --NotebookApp.allow_credentials=True
pause
