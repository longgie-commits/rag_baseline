@echo off
echo =========================================
echo       RAG Baseline Project Runner
echo =========================================
echo.
echo [1] Build Vector Index (build_index.py)
echo [2] Start Retrieval (retrieve.py)
echo.

set /p choice="Enter your choice (1 or 2): "

if "%choice%"=="1" (
    echo.
    echo Starting build_index.py...
    "%~dp0\.venv\Scripts\python.exe" "%~dp0src\build_index.py"
) else if "%choice%"=="2" (
    echo.
    echo Starting retrieve.py...
    "%~dp0\.venv\Scripts\python.exe" "%~dp0src\retrieve.py"
) else (
    echo Invalid choice.
)

echo.
pause
