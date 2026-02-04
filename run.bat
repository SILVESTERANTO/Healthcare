@echo off
echo ========================================
echo   Cyber Threat Detection System
echo   Starting FastAPI Server...
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
    echo Installing dependencies...
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo.
echo Starting server at http://localhost:8000
echo Press Ctrl+C to stop the server
echo.

REM Kill any existing process on port 8000
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do taskkill /pid %%a /f 2>nul

python app.py

pause
