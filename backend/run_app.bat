@echo off
title Digit Recognition Web App
echo ========================================================
echo        STARTING DIGIT RECOGNITION WEB APP
echo ========================================================
echo.

cd /d "%~dp0"

:: Check if model exists, if not train it
if not exist "model.pkl" (
    echo [1/3] model.pkl not found. Training KNN model...
    python train_model.py
) else (
    echo [1/3] model.pkl found.
)

:: Start FastAPI server in background, then open browser after 3s delay
echo [2/3] Starting FastAPI server on http://127.0.0.1:8000...
echo [3/3] Browser will open automatically once server is ready...

:: Open browser after 3-second delay (gives server time to start)
start "" cmd /c "timeout /t 3 /nobreak >nul && start "" http://127.0.0.1:8000"

:: Start FastAPI server (blocking)
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

pause
