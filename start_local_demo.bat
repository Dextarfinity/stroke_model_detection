@echo off
echo.
echo ========================================
echo   Stroke Detection API - Local Demo
echo ========================================
echo.
echo Installing required packages...
C:/Users/Teacher/Downloads/stroke_dataset_yolo/.venv/Scripts/python.exe -m pip install -r requirements_local.txt

echo.
echo Starting local demo server...
echo.
echo Demo URL: http://localhost:8000/demo
echo API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

C:/Users/Teacher/Downloads/stroke_dataset_yolo/.venv/Scripts/python.exe local_demo.py

pause