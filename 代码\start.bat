@echo off
chcp 65001 >nul
title Water AI System Launcher

echo ========================================
echo   AI Water Quality Monitoring System
echo ========================================
echo.

set VENV_PYTHON=c:\water-llm\conference\Scripts\python.exe

:: 1. Start Django backend
echo [1/2] Starting Django backend (port 8000)...
start "Django-API" cmd /c "%VENV_PYTHON% c:\water-llm\iot_backend\manage.py runserver 0.0.0.0:8000"
timeout /t 2 /nobreak >nul
echo   [OK] Django backend started
echo.

:: 2. Start Streamlit frontend
echo [2/2] Starting Streamlit frontend (port 8501)...
start "Streamlit-UI" cmd /c "cd /d c:\water-llm && %VENV_PYTHON% -m streamlit run src/app.py --server.port 8501 --server.headless true"
timeout /t 3 /nobreak >nul
echo   [OK] Streamlit frontend started
echo.

:: 3. Show access URLs
echo ========================================
echo   All services started!
echo ========================================
echo.
echo   Frontend : http://localhost:8501
echo   API      : http://127.0.0.1:8000/api/
echo   Admin    : http://127.0.0.1:8000/admin/
echo.
start http://localhost:8501
echo   Browser opened automatically
echo.
pause
