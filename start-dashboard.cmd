@echo off
setlocal
title Quito Solar - close this window to stop
cd /d "%~dp0"

if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0src\serve_dashboard.py"
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 "%~dp0src\serve_dashboard.py"
    ) else (
        python "%~dp0src\serve_dashboard.py"
    )
)

if errorlevel 1 (
    echo.
    echo Could not start the dashboard. Check the message above.
    echo Python 3 is required. No extra packages are needed.
    pause
)
endlocal
