@echo off
cd /d "%~dp0"
title AI Horde Worker

echo ============================================
echo   AI Horde Worker
echo ============================================
echo.
echo Setting up (if needed), downloading models, then starting the worker...
echo (Press Ctrl+C to stop the worker gracefully)
echo.

REM The bridge path: ensure the environment, download/verify models, then run the headless worker.
call "%~dp0runtime.cmd" launch bridge %*
set "RC=%errorlevel%"
if not "%RC%"=="0" (
    echo.
    echo ERROR: The worker exited with code %RC%. Check the output above.
)
pause
exit /b %RC%
