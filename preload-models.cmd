@echo off
cd /d "%~dp0"
title AI Horde Worker - Preload Models
REM Download/verify the configured models, then exit (no worker started).
call "%~dp0runtime.cmd" preload
set "RC=%errorlevel%"
pause
exit /b %RC%
