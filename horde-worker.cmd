@echo off
cd /d "%~dp0"
title AI Horde Worker

REM Default: serve the dashboard in your web browser. Pass --terminal for the in-terminal UI.
REM Power users can bind the LAN with: horde-worker.cmd --host 0.0.0.0  (unauthenticated; opt-in).
if /I "%~1"=="--terminal" (
    call "%~dp0runtime.cmd" launch terminal
    goto :done
)

echo Starting the AI Horde Worker dashboard...
echo This window runs the worker: closing it (or pressing Ctrl+C here) stops the worker.
echo Closing just the dashboard window/tab leaves the worker running; reopen to reconnect.
echo Pass --terminal for the in-terminal UI.
echo.
call "%~dp0runtime.cmd" launch web %*

:done
REM %errorlevel% is read outside the if-block so it reflects the call's real exit code (a parenthesized
REM block would expand it at parse time, before the call ran).
exit /b %errorlevel%
