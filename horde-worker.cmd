@echo off
cd /d "%~dp0"
title AI Horde Worker

:Isolation
SET PYTHONNOUSERSITE=1
SET PYTHONPATH=
SET CONDA_SHLVL=

REM Default: serve the dashboard in your web browser. Pass --terminal for the in-terminal UI.
REM Power users can bind the LAN with: horde-worker.cmd --host 0.0.0.0  (unauthenticated; opt-in).
if /I "%~1"=="--terminal" goto TERMINAL

echo Starting the AI Horde Worker dashboard...
echo This window runs the worker: closing it (or pressing Ctrl+C here) stops the worker.
echo Closing just the dashboard window/tab leaves the worker running; reopen to reconnect.
echo Pass --terminal for the in-terminal UI.
echo.
call "%~dp0runtime.cmd" horde-worker-web %*
goto END

:TERMINAL
call "%~dp0runtime.cmd" horde-worker
goto END

:END
