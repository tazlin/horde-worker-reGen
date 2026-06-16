@echo off
REM Install / update the environment. Thin wrapper kept under its historical name (docs, shortcuts and
REM muscle memory reference it): it delegates to runtime.cmd, which ensures uv and runs `bootstrap.py sync`.
cd /d "%~dp0"
call "%~dp0runtime.cmd" sync %*
set "RC=%errorlevel%"
REM When the launcher bootstraps on first run it flows straight into starting the worker; only pause when
REM a user ran this script directly (and interactively).
if not defined HORDE_WORKER_FROM_LAUNCHER if not defined HORDE_WORKER_NONINTERACTIVE pause
exit /b %RC%
