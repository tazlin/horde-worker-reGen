@echo off
cd /d "%~dp0"
title AI Horde Worker

REM Three peer launch options:
REM   (default)    web dashboard in your browser (a pop-up window).
REM   --terminal   the in-terminal Textual UI (console, no browser).
REM   --headless   no UI: run the worker in the foreground, printing to this console.
REM Power users can bind the LAN with: horde-worker.cmd --host 0.0.0.0  (unauthenticated; opt-in).
REM The branches goto labels (not parenthesized blocks) so the rest-of-line capture and %errorlevel%
REM are read at run time, not parse time. REST holds every argument after the leading mode flag.
if /I "%~1"=="--web" goto :mode_web
if /I "%~1"=="--terminal" goto :mode_terminal
if /I "%~1"=="--headless" goto :mode_headless
goto :default_web

:mode_web
for /f "tokens=1,*" %%a in ("%*") do set "REST=%%b"
call "%~dp0runtime.cmd" launch web %REST%
goto :done

:mode_terminal
for /f "tokens=1,*" %%a in ("%*") do set "REST=%%b"
call "%~dp0runtime.cmd" launch terminal %REST%
goto :done

:mode_headless
for /f "tokens=1,*" %%a in ("%*") do set "REST=%%b"
call "%~dp0runtime.cmd" launch bridge %REST%
goto :done

:default_web
echo Starting the AI Horde Worker dashboard...
echo This window runs the worker: closing it (or pressing Ctrl+C here) stops the worker.
echo Closing just the dashboard window/tab leaves the worker running; reopen to reconnect.
echo Pass --terminal for the in-terminal UI, or --headless for no UI.
echo.
call "%~dp0runtime.cmd" launch web %*

:done
REM %errorlevel% is read outside any if-block so it reflects the call's real exit code (a parenthesized
REM block would expand it at parse time, before the call ran).
exit /b %errorlevel%
