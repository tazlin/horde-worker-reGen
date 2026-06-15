@echo off
cd /d "%~dp0"

:Isolation
SET PYTHONNOUSERSITE=1
SET PYTHONPATH=
SET CONDA_SHLVL=

IF EXIST ".venv" GOTO APP

:INSTALL
call "%~dp0update-runtime.cmd"

:APP
REM Prefer the bundled uv (installer/end-user setups); fall back to a uv on PATH (dev checkouts).
SET "HORDE_UV=%~dp0bin\uv.exe"
IF NOT EXIST "%HORDE_UV%" SET "HORDE_UV=uv"
"%HORDE_UV%" run --no-sync %*
