@echo off
cd /d "%~dp0"

:Isolation
SET PYTHONNOUSERSITE=1
SET PYTHONPATH=
SET CONDA_SHLVL=

REM Match update-runtime.cmd: keep uv's cache on the install drive (next to .venv) rather than the home drive.
if not defined UV_CACHE_DIR set "UV_CACHE_DIR=%~dp0bin\uv_cache"

IF EXIST ".venv" GOTO APP

:INSTALL
call "%~dp0update-runtime.cmd"

:APP
REM Prefer the bundled uv (installer/end-user setups); fall back to a uv on PATH (dev checkouts).
SET "HORDE_UV=%~dp0bin\uv.exe"
IF NOT EXIST "%HORDE_UV%" SET "HORDE_UV=uv"
"%HORDE_UV%" run --no-sync %*
