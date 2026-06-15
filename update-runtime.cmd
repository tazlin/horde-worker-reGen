@echo off
cd /d "%~dp0"
title AI Horde Worker - Update Runtime

echo ============================================
echo   AI Horde Worker - Install / Update
echo ============================================
echo.

:Isolation
SET PYTHONNOUSERSITE=1
SET PYTHONPATH=
SET CONDA_SHLVL=

REM Keep uv's package cache next to the install instead of on the home drive (%LOCALAPPDATA%). The cache
REM is several GB; defaulting it here means it lands on the drive the user chose for the worker, and stays on
REM the same volume as .venv so uv can hardlink instead of falling back to slow full-copies. Respect an
REM existing UV_CACHE_DIR so power users / dev checkouts can still point at a shared global cache.
if not defined UV_CACHE_DIR set "UV_CACHE_DIR=%~dp0bin\uv_cache"

REM Windows long-path support is a SYSTEM-WIDE setting (HKLM) and needs admin, so it is opt-in, never
REM automatic: we do not change your system unless you ask. Set HORDE_WORKER_ENABLE_LONG_PATHS=1 to opt in
REM (only needed if you hit "path too long" errors; keeping the install path short usually avoids them).
if defined HORDE_WORKER_ENABLE_LONG_PATHS (
    echo Enabling Windows long-path support system-wide ^(requires administrator^)...
    Reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v "LongPathsEnabled" /t REG_DWORD /d "1" /f
)

REM The uv extra is the torch *build* (cu126/cu130/cu132/cpu), all on the latest torch (2.12.0).
REM   BUILD precedence: --cu126/--cu130/--cu132/--cpu flag > HORDE_WORKER_BACKEND > bin\backend (the
REM                     build the installer detected from the GPU driver) > cu126 (broadest CUDA-12
REM                     build: runs on any CUDA 12.6+ driver and, via driver back-compat, CUDA 13).
REM Older torch lines and ROCm are not locked; install those ad-hoc (see pyproject.toml).
SET "BUILD="
for %%a in (%*) do (
    if /I "%%a"=="--cpu" SET BUILD=cpu
    if /I "%%a"=="--cu126" SET BUILD=cu126
    if /I "%%a"=="--cu130" SET BUILD=cu130
    if /I "%%a"=="--cu132" SET BUILD=cu132
)
if not defined BUILD if defined HORDE_WORKER_BACKEND set "BUILD=%HORDE_WORKER_BACKEND%"
if not defined BUILD if exist "%~dp0bin\backend" set /p BUILD=<"%~dp0bin\backend"
if not defined BUILD set "BUILD=cu126"

REM torch 2.12.0 publishes cu126 (not cu128) for CUDA 12; map a legacy cu128 request so an existing
REM install whose bin\backend still says cu128 keeps working.
if /I "%BUILD%"=="cu128" (
    echo Note: torch 2.12.0 has no cu128 build; using cu126 ^(runs on any CUDA 12.6+ driver^).
    set "BUILD=cu126"
)

set "GPU_EXTRA=%BUILD%"
REM Only the builds 2.12.0 actually publishes are locked. If an unsupported build was requested
REM (e.g. rocm), explain the ad-hoc paths rather than failing cryptically.
findstr /b /c:"%GPU_EXTRA% = " "%~dp0pyproject.toml" >nul 2>&1
if not errorlevel 1 goto :have_extra
echo ERROR: '%GPU_EXTRA%' is not a locked build. The lock provides the latest torch (2.12.0) for:
echo        cu126 (CUDA 12.6+), cu130 (CUDA 13.x), cu132 (CUDA 13.2), cpu.
echo        For AMD/ROCm or an older torch line, install ad-hoc, e.g.:
echo        UV_TORCH_BACKEND=auto uv pip install torch torchvision torchaudio
if not defined HORDE_WORKER_NONINTERACTIVE pause
exit /b 1
:have_extra

REM Install uv if not present
if not exist "%~dp0bin\uv.exe" (
    echo Downloading uv package manager...
    powershell -ExecutionPolicy ByPass -NoProfile -c "$env:UV_INSTALL_DIR='%~dp0bin'; irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to download uv. Check your internet connection.
        if not defined HORDE_WORKER_NONINTERACTIVE pause
        exit /b 1
    )
    echo Done.
    echo.
)

echo Installing dependencies for GPU backend: %GPU_EXTRA%
echo (This may take a few minutes on first run...)
echo.
"%~dp0bin\uv.exe" sync --locked --extra %GPU_EXTRA%
if errorlevel 1 (
    echo.
    echo ERROR: Installation failed.
    echo   - Try deleting the .venv folder and running this script again.
    echo   - If the problem persists, ask for help in #local-workers on Discord.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Installation complete!
echo ============================================
echo.
REM When the launcher (runtime.cmd) bootstraps the environment on first run it sets this flag and then
REM goes straight on to start the worker, so skip the manual "next steps" and the keypress: the dashboard
REM opens without an unexpected "Press any key" interrupting first launch. A user who runs this script
REM directly still gets both.
if not defined HORDE_WORKER_FROM_LAUNCHER (
    echo Next steps:
    echo   1. Edit bridgeData.yaml with your API key and worker name
    echo   2. Run horde-bridge.cmd to start the worker
    echo      ^(or horde-worker.cmd for the interactive launcher^)
    echo.
    if not defined HORDE_WORKER_NONINTERACTIVE pause
)
