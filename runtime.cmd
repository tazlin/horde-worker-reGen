@echo off
REM Single Windows entry point: make sure uv exists, then hand every argument to the Python bootstrap
REM brain (bootstrap.py). All install/update/launch logic lives in Python now; this file's only
REM irreducible job is getting uv, which is the one thing that cannot yet be done in Python.
cd /d "%~dp0"

:Isolation
SET PYTHONNOUSERSITE=1
SET PYTHONPATH=
SET CONDA_SHLVL=

REM Keep uv's cache, the managed Python, and downloaded models in a peered data dir: a sibling of the
REM worker folder (same name with a -data suffix) that survives deleting or reinstalling the worker folder,
REM so a user starting fresh cannot lose their cached deps or model weights. HORDE_WORKER_DATA_DIR overrides
REM the location. This must match worker_bootstrap\paths.py:data_root. Respect caller-set values for each.
for %%I in ("%~dp0.") do set "WORKER_ROOT=%%~fI"
if not defined HORDE_WORKER_DATA_DIR set "HORDE_WORKER_DATA_DIR=%WORKER_ROOT%-data"
if not exist "%HORDE_WORKER_DATA_DIR%" md "%HORDE_WORKER_DATA_DIR%"
if not defined UV_CACHE_DIR set "UV_CACHE_DIR=%HORDE_WORKER_DATA_DIR%\uv_cache"
if not defined UV_PYTHON_INSTALL_DIR set "UV_PYTHON_INSTALL_DIR=%HORDE_WORKER_DATA_DIR%\python"
REM AIWORKER_CACHE_HOME intentionally unset here: setting it would outrank `cache_home` in bridgeData.yaml.
REM The worker applies the peered <data>\models default at lowest precedence from HORDE_WORKER_DATA_DIR, so
REM the ladder stays env var > cache_home > peered default. See worker_bootstrap/load_env_vars.py.
REM Self-contained install: use a uv-managed CPython, not a system one that a user could later uninstall.
if not defined UV_PYTHON_PREFERENCE set "UV_PYTHON_PREFERENCE=only-managed"

call :ensure_uv
if errorlevel 1 exit /b 1

REM --no-project + PEP 723 inline metadata means uv ignores the project here and runs bootstrap.py in a
REM tiny stdlib-only environment, so it works before .venv exists. --python 3.12 pins a managed CPython
REM rather than grabbing an ambient (e.g. conda) interpreter.
"%~dp0bin\uv.exe" run --python 3.12 --no-project --script "%~dp0bootstrap.py" %*
exit /b %errorlevel%

REM ---------------------------------------------------------------------------
:ensure_uv
if exist "%~dp0bin\uv.exe" exit /b 0
echo Downloading uv package manager...
if not exist "%~dp0bin" md "%~dp0bin"

REM Pinned for reproducibility; override with HORDE_WORKER_UV_VERSION to bump without editing this file.
set "UV_VERSION=0.11.21"
if defined HORDE_WORKER_UV_VERSION set "UV_VERSION=%HORDE_WORKER_UV_VERSION%"
set "UV_ZIP=uv-x86_64-pc-windows-msvc.zip"
if /I "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "UV_ZIP=uv-aarch64-pc-windows-msvc.zip"
set "UV_URL=https://github.com/astral-sh/uv/releases/download/%UV_VERSION%/%UV_ZIP%"

REM Prefer in-box curl.exe + tar.exe (Windows 10 1803+). This deliberately avoids the old
REM "powershell -c irm https://astral.sh/uv/install.ps1 | iex": a nested Windows PowerShell launched from
REM cmd inherits a pwsh-7-polluted PSModulePath and then fails to load Microsoft.PowerShell.Security, so
REM even Get-ExecutionPolicy throws. A plain HTTPS download has none of that fragility.
set "CURL=%SystemRoot%\System32\curl.exe"
set "TAR=%SystemRoot%\System32\tar.exe"
if not exist "%CURL%" goto :ensure_uv_ps
if not exist "%TAR%" goto :ensure_uv_ps
"%CURL%" -fL --retry 3 -o "%~dp0bin\uv.zip" "%UV_URL%"
if errorlevel 1 goto :ensure_uv_ps
REM The uv zip has uv.exe at its root, so extracting into bin\ lands bin\uv.exe directly.
"%TAR%" -xf "%~dp0bin\uv.zip" -C "%~dp0bin"
if errorlevel 1 goto :ensure_uv_ps
del "%~dp0bin\uv.zip" >nul 2>&1
if exist "%~dp0bin\uv.exe" exit /b 0

:ensure_uv_ps
REM Fallback for pre-1803 Windows (no in-box curl/tar). Reset PSModulePath to the system path first so a
REM pwsh-polluted environment cannot make Windows PowerShell load the wrong (CoreCLR) modules, and opt
REM into TLS 1.2 (WinPS 5.1 still defaults to 1.0).
del "%~dp0bin\uv.zip" >nul 2>&1
echo curl/tar unavailable; falling back to the PowerShell uv installer...
set "PSModulePath=%SystemRoot%\System32\WindowsPowerShell\v1.0\Modules"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12; $env:UV_INSTALL_DIR='%~dp0bin'; irm https://astral.sh/uv/install.ps1 | iex"
if exist "%~dp0bin\uv.exe" exit /b 0
echo.
echo ERROR: Could not install uv (the package manager).
echo   - Confirm GitHub and astral.sh are reachable (proxy/firewall?).
echo   - Or place a uv.exe in "%~dp0bin" and re-run.
exit /b 1
