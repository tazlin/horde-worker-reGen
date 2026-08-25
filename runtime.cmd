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
REM Cache mode: "shared" leaves UV_CACHE_DIR unset so uv uses its own default (system) cache a power user
REM already populates for other projects (no 7-10 GB duplicate); the worker then never auto-prunes it.
REM "isolated" (default) keeps a private cache in the data dir that we can prune safely. Must match
REM worker_bootstrap\paths.py:uv_cache_mode. Respect a caller-set UV_CACHE_DIR in either mode.
if /I not "%HORDE_WORKER_UV_CACHE_MODE%"=="shared" if not defined UV_CACHE_DIR set "UV_CACHE_DIR=%HORDE_WORKER_DATA_DIR%\uv_cache"
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
REM
REM --cache-dir gives THIS parent `uv run` its own tiny cache, deliberately NOT the worker UV_CACHE_DIR the
REM children use. `uv run --script` holds a shared (read) lock on its cache's .lock for the whole script
REM lifetime, while the post-sync `uv cache prune` child wants an exclusive (write) lock on the same file.
REM Pointing them at the same cache deadlocks prune until it times out (a ~5 min apparent hang). UV_CACHE_DIR
REM stays set, so the sync/prune children inherit the worker cache; only this parent is moved.
"%~dp0bin\uv.exe" run --python 3.12 --no-project --cache-dir "%HORDE_WORKER_DATA_DIR%\bootstrap_cache" --script "%~dp0bootstrap.py" %*
exit /b %errorlevel%

REM ---------------------------------------------------------------------------
:ensure_uv
REM This version MUST match [tool.uv] required-version in pyproject.toml. test_uv_version_consistency.py
REM enforces this: uv checks its version at runtime against required-version, so the version we download
REM here must satisfy it. Override with HORDE_WORKER_UV_VERSION to bump without editing this file.
set "UV_VERSION=0.12.1"
if defined HORDE_WORKER_UV_VERSION set "UV_VERSION=%HORDE_WORKER_UV_VERSION%"
set "UV_ACTUAL="
if not exist "%~dp0bin\uv.exe" goto :download_uv
REM Probe from outside the project: an older uv may enforce the new pyproject.toml pin even for a version
REM check, which would prevent bootstrap from discovering and repairing the mismatch.
set "UV_PROBE_FILE=%HORDE_WORKER_DATA_DIR%\uv-version-probe-%RANDOM%.txt"
pushd "%HORDE_WORKER_DATA_DIR%"
"%~dp0bin\uv.exe" --version > "%UV_PROBE_FILE%" 2>nul
popd
for /f "usebackq tokens=1,2" %%A in ("%UV_PROBE_FILE%") do if /I "%%A"=="uv" set "UV_ACTUAL=%%B"
del "%UV_PROBE_FILE%" >nul 2>&1
if "%UV_ACTUAL%"=="%UV_VERSION%" exit /b 0
if not defined UV_ACTUAL set "UV_ACTUAL=unknown"
echo Updating uv package manager from %UV_ACTUAL% to %UV_VERSION%...
goto :download_uv_payload

:download_uv
echo Downloading uv package manager...

:download_uv_payload
if not exist "%~dp0bin" md "%~dp0bin"
set "UV_ARCHIVE=uv-x86_64-pc-windows-msvc.zip"
if /I "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "UV_ARCHIVE=uv-aarch64-pc-windows-msvc.zip"
set "UV_URL=https://github.com/astral-sh/uv/releases/download/%UV_VERSION%/%UV_ARCHIVE%"
set "UV_DOWNLOAD=%~dp0bin\.uv-download-%RANDOM%.zip"
set "UV_CHECKSUM=%UV_DOWNLOAD%.sha256"
set "UV_STAGE=%~dp0bin\.uv-stage-%RANDOM%"
set "UV_EXPECTED="
set "UV_CANDIDATE="

REM Prefer in-box curl.exe + tar.exe (Windows 10 1803+). This deliberately avoids the old
REM remote-script PowerShell installer: a nested Windows PowerShell launched from
REM cmd inherits a pwsh-7-polluted PSModulePath and then fails to load Microsoft.PowerShell.Security, so
REM even Get-ExecutionPolicy throws. A plain HTTPS download has none of that fragility.
set "CURL=%SystemRoot%\System32\curl.exe"
set "TAR=%SystemRoot%\System32\tar.exe"
if not exist "%CURL%" goto :ensure_uv_ps
if not exist "%TAR%" goto :ensure_uv_ps
if not exist "%SystemRoot%\System32\certutil.exe" goto :ensure_uv_ps
"%CURL%" -fL --retry 3 -o "%UV_CHECKSUM%" "%UV_URL%.sha256"
if errorlevel 1 goto :ensure_uv_ps
"%CURL%" -fL --retry 3 -o "%UV_DOWNLOAD%" "%UV_URL%"
if errorlevel 1 goto :ensure_uv_ps
for /f "usebackq tokens=1" %%H in ("%UV_CHECKSUM%") do if not defined UV_EXPECTED set "UV_EXPECTED=%%H"
if not defined UV_EXPECTED goto :ensure_uv_ps
if "%UV_EXPECTED:~63,1%"=="" goto :ensure_uv_ps
if not "%UV_EXPECTED:~64,1%"=="" goto :ensure_uv_ps
echo(%UV_EXPECTED%| "%SystemRoot%\System32\findstr.exe" /R /X "[0-9A-Fa-f][0-9A-Fa-f]*" >nul
if errorlevel 1 goto :ensure_uv_ps
"%SystemRoot%\System32\certutil.exe" -hashfile "%UV_DOWNLOAD%" SHA256 | findstr /L /I /X /C:"%UV_EXPECTED%" >nul
if errorlevel 1 goto :ensure_uv_ps
md "%UV_STAGE%" >nul 2>&1
"%TAR%" -xf "%UV_DOWNLOAD%" -C "%UV_STAGE%"
if errorlevel 1 goto :ensure_uv_ps
for /r "%UV_STAGE%" %%F in (uv.exe) do if not defined UV_CANDIDATE set "UV_CANDIDATE=%%F"
if not defined UV_CANDIDATE goto :ensure_uv_ps
set "UV_ACTUAL="
pushd "%HORDE_WORKER_DATA_DIR%"
"%UV_CANDIDATE%" --version > "%UV_STAGE%\version.txt" 2>nul
popd
for /f "usebackq tokens=1,2" %%A in ("%UV_STAGE%\version.txt") do if /I "%%A"=="uv" set "UV_ACTUAL=%%B"
if not "%UV_ACTUAL%"=="%UV_VERSION%" goto :ensure_uv_ps
move /Y "%UV_CANDIDATE%" "%~dp0bin\uv.exe" >nul
if errorlevel 1 goto :ensure_uv_ps
call :cleanup_uv_download
exit /b 0

:ensure_uv_ps
REM Fallback for pre-1803 Windows or a failed native tool. PowerShell downloads the same exact archive and
REM checksum into staging, verifies and probes it, then publishes it. The old uv remains untouched until
REM every check passes. Reset PSModulePath so a pwsh-polluted environment cannot break Windows PowerShell.
call :cleanup_uv_download
set "UV_DOWNLOAD=%~dp0bin\.uv-download-%RANDOM%.zip"
set "UV_CHECKSUM=%UV_DOWNLOAD%.sha256"
set "UV_STAGE=%~dp0bin\.uv-stage-%RANDOM%"
echo Native uv download unavailable; retrying with verified PowerShell download...
set "PSModulePath=%SystemRoot%\System32\WindowsPowerShell\v1.0\Modules"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing -Uri ($env:UV_URL + '.sha256') -OutFile $env:UV_CHECKSUM; Invoke-WebRequest -UseBasicParsing -Uri $env:UV_URL -OutFile $env:UV_DOWNLOAD; $expected=((Get-Content -Raw $env:UV_CHECKSUM).Trim() -split '\s+')[0]; if ($expected -notmatch '^[0-9a-fA-F]{64}$') { throw 'Malformed uv checksum.' }; $actual=(Get-FileHash -Algorithm SHA256 $env:UV_DOWNLOAD).Hash; if ($actual -ne $expected) { throw 'uv checksum mismatch.' }; New-Item -ItemType Directory -Force $env:UV_STAGE | Out-Null; Expand-Archive -Force $env:UV_DOWNLOAD $env:UV_STAGE; $candidate=Get-ChildItem $env:UV_STAGE -Filter uv.exe -Recurse | Select-Object -First 1; if ($null -eq $candidate) { throw 'uv.exe missing from archive.' }; Unblock-File $candidate.FullName; $reported=(& $candidate.FullName --version); if (($reported -split '\s+')[1] -ne $env:UV_VERSION) { throw ('uv version mismatch: ' + $reported) }; $target=Join-Path (Split-Path $env:UV_DOWNLOAD -Parent) 'uv.exe'; Move-Item -Force $candidate.FullName $target"
if errorlevel 1 goto :uv_download_failed
call :cleanup_uv_download
exit /b 0

:uv_download_failed
call :cleanup_uv_download
echo.
echo ERROR: Could not install uv (the package manager).
echo   - Confirm GitHub Releases is reachable (proxy/firewall?).
echo   - Or place a uv.exe in "%~dp0bin" and re-run.
exit /b 1

:cleanup_uv_download
if defined UV_DOWNLOAD del "%UV_DOWNLOAD%" >nul 2>&1
if defined UV_CHECKSUM del "%UV_CHECKSUM%" >nul 2>&1
if defined UV_STAGE if exist "%UV_STAGE%" rmdir /s /q "%UV_STAGE%" >nul 2>&1
exit /b 0
