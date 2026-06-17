@echo off
REM One-touch full in-place upgrade for the AI Horde Worker (Windows).
REM
REM Re-runs the official one-line installer against THIS folder: it downloads the latest release, extracts
REM it over the worker source, and re-syncs dependencies. The peered data dir (this folder's -data sibling
REM holding the uv cache, managed Python, and models) is left untouched, so an upgrade never re-downloads
REM models. For a deps-only re-sync without fetching a new release, use update-runtime.cmd instead.
REM
REM The installer is run from memory (irm | iex), never from a copy inside this folder, because the upgrade
REM overwrites this folder's files; piping the freshly downloaded script avoids overwriting a running one.

REM Point the installer at this install (full path, no trailing backslash), accept the notice (consent was
REM recorded at first install), and do not auto-launch the dashboard after the upgrade.
for %%I in ("%~dp0.") do set "HORDE_WORKER_DIR=%%~fI"
set "HORDE_WORKER_ASSUME_YES=1"
set "HORDE_WORKER_NO_LAUNCH=1"

powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/tazlin/horde-worker-reGen/main/install.ps1 | iex"
exit /b %errorlevel%
