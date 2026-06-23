@echo off
REM One-touch full in-place upgrade for the AI Horde Worker (Windows).
REM
REM Delegates to the bootstrap brain's `update` action: it queries the latest release, downloads the
REM release bundle directly, verifies its published SHA-256 BEFORE writing anything, then overlays the
REM worker's Python source and lockfile and re-syncs dependencies. The peered data dir (this folder's
REM -data sibling holding the uv cache, managed Python, and models) is left untouched, so an upgrade never
REM re-downloads models. The shell shims (these .cmd/.sh launchers) are intentionally not overwritten, so
REM the script driving this update is never modified out from under itself.
REM
REM This replaces the previous "pipe the remote installer into PowerShell" approach: nothing is executed
REM from the network, and the download is integrity-checked before it touches the install. For a deps-only
REM re-sync without fetching a new release, use update-runtime.cmd instead.
cd /d "%~dp0"
call "%~dp0runtime.cmd" update --yes %*
set "RC=%errorlevel%"
REM When the launcher bootstraps on first run it flows straight into starting the worker; only pause when
REM a user ran this script directly (and interactively).
if not defined HORDE_WORKER_FROM_LAUNCHER if not defined HORDE_WORKER_NONINTERACTIVE pause
exit /b %RC%
