# One-line installer for the AI Horde Worker (Windows).
#
#   irm https://raw.githubusercontent.com/Haidra-Org/horde-worker-reGen/main/install.ps1 | iex
#
# Downloads the latest release, then hands off to the bundled runtime.cmd, which installs uv and runs the
# Python bootstrap (GPU detection, dependency sync, config seeding, launch). No pre-installed Python or git
# needed. Re-running it updates in place.
#
# Options come from environment variables (so they work with the irm | iex form):
#   $env:HORDE_WORKER_DIR        install location (default: .\HordeWorker in the current directory)
#   $env:HORDE_WORKER_BACKEND    cu126 | cu130 | cu132 | cpu (default: detected from the GPU driver)
#   $env:HORDE_WORKER_NO_LAUNCH  set to skip auto-launching the dashboard after install

#Requires -Version 5.1
$ErrorActionPreference = "Stop"

# Windows PowerShell 5.1 defaults to TLS 1.0/1.1; GitHub requires 1.2+, so opt in before downloading.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$Owner = "tazlin"
$Repo = "horde-worker-reGen"
$Asset = "horde-worker-reGen.zip"
$ReleaseUrl = "https://github.com/$Owner/$Repo/releases/latest/download/$Asset"

function Get-Option([string]$Name, [string]$Default) {
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ($value) { return $value }
    return $Default
}

function New-Shortcut([string]$LinkPath, [string]$TargetPath, [string]$WorkingDir) {
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($LinkPath)
    $shortcut.TargetPath = $TargetPath
    $shortcut.WorkingDirectory = $WorkingDir
    $shortcut.Description = "AI Horde Worker dashboard"
    $shortcut.Save()
}

function Expand-ReleaseZip([string]$ZipPath, [string]$Destination) {
    # Prefer in-box curl.exe + tar.exe (Windows 10 1803+): tar unpacks a .zip on Windows and, unlike
    # Expand-Archive, needs no PowerShell module to autoload (which can fail under a pwsh-polluted
    # PSModulePath). Fall back to Expand-Archive on older Windows that lacks tar.
    $tar = Join-Path $env:SystemRoot "System32\tar.exe"
    if (Test-Path $tar) {
        & $tar -xf $ZipPath -C $Destination
        if ($LASTEXITCODE -eq 0) { return }
    }
    Expand-Archive -Path $ZipPath -DestinationPath $Destination -Force
}

# Default into a named subfolder of the current directory, not the home drive: the worker plus its model
# downloads run to many GB, so installing onto whatever drive the user cd'd to is far less surprising than
# quietly filling up %LOCALAPPDATA%. A subfolder keeps the bundle self-contained.
$InstallDir = Get-Option "HORDE_WORKER_DIR" (Join-Path (Get-Location).Path "HordeWorker")
if ($args.Count -ge 1 -and $args[0]) { $InstallDir = [string]$args[0] }
if ($InstallDir -match "\s") {
    Write-Error "The install path must not contain spaces (PyTorch and uv dislike them): $InstallDir"
    exit 1
}

Write-Host ""
Write-Host "=== AI Horde Worker installer ===" -ForegroundColor Cyan
Write-Host "Install location: $InstallDir"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

$tmpZip = Join-Path ([System.IO.Path]::GetTempPath()) ("horde-worker-" + [System.Guid]::NewGuid().ToString() + ".zip")
Write-Host "Downloading the latest release..."
$curl = Join-Path $env:SystemRoot "System32\curl.exe"
if (Test-Path $curl) {
    & $curl -fL --retry 3 -o $tmpZip $ReleaseUrl
    if ($LASTEXITCODE -ne 0) { Write-Error "Download failed (curl exit $LASTEXITCODE) from $ReleaseUrl"; exit 1 }
} else {
    Invoke-WebRequest -Uri $ReleaseUrl -OutFile $tmpZip -UseBasicParsing
}

Write-Host "Extracting..."
Expand-ReleaseZip -ZipPath $tmpZip -Destination $InstallDir
Remove-Item $tmpZip -Force

# Everything else (install uv, detect the GPU, seed bridgeData.yaml, sync dependencies) is the bootstrap's
# job now, so the exact same logic runs for the one-liner, the graphical installer and winget. runtime.cmd
# installs uv via in-box curl/tar (no fragile nested PowerShell) and then runs bootstrap.py. We pass
# --no-launch and start the dashboard ourselves below, after creating shortcuts. A pre-set
# $env:HORDE_WORKER_BACKEND still overrides detection (e.g. 'cpu' to opt into a CPU-only AMD install).
Write-Host "Setting up the environment. The first run downloads Python and PyTorch and can take several minutes..."
$env:HORDE_WORKER_NONINTERACTIVE = "1"
& (Join-Path $InstallDir "runtime.cmd") install --no-launch
if ($LASTEXITCODE -ne 0) {
    Write-Error "Environment setup failed (see the output above). Deleting the .venv folder and re-running often helps."
    exit 1
}
# Trust the artifact, not just the exit code: a real install must have produced a virtual environment.
if (-not (Test-Path (Join-Path $InstallDir ".venv"))) {
    Write-Error "Environment setup did not produce a .venv. See the output above; delete .venv and re-run."
    exit 1
}

Write-Host ""
Write-Host "Installation complete." -ForegroundColor Green
Write-Host "Installed at: $InstallDir"

# Create shortcuts so reopening the dashboard later is one click. Per-user only, opt-out via
# HORDE_WORKER_NO_SHORTCUTS; best-effort.
$launcher = Join-Path $InstallDir "horde-worker.cmd"
$madeShortcut = $false
if (Get-Option "HORDE_WORKER_NO_SHORTCUTS" "") {
    Write-Host "Skipping shortcut creation (HORDE_WORKER_NO_SHORTCUTS is set)."
} else {
    Write-Host "Creating 'AI Horde Worker' shortcuts (Desktop + Start Menu; set HORDE_WORKER_NO_SHORTCUTS to skip)..."
    foreach ($shortcutDir in @([Environment]::GetFolderPath("Programs"), [Environment]::GetFolderPath("Desktop"))) {
        if (-not $shortcutDir) { continue }
        try {
            New-Shortcut -LinkPath (Join-Path $shortcutDir "AI Horde Worker.lnk") -TargetPath $launcher -WorkingDir $InstallDir
            $madeShortcut = $true
        } catch {
            Write-Host "Note: could not create a shortcut in $shortcutDir ($($_.Exception.Message))."
        }
    }
}

Write-Host ""
Write-Host "To open the dashboard again later:" -ForegroundColor Cyan
if ($madeShortcut) {
    Write-Host "  - click the 'AI Horde Worker' shortcut on your Desktop or in the Start Menu, or"
}
Write-Host "  - run horde-worker.cmd in $InstallDir."
Write-Host "To update later: re-run the same install command (or 'winget upgrade Haidra.HordeWorker')."
Write-Host ""

if (Get-Option "HORDE_WORKER_NO_LAUNCH" "") {
    Write-Host "Start it whenever you're ready using a shortcut above."
} else {
    Write-Host "Opening the worker dashboard in your browser now..."
    Start-Process -FilePath $launcher -WorkingDirectory $InstallDir
}
