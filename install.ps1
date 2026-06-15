# One-line installer for the AI Horde Worker (Windows).
#
#   irm https://raw.githubusercontent.com/Haidra-Org/horde-worker-reGen/main/install.ps1 | iex
#
# Downloads the latest release, builds the environment with uv (no pre-installed Python or git needed),
# seeds the config, and opens the dashboard in your browser. Re-running it updates in place.
#
# Options come from environment variables (so they work with the irm | iex form):
#   $env:HORDE_WORKER_DIR       install location (default: %LOCALAPPDATA%\HordeWorker)
#   $env:HORDE_WORKER_BACKEND   cu128 | cpu (default: cu128 if an NVIDIA GPU is detected, else cpu)
#   $env:HORDE_WORKER_NO_LAUNCH set to skip auto-launching the dashboard after install

#Requires -Version 5.1
$ErrorActionPreference = "Stop"

# Windows PowerShell 5.1 defaults to TLS 1.0/1.1; GitHub requires 1.2+, so opt in before downloading.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$Owner = "Haidra-Org"
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

function Test-NvidiaGpu {
    # nvidia-smi on PATH is the happy path, but a freshly installed driver often does not add it to
    # PATH. Fall back to its default location and to the video-controller list so we do not mistake a
    # real NVIDIA card for a CPU-only machine and quietly install the (~100x slower) CPU build.
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { return $true }
    if (Test-Path (Join-Path $env:SystemRoot "System32\nvidia-smi.exe")) { return $true }
    try {
        foreach ($c in Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop) {
            if ($c.Name -match "NVIDIA") { return $true }
        }
    } catch { }
    return $false
}

function Test-AmdGpu {
    try {
        foreach ($c in Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop) {
            if ($c.Name -match "AMD|Radeon") { return $true }
        }
    } catch { }
    return $false
}

$InstallDir = Get-Option "HORDE_WORKER_DIR" (Join-Path $env:LOCALAPPDATA "HordeWorker")
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
Invoke-WebRequest -Uri $ReleaseUrl -OutFile $tmpZip -UseBasicParsing

Write-Host "Extracting..."
Expand-Archive -Path $tmpZip -DestinationPath $InstallDir -Force
Remove-Item $tmpZip -Force

# GPU backend: explicit override wins; otherwise detect the hardware. We never silently fall back to
# the CPU build when a GPU is present, because that is ~100x slower and just looks like the worker is
# "broken" to someone who does not know to check. (ROCm is Linux-only; DirectML is temporarily removed.)
$Backend = Get-Option "HORDE_WORKER_BACKEND" ""
if ($Backend) {
    Write-Host "GPU backend: $Backend (from HORDE_WORKER_BACKEND)"
} elseif (Test-NvidiaGpu) {
    $Backend = "cu128"
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        Write-Host "Note: an NVIDIA GPU was detected but 'nvidia-smi' is not on PATH; using the CUDA build anyway." -ForegroundColor Yellow
    }
    Write-Host "GPU backend: cu128 (NVIDIA GPU detected)"
} elseif (Test-AmdGpu) {
    Write-Error @'
An AMD GPU was detected, but Windows GPU acceleration is currently unavailable
(DirectML is temporarily removed and ROCm is Linux-only). Installing now would use the
CPU build, which is roughly 100x slower.

If you understand that and still want to run on CPU, re-run with:
    $env:HORDE_WORKER_BACKEND = 'cpu'
On Linux, an AMD card can use the ROCm build instead.
'@
    exit 1
} else {
    Write-Host "No NVIDIA or AMD GPU detected; using the CPU build." -ForegroundColor Yellow
    Write-Host "CPU is roughly 100x slower than a GPU and is mainly useful for testing." -ForegroundColor Yellow
    Write-Host "If you do have an NVIDIA GPU, install its drivers and re-run, or set `$env:HORDE_WORKER_BACKEND='cu128'." -ForegroundColor Yellow
    $Backend = "cpu"
}

# Seed the config from the template on a fresh install (never clobbers an existing bridgeData.yaml).
$bridge = Join-Path $InstallDir "bridgeData.yaml"
$template = Join-Path $InstallDir "bridgeData_template.yaml"
if ((-not (Test-Path $bridge)) -and (Test-Path $template)) {
    Copy-Item $template $bridge
}

# Co-locate uv's package cache with the install so it lands on the chosen drive (not the home drive) and
# stays on the same volume as .venv for hardlinking. update-runtime.cmd applies the same default; setting it
# here too makes the decision visible at the entry point. Respect a user-set UV_CACHE_DIR.
if (-not $env:UV_CACHE_DIR) {
    $env:UV_CACHE_DIR = Join-Path (Resolve-Path $InstallDir).Path "bin\uv_cache"
}

Write-Host "Setting up the environment. The first run downloads Python and PyTorch and can take several minutes..."
$env:HORDE_WORKER_NONINTERACTIVE = "1"
$updateRuntime = Join-Path $InstallDir "update-runtime.cmd"
if ($Backend -eq "cpu") {
    & $updateRuntime "--cpu"
} else {
    & $updateRuntime
}
if ($LASTEXITCODE -ne 0) {
    Write-Error "Environment setup failed. See the output above; deleting the .venv folder and re-running often helps."
    exit 1
}

Write-Host ""
Write-Host "Installation complete." -ForegroundColor Green
Write-Host "Installed at: $InstallDir"

# Create shortcuts so reopening the dashboard later is one click. These are per-user (Start Menu /
# Desktop) only, never system-wide, and opt-out via HORDE_WORKER_NO_SHORTCUTS; best-effort either way.
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
