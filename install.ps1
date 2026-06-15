# One-line installer for the AI Horde Worker (Windows).
#
#   irm https://raw.githubusercontent.com/Haidra-Org/horde-worker-reGen/main/install.ps1 | iex
#
# Downloads the latest release, builds the environment with uv (no pre-installed Python or git needed),
# seeds the config, and opens the dashboard in your browser. Re-running it updates in place.
#
# Options come from environment variables (so they work with the irm | iex form):
#   $env:HORDE_WORKER_DIR       install location (default: .\HordeWorker in the current directory)
#   $env:HORDE_WORKER_BACKEND   cu126 | cu130 | cu132 | cpu (default: detected from the GPU driver)
#   $env:HORDE_WORKER_NO_LAUNCH set to skip auto-launching the dashboard after install

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

# Default into a named subfolder of the current directory, not the home drive: the worker plus its model
# downloads run to many GB, so installing onto whatever drive the user cd'd to (and chose to run this from)
# is far less surprising than quietly filling up %LOCALAPPDATA%. A subfolder (rather than the bare CWD)
# keeps the loose-file bundle self-contained and avoids overwriting unrelated files already sitting there.
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
Invoke-WebRequest -Uri $ReleaseUrl -OutFile $tmpZip -UseBasicParsing

Write-Host "Extracting..."
Expand-Archive -Path $tmpZip -DestinationPath $InstallDir -Force
Remove-Item $tmpZip -Force

# GPU backend: explicit override wins; otherwise detect the hardware. We never silently fall back to
# the CPU build when a GPU is present, because that is ~100x slower and just looks like the worker is
# "broken" to someone who does not know to check. (ROCm is Linux-only; DirectML is temporarily removed.)
# The hardware check is shared with the graphical installer via detect-backend.ps1 (shipped in the
# bundle we just extracted), so both installers protect users with the exact same logic. We run it in a
# child PowerShell with -ExecutionPolicy Bypass so a locked-down machine policy cannot block it.
$Backend = Get-Option "HORDE_WORKER_BACKEND" ""
if ($Backend) {
    Write-Host "GPU backend: $Backend (from HORDE_WORKER_BACKEND)"
} else {
    $detectScript = Join-Path $InstallDir "detect-backend.ps1"
    $detected = (& powershell -NoProfile -ExecutionPolicy Bypass -File $detectScript).Trim()
    if (-not $detected) {
        # Fail loud rather than silently choosing the CPU build, which would be the wrong call on a GPU box.
        Write-Error "GPU detection failed (detect-backend.ps1 produced no result). Set `$env:HORDE_WORKER_BACKEND to a CUDA build (cu126/cu130/cu132) or 'cpu' and re-run."
        exit 1
    }
    if ($detected -match '^cu\d+$') {
        # detect-backend.ps1 picks the CUDA build from the driver's max CUDA version (cu130 on a
        # CUDA 13+ driver, otherwise cu126). Accept the whole cu* family so a new build flows through
        # without editing this installer.
        $Backend = $detected
        Write-Host "GPU backend: $detected (NVIDIA GPU detected)"
    } elseif ($detected -eq "amd-unsupported") {
        Write-Error @'
An AMD GPU was detected, but Windows GPU acceleration is currently unavailable
(DirectML is temporarily removed and ROCm is Linux-only). Installing now would use the
CPU build, which is roughly 100x slower.

If you understand that and still want to run on CPU, re-run with:
    $env:HORDE_WORKER_BACKEND = 'cpu'
On Linux, an AMD card can use the ROCm build instead.
'@
        exit 1
    } elseif ($detected -eq "cpu") {
        Write-Host "No NVIDIA or AMD GPU detected; using the CPU build." -ForegroundColor Yellow
        Write-Host "CPU is roughly 100x slower than a GPU and is mainly useful for testing." -ForegroundColor Yellow
        Write-Host "If you do have an NVIDIA GPU, install its drivers and re-run, or set `$env:HORDE_WORKER_BACKEND='cu126'." -ForegroundColor Yellow
        $Backend = "cpu"
    } else {
        Write-Error "GPU detection returned an unrecognized token '$detected'. Set `$env:HORDE_WORKER_BACKEND to a CUDA build (cu126/cu130/cu132) or 'cpu' and re-run."
        exit 1
    }
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
# Hand the resolved build to update-runtime via the env var it already honours. This carries any
# CUDA build (cu126/cu128/cu130), not just cpu, so a detected cu130 is not silently downgraded to the
# update-runtime default.
$env:HORDE_WORKER_BACKEND = $Backend
$updateRuntime = Join-Path $InstallDir "update-runtime.cmd"
& $updateRuntime
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
