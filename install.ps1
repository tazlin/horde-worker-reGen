# One-line installer for the AI Horde Worker (Windows).
#
#   irm https://raw.githubusercontent.com/Haidra-Org/horde-worker-reGen/main/install.ps1 | iex
#
# Downloads the latest release, shows a notice of what will be installed and from where, asks for
# confirmation, then hands off to the bundled runtime.cmd, which installs uv and runs the Python bootstrap
# (GPU detection, dependency sync, config seeding, launch). It provides its own private Python; for git it
# uses an existing git if you have one, and otherwise fetches a portable git on Windows. Re-running it
# updates in place.
#
# Options come from environment variables (so they work with the irm | iex form):
#   $env:HORDE_WORKER_DIR         install location (default: .\HordeWorker in the current directory)
#   $env:HORDE_WORKER_REPO        install from a fork (owner/repo; default Haidra-Org/horde-worker-reGen)
#   $env:HORDE_WORKER_BACKEND     cu126 | cu130 | cu132 | rocm-windows | cpu
#                                  (default: detected from the GPU driver)
#   $env:HORDE_WORKER_FEATURES    optional feature extras: comma/space list of post-processing, controlnet,
#                                 or 'none' (default: all on NVIDIA/CPU, none on other backends)
#   $env:HORDE_WORKER_VERSION     install a specific release tag (e.g. v1.2.3) instead of the latest release
#   $env:HORDE_WORKER_ASSUME_YES  accept the install notice without prompting (required when non-interactive)
#   $env:HORDE_WORKER_SHORTCUTS   create Desktop/Start Menu shortcuts without prompting
#   $env:HORDE_WORKER_NO_SHORTCUTS skip shortcut creation entirely
#   $env:HORDE_WORKER_NO_LAUNCH   skip the "Start now?" prompt and do not launch after install

#Requires -Version 5.1
$ErrorActionPreference = "Stop"

# Windows PowerShell 5.1 defaults to TLS 1.0/1.1; GitHub requires 1.2+, so opt in before downloading.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

# The owner/repo to install from. Defaults to the canonical production repo; a fork overrides it by setting
# HORDE_WORKER_REPO (e.g. baked into its own one-liner) rather than editing this file, so the committed
# default never diverges from upstream. The resolved value is recorded in bin/install-info, so the in-place
# self-updater pulls future releases from the same origin (see worker_bootstrap/updater.py resolve_update_repo).
$RepoSlug = [Environment]::GetEnvironmentVariable("HORDE_WORKER_REPO")
if (-not $RepoSlug) { $RepoSlug = "Haidra-Org/horde-worker-reGen" }
if ($RepoSlug -notmatch "^[^/]+/[^/]+$") { Write-Error "HORDE_WORKER_REPO must be 'owner/repo' (got '$RepoSlug')."; exit 1 }
$Owner, $Repo = $RepoSlug.Split("/", 2)
$Asset = "horde-worker-reGen.zip"
$VersionTag = [Environment]::GetEnvironmentVariable("HORDE_WORKER_VERSION")
if ($VersionTag -and $VersionTag -match "[\s/\\]") {
    Write-Error "HORDE_WORKER_VERSION must be a single release tag with no spaces or slashes (got '$VersionTag')."
    exit 1
}
if ($VersionTag) {
    $ReleaseUrl = "https://github.com/$Owner/$Repo/releases/download/$VersionTag/$Asset"
} else {
    $ReleaseUrl = "https://github.com/$Owner/$Repo/releases/latest/download/$Asset"
}

function Get-Option([string]$Name, [string]$Default) {
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ($value) { return $value }
    return $Default
}

function Read-YesNo([string]$Prompt, [bool]$DefaultYes = $false) {
    $suffix = if ($DefaultYes) { "[Y/n]" } else { "[y/N]" }
    $answer = (Read-Host "$Prompt $suffix").Trim().ToLower()
    if (-not $answer) { return $DefaultYes }
    return ($answer -eq "y" -or $answer -eq "yes")
}

function Read-LaunchChoice([string]$Prompt) {
    while ($true) {
        $answer = (Read-Host $Prompt).Trim().ToLower()
        switch ($answer) {
            { $_ -in "y", "yes" }       { return "web" }
            { $_ -in "n", "no", "" }    { return "no" }
            "t"                          { return "terminal" }
            "h"                          { return "headless" }
            default {
                Write-Host "Please enter y, n, t, or h." -ForegroundColor Yellow
            }
        }
    }
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

# Guard against running the installer from inside an existing horde-worker installation.
# When no explicit destination was given, the default (.\HordeWorker) would create a
# nested copy. Presence of runtime.cmd in the CWD is a reliable sentinel for an existing install.
if (-not (Get-Option "HORDE_WORKER_DIR" "") -and $args.Count -eq 0 -and (Test-Path (Join-Path (Get-Location).Path "runtime.cmd"))) {
    Write-Host "ERROR: the current directory looks like an existing horde-worker installation (runtime.cmd is here)." -ForegroundColor Red
    Write-Host "       Installing from here would create a nested copy at: $InstallDir" -ForegroundColor Red
    Write-Host "       To update the current install, run:  update.cmd" -ForegroundColor Yellow
    Write-Host "       To install elsewhere, cd to another directory first, or set:" -ForegroundColor Yellow
    Write-Host "         `$env:HORDE_WORKER_DIR = 'C:\path\to\new-location'" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "=== AI Horde Worker installer ===" -ForegroundColor Cyan
Write-Host "Install location: $InstallDir"

# Download and extract the release into a temp dir, then lay it down and overlay it onto the install with
# the same mirror-pruning logic the self-updater uses (runtime.cmd apply-bundle, below). Expanding straight
# into the install only overwrites files and never removes ones the new release dropped, so a reinstall over
# an older version would leave a renamed/removed module on the import path to shadow the new code.
$tmpDir = Join-Path ([System.IO.Path]::GetTempPath()) ("horde-worker-" + [System.Guid]::NewGuid().ToString())
$bundleDir = Join-Path $tmpDir "bundle"
New-Item -ItemType Directory -Force -Path $bundleDir | Out-Null
$tmpZip = Join-Path $tmpDir "horde-worker.zip"

Write-Host "Downloading the latest release..."
$curl = Join-Path $env:SystemRoot "System32\curl.exe"
if (Test-Path $curl) {
    & $curl -fL --retry 3 -o $tmpZip $ReleaseUrl
    if ($LASTEXITCODE -ne 0) { Write-Error "Download failed (curl exit $LASTEXITCODE) from $ReleaseUrl"; exit 1 }
} else {
    Invoke-WebRequest -Uri $ReleaseUrl -OutFile $tmpZip -UseBasicParsing
}

Write-Host "Extracting..."
Expand-ReleaseZip -ZipPath $tmpZip -Destination $bundleDir

# Lay the bundle down (shims, bootstrap, source). This overwrites in place but, like a plain expand, does
# not by itself prune modules the new release dropped; the apply-bundle overlay below removes those. The
# shims must be written here, not by the overlay, so the overlay can run *through* runtime.cmd without
# overwriting the running launcher mid-run.
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Copy-Item -Path (Join-Path $bundleDir "*") -Destination $InstallDir -Recurse -Force

# Record how this worker was installed and from where, so the in-place self-updater pulls future releases
# from the same origin (this fork/account) rather than a hardcoded default. Lives under bin/ (preserved
# across updates, removed on uninstall). Written without a BOM so the bootstrap parses the first key cleanly.
$binDir = Join-Path $InstallDir "bin"
New-Item -ItemType Directory -Force -Path $binDir | Out-Null
[System.IO.File]::WriteAllText((Join-Path $binDir "install-info"), "method=one-line`nrepo=$Owner/$Repo`n")

# Show what is about to be installed (and from where) and get consent before any heavy download. We do it
# here, where the user is at a console, then pass HORDE_WORKER_ASSUME_YES so the bootstrap does not prompt
# again. Honour a pre-set HORDE_WORKER_ASSUME_YES for unattended installs.
$noticePath = Join-Path $InstallDir "INSTALL_NOTICE.txt"
if (Test-Path $noticePath) {
    Write-Host ""
    Get-Content $noticePath | ForEach-Object { Write-Host $_ }
    Write-Host ""
}
if (-not (Get-Option "HORDE_WORKER_ASSUME_YES" "")) {
    if ([Environment]::UserInteractive) {
        if (-not (Read-YesNo "Proceed with installation?")) {
            Write-Host "Installation cancelled. The downloaded files are in $InstallDir; delete that folder to remove them."
            exit 1
        }
        $env:HORDE_WORKER_ASSUME_YES = "1"
    } else {
        Write-Error "This is a non-interactive session, so it cannot ask you to accept the notice above. Re-run with `$env:HORDE_WORKER_ASSUME_YES='1' to accept it, or use the graphical installer (HordeWorker-Setup.exe)."
        exit 1
    }
}

# Overlay the freshly downloaded bundle through the shared, mirror-pruning overlay (same as the self-updater)
# so a reinstall prunes modules renamed/removed since the installed version, while .venv, bin, models, and
# bridgeData.yaml are preserved. This also fetches uv (into bin\), which the dependency sync below reuses.
& (Join-Path $InstallDir "runtime.cmd") apply-bundle "$bundleDir"
if ($LASTEXITCODE -ne 0) { Write-Error "Could not lay down the worker files (see the output above)."; exit 1 }
Remove-Item -Recurse -Force $tmpDir -ErrorAction SilentlyContinue

# Everything else (install uv, detect the GPU, seed bridgeData.yaml, sync dependencies) is the bootstrap's
# job now, so the exact same logic runs for the one-liner, the graphical installer and winget. runtime.cmd
# installs uv via in-box curl/tar (no fragile nested PowerShell) and then runs bootstrap.py. We pass
# --no-launch and start the dashboard ourselves below, after creating shortcuts. A pre-set
# $env:HORDE_WORKER_BACKEND still overrides detection (e.g. 'cpu' for CPU-only, or 'rocm-windows' for AMD).
Write-Host "Setting up the environment. The first run downloads Python and PyTorch and can take several minutes..."
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
Write-Host "Models, the uv cache, and Python live in $InstallDir-data (a sibling folder)."
Write-Host "That data folder is preserved if you delete or reinstall the worker folder, so your models are"
Write-Host "not lost. Set `$env:HORDE_WORKER_DATA_DIR before installing to put it elsewhere (e.g. another drive)."

# Shortcuts are opt-in (conservative default): we ask, defaulting to No. HORDE_WORKER_SHORTCUTS creates
# them without asking (for unattended installs); HORDE_WORKER_NO_SHORTCUTS skips entirely. Per-user only,
# best-effort.
$launcher = Join-Path $InstallDir "horde-worker.cmd"
$madeShortcut = $false
$wantShortcuts = $false
if (Get-Option "HORDE_WORKER_NO_SHORTCUTS" "") {
    Write-Host "Skipping shortcut creation (HORDE_WORKER_NO_SHORTCUTS is set)."
} elseif (Get-Option "HORDE_WORKER_SHORTCUTS" "") {
    $wantShortcuts = $true
} elseif ([Environment]::UserInteractive) {
    $wantShortcuts = Read-YesNo "Create 'AI Horde Worker' shortcuts on your Desktop and Start Menu?"
}
if ($wantShortcuts) {
    Write-Host "Creating 'AI Horde Worker' shortcuts (Desktop + Start Menu)..."
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
Write-Host "  - run horde-worker.cmd in $InstallDir (add --terminal for the in-terminal UI, --headless for no UI)."
Write-Host "To update later: run update.cmd in $InstallDir, or re-run the same install command"
Write-Host "  (either keeps your $InstallDir-data folder intact)."
Write-Host ""

if (Get-Option "HORDE_WORKER_NO_LAUNCH" "") {
    Write-Host "Start it whenever you're ready using a shortcut above."
} elseif ([Environment]::UserInteractive) {
    $choice = Read-LaunchChoice "Start the worker now? [(y)es / (n)o / (t)erminal UI / (h)eadless]"
    switch ($choice) {
        "web" {
            Write-Host "Opening the worker dashboard in your browser now..."
            Start-Process -FilePath $launcher -WorkingDirectory $InstallDir
        }
        "terminal" {
            Write-Host "Starting the in-terminal UI..."
            & $launcher --terminal
        }
        "headless" {
            Write-Host "Starting the worker in headless mode..."
            Start-Process -FilePath $launcher -ArgumentList "--headless" -WorkingDirectory $InstallDir -WindowStyle Hidden
        }
        default {
            Write-Host "Start it whenever you're ready using a shortcut above."
        }
    }
} else {
    Write-Host "Start it whenever you're ready using a shortcut above."
}
