#requires -Version 5.1
<#
.SYNOPSIS
  Build the AI Horde Worker release bundle (zip) and the graphical Windows installer
  locally, mirroring .github/workflows/release.yml so the packaging and installer schemes
  can be verified on a developer machine without pushing a tag or invoking GitHub CI.

.DESCRIPTION
  Replicates, in order, what release.yml does:
    1. (optional) Guard against a leftover local 'path =' source in pyproject.toml.
    2. Stage the bundle from packaging/bundle-include.txt plus the horde_worker_regen package.
    3. Pack: zip the stage, write SHA256SUMS, verify uv.lock ships.
    4. Installer: locate ISCC (any installed Inno Setup version, any drive) and compile
       packaging/inno/HordeWorker.iss into HordeWorker-Setup.exe.
    5. (optional, -SmokeTest) install the produced .exe silently into a temp folder, assert
       the file layout and detected backend, then uninstall.

  Outputs land at the repo root (stage/, horde-worker-reGen.zip, SHA256SUMS) and in dist/
  (HordeWorker-Setup.exe). All of these are gitignored.

  Targets Windows PowerShell 5.1 (the default on a clean Windows box and what install.ps1
  assumes), so it runs without installing PowerShell 7.

.PARAMETER Version
  Version baked into the installer (MyAppVersion). Default: 0.0.0-dev.

.PARAMETER Iscc
  Full path to ISCC.exe. If omitted: the HORDE_ISCC env var, then PATH, then every fixed
  drive's "Program Files*\Inno Setup *\ISCC.exe" are searched (highest match wins).

.PARAMETER SkipInstaller
  Stage and pack only. Inno Setup is not required.

.PARAMETER SkipPack
  Stage and build the installer only (no zip / SHA256SUMS).

.PARAMETER StrictGuard
  Make the pyproject local-path-source guard fatal, exactly as CI is. Default: warn only,
  so the installer scheme can still be verified while the dev checkout is pinned to a local
  hordelib for engine work.

.PARAMETER SmokeTest
  After building, install the .exe silently into a temp folder, verify it, then uninstall.
  Shortcuts are opt-in (unchecked tasks), so a silent install creates none. Cannot be combined
  with -SkipInstaller.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File packaging\build-local.ps1
  Full local mirror of CI: stage + zip + installer.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File packaging\build-local.ps1 -SkipPack -SmokeTest
  Build just the installer and exercise a real silent install/uninstall cycle.
#>
[CmdletBinding()]
param(
    [string]$Version = '0.0.0-dev',
    [string]$Iscc,
    [switch]$SkipInstaller,
    [switch]$SkipPack,
    [switch]$StrictGuard,
    [switch]$SmokeTest
)

$ErrorActionPreference = 'Stop'

function Write-Step { param([string]$Message) Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Note { param([string]$Message) Write-Host "    $Message" -ForegroundColor Gray }
function Write-Warn { param([string]$Message) Write-Host "WARNING: $Message" -ForegroundColor Yellow }

function Get-Sha256Lower {
    # Computed via .NET rather than Get-FileHash so it does not depend on the
    # Microsoft.PowerShell.Utility module autoloading (which can fail when PSModulePath is
    # polluted, e.g. when this script is launched from a non-PowerShell shell).
    param([Parameter(Mandatory = $true)][string]$Path)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $stream = [System.IO.File]::OpenRead($Path)
        try { $bytes = $sha.ComputeHash($stream) } finally { $stream.Dispose() }
    } finally { $sha.Dispose() }
    return ([System.BitConverter]::ToString($bytes) -replace '-', '').ToLower()
}

function Find-Iscc {
    if ($env:HORDE_ISCC -and (Test-Path $env:HORDE_ISCC)) { return (Resolve-Path $env:HORDE_ISCC).Path }

    $onPath = Get-Command 'ISCC.exe' -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }

    $found = @()
    foreach ($drive in Get-PSDrive -PSProvider FileSystem -ErrorAction SilentlyContinue) {
        foreach ($pf in @('Program Files', 'Program Files (x86)')) {
            $glob = Join-Path $drive.Root (Join-Path $pf 'Inno Setup *\ISCC.exe')
            $found += Get-ChildItem -Path $glob -ErrorAction SilentlyContinue
        }
    }
    # Prefer the highest version directory (so "Inno Setup 7" beats "Inno Setup 6").
    $best = $found | Sort-Object { $_.Directory.Name } -Descending | Select-Object -First 1
    if ($best) { return $best.FullName }
    return $null
}

function Invoke-SmokeTest {
    param([Parameter(Mandatory = $true)][string]$Exe)

    Write-Step 'Smoke-testing the installer (silent install -> verify -> uninstall)'
    if (-not (Test-Path $Exe)) { throw "Installer not found: $Exe" }

    # PyTorch / uv reject paths with spaces, and so does the installer's destination-page check.
    # $env:TEMP usually has none; fall back to a short fixed path if it does.
    $base = $env:TEMP
    if ((-not $base) -or ($base -match ' ')) { $base = 'C:\hw-smoke' }
    $target = Join-Path $base ('AIHordeWorker-smoke-' + [System.Guid]::NewGuid().ToString('N').Substring(0, 8))
    $log = "$target.install.log"

    Write-Note "Installing to $target"
    $install = Start-Process -FilePath $Exe -Wait -PassThru -ArgumentList @(
        '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART',
        '/MERGETASKS=!desktopicon', "/DIR=$target", "/LOG=$log"
    )
    if ($install.ExitCode -ne 0) { throw "Silent install failed (exit $($install.ExitCode)); see $log" }

    $expected = @(
        'horde-worker.cmd', 'runtime.cmd', 'update-runtime.cmd',
        'bridgeData.yaml', 'detect-backend.ps1',
        'INSTALL_NOTICE.txt', 'THIRD-PARTY-NOTICES.md',
        'bootstrap.py', 'worker_bootstrap\cli.py',
        'horde_worker_regen', 'unins000.exe', 'bin\backend', 'bin\install-consent'
    )
    foreach ($item in $expected) {
        if (-not (Test-Path (Join-Path $target $item))) { throw "Expected installed item missing: $item" }
    }
    $backend = (Get-Content (Join-Path $target 'bin\backend') -Raw).Trim()
    Write-Note "Install OK. Persisted backend = '$backend'"
    $knownBackends = @('cu126', 'cu128', 'cu130', 'cu132', 'cpu', 'rocm', 'rocm-windows', 'rocm-gfx110x', 'rocm-gfx1151', 'rocm-gfx120x')
    if ($backend -notin $knownBackends) { Write-Warn "Unexpected backend token: '$backend'" }

    Write-Note 'Uninstalling'
    $unins = Start-Process -FilePath (Join-Path $target 'unins000.exe') -Wait -PassThru `
        -ArgumentList @('/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART')
    if ($unins.ExitCode -ne 0) { Write-Warn "Uninstaller exit code $($unins.ExitCode)" }

    # The uninstaller relaunches a copy of itself from %TEMP% and returns immediately; give it a
    # moment to release handles before removing the leftover dir (bridgeData.yaml is kept by design).
    Start-Sleep -Milliseconds 750
    if (Test-Path $target) { Remove-Item -Recurse -Force $target -ErrorAction SilentlyContinue }
    # The log lives beside the install dir, not inside it; on success it is just noise (it is kept
    # on failure, where the throw above points the user at it).
    if (Test-Path $log) { Remove-Item -Force $log -ErrorAction SilentlyContinue }
    Write-Step 'Smoke test passed.'
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $RepoRoot
try {
    $stageDir = Join-Path $RepoRoot 'stage'
    $distDir  = Join-Path $RepoRoot 'dist'
    $zipPath  = Join-Path $RepoRoot 'horde-worker-reGen.zip'
    $sumsPath = Join-Path $RepoRoot 'SHA256SUMS'
    $manifest = Join-Path $RepoRoot 'packaging\bundle-include.txt'
    $pkgDir   = Join-Path $RepoRoot 'horde_worker_regen'
    $issPath  = Join-Path $RepoRoot 'packaging\inno\HordeWorker.iss'

    # --- 1. Guard: a released bundle must not depend on an in-repo path source -------------
    Write-Step 'Checking pyproject.toml for a local path source'
    $pathSource = Select-String -Path (Join-Path $RepoRoot 'pyproject.toml') `
        -Pattern '^\s*[A-Za-z0-9_.-]+\s*=\s*\{[^}]*path\s*=' -ErrorAction SilentlyContinue
    if ($pathSource) {
        $detail = ($pathSource | ForEach-Object { $_.Line.Trim() }) -join "`n      "
        $msg = "pyproject.toml still has a local 'path =' source; a real release built from this " +
               "would fail 'uv sync' on a user machine:`n      $detail"
        if ($StrictGuard) { throw $msg }
        Write-Warn $msg
        Write-Warn 'Continuing anyway (this is fine for local install/pack verification; pass -StrictGuard to fail like CI).'
    } else {
        Write-Note 'No local path source. OK.'
    }

    # --- 2. Stage (mirror release.yml "Stage the runtime bundle") --------------------------
    Write-Step "Staging bundle into $stageDir"
    if (-not (Test-Path $pkgDir)) { throw "Package directory not found: $pkgDir" }
    if (Test-Path $stageDir) { Remove-Item -Recurse -Force $stageDir }
    New-Item -ItemType Directory -Path $stageDir | Out-Null

    foreach ($line in (Get-Content $manifest)) {
        $entry = $line.TrimEnd("`r").Trim()
        if ((-not $entry) -or $entry.StartsWith('#')) { continue }
        $matched = @(Get-ChildItem -Path (Join-Path $RepoRoot $entry) -ErrorAction SilentlyContinue)
        if (-not $matched) { Write-Warn "manifest entry matched no files: $entry"; continue }
        foreach ($file in $matched) { Copy-Item -Path $file.FullName -Destination $stageDir }
    }
    Copy-Item -Recurse -Path $pkgDir -Destination (Join-Path $stageDir 'horde_worker_regen')
    # The stdlib-only bootstrap brain that bootstrap.py imports (copied as a package, like the worker).
    $bootstrapPkg = Join-Path $RepoRoot 'worker_bootstrap'
    if (-not (Test-Path $bootstrapPkg)) { throw "Bootstrap package not found: $bootstrapPkg" }
    Copy-Item -Recurse -Path $bootstrapPkg -Destination (Join-Path $stageDir 'worker_bootstrap')

    # Drop build noise that CI also prunes so the zip matches the published one.
    Get-ChildItem -Path $stageDir -Recurse -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq '__pycache__' -or $_.Name -like '*.egg-info' } |
        ForEach-Object { Remove-Item -Recurse -Force $_.FullName -ErrorAction SilentlyContinue }
    Write-Note ("Staged {0} top-level entries." -f (Get-ChildItem $stageDir).Count)

    # --- 2b. Fetch available release versions for the installer's version picker -----------
    Write-Step 'Fetching release version list for the installer'
    $fetchScript = Join-Path $RepoRoot 'packaging\fetch-releases.ps1'
    $versionsFile = Join-Path $stageDir 'release-versions.txt'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $fetchScript `
        -Repo 'Haidra-Org/horde-worker-reGen' -OutFile $versionsFile -MaxAgeMinutes 60
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $versionsFile)) {
        Write-Warn 'Could not fetch release versions (network may be unavailable). The installer version picker will only offer the default choice.'
    } else {
        $count = (Get-Content $versionsFile | Where-Object { $_ -match '\S' }).Count
        Write-Note "Baked $count release tags into the installer stage."
    }

    # --- 3. Pack: zip + SHA256SUMS + lockfile check ----------------------------------------
    if (-not $SkipPack) {
        Write-Step "Packing $zipPath"
        if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
        Compress-Archive -Path (Join-Path $stageDir '*') -DestinationPath $zipPath
        $hash = Get-Sha256Lower -Path $zipPath
        # Two-space separator matches `sha256sum` output (what CI publishes).
        "$hash  horde-worker-reGen.zip" | Set-Content -Path $sumsPath -Encoding ascii
        Write-Note "SHA256: $hash"

        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $archive = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
        try {
            $hasLock = $archive.Entries | Where-Object { $_.FullName -eq 'uv.lock' -or $_.Name -eq 'uv.lock' }
        } finally { $archive.Dispose() }
        if (-not $hasLock) { throw "uv.lock is missing from the bundle; the installer's 'uv sync --locked' would fail." }
        Write-Note 'uv.lock present in bundle: OK'
    }

    # --- 4. Installer ----------------------------------------------------------------------
    if (-not $SkipInstaller) {
        if (-not $Iscc) { $Iscc = Find-Iscc }
        if (-not $Iscc) {
            throw "ISCC.exe not found. Install Inno Setup (https://jrsoftware.org/isdl.php), " +
                  "set HORDE_ISCC, or pass -Iscc <path-to-ISCC.exe>."
        }
        Write-Step "Compiling installer with $Iscc"
        New-Item -ItemType Directory -Force -Path $distDir | Out-Null
        & $Iscc $issPath "/DStageDir=$stageDir" "/DMyAppVersion=$Version" "/O$distDir"
        if ($LASTEXITCODE -ne 0) { throw "ISCC failed with exit code $LASTEXITCODE" }
        $exe = Join-Path $distDir 'HordeWorker-Setup.exe'
        if (-not (Test-Path $exe)) { throw "ISCC reported success but $exe is missing." }
        Write-Note ("Installer: {0} ({1:N1} MB)" -f $exe, ((Get-Item $exe).Length / 1MB))
    }

    # --- 5. Smoke test ---------------------------------------------------------------------
    if ($SmokeTest) {
        if ($SkipInstaller) { throw '-SmokeTest cannot be combined with -SkipInstaller (it needs the built .exe).' }
        Invoke-SmokeTest -Exe (Join-Path $distDir 'HordeWorker-Setup.exe')
    }

    Write-Step 'Local build complete.'
} finally {
    Pop-Location
}
