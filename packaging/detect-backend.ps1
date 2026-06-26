#Requires -Version 5.1
# Shared GPU detection for the AI Horde Worker Windows installers. Both the one-line installer
# (install.ps1) and the graphical Inno Setup installer call this, so the hardware check that protects
# non-technical users from a silent (~100x slower) CPU install lives in exactly one audited place.
#
# Prints one backend token to stdout:
#   cu132            an NVIDIA GPU on a CUDA 13.2+ driver -> CUDA 13.2 build
#   cu130            an NVIDIA GPU on a CUDA 13.0/13.1 driver -> CUDA 13.0 build
#   cu126            an NVIDIA GPU on a CUDA 12.x driver (or version unreadable) -> CUDA 12.6 build
#   rocm-windows    AMD Windows, supported Radeon/Ryzen AI device for the ROCm Windows PyTorch stack
#   amd-unsupported  an AMD/Radeon GPU was found, but not one of the known installable profiles
#   cpu              no supported GPU found
#
# The CUDA build is the newest the driver's max supported CUDA version can run (nvidia-smi's "CUDA
# Version" header). torch 2.12.0 (the locked line) has no cu128 wheel, so a CUDA 12.x driver gets cu126
# -- a 12.6 build runs on any CUDA 12.6+ driver (and, via NVIDIA driver backward-compatibility, on CUDA
# 13 drivers). A 13.2+ driver gets cu132, a 13.0/13.1 driver gets cu130. This mirrors
# worker_bootstrap.detect (tests/bootstrap/test_detect_parity.py guards that the two stay in step).
#
# This is detection only: callers decide how to message the user and may honour a HORDE_WORKER_BACKEND
# override before calling. Pass -OutFile to also write the token (no trailing newline) to a file, for
# callers that cannot easily capture stdout (the Inno installer reads the file back).
[CmdletBinding()]
param([string]$OutFile = "")

$ErrorActionPreference = "SilentlyContinue"

function Test-NvidiaGpu {
    # nvidia-smi on PATH is the happy path, but a freshly installed driver often does not add it to PATH.
    # Fall back to its default location and to the video-controller list so a real NVIDIA card is never
    # mistaken for a CPU-only machine.
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { return $true }
    if (Test-Path (Join-Path $env:SystemRoot "System32\nvidia-smi.exe")) { return $true }
    try {
        foreach ($c in Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop) {
            if ($c.Name -match "NVIDIA") { return $true }
        }
    } catch { }
    return $false
}

function Get-AmdAdapterNames {
    $names = @()
    try {
        foreach ($c in Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop) {
            if ($c.Name -match "AMD|Radeon") { $names += $c.Name }
        }
    } catch { }
    return $names
}

function Test-AmdGpu {
    return @((Get-AmdAdapterNames)).Count -gt 0
}

function Get-WindowsAmdRocmBackend {
    foreach ($name in Get-AmdAdapterNames) {
        $compact = ($name.ToUpperInvariant() -replace "[\s\-_()]+", "")
        if ($name -match "\bRX\s*(9060|9070|7700|7900)\b" -or $name -match "\b(PRO\s+)?W7900\b") {
            return "rocm-windows"
        }
        if ($name -match "\bAI\s+PRO\s+R9700\b") {
            return "rocm-windows"
        }
        if ($compact -match "RYZENAIMAX|STRIXHALO|RADEON8050S|RADEON8060S") {
            return "rocm-windows"
        }
        if ($compact -match "RYZENAI9(HX)?(365|370|375|465|470|475)") {
            return "rocm-windows"
        }
    }
    return $null
}

function Get-NvidiaCudaVersion {
    # The driver's max supported CUDA runtime version as a [version], parsed from nvidia-smi's header
    # ("... CUDA Version: 13.2"). Returns $null when nvidia-smi is absent or the line cannot be read
    # (e.g. a freshly installed driver that has not added nvidia-smi to PATH); callers treat $null as
    # "assume the safe 12.6 build".
    $exe = $null
    $cmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($cmd) {
        $exe = $cmd.Source
    } elseif (Test-Path (Join-Path $env:SystemRoot "System32\nvidia-smi.exe")) {
        $exe = Join-Path $env:SystemRoot "System32\nvidia-smi.exe"
    }
    if (-not $exe) { return $null }
    try {
        $out = & $exe 2>$null | Out-String
        $match = [regex]::Match($out, "CUDA Version:\s*(\d+)\.(\d+)")
        if ($match.Success) {
            return [version]("{0}.{1}" -f $match.Groups[1].Value, $match.Groups[2].Value)
        }
    } catch { }
    return $null
}

if (Test-NvidiaGpu) {
    # Pick the newest build the driver's max CUDA version can run (mirrors worker_bootstrap.detect._cuda_build).
    $cuda = Get-NvidiaCudaVersion
    if ($cuda -and $cuda -ge [version]"13.2") {
        $token = "cu132"
    } elseif ($cuda -and $cuda -ge [version]"13.0") {
        $token = "cu130"
    } else {
        $token = "cu126"
    }
} elseif (Test-AmdGpu) {
    $amdBackend = Get-WindowsAmdRocmBackend
    if ($amdBackend) {
        $token = $amdBackend
    } else {
        $token = "amd-unsupported"
    }
} else {
    $token = "cpu"
}

if ($OutFile) {
    $dir = Split-Path -Parent $OutFile
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    [System.IO.File]::WriteAllText($OutFile, $token)
}

Write-Output $token
