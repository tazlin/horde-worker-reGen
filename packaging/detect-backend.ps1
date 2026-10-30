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
# Version" or "CUDA UMD Version" header field). torch 2.12.0 (the locked line) has no cu128 wheel, so a
# CUDA 12.x driver gets cu126
# -- a 12.6 build runs on any CUDA 12.6+ driver (and, via NVIDIA driver backward-compatibility, on CUDA
# 13 drivers). A 13.2+ driver gets cu132, a 13.0/13.1 driver gets cu130. The pick is then floored by the
# GPU's compute capability: the cu126 wheel carries kernels only through Hopper (sm_90), so a Blackwell
# card (sm_120/sm_100) -- which does not carry a kernel image in cu126 and would die at the first kernel launch --
# is floored onto cu130 even on a CUDA 12.x driver. This mirrors worker_bootstrap.detect
# (tests/bootstrap/test_detect_parity.py guards that the two stay in step).
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
    # ("... CUDA Version: 13.2", or "... CUDA UMD Version: 13.3" on drivers that renamed the field).
    # Returns $null when nvidia-smi is absent or the line cannot be read (e.g. a freshly installed driver
    # that has not added nvidia-smi to PATH), and callers treat $null as "assume the safe 12.6 build".
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
        $match = [regex]::Match($out, "CUDA(?: UMD)? Version\s*:\s*(\d+)\.(\d+)")
        if ($match.Success) {
            return [version]("{0}.{1}" -f $match.Groups[1].Value, $match.Groups[2].Value)
        }
    } catch { }
    return $null
}

function Get-NvidiaComputeCaps {
    # Every GPU's compute capability (sm_<major><minor>) as a [version], from nvidia-smi's structured
    # query. This is the GPU architecture, not the driver's CUDA ceiling: it decides whether a wheel
    # actually has a kernel image for the card. Returns an empty array when the field is unreadable
    # (old driver / nvidia-smi absent), and callers then do not clamp the build.
    $exe = $null
    $cmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($cmd) {
        $exe = $cmd.Source
    } elseif (Test-Path (Join-Path $env:SystemRoot "System32\nvidia-smi.exe")) {
        $exe = Join-Path $env:SystemRoot "System32\nvidia-smi.exe"
    }
    $caps = @()
    if (-not $exe) { return $caps }
    try {
        $out = & $exe --query-gpu=compute_cap --format=csv,noheader 2>$null | Out-String
        # Whole-line only, mirroring worker_bootstrap.detect._GPU_QUERY_LINE_RE: an NVML failure prints the
        # driver version to stdout, which an unanchored match reads as a capability.
        foreach ($m in [regex]::Matches($out, "(?m)^\s*(\d{1,2})\.(\d)\s*$")) {
            $caps += [version]("{0}.{1}" -f $m.Groups[1].Value, $m.Groups[2].Value)
        }
    } catch { }
    return $caps
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
    # Architecture window clamp (mirrors worker_bootstrap.detect._cuda_build). The driver pick can be
    # outside a card's valid window, since cu126 carries sm_50..sm_90 and the CUDA 13 wheels carry
    # sm_75..sm_120 (pre-Turing dropped in CUDA 13). A build without a kernel image for a card dies at the
    # first kernel launch, so clamp both ways for every card. When one card is pre-Turing and another is
    # Blackwell or newer, a single build cannot run both, so the build for the highest card is used, which
    # is the bootstrap's non-interactive default for mixed cards. Without a readable capability the driver
    # pick is kept.
    $caps = @(Get-NvidiaComputeCaps)
    if ($caps.Count -gt 0) {
        $sorted = @($caps | Sort-Object)
        $lowest = $sorted[0]
        $highest = $sorted[-1]
        if ($lowest -lt [version]"7.5" -and $highest -gt [version]"9.0") {
            $caps = @($highest)
        }
        foreach ($cap in $caps) {
            if ($cap -gt [version]"9.0" -and $token -eq "cu126") {
                # cu126 does not carry Blackwell or newer kernels, so floor onto cu130.
                $token = "cu130"
            } elseif ($cap -lt [version]"7.5") {
                # The CUDA 13 wheels dropped pre-Turing (Maxwell/Pascal/Volta), so hold at cu126.
                $token = "cu126"
            }
        }
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
