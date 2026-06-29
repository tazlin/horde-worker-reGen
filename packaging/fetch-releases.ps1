#requires -Version 5.1
<#
.SYNOPSIS
  Query the GitHub releases API for a repo and write the non-draft tag names to a file,
  one per line, newest first. Used to bake the version list into the Inno installer at
  build time so the wizard does not need a live API call at install time.

.DESCRIPTION
  Writes one tag per line to the output file. Exits 0 on success, 1 on failure (network
  error, rate-limit, etc.). When the output file already exists and is not stale, the
  script can skip the network call.

.PARAMETER Repo
  owner/repo slug (default: Haidra-Org/horde-worker-reGen).

.PARAMETER OutFile
  Path to write the version list to.

.PARAMETER MaxAgeMinutes
  If the output file already exists and is newer than this many minutes, skip the network
  call. Default 0 (always fetch). In CI this should be 0; local builds can set a higher
  value to avoid hitting the API on every run.

.EXAMPLE
  powershell -File packaging\fetch-releases.ps1 -OutFile stage\release-versions.txt
#>
[CmdletBinding()]
param(
    [string]$Repo = 'Haidra-Org/horde-worker-reGen',
    [Parameter(Mandatory = $true)]
    [string]$OutFile,
    [int]$MaxAgeMinutes = 0
)

$ErrorActionPreference = 'Stop'

# Skip the network call when the output is fresh enough.
if ((Test-Path $OutFile) -and ($MaxAgeMinutes -gt 0)) {
    $age = [datetime]::Now - (Get-Item $OutFile).LastWriteTime
    if ($age.TotalMinutes -lt $MaxAgeMinutes) {
        Write-Host "release-versions.txt is fresh ($([math]::Round($age.TotalMinutes, 1)) min old); skipping fetch."
        exit 0
    }
}

Write-Host "Fetching release tags for $Repo from GitHub API..."

try {
    # TLS 1.2 required by GitHub; opt in before the request.
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    $releases = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases?per_page=30" `
        -Headers @{Accept = 'application/vnd.github+json'} `
        -ErrorAction Stop

    $tags = $releases | Where-Object { -not $_.draft } | ForEach-Object { $_.tag_name }

    if (-not $tags) {
        Write-Warning 'No non-draft releases found.'
        exit 1
    }

    $parent = Split-Path $OutFile -Parent
    if ($parent -and -not (Test-Path $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }

    $tags | Out-File -FilePath $OutFile -Encoding ascii
    Write-Host "Wrote $($tags.Count) release tags to $OutFile"
    exit 0
}
catch {
    Write-Warning "Failed to fetch releases: $_"
    exit 1
}
