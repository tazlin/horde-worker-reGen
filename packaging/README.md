# Packaging & local release verification

This directory holds everything used to turn a worker checkout into the artifacts users install:

- `bundle-include.txt` - manifest of root files that go into the release zip (kept honest by
  `tests/test_bundle_manifest.py`).
- `detect-backend.ps1` - the single GPU detector shared by `install.ps1` and the graphical
  installer; prints the CUDA build chosen from the driver's CUDA version (`cu130` on a CUDA 13+
  driver, else `cu128`), or `amd-unsupported` / `cpu`.
- `inno/HordeWorker.iss` - Inno Setup script for the double-click Windows installer
  (`HordeWorker-Setup.exe`). See [`inno/README.md`](inno/README.md).
- `winget/` - the winget manifest.
- `build-local.ps1` - **builds the zip and the installer locally**, mirroring
  `.github/workflows/release.yml`, so the packaging and installer schemes can be verified on a
  developer machine without pushing a tag or waiting on GitHub Actions.

## Why build locally

`release.yml` only runs on a `v*` tag, on GitHub-hosted runners. That makes it slow to iterate on
the packaging itself (manifest drift, a new launcher, an `.iss` change) and impossible to eyeball
the resulting `HordeWorker-Setup.exe` before it is published. `build-local.ps1` reproduces the
same steps locally so a human can verify them first.

## Quick start (Windows)

From the repo root, in PowerShell (Windows PowerShell 5.1 is fine; no PowerShell 7 needed):

```powershell
# Full mirror of CI: stage -> zip + SHA256SUMS -> HordeWorker-Setup.exe
powershell -ExecutionPolicy Bypass -File packaging\build-local.ps1
```

Outputs (all gitignored):

- `stage\` - the exact tree that is zipped and fed to Inno Setup.
- `horde-worker-reGen.zip` + `SHA256SUMS` - what the one-line installer and winget consume.
- `dist\HordeWorker-Setup.exe` - the double-click installer.

### Switches

| Switch | Effect |
|--------|--------|
| *(none)* | Stage + zip + installer. |
| `-SkipInstaller` | Stage + zip only (Inno Setup not required). |
| `-SkipPack` | Stage + installer only (skip the zip / SHA256SUMS). |
| `-Version 1.2.3` | Version baked into the installer (default `0.0.0-dev`). |
| `-Iscc <path>` | Use a specific `ISCC.exe`. Otherwise `HORDE_ISCC`, then `PATH`, then every drive's `Program Files*\Inno Setup *\ISCC.exe` are searched. |
| `-StrictGuard` | Fail (like CI) if `pyproject.toml` still has a local `path =` source. Default: warn only, so the installer can still be built while the checkout is path-pinned to a local hordelib. |
| `-SmokeTest` | After building, silently install into a temp folder, assert the file layout + persisted `bin\backend`, then uninstall. Briefly creates a Start Menu shortcut (removed on uninstall). |

### Examples

```powershell
# Iterate on the .iss only, then exercise a real install/uninstall cycle:
powershell -ExecutionPolicy Bypass -File packaging\build-local.ps1 -SkipPack -SmokeTest

# Verify exactly what a tagged release would produce (fail on a dev path source):
powershell -ExecutionPolicy Bypass -File packaging\build-local.ps1 -StrictGuard
```

## Inno Setup

Install Inno Setup 6 or 7 from <https://jrsoftware.org/isdl.php> (or `choco install innosetup`).
The script finds `ISCC.exe` automatically across versions and drives; pass `-Iscc` or set
`HORDE_ISCC` if it lives somewhere unusual. The `.iss` uses only standard directives, so it
compiles on both Inno Setup 6 (what CI installs via choco) and 7.

## What this does and does not verify

`build-local.ps1` verifies the **packaging and installer mechanics**: that the bundle stages
cleanly, the zip ships `uv.lock`, and the installer compiles, lays files down, persists the
detected backend, and uninstalls. It deliberately does **not** build the Python/PyTorch
environment: that is deferred to the first launch of `horde-worker.cmd` on the target machine,
exactly as in production. To verify that end of the flow, run the installed `horde-worker.cmd`
(or unzip the bundle and run it) on a real machine with the relevant GPU.

The `-SmokeTest` install runs `detect-backend.ps1` against your real hardware. On a CPU-only or
unsupported-AMD machine it auto-answers the advisory dialog (`/SUPPRESSMSGBOXES`) and proceeds, so
the persisted backend it reports reflects this box, not necessarily the target's.
