# Graphical Windows installer (Inno Setup)

`HordeWorker.iss` builds `HordeWorker-Setup.exe`, the double-click installer for non-technical Windows
users. It is a thin wrapper around the same release bundle the one-line installer and winget consume (the
CI staging directory), so there is no second copy of the install logic to keep in sync.

## What it does

- Per-user install (no administrator / UAC prompt) under `%LOCALAPPDATA%\Programs\AIHordeWorker` by
  default; the user can pick another drive on the destination page. Paths containing spaces are rejected
  (PyTorch and uv fail on them).
- On a re-run with an existing install, presents an "existing installation found" choice page: **update in
  place** (reuses the previous folder, skips the destination page), **move to a new location** (removes the
  current install first, then lets the user pick a new folder), or **uninstall and exit**. Inno's default of
  silently reusing the previous folder is disabled (`UsePreviousAppDir=no`) so this choice is always offered.
  Models and the dependency cache live in the sibling `…-data` folder and are not moved when relocating, so a
  new location re-downloads them on first launch.
- Detects the GPU once during the wizard via the shared `detect-backend.ps1`, warns on CPU-only, and offers
  a CPU-only path (or cancels) for unsupported AMD-on-Windows. It writes the result to `bin\backend`, which
  the deferred first-launch bootstrap reads so the correct PyTorch build is installed.
- Does **not** build the Python environment itself: first launch of `horde-worker.cmd` does that (and opens
  the browser wizard), exactly like every other install path.
- Seeds `bridgeData.yaml` from the template (only if absent) and never deletes it on uninstall.
- Creates Start Menu and (optional) Desktop shortcuts, and a real uninstaller that removes the generated
  `.venv` and `bin` but preserves `bridgeData.yaml` and the model cache (which lives outside the install dir).

## Building locally

Install Inno Setup 6 or 7 from <https://jrsoftware.org/isdl.php> (or `choco install innosetup`),
then run the one-command local build from the repo root, which stages the bundle and compiles
this `.iss` exactly the way CI does:

```powershell
# Stage + zip + installer (see packaging/README.md for switches)
powershell -ExecutionPolicy Bypass -File packaging\build-local.ps1

# Just the installer (skip the zip):
powershell -ExecutionPolicy Bypass -File packaging\build-local.ps1 -SkipPack

# Build, then silently install/verify/uninstall to exercise the produced .exe:
powershell -ExecutionPolicy Bypass -File packaging\build-local.ps1 -SkipPack -SmokeTest
```

The installer lands in `dist\HordeWorker-Setup.exe`. `build-local.ps1` finds `ISCC.exe`
automatically across Inno Setup versions and drives (PATH, the `HORDE_ISCC` env var, then every
drive's `Program Files*\Inno Setup *`); pass `-Iscc <path>` if it is installed somewhere unusual.
This `.iss` uses only standard directives, so it compiles on both Inno Setup 6 (what CI installs
via choco) and 7. See [`../README.md`](../README.md) for the full local build/test reference.

To compile the `.iss` by hand instead, stage the bundle first (run `build-local.ps1 -SkipPack
-SkipInstaller`, or replicate the "Stage the runtime bundle" step from
`.github/workflows/release.yml`) and then:

```powershell
& "<path-to>\ISCC.exe" packaging\inno\HordeWorker.iss /DStageDir="$(Resolve-Path stage)" /DMyAppVersion=0.0.0-dev
```

## Code signing (deferred)

The installer currently ships **unsigned**, so the first download shows a one-click SmartScreen
"More info -> Run anyway". To remove that warning later (the plan is Azure Trusted Signing, ~$10/mo):

1. Register a SignTool in your ISCC configuration, e.g.

   ```text
   hordesign=$qC:\Path\To\signtool.exe$q sign /fd sha256 /tr http://timestamp.url /td sha256 $f
   ```

2. Uncomment the `SignTool=hordesign $f` line in `HordeWorker.iss`.
3. Wire the same signing step into the `windows-installer` job in `.github/workflows/release.yml` and drop
   the SmartScreen note/screenshot from the top-level `README.md`.
