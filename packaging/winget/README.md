# winget manifest for `Haidra.HordeWorker`

These manifests publish the worker to the [Windows Package Manager](https://github.com/microsoft/winget-pkgs)
so Windows users get a trusted, auto-updating install without a paid code-signing certificate:

```powershell
winget install Haidra.HordeWorker
winget upgrade Haidra.HordeWorker
```

The package is the same release zip the one-line installer uses (`horde-worker-reGen.zip`). winget extracts
it and exposes `horde-worker` on PATH; the first run bootstraps the environment (uv + PyTorch) via
`runtime.cmd`, exactly like the script install.

## Choosing where it installs (and how much disk you need)

winget installs portable packages under your profile by default
(`%LOCALAPPDATA%\Microsoft\WinGet\Packages\`, i.e. the C: drive), with the `horde-worker` alias on PATH
under `%LOCALAPPDATA%\Microsoft\WinGet\Links`. To put it on another drive:

- **Per install:**

  ```powershell
  winget install Haidra.HordeWorker --location "D:\HordeWorker"
  ```

- **As a persistent default** for every portable package (`winget settings`):

  ```jsonc
  "installBehavior": {
      "portablePackageUserRoot":    "D:/WinGet/Packages",          // used with --scope user (default)
      "portablePackageMachineRoot": "D:/WinGet/Packages/Machine"   // used with --scope machine
  }
  ```

  Both must be absolute paths and apply only to `portable`-type packages. Choose which one is used with
  `--scope user` (the default) or `--scope machine`.

> **`--location` does not move the models.** It controls where the bundle extracts, and therefore where
> `.venv` and the uv package cache (`bin\uv_cache`) are created. The model files are separate and far
> larger, and winget does **not** place them: they default to a `models\` folder next to wherever you
> launch `horde-worker` from, or to `AIWORKER_CACHE_HOME`. To keep everything off C:, also point the
> models at the same drive before the first run, either by setting `cache_home:` in `bridgeData.yaml`
> or with an environment variable:
>
> ```powershell
> setx AIWORKER_CACHE_HOME "D:\HordeWorker\models"   # persists; reopen the terminal for it to apply
> ```

### Disk space

Plan generously; this is a heavyweight install, and most of it is unavoidable:

- **Environment floor, before any models:** roughly **10-15 GB** for the CUDA build (the `.venv` alone is
  ~7-10 GB). PyTorch and the NVIDIA CUDA libraries it pulls in are the bulk, and that cost cannot be
  avoided on a GPU install. The CPU-only build is smaller, around 3-5 GB. This figure covers the
  extracted bundle, the `.venv`, and uv's wheel cache together (the `.venv` hardlinks from the cache, so
  on the same drive they largely share storage rather than doubling it).
- **Models, on top of that:** each model is roughly **2-8 GB**, and a useful multi-model selection runs
  to **tens or hundreds of GB**. These land wherever `cache_home` / `AIWORKER_CACHE_HOME` points.

After a successful install you can reclaim a few GB of cached wheel archives with `uv cache prune` (run
from the install folder); the tradeoff is re-downloading them on the next update.

## Releasing a new version (automated)

Submission is automated by `.github/workflows/winget.yml` (the `winget-releaser` action, Komac under the
hood), so the per-release version bump and SHA256 are handled for you. It is **opt-in** and stays inactive
until a maintainer enables it once:

1. Fork [`microsoft/winget-pkgs`](https://github.com/microsoft/winget-pkgs) under an account/bot you control.
2. Create a classic Personal Access Token that can push to that fork (`public_repo` scope) and add it as the
   repository secret **`WINGET_TOKEN`**.
3. Set the repository variable **`WINGET_ENABLED`** to `true`.

After that, every published GitHub release opens a `winget-pkgs` PR for `Haidra.HordeWorker` automatically
(it reads `horde-worker-reGen.zip` from the release and computes the hash). You can also run it manually via
the workflow's **Run workflow** button with a tag.

## Manual fallback

The three static manifests here are kept as a `winget validate` reference and a manual fallback. To submit by
hand: bump `PackageVersion` in all three files, point `InstallerUrl` at the new tag, set `InstallerSha256`
(from the release's `SHA256SUMS` asset, or `sha256sum horde-worker-reGen.zip`), then:

```powershell
winget validate --manifest packaging\winget
winget install --manifest packaging\winget   # local smoke test in a sandbox
wingetcreate submit --token <gh-token> packaging\winget
```
