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
