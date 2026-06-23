# Update the worker

Updating matches how you installed. In every case: stop the worker first, update, then start it again.
Announcements for new versions go out on [Discord](https://discord.gg/3DxrhksKzn).

The worker checks for new releases in the background and tells you when one is available: the
dashboard shows a notification, and a headless/console worker logs it at startup and in its periodic
status report. Set `HORDE_WORKER_NO_UPDATE_CHECK=1` to disable the check.

## Steps

1. **Stop** the worker (`Ctrl+C`, or Quit in the dashboard).
2. **Update**, matching how you installed:

   | Installed with | Update with |
   |----------------|-------------|
   | One-line installer or `.exe` | Run `update.cmd` (Windows) or `update.sh` (Linux/macOS) in your worker folder, or re-run the same installer. Both download the latest release and update in place, leaving your peered `<worker>-data` folder (models, cache, Python) untouched. |
   | Git clone | `git pull`, then `update-runtime.cmd` (or the `.sh` / `-rocm` variant). The self-updater detects a git checkout and stays out of the way, so it never overlays your working tree. |
   | Zip | Download the [latest zip](https://github.com/Haidra-Org/horde-worker-reGen/archive/refs/heads/main.zip), extract over the existing folder, then `update-runtime.cmd` |

3. **Start** the worker again.

Script names above assume Windows and NVIDIA. For Linux use the `.sh` scripts; for AMD use the `-rocm`
variants.

## When the worker offers to update itself

A worker installed by the one-line installer or the `.exe` checks for a newer release on launch and, by
default, asks before applying it. You can answer:

- **Yes** (the default): download, verify, and apply the update, then continue starting.
- **No**: skip it for this launch only. You will be asked again next time.
- **skip**: skip this specific version and don't ask again until a *newer* one is released.

Tune the behaviour with `HORDE_WORKER_AUTO_UPDATE`: `prompt` (default, ask interactively / notify when
headless), `auto` (apply without asking), or `off` (never check or self-update). To check or apply manually
at any time, run `update.cmd --check` (report only) or `update.cmd` (apply). Git-clone installs never
self-update (see the table above).

## Release channels (stable and beta)

By default the worker follows the **stable** channel and only ever sees stable `vX.Y.Z` releases. Betas are
published as GitHub pre-releases tagged `vX.Y.Z-beta.N` and are hidden from stable users.

To opt into betas, set `HORDE_WORKER_UPDATE_CHANNEL=beta`; set it back to `stable` (or unset it) to leave.
Channel handling is version-aware, so:

- A stable worker is never offered a beta.
- If you are already running a beta build, the worker automatically follows the beta channel, keeps you on
  the newest beta, and moves you onto the matching stable release once it ships.
- A beta is **never** rolled back to an older stable: the worker only ever moves you to a strictly newer
  version.

## Where releases are pulled from

The self-updater pulls future releases from the same repository you originally installed from (recorded in
`bin/install-info` at install time), so a fork or staging install updates itself from the right place. Set
`HORDE_WORKER_UPDATE_REPO=owner/repo` to override the origin explicitly.

## Download preview and managing disk use

A managed install (one-line installer or `.exe`) previews what a dependency sync would
download before fetching anything, and prunes superseded wheels afterwards so the cache does not grow
without bound. The defaults are safe and need no configuration; the knobs below are for tuning.

PyTorch is the bulk of the download (~1.5 GB+). It only changes version when a new *release* ships a new
lockfile, so most updates download little. When a release does bump torch, the worker shows a short table
(what changes, to which versions, and an approximate download size) before proceeding.

**Limping along on the installed torch.** If you would rather not pull a fresh ~1.5 GB torch, you can
keep the version you already have, *as long as nothing in the new release actually requires the newer
torch*. The worker checks this with uv: if the older torch still resolves, the upgrade is "optional" and
can be held; if a dependency genuinely needs the newer one, it is "mandatory" and cannot be skipped.

- `update-runtime.cmd --hold-torch` (or `update.sh ... --hold-torch`) holds torch/torchvision for this
  run and updates everything else. Equivalent env var: `HORDE_WORKER_SYNC_HOLD=1`.
- On an interactive terminal, a torch download above the confirm threshold (default 1500 MB) prompts you
  to **[U]pgrade / [H]old / [C]ancel**. Tune with `--confirm-above-mb N`.
- Non-interactive runs (the installer, CI) default to taking the upgrade. Set
  `HORDE_WORKER_SYNC_HEADLESS_POLICY=hold` (or `--headless-policy hold`) to hold instead.
- `--no-sync-preview` (or `HORDE_WORKER_SYNC_PREVIEW=0`) skips the preview entirely and always installs
  exactly the locked versions, as before.

**Cache disk management.** By default the worker keeps a private uv cache in the peered `<worker>-data`
folder and runs `uv cache prune` after a successful sync to reclaim old wheels. This never touches
anything outside the worker.

- If you already use `uv` for other projects and do not want a second multi-GB cache, set
  `HORDE_WORKER_UV_CACHE_MODE=shared` (or `--cache-mode shared`). The worker then uses uv's normal
  system cache and **never auto-prunes it** (it is not ours to clean), so your other projects are safe.
- Disable pruning entirely with `--no-prune` or `HORDE_WORKER_SYNC_PRUNE=0`.
- Point the cache anywhere by setting `UV_CACHE_DIR` yourself; a cache you set is never auto-pruned.

## If you manage your own virtualenv

If you installed into your own environment (see [Choose a PyTorch build](choose-a-pytorch-build.md)),
re-sync your dependencies every time you `git pull`, matching your GPU's CUDA version. For example, for
a CUDA 13.0/13.1 driver:

```bash
python -m pip install -r requirements.txt -U --extra-index-url https://download.pytorch.org/whl/cu130
```

Use `cu132` for a CUDA 13.2+ driver, `cu126` for a CUDA 12.6+ driver, or `rocm6.4` for AMD.

## Antivirus note

Some antivirus software (for example Avast) can interfere with downloads. If you see
`CRYPT_E_NO_REVOCATION_CHECK` errors during an update, temporarily disable it. More fixes are in
[Troubleshooting](troubleshoot.md).
