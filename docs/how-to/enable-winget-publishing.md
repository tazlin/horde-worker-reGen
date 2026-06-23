# Enable winget publishing (maintainers)

**Status: paused.** The worker is *not* distributed via the Windows Package Manager (winget), and nothing
user-facing points at it. Publishing to winget requires a one-time manual onboarding with Microsoft's
community repo that is not worth carrying on top of every release until we commit to the channel, so it is
intentionally switched off in a way that cannot fire by accident.

This page is the runbook for turning it back on. It doubles as the record of everything that was removed or
disabled, so re-enabling is just reversing each item below.

## What "paused" means today

- **The publish workflow is a hard no-op.** `.github/workflows/winget.yml` still exists, but its `publish`
  job is gated `if: false`, *independent of* the `WINGET_ENABLED` repository variable. It will not submit
  anything even if a release is published, `workflow_dispatch` is invoked, or `WINGET_ENABLED` is set to
  `true`. Flipping that variable alone does nothing.
- **The per-release version guard is skipped.** The two winget assertions in
  `tests/test_packaging_versions.py` are `@pytest.mark.skip`-ped, so a release bump no longer has to also
  sync the manifests to keep CI green. (The non-winget guard in that file still runs.)
- **No user-facing pointer mentions winget.** The install/update docs, the README, the in-app "update
  available" notices, and the installer's closing message were scrubbed of winget instructions.
- **The dormant machinery is kept in-tree** so re-enabling is edits, not reconstruction: the three
  manifests and `packaging/winget/README.md`, `packaging/sync-winget-version.py`, and the workflow file all
  remain.
- **The self-updater's defensive winget handling is unchanged.** `worker_bootstrap/updater.py` still detects
  an install under a WinGet `Packages` path and declines to self-update it (deferring to `winget upgrade`).
  This is unreachable while no winget package exists and needs no change when re-enabling.

## Re-enable checklist

1. **Complete the one-time winget-pkgs onboarding (the manual first step).**
   - Fork [`microsoft/winget-pkgs`](https://github.com/microsoft/winget-pkgs) under an account or bot you
     control.
   - Sync the manifests under `packaging/winget/` to the current release and its real hash:
     `python packaging/sync-winget-version.py <version> --sha256 <hash-from-SHA256SUMS>`, then
     `winget validate --manifest packaging\winget`.
   - Open the **first** `Haidra.HordeWorker` PR to `microsoft/winget-pkgs` manually (e.g.
     `wingetcreate submit --token <gh-token> packaging\winget`) and get it merged by Microsoft's
     moderators. Automation only works once the package exists in the community repo.

2. **Set the repository secret and variable.**
   - Secret **`WINGET_TOKEN`**: a classic PAT that can push to your fork of `microsoft/winget-pkgs`.
   - Variable **`WINGET_ENABLED`**: set to `true`.

3. **Un-gate the workflow.** In `.github/workflows/winget.yml`, change the `publish` job's
   `if: false` back to `if: ${{ vars.WINGET_ENABLED == 'true' }}` and update the header comment to drop the
   "PAUSED" note.

4. **Re-activate the version guard.** Remove the `@pytest.mark.skip` from `test_winget_package_versions_match_version`
   and `test_winget_installer_url_tag_matches_version` in `tests/test_packaging_versions.py`, and make sure
   each release syncs the manifests (run `packaging/sync-winget-version.py`, or let the workflow compute the
   hash) so those tests stay green.

5. **Restore the user-facing pointers** (each was removed; add it back):
   - `README.md` — the `winget install Haidra.HordeWorker` line in the Windows command-line install.
   - `docs/how-to/install.md` — the `winget` scripted-install block, and the two "to another drive with
     winget" notes (install path + disk space sections).
   - `docs/tutorials/getting-started.md` — "Other ways to install (winget, ...)".
   - `docs/how-to/update-the-worker.md` — the `winget` row in the update table, the "winget and git-clone
     installs never self-update" sentence, and "(one-line installer, `.exe`, or winget)" in the download
     preview section.
   - `docs/how-to/troubleshoot.md` — the SmartScreen note's "`winget install` avoids the prompt."
   - `install.ps1` — the closing "To update later" message's `winget upgrade Haidra.HordeWorker` option.
   - In-app update notices: `horde_worker_regen/run_worker.py`, `horde_worker_regen/tui/app.py`,
     `horde_worker_regen/reporting/status_reporter.py`, and the docstring in
     `horde_worker_regen/update_check.py` — add `winget upgrade Haidra.HordeWorker` back to the remedy text,
     and restore the assertion in `tests/test_status_reporter_version_nag.py`.
   - Remove the "Paused" banner at the top of `packaging/winget/README.md`.

6. **Verify.** Push a throwaway tag (or run the workflow via `workflow_dispatch` with a tag) and confirm a
   `winget-pkgs` PR is opened for `Haidra.HordeWorker`.
