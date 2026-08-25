# Update bootstrap reference

The update bootstrap owns the transition from installed source to a runnable, locked environment. This
page records its exact preservation, uv-repair, synchronization, and installation-ownership contracts.
For update steps, see [Update the worker](../how-to/update-the-worker.md). For the design rationale, see
[Updates and bootstrap](../explanation/updates_and_bootstrap.md).

## Installation ownership

| Installation | Source owner | `update` behavior |
|--------------|--------------|-------------------|
| Managed installer | Worker updater | Discover and verify a release, overlay source, then synchronize dependencies. |
| Git checkout | Git | Leave source unchanged and synchronize dependencies. |
| Extracted zip | Operator | Synchronize dependencies after the operator replaces source. |

## Managed overlay contract

The release overlay preserves `bridgeData.yaml`, `.venv`, `bin/`, `logs/`, launcher shims, and the sibling
`<worker>-data` directory. It mirror-prunes the worker's Python import roots so a
module removed from a release cannot remain importable from the old installation.

Before invalidating the sync stamp or changing source, the updater copies every overlay target to
`bin/update-backup` and atomically publishes `bin/update-transaction.json`. The transaction marker lists
only validated, source-owned top-level paths. Preserved state and launcher shims cannot be recovery targets.

The marker is removed only after the complete overlay succeeds. On an ordinary error or `Ctrl+C`, the
updater restores the snapshot immediately. If the process is terminated, `bootstrap.py` restores it before
importing `worker_bootstrap`. The marker remains if restoration fails, allowing another invocation to retry
from the same backup. A completed update removes the backup on a best-effort basis after committing.

The updater invalidates `.venv/.horde-sync-stamp` during the transaction. After a successful overlay, a
later launch synchronizes the environment. After rollback, a missing stamp can cause one redundant sync of
the restored lockfile, which is safe.

## uv compatibility contract

`pyproject.toml` specifies an exact `[tool.uv] required-version`. Before a project-aware uv command, the
bootstrap verifies the selected executable against that version. If necessary, it:

1. downloads the exact platform archive and its adjacent SHA-256 file from the uv GitHub release;
2. verifies the archive digest;
3. extracts only `uv` or `uv.exe` to a temporary sibling file;
4. probes that executable from outside the project; and
5. atomically publishes `bin/uv-<version>` or `bin/uv-<version>.exe`.

A checksum or version failure leaves an existing sidecar unchanged. The repair does not run `uv self
update`, replace a PATH-provided uv, or replace the executable hosting the current bootstrap process.
Network-body truncation and filesystem publication failures are normalized into `UvCompatibilityError`,
which the command dispatcher reports without a traceback.

Current `runtime.cmd` and `runtime.sh` launchers apply the same staging rule to the uv executable needed to
start `bootstrap.py`: download the exact archive and adjacent checksum, verify SHA-256, extract into a
temporary directory, probe the version outside the project, then replace `bin/uv` only after every check
succeeds. Installers record `launcher_generation=2` in `bin/install-info`.

Managed self-update preserves the launcher currently driving it. A managed installation without the current
generation remains launchable and receives all Python bootstrap changes, but launch reports that the latest
installer must be run once to refresh those scripts. Git pulls and extracted-zip replacement update scripts
as part of source ownership.

## Compatibility by update entry point

| Entry point | Source and environment behavior | Launcher behavior |
|-------------|---------------------------------|-------------------|
| `update.cmd` or `update.sh` from a managed install | The installed bootstrap overlays the new Python updater, then the new updater synchronizes the locked environment. v18.0.3 has the required bootstrap command and uv pin for this handoff. | Preserves the script driving the command. A generation notice requests one installer refresh when needed. |
| Current one-line installer over an existing modern install | Invokes the intact installed bootstrap against the downloaded bundle before changing source. A failed installed bootstrap falls back to an idempotent complete copy and retries. Fresh and legacy installs use that complete-copy path immediately. | Copies downloaded launchers only after the installed launcher returns. |
| Graphical installer | Inno Setup prunes and relays bundled source from outside the worker process. | Replaces launchers because the installer executable, rather than a worker launcher, drives the operation. |
| Git pull | Git owns and replaces source. The following update command creates or synchronizes the uv environment. | Git replaces tracked scripts before they are invoked. |
| Extracted zip | The archive tool owns replacement. The following runtime update synchronizes dependencies. | The archive replaces scripts before they are invoked. |

The managed updater does not rewrite a running `.cmd` or `.sh` file. Command interpreters may read those
files incrementally, so replacing one in place can change the remainder of the active command. Evolving
repair behavior therefore lives in `bootstrap.py` and `worker_bootstrap/`; a standalone installer refreshes
the small irreducible launcher after the old invocation has returned.

## Synchronization contract

The bootstrap compares the current lockfile fingerprint with `.venv/.horde-sync-stamp`. A mismatch runs a
locked dependency sync before any worker entry point imports project code. The stamp is written only after
a successful sync. A failed sync remains eligible for retry on the next update or launch.

An applying `update` attempts local synchronization even when release discovery fails. It returns nonzero
because source availability is unknown, while explicitly reporting that dependencies are current and the
worker can start. All shared sync flags, including `--cache-mode`, are applied before every update branch.

The selected compute backend and optional feature groups form part of the synchronization input. The
[command-line reference](cli.md#update-and-dependency-sync) lists the accepted backend, preview, cache, and
torch-hold controls.

## Code map and coverage

| Responsibility | File and symbol | Contract tests |
|----------------|-----------------|----------------|
| Pre-import recovery | `bootstrap.py`, `_recover_interrupted_update` | `tests/bootstrap/test_updater.py` |
| Bootstrap dispatch and pre-launch synchronization | `worker_bootstrap/cli.py`, `main`, `_cmd_update`, `_ensure_synced` | `tests/bootstrap/test_cli.py` |
| Exact uv discovery, verification, and publication | `worker_bootstrap/uvbin.py`, `ensure_compatible_uv`, `_download_verified_uv` | `tests/bootstrap/test_uvbin.py` |
| Release discovery, integrity verification, transaction, and overlay | `worker_bootstrap/updater.py`, `check_for_update`, `perform_update`, `apply_bundle`, `restore_interrupted_update` | `tests/bootstrap/test_updater.py` |
| Platform uv bootstrap and launcher generation | `runtime.cmd`, `runtime.sh`, `worker_bootstrap/updater.py`, `launchers_need_refresh` | `tests/test_uv_version_consistency.py`, `tests/test_self_updater.py` |
| Isolated subprocess environment and locked sync | `worker_bootstrap/runner.py`, `build_child_env`, `uv_sync` | `tests/bootstrap/test_runner.py` |
| Persistent paths and lock fingerprint | `worker_bootstrap/paths.py`, `data_root`, `sync_stamp_file` | `tests/bootstrap/test_paths.py`, `tests/bootstrap/test_cli.py` |
