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

The updater invalidates `.horde-sync-stamp` before overlaying source. A later launch therefore synchronizes
dependencies even when the overlay or its first synchronization attempt was interrupted.

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

## Synchronization contract

The bootstrap compares the current lockfile fingerprint with `.venv/.horde-sync-stamp`. A mismatch runs a
locked dependency sync before any worker entry point imports project code. The stamp is written only after
a successful sync. A failed sync remains eligible for retry on the next update or launch.

The selected compute backend and optional feature groups form part of the synchronization input. The
[command-line reference](cli.md#update-and-dependency-sync) lists the accepted backend, preview, cache, and
torch-hold controls.

## Code map and coverage

| Responsibility | File and symbol | Contract tests |
|----------------|-----------------|----------------|
| Bootstrap dispatch and pre-launch synchronization | `worker_bootstrap/cli.py`, `main`, `_cmd_update`, `_ensure_synced` | `tests/bootstrap/test_cli.py` |
| Exact uv discovery, verification, and publication | `worker_bootstrap/uvbin.py`, `ensure_compatible_uv`, `_download_verified_uv` | `tests/bootstrap/test_uvbin.py` |
| Release discovery, integrity verification, and overlay | `worker_bootstrap/updater.py`, `check_for_update`, `perform_update`, `apply_bundle` | `tests/bootstrap/test_updater.py` |
| Isolated subprocess environment and locked sync | `worker_bootstrap/runner.py`, `build_child_env`, `uv_sync` | `tests/bootstrap/test_runner.py` |
| Persistent paths and lock fingerprint | `worker_bootstrap/paths.py`, `data_root`, `sync_stamp_file` | `tests/bootstrap/test_paths.py`, `tests/bootstrap/test_cli.py` |
