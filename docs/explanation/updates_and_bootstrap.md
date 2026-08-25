# Updates and bootstrap

Worker updates coordinate three independently persistent layers: source files, the Python dependency
environment, and operator data. The bootstrap establishes a compatible package manager before it touches
the dependency environment, then prevents worker code from starting until that environment matches the
release lockfile.

For the operator procedure, see [Update the worker](../how-to/update-the-worker.md). Exact commands, flags,
environment variables, release channels, and origin precedence are in the
[command-line reference](../reference/cli.md#update-and-dependency-sync). The exact bootstrap contracts,
implementation map, and test coverage are in the [bootstrap reference](../reference/update_bootstrap.md).

## Ownership depends on the installation method

A managed installation lets the worker updater download and apply release bundles. A git installation
leaves source ownership to git: `git pull` changes the source, and `update` performs dependency
synchronization without overlaying the checkout. An extracted zip relies on the operator to replace source
files before running the dependency synchronizer.

The split prevents the self-updater from rewriting a developer checkout or disagreeing with an external
package manager about the installed version. It costs one extra `git pull` step for git installations.

## A managed update preserves expensive and operator-owned state

The release overlay replaces application source and the lockfile. It preserves:

- `bridgeData.yaml`;
- `.venv` during the source overlay;
- `bin/`, including the private uv executable and recorded backend;
- logs; and
- the sibling `<worker>-data` directory containing models, managed Python, and caches.

Python import roots are mirror-pruned during the overlay. A module removed by a release therefore cannot
remain on disk and shadow its replacement. The dependency environment is retained because rebuilding a
GPU environment is expensive, then reconciled against the new lockfile before launch.

Before changing source, the updater snapshots every source-owned top-level target under the preserved
`bin/` directory and publishes an in-progress marker. Completion clears the marker before deleting the
backup. An ordinary failure restores immediately. After termination or power loss, `bootstrap.py` sees the
marker and restores the complete previous source before it imports `worker_bootstrap`.

## Bootstrap precedes the project environment

The platform launcher needs uv to run the standard-library-only `bootstrap.py`; the bootstrap needs to
work even when `.venv` is missing or incompatible. The launcher therefore runs bootstrap outside the uv
project, using a managed Python 3.12 and a small separate bootstrap cache. A current launcher downloads uv
into staging, verifies the release archive's published SHA-256 and reported version, then replaces its
private executable.

The project pins an exact uv version. Before any project-aware command, the bootstrap compares the selected
uv with that pin. A compatible versioned sidecar is reused. For a mismatch, bootstrap downloads and verifies
the required release, confirms its version outside the project, and publishes it atomically.

The repair never invokes `uv self update`. That command is unavailable when uv was installed in unmanaged
mode. It also never replaces a PATH-provided uv or the executable currently hosting bootstrap, which can be
locked on Windows.

## Source and dependencies converge before launch

The updater invalidates the lock fingerprint before overlaying source. On the next update or launch, a
missing or mismatched fingerprint forces `uv sync --locked` for the recorded compute backend. Worker entry
points run only after uv compatibility and dependency synchronization succeed.

This ordering turns an interrupted update into a retryable state. A partial source overlay is rolled back
before bootstrap imports, and new source can exist beside an old dependency environment only until locked
synchronization succeeds. Worker code does not enter that mismatched environment.

## Failure properties and costs

The design provides these properties:

- A checksum or version mismatch never replaces the last private uv.
- An unavailable uv release stops before the project environment or worker is entered.
- A partial source overlay restores every source target from its pre-update snapshot.
- A failed dependency sync leaves the fingerprint invalid, so the next launch retries it.
- A failed release check does not prevent synchronization of the installed source and lockfile.
- Git-owned source is never overlaid by the managed updater.
- Old micromamba files do not participate in the new uv-managed environment and can remain during migration.

Network access, free disk space, and write permission remain external requirements. When one is absent,
startup stops with an actionable bootstrap or synchronization error and retries on the next invocation.
The temporary source snapshot requires enough free space for one additional copy of source, excluding the
environment, models, caches, and logs.

## Alternatives and tradeoffs

Using `uv self update` would reduce bootstrap code, but unmanaged uv installations reject it and a running
Windows executable may be locked. Direct verified artifacts work for managed and unmanaged provenance.

Overwriting launcher scripts during an update would deliver launcher fixes immediately, but command
interpreters can still be reading those files. The updater preserves launchers and places evolving logic in
the standard-library bootstrap package, which is safe to replace after import. A fresh invocation loads the
new package. Installers record a launcher generation. An older managed installation can consume new Python
bootstrap behavior, then asks the operator to re-run the installer once to refresh the preserved launchers.

This boundary cannot be retroactive: an update initiated by an older updater uses that updater until the
overlay finishes. Transactional rollback protects updates initiated after the fixed bootstrap has landed.
Re-running the current installer is the recovery path if the transition from an older generation is itself
interrupted.

The one-line installer avoids that running-script constraint because the downloaded installer process owns
the operation. For an existing bootstrap-era install, it first asks the intact installed launcher to apply
the bundle. After that launcher returns, the installer copies the current launchers and records their
generation. A fresh or pre-bootstrap install needs a complete copy first; rerunning the installer repeats
that copy and repairs an interrupted attempt. If an existing bootstrap is damaged and cannot apply the
bundle, the installer falls back to the same complete copy and retries through the downloaded launcher.

Mutating the old micromamba environment would save a second environment directory during migration, but it
would cross Python and dependency-management generations. Creating the locked uv environment alongside it
makes the old installation inert and keeps migration deterministic.
