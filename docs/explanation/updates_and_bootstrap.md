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

## Bootstrap precedes the project environment

The platform launcher needs uv to run the standard-library-only `bootstrap.py`; the bootstrap needs to
work even when `.venv` is missing or incompatible. The launcher therefore runs bootstrap outside the uv
project, using a managed Python 3.12 and a small separate bootstrap cache.

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

This ordering turns an interrupted update into a retryable state. New source can exist beside an old
environment temporarily, but worker code does not enter that environment. A later launch resumes repair
and synchronization instead of producing an application import failure.

## Failure properties and costs

The design provides these properties:

- A checksum or version mismatch never replaces the last private uv.
- An unavailable uv release stops before the project environment or worker is entered.
- A failed dependency sync leaves the fingerprint invalid, so the next launch retries it.
- Git-owned source is never overlaid by the managed updater.
- Old micromamba files do not participate in the new uv-managed environment and can remain during migration.

Network access, free disk space, and write permission remain external requirements. When one is absent,
startup stops with a bootstrap or synchronization error and retries on the next invocation. Atomic
publication protects uv itself; the source overlay is not a full release rollback mechanism.

## Alternatives and tradeoffs

Using `uv self update` would reduce bootstrap code, but unmanaged uv installations reject it and a running
Windows executable may be locked. Direct verified artifacts work for managed and unmanaged provenance.

Overwriting launcher scripts during an update would deliver launcher fixes immediately, but command
interpreters can still be reading those files. The updater preserves launchers and places evolving logic in
the standard-library bootstrap package, which is safe to replace after import. A fresh invocation loads the
new package.

Mutating the old micromamba environment would save a second environment directory during migration, but it
would cross Python and dependency-management generations. Creating the locked uv environment alongside it
makes the old installation inert and keeps migration deterministic.
