"""Provisioning plan and stamp bookkeeping for the second (image-utilities) virtual environment.

The worker's ``horde-image-utilities`` capability service runs from its own virtual environment (see
:func:`worker_bootstrap.paths.utilities_venv_dir`) so its native, accelerator-gated dependencies never
share a resolution with the worker's ``.venv``. That environment is defined by a self-contained uv project
under ``requirements/utilities/`` (a ``pyproject.toml`` plus a committed ``uv.lock``); the bootstrap
provisions the venv from it with ``uv sync --locked``. A locked sync installs from the lockfile's recorded
sources and never re-resolves, so every machine gets byte-identical wheels: the utilities venv reuses the
main ``.venv``'s cached torch/CUDA stack instead of re-downloading it, and can never be pulled onto a box's
ambient extra-index (the failure a floating ``uv pip install -r requirements.txt`` is prone to). This module
owns the pure, side-effect-light pieces of keeping that venv in step:

- :func:`plan_utilities_provision` builds the argv command that syncs the venv from the lock, without
  running it (execution lives in :mod:`worker_bootstrap.runner`, mirroring the builder/runner split the rest
  of the bootstrap uses).
- :func:`utilities_provision_wanted` decides, from the same feature resolution the main sync uses, whether
  the utilities venv is wanted at all.
- :func:`needs_provision` compares the recorded stamp against the current utilities lock so an up-to-date
  venv is left untouched.
- :func:`torch_hold` reports a worker environment that is limping along on an older torch than the
  utilities lock pins, which defers provisioning so the held torch is not fetched a second time.

Like the rest of the bootstrap brain, this module is standard-library only (see
``tests/bootstrap/test_stdlib_only.py``): it runs via ``uv run --script`` before any project venv exists.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path

from worker_bootstrap import backend, paths, sync_plan

__all__ = [
    "UTILITIES_PYTHON_VERSION",
    "TorchHold",
    "UtilitiesProvisionError",
    "UtilitiesStamp",
    "locked_torch_version",
    "needs_provision",
    "plan_utilities_provision",
    "read_utilities_stamp",
    "torch_hold",
    "utilities_provision_wanted",
    "write_utilities_stamp",
]

# The utilities venv is pinned to the same managed CPython the worker's own bootstrap shims pin
# (runtime.cmd / runtime.sh use ``uv run --python 3.12``) and the same window the utilities project's
# ``requires-python`` declares, so both environments resolve against one interpreter line.
UTILITIES_PYTHON_VERSION = "3.12"

_STAMP_BACKEND_TOKEN_KEY = "backend_token"
_STAMP_LOCK_SHA256_KEY = "lock_sha256"


class UtilitiesProvisionError(RuntimeError):
    """Raised when creating or populating the utilities venv fails, carrying an actionable message."""


@dataclass(frozen=True)
class UtilitiesStamp:
    """Represents the provenance of a provisioned utilities venv (its backend token and lock digest).

    Attributes:
        backend_token: The locked torch-build token (build extra) the venv was synced for (e.g. ``cu132``).
        lock_sha256: The SHA256 hex digest of the utilities ``uv.lock`` it was synced from.
    """

    backend_token: str
    lock_sha256: str

    def to_dict(self) -> dict[str, str]:
        """Return the JSON-serialisable mapping persisted to the stamp file."""
        return {
            _STAMP_BACKEND_TOKEN_KEY: self.backend_token,
            _STAMP_LOCK_SHA256_KEY: self.lock_sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> UtilitiesStamp | None:
        """Parse a stamp mapping, returning None when either required field is missing or not a string."""
        token = data.get(_STAMP_BACKEND_TOKEN_KEY)
        digest = data.get(_STAMP_LOCK_SHA256_KEY)
        if not isinstance(token, str) or not isinstance(digest, str):
            return None
        return cls(backend_token=token, lock_sha256=digest)


def utilities_provision_wanted(*, token: str, env_value: str | None = None) -> bool:
    """Return whether the utilities venv is wanted for *token* under the resolved feature set.

    Derived from the same resolution the main sync uses
    (:func:`worker_bootstrap.backend.desired_feature_extras`): the utilities venv is wanted exactly when
    that feature set is non-empty. Full builds (NVIDIA/CPU) therefore want it by default, lean backends do
    not unless opted in via ``HORDE_WORKER_FEATURES``, and an explicit ``none`` opts out everywhere.

    Raises:
        ValueError: When ``env_value`` names an extra that is not a known feature extra (propagated from
            :func:`worker_bootstrap.backend.desired_feature_extras`).
    """
    return bool(backend.desired_feature_extras(token, env_value=env_value))


def plan_utilities_provision(
    *,
    uv: str,
    backend_token: str,
    root: Path | None = None,
) -> list[list[str]]:
    """Return the argv command that syncs the utilities venv from its lock, without running it.

    One command is produced: ``uv sync --locked --reinstall --project <requirements/utilities> --extra
    <build> --python <version>``. Run with ``UV_PROJECT_ENVIRONMENT`` pointed at the peered utilities venv
    (see :func:`worker_bootstrap.runner.provision_utilities`), this creates the venv if absent and reconciles
    it to the committed lock: deterministic, non-interactive, and cache-shared with the main ``.venv``. This
    is a pure builder (it runs nothing), mirroring the builder/runner split in :mod:`worker_bootstrap.runner`.

    ``--locked`` is deliberate: it asserts the lock is current and installs strictly from it, so the utilities
    resolution can never diverge from the committed one (the property that keeps it off a box's ambient
    extra-index and sharing the main venv's cached wheels).

    ``--reinstall`` is deliberate too. Provisioning runs only when the venv is stale (a changed lock digest or
    backend token, per :func:`needs_provision`), and a plain ``uv sync`` reconciles by package name and
    version: it leaves an already-installed distribution untouched even when the lock now points the same
    version at a different source. The utilities lock does exactly that (onnxruntime-gpu keeps one version but
    routes per build to a CUDA-matched index), so without ``--reinstall`` a re-provision would keep the old
    wheel and then stamp success, wedging the venv until it was deleted by hand. Reinstalling reconciles the
    actual wheels to the lock; cached wheels re-link rather than re-download, and the cost stays bounded
    because this fires only on a genuine change. The result is returned as a one-element list of commands so
    the runner's execute-each loop stays uniform.

    Args:
        uv: The uv executable to invoke.
        backend_token: The locked torch-build token whose build extra to sync (e.g. ``cu132``).
        root: The install root (defaults to :func:`worker_bootstrap.paths.install_root`).
    """
    root = root or paths.install_root()
    sync_command = [
        uv,
        "sync",
        "--locked",
        "--reinstall",
        "--project",
        str(paths.utilities_project_dir(root)),
        "--extra",
        backend_token,
        "--python",
        UTILITIES_PYTHON_VERSION,
    ]
    return [sync_command]


def _sha256_file(path: Path) -> str:
    """Return the SHA256 hex digest of the file at *path* (empty string when it cannot be read)."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def read_utilities_stamp(root: Path | None = None) -> UtilitiesStamp | None:
    """Return the recorded utilities-venv stamp, or None when it is absent or unreadable/malformed."""
    try:
        raw = paths.utilities_stamp_file(root).read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return UtilitiesStamp.from_dict(data)


def write_utilities_stamp(*, backend_token: str, root: Path | None = None) -> None:
    """Record the backend token and utilities-lock digest the venv was just synced against."""
    stamp = UtilitiesStamp(
        backend_token=backend_token,
        lock_sha256=_sha256_file(paths.utilities_lock_file(root)),
    )
    path = paths.utilities_stamp_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stamp.to_dict(), indent=2), encoding="utf-8")


@dataclass(frozen=True)
class TorchHold:
    """Represents a worker torch installation that does not match the utilities lock target.

    Attributes:
        installed_version: The torch version installed in the worker's own venv, local build tag included
            (e.g. ``2.12.1+cu132``).
        locked_version: The torch build the utilities environment would install (e.g. ``2.14.0+cu132``).
    """

    installed_version: str
    locked_version: str


def _torch_local_build(backend_token: str) -> str:
    """Return the local-version suffix used by the selected torch wheel index."""
    if backend_token == "cpu" and sys.platform == "darwin":
        return ""
    return f"+{backend_token}"


def locked_torch_version(*, backend_token: str, root: Path | None = None) -> str | None:
    """Return the torch build the utilities lock resolves for a backend, or None when unknown.

    The utilities project routes torch to a per-build wheel index, so its lock carries one ``torch`` entry
    per build (``2.14.0+cu126``, ``2.14.0+cpu``, ...). Only the entry whose local build matches
    ``backend_token`` is considered; CPU wheels on macOS have no local suffix. If duplicate matching entries
    exist, the newest wins independent of lock order. None means "cannot tell" (absent or unparseable lock,
    no matching torch entry), which callers read as "no hold": a lock the bootstrap cannot parse must never
    block provisioning.

    Args:
        backend_token: The selected torch build token (for example, ``cu132`` or ``cpu``).
        root: The install root (defaults to :func:`worker_bootstrap.paths.install_root`).

    Returns:
        The newest resolved torch build for the backend, or None when the lock has no matching readable entry.
    """
    try:
        with paths.utilities_lock_file(root).open("rb") as handle:
            lock_data = tomllib.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, tomllib.TOMLDecodeError) as error:
        warnings.warn(f"Could not read the image-utilities lock's torch version: {error}", stacklevel=2)
        return None
    locked_packages = lock_data.get("package")
    if not isinstance(locked_packages, list):
        return None
    expected_local_build = _torch_local_build(backend_token)
    matching_torch_versions: list[str] = []
    for locked_package in locked_packages:
        if not isinstance(locked_package, dict) or locked_package.get("name") != "torch":
            continue
        package_version = locked_package.get("version")
        if not isinstance(package_version, str) or not package_version:
            continue
        _, separator, local_build = package_version.partition("+")
        package_local_build = f"+{local_build}" if separator else ""
        if package_local_build.lower() == expected_local_build.lower():
            matching_torch_versions.append(package_version)
    newest_torch_version: str | None = None
    for torch_version in matching_torch_versions:
        if newest_torch_version is None or sync_plan.version_at_least(torch_version, newest_torch_version):
            newest_torch_version = torch_version
    return newest_torch_version


def torch_hold(*, backend_token: str, root: Path | None = None) -> TorchHold | None:
    """Return the torch hold that defers utilities provisioning, or None when there is no hold.

    The worker's own environment can be left on an older torch on purpose (limping along through a large
    optional upgrade) while the regenerated utilities lock pins the newer one. Syncing the lane from that lock
    would download the very torch the hold declined and leave the two environments on different builds, so
    provisioning waits for a matching main-environment build. Any known mismatch is deferred, including a
    different accelerator build or a main environment ahead of the lock; otherwise provisioning would
    download a second accelerator stack and break the same-build invariant. None whenever either version is
    unknown, so neither an unreadable lock nor a venv without torch can block provisioning.

    Args:
        backend_token: The selected torch build token (for example, ``cu132`` or ``cpu``).
        root: The install root (defaults to :func:`worker_bootstrap.paths.install_root`).

    Returns:
        The mismatch that requires deferral, or None when utilities provisioning may proceed.
    """
    installed_torch_version = sync_plan.installed_versions(paths.venv_dir(root)).get("torch")
    locked_torch_build = locked_torch_version(backend_token=backend_token, root=root)
    if installed_torch_version is None or locked_torch_build is None:
        return None
    if installed_torch_version.lower() == locked_torch_build.lower():
        return None
    return TorchHold(installed_version=installed_torch_version, locked_version=locked_torch_build)


def needs_provision(*, backend_token: str, root: Path | None = None) -> bool:
    """Return whether the utilities venv must be (re)provisioned for *backend_token*.

    False when there is no committed utilities lock (nothing to sync from), so the caller can stay a no-op
    until the utilities project ships. Otherwise True when the venv interpreter is missing, when no stamp is
    recorded, or when the recorded token or lock digest no longer match the committed lock; False when a
    matching stamp proves the venv is already up to date.
    """
    lock = paths.utilities_lock_file(root)
    if not lock.is_file():
        return False
    if not paths.utilities_python(root).is_file():
        return True
    stamp = read_utilities_stamp(root)
    if stamp is None:
        return True
    return stamp.backend_token != backend_token or stamp.lock_sha256 != _sha256_file(lock)
