"""Provisioning plan and stamp bookkeeping for the second (image-utilities) virtual environment.

The worker's ``horde-image-utilities`` capability service runs from its own virtual environment (see
:func:`worker_bootstrap.paths.utilities_venv_dir`) so its native, accelerator-gated dependencies never
share a resolution with the worker's ``.venv``. This module owns the pure, side-effect-light pieces of
keeping that venv in step:

- :func:`utilities_requirements_file` names the CI-compiled pin file for a backend token.
- :func:`plan_utilities_provision` builds the argv command lists that create the venv and install into it,
  without running them (execution lives in :mod:`worker_bootstrap.runner`, mirroring the builder/runner
  split the rest of the bootstrap uses).
- :func:`utilities_provision_wanted` decides, from the same feature resolution the main sync uses, whether
  the utilities venv is wanted at all.
- :func:`needs_provision` compares the recorded stamp against the current requirements pin so an
  up-to-date venv is left untouched.

Like the rest of the bootstrap brain, this module is standard-library only (see
``tests/bootstrap/test_stdlib_only.py``): it runs via ``uv run --script`` before any project venv exists.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from worker_bootstrap import backend, paths

__all__ = [
    "UtilitiesProvisionError",
    "UtilitiesStamp",
    "needs_provision",
    "plan_utilities_provision",
    "read_utilities_stamp",
    "utilities_provision_wanted",
    "utilities_requirements_file",
    "write_utilities_stamp",
]

# The utilities venv is pinned to the same managed CPython the worker's own bootstrap shims pin
# (runtime.cmd / runtime.sh use ``uv run --python 3.12``) and the same window pyproject's
# ``requires-python`` declares, so both environments resolve against one interpreter line.
UTILITIES_PYTHON_VERSION = "3.12"

# A pip requirements line carries a hash pin as ``--hash=sha256:...``; its presence is what tells the
# install step it may (and should) pass ``--require-hashes``.
_HASH_MARKER = "--hash="

_STAMP_BACKEND_TOKEN_KEY = "backend_token"
_STAMP_REQUIREMENTS_SHA256_KEY = "requirements_sha256"


class UtilitiesProvisionError(RuntimeError):
    """Raised when creating or populating the utilities venv fails, carrying an actionable message."""


@dataclass(frozen=True)
class UtilitiesStamp:
    """Represents the provenance of a provisioned utilities venv (its backend token and requirements pin).

    Attributes:
        backend_token: The locked torch-build token the venv was provisioned for (e.g. ``cu132``).
        requirements_sha256: The SHA256 hex digest of the requirements file it was installed from.
    """

    backend_token: str
    requirements_sha256: str

    def to_dict(self) -> dict[str, str]:
        """Return the JSON-serialisable mapping persisted to the stamp file."""
        return {
            _STAMP_BACKEND_TOKEN_KEY: self.backend_token,
            _STAMP_REQUIREMENTS_SHA256_KEY: self.requirements_sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> UtilitiesStamp | None:
        """Parse a stamp mapping, returning None when either required field is missing or not a string."""
        token = data.get(_STAMP_BACKEND_TOKEN_KEY)
        digest = data.get(_STAMP_REQUIREMENTS_SHA256_KEY)
        if not isinstance(token, str) or not isinstance(digest, str):
            return None
        return cls(backend_token=token, requirements_sha256=digest)


def utilities_requirements_file(*, token: str, root: Path | None = None) -> Path:
    """Return the CI-compiled requirements pin for *token* (``requirements/utilities.<token>.txt``).

    The file is a hashed, fully-resolved pin of ``horde-image-utilities`` with its server and
    accelerator-matched feature extras, compiled per backend token in CI. A token with no committed file
    simply has nothing to provision yet.
    """
    return (root or paths.install_root()) / "requirements" / f"utilities.{token}.txt"


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


def _requirements_carry_hashes(path: Path) -> bool:
    """Return whether the requirements file at *path* contains hash pins (``--hash=``)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return _HASH_MARKER in text


def plan_utilities_provision(
    *,
    uv: str,
    backend_token: str,
    root: Path | None = None,
    require_hashes: bool | None = None,
) -> list[list[str]]:
    """Return the argv command lists that create and populate the utilities venv, without running them.

    Two commands are produced: ``uv venv --clear <utilities-venv> --python <version>`` to create the
    environment against the pinned managed CPython, then ``uv pip install --python <interpreter>
    [--require-hashes] -r <requirements>`` to install the CI-compiled pin into it. This is a pure builder
    (it runs nothing), mirroring the builder/runner split in :mod:`worker_bootstrap.runner`;
    :func:`worker_bootstrap.runner.provision_utilities` executes the result.

    ``--clear`` keeps the create step fully managed and non-interactive: uv >=0.8 prompts before replacing
    an existing venv and uv >=0.10 requires ``--clear`` to replace one at all, so without it a reprovision
    would hang on a "replace it?" prompt. Provisioning only runs when the venv is absent or stale (a
    backend or pin change), where a clean venv is exactly what is wanted, so clearing never discards a
    healthy up-to-date environment. The install reuses the worker's peered uv cache and managed CPython
    because :func:`worker_bootstrap.runner.run_uv` runs every command through
    :func:`worker_bootstrap.runner.build_child_env`, so the utilities venv shares the same cache and
    ``uv cache prune`` as the main ``.venv`` rather than re-downloading torch into a private cache.

    Args:
        uv: The uv executable to invoke.
        backend_token: The locked torch-build token whose requirements pin to install.
        root: The install root (defaults to :func:`worker_bootstrap.paths.install_root`).
        require_hashes: Whether to pass ``--require-hashes``. When None, it is auto-detected from whether
            the requirements file carries ``--hash=`` pins, so a not-yet-hashed placeholder still installs.
    """
    root = root or paths.install_root()
    requirements = utilities_requirements_file(token=backend_token, root=root)
    use_require_hashes = _requirements_carry_hashes(requirements) if require_hashes is None else require_hashes

    create_command = [
        uv,
        "venv",
        "--clear",
        str(paths.utilities_venv_dir(root)),
        "--python",
        UTILITIES_PYTHON_VERSION,
    ]
    install_command = [uv, "pip", "install", "--python", str(paths.utilities_python(root))]
    if use_require_hashes:
        install_command.append("--require-hashes")
    install_command += ["-r", str(requirements)]

    return [create_command, install_command]


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
    """Record the backend token and requirements digest the utilities venv was just provisioned against."""
    requirements = utilities_requirements_file(token=backend_token, root=root)
    stamp = UtilitiesStamp(backend_token=backend_token, requirements_sha256=_sha256_file(requirements))
    path = paths.utilities_stamp_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stamp.to_dict(), indent=2), encoding="utf-8")


def needs_provision(*, backend_token: str, root: Path | None = None) -> bool:
    """Return whether the utilities venv must be (re)provisioned for *backend_token*.

    False when there is no committed requirements pin for the token (nothing to install yet), so the
    caller can stay a no-op until CI publishes the pin. Otherwise True when the venv interpreter is
    missing, when no stamp is recorded, or when the recorded token or requirements digest no longer match
    the current pin; False when a matching stamp proves the venv is already up to date.
    """
    requirements = utilities_requirements_file(token=backend_token, root=root)
    if not requirements.is_file():
        return False
    if not paths.utilities_python(root).is_file():
        return True
    stamp = read_utilities_stamp(root)
    if stamp is None:
        return True
    return stamp.backend_token != backend_token or stamp.requirements_sha256 != _sha256_file(requirements)
