"""Find or fetch the executable for a text backend the worker launches itself.

Provisioning is per backend: koboldcpp ships as a prebuilt single-file binary the bootstrap package pins
and downloads; other backends will have their own arrangement (an installed console script, a container,
a separate environment). This module maps the horde's backend name to whatever obtains that backend's
executable, so the launch path asks one question and adding a backend is one more entry here.

The bootstrap package (`worker_bootstrap`) is bundled beside the worker in a release and present in a
checkout, but it is not part of the wheel, so it is imported only when koboldcpp is actually being
provisioned and its absence is reported as a clear error rather than an import failure at start-up.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from horde_model_reference.text_backend_names import TEXT_BACKENDS


class TextBackendProvisionError(RuntimeError):
    """The executable for a backend could not be obtained."""


def _provision_koboldcpp() -> Path:
    """Return the pinned koboldcpp binary, downloading and verifying it on first use.

    Raises:
        TextBackendProvisionError: The bootstrap package is not available, or the download, checksum or
            version probe failed.
    """
    try:
        from worker_bootstrap.koboldcpp_bin import KoboldcppProvisionError, ensure_koboldcpp
    except ImportError as error:
        raise TextBackendProvisionError(
            "koboldcpp provisioning needs the worker_bootstrap package, which this install does not carry; "
            "set text_backend_executable to a koboldcpp binary instead.",
        ) from error
    try:
        return ensure_koboldcpp()
    except KoboldcppProvisionError as error:
        raise TextBackendProvisionError(str(error)) from error


ExecutableProvisioner = Callable[[], Path]
"""Obtains one backend's executable, downloading it if the backend is distributed that way."""

EXECUTABLE_PROVISIONERS: dict[TEXT_BACKENDS, ExecutableProvisioner] = {
    TEXT_BACKENDS.koboldcpp: _provision_koboldcpp,
}
"""The backends the worker can obtain an executable for, keyed by the horde's backend name."""


def provision_executable(kind: TEXT_BACKENDS) -> Path:
    """Return the executable for ``kind``, obtaining it first when the backend is distributed that way.

    Blocking: a first koboldcpp provision downloads hundreds of megabytes. Callers on an event loop run
    it in a thread.

    Raises:
        TextBackendProvisionError: No provisioner exists for ``kind``, or provisioning failed.
    """
    provisioner = EXECUTABLE_PROVISIONERS.get(kind)
    if provisioner is None:
        supported = ", ".join(sorted(str(member) for member in EXECUTABLE_PROVISIONERS))
        raise TextBackendProvisionError(
            f"The worker cannot obtain an executable for text backend {kind!s}; it can obtain: {supported}. "
            "Set text_backend_executable to the program's path instead.",
        )
    return provisioner()


__all__ = [
    "EXECUTABLE_PROVISIONERS",
    "ExecutableProvisioner",
    "TextBackendProvisionError",
    "provision_executable",
]
