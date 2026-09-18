"""Find or fetch what a text backend the worker launches itself needs: its program and its model file.

Provisioning the program is per backend: koboldcpp ships as a prebuilt single-file binary the bootstrap
package pins and downloads; other backends will have their own arrangement (an installed console script, a
container, a separate environment). This module maps the horde's backend name to whatever obtains that
backend's executable, so the launch path asks one question and adding a backend is one more entry here.

Provisioning the model file is backend-neutral, because a model the reference declares a file for is the
same file whichever program loads it. The reference package's own downloader does the work: it resumes an
interrupted transfer with a `Range` request, hashes while it streams, and fails the fetch on a digest
mismatch, so the worker writes no downloader of its own and a dropped multi-gigabyte transfer costs the
bytes already on disk rather than all of them.

Which executable is obtained depends on the compute path the backend will be launched on, because a
koboldcpp asset carries either CUDA or Vulkan-and-CPU and not both. The path comes from the caller
(:func:`effective_text_backend_accelerator`), never from config read in here, so the binary on disk and the
flag on the command line cannot be decided by two different answers.

The bootstrap package (`worker_bootstrap`) is bundled beside the worker in a release and present in a
checkout, but it is not part of the wheel, so it is imported only when koboldcpp is actually being
provisioned and its absence is reported as a clear error rather than an import failure at start-up.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from horde_model_reference.download_engine import download_addressed_file
from horde_model_reference.text_backend_names import TEXT_BACKENDS
from loguru import logger

from horde_worker_regen.compute_mode import TextBackendAccelerator, resolve_text_backend_accelerator

if TYPE_CHECKING:
    from horde_worker_regen.text_backends.model_catalogue import ResolvedTextModel


class TextBackendProvisionError(RuntimeError):
    """Something a text backend needs could not be obtained: its executable, or its model file."""


def detect_text_backend_accelerator() -> TextBackendAccelerator:
    """Return the compute path this host's hardware calls for, for an install that declares no backend.

    The hardware probe lives in the bootstrap package, which a wheel-only install does not carry. Nothing
    can be probed without it, so the answer stays CUDA, which renders the command line the worker rendered
    before a compute path could be chosen at all.
    """
    try:
        from worker_bootstrap.koboldcpp_bin import detected_accelerator
    except ImportError:
        logger.warning(
            "This install declares no compute backend and does not carry worker_bootstrap, so the text "
            "backend's hardware cannot be probed; assuming CUDA. Set `text_backend_accelerator` to choose.",
        )
        return TextBackendAccelerator.CUDA
    return TextBackendAccelerator(detected_accelerator())


def effective_text_backend_accelerator(
    configured: TextBackendAccelerator,
    *,
    backend_file: Path | None = None,
) -> TextBackendAccelerator:
    """Return the compute path to provision and launch a text backend for.

    What the operator or the install declares wins; only an install that declares nothing falls back to
    probing the hardware.

    Args:
        configured: The operator's `text_backend_accelerator`.
        backend_file: Override the install sentinel's location (for tests).
    """
    declared = resolve_text_backend_accelerator(configured, backend_file=backend_file)
    if declared is not None:
        return declared
    return detect_text_backend_accelerator()


def _provision_koboldcpp(accelerator: TextBackendAccelerator) -> Path:
    """Return the pinned koboldcpp binary for *accelerator*, downloading and verifying it on first use.

    Raises:
        TextBackendProvisionError: The bootstrap package is not available, or the download, checksum or
            version probe failed.
    """
    try:
        from worker_bootstrap.koboldcpp_bin import (
            KoboldcppProvisionError,
            ensure_koboldcpp,
            variant_for_accelerator,
        )
    except ImportError as error:
        raise TextBackendProvisionError(
            "koboldcpp provisioning needs the worker_bootstrap package, which this install does not carry; "
            "set text_backend_executable to a koboldcpp binary instead.",
        ) from error
    try:
        return ensure_koboldcpp(variant=variant_for_accelerator(accelerator))
    except KoboldcppProvisionError as error:
        raise TextBackendProvisionError(str(error)) from error


ExecutableProvisioner = Callable[[TextBackendAccelerator], Path]
"""Obtains one backend's executable for a compute path, downloading it if it is distributed that way."""

EXECUTABLE_PROVISIONERS: dict[TEXT_BACKENDS, ExecutableProvisioner] = {
    TEXT_BACKENDS.koboldcpp: _provision_koboldcpp,
}
"""The backends the worker can obtain an executable for, keyed by the horde's backend name."""


def provision_executable(kind: TEXT_BACKENDS, *, accelerator: TextBackendAccelerator) -> Path:
    """Return the executable for ``kind``, obtaining it first when the backend is distributed that way.

    Blocking: a first koboldcpp provision downloads hundreds of megabytes. Callers on an event loop run
    it in a thread.

    Args:
        kind: Which backend to obtain an executable for.
        accelerator: The compute path the backend will be launched on, which decides which build is
            obtained. It is passed in rather than read from config here so that one answer serves both the
            binary and the launch flag.

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
    return provisioner(accelerator)


def _file_is_complete(path: Path, declared_size_bytes: int | None) -> bool:
    """Return whether *path* holds the whole file, by the size the reference declares for it.

    Size rather than digest: hashing gigabytes on every launch would add minutes to a start-up that
    otherwise takes seconds, and the digest was already checked when the file was fetched. An interrupted
    transfer leaves a short file, which is what this catches.
    """
    if not path.is_file():
        return False
    if declared_size_bytes is None:
        return True
    return path.stat().st_size == declared_size_bytes


def _unfetchable_file_message(resolved: ResolvedTextModel) -> str:
    """Return the message for a model file that is not usable and cannot be fetched."""
    if resolved.path.is_file():
        return (
            f"The text model file for {resolved.canonical_name} at {resolved.path} is "
            f"{resolved.path.stat().st_size} bytes where {resolved.declared_size_bytes} is expected, and no "
            "origin for it is on record, so the worker cannot replace it. Put a complete copy at that path, "
            "or point `text_model` at one elsewhere."
        )
    return (
        f"The text model file for {resolved.canonical_name} is not at {resolved.path} and no origin for it "
        "is on record, so the worker cannot fetch it. Put the file at that path, set `text_models_dir` to "
        "where you keep it, or point `text_model` at the file directly."
    )


def ensure_text_model_file(resolved: ResolvedTextModel) -> Path:
    """Return the resolved model's file, fetching it first when it is absent and has an origin on record.

    Blocking: a first fetch downloads gigabytes. Callers on an event loop run it in a thread.

    Progress is logged once at each end rather than per chunk, because a chunk-rate line on a multi-gigabyte
    transfer would bury every other line in the log for the minutes it runs.

    Args:
        resolved: The model the operator's `text_model` resolved to.

    Returns:
        The path to the complete file.

    Raises:
        TextBackendProvisionError: The file is unusable and no origin for it is on record, or the fetch
            failed or produced a file that did not match the declared digest.
    """
    path = resolved.path
    declared_size_bytes = resolved.declared_size_bytes
    if _file_is_complete(path, declared_size_bytes):
        return path

    origin_url = resolved.fetch_url
    declared_file = resolved.declared_file
    if origin_url is None or declared_file is None:
        raise TextBackendProvisionError(_unfetchable_file_message(resolved))

    partial_path = Path(f"{path}.part")
    resumed = partial_path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Fetching text model {resolved.canonical_name} ({declared_size_bytes} bytes) to {path}"
        f"{', resuming a partial transfer' if resumed else ''}",
    )
    started_at = time.monotonic()
    outcome = download_addressed_file(origin_url, path, sha256=declared_file.sha256sum)
    elapsed_seconds = time.monotonic() - started_at
    if not outcome.success:
        raise TextBackendProvisionError(
            f"Fetching the text model file for {resolved.canonical_name} from {origin_url} failed after "
            f"{elapsed_seconds:.0f}s; the reference downloader verified {outcome.bytes_written} bytes "
            f"against {declared_file.sha256sum}. Check the log for the transfer's own errors, or put the "
            f"file at {path} yourself.",
        )
    logger.info(
        f"Fetched text model {resolved.canonical_name} in {elapsed_seconds:.0f}s "
        f"({outcome.bytes_written} bytes, {'resumed' if resumed else 'fresh transfer'})",
    )
    return path


__all__ = [
    "EXECUTABLE_PROVISIONERS",
    "ExecutableProvisioner",
    "TextBackendProvisionError",
    "detect_text_backend_accelerator",
    "effective_text_backend_accelerator",
    "ensure_text_model_file",
    "provision_executable",
]
