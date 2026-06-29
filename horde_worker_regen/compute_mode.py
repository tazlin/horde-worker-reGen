"""The worker's *intended* compute backend, read from the install sentinel without importing torch.

The managed installer records which torch build it installed in ``bin/backend`` (written by
``worker_bootstrap/backend.py``): ``cu126``/``cu130``/``cu132`` for NVIDIA, ``rocm``/``rocm-windows``
for AMD, or ``cpu`` for a CPU-only build. A ``cpu`` token is the signal that this install runs in
**CPU / alchemist-only mode**: image generation (the "dreamer" role) is disabled by definition because
CPU inference is impractically slow, while the pure-CPU alchemy forms (upscale, face-fix, caption,
interrogation) stay on offer.

This module is the torch-free reader of that intent. The long-lived orchestrator process and the TUI
are deliberately torch-free (see ``AGENTS.md``), so they must not load torch just to learn the compute
mode; reading a one-line file is enough. The runtime ground truth (what hardware actually answered) is
the separate, heavier :func:`~horde_worker_regen.utils.accelerator_probe.probe_accelerators`; this
module also offers :func:`reconcile_with_probe` to flag the case where declared intent and observed
hardware disagree (a GPU install whose driver is broken, or a CPU install on a box that does have a
GPU).

The install root is located the same way the rest of the worker locates its writable files: relative to
the current working directory, which the launch shims set to the install folder (matching
``app_state.py``'s use of ``Path.cwd()``). The ``HORDE_WORKER_BACKEND`` environment variable, which the
bootstrap also honours, takes precedence so a one-off override needs no file edit.
"""

from __future__ import annotations

import enum
import os
from pathlib import Path

from loguru import logger

_BACKEND_ENV = "HORDE_WORKER_BACKEND"
_CPU_TOKEN = "cpu"

# Accelerator kinds (as reported by the accelerator probe / hordelib) that count as a real GPU/NPU for
# the intent-vs-reality reconciliation. Anything not here (notably ``cpu``) is "no accelerator".
_ACCELERATED_KINDS = frozenset({"cuda", "rocm", "xpu", "npu", "mlu", "mps", "directml"})


class ComputeMode(enum.StrEnum):
    """The compute backend an install is configured to use."""

    CPU = "cpu"
    """CPU-only build: image generation is disabled; the worker runs in alchemist-only mode."""
    ACCELERATED = "accelerated"
    """A GPU/accelerator build (CUDA, ROCm, XPU, ...): full image generation is available."""


def _candidate_backend_files() -> list[Path]:
    """Return the ``bin/backend`` paths to try, most-authoritative first.

    The launch shims run the worker with the install folder as the working directory, so ``cwd/bin`` is
    the primary location. The package-relative path (``horde_worker_regen``'s parent) is a fallback for an
    editable/dev checkout launched from elsewhere.
    """
    candidates = [Path.cwd() / "bin" / "backend"]
    package_root = Path(__file__).resolve().parent.parent
    package_candidate = package_root / "bin" / "backend"
    if package_candidate not in candidates:
        candidates.append(package_candidate)
    return candidates


def read_backend_token(*, backend_file: Path | None = None) -> str | None:
    """Return the declared backend token (``cu132``/``rocm``/``cpu``/...), or None when not recorded.

    Precedence matches the bootstrap: the ``HORDE_WORKER_BACKEND`` environment variable first, then the
    persisted ``bin/backend`` file. An explicit ``backend_file`` overrides the file location (for tests).
    A retired ``cu128`` token is normalised to ``cu126`` so an old sentinel still classifies correctly.
    """
    env_value = os.environ.get(_BACKEND_ENV)
    if env_value and env_value.strip():
        return _normalize(env_value.strip())

    files = [backend_file] if backend_file is not None else _candidate_backend_files()
    for path in files:
        if path is None:
            continue
        try:
            token = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if token:
            return _normalize(token)
    return None


def _normalize(token: str) -> str:
    """Lower-case a token and fold the retired ``cu128`` onto ``cu126`` (mirrors backend.remap_legacy)."""
    cleaned = token.strip().lower()
    return "cu126" if cleaned == "cu128" else cleaned


def intended_compute_mode(*, backend_file: Path | None = None) -> ComputeMode | None:
    """Classify the declared backend into a :class:`ComputeMode`, or None when no sentinel exists.

    None means "unknown" (no managed install sentinel and no env override): callers should preserve their
    existing, backend-agnostic behaviour rather than assume CPU, so a hand-rolled dev install is never
    silently stripped of image generation.
    """
    token = read_backend_token(backend_file=backend_file)
    if token is None:
        return None
    return ComputeMode.CPU if token == _CPU_TOKEN else ComputeMode.ACCELERATED


def is_cpu_only_install(*, backend_file: Path | None = None) -> bool:
    """Return whether this install is explicitly CPU-only (alchemist-only mode).

    True only when the sentinel/env explicitly says ``cpu``; an unknown intent returns False so the worker
    keeps offering image generation unless it has positively been told this is a CPU install.
    """
    return intended_compute_mode(backend_file=backend_file) is ComputeMode.CPU


def reconcile_with_probe(probed_kinds: list[str], *, backend_file: Path | None = None) -> str | None:
    """Return a human-readable warning when declared intent disagrees with the probed hardware, else None.

    ``probed_kinds`` is the list of ``AcceleratorKind`` strings from
    :func:`~horde_worker_regen.utils.accelerator_probe.probe_accelerators` (e.g. ``["cuda"]`` or
    ``["cpu"]``). Two mismatches matter:

    * intent says a GPU build but the probe found only CPU: a masked or broken GPU/driver, which would
      otherwise surface as a stream of faults or a silent 100x-slower run;
    * intent says CPU but the probe found a real accelerator: the user is leaving a usable GPU idle and
      may want to switch back with the GPU build.
    """
    mode = intended_compute_mode(backend_file=backend_file)
    if mode is None:
        return None
    has_accelerator = any(kind in _ACCELERATED_KINDS for kind in probed_kinds)
    if mode is ComputeMode.ACCELERATED and not has_accelerator:
        return (
            "Configured for a GPU build (bin/backend is not 'cpu') but no accelerator was detected; the "
            "worker would run on CPU (~100x slower) or fault. Check your GPU driver, or reinstall the CPU "
            "build for alchemist-only mode."
        )
    if mode is ComputeMode.CPU and has_accelerator:
        return (
            "Configured for CPU-only / alchemist-only mode (bin/backend is 'cpu') but a GPU was detected; "
            "image generation stays disabled. Reinstall a GPU build (e.g. update-runtime --cu132) to use "
            "the accelerator."
        )
    return None


def log_compute_mode_reconciliation(probed_kinds: list[str], *, backend_file: Path | None = None) -> None:
    """Emit the :func:`reconcile_with_probe` warning, if any, at warning level."""
    message = reconcile_with_probe(probed_kinds, backend_file=backend_file)
    if message is not None:
        logger.warning(message)


def compute_mode_display_label(*, backend_file: Path | None = None) -> str | None:
    """Return a short UI label for the compute mode, or None when it is not noteworthy.

    Returns a label only for a CPU / alchemist-only install (the unusual, context-changing case the UI
    should call out); an accelerated or unknown install returns None so a GPU dashboard is unchanged.
    """
    if is_cpu_only_install(backend_file=backend_file):
        return "CPU (alchemist-only)"
    return None


__all__ = [
    "ComputeMode",
    "compute_mode_display_label",
    "intended_compute_mode",
    "is_cpu_only_install",
    "log_compute_mode_reconciliation",
    "read_backend_token",
    "reconcile_with_probe",
]
