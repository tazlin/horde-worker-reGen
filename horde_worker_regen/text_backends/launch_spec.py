"""What the worker needs to know to start a text backend process, independent of which backend it is.

A launch has two halves. The operator-facing half is the same for every backend: which executable, which
model, which loopback port, which card and compute path, how much of the model to offload, how much context
to allocate, how many generations to run at once, where the program's own output goes. That is
:class:`TextBackendLaunchSettings`. The backend-specific half
is how those settings become a command line, and that lives in one module per backend (see
[`launch`][horde_worker_regen.text_backends.launch]); its output is :class:`TextBackendLaunchSpec`, which is
all the process supervisor ever sees.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from horde_worker_regen.compute_mode import TextBackendAccelerator

LOOPBACK_HOST = "127.0.0.1"
"""Every text backend listens on loopback only; the worker is its sole client."""

SOURCE_SCRIPT_SUFFIX = ".py"
"""An executable with this suffix is a source checkout's entry script and is run by the worker's interpreter."""


class TextBackendLaunchSettings(BaseModel):
    """Represents the operator's choices for running a text backend, before any backend-specific rendering."""

    model_config = ConfigDict(frozen=True)

    executable: Path
    """The backend program."""
    model_path: Path
    """The model artefact the backend loads (a GGUF for llama.cpp-based backends, a repository path or id
    for others)."""
    port: int
    """Loopback port the backend listens on."""
    device_index: int | None
    """Stable device index of the card the backend uses; None runs it without a device flag."""
    gpu_layers: int
    """How much of the model to place on the card, in the backend's own unit (layers for llama.cpp-based
    backends); a large value means all of it."""
    context_length: int
    """The context size the backend allocates for."""
    log_path: Path
    """Where the backend's stdout and stderr are appended."""
    accelerator: TextBackendAccelerator = TextBackendAccelerator.CUDA
    """Which compute path to launch on. The default renders what the worker rendered before the path could
    be chosen, so a caller that sets nothing is unchanged; a launch resolves it from the install instead."""
    parallel_requests: int = 1
    """How many generations the backend may run at once; one leaves the backend's own default in place."""

    @field_validator("accelerator")
    @classmethod
    def _accelerator_is_concrete(cls, value: TextBackendAccelerator) -> TextBackendAccelerator:
        """Validate that the compute path was resolved, since no backend has a command line for ``auto``."""
        if value is TextBackendAccelerator.AUTO:
            raise ValueError(
                "`accelerator` must name a concrete compute path by launch time; resolve it with "
                "`text_backends.provision.effective_text_backend_accelerator` first.",
            )
        return value


class TextBackendLaunchSpec(BaseModel):
    """Represents one concrete command line for a text backend, ready to start."""

    model_config = ConfigDict(frozen=True)

    executable: Path
    """The backend program: a binary, or a Python script run from a source checkout."""
    arguments: tuple[str, ...]
    """Arguments after the executable, already rendered as strings."""
    port: int
    """The loopback port the backend will listen on; `base_url` is built from it."""
    log_path: Path
    """Where the backend's own stdout and stderr go."""
    device_index: int | None = None
    """The stable device index the backend was told to use, for footprint measurement; None off-GPU."""

    @property
    def runs_from_source(self) -> bool:
        """Whether the executable is a Python script rather than a program the OS can start directly."""
        return self.executable.suffix.lower() == SOURCE_SCRIPT_SUFFIX

    @property
    def command(self) -> list[str]:
        """Return the full argv.

        A script is run by the worker's own interpreter: a backend checkout (a patched build, or a build
        for a platform without a release binary) keeps its native libraries beside the script and imports
        only packages the worker already carries, so no second environment is needed. A script launched this
        way is the server itself, not a bootloader, which the supervisor's tree resolution already tolerates.
        """
        if self.runs_from_source:
            return [sys.executable, str(self.executable), *self.arguments]
        return [str(self.executable), *self.arguments]

    @property
    def base_url(self) -> str:
        """Return the HTTP base URL a driver targets."""
        return f"http://{LOOPBACK_HOST}:{self.port}"


__all__ = [
    "LOOPBACK_HOST",
    "SOURCE_SCRIPT_SUFFIX",
    "TextBackendLaunchSettings",
    "TextBackendLaunchSpec",
]
