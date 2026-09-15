"""What the worker needs to know to start a text backend process, independent of which backend it is.

A launch has two halves. The operator-facing half is the same for every backend: which executable, which
model, which loopback port, which card, how much of the model to offload, how much context to allocate,
where the program's own output goes. That is :class:`TextBackendLaunchSettings`. The backend-specific half
is how those settings become a command line, and that lives in one module per backend (see
[`launch`][horde_worker_regen.text_backends.launch]); its output is :class:`TextBackendLaunchSpec`, which is
all the process supervisor ever sees.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

LOOPBACK_HOST = "127.0.0.1"
"""Every text backend listens on loopback only; the worker is its sole client."""


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
    """Stable device index of the card the backend uses; None runs it without an accelerator flag."""
    gpu_layers: int
    """How much of the model to place on the card, in the backend's own unit (layers for llama.cpp-based
    backends); a large value means all of it."""
    context_length: int
    """The context size the backend allocates for."""
    log_path: Path
    """Where the backend's stdout and stderr are appended."""


class TextBackendLaunchSpec(BaseModel):
    """Represents one concrete command line for a text backend, ready to start."""

    model_config = ConfigDict(frozen=True)

    executable: Path
    """The backend program."""
    arguments: tuple[str, ...]
    """Arguments after the executable, already rendered as strings."""
    port: int
    """The loopback port the backend will listen on; `base_url` is built from it."""
    log_path: Path
    """Where the backend's own stdout and stderr go."""
    device_index: int | None = None
    """The stable device index the backend was told to use, for footprint measurement; None off-GPU."""

    @property
    def command(self) -> list[str]:
        """Return the full argv."""
        return [str(self.executable), *self.arguments]

    @property
    def base_url(self) -> str:
        """Return the HTTP base URL a driver targets."""
        return f"http://{LOOPBACK_HOST}:{self.port}"


__all__ = [
    "LOOPBACK_HOST",
    "TextBackendLaunchSettings",
    "TextBackendLaunchSpec",
]
