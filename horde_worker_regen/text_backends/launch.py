"""Turn backend-independent launch settings into the command line for a chosen backend.

Backends are named by the horde's own vocabulary,
[`TEXT_BACKENDS`][horde_model_reference.text_backend_names.TEXT_BACKENDS] from the model reference
package, so the value that picks a launch renderer here is the same value that picks the advertised model
name's backend prefix. Adding a backend is one renderer module plus one entry in
:data:`LAUNCH_SPEC_BUILDERS`; nothing in the supervisor or the flow changes.

Supported today: koboldcpp. Planned: sonar (aphrodite), which speaks the same KoboldAI HTTP API and so
needs only a renderer here, not a new driver.
"""

from __future__ import annotations

from collections.abc import Callable

from horde_model_reference.text_backend_names import TEXT_BACKENDS

from horde_worker_regen.text_backends.koboldcpp_launch import koboldcpp_launch_spec
from horde_worker_regen.text_backends.launch_spec import TextBackendLaunchSettings, TextBackendLaunchSpec

LaunchSpecBuilder = Callable[[TextBackendLaunchSettings], TextBackendLaunchSpec]
"""Renders settings into one backend's command line."""

LAUNCH_SPEC_BUILDERS: dict[TEXT_BACKENDS, LaunchSpecBuilder] = {
    TEXT_BACKENDS.koboldcpp: koboldcpp_launch_spec,
}
"""The backends the worker can launch itself, keyed by the horde's backend name."""


class UnsupportedTextBackendError(ValueError):
    """The requested backend has no launch renderer in this worker."""

    def __init__(self, kind: TEXT_BACKENDS) -> None:
        """Name the requested backend and the ones that can be launched."""
        supported = ", ".join(sorted(str(member) for member in LAUNCH_SPEC_BUILDERS))
        super().__init__(f"The worker cannot launch text backend {kind!s}; it can launch: {supported}.")
        self.kind = kind


def launchable_backends() -> frozenset[TEXT_BACKENDS]:
    """Return the backends this worker knows how to launch."""
    return frozenset(LAUNCH_SPEC_BUILDERS)


def build_launch_spec(kind: TEXT_BACKENDS, settings: TextBackendLaunchSettings) -> TextBackendLaunchSpec:
    """Create the command line for ``kind`` from backend-independent settings.

    Raises:
        UnsupportedTextBackendError: ``kind`` has no renderer here.
    """
    builder = LAUNCH_SPEC_BUILDERS.get(kind)
    if builder is None:
        raise UnsupportedTextBackendError(kind)
    return builder(settings)


__all__ = [
    "LAUNCH_SPEC_BUILDERS",
    "LaunchSpecBuilder",
    "UnsupportedTextBackendError",
    "build_launch_spec",
    "launchable_backends",
]
