"""The only place the worker talks to a text-inference program.

The worker does not implement text inference; it launches a separate program (koboldcpp today, sonar
next) and asks it for generations over HTTP. This package is the whole of that conversation. Everything
above it sees the six verbs of
[`TextBackend`][horde_worker_regen.text_backends.protocol.TextBackend], the result models, and three
exception types, and never sees HTTP.

- [`protocol`][horde_worker_regen.text_backends.protocol]: the contract, the result models and the
  exceptions. Read this first; it explains why the boundary is as narrow as it is.
- [`kobold_api`][horde_worker_regen.text_backends.kobold_api]: the driver for the KoboldAI HTTP API,
  which both backends speak. Generations go through its stream so the worker can see them progress.
- [`fake_text_backend`][horde_worker_regen.text_backends.fake_text_backend]: an in-process stand-in
  for dry runs and tests.

Runs in the main process, so it imports nothing from `hordelib` and nothing that loads torch. Its HTTP
client is the `aiohttp` session the parent already owns, injected by the caller.
"""

from horde_worker_regen.text_backends.fake_text_backend import (
    FakeGenerateCall,
    FakeReadyCall,
    FakeStopCall,
    FakeTextBackend,
)
from horde_worker_regen.text_backends.kobold_api import (
    CAPABILITY_PROBE_GENERATION_KEY,
    STATS_POLL_INTERVAL_SECONDS,
    KoboldApiJsonKeys,
    KoboldApiRoutes,
    KoboldApiTextBackend,
    KoboldApiTimeouts,
)
from horde_worker_regen.text_backends.launch import (
    LAUNCH_SPEC_BUILDERS,
    UnsupportedTextBackendError,
    build_launch_spec,
    launchable_backends,
)
from horde_worker_regen.text_backends.launch_spec import (
    LOOPBACK_HOST,
    TextBackendLaunchSettings,
    TextBackendLaunchSpec,
)
from horde_worker_regen.text_backends.protocol import (
    TextBackend,
    TextBackendBusy,
    TextBackendCapabilities,
    TextBackendDescription,
    TextBackendError,
    TextBackendRejectedPayload,
    TextBackendUnavailable,
    TextGenerationProgress,
    TextGenerationProgressCallback,
    TextGenerationResult,
)

__all__ = [
    "CAPABILITY_PROBE_GENERATION_KEY",
    "LAUNCH_SPEC_BUILDERS",
    "LOOPBACK_HOST",
    "STATS_POLL_INTERVAL_SECONDS",
    "FakeGenerateCall",
    "FakeReadyCall",
    "FakeStopCall",
    "FakeTextBackend",
    "KoboldApiJsonKeys",
    "KoboldApiRoutes",
    "KoboldApiTextBackend",
    "KoboldApiTimeouts",
    "TextBackend",
    "TextBackendBusy",
    "TextBackendCapabilities",
    "TextBackendDescription",
    "TextBackendError",
    "TextBackendRejectedPayload",
    "TextBackendLaunchSettings",
    "TextBackendLaunchSpec",
    "TextBackendUnavailable",
    "TextGenerationProgress",
    "TextGenerationProgressCallback",
    "TextGenerationResult",
    "UnsupportedTextBackendError",
    "build_launch_spec",
    "launchable_backends",
]
