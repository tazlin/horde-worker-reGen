"""The only place the worker talks to a text-inference program.

The worker does not implement text inference; it launches a separate program (koboldcpp today, sonar
next) and asks it for generations over HTTP. This package is the whole of that conversation. Everything
above it sees the five verbs of
[`TextBackend`][horde_worker_regen.text_backends.protocol.TextBackend], two result models, and three
exception types, and never sees HTTP.

- [`protocol`][horde_worker_regen.text_backends.protocol]: the contract, the result models and the
  exceptions. Read this first; it explains why the boundary is as narrow as it is.
- [`kobold_api`][horde_worker_regen.text_backends.kobold_api]: the driver for the KoboldAI HTTP API,
  which both backends speak.
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
    KoboldApiJsonKeys,
    KoboldApiRoutes,
    KoboldApiTextBackend,
    KoboldApiTimeouts,
)
from horde_worker_regen.text_backends.protocol import (
    TextBackend,
    TextBackendBusy,
    TextBackendDescription,
    TextBackendError,
    TextBackendRejectedPayload,
    TextBackendUnavailable,
    TextGenerationResult,
)

__all__ = [
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
    "TextBackendDescription",
    "TextBackendError",
    "TextBackendRejectedPayload",
    "TextBackendUnavailable",
    "TextGenerationResult",
]
