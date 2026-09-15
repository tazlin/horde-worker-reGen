"""An in-process stand-in for a text backend, with no binary and no HTTP.

[`FakeTextBackend`][horde_worker_regen.text_backends.fake_text_backend.FakeTextBackend] satisfies the
same protocol as the real driver and lives in the package rather than under `tests/` because the
worker's dry-run mode uses it. The text flow can be exercised end to end on a machine with no
inference program installed, the way `fake_worker_processes` lets the image pipeline run with no GPU.
Tests use it too, which is the point. A stand-in that only tests trusted would drift from the flow it
is supposed to stand in for.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from loguru import logger

from horde_worker_regen.text_backends.protocol import (
    TextBackendDescription,
    TextBackendError,
    TextBackendUnavailable,
    TextGenerationResult,
)


@dataclass(frozen=True)
class FakeReadyCall:
    """Represents one `ready` call the fake received."""

    deadline_seconds: float


@dataclass(frozen=True)
class FakeGenerateCall:
    """Represents one `generate` call the fake received.

    The payload is the caller's own mapping, kept as it arrived rather than validated into a copy, so an
    assertion about passthrough is an assertion about what the flow actually handed over.
    """

    payload: Mapping[str, object]
    generation_key: str
    deadline_seconds: float


@dataclass(frozen=True)
class FakeStopCall:
    """Represents one `stop` call the fake received."""

    generation_key: str


@dataclass
class _FakeCallLog:
    """The calls the fake has received, in order."""

    ready: list[FakeReadyCall] = field(default_factory=list)
    describe_count: int = 0
    generate: list[FakeGenerateCall] = field(default_factory=list)
    stop: list[FakeStopCall] = field(default_factory=list)


class FakeTextBackend:
    """A [`TextBackend`][horde_worker_regen.text_backends.protocol.TextBackend] that answers from configuration.

    Satisfies the protocol structurally, like the real driver, so the flow cannot tell the two apart by
    type.

    Examples:
        A backend that is busy for the first two generations and then succeeds:

        ```python
        backend = FakeTextBackend(
            description=TextBackendDescription(
                model_name="koboldcpp/a-model",
                max_context_length=4096,
                max_length=512,
            ),
            response_text="hello",
            generate_failures=(TextBackendBusy("busy"), TextBackendBusy("busy"), None),
        )
        ```
    """

    def __init__(
        self,
        *,
        description: TextBackendDescription,
        response_text: str | Callable[[Mapping[str, object]], str] = "",
        latency_seconds: float = 0.0,
        generate_failures: Sequence[TextBackendError | None] = (),
        ready_results: Sequence[bool] = (),
    ) -> None:
        """Create a fake backend that answers as configured.

        Args:
            description: What [`describe`][horde_worker_regen.text_backends.fake_text_backend.FakeTextBackend.describe]
                returns, and therefore what the flow would advertise for this backend.
            response_text: The text every generation returns, or a callable handed the payload and
                returning the text, for a fake whose answer depends on what was asked.
            latency_seconds: How long each generation takes. Compared against the caller's deadline the
                way a real backend's latency would be.
            generate_failures: One entry per successive generation: an exception to raise, or `None` to
                succeed. Once the script runs out, every further generation succeeds. This is how a
                caller scripts "busy twice then succeed", or a rejected payload, without a server.
            ready_results: One entry per successive `ready` call. Once the script runs out, every
                further call answers `True`. This is how a caller rehearses a backend that is still
                loading its weights for the first few polls, which is the normal reading during a cold
                start and the window a readiness gate exists for.
        """
        self._description = description
        self._response_text = response_text
        self._latency_seconds = latency_seconds
        self._generate_failures = tuple(generate_failures)
        self._ready_results = tuple(ready_results)
        self._calls = _FakeCallLog()
        self._closed = False

    @property
    def ready_calls(self) -> tuple[FakeReadyCall, ...]:
        """Every `ready` call received, in order."""
        return tuple(self._calls.ready)

    @property
    def describe_call_count(self) -> int:
        """How many times `describe` was called."""
        return self._calls.describe_count

    @property
    def generate_calls(self) -> tuple[FakeGenerateCall, ...]:
        """Every `generate` call received, in order, with the payload as it arrived."""
        return tuple(self._calls.generate)

    @property
    def stop_calls(self) -> tuple[FakeStopCall, ...]:
        """Every `stop` call received, in order."""
        return tuple(self._calls.stop)

    @property
    def closed(self) -> bool:
        """Whether `close` has been called."""
        return self._closed

    async def ready(self, *, deadline_seconds: float) -> bool:
        """Return the readiness scripted for this call, recording the call.

        A fake backend is configured with the model it claims to have loaded, so it is ready by
        construction unless `ready_results` says otherwise for the first few calls, which is how a
        caller rehearses a backend still loading.

        Args:
            deadline_seconds: Recorded, so a caller can assert the deadline the flow chose.

        Returns:
            The next entry in the configured readiness script, or `True` once it runs out.
        """
        ready_index = len(self._calls.ready)
        self._calls.ready.append(FakeReadyCall(deadline_seconds=deadline_seconds))
        if ready_index >= len(self._ready_results):
            return True
        return self._ready_results[ready_index]

    async def describe(self) -> TextBackendDescription:
        """Return the configured description, recording the call."""
        self._calls.describe_count += 1
        return self._description

    async def generate(
        self,
        payload: Mapping[str, object],
        *,
        generation_key: str,
        deadline_seconds: float,
    ) -> TextGenerationResult:
        """Return the configured text for this payload, or raise the next scripted failure.

        The call is recorded before anything else, so a caller can assert on a payload that then failed.
        A scripted failure is raised immediately, as a real backend's refusal or busy answer arrives
        without waiting for a generation. Otherwise the configured latency elapses, and a latency past
        the caller's deadline ends the way the real driver's timeout would.

        Args:
            payload: The payload, recorded untouched.
            generation_key: The generation's identifier, recorded.
            deadline_seconds: The caller's deadline, honoured against the configured latency.

        Returns:
            The configured response text.

        Raises:
            TextBackendError: The next entry in the configured failure script, whatever it is.
            TextBackendUnavailable: The configured latency exceeds `deadline_seconds`.
        """
        generation_index = len(self._calls.generate)
        self._calls.generate.append(
            FakeGenerateCall(
                payload=payload,
                generation_key=generation_key,
                deadline_seconds=deadline_seconds,
            ),
        )

        scripted_failure = self._scripted_failure(generation_index)
        if scripted_failure is not None:
            raise scripted_failure

        if self._latency_seconds > deadline_seconds:
            await asyncio.sleep(deadline_seconds)
            raise TextBackendUnavailable(
                f"Fake text backend took longer than {deadline_seconds}s to produce generation {generation_key}",
            )

        await asyncio.sleep(self._latency_seconds)
        return TextGenerationResult(text=self._resolve_response_text(payload))

    def _scripted_failure(self, generation_index: int) -> TextBackendError | None:
        """Return the failure scripted for this generation, or `None` when it is meant to succeed."""
        if generation_index >= len(self._generate_failures):
            return None
        return self._generate_failures[generation_index]

    def _resolve_response_text(self, payload: Mapping[str, object]) -> str:
        """Return the configured text, calling the configured callable with the payload when there is one."""
        if callable(self._response_text):
            return self._response_text(payload)
        return self._response_text

    async def stop(self, *, generation_key: str) -> None:
        """Record the stop request and return. There is no generation to abandon.

        Args:
            generation_key: The key the caller wants abandoned, recorded.
        """
        self._calls.stop.append(FakeStopCall(generation_key=generation_key))

    async def close(self) -> None:
        """Mark the fake closed. Idempotent, like the real driver's."""
        if self._closed:
            return
        self._closed = True
        logger.debug(f"Fake text backend for {self._description.model_name} closed")


__all__ = [
    "FakeGenerateCall",
    "FakeReadyCall",
    "FakeStopCall",
    "FakeTextBackend",
]
