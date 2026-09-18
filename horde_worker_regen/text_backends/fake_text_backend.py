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
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from loguru import logger

from horde_worker_regen.text_backends.protocol import (
    TextBackendCapabilities,
    TextBackendCredentialRefused,
    TextBackendDescription,
    TextBackendError,
    TextBackendUnavailable,
    TextGenerationProgress,
    TextGenerationProgressCallback,
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
    capabilities_count: int = 0
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

        A backend that streams three chunks and then stops producing tokens, which is what a stall
        detector has to catch:

        ```python
        backend = FakeTextBackend(
            description=description,
            stream_chunks=("one ", "two ", "three"),
            chunk_interval_seconds=0.01,
            stalls_after_chunks=True,
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
        finish_reason: str | None = None,
        stream_chunks: Sequence[str] = (),
        chunk_interval_seconds: float = 0.0,
        stalls_after_chunks: bool = False,
        generation_stats: bool = False,
        stats_prompt_tokens: int = 0,
        stats_tokens_per_chunk: int = 1,
        credential_refused: bool = False,
    ) -> None:
        """Create a fake backend that answers as configured.

        Args:
            description: What [`describe`][horde_worker_regen.text_backends.fake_text_backend.FakeTextBackend.describe]
                returns, and therefore what the flow would advertise for this backend.
            response_text: The text every generation returns, or a callable handed the payload and
                returning the text, for a fake whose answer depends on what was asked. Ignored when
                `stream_chunks` is given, because then the chunks are the generation.
            latency_seconds: How long each generation takes, on top of whatever the chunks cost.
                Compared against the caller's deadline the way a real backend's latency would be.
            generate_failures: One entry per successive generation: an exception to raise, or `None` to
                succeed. Once the script runs out, every further generation succeeds. This is how a
                caller scripts "busy twice then succeed", or a rejected payload, without a server.
            ready_results: One entry per successive `ready` call. Once the script runs out, every
                further call answers `True`. This is how a caller rehearses a backend that is still
                loading its weights for the first few polls, which is the normal reading during a cold
                start and the window a readiness gate exists for.
            finish_reason: What the generation reports as its reason for stopping. `None` is the answer
                a backend that does not report one gives, sonar's stream among them.
            stream_chunks: The pieces the generation arrives in, reported one at a time through the
                caller's progress callback. Their concatenation is the generated text. Empty means one
                chunk carrying the whole of `response_text`.
            chunk_interval_seconds: How long each chunk takes to arrive.
            stalls_after_chunks: Whether the generation stops producing after its scripted chunks and
                never returns. This is a backend that wedged part way through, which the caller's own
                stall bound is what ends.
            generation_stats: Whether this backend exposes per-request statistics. The switch the real
                driver's capability probe answers, and the same switch here decides whether progress
                carries token counts and a rate at all.
            stats_prompt_tokens: Prompt tokens the statistics report, when `generation_stats` is on.
            stats_tokens_per_chunk: How many tokens each chunk counts for in the statistics, since a
                chunk is an arbitrary number of tokens on a real backend and never one by definition.
            credential_refused: Whether every verb refuses the caller's credential, as a
                password-protected backend does for a worker whose password is wrong. Settable
                afterwards through
                [`credential_refused`][horde_worker_regen.text_backends.fake_text_backend.FakeTextBackend.credential_refused],
                which is how a caller rehearses an operator correcting the password.
        """
        self._description = description
        self._response_text = response_text
        self._latency_seconds = latency_seconds
        self._generate_failures = tuple(generate_failures)
        self._ready_results = tuple(ready_results)
        self._finish_reason = finish_reason
        self._stream_chunks = tuple(stream_chunks)
        self._chunk_interval_seconds = chunk_interval_seconds
        self._stalls_after_chunks = stalls_after_chunks
        self._generation_stats = generation_stats
        self._stats_prompt_tokens = stats_prompt_tokens
        self._stats_tokens_per_chunk = stats_tokens_per_chunk
        self._credential_refused = credential_refused
        self._calls = _FakeCallLog()
        self._closed = False

    @property
    def credential_refused(self) -> bool:
        """Whether every verb refuses the caller's credential."""
        return self._credential_refused

    @credential_refused.setter
    def credential_refused(self, refused: bool) -> None:
        """Set whether the credential is refused, so a corrected password can be rehearsed mid-test."""
        self._credential_refused = refused

    def _raise_if_credential_refused(self, verb: str) -> None:
        """Raise the refusal a password-protected backend answers a wrong credential with.

        Raises:
            TextBackendCredentialRefused: This stand-in is scripted to refuse the caller's credential.
        """
        if not self._credential_refused:
            return
        raise TextBackendCredentialRefused(f"Fake text backend refused the worker's credentials on {verb}")

    @property
    def ready_calls(self) -> tuple[FakeReadyCall, ...]:
        """Every `ready` call received, in order."""
        return tuple(self._calls.ready)

    @property
    def describe_call_count(self) -> int:
        """How many times `describe` was called."""
        return self._calls.describe_count

    @property
    def capabilities_call_count(self) -> int:
        """How many times `capabilities` was called."""
        return self._calls.capabilities_count

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

        Raises:
            TextBackendCredentialRefused: This stand-in is scripted to refuse the caller's credential.
        """
        ready_index = len(self._calls.ready)
        self._calls.ready.append(FakeReadyCall(deadline_seconds=deadline_seconds))
        self._raise_if_credential_refused("ready")
        if ready_index >= len(self._ready_results):
            return True
        return self._ready_results[ready_index]

    async def describe(self) -> TextBackendDescription:
        """Return the configured description, recording the call.

        Raises:
            TextBackendCredentialRefused: This stand-in is scripted to refuse the caller's credential.
        """
        self._calls.describe_count += 1
        self._raise_if_credential_refused("describe")
        return self._description

    async def capabilities(self) -> TextBackendCapabilities:
        """Return the configured capabilities, recording the call.

        Raises:
            TextBackendCredentialRefused: This stand-in is scripted to refuse the caller's credential.
        """
        self._calls.capabilities_count += 1
        self._raise_if_credential_refused("capabilities")
        return TextBackendCapabilities(generation_stats=self._generation_stats)

    async def generate(
        self,
        payload: Mapping[str, object],
        *,
        generation_key: str,
        deadline_seconds: float,
        on_progress: TextGenerationProgressCallback | None = None,
    ) -> TextGenerationResult:
        """Return the configured text for this payload, or raise the next scripted failure.

        The call is recorded before anything else, so a caller can assert on a payload that then failed.
        A scripted failure is raised immediately, as a real backend's refusal or busy answer arrives
        without waiting for a generation. Otherwise the configured chunks arrive one at a time through
        `on_progress`, the configured latency elapses, and a generation that outlives the caller's
        deadline ends the way the real driver's timeout would.

        Args:
            payload: The payload, recorded untouched.
            generation_key: The generation's identifier, recorded.
            deadline_seconds: The caller's deadline, honoured against the configured latency.
            on_progress: Handed one report per chunk, as the real driver does.

        Returns:
            The configured response text, with the configured finish reason and, where this stand-in
            reports statistics, the token counts its final sample held.

        Raises:
            TextBackendCredentialRefused: This stand-in is scripted to refuse the caller's credential.
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
        self._raise_if_credential_refused("generate")

        scripted_failure = self._scripted_failure(generation_index)
        if scripted_failure is not None:
            raise scripted_failure

        if self._latency_seconds > deadline_seconds:
            await asyncio.sleep(deadline_seconds)
            raise TextBackendUnavailable(
                f"Fake text backend took longer than {deadline_seconds}s to produce generation {generation_key}",
            )

        chunks = self._chunks_for(payload)
        await self._stream_chunks_to(chunks, on_progress=on_progress)

        if self._stalls_after_chunks:
            await asyncio.Event().wait()

        await asyncio.sleep(self._latency_seconds)
        return TextGenerationResult(
            text="".join(chunks),
            finish_reason=self._finish_reason,
            prompt_tokens=self._stats_prompt_tokens if self._generation_stats else None,
            generated_tokens=len(chunks) * self._stats_tokens_per_chunk if self._generation_stats else None,
        )

    def _chunks_for(self, payload: Mapping[str, object]) -> tuple[str, ...]:
        """Return the pieces this generation arrives in, which together are the generated text."""
        if self._stream_chunks:
            return self._stream_chunks
        return (self._resolve_response_text(payload),)

    async def _stream_chunks_to(
        self,
        chunks: Sequence[str],
        *,
        on_progress: TextGenerationProgressCallback | None,
    ) -> None:
        """Deliver each chunk after its interval, reporting progress the way the real driver does."""
        started_at = time.monotonic()
        characters_received = 0
        for chunk_index, chunk in enumerate(chunks, start=1):
            if self._chunk_interval_seconds > 0:
                await asyncio.sleep(self._chunk_interval_seconds)
            characters_received += len(chunk)
            if on_progress is None:
                continue
            on_progress(
                self._progress_after(
                    chunks_received=chunk_index,
                    characters_received=characters_received,
                    elapsed_seconds=time.monotonic() - started_at,
                ),
            )

    def _progress_after(
        self,
        *,
        chunks_received: int,
        characters_received: int,
        elapsed_seconds: float,
    ) -> TextGenerationProgress:
        """Return the progress report for a generation that has delivered this many chunks.

        Token counts and a rate appear only for a stand-in whose backend exposes statistics, because
        that is the only place the real driver has them from: a chunk is an arbitrary number of tokens,
        so counting chunks or characters would be a guess wearing a count's clothes.
        """
        if not self._generation_stats:
            return TextGenerationProgress(
                chunks_received=chunks_received,
                characters_received=characters_received,
                elapsed_seconds=elapsed_seconds,
                last_chunk_at=time.monotonic(),
            )

        completion_tokens = chunks_received * self._stats_tokens_per_chunk
        return TextGenerationProgress(
            chunks_received=chunks_received,
            characters_received=characters_received,
            completion_tokens=completion_tokens,
            prompt_tokens=self._stats_prompt_tokens,
            elapsed_seconds=elapsed_seconds,
            last_chunk_at=time.monotonic(),
            tokens_per_second=completion_tokens / elapsed_seconds if elapsed_seconds > 0 else None,
        )

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
