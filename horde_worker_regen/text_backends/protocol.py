"""The five-verb contract between the worker and a text-inference program it does not own.

A text backend is a separate program (koboldcpp today, sonar next) that the worker launches and then
talks to over HTTP. The worker cannot see inside it, cannot depend on its internals staying put, and
cannot fix it when it misbehaves, so the boundary is deliberately narrow:
[`TextBackend`][horde_worker_regen.text_backends.protocol.TextBackend] exposes `ready`, `describe`,
`generate`, `stop` and `close` and nothing else. Every additional fact the worker tries to learn about a
backend is another place the two programs can disagree, so the protocol grows only when a backend has
demonstrated a stable interface for something the worker actually needs.

Everything above this package (the flow that pops horde jobs, the lifecycle that launches the binary)
sees these five verbs, the two result models, and the three exception types. It never sees HTTP: no
status codes, no routes, no response bodies. That keeps the retry decision where it belongs. The flow
decides what to do about a failure because only the flow knows about the job, the horde and the
operator's configuration; a driver that retried on its own behalf would be making that decision blind.

Public surface:

- [`TextBackend`][horde_worker_regen.text_backends.protocol.TextBackend]: the protocol every driver
  and the in-process fake satisfy.
- [`TextBackendDescription`][horde_worker_regen.text_backends.protocol.TextBackendDescription] and
  [`TextGenerationResult`][horde_worker_regen.text_backends.protocol.TextGenerationResult]: the two
  values the protocol returns.
- [`TextBackendError`][horde_worker_regen.text_backends.protocol.TextBackendError] and its three
  subclasses: every way a call can fail.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class TextBackendError(RuntimeError):
    """Base class for every failure a text backend call reports.

    Callers that do not care which failure occurred (shutdown paths, logging) catch this; callers that
    decide what happens to the job catch the three subclasses, which each name one outcome.
    """


class TextBackendUnavailable(TextBackendError):
    """The backend could not be reached or answered in a way that says it is not working.

    Raised for a refused or reset connection, a name that does not resolve, a request that outlived its
    deadline, a server error other than "busy", and a successful status carrying a body the driver
    cannot read. The flow treats the backend as down: it stops popping jobs and goes back to polling
    [`TextBackend.ready`][horde_worker_regen.text_backends.protocol.TextBackend.ready] until the
    backend answers again.
    """


class TextBackendBusy(TextBackendError):
    """The backend is already generating and will not accept a second request.

    The KoboldAI convention for this is HTTP 503, which both koboldcpp and sonar send when their single
    generation slot is taken. The flow waits and retries the same job: the backend is healthy and the
    payload is fine, so nothing about the job needs to change.
    """


class TextBackendRejectedPayload(TextBackendError):
    """The backend refused the payload and will refuse it again.

    Raised for the statuses a backend uses to say the request itself is wrong. The job cannot succeed
    on this backend no matter how often it is offered, so the flow faults it back to the horde without
    retrying rather than burning the same failure repeatedly.
    """


class TextBackendDescription(BaseModel):
    """Represents what the worker advertises to the horde on behalf of a text backend.

    Every field comes from the backend's own answer rather than from worker configuration, because the
    backend is the only thing that knows which weights it actually loaded and what caps it will honour.
    Advertising anything else is a promise the worker cannot keep.
    """

    # The backend's own name for its model is `model_name`, and the flow's job of deriving an advertised
    # horde name from it reads that spelling; pydantic's `model_` guard is dropped rather than the field
    # renamed to something the protocol's users would have to translate.
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model_name: str
    """The model name exactly as the backend reports it, prefix and all."""
    max_context_length: int
    """The largest prompt-plus-generation token count the backend says it will accept."""
    max_length: int
    """The largest number of tokens the backend says it will generate for one request."""
    soft_prompts: tuple[str, ...] = ()
    """The soft prompts the backend exposes; empty when it has none or does not implement them."""


class TextGenerationResult(BaseModel):
    """Represents one completed text generation.

    A model rather than a bare `str` even with a single field, so that a later addition (the seed a
    backend used, per-request token counts) extends the model instead of changing every signature that
    carries a generation result.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    """The generated text, exactly as the backend returned it."""


@runtime_checkable
class TextBackend(Protocol):
    """The whole of what the worker may ask a text-inference program to do.

    Implementations are drivers for one HTTP dialect
    ([`KoboldApiTextBackend`][horde_worker_regen.text_backends.kobold_api.KoboldApiTextBackend]) or
    in-process stand-ins
    ([`FakeTextBackend`][horde_worker_regen.text_backends.fake_text_backend.FakeTextBackend]). They
    never retry and never decide a job's fate: a call either returns its value or raises one of the
    three [`TextBackendError`][horde_worker_regen.text_backends.protocol.TextBackendError] subclasses,
    and the caller acts on that.

    Runtime-checkable so a test or a dry-run assembly can assert that a stand-in still speaks the
    protocol; the check confirms the methods are present, not that they behave.
    """

    async def ready(self, *, deadline_seconds: float) -> bool:
        """Return whether the backend is up with a model loaded, answering within the deadline.

        One request, one answer. Returns `False` on a connection failure, a timeout, or an answer the
        driver cannot make sense of, rather than raising: the caller polls this while a freshly launched
        backend is still starting, and a refused connection is the expected reading during that window,
        not an error. Callers infer nothing else about liveness from anywhere else.

        Args:
            deadline_seconds: How long to wait for the answer before reporting the backend not ready.

        Returns:
            `True` when the backend answered and has a model loaded, `False` otherwise.
        """
        ...

    async def describe(self) -> TextBackendDescription:
        """Return the facts the worker advertises to the horde for this backend.

        Called once the backend is ready and again after anything that could have changed which model
        is loaded.

        Returns:
            The backend's reported model name, its context and generation caps, and its soft prompts.

        Raises:
            TextBackendUnavailable: The backend did not answer, or answered something unreadable.
        """
        ...

    async def generate(
        self,
        payload: Mapping[str, object],
        *,
        generation_key: str,
        deadline_seconds: float,
    ) -> TextGenerationResult:
        """Return the text the backend generates for a popped horde payload.

        The payload is forwarded verbatim. The horde owns its shape and the backend owns its
        interpretation, so a key the driver filtered, renamed or defaulted would be a key the worker is
        now responsible for getting right on every future horde and backend release. The only addition
        is the generation key, which exists so
        [`stop`][horde_worker_regen.text_backends.protocol.TextBackend.stop] can name this request
        later.

        Args:
            payload: The popped horde payload, forwarded untouched.
            generation_key: An identifier for this generation, unique among in-flight requests.
            deadline_seconds: How long to wait for the generation before giving up on it.

        Returns:
            The generated text.

        Raises:
            TextBackendBusy: The backend is already generating; the same job can be offered again.
            TextBackendRejectedPayload: The backend refuses this payload and always will.
            TextBackendUnavailable: The backend is unreachable, too slow, or answered unreadably.
        """
        ...

    async def stop(self, *, generation_key: str) -> None:
        """Ask the backend to abandon the generation with this key.

        Used on shutdown and when a generation outlives its deadline. Best effort by design: a backend
        with no abort route, or one that has already exited, leaves nothing worth failing over, so the
        outcome is logged at debug and the call returns. Never raises.

        Args:
            generation_key: The key passed to the
                [`generate`][horde_worker_regen.text_backends.protocol.TextBackend.generate] call being
                abandoned.
        """
        ...

    async def close(self) -> None:
        """Release the driver's own resources. Idempotent, so shutdown paths may call it more than once."""
        ...


__all__ = [
    "TextBackend",
    "TextBackendBusy",
    "TextBackendDescription",
    "TextBackendError",
    "TextBackendRejectedPayload",
    "TextBackendUnavailable",
    "TextGenerationResult",
]
