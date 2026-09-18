"""The driver for backends that speak the KoboldAI HTTP API.

koboldcpp and sonar are different programs with different internals, but both expose the KoboldAI
routes, so one driver serves both and the worker does not grow a branch per backend.
[`KoboldApiTextBackend`][horde_worker_regen.text_backends.kobold_api.KoboldApiTextBackend] is the only
place in the worker that knows those routes exist; above it there is just the protocol in
[`protocol`][horde_worker_regen.text_backends.protocol].

Generation is driven through the backend's server-sent-events stream rather than the blocking route.
Both carry the same payload and finish at the same moment, and the stream costs nothing in throughput,
but the stream says something while it runs: how much text has arrived and when. Without that the worker
learns nothing until a generation returns, so a slow generation and a wedged backend look identical for
minutes at a time. Token counts are a separate question the stream cannot answer, because one record
carries an arbitrary number of tokens; those come from the statistics route on a backend that has one.
The blocking route remains the fallback for a backend whose stream route is missing or refuses to start.

Two choices are worth stating because they look like omissions:

- The driver never retries. A failure becomes one of the three protocol exceptions and goes straight
  up. Whether to wait and re-offer the job, fault it back to the horde, or stop popping altogether
  depends on the job and the worker's state, neither of which the driver can see.
- The driver never touches the payload. It forwards what the horde sent and adds the generation key.
  Filtering or defaulting a key would make the worker responsible for a shape owned by the horde on one
  side and interpreted by a program the worker does not control on the other.

Routes and response shapes are those koboldcpp and sonar both implement, and the driver reads only the
caps koboldcpp is willing to advertise to the horde.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import Final, TypeVar

import aiohttp
from aiohttp import hdrs
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from horde_worker_regen.text_backends.protocol import (
    TextBackendBusy,
    TextBackendCapabilities,
    TextBackendCredentialRefused,
    TextBackendDescription,
    TextBackendRejectedPayload,
    TextBackendUnavailable,
    TextGenerationProgress,
    TextGenerationProgressCallback,
    TextGenerationResult,
)


class KoboldApiRoutes:
    """The KoboldAI routes this driver uses, and only those."""

    MODEL: Final = "/api/v1/model"
    """Answers with the loaded model's name. Both backends implement it, and a successful answer proves
    a model is loaded, which is exactly what readiness means here. `/api/v1/info/version` and `/health`
    answer while a backend is still loading weights, so neither can stand in for it."""
    MAX_CONTEXT_LENGTH: Final = "/api/v1/config/max_context_length"
    """The context cap koboldcpp advertises to the horde, which is not always its true cap. The true one
    lives at `/api/extra/true_max_context_length`; koboldcpp deliberately keeps that off the horde and so
    does the worker."""
    MAX_LENGTH: Final = "/api/v1/config/max_length"
    """The generation-length cap the backend advertises to the horde."""
    SOFT_PROMPTS_LIST: Final = "/api/v1/config/soft_prompts_list"
    """The soft prompts the backend exposes. Optional: a backend may answer 404 or an empty list."""
    GENERATE: Final = "/api/v1/generate"
    """Runs one blocking generation and answers with the generated text. The fallback route."""
    GENERATE_STREAM: Final = "/api/extra/generate/stream"
    """Runs one generation and streams it back, same body as `GENERATE`. An extension route, and the
    one generations normally go through, so a backend without it falls back to the blocking route."""
    GENERATION_STATS: Final = "/api/extra/generate/stats"
    """Answers per-request token counts and timings for a generation named by key. Present only on a
    patched build; a stock backend has no such route and every count stays unknown."""
    PERF: Final = "/api/extra/perf"
    """Answers counters for the last finished generation. The only token counts a stock backend offers,
    and only for a generation that ran on its own."""
    ABORT: Final = "/api/extra/abort"
    """Abandons the generation naming a key. An extension route, so a backend may not have it."""


class KoboldApiJsonKeys:
    """The JSON keys the KoboldAI routes use, for the response models below."""

    RESULT: Final = "result"
    VALUE: Final = "value"
    VALUES: Final = "values"
    RESULTS: Final = "results"
    TEXT: Final = "text"
    TOKEN: Final = "token"
    """The field on a stream record carrying that record's text, which is one or more tokens."""
    FINISH_REASON: Final = "finish_reason"
    """Why a generation ended. Set on the last stream record; `None` on every record before it, and
    absent altogether from sonar's records."""
    GENERATION_KEY: Final = "genkey"
    """The KoboldAI field name for a generation's identifier, used on generate, abort and stats."""
    FOUND: Final = "found"
    """Whether the statistics route recognised the generation key it was asked about."""
    PROMPT_TOKENS: Final = "prompt_tokens"
    COMPLETION_TOKENS: Final = "completion_tokens"
    GENERATION_SECONDS: Final = "generation_seconds"
    LAST_TOKEN_COUNT: Final = "last_token_count"
    LAST_INPUT_COUNT: Final = "last_input_count"
    SUCCESS: Final = "success"
    DONE: Final = "done"


class KoboldApiTimeouts:
    """Deadlines for the calls whose protocol signature carries none.

    `ready` and `generate` are given a deadline by their caller, because how long to wait for them is a
    scheduling decision. The metadata reads, the statistics samples and the abort are small, fixed
    requests against a local program, so they carry their own ceiling rather than inheriting the
    session's default, which is minutes long and would let a wedged backend hold a shutdown open.
    """

    DESCRIBE_SECONDS: Final = 10.0
    ABORT_SECONDS: Final = 10.0
    STATS_SECONDS: Final = 5.0
    PERF_SECONDS: Final = 5.0


STATS_POLL_INTERVAL_SECONDS: Final = 0.5
"""How often the statistics route is sampled while a generation's stream is open.

Sized against what the numbers are for: a progress row a person reads, which gains nothing from a faster
clock, against a request per sample on a program that is busy generating. The route reads counters under
a lock the stream poll already takes every twenty milliseconds, so the cost is the HTTP round trip.
"""

CAPABILITY_PROBE_GENERATION_KEY: Final = "regen-capability-probe"
"""The key the statistics route is probed with: deliberately not a generation key anything will use.

The probe asks whether the route exists at all, so it wants an answer that costs the backend nothing.
A key belonging to no generation gets `found: false` from a backend that has the route and a 404 from
one that does not, which is the whole of the question.
"""

_SSE_DATA_FIELD_PREFIX: Final = "data:"
"""The server-sent-events field carrying a record's payload. Both backends send exactly one per record."""

_AUTHORIZATION_BEARER_PREFIX: Final = "Bearer "
"""koboldcpp launched with `--password` accepts the password as a bearer token."""

_REJECTED_PAYLOAD_STATUSES: Final = frozenset({HTTPStatus.BAD_REQUEST, HTTPStatus.UNPROCESSABLE_ENTITY})
"""The statuses that mean the request itself is wrong, so re-offering the same job cannot help."""

_REFUSED_CREDENTIAL_STATUSES: Final = frozenset({HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN})
"""The statuses that mean the credential is wrong rather than the request or the backend.

koboldcpp guards its text routes with the password it was launched with and answers 401 to a request
carrying the wrong one or none; 403 is here because a backend behind a proxy answers that way for the
same thing. The model route is the exception: it answers 200 naming `koboldcpp/protected-model`, so a
refused credential can first show at the generation rather than at the readiness probe.
"""

_BODY_SUMMARY_CHARACTER_LIMIT: Final = 300
"""How much of an unreadable body is quoted in an exception message: enough to recognise what answered."""


class _RouteNotImplemented(TextBackendUnavailable):
    """A route answered 404, so this backend does not implement it.

    A subclass of `TextBackendUnavailable` so a caller that does not care still sees the failure the
    protocol promises, while the reads of optional routes catch this narrower type and carry on.
    """


class _StreamUnusable(Exception):
    """The stream could not be started, or started and produced nothing, so this request goes blocking.

    Not a `TextBackendError`: nothing above the driver ever sees it, because the only place it is raised
    is inside the one call that catches it and falls back. A generation that had already begun to arrive
    and then broke is a different thing entirely and is reported as an unavailable backend, because the
    text it produced is incomplete and must not be submitted as an answer.
    """


@dataclass(frozen=True)
class _RawResponse:
    """One HTTP answer reduced to what the driver decides on: the status and the undecoded body."""

    status: int
    body_text: str


class _KoboldResponseModel(BaseModel):
    """Base for the response models, frozen and tolerant of fields the driver does not read."""

    model_config = ConfigDict(frozen=True, extra="ignore")


_KoboldResponseModelT = TypeVar("_KoboldResponseModelT", bound=_KoboldResponseModel)


class _KoboldModelNameResponse(_KoboldResponseModel):
    """Represents the answer from the model route, whose only field is the loaded model's name."""

    result: str


class _KoboldConfigValueResponse(_KoboldResponseModel):
    """Represents the answer from either `config/...` cap route."""

    value: int


class _KoboldSoftPromptsResponse(_KoboldResponseModel):
    """Represents the answer from the soft-prompts route, which is an empty list on both backends today."""

    values: tuple[str, ...] = ()


class _KoboldGenerationEntry(_KoboldResponseModel):
    """Represents one entry of a blocking generate answer's `results` list."""

    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class _KoboldGenerateResponse(_KoboldResponseModel):
    """Represents a successful blocking generate answer.

    `results` is required and non-empty: a 200 carrying neither is a backend answering nonsense, which
    the driver reports as unavailable rather than inventing an empty generation.
    """

    results: tuple[_KoboldGenerationEntry, ...] = Field(min_length=1)


class _KoboldStreamRecord(_KoboldResponseModel):
    """Represents one record of a generation stream.

    `token` is that record's text, which is one or more tokens depending on how the backend buffered
    them. `finish_reason` is set on the last record of a koboldcpp stream and never present at all on
    sonar's, so its absence says nothing about the stream being healthy.
    """

    token: str = ""
    finish_reason: str | None = None


class _KoboldGenerationStats(_KoboldResponseModel):
    """Represents the statistics route's answer about one generation.

    Every field past `found` is optional because the route answers two different shapes: the full one
    for a request the backend is running in its batching path, and a smaller one for a request on the
    legacy single-generation path, which knows its prompt tokens only once it has finished.
    """

    found: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    generation_seconds: float | None = None
    finished: bool = False


class _KoboldPerfResponse(_KoboldResponseModel):
    """Represents the counters a backend keeps about the generation that finished most recently.

    Process-global and overwritten by every generation, so they describe one generation only when one
    generation ran.
    """

    last_token_count: int | None = None
    last_input_count: int | None = None


class _KoboldAbortResponse(_KoboldResponseModel):
    """Represents an abort answer.

    koboldcpp sends the two flags as the strings `"true"` and `"false"` rather than JSON booleans, and
    sonar answers `{}`, so both fields are strings with a default and are only logged.
    """

    success: str = ""
    done: str = ""


def _summarise_body(body_text: str) -> str:
    """Return a short, single-line rendering of a response body for an exception or log message."""
    collapsed = " ".join(body_text.split())
    if len(collapsed) <= _BODY_SUMMARY_CHARACTER_LIMIT:
        return collapsed
    return f"{collapsed[:_BODY_SUMMARY_CHARACTER_LIMIT]}..."


def _reported_count(count: int | None) -> int | None:
    """Return a token count the backend actually knows, or None for the zero it sends when it does not.

    The statistics route answers zero prompt tokens for a generation on the legacy path that has not
    finished, and the perf counters are zero before any generation has run, so a zero is "not known
    yet" rather than "no tokens".
    """
    if count is None or count <= 0:
        return None
    return count


class _StreamedGeneration:
    """Accumulates one generation's stream and statistics into the progress its caller is shown.

    Two writers, one reader: the loop reading the stream appends text, and the statistics sampler (a
    task running beside it) records counts and timings. Both then build a
    [`TextGenerationProgress`][horde_worker_regen.text_backends.protocol.TextGenerationProgress] from
    whatever is known at that moment, so a chunk report carries the last counts seen and a statistics
    report carries the text arrived so far.

    Thread Safety:
        Not thread-safe and does not need to be: both writers run on the same event loop, and neither
        awaits between reading a field and writing it.
    """

    def __init__(self, *, started_at: float) -> None:
        """Begin tracking a generation that was requested at `started_at`.

        Args:
            started_at: The `time.monotonic()` reading when the request went out, which every elapsed
                figure is measured from.
        """
        self._started_at = started_at
        self._text_pieces: list[str] = []
        self._chunks_received = 0
        self._characters_received = 0
        self._last_chunk_at: float | None = None
        self._finish_reason: str | None = None
        self._prompt_tokens: int | None = None
        self._completion_tokens: int | None = None
        self._tokens_per_second: float | None = None
        self._last_counted_sample: tuple[float, int] | None = None
        """When the last sample that carried a generated-token count arrived, and that count."""

    @property
    def text(self) -> str:
        """The generated text as far as it has arrived."""
        return "".join(self._text_pieces)

    @property
    def chunks_received(self) -> int:
        """How many stream records have arrived, which is not a token count."""
        return self._chunks_received

    @property
    def finish_reason(self) -> str | None:
        """Why the backend said it stopped, or None when it has not said."""
        return self._finish_reason

    @property
    def prompt_tokens(self) -> int | None:
        """Prompt tokens from the last statistics sample that knew them, or None."""
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int | None:
        """Generated tokens from the last statistics sample that knew them, or None."""
        return self._completion_tokens

    def note_record(self, record: _KoboldStreamRecord) -> None:
        """Take in one stream record: its text, its arrival, and a finish reason if it carries one."""
        self._chunks_received += 1
        self._last_chunk_at = time.monotonic()
        if record.token:
            self._text_pieces.append(record.token)
            self._characters_received += len(record.token)
        if record.finish_reason is not None:
            self._finish_reason = record.finish_reason

    def note_stats(self, stats: _KoboldGenerationStats) -> None:
        """Take in one statistics sample, keeping only the figures it actually knows.

        A sample that has lost a count it reported before (the legacy path answers zero prompt tokens
        until the generation finishes) leaves the earlier figure in place rather than unsetting it.
        """
        prompt_tokens = _reported_count(stats.prompt_tokens)
        if prompt_tokens is not None:
            self._prompt_tokens = prompt_tokens

        completion_tokens = _reported_count(stats.completion_tokens)
        if completion_tokens is None:
            return
        self._completion_tokens = completion_tokens
        sampled_at = time.monotonic()
        # The backend's own generation seconds first, never the worker's elapsed time since the request:
        # elapsed covers queueing and prompt processing, so dividing by it would report a rate the backend
        # never ran at. A backend that counts tokens live but only stores its timings at finish reports
        # zero seconds throughout; there the growth between two counted samples over the wall clock
        # between them is generation time alone, so it is the rate the backend is running at.
        if stats.generation_seconds is not None and stats.generation_seconds > 0:
            self._tokens_per_second = completion_tokens / stats.generation_seconds
        elif self._last_counted_sample is not None:
            previous_at, previous_count = self._last_counted_sample
            grown = completion_tokens - previous_count
            if grown > 0 and sampled_at > previous_at:
                self._tokens_per_second = grown / (sampled_at - previous_at)
        self._last_counted_sample = (sampled_at, completion_tokens)

    def snapshot(self) -> TextGenerationProgress:
        """Return what is known about this generation right now."""
        return TextGenerationProgress(
            chunks_received=self._chunks_received,
            characters_received=self._characters_received,
            completion_tokens=self._completion_tokens,
            prompt_tokens=self._prompt_tokens,
            elapsed_seconds=time.monotonic() - self._started_at,
            last_chunk_at=self._last_chunk_at,
            tokens_per_second=self._tokens_per_second,
        )


def _report_progress(
    on_progress: TextGenerationProgressCallback | None,
    generation: _StreamedGeneration,
) -> None:
    """Hand the caller the current progress, doing nothing when it asked for none."""
    if on_progress is None:
        return
    on_progress(generation.snapshot())


class KoboldApiTextBackend:
    """A [`TextBackend`][horde_worker_regen.text_backends.protocol.TextBackend] over the KoboldAI HTTP API.

    Satisfies the protocol structurally; it does not inherit from it, so nothing above this module has
    to know the driver exists to accept one.

    Connection Management:
        The `aiohttp` session is injected rather than created here. The parent process already owns one
        session through `ApiSessions`, and that one instance owns connection pooling and its own
        shutdown; a driver that made its own would duplicate both and leak a connector on every backend
        replacement. Consequently
        [`close`][horde_worker_regen.text_backends.kobold_api.KoboldApiTextBackend.close] releases only
        the driver's own state and leaves the session alone.

    Concurrency:
        Safe to drive several generations at once on one event loop; each owns its own stream and its
        own statistics sampler. The one figure that cannot be shared is the perf counters, which are
        process-global in the backend, so they are read only when this driver has a single generation
        running.
    """

    def __init__(self, base_url: str, session: aiohttp.ClientSession, *, password: str | None = None) -> None:
        """Create a driver for the backend serving at `base_url`.

        Args:
            base_url: Where the backend listens, with or without a trailing slash and optionally with a
                path prefix (both backends match their routes by suffix).
            session: The caller's shared session. Not closed by this driver.
            password: The password the backend was launched with, if any, sent as a bearer token.
        """
        self._base_url = base_url.rstrip("/")
        self._session = session
        self._password = password
        self._closed = False
        self._abort_route_missing_logged = False
        self._stream_route_missing = False
        self._blocking_fallback_logged = False
        self._generation_stats_present: bool | None = None
        self._generations_in_flight = 0

    @property
    def base_url(self) -> str:
        """The backend's base URL with any trailing slash removed, as used to build every request."""
        return self._base_url

    def _url(self, route: str) -> str:
        """Return the absolute URL for one route.

        Args:
            route: One of the paths in
                [`KoboldApiRoutes`][horde_worker_regen.text_backends.kobold_api.KoboldApiRoutes].

        Returns:
            The base URL with the route appended.
        """
        return f"{self._base_url}{route}"

    def _headers(self) -> dict[str, str]:
        """Return the request headers, carrying the bearer token only when a password was configured."""
        if self._password is None:
            return {}
        return {hdrs.AUTHORIZATION: f"{_AUTHORIZATION_BEARER_PREFIX}{self._password}"}

    async def _request(
        self,
        *,
        method: str,
        route: str,
        timeout_seconds: float,
        json_body: Mapping[str, object] | None = None,
    ) -> _RawResponse:
        """Return the status and body of one request, letting transport failures propagate.

        Deliberately does no error mapping: each verb maps transport failures to the protocol exception
        its own contract names, and `ready` maps them to `False` instead.

        Raises:
            aiohttp.ClientError: The request could not be completed.
            TimeoutError: The answer did not arrive within `timeout_seconds`.
        """
        async with self._session.request(
            method,
            self._url(route),
            json=json_body,
            headers=self._headers(),
            timeout=aiohttp.ClientTimeout(total=timeout_seconds),
        ) as response:
            return _RawResponse(status=response.status, body_text=await response.text())

    async def _read_json(
        self,
        *,
        route: str,
        response_model: type[_KoboldResponseModelT],
        timeout_seconds: float,
    ) -> _KoboldResponseModelT:
        """Return the parsed answer from a GET, as `response_model`.

        Raises:
            _RouteNotImplemented: The backend answered 404, so it has no such route.
            TextBackendCredentialRefused: The backend refused the worker's credentials.
            TextBackendUnavailable: The backend did not answer, answered another non-200 status, or
                answered a body that is not the expected shape.
        """
        try:
            raw = await self._request(method=hdrs.METH_GET, route=route, timeout_seconds=timeout_seconds)
        except (aiohttp.ClientError, TimeoutError) as request_error:
            raise TextBackendUnavailable(
                f"Text backend at {self._base_url} did not answer {route}: {request_error}",
            ) from request_error

        self._raise_for_refused_credential(raw, detail=f"reading {route}")

        if raw.status == HTTPStatus.NOT_FOUND:
            raise _RouteNotImplemented(
                f"Text backend at {self._base_url} has no {route} route",
            )

        if raw.status != HTTPStatus.OK:
            raise TextBackendUnavailable(
                f"Text backend at {self._base_url} answered {route} with status {raw.status}: "
                f"{_summarise_body(raw.body_text)}",
            )

        try:
            return response_model.model_validate_json(raw.body_text)
        except ValidationError as parse_error:
            raise TextBackendUnavailable(
                f"Text backend at {self._base_url} answered {route} with an unreadable body: "
                f"{_summarise_body(raw.body_text)}",
            ) from parse_error

    def _raise_for_refused_credential(self, raw: _RawResponse, *, detail: str) -> None:
        """Raise when an answer says the worker's credentials were refused, returning otherwise.

        Every route goes through this before its own status handling, so a refused credential is never
        mistaken for a route the backend lacks or for a backend that is down: those are polled through,
        and this one never resolves on its own.

        Raises:
            TextBackendCredentialRefused: The status was 401 or 403.
        """
        if raw.status not in _REFUSED_CREDENTIAL_STATUSES:
            return
        raise TextBackendCredentialRefused(
            f"Text backend at {self._base_url} refused the worker's credentials {detail} "
            f"(status {raw.status}): {_summarise_body(raw.body_text)}",
        )

    async def ready(self, *, deadline_seconds: float) -> bool:
        """Return whether the model route answers within the deadline.

        Every failure is a `False`, logged at debug: the caller polls this while a backend is still
        starting, so a refused connection is the normal reading during that window rather than
        something to raise about.

        Args:
            deadline_seconds: How long to wait for the answer.

        Returns:
            `True` when the backend named a loaded model, `False` otherwise.

        Raises:
            TextBackendCredentialRefused: The backend refused the worker's credentials, which polling
                cannot fix.
        """
        try:
            await self._read_json(
                route=KoboldApiRoutes.MODEL,
                response_model=_KoboldModelNameResponse,
                timeout_seconds=deadline_seconds,
            )
        except TextBackendUnavailable as not_ready:
            logger.debug(f"Text backend at {self._base_url} is not ready: {not_ready}")
            return False

        return True

    async def describe(self) -> TextBackendDescription:
        """Return the model name and caps the backend advertises, plus any soft prompts it exposes.

        Raises:
            TextBackendUnavailable: Any of the required routes did not answer or answered unreadably.
        """
        model_response = await self._read_json(
            route=KoboldApiRoutes.MODEL,
            response_model=_KoboldModelNameResponse,
            timeout_seconds=KoboldApiTimeouts.DESCRIBE_SECONDS,
        )
        max_context_length_response = await self._read_json(
            route=KoboldApiRoutes.MAX_CONTEXT_LENGTH,
            response_model=_KoboldConfigValueResponse,
            timeout_seconds=KoboldApiTimeouts.DESCRIBE_SECONDS,
        )
        max_length_response = await self._read_json(
            route=KoboldApiRoutes.MAX_LENGTH,
            response_model=_KoboldConfigValueResponse,
            timeout_seconds=KoboldApiTimeouts.DESCRIBE_SECONDS,
        )

        return TextBackendDescription(
            model_name=model_response.result,
            max_context_length=max_context_length_response.value,
            max_length=max_length_response.value,
            soft_prompts=await self._read_soft_prompts(),
        )

    async def _read_soft_prompts(self) -> tuple[str, ...]:
        """Return the soft prompts the backend exposes, empty when it exposes or implements none.

        Soft prompts are optional on both backends, so a missing route reads as "none" rather than as a
        broken backend; anything else the route does wrong is still a failure.

        Raises:
            TextBackendUnavailable: The route answered a status other than 200 or 404, or an unreadable
                body.
        """
        try:
            soft_prompts_response = await self._read_json(
                route=KoboldApiRoutes.SOFT_PROMPTS_LIST,
                response_model=_KoboldSoftPromptsResponse,
                timeout_seconds=KoboldApiTimeouts.DESCRIBE_SECONDS,
            )
        except _RouteNotImplemented as no_soft_prompts_route:
            logger.debug(f"{no_soft_prompts_route}; advertising no soft prompts")
            return ()

        return soft_prompts_response.values

    # region capabilities

    async def capabilities(self) -> TextBackendCapabilities:
        """Return which optional routes this build has, probing each at most once.

        Returns:
            Whether per-request generation statistics can be read from this backend.

        Raises:
            TextBackendCredentialRefused: The backend refused the worker's credentials.
        """
        return TextBackendCapabilities(generation_stats=await self._generation_stats_available())

    async def _generation_stats_available(self) -> bool:
        """Return whether the statistics route answers, probing it the first time this is asked."""
        if self._generation_stats_present is None:
            self._generation_stats_present = await self._probe_generation_stats()
        return self._generation_stats_present

    async def _probe_generation_stats(self) -> bool:
        """Return whether the statistics route exists, asked once with a key no generation will use.

        Anything but a 200 is taken as the route being absent, which is what a stock build answers, and
        is not retried: the route is part of the program, so a build that does not have it now will not
        grow one while it runs. A refused credential is the exception, because the answer then describes
        the worker's password rather than the build.

        Raises:
            TextBackendCredentialRefused: The backend refused the worker's credentials.
        """
        try:
            raw = await self._request(
                method=hdrs.METH_POST,
                route=KoboldApiRoutes.GENERATION_STATS,
                timeout_seconds=KoboldApiTimeouts.STATS_SECONDS,
                json_body={KoboldApiJsonKeys.GENERATION_KEY: CAPABILITY_PROBE_GENERATION_KEY},
            )
        except (aiohttp.ClientError, TimeoutError) as probe_error:
            logger.debug(
                f"Text backend at {self._base_url} did not answer the statistics probe, so token counts "
                f"will be unknown for this backend: {probe_error}",
            )
            return False

        self._raise_for_refused_credential(raw, detail="probing the statistics route")

        if raw.status != HTTPStatus.OK:
            logger.debug(
                f"Text backend at {self._base_url} answered the statistics probe with status "
                f"{raw.status}; it has no {KoboldApiRoutes.GENERATION_STATS} route",
            )
            return False

        logger.debug(f"Text backend at {self._base_url} answers per-request generation statistics")
        return True

    # endregion

    # region generate

    async def generate(
        self,
        payload: Mapping[str, object],
        *,
        generation_key: str,
        deadline_seconds: float,
        on_progress: TextGenerationProgressCallback | None = None,
    ) -> TextGenerationResult:
        """Return the text the backend generates for the popped horde payload.

        Driven through the stream, which reports the text as it arrives; the blocking route serves the
        same request for a backend whose stream route is missing or will not start, and that decision is
        announced once so an operator can see which route their generations are taking.

        The payload is forwarded as it arrived; the generation key is the one field the driver sets, and
        it is what [`stop`][horde_worker_regen.text_backends.kobold_api.KoboldApiTextBackend.stop] names
        later.

        Args:
            payload: The popped horde payload, forwarded untouched.
            generation_key: The identifier for this generation.
            deadline_seconds: How long to wait for the generation.
            on_progress: Called on every chunk of text and every statistics sample, or None for none.

        Returns:
            The generated text, its finish reason where the backend gives one, and its token counts
            where anything the backend offers reported them.

        Raises:
            TextBackendBusy: The backend answered 503, its generation slots being taken.
            TextBackendCredentialRefused: The backend refused the worker's credentials.
            TextBackendRejectedPayload: The backend refused the payload itself.
            TextBackendUnavailable: The backend was unreachable, outlived the deadline, failed with a
                server error, answered without a usable generation, or stopped part way through one.
        """
        self._generations_in_flight += 1
        try:
            if not self._stream_route_missing:
                try:
                    return await self._generate_streamed(
                        payload,
                        generation_key=generation_key,
                        deadline_seconds=deadline_seconds,
                        on_progress=on_progress,
                    )
                except _RouteNotImplemented as no_stream_route:
                    self._stream_route_missing = True
                    self._note_blocking_fallback(str(no_stream_route))
                except _StreamUnusable as stream_unusable:
                    self._note_blocking_fallback(str(stream_unusable))

            return await self._generate_blocking(
                payload,
                generation_key=generation_key,
                deadline_seconds=deadline_seconds,
            )
        finally:
            self._generations_in_flight -= 1

    def _note_blocking_fallback(self, reason: str) -> None:
        """Say once that generations are going through the blocking route, and why.

        Once per driver rather than once per generation: the reason is a property of the backend build
        (or of a backend that refuses to stream at all), so repeating it per job would be one line per
        job for the life of the worker.
        """
        if self._blocking_fallback_logged:
            logger.debug(f"Falling back to {KoboldApiRoutes.GENERATE} for this generation: {reason}")
            return
        self._blocking_fallback_logged = True
        logger.info(
            f"Text generations are going through {KoboldApiRoutes.GENERATE} rather than the stream: "
            f"{reason}. They will still complete; the worker just learns nothing about them until they do.",
        )

    def _generation_request_body(self, payload: Mapping[str, object], *, generation_key: str) -> dict[str, object]:
        """Return the body for one generation: the horde's payload plus the key naming this request."""
        request_body: dict[str, object] = dict(payload)
        request_body[KoboldApiJsonKeys.GENERATION_KEY] = generation_key
        return request_body

    async def _generate_blocking(
        self,
        payload: Mapping[str, object],
        *,
        generation_key: str,
        deadline_seconds: float,
    ) -> TextGenerationResult:
        """Return the generation from the blocking route, which says nothing until it is finished.

        Raises:
            TextBackendBusy: The backend answered 503.
            TextBackendCredentialRefused: The backend refused the worker's credentials.
            TextBackendRejectedPayload: The backend refused the payload.
            TextBackendUnavailable: Anything else, including a 200 with no usable generation.
        """
        try:
            raw = await self._request(
                method=hdrs.METH_POST,
                route=KoboldApiRoutes.GENERATE,
                timeout_seconds=deadline_seconds,
                json_body=self._generation_request_body(payload, generation_key=generation_key),
            )
        except (aiohttp.ClientError, TimeoutError) as request_error:
            raise TextBackendUnavailable(
                f"Text backend at {self._base_url} did not complete generation {generation_key} within "
                f"{deadline_seconds}s: {request_error}",
            ) from request_error

        self._raise_for_generate_status(raw, generation_key=generation_key)

        try:
            generate_response = _KoboldGenerateResponse.model_validate_json(raw.body_text)
        except ValidationError as parse_error:
            raise TextBackendUnavailable(
                f"Text backend at {self._base_url} answered generation {generation_key} with no usable "
                f"'{KoboldApiJsonKeys.RESULTS}' entry: {_summarise_body(raw.body_text)}",
            ) from parse_error

        # The blocking answer carries its own counts, so this path needs no second request to learn
        # them and is not subject to the perf counters' one-generation-at-a-time limitation.
        generation = generate_response.results[0]
        return TextGenerationResult(
            text=generation.text,
            finish_reason=generation.finish_reason,
            prompt_tokens=_reported_count(generation.prompt_tokens),
            generated_tokens=_reported_count(generation.completion_tokens),
        )

    async def _generate_streamed(
        self,
        payload: Mapping[str, object],
        *,
        generation_key: str,
        deadline_seconds: float,
        on_progress: TextGenerationProgressCallback | None,
    ) -> TextGenerationResult:
        """Return the generation assembled from the stream, reporting each record as it arrives.

        Raises:
            _RouteNotImplemented: This backend has no stream route.
            _StreamUnusable: The stream would not start, or started and ended without one record.
            TextBackendBusy: The backend answered 503.
            TextBackendCredentialRefused: The backend refused the worker's credentials.
            TextBackendRejectedPayload: The backend refused the payload.
            TextBackendUnavailable: The generation outlived its deadline, or the stream broke after
                text had already arrived, which leaves an incomplete answer that must not be submitted.
        """
        generation = _StreamedGeneration(started_at=time.monotonic())
        stats_sampler: asyncio.Task[None] | None = None
        try:
            async with self._session.post(
                self._url(KoboldApiRoutes.GENERATE_STREAM),
                json=self._generation_request_body(payload, generation_key=generation_key),
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=deadline_seconds),
            ) as response:
                if response.status != HTTPStatus.OK:
                    self._raise_for_stream_status(
                        _RawResponse(status=response.status, body_text=await response.text()),
                        generation_key=generation_key,
                    )

                if await self._generation_stats_available():
                    stats_sampler = asyncio.create_task(
                        self._sample_generation_stats(
                            generation_key=generation_key,
                            generation=generation,
                            on_progress=on_progress,
                        ),
                    )

                async for record in self._read_stream_records(response):
                    generation.note_record(record)
                    _report_progress(on_progress, generation)
        except TimeoutError as stream_timeout:
            raise TextBackendUnavailable(
                f"Text backend at {self._base_url} did not complete generation {generation_key} within "
                f"{deadline_seconds}s: {stream_timeout}",
            ) from stream_timeout
        except (aiohttp.ClientError, ValidationError, UnicodeDecodeError) as stream_error:
            if generation.chunks_received == 0:
                raise _StreamUnusable(
                    f"the stream for generation {generation_key} failed before any text arrived: {stream_error}",
                ) from stream_error
            raise TextBackendUnavailable(
                f"Text backend at {self._base_url} stopped part way through generation {generation_key} "
                f"after {generation.chunks_received} chunk(s): {stream_error}",
            ) from stream_error
        finally:
            await _cancel_sampler(stats_sampler)

        if generation.chunks_received == 0:
            raise _StreamUnusable(f"the stream for generation {generation_key} ended without a single record")

        prompt_tokens, generated_tokens = await self._finished_token_counts(generation)
        return TextGenerationResult(
            text=generation.text,
            finish_reason=generation.finish_reason,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
        )

    def _raise_for_stream_status(self, raw: _RawResponse, *, generation_key: str) -> None:
        """Raise what a non-200 on the stream route means, which is not always what the flow sees.

        A busy backend and a refused payload are the same answers the blocking route gives and mean the
        same things, so they go straight up. A missing route and any other failure are about the stream
        rather than about the job, so they end in the blocking route instead of failing the generation.

        Raises:
            _RouteNotImplemented: The status was 404.
            TextBackendBusy: The status was 503.
            TextBackendCredentialRefused: The status was 401 or 403.
            TextBackendRejectedPayload: The status says the payload itself is wrong.
            _StreamUnusable: Any other non-200 status.
        """
        self._raise_for_refused_credential(raw, detail=f"starting generation {generation_key}")

        failure_detail = (
            f"backend at {self._base_url}, generation {generation_key}, status {raw.status}: "
            f"{_summarise_body(raw.body_text)}"
        )

        if raw.status == HTTPStatus.NOT_FOUND:
            raise _RouteNotImplemented(
                f"Text backend at {self._base_url} has no {KoboldApiRoutes.GENERATE_STREAM} route",
            )

        if raw.status == HTTPStatus.SERVICE_UNAVAILABLE:
            raise TextBackendBusy(f"Text backend is already generating ({failure_detail})")

        if raw.status in _REJECTED_PAYLOAD_STATUSES:
            raise TextBackendRejectedPayload(f"Text backend refused the payload ({failure_detail})")

        raise _StreamUnusable(f"the stream route refused to start ({failure_detail})")

    async def _read_stream_records(self, response: aiohttp.ClientResponse) -> AsyncIterator[_KoboldStreamRecord]:
        """Yield each record of a server-sent-events stream as it arrives.

        Both backends send one `data:` field per record and nothing the driver needs from the other
        fields, so the framing is read by picking those lines out rather than by assembling whole
        events. A record's payload is JSON on a single line, because both encode it with an escaping
        JSON writer, so a line break can only ever be the framing's own.

        Raises:
            UnicodeDecodeError: A line was not text.
            ValidationError: A record's payload was not the shape both backends send.
        """
        async for raw_line in response.content:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith(_SSE_DATA_FIELD_PREFIX):
                continue
            yield _KoboldStreamRecord.model_validate_json(line[len(_SSE_DATA_FIELD_PREFIX) :].strip())

    async def _sample_generation_stats(
        self,
        *,
        generation_key: str,
        generation: _StreamedGeneration,
        on_progress: TextGenerationProgressCallback | None,
    ) -> None:
        """Poll the statistics route for this generation until the caller cancels the task.

        Runs only while the stream is open and only on a backend the probe found the route on. A sample
        the backend cannot answer stops the sampling for this generation rather than filling the log
        with one failure per half-second.
        """
        while True:
            await asyncio.sleep(STATS_POLL_INTERVAL_SECONDS)
            stats = await self._read_generation_stats(generation_key)
            if stats is None:
                return
            if not stats.found:
                continue
            generation.note_stats(stats)
            _report_progress(on_progress, generation)

    async def _read_generation_stats(self, generation_key: str) -> _KoboldGenerationStats | None:
        """Return one statistics sample for a generation, or None when the route did not answer one.

        A 404 also settles the capability the other way: a route the backend answered the probe for and
        now denies is a route it does not have, so nothing asks again.
        """
        try:
            raw = await self._request(
                method=hdrs.METH_POST,
                route=KoboldApiRoutes.GENERATION_STATS,
                timeout_seconds=KoboldApiTimeouts.STATS_SECONDS,
                json_body={KoboldApiJsonKeys.GENERATION_KEY: generation_key},
            )
        except (aiohttp.ClientError, TimeoutError) as request_error:
            logger.debug(
                f"Text backend at {self._base_url} did not answer statistics for generation "
                f"{generation_key}: {request_error}",
            )
            return None

        if raw.status == HTTPStatus.NOT_FOUND:
            self._generation_stats_present = False
            logger.debug(
                f"Text backend at {self._base_url} has no {KoboldApiRoutes.GENERATION_STATS} route; "
                f"token counts will be unknown for this backend",
            )
            return None

        if raw.status != HTTPStatus.OK:
            logger.debug(
                f"Text backend at {self._base_url} answered statistics for generation {generation_key} "
                f"with status {raw.status}: {_summarise_body(raw.body_text)}",
            )
            return None

        try:
            return _KoboldGenerationStats.model_validate_json(raw.body_text)
        except ValidationError as parse_error:
            logger.debug(
                f"Text backend at {self._base_url} answered statistics for generation {generation_key} "
                f"unreadably: {_summarise_body(raw.body_text)} ({parse_error})",
            )
            return None

    async def _finished_token_counts(self, generation: _StreamedGeneration) -> tuple[int | None, int | None]:
        """Return the prompt and generated token counts for a finished stream, unknown where they are.

        The statistics route's last sample is the honest answer wherever there is one. Without it the
        only counts a backend offers are the perf counters, which are process-global and describe
        whichever generation finished most recently, so they are read only when this driver had a single
        generation running and could not be reading someone else's.
        """
        if generation.completion_tokens is not None or generation.prompt_tokens is not None:
            return generation.prompt_tokens, generation.completion_tokens

        if self._generations_in_flight > 1:
            return None, None

        perf = await self._read_perf_counters()
        if perf is None:
            return None, None
        return _reported_count(perf.last_input_count), _reported_count(perf.last_token_count)

    async def _read_perf_counters(self) -> _KoboldPerfResponse | None:
        """Return the backend's last-generation counters, or None when it has no such route or is mute."""
        try:
            return await self._read_json(
                route=KoboldApiRoutes.PERF,
                response_model=_KoboldPerfResponse,
                timeout_seconds=KoboldApiTimeouts.PERF_SECONDS,
            )
        except TextBackendUnavailable as no_counters:
            logger.debug(f"{no_counters}; the finished generation's token counts stay unknown")
            return None

    def _raise_for_generate_status(self, raw: _RawResponse, *, generation_key: str) -> None:
        """Raise the protocol exception a generate status calls for, returning only on success.

        Raises:
            TextBackendBusy: The status was 503.
            TextBackendCredentialRefused: The status was one of
                [`_REFUSED_CREDENTIAL_STATUSES`][horde_worker_regen.text_backends.kobold_api].
            TextBackendRejectedPayload: The status was one of
                [`_REJECTED_PAYLOAD_STATUSES`][horde_worker_regen.text_backends.kobold_api].
            TextBackendUnavailable: Any other non-200 status.
        """
        if raw.status == HTTPStatus.OK:
            return

        self._raise_for_refused_credential(raw, detail=f"running generation {generation_key}")

        failure_detail = (
            f"backend at {self._base_url}, generation {generation_key}, status {raw.status}: "
            f"{_summarise_body(raw.body_text)}"
        )

        if raw.status == HTTPStatus.SERVICE_UNAVAILABLE:
            raise TextBackendBusy(f"Text backend is already generating ({failure_detail})")

        if raw.status in _REJECTED_PAYLOAD_STATUSES:
            raise TextBackendRejectedPayload(f"Text backend refused the payload ({failure_detail})")

        raise TextBackendUnavailable(f"Text backend failed the generation ({failure_detail})")

    # endregion

    async def stop(self, *, generation_key: str) -> None:
        """Ask the backend to abandon the generation with this key, best effort.

        The outcome is logged and never raised, a refused credential included. This runs on shutdown and
        on a generation the flow has already decided the fate of, where the backend may have no abort
        route, may refuse the worker's password, or may already be gone; in none of those cases is there
        anything left for the caller to do about it, and raising would displace the failure being handled.

        Args:
            generation_key: The key the abandoned generation was issued with.
        """
        try:
            raw = await self._request(
                method=hdrs.METH_POST,
                route=KoboldApiRoutes.ABORT,
                timeout_seconds=KoboldApiTimeouts.ABORT_SECONDS,
                json_body={KoboldApiJsonKeys.GENERATION_KEY: generation_key},
            )
        except (aiohttp.ClientError, TimeoutError) as request_error:
            logger.debug(
                f"Text backend at {self._base_url} did not answer an abort for generation "
                f"{generation_key}: {request_error}",
            )
            return

        if raw.status == HTTPStatus.NOT_FOUND:
            if not self._abort_route_missing_logged:
                self._abort_route_missing_logged = True
                logger.debug(
                    f"Text backend at {self._base_url} has no {KoboldApiRoutes.ABORT} route; "
                    f"generations cannot be abandoned on this backend",
                )
            return

        if raw.status != HTTPStatus.OK:
            logger.debug(
                f"Text backend at {self._base_url} answered an abort for generation {generation_key} "
                f"with status {raw.status}: {_summarise_body(raw.body_text)}",
            )
            return

        try:
            abort_response = _KoboldAbortResponse.model_validate_json(raw.body_text)
        except ValidationError as parse_error:
            logger.debug(
                f"Text backend at {self._base_url} answered an abort for generation {generation_key} "
                f"unreadably: {_summarise_body(raw.body_text)} ({parse_error})",
            )
            return

        logger.debug(
            f"Text backend at {self._base_url} abort for generation {generation_key}: "
            f"{KoboldApiJsonKeys.SUCCESS}={abort_response.success!r} "
            f"{KoboldApiJsonKeys.DONE}={abort_response.done!r}",
        )

    async def close(self) -> None:
        """Release the driver's own state. The injected session belongs to the caller and stays open."""
        if self._closed:
            return
        self._closed = True
        logger.debug(f"Text backend driver for {self._base_url} closed")


async def _cancel_sampler(sampler: asyncio.Task[None] | None) -> None:
    """Stop a statistics sampler and wait for it to notice, so no task outlives its generation."""
    if sampler is None:
        return
    sampler.cancel()
    # The sampler loops until it is cancelled, so its cancellation is the expected end of it; letting
    # that out would turn a finished generation into a cancelled one.
    with contextlib.suppress(asyncio.CancelledError):
        await sampler


__all__ = [
    "CAPABILITY_PROBE_GENERATION_KEY",
    "STATS_POLL_INTERVAL_SECONDS",
    "KoboldApiJsonKeys",
    "KoboldApiRoutes",
    "KoboldApiTextBackend",
    "KoboldApiTimeouts",
]
