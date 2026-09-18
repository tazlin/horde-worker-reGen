"""The KoboldAI driver is the worker's whole conversation with a program it does not control.

These tests drive the real driver against a local `aiohttp` application whose answers are shaped like
koboldcpp's: the same routes, the same JSON keys, the same server-sent-events framing, the same string
booleans on abort, and the same 503 for "one generation at a time and I am on it". Three things they pin
above all: the horde payload reaches the backend exactly as it arrived with only the generation key
added, a generation is driven through the stream and reports itself as it arrives, and every failure
mode surfaces as one of the four protocol exceptions so the flow never has to look at HTTP.
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Final

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from horde_worker_regen.text_backends import (
    CAPABILITY_PROBE_GENERATION_KEY,
    KoboldApiJsonKeys,
    KoboldApiRoutes,
    KoboldApiTextBackend,
    TextBackend,
    TextBackendBusy,
    TextBackendCredentialRefused,
    TextBackendRejectedPayload,
    TextBackendUnavailable,
    TextGenerationProgress,
    kobold_api,
)

_MODEL_NAME = "koboldcpp/a-model.gguf"
_PROTECTED_MODEL_NAME = "koboldcpp/protected-model"
"""What koboldcpp names on its model route to a reader whose password it does not accept.

The route is not guarded like the rest: it answers 200 to anyone, so a wrong password reads as a healthy
backend until something behind the guard is asked.
"""

_MAX_CONTEXT_LENGTH = 4096
_MAX_LENGTH = 512
_STREAM_CHUNKS: Final = ("the backend's ", "own ", "words")
"""One generation as the stream delivers it: several tokens per record, which is what a real one sends."""
_GENERATED_TEXT = "".join(_STREAM_CHUNKS)
_STREAM_FINISH_REASON = "length"
_GENERATION_KEY = "FEDCBA98"

_BUSY_BODY = {"detail": {"msg": "Server is busy; please try again later.", "type": "service_unavailable"}}
"""koboldcpp's own 503 body, so the driver is exercised against the shape it will really see."""

_UNAUTHORIZED_BODY = {
    "detail": {"error": "Unauthorized", "msg": "Authentication key is missing or invalid.", "type": "unauthorized"},
}
"""koboldcpp's own 401 body, sent by every route it guards with the password it was launched with."""

_SLOW_ANSWER_SECONDS = 0.5
"""Long enough to outlast a test's deadline, short enough that the server shuts down without waiting."""

_SHORT_DEADLINE_SECONDS = 0.05

_RECORD_DELAY_SECONDS = 0.05
"""Spacing between stream records, so a test can act while a generation is still arriving."""

_FAST_STATS_POLL_SECONDS = 0.01
"""The statistics poll interval under test: several samples inside one short streamed generation."""

_STATS_SAMPLES: Final = (
    {
        KoboldApiJsonKeys.FOUND: True,
        "batched": True,
        "state": 2,
        "slot": 1,
        KoboldApiJsonKeys.PROMPT_TOKENS: 12,
        KoboldApiJsonKeys.COMPLETION_TOKENS: 20,
        KoboldApiJsonKeys.GENERATION_SECONDS: 1.0,
        "finished": False,
    },
    {
        KoboldApiJsonKeys.FOUND: True,
        "batched": True,
        "state": 2,
        "slot": 1,
        KoboldApiJsonKeys.PROMPT_TOKENS: 12,
        KoboldApiJsonKeys.COMPLETION_TOKENS: 40,
        KoboldApiJsonKeys.GENERATION_SECONDS: 2.0,
        "finished": False,
    },
)
"""A patched backend's answers about one generation in flight, the second repeating once the script ends."""


def _default_stream_records() -> tuple[dict[str, object], ...]:
    """Return koboldcpp's own stream shape: a null finish reason per record and a real one on the last."""
    records: list[dict[str, object]] = [
        {KoboldApiJsonKeys.TOKEN: chunk, KoboldApiJsonKeys.FINISH_REASON: None} for chunk in _STREAM_CHUNKS[:-1]
    ]
    records.append(
        {KoboldApiJsonKeys.TOKEN: _STREAM_CHUNKS[-1], KoboldApiJsonKeys.FINISH_REASON: _STREAM_FINISH_REASON}
    )
    return tuple(records)


def _sonar_stream_records() -> tuple[dict[str, object], ...]:
    """Return sonar's stream shape, which carries a token and never a finish reason at all."""
    return tuple({KoboldApiJsonKeys.TOKEN: chunk} for chunk in _STREAM_CHUNKS)


@dataclass
class _KoboldBackendBehaviour:
    """How the fake KoboldAI server should answer, and what it recorded while doing so.

    Defaults describe a healthy stock koboldcpp: a model loaded, no soft prompts, a working stream and
    abort route, and no statistics route, which is what an unpatched build has.  Each test changes only
    the one thing it is about.
    """

    model_name: str = _MODEL_NAME
    model_status: int = HTTPStatus.OK
    model_delay_seconds: float = 0.0
    required_password: str | None = None
    """The password the backend was launched with, or `None` for one launched without.

    Guards the routes koboldcpp guards (the generation routes, the statistics route and abort) with the
    401 it answers; the model route stays open and names a placeholder instead, as the real one does.
    """
    max_context_length: int = _MAX_CONTEXT_LENGTH
    max_length: int = _MAX_LENGTH
    soft_prompts: tuple[str, ...] | None = ()
    """The list the soft-prompts route answers with; `None` makes that route answer 404."""
    generate_status: int = HTTPStatus.OK
    """The status both generation routes answer with, since a backend refuses a request either way."""
    generate_body: object | None = None
    """An explicit blocking generate body, or `None` for koboldcpp's faithful `results` answer."""
    generate_delay_seconds: float = 0.0
    stream_route_present: bool = True
    """Whether the stream route exists. `False` answers 404 there, as a backend without one does."""
    stream_records: tuple[Mapping[str, object], ...] | None = None
    """The records the stream sends, or `None` for koboldcpp's own shape over `_STREAM_CHUNKS`."""
    stream_record_delay_seconds: float = 0.0
    stream_breaks_after_records: int | None = None
    """Cut the connection after this many records, standing in for a backend that died mid-generation."""
    stats_route_present: bool = False
    stats_samples: tuple[Mapping[str, object], ...] = _STATS_SAMPLES
    perf_last_token_count: int = 0
    perf_last_input_count: int = 0
    abort_route_present: bool = True
    abort_requested: asyncio.Event = field(default_factory=asyncio.Event)
    recorded_generate_bodies: list[object] = field(default_factory=list)
    """Every body either generation route received, so passthrough is asserted whichever one was used."""
    recorded_generate_authorizations: list[str | None] = field(default_factory=list)
    recorded_abort_bodies: list[object] = field(default_factory=list)
    recorded_stats_bodies: list[object] = field(default_factory=list)
    blocking_generate_count: int = 0
    stream_generate_count: int = 0
    perf_request_count: int = 0
    stats_sample_index: int = 0


def _build_kobold_app(behaviour: _KoboldBackendBehaviour) -> web.Application:
    """Return an application answering the KoboldAI routes the way `behaviour` says to."""

    def credential_refused(request: web.Request) -> bool:
        """Return whether this request carries something other than the password the backend wants."""
        if behaviour.required_password is None:
            return False
        return request.headers.get(aiohttp.hdrs.AUTHORIZATION) != f"Bearer {behaviour.required_password}"

    def unauthorized() -> web.Response:
        """Return the refusal koboldcpp answers a request whose credential it will not accept."""
        return web.json_response(_UNAUTHORIZED_BODY, status=HTTPStatus.UNAUTHORIZED)

    async def handle_model(request: web.Request) -> web.Response:
        """Answer the model route, which is what readiness and the advertised name both read."""
        if behaviour.model_delay_seconds > 0:
            await asyncio.sleep(behaviour.model_delay_seconds)
        if behaviour.model_status != HTTPStatus.OK:
            return web.json_response({"detail": "no model"}, status=behaviour.model_status)
        if credential_refused(request):
            return web.json_response({KoboldApiJsonKeys.RESULT: _PROTECTED_MODEL_NAME})
        return web.json_response({KoboldApiJsonKeys.RESULT: behaviour.model_name})

    async def handle_max_context_length(request: web.Request) -> web.Response:
        """Answer the horde-facing context cap."""
        return web.json_response({KoboldApiJsonKeys.VALUE: behaviour.max_context_length})

    async def handle_max_length(request: web.Request) -> web.Response:
        """Answer the horde-facing generation-length cap."""
        return web.json_response({KoboldApiJsonKeys.VALUE: behaviour.max_length})

    async def handle_soft_prompts(request: web.Request) -> web.Response:
        """Answer the optional soft-prompts route, or 404 as a backend without one does."""
        if behaviour.soft_prompts is None:
            return web.json_response({"detail": "not found"}, status=HTTPStatus.NOT_FOUND)
        return web.json_response({KoboldApiJsonKeys.VALUES: list(behaviour.soft_prompts)})

    async def record_generation_request(request: web.Request) -> None:
        """Record what either generation route was sent, which is what a passthrough assertion reads."""
        behaviour.recorded_generate_bodies.append(await request.json())
        behaviour.recorded_generate_authorizations.append(request.headers.get(aiohttp.hdrs.AUTHORIZATION))

    def refusal_for(status: int) -> web.Response:
        """Return the answer a backend gives for a request it will not run, with koboldcpp's busy body."""
        if status == HTTPStatus.SERVICE_UNAVAILABLE:
            return web.json_response(_BUSY_BODY, status=HTTPStatus.SERVICE_UNAVAILABLE)
        return web.json_response({"detail": "nope"}, status=status)

    async def handle_generate(request: web.Request) -> web.Response:
        """Answer the blocking route, which says nothing at all until the generation is finished."""
        behaviour.blocking_generate_count += 1
        await record_generation_request(request)
        if credential_refused(request):
            return unauthorized()
        if behaviour.generate_delay_seconds > 0:
            await asyncio.sleep(behaviour.generate_delay_seconds)
        if behaviour.generate_status != HTTPStatus.OK:
            return refusal_for(behaviour.generate_status)
        if behaviour.generate_body is not None:
            return web.json_response(behaviour.generate_body)
        return web.json_response(
            {
                KoboldApiJsonKeys.RESULTS: [
                    {
                        KoboldApiJsonKeys.TEXT: _GENERATED_TEXT,
                        KoboldApiJsonKeys.FINISH_REASON: _STREAM_FINISH_REASON,
                        KoboldApiJsonKeys.PROMPT_TOKENS: 12,
                        KoboldApiJsonKeys.COMPLETION_TOKENS: 34,
                    },
                ],
            },
        )

    async def handle_generate_stream(request: web.Request) -> web.StreamResponse:
        """Stream the generation back the way koboldcpp does, one `event: message` record per chunk."""
        behaviour.stream_generate_count += 1
        await record_generation_request(request)
        if credential_refused(request):
            return unauthorized()
        if not behaviour.stream_route_present:
            return web.json_response({"detail": "not found"}, status=HTTPStatus.NOT_FOUND)
        if behaviour.generate_delay_seconds > 0:
            await asyncio.sleep(behaviour.generate_delay_seconds)
        if behaviour.generate_status != HTTPStatus.OK:
            return refusal_for(behaviour.generate_status)

        records = behaviour.stream_records if behaviour.stream_records is not None else _default_stream_records()
        response = web.StreamResponse()
        response.content_type = "text/event-stream"
        await response.prepare(request)
        for records_sent, record in enumerate(records, start=1):
            if behaviour.stream_record_delay_seconds > 0:
                await asyncio.sleep(behaviour.stream_record_delay_seconds)
            if behaviour.abort_requested.is_set():
                break
            await response.write(f"event: message\ndata: {json.dumps(dict(record))}\n\n".encode())
            if records_sent == behaviour.stream_breaks_after_records:
                # A backend that died part way through: the connection goes without the body ever being
                # completed, which is what the driver has to tell apart from a generation that ended.
                transport = request.transport
                assert transport is not None, "a prepared response always has a transport"
                transport.abort()
                return response
        await response.write_eof()
        return response

    async def handle_generation_stats(request: web.Request) -> web.Response:
        """Answer per-request statistics as a patched build does, or 404 as a stock build does."""
        body = await request.json()
        behaviour.recorded_stats_bodies.append(body)
        if credential_refused(request):
            return unauthorized()
        if not behaviour.stats_route_present:
            return web.json_response({"detail": "not found"}, status=HTTPStatus.NOT_FOUND)
        if body.get(KoboldApiJsonKeys.GENERATION_KEY) == CAPABILITY_PROBE_GENERATION_KEY:
            return web.json_response({KoboldApiJsonKeys.FOUND: False})
        sample_index = min(behaviour.stats_sample_index, len(behaviour.stats_samples) - 1)
        behaviour.stats_sample_index += 1
        if sample_index < 0:
            return web.json_response({KoboldApiJsonKeys.FOUND: False})
        return web.json_response(dict(behaviour.stats_samples[sample_index]))

    async def handle_perf(request: web.Request) -> web.Response:
        """Answer the counters describing whichever generation finished most recently."""
        behaviour.perf_request_count += 1
        return web.json_response(
            {
                KoboldApiJsonKeys.LAST_TOKEN_COUNT: behaviour.perf_last_token_count,
                KoboldApiJsonKeys.LAST_INPUT_COUNT: behaviour.perf_last_input_count,
                "idle": 1,
                "queue": 0,
            },
        )

    async def handle_abort(request: web.Request) -> web.Response:
        """Record the abort, end any stream in flight, and answer koboldcpp's two string flags."""
        behaviour.recorded_abort_bodies.append(await request.json())
        if credential_refused(request):
            return unauthorized()
        behaviour.abort_requested.set()
        return web.json_response({KoboldApiJsonKeys.SUCCESS: "true", KoboldApiJsonKeys.DONE: "true"})

    app = web.Application()
    app.router.add_get(KoboldApiRoutes.MODEL, handle_model)
    app.router.add_get(KoboldApiRoutes.MAX_CONTEXT_LENGTH, handle_max_context_length)
    app.router.add_get(KoboldApiRoutes.MAX_LENGTH, handle_max_length)
    app.router.add_get(KoboldApiRoutes.SOFT_PROMPTS_LIST, handle_soft_prompts)
    app.router.add_post(KoboldApiRoutes.GENERATE, handle_generate)
    app.router.add_post(KoboldApiRoutes.GENERATE_STREAM, handle_generate_stream)
    app.router.add_post(KoboldApiRoutes.GENERATION_STATS, handle_generation_stats)
    app.router.add_get(KoboldApiRoutes.PERF, handle_perf)
    if behaviour.abort_route_present:
        app.router.add_post(KoboldApiRoutes.ABORT, handle_abort)
    return app


@asynccontextmanager
async def _running_backend(
    behaviour: _KoboldBackendBehaviour,
    *,
    password: str | None = None,
) -> AsyncIterator[KoboldApiTextBackend]:
    """Serve `behaviour` on an ephemeral port and yield a driver pointed at it."""
    server = TestServer(_build_kobold_app(behaviour))
    await server.start_server()
    session = aiohttp.ClientSession()
    try:
        yield KoboldApiTextBackend(f"http://{server.host}:{server.port}", session, password=password)
    finally:
        await session.close()
        await server.close()


@asynccontextmanager
async def _backend_on_a_closed_port() -> AsyncIterator[KoboldApiTextBackend]:
    """Yield a driver pointed at a port that had a server and now has nothing, so connecting is refused."""
    server = TestServer(web.Application())
    await server.start_server()
    closed_url = f"http://{server.host}:{server.port}"
    await server.close()
    session = aiohttp.ClientSession()
    try:
        yield KoboldApiTextBackend(closed_url, session)
    finally:
        await session.close()


class _ProgressRecorder:
    """Collects every progress report a generation made, which is what the flow would be reading."""

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.reports: list[TextGenerationProgress] = []
        self.first_chunk = asyncio.Event()

    def __call__(self, progress: TextGenerationProgress) -> None:
        """Record one report and wake anything waiting for the generation to have started arriving."""
        self.reports.append(progress)
        if progress.chunks_received > 0:
            self.first_chunk.set()

    @property
    def chunk_reports(self) -> list[TextGenerationProgress]:
        """The reports that carried a chunk, in order, ignoring any that only carried statistics."""
        seen_chunks = 0
        carried_a_chunk: list[TextGenerationProgress] = []
        for report in self.reports:
            if report.chunks_received > seen_chunks:
                seen_chunks = report.chunks_received
                carried_a_chunk.append(report)
        return carried_a_chunk


async def test_driver_satisfies_the_protocol() -> None:
    """The driver is accepted as a `TextBackend` without inheriting from it."""
    async with _running_backend(_KoboldBackendBehaviour()) as backend:
        assert isinstance(backend, TextBackend)


async def test_base_url_drops_a_trailing_slash() -> None:
    """An operator-written URL with a trailing slash must not produce a double slash in every route."""
    session = aiohttp.ClientSession()
    try:
        backend = KoboldApiTextBackend("http://127.0.0.1:5001/", session)

        assert backend.base_url == "http://127.0.0.1:5001"
    finally:
        await session.close()


async def test_ready_is_true_when_the_model_route_answers() -> None:
    """A named model on the model route is the whole of what readiness means."""
    async with _running_backend(_KoboldBackendBehaviour()) as backend:
        assert await backend.ready(deadline_seconds=5.0) is True


async def test_ready_is_false_when_the_connection_is_refused() -> None:
    """Polling a backend that has not opened its port yet reads `False`, never an exception."""
    async with _backend_on_a_closed_port() as backend:
        assert await backend.ready(deadline_seconds=5.0) is False


async def test_ready_is_false_when_the_answer_arrives_after_the_deadline() -> None:
    """A backend too slow to answer within the deadline is not ready, whatever it eventually says."""
    behaviour = _KoboldBackendBehaviour(model_delay_seconds=_SLOW_ANSWER_SECONDS)

    async with _running_backend(behaviour) as backend:
        assert await backend.ready(deadline_seconds=_SHORT_DEADLINE_SECONDS) is False


async def test_ready_is_false_when_the_model_route_errors() -> None:
    """A model route answering a server error says nothing about a loaded model."""
    behaviour = _KoboldBackendBehaviour(model_status=HTTPStatus.INTERNAL_SERVER_ERROR)

    async with _running_backend(behaviour) as backend:
        assert await backend.ready(deadline_seconds=5.0) is False


async def test_describe_reports_the_backend_name_and_caps_verbatim() -> None:
    """The advertised facts come from the backend's own answers, prefix included."""
    behaviour = _KoboldBackendBehaviour(soft_prompts=("a_soft_prompt", "another_soft_prompt"))

    async with _running_backend(behaviour) as backend:
        description = await backend.describe()

    assert description.model_name == _MODEL_NAME
    assert description.max_context_length == _MAX_CONTEXT_LENGTH
    assert description.max_length == _MAX_LENGTH
    assert description.soft_prompts == ("a_soft_prompt", "another_soft_prompt")


async def test_describe_reports_no_soft_prompts_when_the_route_is_absent() -> None:
    """Soft prompts are optional, so a backend without the route still describes."""
    behaviour = _KoboldBackendBehaviour(soft_prompts=None)

    async with _running_backend(behaviour) as backend:
        description = await backend.describe()

    assert description.soft_prompts == ()
    assert description.model_name == _MODEL_NAME


async def test_describe_reports_no_soft_prompts_when_the_list_is_empty() -> None:
    """Both backends answer an empty list today; that is "none", not a failure."""
    async with _running_backend(_KoboldBackendBehaviour()) as backend:
        description = await backend.describe()

    assert description.soft_prompts == ()


async def test_describe_raises_unavailable_when_a_required_route_errors() -> None:
    """A fact the backend will not report cannot be advertised, so describing fails loudly."""
    behaviour = _KoboldBackendBehaviour(model_status=HTTPStatus.INTERNAL_SERVER_ERROR)

    async with _running_backend(behaviour) as backend:
        with pytest.raises(TextBackendUnavailable):
            await backend.describe()


async def test_describe_raises_unavailable_when_the_backend_is_not_listening() -> None:
    """Describing a backend that has gone away is a backend-down failure, not a silent default."""
    async with _backend_on_a_closed_port() as backend:
        with pytest.raises(TextBackendUnavailable):
            await backend.describe()


async def test_capabilities_finds_the_statistics_route_on_a_patched_backend() -> None:
    """A build with the statistics route is what lets a text job report token counts at all."""
    behaviour = _KoboldBackendBehaviour(stats_route_present=True)

    async with _running_backend(behaviour) as backend:
        capabilities = await backend.capabilities()

    assert capabilities.generation_stats is True
    assert behaviour.recorded_stats_bodies == [
        {KoboldApiJsonKeys.GENERATION_KEY: CAPABILITY_PROBE_GENERATION_KEY},
    ], "the probe asks about a key no generation will ever use"


async def test_capabilities_is_probed_once_and_the_answer_reused() -> None:
    """The route is part of the program, so a build without one will not grow one while it runs."""
    behaviour = _KoboldBackendBehaviour(stats_route_present=False)

    async with _running_backend(behaviour) as backend:
        first = await backend.capabilities()
        second = await backend.capabilities()

    assert first.generation_stats is False
    assert second.generation_stats is False
    assert len(behaviour.recorded_stats_bodies) == 1


async def test_a_streamed_generation_assembles_the_text_and_reports_every_chunk() -> None:
    """The stream is how the worker learns anything at all before a generation has finished."""
    behaviour = _KoboldBackendBehaviour()
    progress = _ProgressRecorder()

    async with _running_backend(behaviour) as backend:
        result = await backend.generate(
            {"prompt": "a prompt"},
            generation_key=_GENERATION_KEY,
            deadline_seconds=10.0,
            on_progress=progress,
        )

    assert result.text == _GENERATED_TEXT
    assert behaviour.stream_generate_count == 1
    assert behaviour.blocking_generate_count == 0
    assert [report.chunks_received for report in progress.chunk_reports] == [1, 2, 3]
    assert [report.characters_received for report in progress.chunk_reports] == [
        len(_STREAM_CHUNKS[0]),
        len(_STREAM_CHUNKS[0]) + len(_STREAM_CHUNKS[1]),
        len(_GENERATED_TEXT),
    ]
    assert progress.chunk_reports[-1].last_chunk_at is not None


async def test_the_final_stream_record_carries_the_finish_reason() -> None:
    """Why a generation stopped is the backend's own word for it, and only the last record has it."""
    async with _running_backend(_KoboldBackendBehaviour()) as backend:
        result = await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=10.0)

    assert result.finish_reason == _STREAM_FINISH_REASON


async def test_a_stream_without_any_finish_reason_still_produces_the_generation() -> None:
    """A sonar record carries a token and nothing else, so an absent finish reason is not a failure."""
    behaviour = _KoboldBackendBehaviour(stream_records=_sonar_stream_records())
    progress = _ProgressRecorder()

    async with _running_backend(behaviour) as backend:
        result = await backend.generate(
            {},
            generation_key=_GENERATION_KEY,
            deadline_seconds=10.0,
            on_progress=progress,
        )

    assert result.text == _GENERATED_TEXT
    assert result.finish_reason is None
    assert len(progress.chunk_reports) == len(_STREAM_CHUNKS)


async def test_statistics_samples_reach_the_callback_with_token_counts_and_a_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chunk is an arbitrary number of tokens, so counts and a rate can only come from the backend."""
    monkeypatch.setattr(kobold_api, "STATS_POLL_INTERVAL_SECONDS", _FAST_STATS_POLL_SECONDS)
    behaviour = _KoboldBackendBehaviour(
        stats_route_present=True,
        stream_record_delay_seconds=_RECORD_DELAY_SECONDS,
    )
    progress = _ProgressRecorder()

    async with _running_backend(behaviour) as backend:
        result = await backend.generate(
            {},
            generation_key=_GENERATION_KEY,
            deadline_seconds=10.0,
            on_progress=progress,
        )

    counted_reports = [report for report in progress.reports if report.completion_tokens is not None]
    assert counted_reports, "a backend with the statistics route must report token counts while it runs"
    assert counted_reports[-1].completion_tokens == 40
    assert counted_reports[-1].prompt_tokens == 12
    assert counted_reports[-1].tokens_per_second == pytest.approx(20.0)
    assert result.prompt_tokens == 12
    assert result.generated_tokens == 40
    assert behaviour.perf_request_count == 0, "a backend that counted the tokens is not asked again"


def test_a_rate_is_measured_between_counted_samples_when_the_backend_reports_no_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backend that stores its timings only at finish answers zero seconds live; the count growth still rates it."""
    clock = {"now": 100.0}
    monkeypatch.setattr(kobold_api.time, "monotonic", lambda: clock["now"])
    generation = kobold_api._StreamedGeneration(started_at=clock["now"])

    def sample(completion_tokens: int) -> kobold_api._KoboldGenerationStats:
        return kobold_api._KoboldGenerationStats.model_validate(
            {
                KoboldApiJsonKeys.FOUND: True,
                KoboldApiJsonKeys.PROMPT_TOKENS: 0,
                KoboldApiJsonKeys.COMPLETION_TOKENS: completion_tokens,
                KoboldApiJsonKeys.GENERATION_SECONDS: 0.0,
            },
        )

    generation.note_stats(sample(10))
    assert generation.snapshot().tokens_per_second is None, "one sample is a count, not a rate"

    clock["now"] = 100.5
    generation.note_stats(sample(60))
    assert generation.snapshot().tokens_per_second == pytest.approx(100.0)

    clock["now"] = 101.0
    generation.note_stats(sample(60))
    assert generation.snapshot().tokens_per_second == pytest.approx(100.0), "no growth keeps the last rate"

    clock["now"] = 102.0
    generation.note_stats(sample(80))
    assert generation.snapshot().tokens_per_second == pytest.approx(20.0)


async def test_a_backend_without_statistics_is_asked_once_and_reports_no_counts() -> None:
    """A stock build has no statistics route, and a rate guessed from characters would be a lie."""
    behaviour = _KoboldBackendBehaviour(stats_route_present=False)
    progress = _ProgressRecorder()

    async with _running_backend(behaviour) as backend:
        for _generation in range(2):
            await backend.generate(
                {},
                generation_key=_GENERATION_KEY,
                deadline_seconds=10.0,
                on_progress=progress,
            )

    assert len(behaviour.recorded_stats_bodies) == 1, "the route was probed once and then left alone"
    assert all(report.completion_tokens is None for report in progress.reports)
    assert all(report.prompt_tokens is None for report in progress.reports)
    assert all(report.tokens_per_second is None for report in progress.reports)


async def test_a_streamed_generation_without_statistics_takes_its_counts_from_the_perf_route() -> None:
    """The perf counters describe the generation that just finished, which is this one when it ran alone."""
    behaviour = _KoboldBackendBehaviour(
        stats_route_present=False,
        perf_last_token_count=160,
        perf_last_input_count=20,
    )

    async with _running_backend(behaviour) as backend:
        result = await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=10.0)

    assert result.generated_tokens == 160
    assert result.prompt_tokens == 20
    assert behaviour.perf_request_count == 1


async def test_a_missing_stream_route_falls_back_to_the_blocking_call() -> None:
    """A backend with no stream route still generates; the worker just learns nothing until it answers."""
    behaviour = _KoboldBackendBehaviour(stream_route_present=False)

    async with _running_backend(behaviour) as backend:
        first = await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=10.0)
        second = await backend.generate({}, generation_key="a-second-generation", deadline_seconds=10.0)

    assert first.text == _GENERATED_TEXT
    assert second.text == _GENERATED_TEXT
    assert first.finish_reason == _STREAM_FINISH_REASON
    assert first.prompt_tokens == 12
    assert first.generated_tokens == 34
    assert behaviour.blocking_generate_count == 2
    assert behaviour.stream_generate_count == 1, "a route that is missing is not asked for again"


async def test_a_stream_that_breaks_mid_way_fails_the_generation() -> None:
    """Half a generation reads as a whole one to a requester, so a broken stream is a failure, not an answer."""
    behaviour = _KoboldBackendBehaviour(
        stream_breaks_after_records=1,
        stream_record_delay_seconds=_RECORD_DELAY_SECONDS,
    )

    async with _running_backend(behaviour) as backend:
        with pytest.raises(TextBackendUnavailable):
            await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=10.0)

    assert behaviour.blocking_generate_count == 0, "a generation that had begun must not be silently re-run"


async def test_aborting_a_generation_mid_stream_returns_the_text_that_arrived() -> None:
    """An abandoned generation on koboldcpp answers with what it had, which the flow then discards."""
    behaviour = _KoboldBackendBehaviour(stream_record_delay_seconds=_RECORD_DELAY_SECONDS)
    progress = _ProgressRecorder()

    async with _running_backend(behaviour) as backend:
        generation = asyncio.create_task(
            backend.generate(
                {},
                generation_key=_GENERATION_KEY,
                deadline_seconds=10.0,
                on_progress=progress,
            ),
        )
        await asyncio.wait_for(progress.first_chunk.wait(), timeout=5.0)
        await backend.stop(generation_key=_GENERATION_KEY)
        result = await asyncio.wait_for(generation, timeout=5.0)

    assert behaviour.recorded_abort_bodies == [{KoboldApiJsonKeys.GENERATION_KEY: _GENERATION_KEY}]
    assert result.text != _GENERATED_TEXT, "an abandoned generation cannot have produced all of its text"
    assert _GENERATED_TEXT.startswith(result.text)


async def test_generate_forwards_the_payload_untouched_and_adds_only_the_generation_key() -> None:
    """The horde owns the payload's shape, so every key, nested or unfamiliar, arrives as it was sent."""
    payload: dict[str, object] = {
        "prompt": "a prompt the horde wrote",
        "max_length": 80,
        "temperature": 0.75,
        "stop_sequence": ["###", "\n\n"],
        "sampler_order": [6, 0, 1, 3, 4, 2, 5],
        "dynatemp_range": 0.0,
        "use_default_badwordsids": False,
        "logit_bias": {"1337": 4.5, "42": -2.25},
        "a_key_this_worker_has_never_heard_of": {"nested": {"deeper": [1, 2, {"deepest": True}]}},
    }
    payload_as_sent = copy.deepcopy(payload)
    behaviour = _KoboldBackendBehaviour()

    async with _running_backend(behaviour) as backend:
        result = await backend.generate(payload, generation_key=_GENERATION_KEY, deadline_seconds=10.0)

    assert result.text == _GENERATED_TEXT
    assert behaviour.recorded_generate_bodies == [
        {**payload_as_sent, KoboldApiJsonKeys.GENERATION_KEY: _GENERATION_KEY},
    ]
    assert payload == payload_as_sent, "the caller's own payload must not be mutated"


async def test_generate_maps_busy_to_the_busy_exception() -> None:
    """503 is the KoboldAI way of saying the generation slots are taken, so the job can be re-offered."""
    behaviour = _KoboldBackendBehaviour(generate_status=HTTPStatus.SERVICE_UNAVAILABLE)

    async with _running_backend(behaviour) as backend:
        with pytest.raises(TextBackendBusy):
            await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=10.0)

    assert behaviour.blocking_generate_count == 0, "a busy backend is busy on either route"


@pytest.mark.parametrize(
    "rejecting_status",
    [
        pytest.param(HTTPStatus.BAD_REQUEST, id="bad-request"),
        pytest.param(HTTPStatus.UNPROCESSABLE_ENTITY, id="unprocessable-entity"),
    ],
)
async def test_generate_maps_a_refused_payload_to_the_rejected_exception(rejecting_status: HTTPStatus) -> None:
    """A payload the backend refuses will be refused again, so the flow must be told not to retry."""
    behaviour = _KoboldBackendBehaviour(generate_status=rejecting_status)

    async with _running_backend(behaviour) as backend:
        with pytest.raises(TextBackendRejectedPayload):
            await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=10.0)

    assert behaviour.blocking_generate_count == 0, "a refused payload is refused on either route"


@pytest.mark.parametrize(
    "failing_status",
    [
        pytest.param(HTTPStatus.INTERNAL_SERVER_ERROR, id="internal-server-error"),
        pytest.param(HTTPStatus.BAD_GATEWAY, id="bad-gateway"),
        pytest.param(HTTPStatus.NOT_FOUND, id="no-generate-route"),
    ],
)
async def test_generate_maps_every_other_failing_status_to_unavailable(failing_status: HTTPStatus) -> None:
    """Anything that is neither busy nor a refused payload reads as a backend that is not working."""
    behaviour = _KoboldBackendBehaviour(generate_status=failing_status)

    async with _running_backend(behaviour) as backend:
        with pytest.raises(TextBackendUnavailable):
            await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=10.0)

    assert behaviour.blocking_generate_count == 1, "a stream that would not start is tried blocking once"


@pytest.mark.parametrize(
    "malformed_body",
    [
        pytest.param({"detail": "something else entirely"}, id="no-results-key"),
        pytest.param({KoboldApiJsonKeys.RESULTS: []}, id="empty-results"),
        pytest.param({KoboldApiJsonKeys.RESULTS: [{"finish_reason": "length"}]}, id="result-without-text"),
    ],
)
async def test_generate_treats_a_malformed_success_body_as_unavailable(malformed_body: object) -> None:
    """A 200 with no usable generation is a backend answering nonsense, and the message says what it sent."""
    behaviour = _KoboldBackendBehaviour(generate_body=malformed_body, stream_route_present=False)

    async with _running_backend(behaviour) as backend:
        with pytest.raises(TextBackendUnavailable) as raised:
            await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=10.0)

    assert KoboldApiJsonKeys.RESULTS in str(raised.value)


async def test_a_stream_that_ends_without_a_record_falls_back_to_the_blocking_call() -> None:
    """A route that opens and produces nothing has not generated anything, so the request is run again."""
    behaviour = _KoboldBackendBehaviour(stream_records=())

    async with _running_backend(behaviour) as backend:
        result = await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=10.0)

    assert result.text == _GENERATED_TEXT
    assert behaviour.blocking_generate_count == 1


async def test_generate_past_the_deadline_is_unavailable() -> None:
    """A generation that outlives its deadline is abandoned by the driver rather than waited on."""
    behaviour = _KoboldBackendBehaviour(generate_delay_seconds=_SLOW_ANSWER_SECONDS)

    async with _running_backend(behaviour) as backend:
        with pytest.raises(TextBackendUnavailable):
            await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=_SHORT_DEADLINE_SECONDS)

    assert behaviour.blocking_generate_count == 0, "a deadline is not a reason to run the generation twice"


async def test_generate_raises_unavailable_when_the_backend_is_not_listening() -> None:
    """A backend that died mid-run fails the generation as unavailable, not as a bad payload."""
    async with _backend_on_a_closed_port() as backend:
        with pytest.raises(TextBackendUnavailable):
            await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=5.0)


async def test_generate_sends_the_password_as_a_bearer_token() -> None:
    """Koboldcpp launched with a password wants that password as a bearer token."""
    behaviour = _KoboldBackendBehaviour()

    async with _running_backend(behaviour, password="a-launch-password") as backend:
        await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=10.0)

    assert behaviour.recorded_generate_authorizations == ["Bearer a-launch-password"]


async def test_generate_sends_no_authorization_header_without_a_password() -> None:
    """An unprotected backend must not be sent an empty credential."""
    behaviour = _KoboldBackendBehaviour()

    async with _running_backend(behaviour) as backend:
        await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=10.0)

    assert behaviour.recorded_generate_authorizations == [None]


async def test_a_protected_backend_serves_a_driver_carrying_its_password() -> None:
    """The password reaches every guarded route, so a protected backend behaves like any other."""
    behaviour = _KoboldBackendBehaviour(required_password="a-launch-password", stats_route_present=True)

    async with _running_backend(behaviour, password="a-launch-password") as backend:
        assert await backend.ready(deadline_seconds=5.0) is True
        description = await backend.describe()
        capabilities = await backend.capabilities()
        result = await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=10.0)

    assert description.model_name == _MODEL_NAME
    assert capabilities.generation_stats is True
    assert result.text == _GENERATED_TEXT


async def test_generate_raises_a_credential_refusal_when_the_password_is_wrong() -> None:
    """The refusal is its own failure: polling a backend that will not accept the worker never ends."""
    behaviour = _KoboldBackendBehaviour(required_password="the-real-password")

    async with _running_backend(behaviour, password="the-wrong-password") as backend:
        with pytest.raises(TextBackendCredentialRefused):
            await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=10.0)

    assert behaviour.blocking_generate_count == 0, "a refused credential is not a reason to try the other route"


async def test_capabilities_raises_a_credential_refusal_rather_than_reporting_a_missing_route() -> None:
    """A route behind a credential the worker does not have says nothing about whether the build has it."""
    behaviour = _KoboldBackendBehaviour(required_password="the-real-password", stats_route_present=True)

    async with _running_backend(behaviour) as backend:
        with pytest.raises(TextBackendCredentialRefused):
            await backend.capabilities()


async def test_ready_raises_a_credential_refusal_when_the_model_route_itself_is_guarded() -> None:
    """A backend (or a proxy in front of one) that guards every route is refusing, not starting up."""
    behaviour = _KoboldBackendBehaviour(model_status=HTTPStatus.UNAUTHORIZED)

    async with _running_backend(behaviour) as backend:
        with pytest.raises(TextBackendCredentialRefused):
            await backend.ready(deadline_seconds=5.0)


async def test_a_protected_backend_names_a_placeholder_model_to_a_worker_it_will_not_serve() -> None:
    """Koboldcpp answers its model route to anyone, which is why the refusal shows at the routes behind it.

    A worker whose password is wrong sees a ready backend describing a model it has never heard of, so
    the readiness probe alone cannot be where a credential is judged.
    """
    behaviour = _KoboldBackendBehaviour(required_password="the-real-password")

    async with _running_backend(behaviour, password="the-wrong-password") as backend:
        assert await backend.ready(deadline_seconds=5.0) is True
        description = await backend.describe()

    assert description.model_name == _PROTECTED_MODEL_NAME


async def test_stop_names_the_generation_on_the_abort_route() -> None:
    """The abort names one generation by its key, so another request in flight is not cancelled with it."""
    behaviour = _KoboldBackendBehaviour()

    async with _running_backend(behaviour) as backend:
        await backend.stop(generation_key=_GENERATION_KEY)

    assert behaviour.recorded_abort_bodies == [{KoboldApiJsonKeys.GENERATION_KEY: _GENERATION_KEY}]


async def test_stop_returns_quietly_when_the_backend_has_no_abort_route() -> None:
    """A backend with no abort route leaves nothing worth failing a shutdown over."""
    behaviour = _KoboldBackendBehaviour(abort_route_present=False)

    async with _running_backend(behaviour) as backend:
        await backend.stop(generation_key=_GENERATION_KEY)
        await backend.stop(generation_key="a-second-generation")

    assert behaviour.recorded_abort_bodies == []


async def test_stop_returns_quietly_when_the_backend_is_gone() -> None:
    """Stopping a generation on a backend that has already exited is the expected shutdown ordering."""
    async with _backend_on_a_closed_port() as backend:
        await backend.stop(generation_key=_GENERATION_KEY)


async def test_stop_returns_quietly_when_the_backend_refuses_the_credential() -> None:
    """Abandoning a generation runs where a failure is already being handled, so it raises nothing."""
    behaviour = _KoboldBackendBehaviour(required_password="the-real-password")

    async with _running_backend(behaviour, password="the-wrong-password") as backend:
        await backend.stop(generation_key=_GENERATION_KEY)


async def test_close_is_idempotent_and_leaves_the_session_open() -> None:
    """The session belongs to the caller, so closing the driver twice must not touch it."""
    async with _running_backend(_KoboldBackendBehaviour()) as backend:
        await backend.close()
        await backend.close()

        assert await backend.ready(deadline_seconds=5.0) is True
