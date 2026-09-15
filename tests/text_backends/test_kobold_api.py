"""The KoboldAI driver is the worker's whole conversation with a program it does not control.

These tests drive the real driver against a local `aiohttp` application whose answers are shaped like
koboldcpp's: the same routes, the same JSON keys, the same string booleans on abort, and the same 503
for "one generation at a time and I am on it". Two things they pin above all: the horde payload reaches
the backend exactly as it arrived, with only the generation key added, and every failure mode surfaces
as one of the three protocol exceptions so the flow never has to look at HTTP.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from http import HTTPStatus

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from horde_worker_regen.text_backends import (
    KoboldApiJsonKeys,
    KoboldApiRoutes,
    KoboldApiTextBackend,
    TextBackend,
    TextBackendBusy,
    TextBackendRejectedPayload,
    TextBackendUnavailable,
)

_MODEL_NAME = "koboldcpp/a-model.gguf"
_MAX_CONTEXT_LENGTH = 4096
_MAX_LENGTH = 512
_GENERATED_TEXT = "the backend's own words"
_GENERATION_KEY = "FEDCBA98"

_BUSY_BODY = {"detail": {"msg": "Server is busy; please try again later.", "type": "service_unavailable"}}
"""koboldcpp's own 503 body, so the driver is exercised against the shape it will really see."""

_SLOW_ANSWER_SECONDS = 0.5
"""Long enough to outlast a test's deadline, short enough that the server shuts down without waiting."""

_SHORT_DEADLINE_SECONDS = 0.05


@dataclass
class _KoboldBackendBehaviour:
    """How the fake KoboldAI server should answer, and what it recorded while doing so.

    Defaults describe a healthy koboldcpp with a model loaded, no soft prompts and a working abort
    route; each test changes only the one thing it is about.
    """

    model_name: str = _MODEL_NAME
    model_status: int = HTTPStatus.OK
    model_delay_seconds: float = 0.0
    max_context_length: int = _MAX_CONTEXT_LENGTH
    max_length: int = _MAX_LENGTH
    soft_prompts: tuple[str, ...] | None = ()
    """The list the soft-prompts route answers with; `None` makes that route answer 404."""
    generate_status: int = HTTPStatus.OK
    generate_body: object | None = None
    """An explicit generate body, or `None` for koboldcpp's faithful `results` answer."""
    generate_delay_seconds: float = 0.0
    abort_route_present: bool = True
    recorded_generate_bodies: list[object] = field(default_factory=list)
    recorded_generate_authorizations: list[str | None] = field(default_factory=list)
    recorded_abort_bodies: list[object] = field(default_factory=list)


def _build_kobold_app(behaviour: _KoboldBackendBehaviour) -> web.Application:
    """Return an application answering the KoboldAI routes the way `behaviour` says to."""

    async def handle_model(request: web.Request) -> web.Response:
        """Answer the model route, which is what readiness and the advertised name both read."""
        if behaviour.model_delay_seconds > 0:
            await asyncio.sleep(behaviour.model_delay_seconds)
        if behaviour.model_status != HTTPStatus.OK:
            return web.json_response({"detail": "no model"}, status=behaviour.model_status)
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

    async def handle_generate(request: web.Request) -> web.Response:
        """Record the forwarded request and answer as the behaviour dictates."""
        behaviour.recorded_generate_bodies.append(await request.json())
        behaviour.recorded_generate_authorizations.append(request.headers.get(aiohttp.hdrs.AUTHORIZATION))
        if behaviour.generate_delay_seconds > 0:
            await asyncio.sleep(behaviour.generate_delay_seconds)
        if behaviour.generate_status == HTTPStatus.SERVICE_UNAVAILABLE:
            return web.json_response(_BUSY_BODY, status=HTTPStatus.SERVICE_UNAVAILABLE)
        if behaviour.generate_status != HTTPStatus.OK:
            return web.json_response({"detail": "nope"}, status=behaviour.generate_status)
        if behaviour.generate_body is not None:
            return web.json_response(behaviour.generate_body)
        return web.json_response(
            {
                KoboldApiJsonKeys.RESULTS: [
                    {
                        KoboldApiJsonKeys.TEXT: _GENERATED_TEXT,
                        "finish_reason": "length",
                        "prompt_tokens": 12,
                        "completion_tokens": 34,
                    },
                ],
            },
        )

    async def handle_abort(request: web.Request) -> web.Response:
        """Record the abort and answer koboldcpp's two flags, which are strings rather than booleans."""
        behaviour.recorded_abort_bodies.append(await request.json())
        return web.json_response({KoboldApiJsonKeys.SUCCESS: "true", KoboldApiJsonKeys.DONE: "true"})

    app = web.Application()
    app.router.add_get(KoboldApiRoutes.MODEL, handle_model)
    app.router.add_get(KoboldApiRoutes.MAX_CONTEXT_LENGTH, handle_max_context_length)
    app.router.add_get(KoboldApiRoutes.MAX_LENGTH, handle_max_length)
    app.router.add_get(KoboldApiRoutes.SOFT_PROMPTS_LIST, handle_soft_prompts)
    app.router.add_post(KoboldApiRoutes.GENERATE, handle_generate)
    if behaviour.abort_route_present:
        app.router.add_post(KoboldApiRoutes.ABORT, handle_abort)
    return app


@asynccontextmanager
async def _running_backend(
    behaviour: _KoboldBackendBehaviour,
    *,
    api_key: str | None = None,
) -> AsyncIterator[KoboldApiTextBackend]:
    """Serve `behaviour` on an ephemeral port and yield a driver pointed at it."""
    server = TestServer(_build_kobold_app(behaviour))
    await server.start_server()
    session = aiohttp.ClientSession()
    try:
        yield KoboldApiTextBackend(f"http://{server.host}:{server.port}", session, api_key=api_key)
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
    """503 is the KoboldAI way of saying the single generation slot is taken, so the job can be re-offered."""
    behaviour = _KoboldBackendBehaviour(generate_status=HTTPStatus.SERVICE_UNAVAILABLE)

    async with _running_backend(behaviour) as backend:
        with pytest.raises(TextBackendBusy):
            await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=10.0)


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
    behaviour = _KoboldBackendBehaviour(generate_body=malformed_body)

    async with _running_backend(behaviour) as backend:
        with pytest.raises(TextBackendUnavailable) as raised:
            await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=10.0)

    assert KoboldApiJsonKeys.RESULTS in str(raised.value)


async def test_generate_past_the_deadline_is_unavailable() -> None:
    """A generation that outlives its deadline is abandoned by the driver rather than waited on."""
    behaviour = _KoboldBackendBehaviour(generate_delay_seconds=_SLOW_ANSWER_SECONDS)

    async with _running_backend(behaviour) as backend:
        with pytest.raises(TextBackendUnavailable):
            await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=_SHORT_DEADLINE_SECONDS)


async def test_generate_raises_unavailable_when_the_backend_is_not_listening() -> None:
    """A backend that died mid-run fails the generation as unavailable, not as a bad payload."""
    async with _backend_on_a_closed_port() as backend:
        with pytest.raises(TextBackendUnavailable):
            await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=5.0)


async def test_generate_sends_the_api_key_as_a_bearer_token() -> None:
    """Koboldcpp launched with a password wants that password as a bearer token."""
    behaviour = _KoboldBackendBehaviour()

    async with _running_backend(behaviour, api_key="a-launch-password") as backend:
        await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=10.0)

    assert behaviour.recorded_generate_authorizations == ["Bearer a-launch-password"]


async def test_generate_sends_no_authorization_header_without_a_key() -> None:
    """An unprotected backend must not be sent an empty credential."""
    behaviour = _KoboldBackendBehaviour()

    async with _running_backend(behaviour) as backend:
        await backend.generate({}, generation_key=_GENERATION_KEY, deadline_seconds=10.0)

    assert behaviour.recorded_generate_authorizations == [None]


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


async def test_close_is_idempotent_and_leaves_the_session_open() -> None:
    """The session belongs to the caller, so closing the driver twice must not touch it."""
    async with _running_backend(_KoboldBackendBehaviour()) as backend:
        await backend.close()
        await backend.close()

        assert await backend.ready(deadline_seconds=5.0) is True
