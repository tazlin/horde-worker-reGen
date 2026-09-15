"""The driver for backends that speak the KoboldAI HTTP API.

koboldcpp and sonar are different programs with different internals, but both expose the KoboldAI
routes, so one driver serves both and the worker does not grow a branch per backend.
[`KoboldApiTextBackend`][horde_worker_regen.text_backends.kobold_api.KoboldApiTextBackend] is the only
place in the worker that knows those routes exist; above it there is just the five-verb protocol in
[`protocol`][horde_worker_regen.text_backends.protocol].

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

from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import Final, TypeVar

import aiohttp
from aiohttp import hdrs
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from horde_worker_regen.text_backends.protocol import (
    TextBackendBusy,
    TextBackendDescription,
    TextBackendRejectedPayload,
    TextBackendUnavailable,
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
    """Runs one blocking generation and answers with the generated text."""
    ABORT: Final = "/api/extra/abort"
    """Abandons the generation naming a key. An extension route, so a backend may not have it."""


class KoboldApiJsonKeys:
    """The JSON keys the KoboldAI routes use, for the response models below."""

    RESULT: Final = "result"
    VALUE: Final = "value"
    VALUES: Final = "values"
    RESULTS: Final = "results"
    TEXT: Final = "text"
    GENERATION_KEY: Final = "genkey"
    """The KoboldAI field name for a generation's identifier, used on both generate and abort."""
    SUCCESS: Final = "success"
    DONE: Final = "done"


class KoboldApiTimeouts:
    """Deadlines for the calls whose protocol signature carries none.

    `ready` and `generate` are given a deadline by their caller, because how long to wait for them is a
    scheduling decision. The metadata reads and the abort are small, fixed requests against a local
    program, so they carry their own ceiling rather than inheriting the session's default, which is
    minutes long and would let a wedged backend hold a shutdown open.
    """

    DESCRIBE_SECONDS: Final = 10.0
    ABORT_SECONDS: Final = 10.0


_AUTHORIZATION_BEARER_PREFIX: Final = "Bearer "
"""koboldcpp launched with `--password` accepts the password as a bearer token."""

_REJECTED_PAYLOAD_STATUSES: Final = frozenset({HTTPStatus.BAD_REQUEST, HTTPStatus.UNPROCESSABLE_ENTITY})
"""The statuses that mean the request itself is wrong, so re-offering the same job cannot help."""

_BODY_SUMMARY_CHARACTER_LIMIT: Final = 300
"""How much of an unreadable body is quoted in an exception message: enough to recognise what answered."""


class _RouteNotImplemented(TextBackendUnavailable):
    """A route answered 404, so this backend does not implement it.

    A subclass of `TextBackendUnavailable` so a caller that does not care still sees the failure the
    protocol promises, while the reads of optional routes catch this narrower type and carry on.
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
    """Represents one entry of a generate answer's `results` list."""

    text: str


class _KoboldGenerateResponse(_KoboldResponseModel):
    """Represents a successful generate answer.

    `results` is required and non-empty: a 200 carrying neither is a backend answering nonsense, which
    the driver reports as unavailable rather than inventing an empty generation.
    """

    results: tuple[_KoboldGenerationEntry, ...] = Field(min_length=1)


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
    """

    def __init__(self, base_url: str, session: aiohttp.ClientSession, *, api_key: str | None = None) -> None:
        """Create a driver for the backend serving at `base_url`.

        Args:
            base_url: Where the backend listens, with or without a trailing slash and optionally with a
                path prefix (both backends match their routes by suffix).
            session: The caller's shared session. Not closed by this driver.
            api_key: The password a backend was launched with, if any, sent as a bearer token.
        """
        self._base_url = base_url.rstrip("/")
        self._session = session
        self._api_key = api_key
        self._closed = False
        self._abort_route_missing_logged = False

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
        """Return the request headers, carrying the bearer token only when a key was configured."""
        if self._api_key is None:
            return {}
        return {hdrs.AUTHORIZATION: f"{_AUTHORIZATION_BEARER_PREFIX}{self._api_key}"}

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
            TextBackendUnavailable: The backend did not answer, answered another non-200 status, or
                answered a body that is not the expected shape.
        """
        try:
            raw = await self._request(method=hdrs.METH_GET, route=route, timeout_seconds=timeout_seconds)
        except (aiohttp.ClientError, TimeoutError) as request_error:
            raise TextBackendUnavailable(
                f"Text backend at {self._base_url} did not answer {route}: {request_error}",
            ) from request_error

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

    async def ready(self, *, deadline_seconds: float) -> bool:
        """Return whether the model route answers within the deadline.

        Every failure is a `False`, logged at debug: the caller polls this while a backend is still
        starting, so a refused connection is the normal reading during that window rather than
        something to raise about.

        Args:
            deadline_seconds: How long to wait for the answer.

        Returns:
            `True` when the backend named a loaded model, `False` otherwise.
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

    async def generate(
        self,
        payload: Mapping[str, object],
        *,
        generation_key: str,
        deadline_seconds: float,
    ) -> TextGenerationResult:
        """Return the text the backend generates for the popped horde payload.

        The payload is forwarded as it arrived; the generation key is the one field the driver sets, and
        it is what [`stop`][horde_worker_regen.text_backends.kobold_api.KoboldApiTextBackend.stop] names
        later.

        Args:
            payload: The popped horde payload, forwarded untouched.
            generation_key: The identifier for this generation.
            deadline_seconds: How long to wait for the generation.

        Returns:
            The generated text.

        Raises:
            TextBackendBusy: The backend answered 503, its single generation slot being taken.
            TextBackendRejectedPayload: The backend refused the payload itself.
            TextBackendUnavailable: The backend was unreachable, outlived the deadline, failed with a
                server error, or answered 200 without a usable generation.
        """
        request_body: dict[str, object] = dict(payload)
        request_body[KoboldApiJsonKeys.GENERATION_KEY] = generation_key

        try:
            raw = await self._request(
                method=hdrs.METH_POST,
                route=KoboldApiRoutes.GENERATE,
                timeout_seconds=deadline_seconds,
                json_body=request_body,
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

        return TextGenerationResult(text=generate_response.results[0].text)

    def _raise_for_generate_status(self, raw: _RawResponse, *, generation_key: str) -> None:
        """Raise the protocol exception a generate status calls for, returning only on success.

        Raises:
            TextBackendBusy: The status was 503.
            TextBackendRejectedPayload: The status was one of
                [`_REJECTED_PAYLOAD_STATUSES`][horde_worker_regen.text_backends.kobold_api].
            TextBackendUnavailable: Any other non-200 status.
        """
        if raw.status == HTTPStatus.OK:
            return

        failure_detail = (
            f"backend at {self._base_url}, generation {generation_key}, status {raw.status}: "
            f"{_summarise_body(raw.body_text)}"
        )

        if raw.status == HTTPStatus.SERVICE_UNAVAILABLE:
            raise TextBackendBusy(f"Text backend is already generating ({failure_detail})")

        if raw.status in _REJECTED_PAYLOAD_STATUSES:
            raise TextBackendRejectedPayload(f"Text backend refused the payload ({failure_detail})")

        raise TextBackendUnavailable(f"Text backend failed the generation ({failure_detail})")

    async def stop(self, *, generation_key: str) -> None:
        """Ask the backend to abandon the generation with this key, best effort.

        The outcome is logged and never raised. This runs on shutdown and on a generation that outlived
        its deadline, where the backend may have no abort route or may already be gone, and in neither
        case is there anything left for the caller to do about it.

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


__all__ = [
    "KoboldApiJsonKeys",
    "KoboldApiRoutes",
    "KoboldApiTextBackend",
    "KoboldApiTimeouts",
]
