"""Parent-side polling of live per-model queue demand from the AI Horde ``/v2/status/models`` endpoint.

The horde publishes, per active image model, how much work is queued and how many worker threads serve it.
Scheduling and pop-shaping code on the parent wants that signal to bias which models the worker offers and
loads toward where the demand actually is. This module fetches that signal on a jittered interval and exposes
the most recent good reading as an immutable :class:`DemandSnapshot`.

The poller is torch-free and asyncio-native. It never touches the process manager: the API submission
dependency is injected at construction (a callable matching ``AIHordeAPIAsyncClientSession.submit_request``),
so the same loop drives against a real session in the worker and against a stub in tests. A transient API
failure never propagates out of the loop and never discards the last good snapshot; consecutive failures grow
the retry delay up to a ceiling, and warnings are throttled so a sustained outage does not flood the log.

Public surface:

- :class:`ModelDemandRecord`: one model's queued work, job count, eta, worker count and average speed.
- :class:`DemandSnapshot`: an immutable name-keyed reading with a fetch timestamp and staleness helpers.
- :class:`ModelDemandPoller`: the long-lived polling loop plus :meth:`ModelDemandPoller.latest` accessor.
- :func:`parse_snapshot`: the pure response-to-snapshot conversion, usable without the loop.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from horde_sdk import RequestErrorResponse
from horde_sdk.ai_horde_api.apimodels.status import (
    HordeStatusModelsAllRequest,
    HordeStatusModelsAllResponse,
)
from horde_sdk.ai_horde_api.consts import MODEL_TYPE
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = [
    "DemandSnapshot",
    "ModelDemandPoller",
    "ModelDemandRecord",
    "StatusModelsSubmitter",
    "parse_snapshot",
]

_DEFAULT_POLL_INTERVAL_SECONDS = 90.0
"""Baseline delay between successful demand polls; demand shifts slowly enough that a tighter cadence would
spend request budget without changing scheduling decisions."""

_DEFAULT_JITTER_FRACTION = 0.15
"""Fraction of the delay added as random jitter so many workers polling the same endpoint do not synchronise
into a thundering herd."""

_MAX_BACKOFF_SECONDS = 300.0
"""Ceiling on the retry delay after repeated failures; the loop keeps retrying at this cadence rather than
giving up, since the endpoint recovering must be noticed without an operator."""

_WARNING_THROTTLE_SECONDS = 120.0
"""Minimum spacing between failure warnings so a sustained outage logs periodically instead of every retry."""

_DEFAULT_STALE_AGE_SECONDS = 300.0
"""Age past which a snapshot is treated as stale by :meth:`DemandSnapshot.is_stale` when no override is given;
consumers past this horizon should discount the reading rather than trust it as current demand."""


class StatusModelsSubmitter(Protocol):
    """The injected API dependency: submit a status-models request and return its response.

    Structurally matches ``AIHordeAPIAsyncClientSession.submit_request`` so the worker can pass its live
    session's bound method directly, while tests pass a stub coroutine.
    """

    async def __call__(
        self,
        api_request: HordeStatusModelsAllRequest,
        expected_response_type: type[HordeStatusModelsAllResponse],
        /,
    ) -> HordeStatusModelsAllResponse | RequestErrorResponse:
        """Return the endpoint's response, or a :class:`RequestErrorResponse` on an API-level error.

        The parameters are positional-only so any implementation, including the live session's
        ``submit_request`` bound method and test stubs, satisfies the protocol regardless of its parameter
        names.
        """
        ...


@dataclass(frozen=True)
class ModelDemandRecord:
    """Represents one image model's live queue demand as reported by ``/v2/status/models``.

    ``queued`` is the outstanding work in megapixelsteps, ``jobs`` the count of queued jobs, ``eta_seconds``
    the horde's estimate to clear that queue, ``worker_count`` the number of worker threads serving the model,
    and ``performance`` its reported average generation speed. Every field is optional because the endpoint
    omits values it has no data for.
    """

    queued: float | None
    jobs: float | None
    eta_seconds: int | None
    worker_count: int | None
    performance: float | None


@dataclass(frozen=True)
class DemandSnapshot:
    """Represents an immutable name-keyed reading of per-model demand taken at ``fetched_at``.

    ``records`` maps model name to its :class:`ModelDemandRecord`; ``fetched_at`` is the wall-clock time the
    reading was parsed. The snapshot is a value object: consumers hold it and query it, and a fresh snapshot
    replaces it wholesale rather than mutating it in place.
    """

    records: Mapping[str, ModelDemandRecord]
    fetched_at: float

    def is_stale(self, now: float, max_age_seconds: float = _DEFAULT_STALE_AGE_SECONDS) -> bool:
        """Return whether this reading is older than ``max_age_seconds`` relative to ``now``."""
        return (now - self.fetched_at) > max_age_seconds

    def queued_per_worker(self, model: str) -> float | None:
        """Return the model's queued work divided by its serving-worker count plus one, or ``None``.

        The plus-one keeps the ratio finite when no worker yet serves the model (the horde reports the queue
        before any worker picks it up) and biases the signal toward genuinely under-served models. Returns
        ``None`` when the model is absent from the snapshot or reported no queued work, so callers can treat
        an unknown model as carrying no demand rather than zero-versus-missing ambiguity.
        """
        record = self.records.get(model)
        if record is None or record.queued is None:
            return None
        worker_count = record.worker_count if record.worker_count is not None else 0
        return record.queued / (worker_count + 1)


def parse_snapshot(response: HordeStatusModelsAllResponse, now: float) -> DemandSnapshot:
    """Convert a status-models response into an immutable :class:`DemandSnapshot` stamped at ``now``.

    Models the endpoint returns without a name are dropped (they cannot be keyed or matched to a local model),
    and a duplicate name keeps the last entry seen. Pure and side-effect free so tests and the dry-run harness
    can feed canned responses without the polling loop.
    """
    records: dict[str, ModelDemandRecord] = {}
    for active_model in response.root:
        if active_model.name is None:
            continue
        records[active_model.name] = ModelDemandRecord(
            queued=active_model.queued,
            jobs=active_model.jobs,
            eta_seconds=active_model.eta,
            worker_count=active_model.count,
            performance=active_model.performance,
        )
    return DemandSnapshot(records=records, fetched_at=now)


class ModelDemandPoller:
    """Long-lived loop that refreshes and exposes the latest per-model demand snapshot.

    The loop submits an image-type status-models request on a jittered interval, parses the response into a
    :class:`DemandSnapshot`, and retains it for :meth:`latest`. Failures (transport exceptions or API-level
    :class:`RequestErrorResponse`) never propagate out of the loop and never discard the last good snapshot;
    consecutive failures grow the retry delay geometrically up to :data:`_MAX_BACKOFF_SECONDS`, and warnings
    are throttled so a sustained outage logs periodically rather than on every retry. Cancellation propagates
    so the owning task set can shut the loop down cleanly.

    Thread Safety:
        Single-consumer within one event loop. It holds no locks and is not safe to drive from multiple
        loops concurrently.
    """

    def __init__(
        self,
        submit_request: StatusModelsSubmitter,
        *,
        interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        jitter_fraction: float = _DEFAULT_JITTER_FRACTION,
        max_backoff_seconds: float = _MAX_BACKOFF_SECONDS,
        time_source: Callable[[], float] = time.time,
        random_source: Callable[[], float] = random.random,
    ) -> None:
        """Configure the poller against an injected API submission dependency.

        Args:
            submit_request: The status-models submission callable, matching
                ``AIHordeAPIAsyncClientSession.submit_request``.
            interval_seconds: Baseline delay between successful polls.
            jitter_fraction: Fraction of the delay added as random jitter to desynchronise pollers.
            max_backoff_seconds: Ceiling on the failure-driven retry delay.
            time_source: Wall-clock source, injectable for deterministic tests.
            random_source: Source of a ``[0, 1)`` float for jitter, injectable for deterministic tests.
        """
        self._submit_request = submit_request
        self._interval_seconds = interval_seconds
        self._jitter_fraction = jitter_fraction
        self._max_backoff_seconds = max_backoff_seconds
        self._time_source = time_source
        self._random_source = random_source

        self._latest: DemandSnapshot | None = None
        self._consecutive_failures = 0
        self._last_warning_at: float | None = None

    def latest(self) -> DemandSnapshot | None:
        """Return the most recent good demand snapshot, or ``None`` before the first successful poll."""
        return self._latest

    def seed(self, snapshot: DemandSnapshot) -> None:
        """Install ``snapshot`` as the latest reading without polling.

        The harness and tests use this to hand the pool a synthetic demand signal in sessions where the
        polling loop never runs (dry-run and canned-pop soaks have no live API to query). Seeding follows the
        same staleness contract as a polled reading: consumers judge freshness by the snapshot's own
        ``fetched_at``, so a seeder that wants the reading to stay fresh re-seeds on a cadence.
        """
        self._latest = snapshot

    def _compute_delay(self, consecutive_failures: int) -> float:
        """Return the pre-jitter delay before the next poll given the consecutive-failure count.

        A success (zero failures) uses the baseline interval; each additional consecutive failure doubles the
        delay geometrically, capped at ``max_backoff_seconds``, so a flapping endpoint is retried gently
        without abandoning it.
        """
        if consecutive_failures <= 0:
            return self._interval_seconds
        backoff = self._interval_seconds * (2 ** (consecutive_failures - 1))
        return min(backoff, self._max_backoff_seconds)

    def _apply_jitter(self, delay: float) -> float:
        """Return ``delay`` widened by a random fraction of itself, bounded by ``jitter_fraction``."""
        if self._jitter_fraction <= 0:
            return delay
        return delay * (1 + self._jitter_fraction * self._random_source())

    def _maybe_warn(self, failure_description: str) -> None:
        """Log a throttled failure warning, retaining the last good snapshot without raising."""
        now = self._time_source()
        recently_warned = (
            self._last_warning_at is not None and (now - self._last_warning_at) < _WARNING_THROTTLE_SECONDS
        )
        if recently_warned:
            return
        self._last_warning_at = now
        logger.warning(
            f"Model-demand poll failed ({self._consecutive_failures} consecutive): {failure_description}. "
            "Retaining last good snapshot.",
        )

    async def _poll_once(self) -> bool:
        """Perform one poll, updating the retained snapshot on success. Return whether it succeeded.

        Never raises for transport or API-level failures: those are logged (throttled) and reported as a
        ``False`` return so the loop can back off. ``asyncio.CancelledError`` is allowed to propagate so
        shutdown is not swallowed.
        """
        request = HordeStatusModelsAllRequest(type=MODEL_TYPE.image)
        try:
            response = await self._submit_request(request, HordeStatusModelsAllResponse)
        except asyncio.CancelledError:
            raise
        except Exception as exception:  # noqa: BLE001 - a poll failure must never break the loop
            self._consecutive_failures += 1
            self._maybe_warn(f"{type(exception).__name__}: {exception}")
            return False

        if isinstance(response, RequestErrorResponse):
            self._consecutive_failures += 1
            self._maybe_warn(f"API error: {response.message}")
            return False

        self._latest = parse_snapshot(response, self._time_source())
        self._consecutive_failures = 0
        self._last_warning_at = None
        return True

    async def _sleep_before_next_poll(self, stop_event: asyncio.Event) -> None:
        """Wait the jittered inter-poll delay, waking early if ``stop_event`` is set."""
        delay = self._apply_jitter(self._compute_delay(self._consecutive_failures))
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=delay)
        except TimeoutError:
            return

    async def run(self, stop_event: asyncio.Event) -> None:
        """Poll model demand until ``stop_event`` is set, retaining the latest good snapshot throughout.

        Each iteration polls once and then sleeps the interval (grown by backoff after failures), waking early
        when ``stop_event`` is set. Cancellation propagates to end the loop promptly.
        """
        while not stop_event.is_set():
            await self._poll_once()
            if stop_event.is_set():
                return
            await self._sleep_before_next_poll(stop_event)
