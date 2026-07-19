"""Tests for the parent-side model-demand poller and its pure snapshot parsing.

Covers the response-to-snapshot conversion, the staleness and queued-per-worker helpers, and the polling
loop's failure tolerance (last good snapshot retained across exceptions and API errors, retry backoff growth),
all against a stubbed submission callable with an injected clock so no real network or wall-clock sleeps run.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest
from horde_sdk import RequestErrorResponse
from horde_sdk.ai_horde_api.apimodels.base import ActiveModel
from horde_sdk.ai_horde_api.apimodels.status import (
    HordeStatusModelsAllRequest,
    HordeStatusModelsAllResponse,
)
from horde_sdk.ai_horde_api.consts import MODEL_TYPE

from horde_worker_regen.process_management.scheduling.model_demand_poller import (
    DemandSnapshot,
    ModelDemandPoller,
    ModelDemandRecord,
    parse_snapshot,
)

_Submitter = Callable[
    [HordeStatusModelsAllRequest, type[HordeStatusModelsAllResponse]],
    Awaitable[HordeStatusModelsAllResponse | RequestErrorResponse],
]


def _active_model(
    name: str | None,
    *,
    queued: float | None = None,
    count: int | None = None,
    eta: int | None = None,
    jobs: float | None = None,
    performance: float | None = None,
) -> ActiveModel:
    """Build an image-type ActiveModel carrying only the fields the poller reads."""
    return ActiveModel(
        type=MODEL_TYPE.image,
        name=name,
        queued=queued,
        count=count,
        eta=eta,
        jobs=jobs,
        performance=performance,
    )


def _response(models: list[ActiveModel]) -> HordeStatusModelsAllResponse:
    """Wrap active models in the endpoint's root response."""
    return HordeStatusModelsAllResponse(root=models)


class TestParseSnapshot:
    """The pure response-to-snapshot conversion."""

    def test_maps_every_named_model_field(self) -> None:
        """Each named model's queue fields land in its record and the fetch time is stamped."""
        response = _response(
            [_active_model("alpha", queued=100.0, count=2, eta=45, jobs=7.0, performance=9.5)],
        )
        snapshot = parse_snapshot(response, now=1000.0)
        record = snapshot.records["alpha"]
        assert record == ModelDemandRecord(
            queued=100.0,
            jobs=7.0,
            eta_seconds=45,
            worker_count=2,
            performance=9.5,
        )
        assert snapshot.fetched_at == 1000.0

    def test_drops_models_without_a_name(self) -> None:
        """A model the endpoint returns without a name cannot be keyed and is dropped."""
        response = _response([_active_model(None, queued=5.0), _active_model("beta", queued=1.0)])
        snapshot = parse_snapshot(response, now=0.0)
        assert set(snapshot.records) == {"beta"}

    def test_duplicate_name_keeps_last(self) -> None:
        """A repeated model name keeps the last entry seen."""
        response = _response([_active_model("gamma", queued=1.0), _active_model("gamma", queued=2.0)])
        snapshot = parse_snapshot(response, now=0.0)
        assert snapshot.records["gamma"].queued == 2.0


class TestSnapshotHelpers:
    """Staleness and queued-per-worker accessors on the immutable snapshot."""

    def test_is_stale_boundary(self) -> None:
        """A snapshot is stale strictly past the max age, not at it."""
        snapshot = DemandSnapshot(records={}, fetched_at=0.0)
        assert not snapshot.is_stale(now=300.0, max_age_seconds=300.0)
        assert snapshot.is_stale(now=300.1, max_age_seconds=300.0)

    def test_queued_per_worker_divides_by_worker_count_plus_one(self) -> None:
        """The ratio uses worker_count + 1 so an unserved queue stays finite."""
        snapshot = DemandSnapshot(
            records={
                "alpha": ModelDemandRecord(queued=30.0, jobs=None, eta_seconds=None, worker_count=2, performance=None)
            },
            fetched_at=0.0,
        )
        assert snapshot.queued_per_worker("alpha") == pytest.approx(10.0)

    def test_queued_per_worker_missing_worker_count_treated_as_zero(self) -> None:
        """A model with no reported worker count divides by one."""
        snapshot = DemandSnapshot(
            records={
                "alpha": ModelDemandRecord(
                    queued=8.0, jobs=None, eta_seconds=None, worker_count=None, performance=None
                )
            },
            fetched_at=0.0,
        )
        assert snapshot.queued_per_worker("alpha") == pytest.approx(8.0)

    def test_queued_per_worker_absent_or_no_queue_is_none(self) -> None:
        """An unknown model, or one with no queued work, reports None rather than zero."""
        snapshot = DemandSnapshot(
            records={
                "alpha": ModelDemandRecord(queued=None, jobs=None, eta_seconds=None, worker_count=1, performance=None)
            },
            fetched_at=0.0,
        )
        assert snapshot.queued_per_worker("absent") is None
        assert snapshot.queued_per_worker("alpha") is None


class TestPollerFailureTolerance:
    """The loop keeps the last good snapshot across failures and grows its retry delay."""

    @pytest.mark.asyncio
    async def test_success_populates_latest(self) -> None:
        """A first successful poll exposes a snapshot; before it, latest is None."""

        async def submit(
            request: HordeStatusModelsAllRequest, response_type: type[HordeStatusModelsAllResponse]
        ) -> HordeStatusModelsAllResponse:
            return _response([_active_model("alpha", queued=10.0, count=1)])

        poller = ModelDemandPoller(submit, interval_seconds=0.001, jitter_fraction=0.0, time_source=lambda: 500.0)
        assert poller.latest() is None
        succeeded = await poller._poll_once()
        assert succeeded
        latest = poller.latest()
        assert latest is not None
        assert latest.records["alpha"].queued == 10.0
        assert latest.fetched_at == 500.0

    @pytest.mark.asyncio
    async def test_exception_retains_last_good_snapshot(self) -> None:
        """A transport exception is swallowed, the prior snapshot is kept, and the failure count grows."""
        should_raise = False

        async def submit(
            request: HordeStatusModelsAllRequest, response_type: type[HordeStatusModelsAllResponse]
        ) -> HordeStatusModelsAllResponse:
            if should_raise:
                raise ConnectionError("endpoint down")
            return _response([_active_model("alpha", queued=10.0, count=1)])

        poller = ModelDemandPoller(submit, interval_seconds=0.001, jitter_fraction=0.0, time_source=lambda: 0.0)
        assert await poller._poll_once()
        good_snapshot = poller.latest()

        should_raise = True
        assert not await poller._poll_once()
        assert not await poller._poll_once()

        assert poller.latest() is good_snapshot
        assert poller._consecutive_failures == 2

    @pytest.mark.asyncio
    async def test_api_error_response_counts_as_failure(self) -> None:
        """A RequestErrorResponse is a failure: no snapshot update, failure count grows."""

        async def submit(
            request: HordeStatusModelsAllRequest, response_type: type[HordeStatusModelsAllResponse]
        ) -> RequestErrorResponse:
            return RequestErrorResponse(message="rate limited")

        poller = ModelDemandPoller(submit, interval_seconds=0.001, jitter_fraction=0.0, time_source=lambda: 0.0)
        assert not await poller._poll_once()
        assert poller.latest() is None
        assert poller._consecutive_failures == 1

    def test_backoff_grows_geometrically_and_is_capped(self) -> None:
        """The retry delay doubles per consecutive failure up to the ceiling; success uses the interval."""
        poller = ModelDemandPoller(
            _unused_submit,
            interval_seconds=10.0,
            jitter_fraction=0.0,
            max_backoff_seconds=100.0,
        )
        assert poller._compute_delay(0) == 10.0
        assert poller._compute_delay(1) == 10.0
        assert poller._compute_delay(2) == 20.0
        assert poller._compute_delay(3) == 40.0
        assert poller._compute_delay(10) == 100.0

    @pytest.mark.asyncio
    async def test_run_survives_failures_and_ends_with_final_snapshot(self) -> None:
        """The loop runs through a failure and stops itself, exposing the final good snapshot."""
        call_count = 0
        stop_event = asyncio.Event()

        async def submit(
            request: HordeStatusModelsAllRequest, response_type: type[HordeStatusModelsAllResponse]
        ) -> HordeStatusModelsAllResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise TimeoutError("blip")
            if call_count >= 3:
                stop_event.set()
                return _response([_active_model("alpha", queued=99.0, count=1)])
            return _response([_active_model("alpha", queued=1.0, count=1)])

        poller = ModelDemandPoller(submit, interval_seconds=0.001, jitter_fraction=0.0, time_source=lambda: 7.0)
        await asyncio.wait_for(poller.run(stop_event), timeout=5.0)

        latest = poller.latest()
        assert latest is not None
        assert latest.records["alpha"].queued == 99.0
        assert poller._consecutive_failures == 0


async def _unused_submit(
    request: HordeStatusModelsAllRequest,
    response_type: type[HordeStatusModelsAllResponse],
) -> HordeStatusModelsAllResponse:
    """Placeholder submitter for tests that never poll."""
    raise AssertionError("submit should not be called")
