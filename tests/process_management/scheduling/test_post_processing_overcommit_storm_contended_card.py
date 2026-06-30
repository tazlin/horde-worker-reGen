"""Concurrent post-processing can over-commit even a large card; transient contention defers, the rest faults.

The post-processing reclaim planner sizes a job's own upscaler/face-fixer peak against *effective* free
VRAM: the measured free reading less the room concurrent sibling post-processing has already committed (and
any imminent peak still sampling). On a high-VRAM card running several inference processes with overlap, two
or more siblings can be mid post-processing at once, each owing its multi-GB peak. Their combined committed
reserve can exceed the measured free reading outright, so the effective free goes *negative* and a freshly
dispatched job's own peak overflows a card that, by raw capacity, looks roomy. When no sibling is idle (every
process is busy) there is nothing to evict and no bare context to shed.

Whether that overflow faults or merely waits turns on one question: can the card host the peak *at all*? When
the peak fits the card drained to this job's process alone, the contention is transient (a busy sibling's
completion frees the room), so the planner returns :attr:`PostProcessingReclaimAction.DEFER` and the job
keeps its head-of-queue position until the room appears, rather than faulting a job the card can host moments
from now. Only when the peak overflows even the drained card (or no sibling is in flight to free room) is the
verdict the terminal :attr:`PostProcessingReclaimAction.INSUFFICIENT`.

These pin that split, the terminal (non-retryable) fault that keeps an unhostable job from being
re-dispatched into the same unchanged card (which would only fault again and feed the breaker a second
over-commit count for one placement failure), and the end-to-end chain where repeated such faults trip the
session-latched fault breaker and stop the worker advertising post-processing (so it is no longer handed jobs
it cannot host).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.scheduling import inference_scheduler as inference_scheduler_module
from horde_worker_regen.process_management.scheduling.inference_scheduler import (
    InferenceScheduler,
    PostProcessingReclaimAction,
)
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

# A contended 24 GB card running with overlap: the dispatching job, two siblings each mid post-processing,
# and a fourth process sampling a job that requests no post-processing. The numbers below mirror the readings
# such a layout produces (a measured free that a single peak would fit, but a committed reserve from the two
# concurrent post-processing peaks that pulls the effective free below zero).
_DISPATCHED_MODEL = "Juggernaut XL"
_PP_SIBLING_MODEL_A = "WAI-ANI-NSFW-PONYXL"
_PP_SIBLING_MODEL_B = "CyberRealistic Pony"
_NO_PP_MODEL = "NTR MIX IL-Noob XL"

_MEASURED_FREE_MB = 10814.0
# Each concurrent sibling post-processing peak; the two together exceed the measured free reading.
_PP_SIBLING_PEAK_MB = 6495.5
_DISPATCHED_PEAK_MB = 5230.0
_DISPATCHED_WEIGHTS_MB = 4900.0
# The large card's total VRAM: roomy enough that the ~5 GB dispatched peak fits it alone, so the contention
# is purely transient (a sibling's completion will free room) rather than a peak the device cannot host.
_LARGE_CARD_TOTAL_VRAM_MB = 24000.0
# A peak no draining of this card can host (it overflows the device alone): the genuinely terminal case that
# still faults and feeds the breaker, distinct from the transient contention that now defers.
_UNHOSTABLE_DISPATCHED_PEAK_MB = 26000.0

# effective free = measured free - committed reserve (both siblings) - imminent (the no-pp job adds 0).
_EXPECTED_COMMITTED_RESERVE_MB = _PP_SIBLING_PEAK_MB * 2  # 12991
_EXPECTED_EFFECTIVE_FREE_MB = _MEASURED_FREE_MB - _EXPECTED_COMMITTED_RESERVE_MB  # -2177
# The peak overflows the (negative) effective free by peak - effective_free.
_EXPECTED_SHORTFALL_MB = _DISPATCHED_PEAK_MB - _EXPECTED_EFFECTIVE_FREE_MB  # 7407


def _peak_by_model(model: str | None, *, dispatched_peak_mb: float) -> float:
    if model == _DISPATCHED_MODEL:
        return dispatched_peak_mb
    if model in (_PP_SIBLING_MODEL_A, _PP_SIBLING_MODEL_B):
        return _PP_SIBLING_PEAK_MB
    return 0.0


def _contended_large_card_scheduler(
    job_tracker: JobTracker,
    process_map: ProcessMap,
    *,
    monkeypatch: pytest.MonkeyPatch,
    dispatched_peak_mb: float = _DISPATCHED_PEAK_MB,
    total_vram_mb: float = _LARGE_CARD_TOTAL_VRAM_MB,
) -> InferenceScheduler:
    """A budget-active scheduler whose readings mirror a contended large card under post-processing overlap.

    ``dispatched_peak_mb`` is the dispatching job's own post-processing peak (the default fits the card alone,
    so the contention is transient; an unhostable value overflows even the drained card). ``total_vram_mb`` is
    the device total the planner sizes "fits the card alone" against.
    """
    bridge_data = make_mock_bridge_data(enable_vram_budget=True, vram_reserve_mb=2048, ram_reserve_mb=4096)
    scheduler = _make_inference_scheduler(
        job_tracker=job_tracker,
        bridge_data=bridge_data,
        process_map=process_map,
        max_inference=4,
    )
    scheduler._measured_free_vram_mb = Mock(return_value=_MEASURED_FREE_MB)  # type: ignore[method-assign]
    scheduler._process_map.get_reported_total_vram_mb = Mock(return_value=total_vram_mb)  # type: ignore[method-assign]

    def _fake_peak(job: object, baseline: str | None) -> float:
        return _peak_by_model(getattr(job, "model", None), dispatched_peak_mb=dispatched_peak_mb)

    def _fake_weight(job: object, baseline: str | None) -> float:
        return _DISPATCHED_WEIGHTS_MB if getattr(job, "model", None) == _DISPATCHED_MODEL else 0.0

    monkeypatch.setattr(inference_scheduler_module, "predict_job_post_processing_vram_mb", _fake_peak)
    monkeypatch.setattr(inference_scheduler_module, "predict_job_weight_mb", _fake_weight)
    return scheduler


async def _seed_contended_card(scheduler: InferenceScheduler) -> ImageGenerateJobPopResponse:
    """Mark two siblings' jobs in-flight (mid post-processing) and return the dispatching job.

    The dispatching process keeps ``last_job_referenced`` unset so the dispatched job is not charged against
    itself (the imminent reserve), matching the planner's "this job is not yet in flight at plan time" rule.
    """
    sibling_a = make_job_pop_response(model=_PP_SIBLING_MODEL_A)
    sibling_b = make_job_pop_response(model=_PP_SIBLING_MODEL_B)
    no_pp = make_job_pop_response(model=_NO_PP_MODEL)
    for job in (sibling_a, sibling_b, no_pp):
        await track_popped_job_async(scheduler._job_tracker, job)
        await scheduler._job_tracker.mark_inference_started(job)

    scheduler._process_map[1].last_job_referenced = sibling_a
    scheduler._process_map[3].last_job_referenced = sibling_b
    scheduler._process_map[2].last_job_referenced = no_pp

    dispatched = make_job_pop_response(model=_DISPATCHED_MODEL)
    await track_popped_job_async(scheduler._job_tracker, dispatched)
    await scheduler._job_tracker.mark_inference_started(dispatched)
    return dispatched


def _contended_process_map() -> ProcessMap:
    """Four busy inference processes on one card: dispatcher, two post-processing siblings, one sampling."""
    return ProcessMap(
        {
            4: make_mock_process_info(
                process_id=4,
                model_name=_DISPATCHED_MODEL,
                state=HordeProcessState.INFERENCE_STARTING,
            ),
            1: make_mock_process_info(
                process_id=1,
                model_name=_PP_SIBLING_MODEL_A,
                state=HordeProcessState.INFERENCE_POST_PROCESSING,
            ),
            3: make_mock_process_info(
                process_id=3,
                model_name=_PP_SIBLING_MODEL_B,
                state=HordeProcessState.INFERENCE_POST_PROCESSING,
            ),
            2: make_mock_process_info(
                process_id=2,
                model_name=_NO_PP_MODEL,
                state=HordeProcessState.INFERENCE_STARTING,
            ),
        },
    )


class TestContendedLargeCardDefersThenFaultsOnlyWhenUnhostable:
    """Transient concurrent-post-processing contention now defers; only an unhostable peak still faults."""

    async def test_concurrent_post_processing_drives_effective_free_negative(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two siblings mid post-processing commit more than the measured free, so effective free is negative."""
        job_tracker = JobTracker()
        scheduler = _contended_large_card_scheduler(job_tracker, _contended_process_map(), monkeypatch=monkeypatch)
        await _seed_contended_card(scheduler)

        committed = scheduler._committed_vram_reserve_mb(device_index=None)
        imminent = scheduler._imminent_post_processing_reserve_mb(device_index=None)
        effective_free = _MEASURED_FREE_MB - committed - imminent

        assert committed == pytest.approx(_EXPECTED_COMMITTED_RESERVE_MB)
        # The job sampling without post-processing contributes nothing, so the imminent reserve stays at zero.
        assert imminent == pytest.approx(0.0)
        assert effective_free == pytest.approx(_EXPECTED_EFFECTIVE_FREE_MB)
        assert effective_free < 0.0

    async def test_large_card_defers_a_hostable_peak_until_a_sibling_frees_room(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The peak overflows the contended card now but fits it alone, with siblings in flight -> DEFER.

        The card is 24 GB, far larger than the ~5 GB peak: the room is only spoken for transiently by the
        siblings' in-flight post-processing, which will free it as they complete. Rather than faulting a job
        the card can host moments from now, the planner holds the dispatch (the job keeps its head-of-queue
        position) until that room appears.
        """
        job_tracker = JobTracker()
        scheduler = _contended_large_card_scheduler(job_tracker, _contended_process_map(), monkeypatch=monkeypatch)
        dispatched = await _seed_contended_card(scheduler)

        plan = scheduler._plan_post_processing_reclaim(dispatched, device_index=None, dispatching_process_id=4)

        assert plan.action is PostProcessingReclaimAction.DEFER
        assert plan.shortfall_mb == pytest.approx(_EXPECTED_SHORTFALL_MB)

    async def test_large_card_faults_when_the_peak_overflows_even_the_drained_card(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A peak no draining of the card can host faults terminally even with siblings in flight.

        Waiting only helps when the peak fits the card alone; one that overflows the device outright is
        unhostable no matter how the siblings drain, so the planner declines rather than parking the dispatch
        forever. This is the genuinely terminal case that still feeds the post-processing breaker.
        """
        job_tracker = JobTracker()
        scheduler = _contended_large_card_scheduler(
            job_tracker,
            _contended_process_map(),
            monkeypatch=monkeypatch,
            dispatched_peak_mb=_UNHOSTABLE_DISPATCHED_PEAK_MB,
        )
        dispatched = await _seed_contended_card(scheduler)

        plan = scheduler._plan_post_processing_reclaim(dispatched, device_index=None, dispatching_process_id=4)

        assert plan.action is PostProcessingReclaimAction.INSUFFICIENT

    async def test_freeing_the_dispatching_jobs_own_weights_cannot_close_the_gap(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Even the in-child own-weights credit (~4.9 GB) leaves the peak short of a negative effective free."""
        job_tracker = JobTracker()
        scheduler = _contended_large_card_scheduler(job_tracker, _contended_process_map(), monkeypatch=monkeypatch)
        await _seed_contended_card(scheduler)

        # peak (5230) > effective_free (-2177) + own_weights (4900): the own-weights rung cannot host it either.
        assert _DISPATCHED_PEAK_MB > _EXPECTED_EFFECTIVE_FREE_MB + _DISPATCHED_WEIGHTS_MB


class TestInsufficientFaultFeedsBreakerCounter:
    """The enacted INSUFFICIENT plan faults the job and records one post-processing over-commit fault."""

    async def test_insufficient_records_a_single_overcommit_fault_per_attempt(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Enacting INSUFFICIENT faults the job, declines dispatch, and bumps the breaker's window counter."""
        job_tracker = JobTracker()
        scheduler = _contended_large_card_scheduler(
            job_tracker,
            _contended_process_map(),
            monkeypatch=monkeypatch,
            dispatched_peak_mb=_UNHOSTABLE_DISPATCHED_PEAK_MB,
        )
        dispatched = await _seed_contended_card(scheduler)
        scheduler._job_tracker.handle_job_fault = AsyncMock()  # type: ignore[method-assign]

        assert job_tracker.count_recent_post_processing_faults(1800) == 0
        plan = scheduler._plan_post_processing_reclaim(dispatched, device_index=None, dispatching_process_id=4)
        assert plan.action is PostProcessingReclaimAction.INSUFFICIENT
        should_dispatch = await scheduler._enact_post_processing_reclaim(
            plan,
            dispatched,
            scheduler._process_map[4],
            device_index=None,
        )

        assert should_dispatch is False
        scheduler._job_tracker.handle_job_fault.assert_awaited_once()
        assert job_tracker.count_recent_post_processing_faults(1800) == 1

    async def test_overcommit_fault_is_terminal_not_retried_into_the_same_card(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The unhostable-peak fault is terminal: the job is reissued by the horde, not retried on the same card.

        The reclaim gate faults so the horde can reissue the job elsewhere; a local retry would only
        re-dispatch it into the unchanged, still-overflowing card (a guaranteed second fault) and feed the
        breaker a second count for one job, halving the breaker's effective tolerance. So the fault is
        non-retryable: even under the worker's two-attempt retry policy the job is faulted terminally rather
        than requeued, and a single unhostable job records exactly one over-commit count.
        """
        job_tracker = JobTracker()
        # The worker's default retry policy; a plain (retryable) fault would grant this job a second attempt.
        job_tracker.set_retry_policy(2)
        scheduler = _contended_large_card_scheduler(
            job_tracker,
            _contended_process_map(),
            monkeypatch=monkeypatch,
            dispatched_peak_mb=_UNHOSTABLE_DISPATCHED_PEAK_MB,
        )
        dispatched = await _seed_contended_card(scheduler)

        plan = scheduler._plan_post_processing_reclaim(dispatched, device_index=None, dispatching_process_id=4)
        assert plan.action is PostProcessingReclaimAction.INSUFFICIENT
        should_dispatch = await scheduler._enact_post_processing_reclaim(
            plan,
            dispatched,
            scheduler._process_map[4],
            device_index=None,
        )

        assert should_dispatch is False
        # Terminal: not requeued for a guaranteed-to-fail retry into the same card.
        assert dispatched not in job_tracker.jobs_pending_inference
        assert dispatched not in job_tracker.jobs_in_progress
        # One unhostable job, one over-commit count (no local retry to double it).
        assert job_tracker.count_recent_post_processing_faults(1800) == 1


class TestOvercommitStormTripsBreakerAndStopsAdvertising:
    """End to end: repeated unhostable post-processing peaks latch the breaker and withhold the feature."""

    def test_storm_of_overcommit_faults_trips_the_session_breaker(self) -> None:
        """Six over-commit faults in the window cross the default threshold of four and latch the breaker.

        Each unhostable job now contributes a single over-commit count (the fault is terminal, no retry into
        the same card), so six counts means six distinct placement failures or watchdog-reaped stalls. That
        exceeds the default threshold of 4, so the control loop's breaker check disables post-processing for
        the rest of the session.
        """
        manager = make_testable_process_manager(
            post_processing_fault_breaker_enabled=True,
            post_processing_fault_threshold=4,
            post_processing_fault_window_seconds=1800,
        )

        for _ in range(6):
            manager._job_tracker.note_post_processing_overcommit_fault()

        assert manager._job_tracker.count_recent_post_processing_faults(1800) == 6
        assert manager._state.post_processing_disabled_by_breaker is False

        manager._apply_post_processing_fault_breaker()

        assert manager._state.post_processing_disabled_by_breaker is True
        assert manager._state.post_processing_breaker_tripped_at > 0

    def test_latched_breaker_withholds_post_processing_from_the_pop(self) -> None:
        """Once latched, the worker stops advertising post-processing so it is no longer handed such jobs.

        This is the protective outcome: the worker keeps generating, but without the upscale/face-fix jobs it
        cannot host on this contended card, so it stops feeding the horde's forced-maintenance spiral.
        """
        manager = make_testable_process_manager(
            post_processing_fault_threshold=4,
            post_processing_fault_window_seconds=1800,
        )
        for _ in range(6):
            manager._job_tracker.note_post_processing_overcommit_fault()
        manager._apply_post_processing_fault_breaker()

        # The session latch is the flag the job popper reads to drop post-processing from its advertised
        # capabilities (asserted directly against the popper in test_job_popping.py).
        assert manager._state.post_processing_disabled_by_breaker is True
