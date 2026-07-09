"""Headroom-aware scaling of the concurrent-overlap gate, now decided by the VRAM arbiter.

The overlap gate exists to stop stacked weight loads and activation peaks from thrashing a sampler
into a step-timeout teardown. It runs two guards. A temporal/structural guard keeps a newcomer off a
running job's memory-hungry startup beat: an extra-large (whole-card tier) model neither joins a busy
card nor shares one, and a heavy pairing (or a batch) must let the running job make size-appropriate
headway first. The memory guard is the VRAM arbiter's: it prices the candidate's marginal device cost
against the cycle-frozen admission floor and answers whether the card can hold the overlap at all.

When the arbiter admits, a heavy pairing's headway relaxes to a small startup-beat constant, since the
over-subscription the strict fractions guard against cannot occur on a card judged able to hold the
newcomer. When the arbiter withholds (the measured floor is over-committed), the overlap is denied for
the cycle whatever the headway. With no cycle snapshot (cold start, arbiter unwired) the memory answer
relaxes to admit and only the temporal guard applies.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.model_sizing import ModelSizeTier
from horde_worker_regen.process_management.resources.vram_arbiter import (
    DeviceVramState,
    MeasuredVramSnapshot,
    VramArbiter,
)
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_HEAVY_A = "sdxl_alpha"
_HEAVY_B = "sdxl_beta"
_EXTRA_LARGE = "flux_like"


def _fitting_state() -> DeviceVramState:
    """A device state with ample measured free room, so the arbiter admits any candidate's memory demand."""
    return DeviceVramState(
        total_vram_mb=100000.0,
        baseline_mb=0.0,
        committed_vram_mb=0.0,
        planned_unmaterialized_mb=0.0,
        committed_is_stale=False,
        device_free_mb=100000.0,
    )


def _over_committed_state() -> DeviceVramState:
    """A device state whose measured free room is exhausted, so the arbiter withholds any candidate."""
    return DeviceVramState(
        total_vram_mb=16000.0,
        baseline_mb=0.0,
        committed_vram_mb=16000.0,
        planned_unmaterialized_mb=0.0,
        committed_is_stale=False,
        device_free_mb=100.0,
    )


def _install_cycle(scheduler, state: DeviceVramState) -> None:  # noqa: ANN001
    """Freeze a crafted arbiter cycle on the scheduler so its overlap memory question is deterministic."""
    arbiter = VramArbiter()
    arbiter.begin_cycle(MeasuredVramSnapshot(devices={0: state}))
    scheduler._vram_arbiter = arbiter


def _make_overlap_scheduler(  # noqa: ANN202
    job_tracker: JobTracker,
    monkeypatch: pytest.MonkeyPatch,
    *,
    tiers: dict[str, ModelSizeTier],
    memory_admits: bool,
    high_performance_mode: bool = False,
    moderate_performance_mode: bool = False,
):
    """A two-slot scheduler with pinned model tiers and a crafted arbiter cycle for the memory question."""
    process_map = ProcessMap({1: make_mock_process_info(1), 2: make_mock_process_info(2)})
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        bridge_data=make_mock_bridge_data(
            max_threads=2,
            high_performance_mode=high_performance_mode,
            moderate_performance_mode=moderate_performance_mode,
        ),
        max_concurrent=2,
        max_inference=2,
    )
    monkeypatch.setattr(
        scheduler,
        "_model_size_tier",
        lambda name: tiers.get(name or "", ModelSizeTier.LIGHT),
    )
    _install_cycle(scheduler, _fitting_state() if memory_admits else _over_committed_state())
    return scheduler


async def _running(job_tracker: JobTracker, model: str, *, n_iter: int = 1):  # noqa: ANN202
    """Put one job for ``model`` in flight."""
    job = make_job_pop_response(model=model, n_iter=n_iter)
    await job_tracker.record_popped_job(job)
    await job_tracker.mark_inference_started(job)
    return job


def _pin_progress(monkeypatch: pytest.MonkeyPatch, scheduler, fraction: float) -> None:  # noqa: ANN001
    """Pin the in-flight job's sampling progress fraction."""
    monkeypatch.setattr(scheduler, "_in_flight_progress_fraction", lambda job: fraction)


_BOTH_HEAVY = {_HEAVY_A: ModelSizeTier.HEAVY, _HEAVY_B: ModelSizeTier.HEAVY}


class TestHeavyPairHeadwayScalesWithHeadroom:
    """Two heavy jobs overlap early on a card whose measured free VRAM absorbs the second peak."""

    async def test_tight_card_keeps_strict_headway(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CONTROL: without ample headroom, a second heavy job still waits for 75% progress."""
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=_BOTH_HEAVY, memory_admits=False)
        await _running(job_tracker, _HEAVY_A)
        _pin_progress(monkeypatch, scheduler, 0.5)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B)) is False

    async def test_ample_card_admits_second_heavy_early(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the candidate's full peak fitting measured free VRAM, modest progress suffices."""
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=_BOTH_HEAVY, memory_admits=True)
        await _running(job_tracker, _HEAVY_A)
        _pin_progress(monkeypatch, scheduler, 0.2)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B)) is True

    async def test_ample_card_still_grants_startup_beat(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even with headroom, the running job keeps a small headway for its memory-hungry startup."""
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=_BOTH_HEAVY, memory_admits=True)
        await _running(job_tracker, _HEAVY_A)
        _pin_progress(monkeypatch, scheduler, 0.05)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B)) is False


class TestPerformanceModeShrinksHeadway:
    """Higher performance modes pull a newcomer into the running job's tail sooner (unpriced-memory path).

    The arbiter is left unwired so the memory question relaxes to admit and the strict fraction (not the
    ample-VRAM relaxation) is the value being scaled, isolating the performance-mode effect.
    """

    async def test_default_mode_keeps_full_headway(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CONTROL: with no performance mode, a second heavy job still waits the full both-heavy headway."""
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=_BOTH_HEAVY, memory_admits=True)
        scheduler._vram_arbiter = None
        await _running(job_tracker, _HEAVY_A)
        _pin_progress(monkeypatch, scheduler, 0.5)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B)) is False

    async def test_high_performance_mode_admits_at_scaled_headway(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same 0.5 progress a default worker blocks admits in high-performance mode (0.75 -> 0.375)."""
        scheduler = _make_overlap_scheduler(
            job_tracker, monkeypatch, tiers=_BOTH_HEAVY, memory_admits=True, high_performance_mode=True
        )
        scheduler._vram_arbiter = None
        await _running(job_tracker, _HEAVY_A)
        _pin_progress(monkeypatch, scheduler, 0.5)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B)) is True

    async def test_high_performance_mode_still_holds_below_scaled_headway(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """High mode shrinks but never removes the headway: below the scaled 0.375 the second heavy waits."""
        scheduler = _make_overlap_scheduler(
            job_tracker, monkeypatch, tiers=_BOTH_HEAVY, memory_admits=True, high_performance_mode=True
        )
        scheduler._vram_arbiter = None
        await _running(job_tracker, _HEAVY_A)
        _pin_progress(monkeypatch, scheduler, 0.3)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B)) is False


class TestBatchBlockIsBoundedByHeadroom:
    """A batched job blocks overlap only while the card cannot absorb the newcomer's peak."""

    async def test_running_batch_blocks_without_headroom(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CONTROL: a batched in-flight job on a tight card keeps the hard block, at any progress."""
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=_BOTH_HEAVY, memory_admits=False)
        await _running(job_tracker, _HEAVY_A, n_iter=4)
        _pin_progress(monkeypatch, scheduler, 0.9)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B)) is False

    async def test_running_batch_admits_late_join_with_headroom(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With ample headroom, a batched in-flight job imposes the strictest headway, not a wall."""
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=_BOTH_HEAVY, memory_admits=True)
        await _running(job_tracker, _HEAVY_A, n_iter=4)
        _pin_progress(monkeypatch, scheduler, 0.8)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B)) is True

    async def test_running_batch_still_holds_early_join_with_headroom(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The batch's headway stays the strictest tier; headroom does not shrink it further."""
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=_BOTH_HEAVY, memory_admits=True)
        await _running(job_tracker, _HEAVY_A, n_iter=4)
        _pin_progress(monkeypatch, scheduler, 0.5)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B)) is False

    async def test_batched_candidate_admitted_with_headroom_and_headway(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A batched candidate may join late once its whole batched peak fits free VRAM."""
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=_BOTH_HEAVY, memory_admits=True)
        await _running(job_tracker, _HEAVY_A)
        _pin_progress(monkeypatch, scheduler, 0.8)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B, n_iter=4)) is True

    async def test_batched_candidate_blocked_without_headroom(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CONTROL: a batched candidate on a tight card never joins a busy card."""
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=_BOTH_HEAVY, memory_admits=False)
        await _running(job_tracker, _HEAVY_A)
        _pin_progress(monkeypatch, scheduler, 0.9)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B, n_iter=4)) is False


class TestExtraLargeContractNeverRelaxes:
    """The whole-card tier's no-co-sampling contract is independent of measured headroom."""

    async def test_running_extra_large_blocks_despite_headroom(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An extra-large in-flight job shares with no one, at any progress, on any card."""
        tiers = {_EXTRA_LARGE: ModelSizeTier.EXTRA_LARGE, _HEAVY_B: ModelSizeTier.HEAVY}
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=tiers, memory_admits=True)
        await _running(job_tracker, _EXTRA_LARGE)
        _pin_progress(monkeypatch, scheduler, 0.95)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B)) is False

    async def test_extra_large_candidate_blocked_despite_headroom(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An extra-large candidate never joins a busy card, whatever the headroom."""
        tiers = {_EXTRA_LARGE: ModelSizeTier.EXTRA_LARGE, _HEAVY_A: ModelSizeTier.HEAVY}
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=tiers, memory_admits=True)
        await _running(job_tracker, _HEAVY_A)
        _pin_progress(monkeypatch, scheduler, 0.95)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_EXTRA_LARGE)) is False


class TestMemoryQuestionIsTheArbiter:
    """The overlap gate's memory question is now the VRAM arbiter's authoritative verdict."""

    async def test_overlap_denied_under_measured_over_commit(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even with the running job well past its headway, an over-committed floor withholds the overlap."""
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=_BOTH_HEAVY, memory_admits=False)
        await _running(job_tracker, _HEAVY_A)
        _pin_progress(monkeypatch, scheduler, 0.95)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B)) is False

    async def test_overlap_allowed_within_capacity(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With capacity for the newcomer and the running job past the startup beat, the overlap admits."""
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=_BOTH_HEAVY, memory_admits=True)
        await _running(job_tracker, _HEAVY_A)
        _pin_progress(monkeypatch, scheduler, 0.95)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B)) is True

    async def test_cold_start_relaxes_memory_to_admit(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no cycle snapshot the memory answer relaxes to admit; only the temporal guard remains."""
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=_BOTH_HEAVY, memory_admits=True)
        scheduler._vram_arbiter = None  # unwired: the memory question relaxes to admit
        await _running(job_tracker, _HEAVY_A)
        _pin_progress(monkeypatch, scheduler, 0.95)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B)) is True

    async def test_disagg_candidate_priced_with_sampler_only_delta(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A disaggregation-class candidate is priced disaggregated, so the decode spike is not double-charged."""
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=_BOTH_HEAVY, memory_admits=True)
        await _running(job_tracker, _HEAVY_A)
        _pin_progress(monkeypatch, scheduler, 0.95)
        monkeypatch.setattr(scheduler, "_is_disaggregation_class_eligible", lambda job: True)

        seen: list[bool] = []

        def _record_delta(job, baseline, *, process_id, disaggregated):  # noqa: ANN001, ANN202
            seen.append(disaggregated)
            return 0.0

        monkeypatch.setattr(scheduler, "_measured_admission_candidate_delta_mb", _record_delta)

        scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B))

        assert seen == [True]
