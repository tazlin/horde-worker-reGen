"""Headroom-aware scaling of the concurrent-overlap gate.

The overlap gate exists to stop stacked weight loads and activation peaks from thrashing a sampler
into a step-timeout teardown. Its headway fractions were sized for the tight-VRAM case, but applied
unconditionally they also price a high-VRAM card as if it were tight: on an all-SDXL queue (every
model HEAVY tier) a second thread may only join at 75% progress, and any batched job blocks overlap
outright, so a two-thread worker converges to ~one effective thread regardless of how much free VRAM
the card actually has.

The scaling pinned here keeps the gate but conditions its strictness on measurement
(`InferenceScheduler._overlap_headroom_ample`): when the device's live free VRAM absorbs the
candidate's full predicted sampling peak plus the configured reserve, the heavy-pair headway drops to
a small constant, and a batched job imposes the strictest headway instead of a hard block. The test is
deliberately conservative in the candidate's favor: dispatch requires the candidate's model to already
be resident, so its weights are on the card and the prediction double-counts them as margin.

What never relaxes: an extra-large (whole-card tier) model neither joins a busy card nor shares one,
whatever the headroom; that contract is the tier's, not the card's. And with no measurement (cold
start, budget disabled, partial config) the gate keeps its strict fractions.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.model_sizing import ModelSizeTier
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_HEAVY_A = "sdxl_alpha"
_HEAVY_B = "sdxl_beta"
_EXTRA_LARGE = "flux_like"

_TIERS = {
    _HEAVY_A: ModelSizeTier.EXTRA_LARGE,  # overridden per test where needed
}


def _make_overlap_scheduler(  # noqa: ANN202
    job_tracker: JobTracker,
    monkeypatch: pytest.MonkeyPatch,
    *,
    tiers: dict[str, ModelSizeTier],
    ample: bool,
):
    """A two-slot scheduler with pinned model tiers and a pinned headroom verdict."""
    process_map = ProcessMap({1: make_mock_process_info(1), 2: make_mock_process_info(2)})
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        bridge_data=make_mock_bridge_data(max_threads=2),
        max_concurrent=2,
        max_inference=2,
    )
    monkeypatch.setattr(
        scheduler,
        "_model_size_tier",
        lambda name: tiers.get(name or "", ModelSizeTier.LIGHT),
    )
    monkeypatch.setattr(
        scheduler,
        "_overlap_headroom_ample",
        lambda job, device_index=None: ample,
    )
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
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=_BOTH_HEAVY, ample=False)
        await _running(job_tracker, _HEAVY_A)
        _pin_progress(monkeypatch, scheduler, 0.5)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B)) is False

    async def test_ample_card_admits_second_heavy_early(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the candidate's full peak fitting measured free VRAM, modest progress suffices."""
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=_BOTH_HEAVY, ample=True)
        await _running(job_tracker, _HEAVY_A)
        _pin_progress(monkeypatch, scheduler, 0.2)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B)) is True

    async def test_ample_card_still_grants_startup_beat(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even with headroom, the running job keeps a small headway for its memory-hungry startup."""
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=_BOTH_HEAVY, ample=True)
        await _running(job_tracker, _HEAVY_A)
        _pin_progress(monkeypatch, scheduler, 0.05)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B)) is False


class TestBatchBlockIsBoundedByHeadroom:
    """A batched job blocks overlap only while the card cannot absorb the newcomer's peak."""

    async def test_running_batch_blocks_without_headroom(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CONTROL: a batched in-flight job on a tight card keeps the hard block, at any progress."""
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=_BOTH_HEAVY, ample=False)
        await _running(job_tracker, _HEAVY_A, n_iter=4)
        _pin_progress(monkeypatch, scheduler, 0.9)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B)) is False

    async def test_running_batch_admits_late_join_with_headroom(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With ample headroom, a batched in-flight job imposes the strictest headway, not a wall."""
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=_BOTH_HEAVY, ample=True)
        await _running(job_tracker, _HEAVY_A, n_iter=4)
        _pin_progress(monkeypatch, scheduler, 0.8)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B)) is True

    async def test_running_batch_still_holds_early_join_with_headroom(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The batch's headway stays the strictest tier; headroom does not shrink it further."""
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=_BOTH_HEAVY, ample=True)
        await _running(job_tracker, _HEAVY_A, n_iter=4)
        _pin_progress(monkeypatch, scheduler, 0.5)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B)) is False

    async def test_batched_candidate_admitted_with_headroom_and_headway(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A batched candidate may join late once its whole batched peak fits free VRAM."""
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=_BOTH_HEAVY, ample=True)
        await _running(job_tracker, _HEAVY_A)
        _pin_progress(monkeypatch, scheduler, 0.8)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B, n_iter=4)) is True

    async def test_batched_candidate_blocked_without_headroom(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CONTROL: a batched candidate on a tight card never joins a busy card."""
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=_BOTH_HEAVY, ample=False)
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
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=tiers, ample=True)
        await _running(job_tracker, _EXTRA_LARGE)
        _pin_progress(monkeypatch, scheduler, 0.95)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B)) is False

    async def test_extra_large_candidate_blocked_despite_headroom(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An extra-large candidate never joins a busy card, whatever the headroom."""
        tiers = {_EXTRA_LARGE: ModelSizeTier.EXTRA_LARGE, _HEAVY_A: ModelSizeTier.HEAVY}
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=tiers, ample=True)
        await _running(job_tracker, _HEAVY_A)
        _pin_progress(monkeypatch, scheduler, 0.95)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_EXTRA_LARGE)) is False


class TestHeadroomVerdict:
    """The headroom predicate is measurement-gated: no reading or an inactive budget reads as tight."""

    def _scheduler(self, job_tracker: JobTracker):  # noqa: ANN202
        return _make_inference_scheduler(
            process_map=ProcessMap({1: make_mock_process_info(1)}),
            job_tracker=job_tracker,
            bridge_data=make_mock_bridge_data(max_threads=2),
            max_concurrent=2,
            max_inference=2,
        )

    def test_no_vram_reading_is_not_ample(self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cold start (no process has reported VRAM) keeps the strict gate."""
        scheduler = self._scheduler(job_tracker)
        monkeypatch.setattr(scheduler, "_budget_active", lambda: True)
        monkeypatch.setattr(scheduler, "_measured_free_vram_mb", lambda device_index=None: None)

        assert scheduler._overlap_headroom_ample(make_job_pop_response(model=_HEAVY_B)) is False

    def test_inactive_budget_is_not_ample(self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the VRAM budget disabled there is no trusted reserve to measure against."""
        scheduler = self._scheduler(job_tracker)
        monkeypatch.setattr(scheduler, "_budget_active", lambda: False)
        monkeypatch.setattr(scheduler, "_measured_free_vram_mb", lambda device_index=None: 20000.0)

        assert scheduler._overlap_headroom_ample(make_job_pop_response(model=_HEAVY_B)) is False

    def test_fitting_verdict_is_ample(self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch) -> None:
        """A candidate whose predicted peak plus reserve fits measured free VRAM reads as ample."""
        scheduler = self._scheduler(job_tracker)
        monkeypatch.setattr(scheduler, "_budget_active", lambda: True)
        monkeypatch.setattr(scheduler, "_measured_free_vram_mb", lambda device_index=None: 20000.0)
        scheduler._vram_budget.set_reserve_mb(2048.0)

        job = make_job_pop_response(model="AlbedoBase XL (SDXL)", width=1024, height=1024)
        assert scheduler._overlap_headroom_ample(job) is True

    def test_short_verdict_is_not_ample(self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch) -> None:
        """CONTROL: the same candidate against a nearly-full card reads as tight."""
        scheduler = self._scheduler(job_tracker)
        monkeypatch.setattr(scheduler, "_budget_active", lambda: True)
        monkeypatch.setattr(scheduler, "_measured_free_vram_mb", lambda device_index=None: 1500.0)
        scheduler._vram_budget.set_reserve_mb(2048.0)

        job = make_job_pop_response(model="AlbedoBase XL (SDXL)", width=1024, height=1024)
        assert scheduler._overlap_headroom_ample(job) is False
