"""Headroom-aware scaling of the concurrent-overlap gate, now decided by the VRAM arbiter.

The overlap gate exists to stop stacked weight loads and activation peaks from thrashing a sampler
into a step-timeout teardown. It runs two guards. A temporal/structural guard keeps a newcomer off a
running job's memory-hungry startup beat: two extra-large (whole-card tier) models never share a card, a
pairing with one extra-large side needs confirmed room and the strictest headway, and a heavy pairing (or a
batch) must let the running job make size-appropriate headway first. The memory guard is the VRAM arbiter's:
it prices the candidate's marginal device cost against the cycle-frozen admission floor and answers whether
the card can hold the overlap at all.

When the arbiter admits, a heavy pairing's headway relaxes to a small startup-beat constant, since the
over-subscription the strict fractions guard against cannot occur on a card judged able to hold the
newcomer. When the arbiter withholds (the measured floor is over-committed), the overlap is denied for
the cycle whatever the headway. With no cycle snapshot (cold start, arbiter unwired) the memory answer
relaxes to admit and only the temporal guard applies.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeProcessState, ModelLoadState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.model_sizing import ModelSizeTier
from horde_worker_regen.process_management.resources.vram_arbiter import (
    DeviceVramState,
    MeasuredVramSnapshot,
    VramArbiter,
)
from horde_worker_regen.process_management.scheduling.clearance_lease import CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from horde_worker_regen.process_management.scheduling.workload_flow import PRELOAD_ADMISSION_FLOW
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_HEAVY_A = "sdxl_alpha"
_HEAVY_B = "sdxl_beta"
_EXTRA_LARGE = "flux_like"
_EXTRA_LARGE_OTHER = "qwen_like"


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
            enable_vram_budget=True,
            vram_reserve_mb=2048,
            ram_reserve_mb=4096,
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


class TestExtraLargePairingIsPriced:
    """A pairing with one extra-large side is admitted on the arbiter's priced verdict, never on headroom alone."""

    async def test_running_extra_large_is_joined_on_confirmed_room(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A smaller job joins an extra-large sampler past the strictest headway once the arbiter admits it."""
        tiers = {_EXTRA_LARGE: ModelSizeTier.EXTRA_LARGE, _HEAVY_B: ModelSizeTier.HEAVY}
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=tiers, memory_admits=True)
        await _running(job_tracker, _EXTRA_LARGE)
        _pin_progress(monkeypatch, scheduler, 0.95)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B)) is True

    async def test_running_extra_large_is_not_joined_under_over_commit(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without room an extra-large sampler shares with no one, at any progress."""
        tiers = {_EXTRA_LARGE: ModelSizeTier.EXTRA_LARGE, _HEAVY_B: ModelSizeTier.HEAVY}
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=tiers, memory_admits=False)
        await _running(job_tracker, _EXTRA_LARGE)
        _pin_progress(monkeypatch, scheduler, 0.95)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B)) is False

    async def test_extra_large_pairing_keeps_the_strictest_headway(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Confirmed room relaxes a heavy pairing's headway but never an extra-large one's."""
        tiers = {_EXTRA_LARGE: ModelSizeTier.EXTRA_LARGE, _HEAVY_B: ModelSizeTier.HEAVY}
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=tiers, memory_admits=True)
        await _running(job_tracker, _EXTRA_LARGE)
        _pin_progress(monkeypatch, scheduler, 0.5)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_HEAVY_B)) is False

    async def test_extra_large_candidate_joins_on_confirmed_room(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An extra-large candidate joins a smaller sampler past the strictest headway once the arbiter admits."""
        tiers = {_EXTRA_LARGE: ModelSizeTier.EXTRA_LARGE, _HEAVY_A: ModelSizeTier.HEAVY}
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=tiers, memory_admits=True)
        await _running(job_tracker, _HEAVY_A)
        _pin_progress(monkeypatch, scheduler, 0.95)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_EXTRA_LARGE)) is True

    async def test_second_extra_large_job_never_joins_despite_headroom(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two extra-large jobs never share a card, whether two copies of one model or two different ones."""
        tiers = {_EXTRA_LARGE: ModelSizeTier.EXTRA_LARGE, _EXTRA_LARGE_OTHER: ModelSizeTier.EXTRA_LARGE}
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=tiers, memory_admits=True)
        await _running(job_tracker, _EXTRA_LARGE)
        _pin_progress(monkeypatch, scheduler, 0.95)

        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_EXTRA_LARGE)) is False
        assert scheduler._concurrent_overlap_allowed(make_job_pop_response(model=_EXTRA_LARGE_OTHER)) is False


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

    async def test_staged_candidate_is_not_charged_its_own_preload_twice(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A staged candidate's own preload charge is netted from the overlay, as the materialisation request nets it.

        The card has room for the candidate exactly once; counting its staged preload beside its own delta would
        veto every overlap a staged job asks for.
        """
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=_BOTH_HEAVY, memory_admits=True)
        _install_cycle(
            scheduler,
            DeviceVramState(
                total_vram_mb=16000.0,
                baseline_mb=0.0,
                committed_vram_mb=0.0,
                planned_unmaterialized_mb=6000.0,
                committed_is_stale=False,
                device_free_mb=8000.0,
                noise_buffer_mb=512.0,
            ),
        )
        scheduler._horde_model_map.update_entry(_HEAVY_B, load_state=ModelLoadState.LOADED_IN_RAM, process_id=2)
        scheduler._reserve_ledger.set_planned(
            PRELOAD_ADMISSION_FLOW, "2", vram_mb=6000.0, target_process_id=2, reserved_at_admit_mb=0.0
        )
        monkeypatch.setattr(
            scheduler,
            "_measured_admission_candidate_delta_mb",
            lambda job, baseline, *, process_id, disaggregated: 6000.0,
        )

        assert scheduler._overlap_memory_verdict(make_job_pop_response(model=_HEAVY_B), target_device_index=None)

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


_FULL_JOB_MB = 6000.0
"""A candidate's full measured delta that the staging card below cannot hold beside the running sample."""


def _staging_card_state() -> DeviceVramState:
    """A card with room for a staged job's encode charge and none for the candidate's full job."""
    return DeviceVramState(
        total_vram_mb=16000.0,
        baseline_mb=0.0,
        committed_vram_mb=11000.0,
        planned_unmaterialized_mb=0.0,
        committed_is_stale=False,
        device_free_mb=5000.0,
        noise_buffer_mb=512.0,
    )


class TestOverlapPricesWhatALeasedDispatchPutsOnTheCard:
    """Under the clearance lease a dispatch beside a running sample only stages the job.

    Its weights land at clearance, which prices the full job, so the overlap gate charges the staging cost.
    It does so only while clearance is expected inside the lease-acquire timeout: past it the staged child
    samples without a grant, and the full price at dispatch is then the only guard.
    """

    async def _leased_scheduler(
        self,
        job_tracker: JobTracker,
        monkeypatch: pytest.MonkeyPatch,
        *,
        lease: bool,
        remaining_seconds: float | None,
        load_seconds: float | None = 3.0,
    ) -> InferenceScheduler:
        """A scheduler with one heavy job sampling on lane 1 and a staging-sized card."""
        scheduler = _make_overlap_scheduler(job_tracker, monkeypatch, tiers=_BOTH_HEAVY, memory_admits=True)
        scheduler._runtime_config.bridge_data.gpu_sampling_lease_enabled = lease
        _install_cycle(scheduler, _staging_card_state())
        running_job = await _running(job_tracker, _HEAVY_A)
        lane = scheduler._process_map[1]
        lane.record_inference_ownership(running_job, attempt_ordinal=1)
        lane.last_process_state = HordeProcessState.INFERENCE_STARTING
        monkeypatch.setattr(
            scheduler,
            "_measured_admission_candidate_delta_mb",
            lambda job, baseline, *, process_id, disaggregated: _FULL_JOB_MB,
        )
        monkeypatch.setattr(scheduler, "_remaining_sampling_seconds", lambda process_info: remaining_seconds)
        monkeypatch.setattr(scheduler._process_map, "recent_vram_load_seconds", lambda device_index: load_seconds)
        return scheduler

    async def test_a_staging_dispatch_beside_a_short_sample_fits(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A sample ending well inside the timeout lets the next job stage at the encode charge."""
        scheduler = await self._leased_scheduler(job_tracker, monkeypatch, lease=True, remaining_seconds=5.0)

        assert scheduler._overlap_memory_verdict(make_job_pop_response(model=_HEAVY_B), target_device_index=0)

    async def test_without_the_lease_the_full_job_is_priced(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CONTROL: without the lease the dispatch is the VRAM moment and the full job does not fit."""
        scheduler = await self._leased_scheduler(job_tracker, monkeypatch, lease=False, remaining_seconds=5.0)

        assert scheduler._overlap_memory_verdict(make_job_pop_response(model=_HEAVY_B), target_device_index=0) is False

    async def test_a_sample_outlasting_the_timeout_keeps_the_full_price(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A staged child would wait out the acquire timeout and sample unpriced, so the full job is priced."""
        scheduler = await self._leased_scheduler(
            job_tracker,
            monkeypatch,
            lease=True,
            remaining_seconds=CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS,
        )

        assert scheduler._overlap_memory_verdict(make_job_pop_response(model=_HEAVY_B), target_device_index=0) is False

    async def test_a_sampler_with_no_trusted_rate_keeps_the_full_price(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A running job whose remaining time is unknown could outlast the timeout."""
        scheduler = await self._leased_scheduler(job_tracker, monkeypatch, lease=True, remaining_seconds=None)

        assert scheduler._overlap_memory_verdict(make_job_pop_response(model=_HEAVY_B), target_device_index=0) is False

    async def test_a_staged_running_lane_keeps_the_full_price(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A lane still waiting for its own clearance is not sampling, so no hand-over time can be estimated."""
        scheduler = await self._leased_scheduler(job_tracker, monkeypatch, lease=True, remaining_seconds=5.0)
        scheduler._process_map[1].last_process_state = HordeProcessState.INFERENCE_PRIMED

        assert scheduler._overlap_memory_verdict(make_job_pop_response(model=_HEAVY_B), target_device_index=0) is False

    async def test_an_unmeasured_load_keeps_the_full_price(
        self, job_tracker: JobTracker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a measured weight-load time on the card the wait for clearance cannot be bounded."""
        scheduler = await self._leased_scheduler(
            job_tracker, monkeypatch, lease=True, remaining_seconds=5.0, load_seconds=None
        )

        assert scheduler._overlap_memory_verdict(make_job_pop_response(model=_HEAVY_B), target_device_index=0) is False
