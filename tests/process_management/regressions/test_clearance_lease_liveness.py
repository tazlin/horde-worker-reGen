"""Regressions for the per-process GPU denoise clearance lease: staging liveness, head protection, and holds.

These pin the throughput unlock the clearance lease exists for (a spare process stages the next job ahead
while another samples, instead of idling), the head-protection invariant (siblings staging under the
encode-only charge never starve the head's full-materialisation room), the ledger charge upgrade at
clearance (encode charge upgraded in place, not double-booked), the slot-duty attribution of a held
clearance, and the pool liveness guarantees (a never-fitting admission degrades rather than wedging; a full
slot cap holds the next grant until a window retires).
"""

from __future__ import annotations

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessState, HordeProcessType
from horde_worker_regen.process_management.resources.admission_identity import admission_noise_buffer_mb
from horde_worker_regen.process_management.scheduling import inference_scheduler as _sched_mod
from horde_worker_regen.process_management.scheduling.clearance_lease import (
    ActiveSampler,
    ClearanceController,
    ClearanceInputs,
    ClearanceLeaseProxy,
    ClearanceWaiter,
    GrantState,
)
from horde_worker_regen.process_management.scheduling.slot_duty import SlotDutyBucket
from horde_worker_regen.process_management.scheduling.workload_flow import DISPATCH_ADMISSION_FLOW
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap  # isort: skip

_ENCODE_MB = 2048.0


class TestStagingLivenessUnlock:
    """A spare process must be allowed to stage the next job ahead under tight-but-encode-fitting device free.

    The live regression: with the lease enabled but staging gated on a flat multi-GB device-free floor, a
    spare inference process sat idle through nearly every sampling snapshot despite queued work, because
    mid-sampling the device rarely cleared that floor. Staging now funds only the encode working set, so a
    spare process stages ahead whenever free net of the reserve covers the encode charge.
    """

    def _vram_process_map(self, free_mb: int) -> ProcessMap:
        proc = make_mock_process_info(0)
        proc.total_vram_mb = 16000
        proc.vram_usage_mb = 16000 - free_mb
        return ProcessMap({0: proc})

    def test_spare_process_stages_ahead_under_tight_free(self) -> None:
        """Free between the encode charge and the old flat floor now stages ahead instead of idling the spare."""
        scheduler = _make_inference_scheduler(
            process_map=self._vram_process_map(2500),  # above the 2048 encode charge, below the old 3000 floor
            max_concurrent=2,
            max_inference=4,
            bridge_data=make_mock_bridge_data(gpu_sampling_lease_enabled=True),
        )
        assert scheduler._max_jobs_in_progress_allowed() == 4

    def test_staging_withheld_when_free_cannot_cover_the_encode_charge(self) -> None:
        """Below the encode charge even staging is withheld, so speculation never over-commits the device."""
        scheduler = _make_inference_scheduler(
            process_map=self._vram_process_map(1500),  # under the 2048 encode charge
            max_concurrent=2,
            max_inference=4,
            bridge_data=make_mock_bridge_data(gpu_sampling_lease_enabled=True),
        )
        assert scheduler._max_jobs_in_progress_allowed() == 2


class TestDispatchReservationCharge:
    """A staged dispatch books only the encode charge; clearance upgrades it in place to the full peak."""

    def _reserved_vram_mb(self, scheduler: object) -> float:
        # The dispatch-flow planned overlay for this run, with no materialisation yet observed on any process.
        ledger = scheduler._reserve_ledger  # type: ignore[attr-defined]
        return ledger.effective_planned_vram_mb_for_flow(DISPATCH_ADMISSION_FLOW, {})

    def test_staging_books_encode_charge_not_full_materialisation(self) -> None:
        """Under the lease a dispatch reserves the encode working set, not the weights-plus-activation peak."""
        scheduler = _make_inference_scheduler(
            bridge_data=make_mock_bridge_data(gpu_sampling_lease_enabled=True),
        )
        proc = make_mock_process_info(0, model_name="stable_diffusion")
        job = make_job_pop_response("stable_diffusion")

        scheduler._record_dispatch_reservation(job, proc, baseline=None, staging_only=True)
        assert self._reserved_vram_mb(scheduler) == _ENCODE_MB

    def test_clearance_upgrades_the_charge_in_place_without_double_booking(self) -> None:
        """Upgrading at clearance re-books the same ledger unit at the full peak, never additively."""
        scheduler = _make_inference_scheduler(
            bridge_data=make_mock_bridge_data(gpu_sampling_lease_enabled=True),
        )
        proc = make_mock_process_info(0, model_name="stable_diffusion")
        job = make_job_pop_response("stable_diffusion")
        scheduler._record_dispatch_reservation(job, proc, baseline=None, staging_only=True)

        # The full materialisation charge for this job (what the upgrade will book).
        full_mb = scheduler._measured_admission_candidate_delta_mb(job, None, process_id=0, disaggregated=False)
        assert full_mb is not None and full_mb > _ENCODE_MB  # the full peak exceeds the encode-only staging charge

        scheduler._upgrade_dispatch_reservation_to_full(job, proc, baseline=None)
        upgraded = self._reserved_vram_mb(scheduler)
        # A single entry upgraded in place: the reserved total is the full peak, not encode + full (double-book).
        assert upgraded == full_mb
        assert upgraded != _ENCODE_MB + full_mb


class TestDispatchReservationDisaggregationPricing:
    """A class-eligible dispatch's full charge is priced sampler-only, matching the other measured-admission sites.

    The upgrade at clearance re-books the dispatch reservation at the full materialisation charge. A
    disaggregation-class-eligible job loads its weights and runs its decode off the sampling process, so that
    full charge is the UNet-only sampler figure, not the monolithic whole-job peak. This pins that the charge
    tracks the job's class eligibility rather than an unconditional monolithic price.
    """

    _WHOLE_JOB_MB = 9000.0
    _SAMPLER_ONLY_MB = 4000.0

    def _pin_predictors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_sched_mod, "predict_job_sampling_vram_mb", lambda _job, _baseline: self._WHOLE_JOB_MB)
        monkeypatch.setattr(
            _sched_mod,
            "predict_job_sampler_only_vram_mb",
            lambda _job, _baseline: self._SAMPLER_ONLY_MB,
        )

    def _reserved_vram_mb(self, scheduler: object) -> float:
        ledger = scheduler._reserve_ledger  # type: ignore[attr-defined]
        return ledger.effective_planned_vram_mb_for_flow(DISPATCH_ADMISSION_FLOW, {})

    def test_class_eligible_full_charge_uses_sampler_only_pricing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A disaggregation-class-eligible job books its full charge at the sampler-only figure, not whole-job."""
        self._pin_predictors(monkeypatch)
        scheduler = _make_inference_scheduler(
            bridge_data=make_mock_bridge_data(gpu_sampling_lease_enabled=True),
        )
        scheduler._is_disaggregation_class_eligible = lambda _job: True
        proc = make_mock_process_info(0, model_name="stable_diffusion")
        job = make_job_pop_response("stable_diffusion")

        scheduler._record_dispatch_reservation(job, proc, baseline=None, staging_only=False)
        assert self._reserved_vram_mb(scheduler) == self._SAMPLER_ONLY_MB

    def test_monolithic_job_full_charge_uses_whole_job_pricing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A job that will not run disaggregated still books the whole-job peak (the predicate governs)."""
        self._pin_predictors(monkeypatch)
        scheduler = _make_inference_scheduler(
            bridge_data=make_mock_bridge_data(gpu_sampling_lease_enabled=True),
        )
        scheduler._is_disaggregation_class_eligible = lambda _job: False
        proc = make_mock_process_info(0, model_name="stable_diffusion")
        job = make_job_pop_response("stable_diffusion")

        scheduler._record_dispatch_reservation(job, proc, baseline=None, staging_only=False)
        assert self._reserved_vram_mb(scheduler) == self._WHOLE_JOB_MB


class TestHeadNotStarvedBySiblingStaging:
    """Siblings staging under the encode charge must never consume the head's full-materialisation room."""

    def test_many_staged_siblings_book_only_encode_each(self) -> None:
        """Three siblings staging book 3x the encode charge, not 3x a full materialisation peak.

        Hostile: were staging to book full materialisation, a few spare processes staging ahead would reserve
        the whole card and starve the head. Charging only the encode working set per staged sibling keeps the
        card's full-materialisation room available for the head's own clearance claim.
        """
        scheduler = _make_inference_scheduler(
            bridge_data=make_mock_bridge_data(gpu_sampling_lease_enabled=True),
        )
        ledger = scheduler._reserve_ledger
        for pid in range(3):
            proc = make_mock_process_info(pid, model_name="stable_diffusion")
            job = make_job_pop_response("stable_diffusion")
            scheduler._record_dispatch_reservation(job, proc, baseline=None, staging_only=True)

        total = ledger.effective_planned_vram_mb_for_flow(DISPATCH_ADMISSION_FLOW, {})
        assert total == 3 * _ENCODE_MB


class TestClearanceAdmission:
    """The clearance admit function prices the full materialisation and upgrades the reservation on a grant."""

    def test_admits_and_upgrades_when_device_has_ample_room(self) -> None:
        """An ample card admits the staged child's clearance and upgrades its reservation to the full peak."""
        scheduler = _make_inference_scheduler(
            bridge_data=make_mock_bridge_data(
                gpu_sampling_lease_enabled=True,
                enable_vram_budget=True,
                vram_reserve_mb=2048,
                ram_reserve_mb=4096,
            ),
            device_free_mb=24000.0,
        )
        proc = make_mock_process_info(0, model_name="stable_diffusion", state=HordeProcessState.INFERENCE_PRIMED)
        job = make_job_pop_response("stable_diffusion")
        proc.last_job_referenced = job
        scheduler._process_map = ProcessMap({0: proc})
        scheduler._record_dispatch_reservation(job, proc, baseline=None, staging_only=True)

        assert scheduler.clearance_admit_process(0) is True
        full_mb = scheduler._measured_admission_candidate_delta_mb(job, None, process_id=0, disaggregated=False)
        assert full_mb is not None
        reserved = scheduler._reserve_ledger.effective_planned_vram_mb_for_flow(DISPATCH_ADMISSION_FLOW, {})
        assert reserved == full_mb  # upgraded from the encode charge on the grant

    def test_missing_job_admits_rather_than_wedging(self) -> None:
        """A primed process with no referenced job is admitted rather than held (liveness over pricing)."""
        scheduler = _make_inference_scheduler(
            bridge_data=make_mock_bridge_data(gpu_sampling_lease_enabled=True),
        )
        proc = make_mock_process_info(0, model_name="stable_diffusion", state=HordeProcessState.INFERENCE_PRIMED)
        proc.last_job_referenced = None
        scheduler._process_map = ProcessMap({0: proc})
        assert scheduler.clearance_admit_process(0) is True


class TestSlotDutyClearanceAttribution:
    """A staged-but-uncleared child's empty sampling slot is attributed to CLEARANCE_HOLD under the lease."""

    def test_primed_uncleared_slot_reads_as_clearance_hold(self) -> None:
        """With the lease on and a primed child not yet sampling, the spare sampling slot names CLEARANCE_HOLD."""
        scheduler = _make_inference_scheduler(
            bridge_data=make_mock_bridge_data(gpu_sampling_lease_enabled=True),
            max_concurrent=2,
        )
        proc = make_mock_process_info(0, model_name="stable_diffusion", state=HordeProcessState.INFERENCE_PRIMED)
        job = make_job_pop_response("stable_diffusion")
        proc.last_job_referenced = job
        scheduler._process_map = ProcessMap({0: proc})

        scheduler.record_slot_duty({})
        _totals, _capacity, hold = scheduler.slot_duty_snapshot()
        assert hold == str(SlotDutyBucket.CLEARANCE_HOLD)


class TestTrailingActiveStateIsNotTheNewJobsGrant:
    """A slot's active state from the previous job is not read as the newly bound job sampling unpriced.

    A disaggregated sampler is re-bound inside the previous job's result tick; for a moment it still reports
    that job's INFERENCE_STARTING while ownership already names the next job. Read as an active grant for the
    new job, the reconciler flags an unpriced window and marks the slot sampling, so the PRIMED that follows for
    the real sample stage is never granted and the child idles until its lease-acquire timeout.
    """

    def test_active_state_older_than_the_ownership_is_not_an_active_grant(self) -> None:
        """The active state reported after the ownership was taken is not read as an active grant for the new job."""
        import time

        scheduler = _make_inference_scheduler(
            bridge_data=make_mock_bridge_data(gpu_sampling_lease_enabled=True),
            max_concurrent=1,
        )
        proc = make_mock_process_info(5, model_name="stable_diffusion", state=HordeProcessState.INFERENCE_STARTING)
        proc.last_process_state_started_at = time.time() - 20.0
        job = make_job_pop_response("stable_diffusion")
        proc.record_inference_ownership(job, attempt_ordinal=1)
        scheduler._process_map = ProcessMap({5: proc})

        inputs = scheduler.build_clearance_inputs(device_index=0)
        assert inputs.active_grants == ()

        # Control: the active state reported after the ownership was taken is this job's own sampling.
        proc.last_process_state_started_at = time.time() + 1.0
        inputs = scheduler.build_clearance_inputs(device_index=0)
        assert [grant.process_id for grant in inputs.active_grants] == [5]


class _FakeSemaphore:
    """A counting semaphore mirroring the multiprocessing primitive, with an optional bound for the clearance."""

    def __init__(self, value: int = 0, *, bound: int | None = None) -> None:
        self._value = value
        self._bound = bound

    def acquire(self, block: bool = True, timeout: float | None = None) -> bool:
        if self._value > 0:
            self._value -= 1
            return True
        return False

    def release(self) -> None:
        if self._bound is not None and self._value >= self._bound:
            raise ValueError("released too many times")
        self._value += 1


def _proxy() -> ClearanceLeaseProxy:
    return ClearanceLeaseProxy(clearance=_FakeSemaphore(0, bound=1), done=_FakeSemaphore())


def _controller(*, slot_cap: int = 1, tail: bool = False) -> ClearanceController:
    return ClearanceController(device_index=0, slot_cap=slot_cap, tail_overlap=tail)


def _inputs(*, staged: tuple[ClearanceWaiter, ...] = (), active: tuple[ActiveSampler, ...] = ()) -> ClearanceInputs:
    return ClearanceInputs(
        staged_waiters=staged,
        active_grants=active,
        device_free_mb=20000.0,
        vram_reserve_mb=2048.0,
        paging_active=False,
    )


class TestPoolLivenessUnderUnfittableClearance:
    """A clearance that never fits must degrade to the child's own timeout path, never wedge the pool."""

    def test_never_fitting_admission_holds_but_pool_stays_live(self) -> None:
        """With admission permanently denied, no grant is ever issued yet the child still retires and reopens."""
        controller = _controller()
        proxy = _proxy()
        controller.register(2, proxy)
        staged = (ClearanceWaiter(process_id=2, priority=1),)

        for _ in range(6):
            result = controller.step(_inputs(staged=staged), admit_fn=lambda _pid: False)
            assert result.held_process_ids == (2,)
        assert controller.held_grant_count == 0  # nothing wedged; no phantom grant is held

        # The child times out on its lease and samples unpriced, then signals done; the slot reopens.
        proxy.release()
        controller.step(_inputs(), admit_fn=lambda _pid: False)
        assert controller.grant_state(2) is GrantState.IDLE


class TestHeavyPairFence:
    """A second heavy clearance is held while the first samples, and granted once the first window retires."""

    def test_second_grant_waits_for_the_first_to_complete(self) -> None:
        """At one slot, a second staged child is not cleared while the first holds the grant, then is after."""
        controller = _controller(slot_cap=1)
        first, second = _proxy(), _proxy()
        controller.register(1, first)
        controller.register(2, second)

        # First child is cleared and begins sampling: the single slot is occupied.
        first_staged = (ClearanceWaiter(process_id=1, priority=1),)
        controller.step(_inputs(staged=first_staged), admit_fn=lambda _pid: True)
        assert controller.grant_state(1) is GrantState.CLEARED

        first_sampling = (ActiveSampler(process_id=1, job_id="job-a", progress_fraction=0.3),)
        second_staged = (ClearanceWaiter(process_id=2, priority=2),)
        # Even with the device saying yes, the slot cap holds the second grant while the first samples.
        result = controller.step(
            _inputs(staged=second_staged, active=first_sampling),
            admit_fn=lambda _pid: True,
        )
        assert 2 not in result.cleared_process_ids
        assert controller.grant_state(2) is GrantState.IDLE

        # The first window completes (child signals done); the freed slot admits the second child.
        first.release()
        result = controller.step(_inputs(staged=second_staged), admit_fn=lambda _pid: True)
        assert result.cleared_process_ids == (2,)
        assert controller.grant_state(2) is GrantState.CLEARED


class TestClearanceResidentWeightCredit:
    """Clearance prices a candidate whose weights the target slot already retains at its activation delta.

    The regression: a slot granted retention for a model, taking its next job for the same model, was held
    at clearance for the whole materialisation peak (weights plus activation) although those weights had
    never left the card and were already excluded from the device-free reading. On a card with room for the
    activation alone the streak stalled until the child timed its lease out and sampled unpriced. A
    disaggregated sampler cannot escape it at all: its sample stage reports no model-load transition, so the
    model map never shows its UNet VRAM-resident however long the slot holds it, and the parent's retention
    record is the only truth that says the weights are there.
    """

    _WEIGHTS_MB = 5000.0
    _PEAK_MB = 6000.0

    def _pin_predictors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Price the candidate at a peak dominated by weights, so the credit decides the fit."""
        monkeypatch.setattr(_sched_mod, "predict_job_sampling_vram_mb", lambda _job, _baseline: self._PEAK_MB)
        monkeypatch.setattr(_sched_mod, "predict_job_weight_mb", lambda _job, _baseline: self._WEIGHTS_MB)

    async def _scheduler_with_staged_waiter(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        device_free_mb: float,
        retained_model: str | None,
    ) -> tuple[object, ImageGenerateJobPopResponse]:
        """A slot primed with a same-model job while a sibling samples, the clearance situation under load."""
        self._pin_predictors(monkeypatch)
        job_tracker = JobTracker()
        scheduler = _make_inference_scheduler(
            bridge_data=make_mock_bridge_data(
                gpu_sampling_lease_enabled=True,
                enable_vram_budget=True,
                vram_reserve_mb=2048,
                ram_reserve_mb=4096,
            ),
            job_tracker=job_tracker,
            device_free_mb=device_free_mb,
            max_concurrent=2,
        )
        waiter = make_mock_process_info(0, model_name="stable_diffusion", state=HordeProcessState.INFERENCE_PRIMED)
        waiter.retained_resident_model = retained_model
        sibling = make_mock_process_info(1, model_name=None, state=HordeProcessState.INFERENCE_STARTING)
        scheduler._process_map = ProcessMap({0: waiter, 1: sibling})

        # A sibling job already sampling on the card: the candidate's activation lands beside a live
        # reservation, so its resident weights buy it a credit rather than an unconditional no-op admit.
        sibling_job = make_job_pop_response("stable_diffusion")
        await track_popped_job_async(job_tracker, sibling_job)
        await job_tracker.mark_inference_started(sibling_job)

        job = make_job_pop_response("stable_diffusion")
        await track_popped_job_async(job_tracker, job)
        assert job.id_ is not None
        assert job_tracker.mark_job_aux_prepared_if_ready(job.id_) is True
        waiter.last_job_referenced = job
        scheduler._record_dispatch_reservation(job, waiter, baseline=None, staging_only=True)
        return scheduler, job

    async def test_retained_resident_clears_when_the_activation_delta_fits(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Weights the slot holds are credited, so a card with room for the activation alone clears the job."""
        scheduler, _job = await self._scheduler_with_staged_waiter(
            monkeypatch,
            device_free_mb=4000.0,  # under the full peak, well over the activation delta
            retained_model="stable_diffusion",
        )
        assert scheduler.clearance_admit_process(0) is True

    async def test_the_same_card_holds_a_candidate_whose_weights_are_not_resident(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The contrast: with nothing retained the weights must still land, so the full peak is charged."""
        scheduler, _job = await self._scheduler_with_staged_waiter(
            monkeypatch,
            device_free_mb=4000.0,
            retained_model=None,
        )
        assert scheduler.clearance_admit_process(0) is False

    async def test_holds_when_even_the_activation_delta_does_not_fit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The credit is bounded to the weights: an activation the card cannot seat is still held."""
        scheduler, _job = await self._scheduler_with_staged_waiter(
            monkeypatch,
            device_free_mb=600.0,  # under the activation delta too
            retained_model="stable_diffusion",
        )
        assert scheduler.clearance_admit_process(0) is False


class TestClearanceNetsTheWaiterOwnStagingCharge:
    """A staged child's own encode reservation must not be counted against its own clearance.

    Dispatch under the lease books the job an encode-only staging reservation in the dispatch flow. At
    clearance the same job is priced at its full peak against measured device truth net of every outstanding
    reservation. If that netting leaves the job's own staging entry in the overlay, the waiter is charged
    twice (its staging charge as "outstanding", its full peak as the candidate) and a card with room for the
    peak is read as short by the whole encode charge. Nothing on the card changes while it waits, so the hold
    only ends when the child samples through its lease-acquire timeout: every such job pays the full timeout
    for room the card had all along. The preload flow already nets a request's own planned charge for exactly
    this reason; the dispatch flow must net it too.
    """

    _WEIGHTS_MB = 5000.0
    _PEAK_MB = 6000.0
    _TOTAL_VRAM_MB = 16000

    async def _lone_staged_waiter(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        room_beyond_peak_mb: float,
    ) -> _sched_mod.InferenceScheduler:
        """One staged child, nothing else on the card, device free set relative to the priced peak.

        ``room_beyond_peak_mb`` is the free VRAM left over once the full peak and the noise buffer are seated:
        positive means the peak fits outright, negative means it does not even with nothing else outstanding.
        The staging reservation is booked exactly as dispatch books it.
        """
        monkeypatch.setattr(_sched_mod, "predict_job_sampling_vram_mb", lambda _job, _baseline: self._PEAK_MB)
        monkeypatch.setattr(_sched_mod, "predict_job_weight_mb", lambda _job, _baseline: self._WEIGHTS_MB)
        noise_mb = admission_noise_buffer_mb(float(self._TOTAL_VRAM_MB))
        job_tracker = JobTracker()
        scheduler = _make_inference_scheduler(
            bridge_data=make_mock_bridge_data(
                gpu_sampling_lease_enabled=True,
                enable_vram_budget=True,
                vram_reserve_mb=2048,
                ram_reserve_mb=4096,
            ),
            job_tracker=job_tracker,
            device_free_mb=self._PEAK_MB + noise_mb + room_beyond_peak_mb,
        )
        waiter = make_mock_process_info(0, model_name="stable_diffusion", state=HordeProcessState.INFERENCE_PRIMED)
        waiter.total_vram_mb = self._TOTAL_VRAM_MB
        scheduler._process_map = ProcessMap({0: waiter})

        job = make_job_pop_response("stable_diffusion")
        await track_popped_job_async(job_tracker, job)
        assert job.id_ is not None
        assert job_tracker.mark_job_aux_prepared_if_ready(job.id_) is True
        await job_tracker.mark_inference_started(job)
        waiter.last_job_referenced = job
        scheduler._record_dispatch_reservation(job, waiter, baseline=None, staging_only=True)
        outstanding = scheduler._reserve_ledger.effective_planned_vram_mb_for_flow(DISPATCH_ADMISSION_FLOW, {})
        assert outstanding == _ENCODE_MB, "precondition: the waiter's own encode charge is the only reservation"
        return scheduler

    async def test_a_card_with_room_for_the_peak_clears_the_lone_waiter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Free covers the full peak plus noise with margin to spare, but not a second copy of the encode charge."""
        scheduler = await self._lone_staged_waiter(monkeypatch, room_beyond_peak_mb=_ENCODE_MB / 4)

        assert scheduler.clearance_admit_process(0) is True, (
            "the waiter's own staging reservation was priced against its own clearance"
        )

    async def test_a_card_short_of_the_peak_still_holds_the_lone_waiter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The netting is bounded to the waiter's own entry: a peak the card cannot seat is still held."""
        scheduler = await self._lone_staged_waiter(monkeypatch, room_beyond_peak_mb=-_ENCODE_MB / 4)

        assert scheduler.clearance_admit_process(0) is False

    async def test_the_waiter_staged_allocation_is_credited_against_its_peak(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """What the child has already put on the card (its encoder, its cache) is not charged a second time.

        The device-free reading already lacks the staged allocation and the priced peak includes it, so a card
        short of the gross peak by less than that allocation seats the remainder.
        """
        shortfall_mb = 1500.0
        scheduler = await self._lone_staged_waiter(monkeypatch, room_beyond_peak_mb=-shortfall_mb)
        scheduler._process_map[0].process_reserved_mb = int(shortfall_mb + 100)

        assert scheduler.clearance_admit_process(0) is True, (
            "the waiter's staged allocation was priced against its own clearance"
        )

    async def test_a_hold_demotes_the_safety_weights_in_place_before_giving_up(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A waiter the card cannot seat gets safety's weights moved to host RAM, not a dead 60 s wait.

        The demotion keeps the safety process and its context; it is asked once, and the hold re-asks the
        admission next pass against the room it returned.
        """
        scheduler = await self._lone_staged_waiter(monkeypatch, room_beyond_peak_mb=-1500.0)
        scheduler._runtime_config.bridge_data.safety_on_gpu = True
        scheduler._runtime_config.bridge_data.whole_card_residency_safety_off_gpu = True
        lifecycle = scheduler._process_lifecycle  # a Mock in this harness
        lifecycle.is_safety_gpu_paused = False  # pyrefly: ignore
        lifecycle.safety_placement_transition_pending = False  # pyrefly: ignore
        safety = make_mock_process_info(9, model_name=None, process_type=HordeProcessType.SAFETY)
        send = safety.pipe_connection.send  # a Mock in this harness
        scheduler._process_map[9] = safety

        assert scheduler.clearance_admit_process(0) is False
        sent = [call.args[0].control_flag for call in send.call_args_list]  # pyrefly: ignore
        assert sent == [HordeControlFlag.DEMOTE_SAFETY_WEIGHTS]
        assert scheduler._safety_weights_demoted is True

        send.reset_mock()  # pyrefly: ignore
        scheduler.clearance_admit_process(0)
        assert send.call_args_list == [], "the demotion is asked once, not every tick"  # pyrefly: ignore

    async def test_the_staged_credit_is_bounded_by_what_is_actually_held(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A staged allocation smaller than the shortfall does not admit: the remainder must still fit."""
        shortfall_mb = 1500.0
        scheduler = await self._lone_staged_waiter(monkeypatch, room_beyond_peak_mb=-shortfall_mb)
        scheduler._process_map[0].process_reserved_mb = int(shortfall_mb - 100)

        assert scheduler.clearance_admit_process(0) is False
