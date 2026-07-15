"""Regressions for the per-process GPU denoise clearance lease: staging liveness, head protection, and holds.

These pin the throughput unlock the clearance lease exists for (a spare process stages the next job ahead
while another samples, instead of idling), the head-protection invariant (siblings staging under the
encode-only charge never starve the head's full-materialisation room), the ledger charge upgrade at
clearance (encode charge upgraded in place, not double-booked), the slot-duty attribution of a held
clearance, and the pool liveness guarantees (a never-fitting admission degrades rather than wedging; a full
slot cap holds the next grant until a window retires).
"""

from __future__ import annotations

from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessState
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
            bridge_data=make_mock_bridge_data(gpu_sampling_lease_enabled=True),
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
