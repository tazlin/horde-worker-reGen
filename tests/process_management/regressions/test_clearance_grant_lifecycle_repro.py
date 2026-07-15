"""Regressions for the clearance-lease grant lifecycle failure that wedged a live worker.

The live sequence: a child was cleared for a fresh job, then a stale ``done`` permit from its previous job
retired that fresh grant, leaving the child sampling with no recorded grant and its slot permanently occupied
by a phantom grant that could never retire. With one slot that stuck grant held the only slot, so the sibling
child sat in ``INFERENCE_PRIMED`` waiting for a clearance that never came until the hung-process watchdog
killed it at the step-timeout. These pin the fix: grant retirement is job-correlated from the process
snapshot (never from counting ``done`` permits), the slot reopens when a child finishes its job, and a child
the controller is deliberately holding is not killed as a hung inference.
"""

from __future__ import annotations

import time

from horde_worker_regen.process_management.lifecycle.horde_process import (
    HordeProcessState,
    HordeProcessType,
)
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.scheduling.clearance_lease import (
    CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS,
    ActiveSampler,
    ClearanceController,
    ClearanceInputs,
    ClearanceLeaseProxy,
    ClearanceWaiter,
    GrantState,
)
from tests.process_management.conftest import make_mock_bridge_data, make_mock_process_info
from tests.process_management.lifecycle.test_process_lifecycle import _make_plm


class _FakeSemaphore:
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


class _StubProxy(ClearanceLeaseProxy):
    def __init__(self) -> None:
        super().__init__(clearance=_FakeSemaphore(0, bound=1), done=_FakeSemaphore())

    def child_signal_done(self) -> None:
        self._done.release()  # type: ignore[attr-defined]


def _controller(*, slot_cap: int = 1) -> ClearanceController:
    return ClearanceController(device_index=0, slot_cap=slot_cap, tail_overlap=False)


def _inputs(
    *,
    staged: tuple[ClearanceWaiter, ...] = (),
    active: tuple[ActiveSampler, ...] = (),
) -> ClearanceInputs:
    return ClearanceInputs(
        staged_waiters=staged,
        active_grants=active,
        device_free_mb=20000.0,
        vram_reserve_mb=2048.0,
        paging_active=False,
    )


def _always(_pid: int) -> bool:
    return True


class TestStaleDonePermitDoesNotRetireFreshGrant:
    """A done permit outliving the job that posted it must never retire a later grant (the live wedge)."""

    def test_mid_job_done_permit_does_not_retire_the_active_grant(self) -> None:
        """A multi-sample job posts a done permit between passes; the grant it holds must survive."""
        controller = _controller()
        proxy = _StubProxy()
        controller.register(3, proxy)

        # Job A: cleared, then sampling.
        controller.step(_inputs(staged=(ClearanceWaiter(process_id=3, priority=1, job_id="job-a"),)), admit_fn=_always)
        controller.step(
            _inputs(active=(ActiveSampler(process_id=3, job_id="job-a", progress_fraction=0.4),)), admit_fn=_always
        )
        assert controller.grant_state(3) is GrantState.SAMPLING

        # Multi-sample job A releases a done permit after its first pass while STILL sampling (hires second pass).
        proxy.child_signal_done()
        controller.step(
            _inputs(active=(ActiveSampler(process_id=3, job_id="job-a", progress_fraction=0.6),)),
            admit_fn=_always,
        )
        # The stale/mid-job done permit must not have retired the live grant.
        assert controller.grant_state(3) is GrantState.SAMPLING
        assert controller.held_grant_count == 1

    def test_leftover_done_from_prior_job_does_not_retire_a_new_jobs_grant(self) -> None:
        """The exact live sequence: cleared for job B with a leftover done from job A pending; grant survives."""
        controller = _controller()
        proxy = _StubProxy()
        controller.register(3, proxy)

        # Job A ran and left two done permits behind (a two-sample job), and the process has since moved on.
        proxy.child_signal_done()
        proxy.child_signal_done()

        # Job B is dispatched to process 3 and the controller clears it into its window.
        result = controller.step(
            _inputs(staged=(ClearanceWaiter(process_id=3, priority=1, job_id="job-b"),)),
            admit_fn=_always,
        )
        assert result.cleared_process_ids == (3,)
        assert controller.grant_state(3) is GrantState.CLEARED

        # A subsequent tick (the leftover done permits now drained-and-discarded) must not retire job B's grant
        # while process 3 is still primed on job B.
        controller.step(
            _inputs(staged=(ClearanceWaiter(process_id=3, priority=1, job_id="job-b"),)),
            admit_fn=_always,
        )
        assert controller.grant_state(3) is GrantState.CLEARED
        assert controller.held_grant_count == 1


class TestSlotReopensAndSiblingClears:
    """A stuck grant must never wedge the sibling: the slot reopens when the first child finishes its job."""

    def test_second_child_clears_once_first_completes(self) -> None:
        """One slot: child 3 samples, child 4 waits, then 3 completes and 4 is cleared (the anti-wedge)."""
        controller = _controller(slot_cap=1)
        controller.register(3, _StubProxy())
        controller.register(4, _StubProxy())

        # Child 3 is cleared and sampling; child 4 is primed and waiting for the single slot.
        controller.step(_inputs(staged=(ClearanceWaiter(process_id=3, priority=1, job_id="job-a"),)), admit_fn=_always)
        both = _inputs(
            staged=(ClearanceWaiter(process_id=4, priority=2, job_id="job-b"),),
            active=(ActiveSampler(process_id=3, job_id="job-a", progress_fraction=0.5),),
        )
        result = controller.step(both, admit_fn=_always)
        assert 4 not in result.cleared_process_ids  # slot full while 3 samples
        assert controller.grant_state(4) is GrantState.IDLE

        # Child 3 finishes job A and leaves the sampling set; the slot reopens and child 4 is cleared.
        result = controller.step(
            _inputs(staged=(ClearanceWaiter(process_id=4, priority=2, job_id="job-b"),)),
            admit_fn=_always,
        )
        assert controller.grant_state(3) is GrantState.IDLE  # retired by job correlation, not a done permit
        assert result.cleared_process_ids == (4,)
        assert controller.grant_state(4) is GrantState.CLEARED


class TestBootShapedSteadyClearPath:
    """A boot-shaped run: two fresh children cycle jobs through the single slot with no phantom accumulation."""

    def test_two_children_alternate_through_one_slot(self) -> None:
        """Clear 3, it samples and completes, clear 4, it samples and completes: no grant is ever stuck held."""
        controller = _controller(slot_cap=1)
        controller.register(3, _StubProxy())
        controller.register(4, _StubProxy())

        # Both primed at boot; only the head-priority child is cleared.
        both_primed = _inputs(
            staged=(
                ClearanceWaiter(process_id=3, priority=1, job_id="job-a"),
                ClearanceWaiter(process_id=4, priority=2, job_id="job-b"),
            ),
        )
        result = controller.step(both_primed, admit_fn=_always)
        assert result.cleared_process_ids == (3,)

        # Child 3 samples to completion.
        controller.step(
            _inputs(
                staged=(ClearanceWaiter(process_id=4, priority=2, job_id="job-b"),),
                active=(ActiveSampler(process_id=3, job_id="job-a", progress_fraction=0.9),),
            ),
            admit_fn=_always,
        )
        # Child 3 completes and leaves; child 4 is cleared into the freed slot.
        result = controller.step(
            _inputs(staged=(ClearanceWaiter(process_id=4, priority=2, job_id="job-b"),)),
            admit_fn=_always,
        )
        assert result.cleared_process_ids == (4,)
        assert controller.grant_state(3) is GrantState.IDLE

        # Child 4 samples and completes; the pool returns to no held grants (no CLEARANCE_HOLD accumulation).
        controller.step(
            _inputs(active=(ActiveSampler(process_id=4, job_id="job-b", progress_fraction=0.9),)),
            admit_fn=_always,
        )
        controller.step(_inputs(), admit_fn=_always)
        assert controller.held_grant_count == 0


class TestWatchdogDoesNotKillClearanceHeldChild:
    """A child the controller is deliberately holding (primed, no step yet) is not killed as hung inference."""

    def _primed_child(self, *, silent_for: float) -> object:
        proc = make_mock_process_info(
            4,
            model_name="stable_diffusion",
            state=HordeProcessState.INFERENCE_PRIMED,
            process_type=HordeProcessType.INFERENCE,
        )
        proc.last_current_step = None  # no sampling step yet: still waiting for clearance
        proc.last_heartbeat_timestamp = time.time() - silent_for
        return proc

    def test_first_step_grace_is_extended_by_the_acquire_timeout_under_the_lease(self) -> None:
        """A primed, not-yet-sampling child under the lease widens its first-step grace by the acquire timeout."""
        plm = _make_plm()
        bridge = make_mock_bridge_data(gpu_sampling_lease_enabled=True)
        bridge.inference_first_step_timeout = 120
        proc = self._primed_child(silent_for=0.0)
        widened = plm._effective_first_step_timeout(bridge, proc, 120)
        assert widened == int(120 + CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS)

    def test_no_lease_first_step_grace_is_unchanged(self) -> None:
        """Without the lease the first-step grace is byte-identical to the configured value."""
        plm = _make_plm()
        bridge = make_mock_bridge_data(gpu_sampling_lease_enabled=False)
        proc = self._primed_child(silent_for=0.0)
        assert plm._effective_first_step_timeout(bridge, proc, 120) == 120

    def test_held_child_within_grace_is_not_flagged_stuck(self) -> None:
        """A clearance-held child silent within the widened grace is not reaped as stuck mid inference."""
        proc = self._primed_child(silent_for=140.0)  # past 120s first-step, within 120+60 widened grace
        process_map = ProcessMap({4: proc})
        widened = int(120 + CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS)
        assert process_map.is_stuck_on_inference(4, 60, widened) is False

    def test_held_child_past_widened_grace_is_still_reaped(self) -> None:
        """Past the bounded widened grace a genuinely wedged primed child is still reaped (liveness preserved)."""
        proc = self._primed_child(silent_for=200.0)  # past 120+60
        process_map = ProcessMap({4: proc})
        widened = int(120 + CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS)
        assert process_map.is_stuck_on_inference(4, 60, widened) is True
