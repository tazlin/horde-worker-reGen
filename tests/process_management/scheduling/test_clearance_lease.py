"""Unit tests for the per-process GPU denoise clearance lease: proxy protocol and controller truth table."""

from __future__ import annotations

from collections.abc import Callable

from hordelib.metrics import JobPhaseMetrics, ModelLoadEvent

from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessState
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.scheduling.clearance_lease import (
    ActiveSampler,
    ClearanceController,
    ClearanceInputs,
    ClearanceLeaseProxy,
    ClearancePlan,
    ClearanceWaiter,
    GrantState,
    TailOverlapDenialReason,
    decide_clearances,
    format_tail_overlap_tally,
)
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import make_mock_process_info, make_testable_process_manager

_MARGIN_MB = 3072.0
_PAD_SECONDS = 1.0
_LOAD_SECONDS = 4.0
"""The measured incoming weight-upload cost the tests size the handoff window from: with the pad, a sampler
with 5.0s or less left is inside the window and one with more is outside it."""


class _FakeSemaphore:
    """A counting semaphore mirroring the multiprocessing primitive's ``acquire(block, timeout)``/``release``.

    An optional ``bound`` reproduces the bounded clearance semaphore: a release past the bound raises
    ``ValueError``, exactly as the production ``BoundedSemaphore`` does, so the controller's double-clear
    absorption is exercised as it runs in the worker.
    """

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

    @property
    def value(self) -> int:
        return self._value


def _held_empty_clearance() -> _FakeSemaphore:
    """A bounded clearance semaphore the parent has already emptied (its single permit acquired)."""
    return _FakeSemaphore(0, bound=1)


class TestProxyProtocol:
    """The child-side proxy grants once per job, passes later samples through, and signals completion."""

    def test_acquire_consumes_grant_once_and_passes_through_same_job(self) -> None:
        """A granted first sample acquires; the second sample of the same job passes through without blocking."""
        clearance = _FakeSemaphore(1, bound=1)  # the parent has granted (permit present)
        proxy = ClearanceLeaseProxy(clearance=clearance, done=_FakeSemaphore())
        proxy.begin_job()

        assert proxy.acquire(True, 5.0) is True
        assert clearance.value == 0  # the grant was consumed
        # Second sample of the same job: pass-through, no further permit needed.
        assert proxy.acquire(True, 5.0) is True
        assert clearance.value == 0

    def test_begin_job_resets_so_next_job_waits_for_its_own_grant(self) -> None:
        """After a job consumes its grant, a fresh job blocks again until the parent clears it."""
        clearance = _FakeSemaphore(1, bound=1)
        proxy = ClearanceLeaseProxy(clearance=clearance, done=_FakeSemaphore())
        proxy.begin_job()
        assert proxy.acquire(True, 5.0) is True  # first job consumes the grant

        proxy.begin_job()  # next job
        # No permit available now, so the next job's first acquire cannot pass through.
        assert proxy.acquire(True, 0.0) is False

    def test_timeout_returns_false_but_still_consumes_grant(self) -> None:
        """A timed-out acquire degrades to unpriced sampling; later samples of that job must not block again."""
        proxy = ClearanceLeaseProxy(clearance=_held_empty_clearance(), done=_FakeSemaphore())
        proxy.begin_job()
        assert proxy.acquire(True, 0.0) is False  # no grant: times out
        # The job now samples unpriced; its second sample passes through rather than paying the timeout again.
        assert proxy.acquire(True, 0.0) is True

    def test_release_signals_done_without_touching_clearance(self) -> None:
        """Release signals the parent through ``done`` and never returns a clearance permit to the child."""
        clearance = _held_empty_clearance()
        done = _FakeSemaphore()
        proxy = ClearanceLeaseProxy(clearance=clearance, done=done)
        proxy.release()
        assert done.value == 1
        assert clearance.value == 0  # release never grants a clearance permit


def _inputs(
    *,
    staged: tuple[ClearanceWaiter, ...] = (),
    active: tuple[ActiveSampler, ...] = (),
    device_free_mb: float | None = 20000.0,
    reserve_mb: float = 2048.0,
    paging: bool = False,
    load_seconds: float | None = _LOAD_SECONDS,
) -> ClearanceInputs:
    return ClearanceInputs(
        staged_waiters=staged,
        active_grants=active,
        device_free_mb=device_free_mb,
        vram_reserve_mb=reserve_mb,
        paging_active=paging,
        incoming_load_seconds=load_seconds,
    )


def _tailing(
    *,
    job_id: str = "job-a",
    remaining: float | None = 2.0,
    progress: float = 0.9,
    process_id: int = 1,
) -> ActiveSampler:
    """An active sampler whose estimated remaining time places it inside the handoff window by default."""
    return ActiveSampler(
        process_id=process_id,
        job_id=job_id,
        progress_fraction=progress,
        remaining_sampling_seconds=remaining,
    )


def _decide(
    inputs: ClearanceInputs,
    *,
    slot_cap: int = 1,
    held: int | None = None,
    tail: bool = True,
    cleared: frozenset[str] = frozenset(),
) -> ClearancePlan:
    # The controller's authoritative held count is the number of active grants unless overridden (a cleared
    # child not yet sampling would raise it above len(active_grants), which some tests exercise explicitly).
    held_grant_count = len(inputs.active_grants) if held is None else held
    return decide_clearances(
        inputs,
        slot_cap=slot_cap,
        held_grant_count=held_grant_count,
        tail_overlap_enabled=tail,
        tail_overlap_pad_seconds=_PAD_SECONDS,
        tail_overlap_margin_mb=_MARGIN_MB,
        already_tail_cleared_job_ids=cleared,
    )


class TestDecideClearances:
    """The pure decision respects the slot cap, head-of-queue order, and the one-per-outgoing tail bonus."""

    def test_free_slot_clears_best_staged_waiter(self) -> None:
        """With a free steady slot and queued waiters, the head-priority waiter is chosen."""
        staged = (ClearanceWaiter(process_id=3, priority=2), ClearanceWaiter(process_id=2, priority=1))
        plan = _decide(_inputs(staged=staged), slot_cap=1)
        assert plan.clear_process_ids == (2,)

    def test_slot_cap_limits_concurrent_grants(self) -> None:
        """A full slot cap (all slots granted) clears no one until a grant retires."""
        active = (ActiveSampler(process_id=1, job_id="job-a", progress_fraction=0.2),)
        staged = (ClearanceWaiter(process_id=2, priority=1),)
        plan = _decide(_inputs(staged=staged, active=active), slot_cap=1, tail=False)
        assert plan.clear_process_ids == ()

    def test_higher_cap_fills_multiple_slots_in_priority_order(self) -> None:
        """A cap above one fills several slots, still head-first."""
        staged = (
            ClearanceWaiter(process_id=5, priority=3),
            ClearanceWaiter(process_id=4, priority=1),
            ClearanceWaiter(process_id=6, priority=2),
        )
        plan = _decide(_inputs(staged=staged), slot_cap=2, tail=False)
        assert plan.clear_process_ids == (4, 6)

    def test_tail_overlap_grants_one_extra_when_outgoing_is_tailing(self) -> None:
        """A full cap plus a tailing sampler and room clears exactly one extra waiter, bound to the outgoing job."""
        staged = (ClearanceWaiter(process_id=2, priority=1),)
        plan = _decide(_inputs(staged=staged, active=(_tailing(),)), slot_cap=1, tail=True)
        assert plan.clear_process_ids == (2,)
        assert plan.tail_cleared_for_job_id == "job-a"
        assert plan.tail_denial is None

    def test_tail_overlap_suppressed_for_already_cleared_outgoing_job(self) -> None:
        """A tail bonus fires once per outgoing sampler: a second tick for the same job clears no extra."""
        staged = (ClearanceWaiter(process_id=2, priority=1),)
        plan = _decide(
            _inputs(staged=staged, active=(_tailing(),)),
            slot_cap=1,
            tail=True,
            cleared=frozenset({"job-a"}),
        )
        assert plan.clear_process_ids == ()
        assert plan.tail_denial is not None
        assert plan.tail_denial.reason is TailOverlapDenialReason.ALREADY_GRANTED

    def test_tail_overlap_disabled_never_grants_extra(self) -> None:
        """With tail overlap off, a full cap admits no handoff grant however advanced the sampler."""
        staged = (ClearanceWaiter(process_id=2, priority=1),)
        plan = _decide(_inputs(staged=staged, active=(_tailing(remaining=0.1),)), slot_cap=1, tail=False)
        assert plan.clear_process_ids == ()
        assert plan.tail_denial is None  # the gate never applied, so nothing refused it

    def test_tail_overlap_held_under_paging(self) -> None:
        """Under WDDM paging the measured free is untrustworthy, so no early clear."""
        staged = (ClearanceWaiter(process_id=2, priority=1),)
        plan = _decide(_inputs(staged=staged, active=(_tailing(),), paging=True), slot_cap=1, tail=True)
        assert plan.clear_process_ids == ()
        assert plan.tail_denial is not None
        assert plan.tail_denial.reason is TailOverlapDenialReason.PAGING_ACTIVE

    def test_tail_overlap_held_below_margin(self) -> None:
        """Measured free net of the reserve below the margin withholds the handoff grant."""
        staged = (ClearanceWaiter(process_id=2, priority=1),)
        # reserve 2048 + margin 3072 = 5120 required; 5000 is short.
        plan = _decide(
            _inputs(staged=staged, active=(_tailing(),), device_free_mb=5000.0),
            slot_cap=1,
            tail=True,
        )
        assert plan.clear_process_ids == ()
        assert plan.tail_denial is not None
        assert plan.tail_denial.reason is TailOverlapDenialReason.HEADROOM_SHORT
        assert plan.tail_denial.headroom_shortfall_mb == 120.0

    def test_tail_overlap_denied_when_no_waiter_is_staged(self) -> None:
        """With nobody staged there is no handoff to make; the denial names the missing waiter."""
        plan = _decide(_inputs(active=(_tailing(),)), slot_cap=1, tail=True)
        assert plan.clear_process_ids == ()
        assert plan.tail_denial is not None
        assert plan.tail_denial.reason is TailOverlapDenialReason.NO_STAGED_WAITER

    def test_tail_overlap_denied_when_the_bonus_slot_is_unused(self) -> None:
        """A steady slot already covers the only staged waiter, so the opened handoff slot goes unused."""
        staged = (ClearanceWaiter(process_id=2, priority=1),)
        plan = _decide(_inputs(staged=staged, active=(_tailing(),)), slot_cap=2, tail=True)
        assert plan.clear_process_ids == (2,)
        assert plan.tail_cleared_for_job_id is None
        assert plan.tail_denial is not None
        assert plan.tail_denial.reason is TailOverlapDenialReason.BONUS_SLOT_UNUSED


class TestTailOverlapTimeTrigger:
    """The handoff fires on the outgoing sampler's remaining *time*, not on a fraction of its step count."""

    def test_fires_as_soon_as_remaining_time_fits_the_load_estimate(self) -> None:
        """A short-remaining sampler opens the window even at a modest progress fraction (a fast job)."""
        staged = (ClearanceWaiter(process_id=2, priority=1),)
        # A 6-second job barely past halfway has under 5s left, so the incoming load fits inside its tail.
        active = (_tailing(remaining=4.9, progress=0.55),)
        plan = _decide(_inputs(staged=staged, active=active), slot_cap=1, tail=True)
        assert plan.clear_process_ids == (2,)
        assert plan.tail_cleared_for_job_id == "job-a"

    def test_slow_job_is_denied_until_its_tail(self) -> None:
        """A deep-but-slow sampler with more time left than the window is refused, and says by how much."""
        staged = (ClearanceWaiter(process_id=2, priority=1),)
        active = (_tailing(remaining=20.0, progress=0.85),)
        plan = _decide(_inputs(staged=staged, active=active), slot_cap=1, tail=True)
        assert plan.clear_process_ids == ()
        assert plan.tail_denial is not None
        assert plan.tail_denial.reason is TailOverlapDenialReason.REMAINING_ABOVE_WINDOW
        assert plan.tail_denial.remaining_seconds == 20.0
        assert plan.tail_denial.load_estimate_seconds == _LOAD_SECONDS

    def test_unknown_rate_never_fires_and_is_reported(self) -> None:
        """A sampler too early for a trusted rate is refused as progress-unknown however deep it reads."""
        staged = (ClearanceWaiter(process_id=2, priority=1),)
        active = (_tailing(remaining=None, progress=0.95),)
        plan = _decide(_inputs(staged=staged, active=active), slot_cap=1, tail=True)
        assert plan.clear_process_ids == ()
        assert plan.tail_denial is not None
        assert plan.tail_denial.reason is TailOverlapDenialReason.PROGRESS_UNKNOWN
        assert plan.tail_denial.progress_fraction == 0.95

    def test_window_falls_back_to_the_default_load_estimate(self) -> None:
        """With no measured upload cost the window is sized from the conservative module default."""
        staged = (ClearanceWaiter(process_id=2, priority=1),)
        # The default estimate is longer than the measured one used elsewhere here, so 6.5s now fits.
        active = (_tailing(remaining=6.5),)
        plan = _decide(_inputs(staged=staged, active=active, load_seconds=None), slot_cap=1, tail=True)
        assert plan.clear_process_ids == (2,)
        assert plan.tail_denial is None

    def test_soonest_finishing_sampler_owns_the_handoff(self) -> None:
        """With several samplers in flight the window is sized against the one about to free a slot."""
        staged = (ClearanceWaiter(process_id=2, priority=1),)
        active = (
            _tailing(job_id="job-slow", remaining=30.0, progress=0.95, process_id=1),
            _tailing(job_id="job-fast", remaining=1.0, progress=0.4, process_id=3),
        )
        plan = _decide(_inputs(staged=staged, active=active), slot_cap=2, held=2, tail=True)
        assert plan.tail_cleared_for_job_id == "job-fast"


class _StubProxy(ClearanceLeaseProxy):
    """A proxy backed by fake semaphores, so the controller drives real grant/done edges under test."""

    def __init__(self) -> None:
        super().__init__(clearance=_held_empty_clearance(), done=_FakeSemaphore())

    @property
    def clearance_value(self) -> int:
        return self._clearance.value  # type: ignore[attr-defined]

    def child_signal_done(self) -> None:
        self._done.release()  # type: ignore[attr-defined]


def _controller(
    *,
    slot_cap: int = 1,
    tail: bool = True,
    clock: Callable[[], float] | None = None,
) -> ClearanceController:
    return ClearanceController(
        device_index=0,
        slot_cap=slot_cap,
        tail_overlap=tail,
        tail_overlap_pad_seconds=_PAD_SECONDS,
        tail_overlap_margin_mb=_MARGIN_MB,
        clock=clock,
    )


def _always_admit(_process_id: int) -> bool:
    return True


def _never_admit(_process_id: int) -> bool:
    return False


class TestControllerClearing:
    """The controller grants clearance at the semaphore edge, guarding double clears and honouring admission."""

    def test_clear_releases_permit_and_marks_cleared(self) -> None:
        """A chosen, admitted waiter has its clearance permit released and its state advanced to CLEARED."""
        controller = _controller()
        proxy = _StubProxy()
        controller.register(2, proxy)
        staged = (ClearanceWaiter(process_id=2, priority=1),)

        result = controller.step(_inputs(staged=staged), admit_fn=_always_admit)

        assert result.cleared_process_ids == (2,)
        assert controller.grant_state(2) is GrantState.CLEARED
        assert proxy.clearance_value == 1  # the child can now acquire its window
        assert controller.held_grant_count == 1

    def test_admission_denied_holds_the_waiter(self) -> None:
        """A waiter whose full-price admission does not fit is held, not cleared: its slot is a reported hold."""
        controller = _controller()
        proxy = _StubProxy()
        controller.register(2, proxy)
        staged = (ClearanceWaiter(process_id=2, priority=1),)

        result = controller.step(_inputs(staged=staged), admit_fn=_never_admit)

        assert result.cleared_process_ids == ()
        assert result.held_process_ids == (2,)
        assert controller.grant_state(2) is GrantState.IDLE
        assert proxy.clearance_value == 0  # no grant issued

    def test_already_cleared_waiter_is_not_cleared_again(self) -> None:
        """A child already holding a grant is never double-cleared, even if it reappears as a staged waiter."""
        controller = _controller()
        proxy = _StubProxy()
        controller.register(2, proxy)
        staged = (ClearanceWaiter(process_id=2, priority=1),)
        controller.step(_inputs(staged=staged), admit_fn=_always_admit)  # first clear
        assert proxy.clearance_value == 1

        # A second tick with the same waiter still listed must not release a second permit.
        controller.step(_inputs(staged=staged), admit_fn=_always_admit)
        assert proxy.clearance_value == 1

    def test_sampling_onset_advances_cleared_to_sampling(self) -> None:
        """Once the cleared child reports denoise progress it is accounted as sampling (still one held slot)."""
        controller = _controller()
        proxy = _StubProxy()
        controller.register(2, proxy)
        staged = (ClearanceWaiter(process_id=2, priority=1),)
        controller.step(_inputs(staged=staged), admit_fn=_always_admit)

        active = (ActiveSampler(process_id=2, job_id="job-x", progress_fraction=0.1),)
        controller.step(_inputs(active=active), admit_fn=_always_admit)
        assert controller.grant_state(2) is GrantState.SAMPLING
        assert controller.held_grant_count == 1

    def test_grant_retires_when_the_child_leaves_the_snapshot(self) -> None:
        """A child that finishes its job and leaves the staged/sampling snapshot frees its slot to idle.

        Retirement is job-correlated from the process snapshot, not from counting done permits: a done permit
        posted here is drained-and-discarded and never drives retirement.
        """
        controller = _controller()
        proxy = _StubProxy()
        controller.register(2, proxy)
        controller.step(
            _inputs(staged=(ClearanceWaiter(process_id=2, priority=1, job_id="job-x"),)),
            admit_fn=_always_admit,
        )
        assert controller.grant_state(2) is GrantState.CLEARED

        proxy.child_signal_done()  # a leftover done permit must not be what retires the grant
        controller.step(_inputs(), admit_fn=_always_admit)  # process 2 has left the snapshot
        assert controller.grant_state(2) is GrantState.IDLE
        assert controller.held_grant_count == 0

    def test_note_child_replaced_discards_state(self) -> None:
        """A replaced child's grant state is dropped so its dead slot never holds a phantom grant."""
        controller = _controller()
        controller.register(2, _StubProxy())
        controller.step(_inputs(staged=(ClearanceWaiter(process_id=2, priority=1),)), admit_fn=_always_admit)
        assert controller.held_grant_count == 1

        controller.note_child_replaced(2)
        assert controller.grant_state(2) is GrantState.IDLE
        assert controller.held_grant_count == 0

    def test_unpriced_sampling_warning_is_edge_triggered(self) -> None:
        """A child sampling with no recorded grant (the timeout path) warns once, not every tick."""
        from loguru import logger

        messages: list[str] = []
        sink_id = logger.add(lambda record: messages.append(record), level="WARNING")
        try:
            controller = _controller()
            controller.register(2, _StubProxy())
            active = (ActiveSampler(process_id=2, job_id="job-y", progress_fraction=0.3),)
            controller.step(_inputs(active=active), admit_fn=_always_admit)
            controller.step(_inputs(active=active), admit_fn=_always_admit)
        finally:
            logger.remove(sink_id)

        unpriced = [m for m in messages if "unpriced sampling" in m]
        assert len(unpriced) == 1
        assert controller.grant_state(2) is GrantState.SAMPLING


class TestTailOverlapObservability:
    """A tail-overlap early clear emits its own INFO signal exactly once per outgoing sampler."""

    def _drive_tail_bonus(self, controller: ClearanceController) -> None:
        """Clear an outgoing sampler, advance it into its tail, and clear one extra waiter on the bonus slot."""
        outgoing = _StubProxy()
        incoming = _StubProxy()
        controller.register(1, outgoing)
        controller.register(2, incoming)
        # Clear the outgoing sampler and advance it to a tailing denoise progress the bonus keys on.
        controller.step(
            _inputs(staged=(ClearanceWaiter(process_id=1, priority=0, job_id="job-a"),)),
            admit_fn=_always_admit,
        )
        # The outgoing sampler holds the only steady slot, so the incoming waiter can only clear on the bonus.
        controller.step(
            _inputs(staged=(ClearanceWaiter(process_id=2, priority=1),), active=(_tailing(),)),
            admit_fn=_always_admit,
        )

    def test_tail_bonus_clear_emits_dedicated_info_line_once(self) -> None:
        """The bonus clear logs one dedicated INFO line carrying the outgoing token, progress, and headroom."""
        from loguru import logger

        messages: list[str] = []
        sink_id = logger.add(lambda record: messages.append(record), level="INFO")
        try:
            controller = _controller(slot_cap=1, tail=True)
            self._drive_tail_bonus(controller)
            # A further tick for the same outgoing sampler must not re-emit (one-per-job dedup).
            controller.step(
                _inputs(staged=(ClearanceWaiter(process_id=3, priority=2),), active=(_tailing(),)),
                admit_fn=_always_admit,
            )
        finally:
            logger.remove(sink_id)

        tail_lines = [m for m in messages if "tail-overlap early clear" in m]
        assert len(tail_lines) == 1
        line = tail_lines[0]
        assert "job-a" in line  # the outgoing sampler's job id token
        assert "0.90" in line  # the outgoing sampler's denoise progress fraction
        assert "17952MB" in line  # measured headroom: 20000 device-free minus the 2048 reserve

    def test_grant_and_denial_tallies_accumulate(self) -> None:
        """The session tally counts the grant and attributes each denied tick to the clause that refused."""
        controller = _controller(slot_cap=1, tail=True)
        self._drive_tail_bonus(controller)
        assert controller.tail_overlap_grant_count == 1

        # The same outgoing sampler is now deduped, and a paging tick is refused earlier in the order.
        controller.step(
            _inputs(staged=(ClearanceWaiter(process_id=3, priority=2),), active=(_tailing(),)),
            admit_fn=_always_admit,
        )
        controller.step(
            _inputs(staged=(ClearanceWaiter(process_id=3, priority=2),), active=(_tailing(),), paging=True),
            admit_fn=_always_admit,
        )
        counts = controller.tail_overlap_denial_counts
        assert counts[TailOverlapDenialReason.ALREADY_GRANTED] == 1
        assert counts[TailOverlapDenialReason.PAGING_ACTIVE] == 1

    def test_denial_line_is_edge_triggered_on_the_reason(self) -> None:
        """An unchanged denial reason logs once; a different clause logs again with the suppressed count."""
        from loguru import logger

        messages: list[str] = []
        sink_id = logger.add(lambda record: messages.append(record), level="DEBUG")
        clock_seconds = 100.0
        try:
            controller = _controller(slot_cap=1, tail=True, clock=lambda: clock_seconds)
            controller.register(1, _StubProxy())
            controller.step(
                _inputs(staged=(ClearanceWaiter(process_id=1, priority=0, job_id="job-a"),)),
                admit_fn=_always_admit,
            )
            slow = _inputs(staged=(ClearanceWaiter(process_id=2, priority=1),), active=(_tailing(remaining=30.0),))
            for _ in range(3):
                controller.step(slow, admit_fn=_always_admit)
            controller.step(
                _inputs(staged=(ClearanceWaiter(process_id=2, priority=1),), active=(_tailing(),), paging=True),
                admit_fn=_always_admit,
            )
        finally:
            logger.remove(sink_id)

        denial_lines = [m for m in messages if "tail-overlap handoff denied" in m]
        assert len(denial_lines) == 2
        assert "remaining" in denial_lines[0]
        assert "30.0s" in denial_lines[0]
        assert "paging" in denial_lines[1]
        assert "suppressed 2 unchanged repeats" in denial_lines[1]


class TestTailOverlapTallyFormatting:
    """The compact tally the duty-cycle line carries."""

    def test_tally_ranks_denial_shares(self) -> None:
        """Grants, denials, and the largest denial shares read straight off the line."""
        line = format_tail_overlap_tally(
            4,
            {
                TailOverlapDenialReason.REMAINING_ABOVE_WINDOW: 72,
                TailOverlapDenialReason.NO_STAGED_WAITER: 35,
                TailOverlapDenialReason.HEADROOM_SHORT: 11,
                TailOverlapDenialReason.PAGING_ACTIVE: 0,
            },
        )
        assert line == "tail-overlap: 4 granted / 118 denied (remaining 61%, no-waiter 30%, headroom 9%)"

    def test_unused_slots_trail_the_line_without_diluting_the_shares(self) -> None:
        """Ticks where the steady slots covered the queue are reported apart from the refusing clauses."""
        line = format_tail_overlap_tally(
            4,
            {
                TailOverlapDenialReason.REMAINING_ABOVE_WINDOW: 72,
                TailOverlapDenialReason.NO_STAGED_WAITER: 35,
                TailOverlapDenialReason.HEADROOM_SHORT: 11,
                TailOverlapDenialReason.BONUS_SLOT_UNUSED: 42,
            },
        )
        assert line == (
            "tail-overlap: 4 granted / 118 denied (remaining 61%, no-waiter 30%, headroom 9%) [slot-unused 42]"
        )

    def test_only_unused_slots_still_reports(self) -> None:
        """A worker that never had to lean on the handoff says so rather than going silent."""
        assert format_tail_overlap_tally(0, {TailOverlapDenialReason.BONUS_SLOT_UNUSED: 9}) == (
            "tail-overlap: 0 granted / 0 denied [slot-unused 9]"
        )

    def test_tally_is_absent_when_the_gate_never_applied(self) -> None:
        """Nothing granted and nothing denied means the handoff never applied, so no tally is composed."""
        assert format_tail_overlap_tally(0, {}) is None


class TestControllerLiveness:
    """A controller that never clears must not wedge the pool: children still sample via the timeout path."""

    def test_pool_stays_live_when_admission_never_fits(self) -> None:
        """With admission permanently denied, no grants are ever issued yet the queue keeps being offered."""
        controller = _controller()
        proxy = _StubProxy()
        controller.register(2, proxy)
        staged = (ClearanceWaiter(process_id=2, priority=1),)

        for _ in range(5):
            result = controller.step(_inputs(staged=staged), admit_fn=_never_admit)
            assert result.held_process_ids == (2,)
        assert controller.held_grant_count == 0

        # The child times out on its lease and samples unpriced; the controller retires it on done and the
        # slot reopens for the next waiter rather than staying wedged.
        proxy.child_signal_done()
        controller.step(_inputs(), admit_fn=_never_admit)
        assert controller.grant_state(2) is GrantState.IDLE


class TestRemainingSamplingEstimate:
    """The scheduler's remaining-time estimate, built from the step position and the first-step timestamp."""

    def _scheduler(self) -> InferenceScheduler:
        return make_testable_process_manager()._inference_scheduler

    def _sampling_process(
        self,
        *,
        current: int | None,
        total: int | None,
        first_step_at: float | None,
    ) -> HordeProcessInfo:
        process_info = make_mock_process_info(1, state=HordeProcessState.INFERENCE_STARTING)
        process_info.last_current_step = current
        process_info.last_total_steps = total
        process_info.current_first_step_at = first_step_at
        return process_info

    def test_extrapolates_the_jobs_own_step_rate(self) -> None:
        """Half a 20-step job in 5 seconds leaves about 5 seconds of sampling."""
        scheduler = self._scheduler()
        scheduler._clock = lambda: 105.0
        process_info = self._sampling_process(current=10, total=20, first_step_at=100.0)
        assert scheduler._remaining_sampling_seconds(process_info) == 5.0

    def test_withholds_an_estimate_early_in_the_loop(self) -> None:
        """Below the minimum progress the rate is dominated by loop entry, so no estimate is offered."""
        scheduler = self._scheduler()
        scheduler._clock = lambda: 101.0
        process_info = self._sampling_process(current=1, total=20, first_step_at=100.0)
        assert scheduler._remaining_sampling_seconds(process_info) is None

    def test_withholds_an_estimate_without_a_reported_position(self) -> None:
        """A process that has reported no step position yields nothing to extrapolate from."""
        scheduler = self._scheduler()
        process_info = self._sampling_process(current=None, total=None, first_step_at=100.0)
        assert scheduler._remaining_sampling_seconds(process_info) is None

    def test_saturated_final_step_reads_as_no_time_left(self) -> None:
        """A sampler sitting on its final step has nothing left to extrapolate: the handoff is due now."""
        scheduler = self._scheduler()
        scheduler._clock = lambda: 110.0
        process_info = self._sampling_process(current=20, total=20, first_step_at=100.0)
        assert scheduler._remaining_sampling_seconds(process_info) == 0.0


class TestIncomingLoadEstimate:
    """The measured weight-upload cost the handoff window is sized from."""

    def test_median_of_recent_uploads_per_card(self) -> None:
        """Only GPU-upload phases count, and the card's own uploads answer for it."""
        process_map = ProcessMap()
        process_map[1] = make_mock_process_info(1, device_index=0)
        process_map.on_job_metrics(
            1,
            JobPhaseMetrics(
                model_loads=[
                    ModelLoadEvent(model_name="m", phase="disk_to_ram", duration_seconds=30.0, timestamp=1.0),
                    ModelLoadEvent(model_name="m", phase="ram_to_vram", duration_seconds=4.0, timestamp=2.0),
                    ModelLoadEvent(model_name="m", phase="ram_to_vram", duration_seconds=6.0, timestamp=3.0),
                ],
            ),
        )
        assert process_map.recent_vram_load_seconds(0) == 5.0
        assert process_map.recent_vram_load_seconds(1) is None

    def test_no_observed_upload_reads_as_unmeasured(self) -> None:
        """A job that found its weights resident reports no upload, so the card stays unmeasured."""
        process_map = ProcessMap()
        process_map[1] = make_mock_process_info(1, device_index=0)
        process_map.on_job_metrics(1, JobPhaseMetrics())
        assert process_map.recent_vram_load_seconds(0) is None
