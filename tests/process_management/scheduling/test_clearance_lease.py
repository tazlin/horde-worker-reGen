"""Unit tests for the per-process GPU denoise clearance lease: proxy protocol and controller truth table."""

from __future__ import annotations

from horde_worker_regen.process_management.scheduling.clearance_lease import (
    ActiveSampler,
    ClearanceController,
    ClearanceInputs,
    ClearanceLeaseProxy,
    ClearancePlan,
    ClearanceWaiter,
    GrantState,
    decide_clearances,
)

_THRESHOLD = 0.8
_MARGIN_MB = 3072.0


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
) -> ClearanceInputs:
    return ClearanceInputs(
        staged_waiters=staged,
        active_grants=active,
        device_free_mb=device_free_mb,
        vram_reserve_mb=reserve_mb,
        paging_active=paging,
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
        tail_overlap_progress_threshold=_THRESHOLD,
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
        active = (ActiveSampler(process_id=1, job_id="job-a", progress_fraction=0.9),)
        staged = (ClearanceWaiter(process_id=2, priority=1),)
        plan = _decide(_inputs(staged=staged, active=active), slot_cap=1, tail=True)
        assert plan.clear_process_ids == (2,)
        assert plan.tail_cleared_for_job_id == "job-a"

    def test_tail_overlap_suppressed_for_already_cleared_outgoing_job(self) -> None:
        """A tail bonus fires once per outgoing sampler: a second tick for the same job clears no extra."""
        active = (ActiveSampler(process_id=1, job_id="job-a", progress_fraction=0.9),)
        staged = (ClearanceWaiter(process_id=2, priority=1),)
        plan = _decide(_inputs(staged=staged, active=active), slot_cap=1, tail=True, cleared=frozenset({"job-a"}))
        assert plan.clear_process_ids == ()

    def test_tail_overlap_disabled_never_grants_extra(self) -> None:
        """With tail overlap off, a full cap admits no handoff grant however advanced the sampler."""
        active = (ActiveSampler(process_id=1, job_id="job-a", progress_fraction=0.99),)
        staged = (ClearanceWaiter(process_id=2, priority=1),)
        plan = _decide(_inputs(staged=staged, active=active), slot_cap=1, tail=False)
        assert plan.clear_process_ids == ()

    def test_tail_overlap_held_below_threshold(self) -> None:
        """A sampler short of its tail does not open the handoff slot."""
        active = (ActiveSampler(process_id=1, job_id="job-a", progress_fraction=0.5),)
        staged = (ClearanceWaiter(process_id=2, priority=1),)
        assert _decide(_inputs(staged=staged, active=active), slot_cap=1, tail=True).clear_process_ids == ()

    def test_tail_overlap_held_under_paging(self) -> None:
        """Under WDDM paging the measured free is untrustworthy, so no early clear."""
        active = (ActiveSampler(process_id=1, job_id="job-a", progress_fraction=0.9),)
        staged = (ClearanceWaiter(process_id=2, priority=1),)
        held = _inputs(staged=staged, active=active, paging=True)
        assert _decide(held, slot_cap=1, tail=True).clear_process_ids == ()

    def test_tail_overlap_held_below_margin(self) -> None:
        """Measured free net of the reserve below the margin withholds the handoff grant."""
        active = (ActiveSampler(process_id=1, job_id="job-a", progress_fraction=0.9),)
        staged = (ClearanceWaiter(process_id=2, priority=1),)
        # reserve 2048 + margin 3072 = 5120 required; 5000 is short.
        held = _inputs(staged=staged, active=active, device_free_mb=5000.0)
        assert _decide(held, slot_cap=1, tail=True).clear_process_ids == ()


class _StubProxy(ClearanceLeaseProxy):
    """A proxy backed by fake semaphores, so the controller drives real grant/done edges under test."""

    def __init__(self) -> None:
        super().__init__(clearance=_held_empty_clearance(), done=_FakeSemaphore())

    @property
    def clearance_value(self) -> int:
        return self._clearance.value  # type: ignore[attr-defined]

    def child_signal_done(self) -> None:
        self._done.release()  # type: ignore[attr-defined]


def _controller(*, slot_cap: int = 1, tail: bool = True) -> ClearanceController:
    return ClearanceController(
        device_index=0,
        slot_cap=slot_cap,
        tail_overlap=tail,
        tail_overlap_progress_threshold=_THRESHOLD,
        tail_overlap_margin_mb=_MARGIN_MB,
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
        active = (ActiveSampler(process_id=1, job_id="job-a", progress_fraction=0.9),)
        # The outgoing sampler holds the only steady slot, so the incoming waiter can only clear on the bonus.
        controller.step(
            _inputs(staged=(ClearanceWaiter(process_id=2, priority=1),), active=active),
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
            active = (ActiveSampler(process_id=1, job_id="job-a", progress_fraction=0.95),)
            controller.step(
                _inputs(staged=(ClearanceWaiter(process_id=3, priority=2),), active=active),
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
