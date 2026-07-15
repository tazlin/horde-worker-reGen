"""Exploratory boundary matrix for the save-our-ship give-up decision across pool-boot shapes.

These cases probe the two-directional liveness invariant of the readiness-gated give-up:

* **Never fault servable work while recovery is still advancing.** A replacement pool that is alive and
  still booting (or a ready lane whose head model is still materialising) is capacity in flight, not an
  unrecoverable wedge. Give-up must not fault jobs the finishing boot would serve.
* **Always give up within bounded time on truly dead recovery.** A boot that never completes because the
  child died, or is pathologically hung, must still reach give-up so the horde can reissue the work.

Their disposition is deliberately left open: each asserts the *invariant* for its shape rather than a
pre-assumed pass/fail, so a failure localises which boundary the current (or a future) implementation gets
wrong. Cases whose boot outlasts the fixed allowance while the pool is alive are expected to fault today
(the defect); the dead and hung cases guard the opposite edge so a remedy that over-corrects into an
unbounded park is caught. The cases are grouped by shape; none is a regression guard until its direction is
settled.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)
from tests.process_management.regressions.test_giveup_respects_in_progress_boot import (
    _BOOT_ALLOWANCE,
    _POOL_READY_GRACE,
    _abandoned_records,
    _disable_pp_reclaim_yield,
    _FakeClock,
    _install_supervisor,
    _latch_structural_queue_wedge,
    _spy_give_up,
)


class TestSlowButHealthyBootIsNotFaulted:
    """Invariant: a replacement pool that is alive and still booting must not have its head faulted.

    The boot durations span the interesting boundary: just under the allowance (the healthy fast boot), just
    over it, and well over it. In every case the boot eventually completes and the pool serves the head, so
    give-up faulting the head at any point before that completion is a liveness violation.
    """

    @pytest.mark.parametrize(
        "allowance_multiple",
        [0.5, 1.33, 1.6, 4.0],
        ids=["just-under", "just-over", "1.6x", "4x"],
    )
    async def test_boot_completes_and_head_is_served(
        self,
        allowance_multiple: float,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A rebuild that boots for ``allowance_multiple`` x the allowance then serves the head, unfaulted."""
        pm = make_testable_process_manager()
        clock = _FakeClock()
        _install_supervisor(pm, clock, clean_streak_seconds=4)
        await _latch_structural_queue_wedge(pm)
        _disable_pp_reclaim_yield(pm, monkeypatch)
        # A terminal give-up would arm the real force-kill backstop and write an abort sentinel file; both
        # outlive the test and kill the interpreter later in the session, so it is neutralised here.
        monkeypatch.setattr(pm, "_abort", lambda: None)
        proc = pm._process_map[0]

        monkeypatch.setattr(
            pm._process_lifecycle,
            "rebuild_inference_pool",
            lambda *, reason: setattr(proc, "last_process_state", HordeProcessState.PROCESS_STARTING),
        )
        monkeypatch.setattr(pm._process_lifecycle, "rebuild_safety_pool", lambda *, reason: None)
        give_up_calls = _spy_give_up(pm, monkeypatch)

        boot_seconds = int(_BOOT_ALLOWANCE * allowance_multiple) + 1
        for _ in range(boot_seconds):
            clock.advance(1)
            pm._recovery_coordinator.run_recovery_supervisor()
            assert proc.is_process_alive() is True  # the boot is progressing, never dead
            # The head must survive the entire live-boot window.
            assert give_up_calls == [], f"head faulted mid-boot at t={clock.now} (x{allowance_multiple})"

        # The boot completes: the slot accepts again, the wedge clears, and the head is served (progress).
        proc.last_process_state = HordeProcessState.WAITING_FOR_JOB
        pm._message_dispatcher._in_queue_deadlock = False
        pm._job_tracker._total_num_completed_jobs += 1
        for _ in range(6):
            clock.advance(1)
            pm._recovery_coordinator.run_recovery_supervisor()

        assert give_up_calls == []
        assert _abandoned_records(pm) == []
        assert len(list(pm._job_tracker.jobs_pending_inference)) == 1


class TestDeadOrHungBootStillGivesUp:
    """Invariant: recovery that is truly dead or pathologically hung must reach give-up in bounded time.

    The opposite edge of the slow-boot cases: give-up is a backstop, not a hold that can be parked forever.
    A child that died mid-boot, or one that never leaves ``PROCESS_STARTING`` at all, must still let the
    escalation fault the unservable work rather than spin indefinitely.
    """

    async def test_child_dies_during_boot_reaches_give_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A replacement child that dies mid-boot (exitcode set) must still escalate to give-up.

        The child is not alive (its exitcode is set) even though its last reported state has not yet been
        reaped from ``PROCESS_STARTING``. A remedy keyed only on the raw starting-count, without checking
        liveness, would park here forever; the invariant is that a dead boot escalates within bounded time.
        """
        pm = make_testable_process_manager()
        clock = _FakeClock()
        _install_supervisor(pm, clock, max_soft_resets=0)
        await _latch_structural_queue_wedge(pm)
        _disable_pp_reclaim_yield(pm, monkeypatch)
        monkeypatch.setattr(pm, "_abort", lambda: None)
        monkeypatch.setattr(pm._process_lifecycle, "rebuild_inference_pool", lambda *, reason: None)

        proc = pm._process_map[0]
        proc.last_process_state = HordeProcessState.PROCESS_STARTING
        # The mock child reports it died mid-boot: no longer alive, exitcode set.
        proc.mp_process.is_alive.return_value = False
        monkeypatch.setattr(proc.mp_process, "exitcode", 1, raising=False)
        assert proc.is_process_alive() is False

        fired_at: float | None = None
        for _ in range(60):
            clock.advance(1)
            pm._recovery_coordinator.run_recovery_supervisor()
            if _abandoned_records(pm):
                fired_at = clock.now
                break

        assert fired_at is not None, "a dead boot never escalated to give-up (unbounded park)"

    async def test_perpetually_starting_child_reaches_give_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A child that stays alive and ``PROCESS_STARTING`` forever must still escalate within bounded time.

        The pathological hang: the replacement process is alive and emitting heartbeats but never reaches an
        accepting state. Holding give-up while a live boot progresses is correct, but the hold must be bounded:
        a boot that never lands has to fault the unservable work eventually rather than park indefinitely.
        """
        pm = make_testable_process_manager()
        clock = _FakeClock()
        _install_supervisor(pm, clock, max_soft_resets=0)
        await _latch_structural_queue_wedge(pm)
        _disable_pp_reclaim_yield(pm, monkeypatch)
        monkeypatch.setattr(pm, "_abort", lambda: None)
        monkeypatch.setattr(pm._process_lifecycle, "rebuild_inference_pool", lambda *, reason: None)

        proc = pm._process_map[0]
        proc.last_process_state = HordeProcessState.PROCESS_STARTING  # alive, never leaves the boot state

        fired_at: float | None = None
        for _ in range(400):
            clock.advance(1)
            pm._recovery_coordinator.run_recovery_supervisor()
            assert proc.is_process_alive() is True  # still a live, heartbeating boot
            if _abandoned_records(pm):
                fired_at = clock.now
                break

        assert fired_at is not None, "a perpetually hung boot never escalated to give-up (unbounded park)"


class TestConsecutiveResetsDuringBoot:
    """Invariant: repeated soft resets whose replacements keep booting still protect servable work.

    With a larger reset budget the escalation may rebuild more than once. While the latest replacement is
    alive and booting, the head must not be faulted on the fixed allowance; the escalation should climb its
    ladder rather than drop servable work each rebuild.
    """

    async def test_alive_booting_replacements_do_not_fault_head_within_patience(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two soft resets, each replacement alive and booting: the head is not faulted within the boot window."""
        pm = make_testable_process_manager()
        clock = _FakeClock()
        _install_supervisor(pm, clock, max_soft_resets=2)
        await _latch_structural_queue_wedge(pm)
        _disable_pp_reclaim_yield(pm, monkeypatch)
        monkeypatch.setattr(pm, "_abort", lambda: None)
        proc = pm._process_map[0]

        monkeypatch.setattr(
            pm._process_lifecycle,
            "rebuild_inference_pool",
            lambda *, reason: setattr(proc, "last_process_state", HordeProcessState.PROCESS_STARTING),
        )
        monkeypatch.setattr(pm._process_lifecycle, "rebuild_safety_pool", lambda *, reason: None)
        give_up_calls = _spy_give_up(pm, monkeypatch)

        # A boot window comfortably longer than the fixed allowance, with the replacement alive throughout.
        for _ in range(int(_BOOT_ALLOWANCE + _POOL_READY_GRACE) * 2):
            clock.advance(1)
            pm._recovery_coordinator.run_recovery_supervisor()

        assert proc.is_process_alive() is True
        assert give_up_calls == []  # servable work was not dropped while replacements kept booting
        assert len(list(pm._job_tracker.jobs_pending_inference)) == 1


class TestJobsPoppedDuringBoot:
    """Invariant: work accepted before the wedge and work popped during the boot are both protected.

    A give-up over a still-booting pool must not fault either the job the wedge formed around or a job the
    worker popped while the replacement was booting; both are servable once the boot completes.
    """

    async def test_pre_wedge_and_mid_boot_jobs_both_survive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A pre-wedge head and a job popped during the boot both survive the whole live-boot window."""
        pm = make_testable_process_manager()
        clock = _FakeClock()
        _install_supervisor(pm, clock)
        await _latch_structural_queue_wedge(pm)
        _disable_pp_reclaim_yield(pm, monkeypatch)
        monkeypatch.setattr(pm, "_abort", lambda: None)  # keep the force-kill backstop out of the interpreter
        proc = pm._process_map[0]

        monkeypatch.setattr(
            pm._process_lifecycle,
            "rebuild_inference_pool",
            lambda *, reason: setattr(proc, "last_process_state", HordeProcessState.PROCESS_STARTING),
        )
        monkeypatch.setattr(pm._process_lifecycle, "rebuild_safety_pool", lambda *, reason: None)
        give_up_calls = _spy_give_up(pm, monkeypatch)

        # Take the soft reset first (the replacement enters the boot window), then pop a second job mid-boot.
        for _ in range(3):
            clock.advance(1)
            pm._recovery_coordinator.run_recovery_supervisor()
        assert proc.last_process_state is HordeProcessState.PROCESS_STARTING
        await track_popped_job_async(pm._job_tracker, make_job_pop_response(model="resident"))
        assert len(list(pm._job_tracker.jobs_pending_inference)) == 2

        for _ in range(int(_BOOT_ALLOWANCE + _POOL_READY_GRACE) * 2):
            clock.advance(1)
            pm._recovery_coordinator.run_recovery_supervisor()

        assert give_up_calls == []  # neither the pre-wedge head nor the mid-boot pop was faulted
        assert len(list(pm._job_tracker.jobs_pending_inference)) == 2


class TestSafetyStillBootingWhileInferenceReady:
    """Invariant: a booting safety pool alone is not a give-up-eligible wedge.

    An alive-but-starting safety process still counts as safety capacity in flight, so the give-up path must
    not fault work waiting on it while a healthy inference lane is ready.
    """

    async def test_booting_safety_is_capacity_in_flight(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Inference ready, safety alive and ``PROCESS_STARTING``: give-up faults nothing and does not abort."""
        pm = make_testable_process_manager()
        clock = _FakeClock()
        _install_supervisor(pm, clock)
        monkeypatch.setattr(pm, "_abort", lambda: None)

        # A ready inference lane serving a queued head, and a safety process still booting.
        pm._process_map[0] = make_mock_process_info(
            0,
            model_name="resident",
            state=HordeProcessState.WAITING_FOR_JOB,
        )
        pm._process_map[1] = make_mock_process_info(
            1,
            model_name=None,
            state=HordeProcessState.PROCESS_STARTING,
            process_type=HordeProcessType.SAFETY,
        )
        await track_popped_job_async(pm._job_tracker, make_job_pop_response(model="resident"))

        coordinator = pm._recovery_coordinator
        assert coordinator.is_inference_pool_ready() is True
        assert coordinator.is_safety_pool_ready() is False  # the safety lane has not finished booting
        assert coordinator.is_safety_capacity_available() is True  # but it is alive: capacity in flight

        before = len(list(pm._job_tracker.jobs_pending_inference))
        # No structural inference wedge is latched, so this direct give-up has no fault trigger: a booting
        # safety pool by itself must not manufacture one.
        coordinator.give_up_on_wedged_jobs(terminal=False)

        assert _abandoned_records(pm) == []
        assert len(list(pm._job_tracker.jobs_pending_inference)) == before  # nothing faulted
