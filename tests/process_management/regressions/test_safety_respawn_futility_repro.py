"""Reproduction: a safety pool that can never start is respawned forever without ever being declared failing.

Three gaps in the safety-process recovery escalation share one consequence: the worker rebuilds a safety
child that dies on every start, indefinitely, while every signal that would escalate stays clear.

1. Laundering. :meth:`ProcessLifecycleManager._replace_all_safety_process` skips the crash-loop record while
   either intentional-replacement flag is set. ``_safety_replacement_intentional_until_ready`` is cleared only
   by :meth:`_observe_safety_pool_readiness`, which requires a *loaded* safety process. A
   child that dies while still ``PROCESS_STARTING`` never loads, so an intentional window opened by a
   whole-card pause never closes and every subsequent crash-driven rebuild is counted as part of the
   placement change. :attr:`ProcessLifecycleManager.safety_pool_failing` stays False for an unbounded
   crash-on-start loop. Contract: the suppression is bounded, so a rebuild loop that never reaches readiness
   is eventually counted and the pool is reported failing.

2. No futility classification. The reap-rebuild-start loop respawns immediately with no ceiling. The only
   breaker is the sliding window (``SAFETY_CRASH_LOOP_MAX`` rebuilds within ``CRASH_LOOP_WINDOW_SECONDS``),
   which a slow cold start outruns: rebuilds spaced wider than the window's per-rebuild budget age out before
   they can accumulate, so the pool respawns forever at a rate the breaker can never see. Contract: the
   rate-independent companion the inference side already has (``CRASH_LOOP_MAX_START_FAILURES``): consecutive
   rebuilds where the child never reaches readiness are structural regardless of spacing, so the pool is
   declared unrecoverable and the wedge assessment sees it.

3. No failure-independent backstop. Job pops can be held at a gate (``no_safety_process`` among them) for
   hours with no jobs flowing. The pop-liveness sentinel logs that, and nothing consumes it: no wedge
   assessment keys on a held gate. Contract: a pop gate held past a threshold with zero job flow is a wedge
   regardless of which gate it is, since the gate's own cause cannot satisfy a completed-work signal.

4. A backstop that recreates its own trigger. The remedy for a held gate is a pool teardown, and a worker on
   its first run is held at ``no_safety_process`` precisely because its safety models are still downloading.
   Tearing the pool down restarts that transfer, so on a link slow enough that the download outlasts the
   wedge window the escalation is the reason the worker never comes up. Contract: while the worker has never
   served a job and its downloads are still gaining bytes, the held-gate escalation stands down; a transfer
   that has stopped moving escalates unchanged.

The controls pin the behavior the fixes must not break: an intentional cycle that does reach readiness stays
uncounted, one crash on a healthy pool is not a loop, the sliding-window threshold still trips without an
intentional window, a rebuild streak broken by a successful load is not structural, and a worker that has
served before is not excused by a download running beside it.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    FEATURE_SAFETY,
    CurrentDownloadStatus,
    DownloadPhase,
    DownloadStatusSnapshot,
)
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.lifecycle.process_lifecycle import (
    CRASH_LOOP_WINDOW_SECONDS,
    SAFETY_CRASH_LOOP_MAX,
    SAFETY_PROCESS_ID,
    SAFETY_RESPAWN_BACKOFF_MAX_SECONDS,
    PauseOwner,
    ProcessLifecycleManager,
)
from horde_worker_regen.process_management.models.model_availability import (
    DOWNLOAD_PROGRESS_STALL_SECONDS,
    ModelAvailability,
)
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from tests.process_management.conftest import make_mock_process_info, make_testable_process_manager

_SLOW_RESPAWN_SECONDS = CRASH_LOOP_WINDOW_SECONDS / 2
"""Spacing between rebuilds that the sliding-window breaker can never accumulate past its threshold.

A cold start this slow lets each rebuild age out of ``CRASH_LOOP_WINDOW_SECONDS`` before enough of them
coincide, which is the condition a rate-independent futility signal has to cover.
"""

_FUTILE_RESPAWN_COUNT = SAFETY_CRASH_LOOP_MAX * 3
"""Rebuild attempts driven per futility scenario: well past any plausible bound on respawning."""

_BACKOFF_CLEARING_RESPAWN_SECONDS = SAFETY_RESPAWN_BACKOFF_MAX_SECONDS * 2
"""Spacing between rebuilds that clears the respawn backoff while staying dense inside the crash-loop window.

Consecutive failed rebuilds are deliberately spaced out by a growing backoff, so a scenario driving rebuild
after rebuild has to move the clock past it or the state machine legitimately declines to respawn yet.
"""

_GATE_HELD_SECONDS = 3600.0
"""How long a pop gate is held in the backstop scenarios: far past any threshold a fix could pick."""

_BRIEF_GATE_HELD_SECONDS = 30.0
"""A gate hold short enough that no threshold should treat it as a wedge."""

_STARTUP_HANG_TIMEOUT_SECONDS = 60.0
"""The startup timeout a wedged safety child is judged against."""

_STARTUP_HANG_SECONDS = _STARTUP_HANG_TIMEOUT_SECONDS * 10
"""How long a wedged safety child has sat in its startup state: far past any plausible timeout."""

_SAFETY_MODEL_BYTES = 675_000_000
"""A stand-in size for the safety weight, only large enough that a quarter of it is plainly a partial file."""

_OPAQUE_GRACE_SECONDS = 900.0
"""The window a caller grants a provisioning step that cannot report bytes; the wedge passes its own."""


class _AdvanceableTime:
    """Stand-in for the ``time`` module whose ``time()`` the test can advance.

    Every other attribute is forwarded to the real module, so code paths that use ``monotonic`` or ``sleep``
    are unaffected. Only the wall clock the crash-loop window is measured against moves.
    """

    def __init__(self) -> None:
        self._offset = 0.0

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - forwards the real module's arbitrary members
        """Return the real ``time`` module's attribute of that name."""
        return getattr(time, name)

    def time(self) -> float:
        """Return the current wall clock, shifted by however far the test has advanced it."""
        return time.time() + self._offset

    def advance(self, seconds: float) -> None:
        """Move the wall clock forward."""
        self._offset += seconds


class _FakeClock:
    """A clock the test advances explicitly, so gate-hold durations are exact."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward."""
        self.now += seconds


def _stub_safety_start(lifecycle: ProcessLifecycleManager, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the rebuild state machine from spawning a child, without changing what it records.

    The replacement flow's start step is the only part that would touch the OS. Everything the scenarios read
    (the recovery history, the intentional-window flags, the pool's failing verdict) is decided around it.
    """
    monkeypatch.setattr(lifecycle, "start_safety_processes", lambda: True)


def _crashed_starting_safety_process() -> HordeProcessInfo:
    """A safety child that died while still ``PROCESS_STARTING`` (it never reached readiness)."""
    crashed = make_mock_process_info(
        SAFETY_PROCESS_ID,
        model_name=None,
        state=HordeProcessState.PROCESS_STARTING,
        process_type=HordeProcessType.SAFETY,
    )
    crashed.mp_process.is_alive.return_value = False  # pyrefly: ignore - Mock stand-in for the child handle
    crashed.mp_process.exitcode = -6  # pyrefly: ignore - Mock stand-in; a native abort, not a clean end
    return crashed


def _drive_rebuild_to_completion(lifecycle: ProcessLifecycleManager) -> None:
    """Tick the safety replacement state machine until it has started a fresh safety process."""
    for _ in range(5):
        if not lifecycle.safety_processes_should_be_replaced:
            return
        lifecycle._replace_all_safety_process()
    raise AssertionError("the safety replacement state machine never completed a rebuild cycle")


def _crash_safety_during_startup(lifecycle: ProcessLifecycleManager) -> None:
    """Run one full cycle of a safety child dying before readiness and being rebuilt."""
    crashed = _crashed_starting_safety_process()
    lifecycle._process_map[SAFETY_PROCESS_ID] = crashed
    assert lifecycle._reap_if_crashed(crashed) is True
    _drive_rebuild_to_completion(lifecycle)


def _hang_safety_during_startup(lifecycle: ProcessLifecycleManager) -> None:
    """Run one full cycle of a live safety child wedged in its own startup being timed out and rebuilt.

    The child never exits, so the crash reaper never sees it; the state-duration timeout is the only thing
    that replaces it.
    """
    hung = make_mock_process_info(
        SAFETY_PROCESS_ID,
        model_name=None,
        state=HordeProcessState.PROCESS_STARTING,
        process_type=HordeProcessType.SAFETY,
    )
    stale = time.time() - _STARTUP_HANG_SECONDS
    hung.last_received_timestamp = stale
    hung.last_heartbeat_timestamp = stale
    hung.last_process_state_started_at = stale
    lifecycle._process_map[SAFETY_PROCESS_ID] = hung
    assert (
        lifecycle._check_and_replace_process(
            hung,
            _STARTUP_HANG_TIMEOUT_SECONDS,
            HordeProcessState.PROCESS_STARTING,
            "seems to be stuck starting",
        )
        is True
    )
    _drive_rebuild_to_completion(lifecycle)


def _safety_child_reaches_readiness(lifecycle: ProcessLifecycleManager) -> None:
    """Put a ready safety process in the map and let the lifecycle observe that readiness."""
    lifecycle._process_map[SAFETY_PROCESS_ID] = make_mock_process_info(
        SAFETY_PROCESS_ID,
        model_name=None,
        state=HordeProcessState.WAITING_FOR_JOB,
        process_type=HordeProcessType.SAFETY,
    )
    lifecycle._observe_safety_pool_readiness()


def _hold_pop_gate(pm: HordeWorkerProcessManager, clock: _FakeClock, gate: str, held_seconds: float) -> None:
    """Report job pops as held at ``gate`` for ``held_seconds``, with no pop reaching the horde since."""
    pm._state.last_pop_gate = gate
    pm._state.last_pop_gate_since = clock.now - held_seconds
    pm._state.last_pop_attempt_completed_at = clock.now - held_seconds


class TestIntentionalWindowStopsLaunderingCrashLoops:
    """An intentional replacement window must not absorb a crash loop that never reaches readiness."""

    def test_crash_on_start_loop_inside_an_intentional_window_is_eventually_counted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A safety child crashing before readiness must trip the pool's failing signal, window open or not.

        The whole-card pause opens the window legitimately; what must not follow is an unbounded rebuild loop
        that the window keeps invisible to every consumer of ``safety_pool_failing``.
        """
        pm = make_testable_process_manager(safety_on_gpu=True)
        lifecycle = pm._process_lifecycle
        _stub_safety_start(lifecycle, monkeypatch)

        fake_time = _AdvanceableTime()
        monkeypatch.setattr(
            "horde_worker_regen.process_management.lifecycle.process_lifecycle.time",
            fake_time,
        )

        assert lifecycle.pause_safety_on_gpu(owner=PauseOwner.WHOLE_CARD) is True
        _drive_rebuild_to_completion(lifecycle)
        assert lifecycle._safety_replacement_intentional_until_ready is True

        for _ in range(_FUTILE_RESPAWN_COUNT):
            _crash_safety_during_startup(lifecycle)
            fake_time.advance(_BACKOFF_CLEARING_RESPAWN_SECONDS)

        assert lifecycle.safety_pool_failing is True, (
            f"{_FUTILE_RESPAWN_COUNT} safety rebuilds, none of which reached readiness, left the pool "
            "reported healthy: the intentional-replacement window suppressed every one of them, so no "
            "consumer of the crash-loop signal can see a pool that cannot start at all"
        )

    def test_intentional_window_closes_on_readiness_and_later_crashes_count(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An intentional cycle that does reach readiness closes its window, and later crashes are counted.

        The suppression exists so a burst of whole-card pauses does not read as a crash loop. That must keep
        working through the real pause entry point, and it must not outlive the placement change it covers.
        """
        pm = make_testable_process_manager(safety_on_gpu=True)
        lifecycle = pm._process_lifecycle
        _stub_safety_start(lifecycle, monkeypatch)

        fake_time = _AdvanceableTime()
        monkeypatch.setattr(
            "horde_worker_regen.process_management.lifecycle.process_lifecycle.time",
            fake_time,
        )

        assert lifecycle.pause_safety_on_gpu(owner=PauseOwner.WHOLE_CARD) is True
        _drive_rebuild_to_completion(lifecycle)
        assert lifecycle._safety_recovery_history == []

        _safety_child_reaches_readiness(lifecycle)
        assert lifecycle._safety_replacement_intentional_until_ready is False

        for _ in range(SAFETY_CRASH_LOOP_MAX + 1):
            _crash_safety_during_startup(lifecycle)
            fake_time.advance(_BACKOFF_CLEARING_RESPAWN_SECONDS)

        assert lifecycle.safety_pool_failing is True


class TestSafetyRespawnFutility:
    """Consecutive rebuilds that never reach readiness must be classified structural, whatever their spacing."""

    def test_slow_consecutive_start_failures_declare_the_pool_unrecoverable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rebuilds too slow for the sliding window must still be recognised as a pool that cannot start.

        A deterministic start failure costs a full cold start each time, so the failures arrive spaced out.
        Counting only within a window means the worker respawns such a pool forever; the streak itself is the
        evidence, and the wedge assessment has to see it.
        """
        pm = make_testable_process_manager()
        lifecycle = pm._process_lifecycle
        coordinator = pm._recovery_coordinator
        _stub_safety_start(lifecycle, monkeypatch)

        fake_time = _AdvanceableTime()
        monkeypatch.setattr(
            "horde_worker_regen.process_management.lifecycle.process_lifecycle.time",
            fake_time,
        )

        for _ in range(_FUTILE_RESPAWN_COUNT):
            _crash_safety_during_startup(lifecycle)
            fake_time.advance(_SLOW_RESPAWN_SECONDS)

        # No safety process ever became ready and none is deferred for headroom, so the pool's own failing
        # verdict is the only thing standing between this streak and the unrecoverable classification.
        assert coordinator.is_safety_pool_ready() is False
        assert lifecycle.has_pending_safety_starts() is False

        assert coordinator.is_safety_pool_unrecoverable() is True, (
            f"{_FUTILE_RESPAWN_COUNT} consecutive safety rebuilds, none reaching readiness, left the pool "
            "classified recoverable because they were spaced wider than the crash-loop window; the worker "
            "would respawn this pool for as long as it runs"
        )
        assert coordinator.assess_wedge() is True, (
            "a safety pool that cannot start is on every job's path, so the wedge assessment must report it "
            "and let the escalation reach its terminal rung"
        )

    def test_a_successful_load_breaks_the_start_failure_streak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A rebuild streak interrupted by a safety process that does reach readiness is not structural.

        Futility is about a pool that never initialises. One that comes up, serves, and later dies is an
        ordinary crash the rebuild path is for, so the streak must not carry across the successful load.
        """
        pm = make_testable_process_manager()
        lifecycle = pm._process_lifecycle
        coordinator = pm._recovery_coordinator
        _stub_safety_start(lifecycle, monkeypatch)

        fake_time = _AdvanceableTime()
        monkeypatch.setattr(
            "horde_worker_regen.process_management.lifecycle.process_lifecycle.time",
            fake_time,
        )

        for _ in range(2):
            _crash_safety_during_startup(lifecycle)
            fake_time.advance(_SLOW_RESPAWN_SECONDS)

        _safety_child_reaches_readiness(lifecycle)
        fake_time.advance(_SLOW_RESPAWN_SECONDS)

        for _ in range(2):
            _crash_safety_during_startup(lifecycle)
            fake_time.advance(_SLOW_RESPAWN_SECONDS)

        assert coordinator.is_safety_pool_unrecoverable() is False

    def test_single_safety_crash_respawns_without_declaring_the_pool_failing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One safety crash is recovered in place and escalates nothing.

        The rebuild path exists for exactly this. Neither the crash-loop signal nor the wedge assessment may
        fire on a lone crash, or every transient safety fault would drive the terminal escalation.
        """
        pm = make_testable_process_manager()
        lifecycle = pm._process_lifecycle
        coordinator = pm._recovery_coordinator
        _stub_safety_start(lifecycle, monkeypatch)

        _crash_safety_during_startup(lifecycle)

        assert lifecycle.safety_pool_failing is False
        assert coordinator.is_safety_pool_unrecoverable() is False
        assert coordinator.assess_wedge() is False

    def test_crash_loop_window_threshold_trips_without_an_intentional_window(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rebuilds inside the crash-loop window trip the failing signal once they pass the threshold.

        The direct pin on ``SAFETY_CRASH_LOOP_MAX``: at the threshold the pool is still merely churning, and
        the rebuild past it is what makes it a loop.
        """
        pm = make_testable_process_manager()
        lifecycle = pm._process_lifecycle
        _stub_safety_start(lifecycle, monkeypatch)

        fake_time = _AdvanceableTime()
        monkeypatch.setattr(
            "horde_worker_regen.process_management.lifecycle.process_lifecycle.time",
            fake_time,
        )

        for _ in range(SAFETY_CRASH_LOOP_MAX):
            _crash_safety_during_startup(lifecycle)
            fake_time.advance(_BACKOFF_CLEARING_RESPAWN_SECONDS)
        assert lifecycle.safety_pool_failing is False

        _crash_safety_during_startup(lifecycle)
        assert lifecycle.safety_pool_failing is True


class TestCrashLoopVerdictNeedsACrash:
    """The unrecoverable verdict drops a job's images, so it requires evidence a child actually died.

    A whole-card pause/restore cycle ends the safety child on the parent's own request and rebuilds it. Enough
    of those in a row are indistinguishable from a crash loop by rebuild count alone, and reading them as one
    faults jobs whose safety checks the outgoing launch had in fact completed.
    """

    def test_parent_initiated_restart_cycles_are_not_an_unrecoverable_pool(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rebuilds past every threshold, with no child reaped, are not an unrecoverable pool."""
        pm = make_testable_process_manager(safety_on_gpu=True)
        lifecycle = pm._process_lifecycle
        coordinator = pm._recovery_coordinator
        _stub_safety_start(lifecycle, monkeypatch)

        fake_time = _AdvanceableTime()
        monkeypatch.setattr(
            "horde_worker_regen.process_management.lifecycle.process_lifecycle.time",
            fake_time,
        )

        assert lifecycle.pause_safety_on_gpu(owner=PauseOwner.WHOLE_CARD) is True
        for _ in range(_FUTILE_RESPAWN_COUNT):
            _drive_rebuild_to_completion(lifecycle)
            fake_time.advance(_BACKOFF_CLEARING_RESPAWN_SECONDS)
            lifecycle._initiate_safety_replacement()

        assert lifecycle.safety_pool_failure_evidence_seen is False
        assert lifecycle.safety_pool_start_failing is True
        assert coordinator.is_safety_pool_ready() is False
        assert coordinator.is_safety_pool_unrecoverable() is False, (
            "restart cycles the parent asked for were counted as a crash loop, so jobs the outgoing launch "
            "was still checking would be faulted for a pool that never crashed"
        )

    def test_a_reaped_child_restores_the_unrecoverable_verdict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Once a safety child actually dies, the same rebuild streak reads as unrecoverable again."""
        pm = make_testable_process_manager()
        lifecycle = pm._process_lifecycle
        coordinator = pm._recovery_coordinator
        _stub_safety_start(lifecycle, monkeypatch)

        fake_time = _AdvanceableTime()
        monkeypatch.setattr(
            "horde_worker_regen.process_management.lifecycle.process_lifecycle.time",
            fake_time,
        )

        for _ in range(_FUTILE_RESPAWN_COUNT):
            _crash_safety_during_startup(lifecycle)
            fake_time.advance(_SLOW_RESPAWN_SECONDS)

        assert lifecycle.safety_pool_failure_evidence_seen is True
        assert coordinator.is_safety_pool_unrecoverable() is True

    def test_children_wedged_in_startup_still_reach_the_unrecoverable_verdict(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A child that hangs in its startup never exits, so the timeout reap is the only failure evidence.

        Requiring a child that *died* would exempt the whole hung-start class (a child wedged in accelerator
        init, say) from the futility verdict and respawn it for the life of the worker.
        """
        pm = make_testable_process_manager()
        lifecycle = pm._process_lifecycle
        coordinator = pm._recovery_coordinator
        _stub_safety_start(lifecycle, monkeypatch)

        fake_time = _AdvanceableTime()
        monkeypatch.setattr(
            "horde_worker_regen.process_management.lifecycle.process_lifecycle.time",
            fake_time,
        )

        for _ in range(_FUTILE_RESPAWN_COUNT):
            _hang_safety_during_startup(lifecycle)
            fake_time.advance(_BACKOFF_CLEARING_RESPAWN_SECONDS)

        assert lifecycle.safety_pool_failure_evidence_seen is True
        assert coordinator.is_safety_pool_ready() is False
        assert coordinator.is_safety_pool_unrecoverable() is True, (
            f"{_FUTILE_RESPAWN_COUNT} safety children were reaped for never finishing their startup and the "
            "pool was still classified recoverable, so the worker would respawn it indefinitely"
        )

    def test_readiness_clears_the_crash_evidence_with_the_streak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A pool that comes up starts a fresh streak, so old deaths do not condemn later rebuilds."""
        pm = make_testable_process_manager()
        lifecycle = pm._process_lifecycle
        _stub_safety_start(lifecycle, monkeypatch)

        _crash_safety_during_startup(lifecycle)
        assert lifecycle.safety_pool_failure_evidence_seen is True

        _safety_child_reaches_readiness(lifecycle)

        assert lifecycle.safety_pool_failure_evidence_seen is False


class TestHeldPopGateBackstop:
    """A pop gate held with no work flowing is a wedge whatever holds it."""

    def test_long_held_gate_with_no_job_flow_is_a_wedge(self) -> None:
        """Pops gated for an hour with nothing completed must register as a wedge.

        This is the failure-independent backstop: the signal is completed work, which no gate-holding failure
        can produce on its own, so it stands whether the gate is safety, inference, or anything later added.
        """
        pm = make_testable_process_manager()
        clock = _FakeClock()
        pm._recovery_coordinator._clock = clock
        _hold_pop_gate(pm, clock, "no_safety_process", _GATE_HELD_SECONDS)

        assert pm._job_tracker.total_num_completed_jobs == 0
        assert pm._recovery_coordinator.assess_wedge() is True, (
            f"job pops were held at a gate for {_GATE_HELD_SECONDS:.0f}s with nothing completed and the "
            "worker reported no wedge: the held gate is disclosed to the log and consumed by nothing, so "
            "the escalation never opens an episode over a worker that is serving nothing at all"
        )

    def test_briefly_held_gate_is_not_a_wedge(self) -> None:
        """A gate held for a short spell is ordinary backpressure and must not register as a wedge."""
        pm = make_testable_process_manager()
        clock = _FakeClock()
        pm._recovery_coordinator._clock = clock
        _hold_pop_gate(pm, clock, "no_safety_process", _BRIEF_GATE_HELD_SECONDS)

        assert pm._recovery_coordinator.assess_wedge() is False

    def test_long_held_gate_with_work_completing_is_not_a_wedge(self) -> None:
        """A gate that has held a long time while jobs still complete is not a wedge.

        The gate name alone proves nothing: a worker whose queue is full holds a gate continuously while
        serving at full rate. Only the absence of completed work makes the hold a wedge.
        """
        pm = make_testable_process_manager()
        clock = _FakeClock()
        pm._recovery_coordinator._clock = clock
        _hold_pop_gate(pm, clock, "no_inference_process", _GATE_HELD_SECONDS)

        pm._job_tracker._total_num_completed_jobs += 5
        pm._state.last_pop_attempt_completed_at = clock.now

        assert pm._recovery_coordinator.assess_wedge() is False


def _safety_download_status(downloaded_bytes: int) -> DownloadStatusSnapshot:
    """A download-process report with the safety-model transfer in flight at ``downloaded_bytes``."""
    return DownloadStatusSnapshot(
        phase=DownloadPhase.DOWNLOADING,
        active=[
            CurrentDownloadStatus(
                model_name="safety models",
                feature=FEATURE_SAFETY,
                target_dir="/models/clip_blip",
                downloaded_bytes=downloaded_bytes,
                total_bytes=_SAFETY_MODEL_BYTES,
            ),
        ],
    )


def _report_download(availability: ModelAvailability, downloaded_bytes: int) -> None:
    """Push one download-process report through the availability holder's single writer."""
    availability.update(
        present=set(),
        currently_downloading="safety models",
        pending=(),
        failed=(),
        status=_safety_download_status(downloaded_bytes),
        scan_complete=True,
    )


class TestProvisioningDownloadDefersTheHeldGateBackstop:
    """A first run whose weights are still arriving is provisioning, and its remedy would restart them."""

    def test_advancing_first_run_download_is_not_a_wedge(self) -> None:
        """A worker that has never served, held at the safety gate with bytes still arriving, is not wedged.

        The escalation's remedy is a pool teardown, which restarts the transfer the gate is waiting on. On a
        link slow enough that the download outlasts the wedge window, escalating is what stops the worker
        ever coming up.
        """
        pm = make_testable_process_manager()
        clock = _FakeClock()
        pm._recovery_coordinator._clock = clock
        pm._model_availability._clock = clock
        _report_download(pm._model_availability, _SAFETY_MODEL_BYTES // 4)
        _hold_pop_gate(pm, clock, "no_safety_process", _GATE_HELD_SECONDS)

        assert pm._model_availability.download_advancing() is True
        assert pm._recovery_coordinator.assess_wedge() is False, (
            "a first-run worker was called wedged while its safety models were still downloading; the "
            "teardown that follows restarts the transfer, so the escalation is what prevents the worker "
            "from ever finishing its first download"
        )

    def test_stalled_download_still_escalates(self) -> None:
        """A transfer that has stopped gaining bytes is a wedge like any other hold.

        The deferral is keyed on movement, not on a download merely being registered, so a dead transfer
        cannot hold recovery off indefinitely.
        """
        pm = make_testable_process_manager()
        clock = _FakeClock()
        pm._recovery_coordinator._clock = clock
        pm._model_availability._clock = clock
        _report_download(pm._model_availability, _SAFETY_MODEL_BYTES // 4)

        clock.advance(DOWNLOAD_PROGRESS_STALL_SECONDS * 2)
        _report_download(pm._model_availability, _SAFETY_MODEL_BYTES // 4)
        _hold_pop_gate(pm, clock, "no_safety_process", _GATE_HELD_SECONDS)

        assert pm._model_availability.download_advancing() is False
        assert pm._recovery_coordinator.assess_wedge() is True

    def test_a_worker_that_has_served_is_not_excused_by_a_download(self) -> None:
        """A download running beside a wedged, previously-serving worker is not provisioning.

        The deferral covers the first run only: once the worker has served, a held gate is a fault in a
        working pool and a background download says nothing about it.
        """
        pm = make_testable_process_manager()
        clock = _FakeClock()
        pm._recovery_coordinator._clock = clock
        pm._model_availability._clock = clock
        _report_download(pm._model_availability, _SAFETY_MODEL_BYTES // 4)
        pm._job_tracker._total_num_completed_jobs += 5
        _hold_pop_gate(pm, clock, "no_safety_process", _GATE_HELD_SECONDS)

        assert pm._model_availability.download_advancing() is True
        assert pm._recovery_coordinator.assess_wedge() is True


class TestDownloadProgressLiveness:
    """``download_advancing`` separates a slow transfer from a stopped one, at any link speed."""

    def test_a_download_that_gains_bytes_keeps_advancing(self) -> None:
        """Reports that each carry more bytes keep the transfer live however far apart they are."""
        clock = _FakeClock()
        availability = ModelAvailability(clock=clock)
        _report_download(availability, 1)
        for step in range(2, 6):
            clock.advance(DOWNLOAD_PROGRESS_STALL_SECONDS * 0.9)
            _report_download(availability, step)
            assert availability.download_advancing() is True

    def test_an_empty_report_ends_the_window(self) -> None:
        """With nothing in flight there is no transfer to call advancing."""
        clock = _FakeClock()
        availability = ModelAvailability(clock=clock)
        _report_download(availability, 1)
        availability.update(
            present={"a"},
            currently_downloading=None,
            pending=(),
            failed=(),
            status=DownloadStatusSnapshot(phase=DownloadPhase.IDLE),
            scan_complete=True,
        )

        assert availability.download_advancing() is False

    def test_a_later_download_starts_its_own_window(self) -> None:
        """A fresh transfer is judged on its own bytes, not against a predecessor's larger total."""
        clock = _FakeClock()
        availability = ModelAvailability(clock=clock)
        _report_download(availability, _SAFETY_MODEL_BYTES // 2)
        availability.update(
            present={"a"},
            currently_downloading=None,
            pending=(),
            failed=(),
            status=DownloadStatusSnapshot(phase=DownloadPhase.IDLE),
            scan_complete=True,
        )

        clock.advance(DOWNLOAD_PROGRESS_STALL_SECONDS * 10)
        _report_download(availability, 1)

        assert availability.download_advancing() is True


def _report_opaque_step(availability: ModelAvailability) -> None:
    """Push a report for a provisioning step that is in flight but can report no bytes.

    The CLIP interrogator fetch is the real one: horde_safety fetches and loads it in a single call, so
    from the parent it is a task at zero bytes for its whole duration.
    """
    availability.update(
        present=set(),
        currently_downloading="safety models",
        pending=(),
        failed=(),
        status=DownloadStatusSnapshot(
            phase=DownloadPhase.DOWNLOADING,
            active=[
                CurrentDownloadStatus(
                    model_name="safety models",
                    feature=FEATURE_SAFETY,
                    target_dir="/models/clip_blip",
                    downloaded_bytes=0,
                    total_bytes=0,
                ),
            ],
        ),
        scan_complete=True,
    )


class TestOpaqueProvisioningStep:
    """A provisioning step with no byte signal is trusted for a bounded window, then treated as any hold."""

    def test_a_byte_less_step_defers_the_backstop_within_its_grace(self) -> None:
        """The CLIP fetch reports nothing, so movement cannot excuse it; its start is all there is to go on."""
        clock = _FakeClock()
        availability = ModelAvailability(clock=clock)
        _report_opaque_step(availability)

        clock.advance(_OPAQUE_GRACE_SECONDS / 2)

        assert availability.download_advancing(opaque_grace_seconds=_OPAQUE_GRACE_SECONDS) is True

    def test_the_grace_expires_so_the_step_cannot_hold_recovery_off(self) -> None:
        """A byte-less step delays the escalation once; it must never cancel it."""
        clock = _FakeClock()
        availability = ModelAvailability(clock=clock)
        _report_opaque_step(availability)

        clock.advance(_OPAQUE_GRACE_SECONDS * 2)

        assert availability.download_advancing(opaque_grace_seconds=_OPAQUE_GRACE_SECONDS) is False

    def test_no_grace_leaves_the_movement_signal_alone(self) -> None:
        """A caller that grants no grace judges on gained bytes only, as it did before the grace existed."""
        clock = _FakeClock()
        availability = ModelAvailability(clock=clock)
        _report_opaque_step(availability)

        assert availability.download_advancing() is False

    def test_a_step_that_starts_reporting_bytes_switches_to_the_movement_signal(self) -> None:
        """Once bytes arrive, the transfer is judged on movement and outlives the start-based grace."""
        clock = _FakeClock()
        availability = ModelAvailability(clock=clock)
        _report_opaque_step(availability)

        clock.advance(_OPAQUE_GRACE_SECONDS * 2)
        _report_download(availability, _SAFETY_MODEL_BYTES // 4)

        assert availability.download_advancing(opaque_grace_seconds=_OPAQUE_GRACE_SECONDS) is True
