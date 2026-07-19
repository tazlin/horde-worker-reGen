"""The post-processing fault breaker: repeated unhostable post-processing peaks disable the feature.

A post-processing peak that cannot be hosted faults the job and the horde reissues it, but a worker that
keeps faulting trips the horde's forced-maintenance (the spiral this guards against). These pin the worker's
self-protective breaker: a rolling-window counter fed by both fault sources (the planner's unhostable-peak
faults and watchdog-reaped post-processing stalls), a trip when the count *exceeds* the threshold, and the
recovery behaviour that replaces the old restart-only latch: the fault-count breaker auto-recovers once the
parent measures the card's free VRAM back above the post-processing peak, a proactive gate withholds
advertising below that peak before any fault, and a one-shot idle-resident reclaim frees the card. A host
without an NVML reading keeps the session latch, and a structural whole-card disable never auto-recovers.
"""

from __future__ import annotations

import time

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.process_manager import (
    POST_PROCESSING_GATE_OPEN_REQUIREMENT_MB,
    POST_PROCESSING_HEADROOM_REQUIREMENT_MB,
    HordeWorkerProcessManager,
)
from tests.process_management.conftest import make_mock_process_info, make_testable_process_manager


class TestPostProcessingFaultCounter:
    """The rolling-window counter the breaker reads, fed by both fault sources."""

    def test_counts_within_window_and_excludes_older(self) -> None:
        """Each recorded fault is counted within the window; a zero-width window counts none."""
        job_tracker = JobTracker()
        assert job_tracker.count_recent_post_processing_faults(600) == 0

        job_tracker.note_post_processing_overcommit_fault()
        job_tracker.note_post_processing_overcommit_fault()

        assert job_tracker.count_recent_post_processing_faults(600) == 2
        # A zero-width window excludes faults recorded a moment ago (the boundary is strict-enough to prune).
        assert job_tracker.count_recent_post_processing_faults(-1) == 0


class TestPostProcessingFaultBreaker:
    """The trip/latch behaviour the control loop drives via ``_apply_post_processing_fault_breaker``."""

    def test_trips_only_after_exceeding_threshold_and_latches(self) -> None:
        """The breaker tolerates exactly the threshold and trips on the next fault, then latches."""
        manager = make_testable_process_manager(
            post_processing_fault_breaker_enabled=True,
            post_processing_fault_threshold=4,
            post_processing_fault_window_seconds=1800,
        )

        # Exactly the threshold is tolerated (the trip is strictly greater-than).
        for _ in range(4):
            manager._job_tracker.note_post_processing_overcommit_fault()
        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is False

        # One more crosses it: the breaker trips and stamps the time.
        manager._job_tracker.note_post_processing_overcommit_fault()
        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is True
        assert manager._state.post_processing_breaker_tripped_at > 0

    def test_session_latched_never_auto_clears(self) -> None:
        """Once tripped the latch persists across later checks (and a soft reset, which never rebuilds state)."""
        manager = make_testable_process_manager(
            post_processing_fault_threshold=1,
            post_processing_fault_window_seconds=600,
        )
        for _ in range(2):
            manager._job_tracker.note_post_processing_overcommit_fault()
        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is True

        # A subsequent check (even with the faults aged out of a tiny window) leaves the latch set: the
        # over-commit is structural, so the breaker clears only on a process restart, not on its own.
        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is True

    def test_disabled_flag_prevents_trip(self) -> None:
        """With the breaker disabled, no number of faults latches it off."""
        manager = make_testable_process_manager(
            post_processing_fault_breaker_enabled=False,
            post_processing_fault_threshold=1,
            post_processing_fault_window_seconds=1800,
        )
        for _ in range(10):
            manager._job_tracker.note_post_processing_overcommit_fault()
        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is False


class TestPostProcessingBreakerAutoRecovery:
    """The fault-count breaker re-enables once the window elapses and a driven card recovers headroom."""

    @staticmethod
    def _trip_breaker(manager: HordeWorkerProcessManager) -> None:
        """Drive the breaker to its latched state with two faults at the currently-mocked wall-clock time."""
        for _ in range(2):
            manager._job_tracker.note_post_processing_overcommit_fault()
        manager._apply_post_processing_fault_breaker()

    def test_stays_disabled_until_window_elapsed_and_headroom_recovered(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Tripped plus window elapsed but insufficient headroom stays disabled; recovered headroom re-enables."""
        manager = make_testable_process_manager(
            post_processing_fault_threshold=1,
            post_processing_fault_window_seconds=60,
            device_free_mb=2000.0,
        )
        base_time = 10_000.0
        monkeypatch.setattr(time, "time", lambda: base_time)
        self._trip_breaker(manager)
        assert manager._state.post_processing_disabled_by_breaker is True
        assert manager._state.post_processing_breaker_auto_recoverable is True

        # The fault window elapses (faults age out) but the card is still below the requirement: stays disabled.
        monkeypatch.setattr(time, "time", lambda: base_time + 61.0)
        manager._last_device_free_mb_by_device[0] = 2000.0
        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is True

        # The card's measured free VRAM recovers above the requirement: the breaker re-enables and clears state.
        manager._last_device_free_mb_by_device[0] = POST_PROCESSING_GATE_OPEN_REQUIREMENT_MB + 500.0
        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is False
        assert manager._state.post_processing_breaker_auto_recoverable is False

    def test_recovered_headroom_does_not_re_enable_while_faults_remain_in_window(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ample headroom cannot re-enable while an over-commit fault is still counted within the window."""
        manager = make_testable_process_manager(
            post_processing_fault_threshold=1,
            post_processing_fault_window_seconds=600,
            device_free_mb=POST_PROCESSING_GATE_OPEN_REQUIREMENT_MB + 500.0,
        )
        base_time = 10_000.0
        monkeypatch.setattr(time, "time", lambda: base_time)
        self._trip_breaker(manager)
        assert manager._state.post_processing_disabled_by_breaker is True

        # Same window, faults not yet aged out: even with ample headroom the latch holds.
        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is True

    def test_no_nvml_reading_keeps_session_latch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A host without a device-free reading cannot evaluate headroom, so the latch stays until restart."""
        manager = make_testable_process_manager(
            post_processing_fault_threshold=1,
            post_processing_fault_window_seconds=60,
            device_free_mb=None,
        )
        base_time = 10_000.0
        monkeypatch.setattr(time, "time", lambda: base_time)
        self._trip_breaker(manager)
        assert manager._state.post_processing_disabled_by_breaker is True

        # The window elapses with no NVML reading: headroom is unmeasurable, so recovery never fires.
        monkeypatch.setattr(time, "time", lambda: base_time + 61.0)
        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is True

    def test_structural_whole_card_latch_never_auto_recovers(self) -> None:
        """A latch set without the auto-recoverable marker (the whole-card conflict) stays disabled."""
        manager = make_testable_process_manager(
            device_free_mb=POST_PROCESSING_GATE_OPEN_REQUIREMENT_MB + 1000.0,
        )
        # The scheduler's whole-card disable sets the latch but leaves ``auto_recoverable`` false (default).
        manager._state.post_processing_disabled_by_breaker = True
        manager._state.post_processing_breaker_auto_recoverable = False

        manager._apply_post_processing_fault_breaker()
        assert manager._state.post_processing_disabled_by_breaker is True


class TestPostProcessingHeadroomGate:
    """The proactive gate withholds advertising below the headroom requirement, before any fault occurs."""

    def test_withholds_below_requirement_and_reopens_after_sustained_headroom(self) -> None:
        """Below the requirement the gate closes; recovery re-opens only after the sustain window holds."""
        manager = make_testable_process_manager(
            device_free_mb=POST_PROCESSING_HEADROOM_REQUIREMENT_MB - 100.0,
        )
        manager._apply_post_processing_headroom_gate()
        assert manager._state.post_processing_withheld_for_headroom is True

        # Recovery above the requirement does not open the gate instantly: the proof window must elapse.
        manager._last_device_free_mb_by_device[0] = POST_PROCESSING_GATE_OPEN_REQUIREMENT_MB + 100.0
        manager._apply_post_processing_headroom_gate()
        assert manager._state.post_processing_withheld_for_headroom is True

        manager._pp_headroom_sustained_since = (
            time.monotonic() - manager._POST_PROCESSING_HEADROOM_SUSTAIN_SECONDS - 1.0
        )
        manager._apply_post_processing_headroom_gate()
        assert manager._state.post_processing_withheld_for_headroom is False

    def test_boot_race_holds_gate_until_sustained(self) -> None:
        """A spacious-looking booting card starts withheld: headroom must hold for the sustain window first.

        This is the live regression: an empty card at boot passes a spot check, a wave of post-processing
        jobs is admitted, and every one faults once the heavy residents land.
        """
        manager = make_testable_process_manager(
            device_free_mb=POST_PROCESSING_GATE_OPEN_REQUIREMENT_MB + 5000.0,
        )
        manager._apply_post_processing_headroom_gate()
        assert manager._state.post_processing_withheld_for_headroom is True

        # A dip below the requirement mid-proof restarts the window rather than crediting prior headroom.
        manager._last_device_free_mb_by_device[0] = POST_PROCESSING_HEADROOM_REQUIREMENT_MB - 1.0
        manager._apply_post_processing_headroom_gate()
        assert manager._pp_headroom_sustained_since is None
        assert manager._state.post_processing_withheld_for_headroom is True

    def test_no_nvml_reading_never_gates(self) -> None:
        """A host without a device-free reading never proactively gates on headroom."""
        manager = make_testable_process_manager(device_free_mb=None)
        manager._apply_post_processing_headroom_gate()
        assert manager._state.post_processing_withheld_for_headroom is False

    def test_disabled_master_switch_never_gates(self) -> None:
        """With the self-protection switch off, the proactive gate does not withhold advertising."""
        manager = make_testable_process_manager(
            post_processing_fault_breaker_enabled=False,
            device_free_mb=POST_PROCESSING_HEADROOM_REQUIREMENT_MB - 500.0,
        )
        manager._apply_post_processing_headroom_gate()
        assert manager._state.post_processing_withheld_for_headroom is False


class TestPostProcessingHeadroomReclaim:
    """The one-shot idle-resident reclaim: one targeted unload per window, only when an idle resident exists."""

    @staticmethod
    def _add_idle_resident(
        manager: HordeWorkerProcessManager,
        *,
        process_id: int = 0,
        model_name: str = "sdxl_resident",
    ) -> HordeProcessInfo:
        """Seat one idle inference process holding ``model_name`` in the manager's process map and return it."""
        resident = make_mock_process_info(
            process_id,
            model_name=model_name,
            state=HordeProcessState.WAITING_FOR_JOB,
        )
        manager._process_map[process_id] = resident
        return resident

    def test_fault_with_idle_resident_issues_one_unload(self) -> None:
        """A recorded over-commit fault with an idle resident issues exactly one targeted VRAM unload."""
        manager = make_testable_process_manager(post_processing_fault_window_seconds=1800)
        resident = self._add_idle_resident(manager)

        manager._job_tracker.note_post_processing_overcommit_fault()
        manager._apply_post_processing_fault_breaker()

        resident.pipe_connection.send.assert_called_once()
        sent_message = resident.pipe_connection.send.call_args.args[0]
        assert sent_message.control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
        assert sent_message.horde_model_name == "sdxl_resident"

    def test_reclaim_is_throttled_within_window(self) -> None:
        """A second over-commit fault within the same window issues no further unload."""
        manager = make_testable_process_manager(post_processing_fault_window_seconds=1800)
        resident = self._add_idle_resident(manager)

        manager._job_tracker.note_post_processing_overcommit_fault()
        manager._apply_post_processing_fault_breaker()
        assert resident.pipe_connection.send.call_count == 1

        # Clear the resident's unload flag so only the window throttle (not the already-unloading guard) can
        # block a second unload; the throttle must hold it to a single reclaim per window.
        resident.last_control_flag = None
        manager._job_tracker.note_post_processing_overcommit_fault()
        manager._apply_post_processing_fault_breaker()
        assert resident.pipe_connection.send.call_count == 1

    def test_busy_only_resident_triggers_no_reclaim(self) -> None:
        """A fault with only a busy resident (no idle resident to yield) issues no unload."""
        manager = make_testable_process_manager(post_processing_fault_window_seconds=1800)
        busy_resident = make_mock_process_info(
            0,
            model_name="sdxl_resident",
            state=HordeProcessState.INFERENCE_PRIMED,
        )
        manager._process_map[0] = busy_resident

        manager._job_tracker.note_post_processing_overcommit_fault()
        manager._apply_post_processing_fault_breaker()

        busy_resident.pipe_connection.send.assert_not_called()

    def test_proactive_gate_closure_attempts_reclaim(self) -> None:
        """Closing the proactive gate (headroom shortage, no fault) also attempts the one-shot reclaim."""
        manager = make_testable_process_manager(
            post_processing_fault_window_seconds=1800,
            device_free_mb=POST_PROCESSING_HEADROOM_REQUIREMENT_MB - 200.0,
        )
        resident = self._add_idle_resident(manager)

        manager._apply_post_processing_headroom_gate()

        assert manager._state.post_processing_withheld_for_headroom is True
        resident.pipe_connection.send.assert_called_once()
        sent_message = resident.pipe_connection.send.call_args.args[0]
        assert sent_message.control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
