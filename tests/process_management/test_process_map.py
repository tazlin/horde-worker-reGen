"""Tests for ProcessMap aggregate queries and state-change handling."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from loguru import logger

from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.messages import HordeHeartbeatType, HordeProcessState
from horde_worker_regen.process_management.process_map import ProcessMap

from .conftest import make_mock_process_info


@contextmanager
def _capture_warnings() -> Iterator[list[str]]:
    """Capture loguru WARNING+ messages emitted inside the block."""
    messages: list[str] = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(handler_id)


class TestNumAvailableInferenceProcesses:
    """Tests for ProcessMap.num_available_inference_processes.

    The count must only ever include *inference* processes; an idle safety or
    download process is not available to run inference.
    """

    def test_idle_inference_process_counts(self) -> None:
        """A single idle inference process should count as available."""
        inf_proc = make_mock_process_info(0, state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: inf_proc})

        assert process_map.num_available_inference_processes() == 1

    def test_idle_safety_process_does_not_count(self) -> None:
        """An idle safety process is not an available inference process."""
        safety_proc = make_mock_process_info(
            10,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.SAFETY,
        )
        process_map = ProcessMap({10: safety_proc})

        assert process_map.num_available_inference_processes() == 0

    def test_busy_inference_process_does_not_count(self) -> None:
        """An inference process mid-inference is not available."""
        inf_proc = make_mock_process_info(0, state=HordeProcessState.INFERENCE_STARTING)
        safety_proc = make_mock_process_info(
            10,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.SAFETY,
        )
        process_map = ProcessMap({0: inf_proc, 10: safety_proc})

        assert process_map.num_available_inference_processes() == 0

    def test_mixed_map_counts_only_idle_inference(self) -> None:
        """One idle + one busy inference process + idle safety should count exactly one."""
        idle_inf = make_mock_process_info(0, state=HordeProcessState.WAITING_FOR_JOB)
        busy_inf = make_mock_process_info(1, state=HordeProcessState.PRELOADING_MODEL)
        safety_proc = make_mock_process_info(
            10,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.SAFETY,
        )
        process_map = ProcessMap({0: idle_inf, 1: busy_inf, 10: safety_proc})

        assert process_map.num_available_inference_processes() == 1


class TestProcessStateTransitions:
    """Tests for the expected-transition table in on_process_state_change."""

    def test_expected_transition_is_silent(self) -> None:
        """A normal preload sequence should not produce warnings."""
        proc = make_mock_process_info(0, state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: proc})

        with _capture_warnings() as warnings:
            process_map.on_process_state_change(0, HordeProcessState.PRELOADING_MODEL)
            process_map.on_process_state_change(0, HordeProcessState.PRELOADED_MODEL)
            process_map.on_process_state_change(0, HordeProcessState.INFERENCE_STARTING)
            process_map.on_process_state_change(0, HordeProcessState.INFERENCE_COMPLETE)

        assert warnings == []
        assert proc.last_process_state is HordeProcessState.INFERENCE_COMPLETE

    def test_unexpected_transition_warns_but_applies(self) -> None:
        """An inference-complete report from an idle process is suspicious but must still apply."""
        proc = make_mock_process_info(0, state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: proc})

        with _capture_warnings() as warnings:
            process_map.on_process_state_change(0, HordeProcessState.INFERENCE_COMPLETE)

        assert len(warnings) == 1
        assert "unexpected state transition" in warnings[0]
        assert proc.last_process_state is HordeProcessState.INFERENCE_COMPLETE

    def test_same_state_report_is_silent(self) -> None:
        """Re-reporting the current state should not warn."""
        proc = make_mock_process_info(0, state=HordeProcessState.INFERENCE_COMPLETE)
        process_map = ProcessMap({0: proc})

        with _capture_warnings() as warnings:
            process_map.on_process_state_change(0, HordeProcessState.INFERENCE_COMPLETE)

        assert warnings == []

    def test_state_started_at_changes_only_on_state_change(self) -> None:
        """Heartbeat and memory liveness must not reset the current state's duration clock."""
        proc = make_mock_process_info(0, state=HordeProcessState.DOWNLOADING_AUX_MODEL)
        process_map = ProcessMap({0: proc})
        proc.last_process_state_started_at = 123.0

        process_map.on_heartbeat(0, HordeHeartbeatType.OTHER)
        process_map.on_memory_report(0, ram_usage_bytes=1)

        assert proc.last_process_state_started_at == 123.0

        process_map.on_process_state_change(0, HordeProcessState.PRELOADING_MODEL)

        assert proc.last_process_state_started_at != 123.0

    def test_unrestricted_states_are_silent_from_anywhere(self) -> None:
        """Idle/teardown states can be entered from any state without warnings."""
        proc = make_mock_process_info(0, state=HordeProcessState.INFERENCE_STARTING)
        process_map = ProcessMap({0: proc})

        with _capture_warnings() as warnings:
            process_map.on_process_state_change(0, HordeProcessState.WAITING_FOR_JOB)
            process_map.on_process_state_change(0, HordeProcessState.PROCESS_ENDING)
            process_map.on_process_state_change(0, HordeProcessState.PROCESS_ENDED)

        assert warnings == []


class TestSamplingProgressReset:
    """Sampling progress (step/it-s) is job-scoped and must not linger past the job that set it.

    These fields are only ever set by INFERENCE_STEP heartbeats; if they are not cleared on reset, an
    idle process reports a stale mid-sampling step and it/s, which a consumer (e.g. the TUI live view)
    renders as a nonsensical "WAITING_FOR_JOB but mid-step" state.
    """

    def _drive_sampling(self, process_map: ProcessMap, process_id: int = 0) -> None:
        """Populate the sampling fields the way a real inference heartbeat would."""
        process_map.on_heartbeat(
            process_id,
            HordeHeartbeatType.INFERENCE_STEP,
            percent_complete=50,
            current_step=15,
            total_steps=30,
            iterations_per_second=8.4,
        )

    def test_reset_heartbeat_state_clears_sampling_progress(self) -> None:
        """reset_heartbeat_state must null the step/it-s fields, not just the percent/step count."""
        proc = make_mock_process_info(0, state=HordeProcessState.INFERENCE_STARTING)
        process_map = ProcessMap({0: proc})
        self._drive_sampling(process_map)
        assert proc.last_current_step == 15

        process_map.reset_heartbeat_state(0)

        assert proc.last_current_step is None
        assert proc.last_total_steps is None
        assert proc.last_iterations_per_second is None

    def test_waiting_for_job_transition_clears_sampling_progress(self) -> None:
        """Returning to WAITING_FOR_JOB (which resets heartbeat state) clears the sampling fields."""
        proc = make_mock_process_info(0, state=HordeProcessState.INFERENCE_STARTING)
        process_map = ProcessMap({0: proc})
        self._drive_sampling(process_map)

        process_map.on_process_state_change(0, HordeProcessState.INFERENCE_COMPLETE)
        process_map.on_process_state_change(0, HordeProcessState.WAITING_FOR_JOB)

        assert proc.last_current_step is None
        assert proc.last_total_steps is None
        assert proc.last_iterations_per_second is None
