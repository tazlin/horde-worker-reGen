"""Tests for HordeProcessInfo state predicates."""

from __future__ import annotations

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from tests.process_management.conftest import make_mock_process_info


class TestIsProcessAlive:
    """Tests for HordeProcessInfo.is_process_alive."""

    def test_running_process_waiting_for_job_is_alive(self) -> None:
        """A process whose OS process is alive and is waiting for a job is alive."""
        proc = make_mock_process_info(0, state=HordeProcessState.WAITING_FOR_JOB)
        assert proc.is_process_alive() is True

    def test_running_process_mid_inference_is_alive(self) -> None:
        """A process actively running inference is alive."""
        proc = make_mock_process_info(0, state=HordeProcessState.INFERENCE_STARTING)
        assert proc.is_process_alive() is True

    def test_ending_process_is_not_alive(self) -> None:
        """A process that reported PROCESS_ENDING is not alive."""
        proc = make_mock_process_info(0, state=HordeProcessState.PROCESS_ENDING)
        assert proc.is_process_alive() is False

    def test_ended_process_is_not_alive(self) -> None:
        """A process that reported PROCESS_ENDED is not alive."""
        proc = make_mock_process_info(0, state=HordeProcessState.PROCESS_ENDED)
        assert proc.is_process_alive() is False

    def test_dead_os_process_is_not_alive(self) -> None:
        """A process whose underlying OS process died is not alive regardless of state."""
        proc = make_mock_process_info(0, state=HordeProcessState.WAITING_FOR_JOB)
        proc.mp_process.is_alive.return_value = False  # type: ignore[attr-defined]
        assert proc.is_process_alive() is False
