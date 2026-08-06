"""Tests for HordeProcessInfo state predicates."""

from __future__ import annotations

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from tests.process_management.conftest import make_job_pop_response, make_mock_process_info


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


class TestInferenceOwnership:
    """Tests for execution ownership independent of resident-model attribution."""

    def test_delayed_result_cannot_retire_newer_dispatch(self) -> None:
        """Retirement is conditional on the job owned by the current process launch."""
        proc = make_mock_process_info(0, state=HordeProcessState.INFERENCE_STARTING)
        older_job = make_job_pop_response()
        newer_job = make_job_pop_response()
        proc.record_inference_ownership(newer_job, attempt_ordinal=2)

        assert proc.retire_inference_ownership(older_job) is False
        assert proc.current_inference_job() == newer_job

    def test_retired_execution_remains_resident_model_attribution(self) -> None:
        """A completed job stops owning execution without erasing what its slot prepared."""
        proc = make_mock_process_info(0, state=HordeProcessState.INFERENCE_STARTING)
        job = make_job_pop_response()
        proc.record_inference_ownership(job, attempt_ordinal=1)

        assert proc.retire_inference_ownership(job) is True
        assert proc.current_inference_job() is None
        assert proc.last_job_referenced == job
