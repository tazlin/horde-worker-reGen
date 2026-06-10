"""Tests for safety evaluation orchestration in HordeWorkerProcessManager."""

from __future__ import annotations

import uuid
from unittest.mock import Mock

from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.messages import HordeProcessState

from .conftest import make_mock_process_info, make_testable_process_manager, queue_job_for_safety_async


class TestStartEvaluateSafety:
    """Tests for start_evaluate_safety."""

    async def test_no_pending_safety_checks_returns_early(self) -> None:
        """If there are no jobs pending safety checks, the method should return early without doing anything."""
        process_manager = make_testable_process_manager()
        await process_manager.start_evaluate_safety()

    async def test_no_safety_process_returns_early(self) -> None:
        """If there are jobs pending safety checks but no safety process, the method should return early."""
        process_manager = make_testable_process_manager()
        job_info = Mock()
        await queue_job_for_safety_async(process_manager._job_tracker, job_info)

        await process_manager.start_evaluate_safety()

        assert job_info in process_manager._job_tracker.jobs_pending_safety_check
        assert job_info not in process_manager._job_tracker.jobs_being_safety_checked

    async def test_successful_safety_eval_moves_job(self) -> None:
        """If a safety evaluation is successful, the job should be moved from pending to being checked."""
        process_manager = make_testable_process_manager()
        safety_proc = make_mock_process_info(
            10,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.SAFETY,
        )
        process_manager._process_map.clear()
        process_manager._process_map.update({10: safety_proc})

        model_record = Mock()
        model_record.model_dump.return_value = {"name": "test"}
        process_manager.stable_diffusion_reference = {"stable_diffusion": model_record}

        job = Mock()
        job.id_ = uuid.uuid4()
        job.model = "stable_diffusion"
        job.payload = Mock()
        job.payload.prompt = "test prompt"
        job.payload.use_nsfw_censor = False

        image_result = Mock()
        image_result.image_base64 = "base64data"

        job_info = Mock()
        job_info.sdk_api_job_info = job
        job_info.job_image_results = [image_result]
        job_info.images_base64 = ["base64data"]

        await queue_job_for_safety_async(process_manager._job_tracker, job_info)

        await process_manager.start_evaluate_safety()

        assert job_info not in process_manager._job_tracker.jobs_pending_safety_check
        assert job_info in process_manager._job_tracker.jobs_being_safety_checked

    async def test_critical_fault_missing_image_results(self) -> None:
        """If job_image_results is None, it should be cleaned up and not cause a crash.

        - It should:
            - log an error about missing image results.
            - remove the job from pending safety checks to avoid blocking the queue.
        """
        process_manager = make_testable_process_manager()
        safety_proc = make_mock_process_info(
            10,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.SAFETY,
        )
        process_manager._process_map.clear()
        process_manager._process_map.update({10: safety_proc})

        process_manager.stable_diffusion_reference = {}

        job = Mock()
        job.id_ = "fault-test"
        job.model = "stable_diffusion"
        job.payload = Mock()
        job.payload.prompt = "prompt"

        job_info = Mock()
        job_info.sdk_api_job_info = job
        job_info.job_image_results = None

        await queue_job_for_safety_async(process_manager._job_tracker, job_info)

        await process_manager.start_evaluate_safety()

        assert job_info not in process_manager._job_tracker.jobs_pending_safety_check

    async def test_critical_fault_missing_job_id(self) -> None:
        """If job id is None, it should be cleaned up and not cause a crash.

        - It should:
            - log an error about missing job id.
            - remove the job from pending safety checks to avoid blocking the queue.
        """
        process_manager = make_testable_process_manager()
        safety_proc = make_mock_process_info(
            10,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.SAFETY,
        )
        process_manager._process_map.clear()
        process_manager._process_map.update({10: safety_proc})

        process_manager.stable_diffusion_reference = {}

        job = Mock()
        job.id_ = None
        job.model = "stable_diffusion"
        job.payload = Mock()
        job.payload.prompt = "prompt"

        job_info = Mock()
        job_info.sdk_api_job_info = job
        job_info.job_image_results = [Mock()]

        await queue_job_for_safety_async(process_manager._job_tracker, job_info)

        await process_manager.start_evaluate_safety()

        assert job_info not in process_manager._job_tracker.jobs_pending_safety_check

    async def test_sd_reference_none_raises(self) -> None:
        """Test that if stable_diffusion_reference is None, a RuntimeError is raised."""
        import pytest

        process_manager = make_testable_process_manager()
        process_manager.stable_diffusion_reference = None

        safety_proc = make_mock_process_info(
            10,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.SAFETY,
        )
        process_manager._process_map.clear()
        process_manager._process_map.update({10: safety_proc})

        job_info = Mock()
        await queue_job_for_safety_async(process_manager._job_tracker, job_info)

        with pytest.raises(RuntimeError, match="stable diffusion reference accessed before it was loaded"):
            await process_manager.start_evaluate_safety()

    async def test_failed_send_returns_early_when_process_not_alive(self) -> None:
        """When send fails and is_process_alive returns False, the method returns early.

        Note: HordeProcessInfo.is_process_alive has a bug where it always returns False
        due to `or HordeProcessState.PROCESS_ENDED` being always truthy. This test
        documents the current (buggy) behavior.
        """
        process_manager = make_testable_process_manager()
        safety_proc = make_mock_process_info(
            10,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.SAFETY,
            safe_send_returns=False,
        )
        process_manager._process_map.clear()
        process_manager._process_map.update({10: safety_proc})

        model_record = Mock()
        model_record.model_dump.return_value = {"name": "test"}
        process_manager.stable_diffusion_reference = {"stable_diffusion": model_record}

        job = Mock()
        job.id_ = uuid.uuid4()
        job.model = "stable_diffusion"
        job.payload = Mock()
        job.payload.prompt = "prompt"
        job.payload.use_nsfw_censor = False

        job_info = Mock()
        job_info.sdk_api_job_info = job
        job_info.job_image_results = [Mock()]
        job_info.images_base64 = ["base64data"]

        await queue_job_for_safety_async(process_manager._job_tracker, job_info)

        await process_manager.start_evaluate_safety()

        # is_process_alive() always returns False due to the operator precedence bug,
        # so the code returns early without setting replacement flag
        assert process_manager._process_lifecycle._safety_processes_should_be_replaced is False
