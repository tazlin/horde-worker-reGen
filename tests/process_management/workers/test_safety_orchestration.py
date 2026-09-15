"""Tests for the safety dispatch contract: the parent's dispatch side and the child's reported states."""

from __future__ import annotations

import uuid
from unittest.mock import Mock

import pytest
from horde_sdk.ai_horde_api.apimodels.base import GenerationID

from horde_worker_regen.consts import AESTHETIC_METADATA_TYPE
from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeProcessHeartbeatMessage,
    HordeProcessState,
    HordeProcessStateChangeMessage,
    HordeSafetyControlMessage,
    HordeSafetyResultMessage,
)
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.simulation._dummy_images import make_dummy_png_bytes
from horde_worker_regen.process_management.workers.safety_process import HordeSafetyProcess
from tests.process_management.conftest import (
    make_mock_model_reference_record,
    make_mock_process_info,
    make_testable_process_manager,
    queue_job_for_safety_async,
)


def _sent_safety_message(safety_proc: object) -> object:
    """Return the control message the orchestrator handed to the safety process's pipe."""
    return safety_proc.pipe_connection.send.call_args.args[0]  # type: ignore[attr-defined]


def _completed_job_info() -> Mock:
    """A finished image job shaped for the safety dispatch path."""
    job = Mock()
    job.id_ = uuid.uuid4()
    job.model = "stable_diffusion"
    job.payload = Mock()
    job.payload.prompt = "test prompt"
    job.payload.use_nsfw_censor = False

    job_info = Mock()
    job_info.sdk_api_job_info = job
    job_info.job_image_results = [Mock()]
    job_info.images_bytes = [b"imgdata"]
    return job_info


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

        model_record = make_mock_model_reference_record("stable_diffusion")
        process_manager.stable_diffusion_reference = {"stable_diffusion": model_record}

        job = Mock()
        job.id_ = uuid.uuid4()
        job.model = "stable_diffusion"
        job.payload = Mock()
        job.payload.prompt = "test prompt"
        job.payload.use_nsfw_censor = False

        image_result = Mock()
        image_result.image_bytes = b"imgdata"

        job_info = Mock()
        job_info.sdk_api_job_info = job
        job_info.job_image_results = [image_result]
        job_info.images_bytes = [b"imgdata"]

        await queue_job_for_safety_async(process_manager._job_tracker, job_info)

        await process_manager.start_evaluate_safety()

        assert job_info not in process_manager._job_tracker.jobs_pending_safety_check
        assert job_info in process_manager._job_tracker.jobs_being_safety_checked

    @pytest.mark.parametrize(
        ("scoring_enabled", "server_supports", "expected_include"),
        [
            (True, True, True),
            (True, False, False),
            (False, True, False),
        ],
    )
    async def test_aesthetic_score_gated_on_scoring_flag_and_server_support(
        self,
        monkeypatch: pytest.MonkeyPatch,
        scoring_enabled: bool,
        server_supports: bool,
        expected_include: bool,
    ) -> None:
        """Aesthetic scoring is requested only when opted in AND the server advertises the metadata type.

        The server rejects a submit carrying a gen_metadata type it does not recognise, so the safety
        control message must not ask for a score until both the operator flag and the server-capability
        probe agree.
        """
        from horde_worker_regen.process_management.workers import safety_orchestrator

        process_manager = make_testable_process_manager()
        process_manager.bridge_data.aesthetic_scoring_enabled = scoring_enabled
        monkeypatch.setattr(
            safety_orchestrator,
            "server_supports_generation_metadata_type",
            lambda metadata_type: server_supports and metadata_type == AESTHETIC_METADATA_TYPE,
        )

        safety_proc = make_mock_process_info(
            10,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.SAFETY,
        )
        process_manager._process_map.clear()
        process_manager._process_map.update({10: safety_proc})

        model_record = make_mock_model_reference_record("stable_diffusion")
        process_manager.stable_diffusion_reference = {"stable_diffusion": model_record}

        job = Mock()
        job.id_ = uuid.uuid4()
        job.model = "stable_diffusion"
        job.payload = Mock()
        job.payload.prompt = "test prompt"
        job.payload.use_nsfw_censor = False

        job_info = Mock()
        job_info.sdk_api_job_info = job
        job_info.job_image_results = [Mock()]
        job_info.images_bytes = [b"imgdata"]

        await queue_job_for_safety_async(process_manager._job_tracker, job_info)

        await process_manager.start_evaluate_safety()

        assert _sent_safety_message(safety_proc).include_aesthetic_score is expected_include

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

    async def test_failed_send_to_live_process_flags_replacement(self) -> None:
        """When the send fails but the safety process is alive, flag it for replacement.

        A live process that cannot receive control messages is unrecoverable from
        the orchestrator's point of view; the lifecycle manager must replace it.
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

        model_record = make_mock_model_reference_record("stable_diffusion")
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
        job_info.images_bytes = [b"imgdata"]

        await queue_job_for_safety_async(process_manager._job_tracker, job_info)

        await process_manager.start_evaluate_safety()

        assert process_manager._process_lifecycle._safety_processes_should_be_replaced is True
        # The job was never moved into being_safety_checked (the send failed), so it
        # must remain pending so a replacement safety process can pick it up.
        assert job_info in process_manager._job_tracker.jobs_pending_safety_check


class TestOneSafetyJobInFlight:
    """The safety lane evaluates one job at a time, so dispatch may only hold one job in flight.

    Sending every finished job at once queues them in the child's pipe, where the orphan watchdog's verdict
    clock runs against jobs the child has not yet read: they time out, each requeue sends a duplicate check,
    and the duplicates deepen the very backlog making verdicts late.
    """

    @staticmethod
    def _manager_with_idle_safety() -> tuple[object, object]:
        """Return a manager whose only process is an idle safety lane, plus that lane."""
        process_manager = make_testable_process_manager()
        safety_proc = make_mock_process_info(
            10,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.SAFETY,
        )
        process_manager._process_map.clear()
        process_manager._process_map.update({10: safety_proc})
        process_manager.stable_diffusion_reference = {
            "stable_diffusion": make_mock_model_reference_record("stable_diffusion"),
        }
        return process_manager, safety_proc

    async def test_dispatch_marks_the_lane_busy(self) -> None:
        """A successful send leaves the lane reading as evaluating, not available."""
        process_manager, safety_proc = self._manager_with_idle_safety()
        job_info = _completed_job_info()
        await queue_job_for_safety_async(process_manager._job_tracker, job_info)

        await process_manager.start_evaluate_safety()

        assert safety_proc.last_process_state == HordeProcessState.JOB_RECEIVED
        assert process_manager._process_map.get_first_available_safety_process() is None

    async def test_second_job_waits_until_the_first_verdict_frees_the_lane(self) -> None:
        """A second finished job stays pending while the first is in flight, and is sent once safety idles."""
        process_manager, safety_proc = self._manager_with_idle_safety()
        first_job_info = _completed_job_info()
        second_job_info = _completed_job_info()
        await queue_job_for_safety_async(process_manager._job_tracker, first_job_info)
        await queue_job_for_safety_async(process_manager._job_tracker, second_job_info)

        await process_manager.start_evaluate_safety()

        assert first_job_info in process_manager._job_tracker.jobs_being_safety_checked
        assert second_job_info in process_manager._job_tracker.jobs_pending_safety_check
        assert safety_proc.pipe_connection.send.call_count == 1

        # A second pump while the child is still evaluating must not push the queued job onto the pipe.
        await process_manager.start_evaluate_safety()

        assert second_job_info in process_manager._job_tracker.jobs_pending_safety_check
        assert safety_proc.pipe_connection.send.call_count == 1

        # The child reports itself idle again once its verdict is away; the queued job goes next.
        process_manager._process_map.on_process_state_change(
            process_id=10,
            new_state=HordeProcessState.WAITING_FOR_JOB,
        )
        await process_manager.start_evaluate_safety()

        assert second_job_info in process_manager._job_tracker.jobs_being_safety_checked
        assert safety_proc.pipe_connection.send.call_count == 2

    async def test_a_replaced_launch_is_not_marked_busy(self) -> None:
        """The busy mark belongs to the launch that was sent to, never to whatever replaced it."""
        process_manager, _ = self._manager_with_idle_safety()
        job_info = _completed_job_info()
        await queue_job_for_safety_async(process_manager._job_tracker, job_info)

        original_send = process_manager._process_map[10].safe_send_message

        def _swap_launch_on_send(message: object) -> bool:
            """Stand a fresh launch in the same slot between the send and the parent's bookkeeping."""
            result = original_send(message)
            replacement = make_mock_process_info(
                10,
                model_name=None,
                state=HordeProcessState.WAITING_FOR_JOB,
                process_type=HordeProcessType.SAFETY,
            )
            replacement.process_launch_identifier = 1
            process_manager._process_map[10] = replacement
            return result

        process_manager._process_map[10].safe_send_message = _swap_launch_on_send  # type: ignore[method-assign]

        await process_manager.start_evaluate_safety()

        assert process_manager._process_map[10].last_process_state == HordeProcessState.WAITING_FOR_JOB


class _RecordingQueue:
    """Captures every message the safety child emits, in order."""

    def __init__(self) -> None:
        """Start with an empty message log."""
        self.messages: list[object] = []

    def put(self, message: object) -> None:
        """Record a message the child sent to the parent."""
        self.messages.append(message)

    def state_changes(self) -> list[HordeProcessState]:
        """Return the sequence of process states reported so far."""
        return [m.process_state for m in self.messages if isinstance(m, HordeProcessStateChangeMessage)]


class _CleanNsfwResult:
    """An all-clear verdict, standing in for horde_safety's NSFWResult."""

    is_nsfw = False
    is_csam = False


class _CleanNsfwChecker:
    """A checker that clears every image without loading any model."""

    def check_for_nsfw(self, **_kwargs: object) -> _CleanNsfwResult:
        """Return an all-clear verdict for the given image."""
        return _CleanNsfwResult()


class TestSafetyChildReportsItsBusyState:
    """The real child brackets an evaluation with JOB_RECEIVED and WAITING_FOR_JOB.

    The parent marks the lane busy at the send; the child's own report is what confirms it actually read the
    job, and the terminal idle report is what releases the lane for the next one. Evaluation also runs under
    a heartbeat, so a check that is merely slow (every off-GPU check is) is distinguishable from a wedged one.
    """

    @staticmethod
    def _bare_safety_process(monkeypatch: pytest.MonkeyPatch) -> tuple[HordeSafetyProcess, _RecordingQueue]:
        """Build a safety process with only the attributes the evaluation path reads, and no models."""
        queue = _RecordingQueue()
        process = HordeSafetyProcess.__new__(HordeSafetyProcess)
        process.process_id = 0
        process.process_launch_identifier = 0
        process.process_message_queue = queue  # type: ignore[assignment]
        process._dry_run_skip_safety = False
        process._nsfw_checker = _CleanNsfwChecker()  # type: ignore[assignment]
        monkeypatch.setattr(process, "_stage_clip", lambda: None)
        monkeypatch.setattr(process, "send_memory_report_message", lambda **_kwargs: None)
        monkeypatch.setattr(process, "release_allocator_cache", lambda: None)
        return process, queue

    def test_evaluation_is_bracketed_by_busy_then_idle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """JOB_RECEIVED precedes the verdict and WAITING_FOR_JOB follows it."""
        process, queue = self._bare_safety_process(monkeypatch)
        job_id = GenerationID(root=uuid.uuid4())

        process._receive_and_handle_control_message(
            HordeSafetyControlMessage(
                control_flag=HordeControlFlag.EVALUATE_SAFETY,
                job_id=job_id,
                prompt="a test prompt",
                censor_nsfw=False,
                sfw_worker=False,
                images_bytes=[make_dummy_png_bytes()],
                horde_model_info=None,
            ),
        )

        states = queue.state_changes()
        assert states[0] == HordeProcessState.JOB_RECEIVED
        assert states[-1] == HordeProcessState.WAITING_FOR_JOB

        results = [m for m in queue.messages if isinstance(m, HordeSafetyResultMessage)]
        assert len(results) == 1
        assert results[0].job_id == job_id
        assert not results[0].safety_evaluations[0].failed

        # The verdict is reported between the two states, so the parent never sees an idle lane holding one.
        state_positions = [i for i, m in enumerate(queue.messages) if isinstance(m, HordeProcessStateChangeMessage)]
        result_position = next(i for i, m in enumerate(queue.messages) if isinstance(m, HordeSafetyResultMessage))
        assert state_positions[0] < result_position < state_positions[-1]

    def test_evaluation_heartbeats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The evaluation body beats, so a slow check is not silence the process watchdog must guess at."""
        process, queue = self._bare_safety_process(monkeypatch)

        process._receive_and_handle_control_message(
            HordeSafetyControlMessage(
                control_flag=HordeControlFlag.EVALUATE_SAFETY,
                job_id=GenerationID(root=uuid.uuid4()),
                prompt="a test prompt",
                censor_nsfw=False,
                sfw_worker=False,
                images_bytes=[make_dummy_png_bytes()],
                horde_model_info=None,
            ),
        )

        assert [m for m in queue.messages if isinstance(m, HordeProcessHeartbeatMessage)]
