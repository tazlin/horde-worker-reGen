"""Tests for the extracted control-loop tick.

The tick is the body of HordeWorkerProcessManager._process_control_loop; with a
no-op sleep injected it can be driven deterministically without wall-clock delays.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState, ModelLoadState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from tests.process_management.conftest import (
    make_mock_job,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)


async def _noop_sleep(_delay: float) -> None:
    return None


def _make_tickable_manager() -> HordeWorkerProcessManager:
    """Build a testable manager whose tick can run without sleeping or printing."""
    process_manager = make_testable_process_manager()
    process_manager._sleep = _noop_sleep
    # Pretend a status message was just printed so the tick does not exercise
    # the status reporter (covered by its own tests).
    process_manager._last_status_message_time = time.time()
    return process_manager


class TestControlLoopTick:
    """Tests for _control_loop_tick."""

    async def test_idle_tick_keeps_running(self) -> None:
        """With nothing to do, a tick should complete and ask to keep looping."""
        process_manager = _make_tickable_manager()

        assert await process_manager._control_loop_tick() is True

    async def test_tick_marks_the_supervisor_alive(self) -> None:
        """Each tick stamps liveness so the TUI judges responsiveness on loop progress, not snapshots."""
        process_manager = _make_tickable_manager()
        supervisor = Mock()
        supervisor.drain_commands.return_value = []
        supervisor.send_snapshot.return_value = True
        process_manager._supervisor = supervisor  # type: ignore[assignment]

        assert await process_manager._control_loop_tick() is True
        supervisor.note_alive.assert_called()

    async def test_tick_requests_shutdown_when_ready(self) -> None:
        """When shutting down with no jobs and no processes, the tick should ask to stop."""
        process_manager = _make_tickable_manager()
        process_manager._state.shutting_down = True

        assert await process_manager._control_loop_tick() is False

    async def test_shutdown_keeps_inference_processes_up_while_queue_remains(self) -> None:
        """During a drain, inference processes are not ended while queued inference work remains.

        The popper has already stopped accepting new work, so the queued jobs are given a chance to
        finish; ending the processes out from under them would fault work that could still complete.
        """
        process_manager = _make_tickable_manager()
        process_manager._state.shutting_down = True
        process_manager._process_lifecycle.end_inference_processes = Mock()  # type: ignore[method-assign]

        await track_popped_job_async(process_manager._job_tracker, make_mock_job())
        assert len(process_manager._job_tracker.jobs_pending_inference) == 1

        assert await process_manager._control_loop_tick() is True
        process_manager._process_lifecycle.end_inference_processes.assert_not_called()

    async def test_shutdown_ends_inference_processes_once_queue_drained(self) -> None:
        """Once no inference job remains pending or in progress, the drain winds the processes down."""
        process_manager = _make_tickable_manager()
        process_manager._state.shutting_down = True
        process_manager._process_lifecycle.end_inference_processes = Mock()  # type: ignore[method-assign]

        await process_manager._control_loop_tick()
        process_manager._process_lifecycle.end_inference_processes.assert_called()

    async def test_shutdown_ends_starting_safety_process_once_safety_queue_drained(self) -> None:
        """A shutdown tick should send END_PROCESS to safety even if it is still starting."""
        process_manager = _make_tickable_manager()
        process_manager._state.shutting_down = True
        safety_proc = make_mock_process_info(
            10,
            model_name=None,
            state=HordeProcessState.PROCESS_STARTING,
            process_type=HordeProcessType.SAFETY,
        )
        process_manager._process_map.update({10: safety_proc})

        assert await process_manager._control_loop_tick() is False

        assert safety_proc.end_intended is True
        safety_proc.pipe_connection.send.assert_called_once()  # type: ignore[attr-defined]
        sent = safety_proc.pipe_connection.send.call_args.args[0]  # pyrefly: ignore
        assert sent.control_flag == HordeControlFlag.END_PROCESS

    async def test_shutdown_keeps_safety_process_up_while_alchemy_form_is_pending(self) -> None:
        """CLIP alchemy uses safety, so safety must not stop until alchemy drains too."""
        process_manager = _make_tickable_manager()
        process_manager._state.shutting_down = True
        safety_proc = make_mock_process_info(
            10,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.SAFETY,
        )
        process_manager._process_map.update({10: safety_proc})
        process_manager._alchemy_coordinator._pending_forms.append(Mock())  # pyrefly: ignore[private-usage]

        assert await process_manager._control_loop_tick() is True

        assert safety_proc.end_intended is False
        safety_proc.pipe_connection.send.assert_not_called()  # type: ignore[attr-defined]

    async def test_tick_dispatches_pending_safety_check(self) -> None:
        """A tick should send a job pending safety check to an available safety process."""
        process_manager = _make_tickable_manager()
        safety_proc = make_mock_process_info(
            10,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.SAFETY,
        )
        process_manager._process_map.update({10: safety_proc})

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

        await process_manager._job_tracker.queue_for_safety(job_info)

        assert await process_manager._control_loop_tick() is True

        assert job_info in process_manager._job_tracker.jobs_being_safety_checked
        assert job_info not in process_manager._job_tracker.jobs_pending_safety_check

    async def test_tick_starts_inference_for_pending_job(self) -> None:
        """A tick should start inference for a pending job whose model is on a free process."""
        process_manager = _make_tickable_manager()
        inf_proc = make_mock_process_info(
            0,
            model_name="stable_diffusion",
            state=HordeProcessState.WAITING_FOR_JOB,
        )
        safety_proc = make_mock_process_info(
            10,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.SAFETY,
        )
        process_manager._process_map.update({0: inf_proc, 10: safety_proc})
        # The scenario is a model already resident on the free process, so the model map carries the VRAM
        # residency: dispatch then materialises nothing new and admission treats it as the no-op it is.
        process_manager._horde_model_map.update_entry(
            "stable_diffusion",
            load_state=ModelLoadState.LOADED_IN_VRAM,
            process_id=0,
        )

        job = await track_popped_job_async(process_manager._job_tracker, make_mock_job())

        assert await process_manager._control_loop_tick() is True

        assert job in process_manager._job_tracker.jobs_in_progress
        inf_proc.pipe_connection.send.assert_called()  # type: ignore[attr-defined]


class TestPeriodicUpdateCheckLoopShutdown:
    """The update-check loop must not pin the process open across its 30-minute interval on shutdown.

    The loop waits a long interval between checks; if it waited it out in a single sleep it would keep
    ``asyncio.gather`` (and so the whole worker process) alive until the next wake-up after a stop was
    requested. With the control loop already finished, nothing would stamp liveness during that gap and
    the TUI would age the worker into a false UNRESPONSIVE long after it had really stopped. The loop must
    instead exit within a poll interval of the shutdown flag flipping.
    """

    async def test_loop_exits_promptly_when_shut_down(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With shut_down set, the loop returns within a couple of poll intervals despite the long interval."""
        process_manager = _make_tickable_manager()
        # Make the poll cadence tight so the bounded wait_for below is comfortably above it, and keep the
        # check interval at its real (long) value so a regression to a single long sleep would hang here.
        monkeypatch.setattr(process_manager, "_UPDATE_CHECK_SHUTDOWN_POLL_SECONDS", 0.01)
        monkeypatch.setattr(
            "horde_worker_regen.update_check.update_check_disabled",
            lambda: False,
        )
        # Setting the flag before the loop starts means its first poll should observe it and break; no real
        # update check should ever run (it sits behind the long interval the loop never reaches here).
        process_manager._state.shut_down = True
        unexpected_check = Mock(side_effect=AssertionError("check_for_update ran during shutdown"))
        monkeypatch.setattr("horde_worker_regen.update_check.check_for_update", unexpected_check)

        await asyncio.wait_for(process_manager._periodic_update_check_loop(), timeout=2.0)

        unexpected_check.assert_not_called()
