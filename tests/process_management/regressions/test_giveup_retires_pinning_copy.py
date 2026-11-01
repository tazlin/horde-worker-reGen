"""The save-our-ship give-up retires the by-name copy that pinned each held job it faults.

Dispatch places a job on the lane holding its model by name. Faulting a held head leaves that copy on its lane,
so the next job of the same model binds to the same copy and is held the same way. The give-up unloads the copy
from RAM under the dispatch gate's reroute keep-rules, so the next job is placed by preload. A faulted job whose
dispatch was never held keeps its lane's copy.
"""

from __future__ import annotations

import time
from unittest.mock import Mock

from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.process_management.ipc.messages import (
    HordeControlFlag,
    HordeControlMessage,
    HordeProcessState,
    ModelLoadState,
)
from horde_worker_regen.process_management.lifecycle.process_info import HordeProcessInfo
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from horde_worker_regen.process_management.scheduling.ledgers.dispatch_holds import HoldRemedy
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)

_WEDGED_MODEL = "Deliberate"
_BUSY_LANE_MODEL = "AlbedoBase XL (SDXL)"
_SIBLING_MODEL = "AbsoluteReality"


def _sent_ram_unload(process_info: HordeProcessInfo) -> bool:
    """Whether the parent sent ``process_info`` an unload-from-RAM control message."""
    send = process_info.pipe_connection.send
    assert isinstance(send, Mock)
    return any(
        isinstance(message := call.args[0], HordeControlMessage)
        and message.control_flag is HordeControlFlag.UNLOAD_MODELS_FROM_RAM
        for call in send.call_args_list
    )


def _seat_copy(pm: HordeWorkerProcessManager, process_id: int, model: str) -> HordeProcessInfo:
    """Hold ``model`` by name in RAM on an idle lane, as a preload leaves it."""
    lane = make_mock_process_info(process_id, model_name=model, state=HordeProcessState.PRELOADED_MODEL)
    pm._process_map[process_id] = lane
    pm._horde_model_map.update_entry(model, load_state=ModelLoadState.LOADED_IN_RAM, process_id=process_id)
    return lane


def _hold_dispatch(pm: HordeWorkerProcessManager, job: ImageGenerateJobPopResponse) -> None:
    """Record ``job``'s dispatch as held with no remedy left, as the dispatch gate does on a terminal DENY."""
    pm._inference_scheduler.dispatch_holds.note_hold(
        str(job.id_),
        reclaim_applied=False,
        room_inputs=None,
        remedy=HoldRemedy.NONE,
    )


def _latch_queue_deadlock(pm: HordeWorkerProcessManager, model: str) -> None:
    """Latch the dispatcher's queue deadlock on ``model`` past the structural-wedge horizon."""
    dispatcher = pm._message_dispatcher
    dispatcher._in_queue_deadlock = True
    dispatcher._queue_deadlock_model = model
    dispatcher._last_queue_deadlock_detected_time = time.time() - 60


class TestGiveUpRetiresThePinningCopy:
    """A give-up unloads the copy that pinned a held job, and leaves other copies alone."""

    async def test_only_the_copy_that_pinned_a_held_job_on_an_idle_lane_is_unloaded(self) -> None:
        """The held head's idle copy gets an unload from RAM. A held job's busy lane and a sibling's copy do not.

        Three pending jobs are faulted. The head and a job for the busy lane's model were held. The busy lane owns a
        dispatched job it has not started sampling, so the pool has no inference in progress and the wedge stands,
        and the lane keeps its copy under the keep-rules. The sibling's job was never held, so nothing showed that
        its idle copy could not serve, and the copy stays.
        """
        pm = make_testable_process_manager()
        idle_copy = _seat_copy(pm, 0, _WEDGED_MODEL)
        busy_copy = _seat_copy(pm, 1, _BUSY_LANE_MODEL)
        busy_copy.record_inference_ownership(make_job_pop_response(model=_BUSY_LANE_MODEL), attempt_ordinal=1)
        sibling_copy = _seat_copy(pm, 2, _SIBLING_MODEL)
        head = await track_popped_job_async(pm._job_tracker, make_job_pop_response(model=_WEDGED_MODEL))
        queued = await track_popped_job_async(pm._job_tracker, make_job_pop_response(model=_BUSY_LANE_MODEL))
        sibling = await track_popped_job_async(pm._job_tracker, make_job_pop_response(model=_SIBLING_MODEL))
        _hold_dispatch(pm, head)
        _hold_dispatch(pm, queued)
        _latch_queue_deadlock(pm, _WEDGED_MODEL)
        coordinator = pm._recovery_coordinator
        assert coordinator.structural_queue_wedge_active() is True, "precondition: the queue must read as wedged"

        coordinator.give_up_on_wedged_jobs()

        pending = list(pm._job_tracker.jobs_pending_inference)
        assert all(job not in pending for job in (head, queued, sibling)), (
            "precondition: the give-up must fault all three jobs"
        )
        assert _sent_ram_unload(idle_copy), "the held head's copy was left on its idle lane"
        assert idle_copy.loaded_horde_model_name is None, "the idle lane still records the held head's model"
        assert not _sent_ram_unload(busy_copy), "the give-up unloaded the copy on a lane that owns a job"
        assert busy_copy.loaded_horde_model_name == _BUSY_LANE_MODEL
        assert not _sent_ram_unload(sibling_copy), "the give-up unloaded the copy of a sibling that was never held"
        assert sibling_copy.loaded_horde_model_name == _SIBLING_MODEL
