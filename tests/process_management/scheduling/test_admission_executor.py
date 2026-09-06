"""The executor hands the head context to the actuators a verdict's commands name."""

from __future__ import annotations

from unittest.mock import Mock

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.vram_arbiter import (
    ActuatorCommand,
    ActuatorCommandKind,
    HeadReclaimContext,
)
from tests.process_management.conftest import make_mock_bridge_data, make_mock_process_info
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_HEAD = HeadReclaimContext(model="model_a", target_process_id=0, max_resident=1)


def _scheduler():  # noqa: ANN202
    target = make_mock_process_info(0, model_name="model_a", state=HordeProcessState.WAITING_FOR_JOB)
    sibling = make_mock_process_info(1, model_name="model_b", state=HordeProcessState.WAITING_FOR_JOB)
    return _make_inference_scheduler(
        process_map=ProcessMap({0: target, 1: sibling}),
        job_tracker=JobTracker(),
        bridge_data=make_mock_bridge_data(enable_vram_budget=True),
        max_inference=2,
    )


def test_the_head_context_reaches_the_eviction_and_reduction_actuators() -> None:
    """Both head-shaped actuators receive the context the executor was handed, once each."""
    scheduler = _scheduler()
    scheduler.evict_idle_model = Mock(return_value=True)  # type: ignore[method-assign]
    scheduler.reduce_live_contexts = Mock(return_value=True)  # type: ignore[method-assign]
    commands = (
        ActuatorCommand(kind=ActuatorCommandKind.EVICT_IDLE_MODEL, device_index=None),
        ActuatorCommand(kind=ActuatorCommandKind.REDUCE_LIVE_CONTEXTS, device_index=None),
    )

    applied = scheduler.executor.execute_actuations(commands, device_index=None, for_head_of_queue=True, head=_HEAD)

    assert applied == commands
    scheduler.evict_idle_model.assert_called_once_with(None, for_head_of_queue=True, head=_HEAD)
    scheduler.reduce_live_contexts.assert_called_once_with(None, head=_HEAD)


def test_an_eviction_on_behalf_of_a_head_spares_its_slot_and_model() -> None:
    """The head's target process anchors the sweep and its model is the one room is made for."""
    scheduler = _scheduler()
    scheduler.unload_models_from_vram = Mock(return_value=True)  # type: ignore[method-assign]

    assert scheduler.evict_idle_model(None, for_head_of_queue=True, head=_HEAD) is True

    scheduler.unload_models_from_vram.assert_called_once_with(
        scheduler._process_map[0],
        under_pressure=True,
        for_head_of_queue=True,
        device_index=None,
        make_room_for_model="model_a",
    )


def test_a_reduction_without_a_head_is_a_no_op() -> None:
    """A context reduction has no depth to collapse to without the head that sized it."""
    scheduler = _scheduler()
    scale = Mock(return_value=1)
    scheduler._process_lifecycle.scale_inference_processes = scale  # type: ignore[method-assign]

    assert scheduler.reduce_live_contexts(None) is False
    scale.assert_not_called()


def test_a_reduction_for_a_vanished_head_is_a_no_op() -> None:
    """A head whose target slot has left the process map cannot be protected, so nothing is reduced."""
    scheduler = _scheduler()
    scale = Mock(return_value=1)
    scheduler._process_lifecycle.scale_inference_processes = scale  # type: ignore[method-assign]
    gone = HeadReclaimContext(model="model_a", target_process_id=9, max_resident=1)

    assert scheduler.reduce_live_contexts(None, head=gone) is False
    scale.assert_not_called()
