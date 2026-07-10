"""Arming of the idle-fill breaker (``wants_idle_fill_candidate``).

The scheduler arms the idle-fill breaker only when the queue head has sat on an idle device past
``idle_fill_threshold_seconds`` (its model still loading, nothing in progress) with a free inference sibling
to run a fill job. Because the underlying head-starvation clock is forced to zero whenever any job is in
progress, this is inert in steady state and fires only for the "stuck doing nothing but downloading" case.
"""

from __future__ import annotations

import time

from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from tests.process_management.conftest import make_mock_bridge_data, make_mock_process_info
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


def _free_sibling_map() -> ProcessMap:
    """A process map whose single inference process is idle and can take a fill job."""
    return ProcessMap({0: make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)})


def _busy_only_map() -> ProcessMap:
    """A process map whose only inference process is busy sampling, so no sibling is free for a fill."""
    return ProcessMap({0: make_mock_process_info(0, model_name="m", state=HordeProcessState.INFERENCE_STARTING)})


def _scheduler(*, process_map: ProcessMap, threshold: int | None = 5, state: WorkerState | None = None):  # noqa: ANN202
    bridge = make_mock_bridge_data()
    bridge.idle_fill_threshold_seconds = threshold
    scheduler = _make_inference_scheduler(
        state=state if state is not None else WorkerState(),
        process_map=process_map,
        bridge_data=bridge,
    )
    return scheduler, bridge


def test_arms_when_head_starved_with_free_sibling() -> None:
    """A head starved past the threshold with a free inference sibling arms the breaker."""
    scheduler, bridge = _scheduler(process_map=_free_sibling_map(), threshold=5)
    scheduler._head_starvation_since = time.time() - 30.0

    scheduler._update_idle_fill_arm(bridge)

    assert scheduler._state.wants_idle_fill_candidate is True


def test_does_not_arm_below_threshold() -> None:
    """A head starved less than the threshold does not arm (a between-jobs gap is not a stall)."""
    scheduler, bridge = _scheduler(process_map=_free_sibling_map(), threshold=5)
    scheduler._head_starvation_since = time.time() - 1.0

    scheduler._update_idle_fill_arm(bridge)

    assert scheduler._state.wants_idle_fill_candidate is False


def test_does_not_arm_in_steady_state() -> None:
    """With no head starvation clock running (a job is in progress), the breaker never arms."""
    scheduler, bridge = _scheduler(process_map=_free_sibling_map(), threshold=5)
    scheduler._head_starvation_since = 0.0  # the clock is zeroed whenever a job is in progress

    scheduler._update_idle_fill_arm(bridge)

    assert scheduler._state.wants_idle_fill_candidate is False


def test_does_not_arm_without_a_free_sibling() -> None:
    """A starved head with no idle inference sibling (nothing to run the fill on) does not arm."""
    scheduler, bridge = _scheduler(process_map=_busy_only_map(), threshold=5)
    scheduler._head_starvation_since = time.time() - 30.0

    scheduler._update_idle_fill_arm(bridge)

    assert scheduler._state.wants_idle_fill_candidate is False


def test_disabled_when_threshold_none() -> None:
    """A None threshold disables idle-fill entirely: it never arms however long the head is starved."""
    scheduler, bridge = _scheduler(process_map=_free_sibling_map(), threshold=None)
    scheduler._head_starvation_since = time.time() - 300.0

    scheduler._update_idle_fill_arm(bridge)

    assert scheduler._state.wants_idle_fill_candidate is False


def test_disarms_and_resets_ladder_when_no_longer_starved() -> None:
    """Once the head is no longer starved, an armed breaker disarms and the ladder resets to rung 0."""
    state = WorkerState()
    state.wants_idle_fill_candidate = True
    state.idle_fill_rung = 2
    scheduler, bridge = _scheduler(process_map=_free_sibling_map(), threshold=5, state=state)
    scheduler._head_starvation_since = 0.0  # head no longer starved

    scheduler._update_idle_fill_arm(bridge)

    assert scheduler._state.wants_idle_fill_candidate is False
    assert scheduler._state.idle_fill_rung == 0


def test_dispatch_clears_the_breaker_and_ladder() -> None:
    """Dispatching a job (clearing the head-starvation timer) disarms idle-fill and resets the ladder."""
    state = WorkerState()
    state.wants_idle_fill_candidate = True
    state.idle_fill_rung = 3
    scheduler, _ = _scheduler(process_map=_free_sibling_map(), threshold=5, state=state)

    scheduler._clear_head_starvation_timer()

    assert scheduler._state.wants_idle_fill_candidate is False
    assert scheduler._state.idle_fill_rung == 0
    assert scheduler._head_starvation_since == 0.0
