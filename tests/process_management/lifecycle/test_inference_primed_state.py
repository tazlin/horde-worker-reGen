"""Behaviour of the INFERENCE_PRIMED / INFERENCE_STARTING split.

A dispatched inference slot is INFERENCE_PRIMED while it stages its pipeline and waits for the GPU sampling
lease, and only advances to INFERENCE_STARTING once it is actually in the ComfyUI denoise loop (the first
sampling step). These tests pin the two behaviours that split is for:

* the reported state distinguishes "primed / waiting for the lease" from "actually sampling"; and
* a primed slot is still counted as busy and owning its card, so nothing mistakes a staging slot for an
  idle one (which would punt the in-progress job or trip a false queue-deadlock verdict), while the
  hang watchdog still covers a slot wedged mid-staging before it ever steps.
"""

from __future__ import annotations

import time

from horde_worker_regen.process_management.ipc.messages import HordeHeartbeatType, HordeProcessState
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from tests.process_management.conftest import make_mock_process_info


def _map_with(state: HordeProcessState) -> ProcessMap:
    process_map = ProcessMap()
    process_map[0] = make_mock_process_info(0, state=state)
    return process_map


def test_first_step_upgrades_primed_to_starting() -> None:
    """The first INFERENCE_STEP moves a primed slot into the sampling state and stamps the first-step clock."""
    process_map = _map_with(HordeProcessState.INFERENCE_PRIMED)
    assert process_map[0].current_first_step_at is None

    process_map.on_heartbeat(0, HordeHeartbeatType.INFERENCE_STEP, current_step=1, total_steps=20)

    assert process_map[0].last_process_state == HordeProcessState.INFERENCE_STARTING
    assert process_map[0].current_first_step_at is not None


def test_non_step_heartbeat_does_not_upgrade_primed() -> None:
    """A liveness (non-step) heartbeat keeps the slot primed: it has not started sampling yet."""
    process_map = _map_with(HordeProcessState.INFERENCE_PRIMED)

    process_map.on_heartbeat(0, HordeHeartbeatType.OTHER)

    assert process_map[0].last_process_state == HordeProcessState.INFERENCE_PRIMED
    assert process_map[0].current_first_step_at is None


def test_primed_slot_counts_as_busy_and_running_a_job() -> None:
    """A primed slot must read as busy / running a job, never as idle.

    If a staging slot read as idle, the orphaned-job watchdog would punt its in-progress job and the
    all-cards-idle premise of the queue-deadlock recovery would falsely hold.
    """
    process_map = _map_with(HordeProcessState.INFERENCE_PRIMED)

    assert process_map[0].is_process_busy() is True
    assert process_map.has_inference_in_progress() is True
    assert process_map.num_busy_with_inference() == 1


def test_primed_slot_wedged_before_first_step_is_reaped_after_grace() -> None:
    """A slot stuck priming (silent, no step) is reaped once past the first-step grace, not before it."""
    process_map = _map_with(HordeProcessState.INFERENCE_PRIMED)
    process_map[0].last_current_step = None  # never stepped

    process_map[0].last_heartbeat_timestamp = time.time() - 30.0
    assert process_map.is_stuck_on_inference(0, inference_step_timeout=10, first_step_timeout=60) is False

    process_map[0].last_heartbeat_timestamp = time.time() - 90.0
    assert process_map.is_stuck_on_inference(0, inference_step_timeout=10, first_step_timeout=60) is True


def test_nonadvancing_step_wedge_is_sampling_only() -> None:
    """The non-advancing-step wedge is a denoise-loop condition, so it never fires on a primed slot."""
    primed = _map_with(HordeProcessState.INFERENCE_PRIMED)
    primed[0].nonadvancing_step_repeats = 999
    assert primed.is_stuck_on_nonadvancing_step(0, repeat_limit=3) is False

    sampling = _map_with(HordeProcessState.INFERENCE_STARTING)
    sampling[0].nonadvancing_step_repeats = 999
    assert sampling.is_stuck_on_nonadvancing_step(0, repeat_limit=3) is True


def test_state_labels_distinguish_priming_from_sampling() -> None:
    """The TUI label for a primed slot is not 'Sampling' (the whole point of the split)."""
    from horde_worker_regen.tui.formatters import label_state

    assert label_state("INFERENCE_PRIMED") == "Priming"
    assert label_state("INFERENCE_STARTING") == "Sampling"
