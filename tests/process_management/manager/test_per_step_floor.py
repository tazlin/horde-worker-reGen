"""Tests for the per-step floor: the fast crawl detector that forces reclaim on demand-paged sampling."""

from __future__ import annotations

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.resources.device_free_governor import GovernorState
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
)


def _seat_sampling_slot(
    *,
    governor_state: GovernorState,
    expected_sampling_seconds: float | None = 30.0,
    ddim_steps: int = 30,
    steps_seen: int = 2,
) -> tuple[object, object]:
    """Build a manager with one sampling inference slot on device 0 at the given governor state."""
    pm = make_testable_process_manager()
    proc = make_mock_process_info(1, model_name="Flux.1", state=HordeProcessState.INFERENCE_STARTING)
    proc.process_type = HordeProcessType.INFERENCE
    proc.device_index = 0
    proc.last_job_referenced = make_job_pop_response(model="Flux.1", ddim_steps=ddim_steps)
    proc.current_job_expected_sampling_seconds = expected_sampling_seconds
    proc.heartbeats_inference_steps = steps_seen
    pm._process_map[1] = proc
    pm._governor_states_by_device[0] = governor_state
    return pm, proc


def test_two_consecutive_slow_steps_under_pressure_trip_and_force_reclaim() -> None:
    """Two steps each >= 3x expected per-step time, on a PRESSURE card, trip the floor and latch reclaim."""
    # expected per-step = 30s / 30 steps = 1.0s; 4.0s observed is 4x, above the 3x floor.
    pm, proc = _seat_sampling_slot(governor_state=GovernorState.PRESSURE)
    proc.last_heartbeat_delta = 4.0

    proc.heartbeats_inference_steps = 2
    pm._observe_inference_step(1)
    assert proc.consecutive_slow_per_steps == 1
    assert proc.current_job_per_step_floor_tripped is False

    proc.heartbeats_inference_steps = 3
    pm._observe_inference_step(1)
    assert proc.current_job_per_step_floor_tripped is True
    assert pm._per_step_floor_triggers == 1
    assert pm._per_step_floor_latch_by_device.get(0) is True


def test_healthy_card_does_not_force_reclaim_or_count() -> None:
    """Slow steps on a HEALTHY card mark the crawl but do not latch reclaim or count a trigger.

    A legitimately heavy job crawling on an uncontended card is not a paging victim.
    """
    pm, proc = _seat_sampling_slot(governor_state=GovernorState.HEALTHY)
    proc.last_heartbeat_delta = 4.0
    proc.heartbeats_inference_steps = 2
    pm._observe_inference_step(1)
    proc.heartbeats_inference_steps = 3
    pm._observe_inference_step(1)

    assert proc.current_job_per_step_floor_tripped is True
    assert pm._per_step_floor_triggers == 0
    assert pm._per_step_floor_latch_by_device.get(0) is None


def test_fast_step_resets_the_crawl_signal() -> None:
    """A step back at healthy pace clears the crawl, so a slot that recovers in place is no kill candidate."""
    pm, proc = _seat_sampling_slot(governor_state=GovernorState.SATURATED)
    proc.last_heartbeat_delta = 4.0
    proc.heartbeats_inference_steps = 2
    pm._observe_inference_step(1)
    proc.heartbeats_inference_steps = 3
    pm._observe_inference_step(1)
    assert proc.current_job_per_step_floor_tripped is True

    # A healthy-paced step (well under 3x) recovers the slot.
    proc.last_heartbeat_delta = 1.0
    proc.heartbeats_inference_steps = 4
    pm._observe_inference_step(1)
    assert proc.current_job_per_step_floor_tripped is False
    assert proc.consecutive_slow_per_steps == 0


def test_first_step_is_skipped() -> None:
    """The first sampling step is skipped: its inter-beat gap includes one-time cold load/encode work."""
    pm, proc = _seat_sampling_slot(governor_state=GovernorState.SATURATED)
    proc.last_heartbeat_delta = 100.0  # a huge first-step gap (cold load), not a per-step crawl
    proc.heartbeats_inference_steps = 1
    pm._observe_inference_step(1)
    assert proc.consecutive_slow_per_steps == 0
    assert proc.current_job_per_step_floor_tripped is False


def test_missing_expected_sampling_time_is_skipped_without_guessing() -> None:
    """No expected sampling time (a cold start with no seed) suppresses the floor rather than guessing."""
    pm, proc = _seat_sampling_slot(governor_state=GovernorState.SATURATED, expected_sampling_seconds=None)
    proc.last_heartbeat_delta = 4.0
    proc.heartbeats_inference_steps = 3
    pm._observe_inference_step(1)
    assert proc.current_job_per_step_floor_tripped is False
    assert pm._per_step_floor_triggers == 0


def test_single_slow_step_does_not_trip() -> None:
    """One slow step is not enough; the floor requires two consecutive slow steps."""
    pm, proc = _seat_sampling_slot(governor_state=GovernorState.SATURATED)
    proc.last_heartbeat_delta = 4.0
    proc.heartbeats_inference_steps = 2
    pm._observe_inference_step(1)
    assert proc.consecutive_slow_per_steps == 1
    assert proc.current_job_per_step_floor_tripped is False
    assert pm._per_step_floor_triggers == 0
