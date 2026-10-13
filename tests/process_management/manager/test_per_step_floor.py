"""Tests for the per-step floor: the fast crawl detector that forces reclaim on demand-paged sampling."""

from __future__ import annotations

from horde_worker_regen.process_management.ipc.messages import (
    HordeHeartbeatType,
    HordeProcessHeartbeatMessage,
    HordeProcessState,
)
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
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


def _seat_staged_slot_and_step(
    *,
    dispatched_at: float | None,
    first_step_at: float | None,
) -> tuple[HordeWorkerProcessManager, str]:
    """Seat an inference slot owning a job with the given stamps, then deliver one INFERENCE_STEP beat.

    The beat goes through the dispatcher's heartbeat handler, the path that stamps the first step and then calls
    the step observer. Returns the manager and the owned job's id.
    """
    pm = make_testable_process_manager()
    proc = make_mock_process_info(1, model_name="Flux.1", state=HordeProcessState.INFERENCE_PRIMED)
    proc.process_type = HordeProcessType.INFERENCE
    job = make_job_pop_response(model="Flux.1", ddim_steps=30)
    assert job.id_ is not None
    proc.record_inference_ownership(job, attempt_ordinal=1)
    proc.current_inference_started_at = dispatched_at
    proc.current_first_step_at = first_step_at
    pm._process_map[1] = proc
    pm._message_dispatcher._handle_heartbeat(_inference_step_beat())
    return pm, str(job.id_)


def _inference_step_beat() -> HordeProcessHeartbeatMessage:
    """Build one INFERENCE_STEP heartbeat from process 1."""
    return HordeProcessHeartbeatMessage(
        process_id=1,
        process_launch_identifier=0,
        info="",
        heartbeat_type=HordeHeartbeatType.INFERENCE_STEP,
        current_step=1,
        total_steps=30,
    )


def test_first_step_caches_dispatch_overhead_for_the_job() -> None:
    """The first step hands the performance model the seconds from dispatch to that step, keyed by job id."""
    pm, job_id = _seat_staged_slot_and_step(dispatched_at=100.0, first_step_at=102.5)

    assert pm._performance_model._overhead_seconds_by_job_id == {job_id: 2.5}


def test_first_step_without_dispatch_stamp_caches_nothing() -> None:
    """A slot with no dispatch instant (a pinned disaggregated sampler, a reset slot) yields no overhead."""
    pm, _job_id = _seat_staged_slot_and_step(dispatched_at=None, first_step_at=None)

    assert pm._performance_model._overhead_seconds_by_job_id == {}


def test_later_step_leaves_cached_overhead_unchanged() -> None:
    """Steps after the first do not re-read the stamps, so a moved first-step instant changes nothing."""
    pm, job_id = _seat_staged_slot_and_step(dispatched_at=100.0, first_step_at=102.5)
    proc = pm._process_map[1]
    proc.current_first_step_at = 200.0

    pm._message_dispatcher._handle_heartbeat(_inference_step_beat())

    assert proc.heartbeats_inference_steps == 2
    assert pm._performance_model._overhead_seconds_by_job_id == {job_id: 2.5}
