"""Tests for the dirty-gated, floor-bounded supervisor snapshot publishing cadence.

Publishing must surface display-relevant change promptly (a changed state signature publishes on the
next tick) while staying quiet when nothing changes (a periodic floor still emits a heartbeat frame so
the TUI can tell a live worker from a hung one). The cadence is isolated here from the full snapshot
build, which is exercised elsewhere.
"""

from __future__ import annotations

import time

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.ipc.supervisor_channel import ModelPoolSeatReadiness
from horde_worker_regen.process_management.jobs.pool_lanes import LaneDecision, PoolLaneState
from horde_worker_regen.process_management.scheduling.model_demand_poller import DemandSnapshot
from horde_worker_regen.process_management.scheduling.model_pool import (
    ModelPool,
    PoolParams,
    PopLane,
    RankedCandidate,
)
from tests.process_management.conftest import make_mock_process_info, make_testable_process_manager


class _Recorder:
    """A stand-in supervisor channel that counts the snapshots it is handed."""

    def __init__(self) -> None:
        self.count = 0
        self.closed = False

    def send_snapshot(self, snapshot: object) -> bool:
        """Record a send and report success (the worker keeps the channel)."""
        self.count += 1
        return True


def test_snapshot_reflects_runtime_cpu_only_torch_build() -> None:
    """A child's CPU-only-build report drops image generation from the snapshot and flags the reason.

    This is what makes the dashboard reshape to alchemist-only (and the popper stop) for a CPU torch build
    whose install sentinel was never set.
    """
    manager = make_testable_process_manager()
    manager._runtime_config.bridge_data.dreamer = True
    manager._runtime_config.bridge_data.alchemist = True

    before = manager._build_worker_state_snapshot()
    assert "image_generation" in before.enabled_workloads
    assert before.torch_build_cpu_only is False

    manager._state.torch_build_cpu_only = True
    manager._state.torch_build_cpu_only_reason = "Installed PyTorch is a CPU-only build; image generation disabled."

    after = manager._build_worker_state_snapshot()
    assert "image_generation" not in after.enabled_workloads
    assert "alchemy" in after.enabled_workloads
    assert after.torch_build_cpu_only is True
    assert after.torch_build_cpu_only_reason


def test_snapshot_omits_model_pool_when_disabled() -> None:
    """A worker with the fixed pool off leaves ``model_pool`` unset, so old supervisors are unaffected."""
    manager = make_testable_process_manager()
    assert manager.bridge_data.model_pool.enabled is False

    snapshot = manager._build_worker_state_snapshot()

    assert snapshot.model_pool is None


def test_snapshot_populates_model_pool_when_enabled() -> None:
    """An enabled pool ships seats/bench/lane/demand-age/budget with monotonic stamps resolved to ages.

    The pool engine keeps its timing in a monotonic clock, so the population must convert each stamp to an
    age or countdown at snapshot time; this drives a real seat plus a real bench entry and asserts no
    resolved age is ever negative.
    """
    manager = make_testable_process_manager()
    manager.bridge_data.model_pool.enabled = True
    manager.bridge_data.model_pool.download_budget_gb = 5.0
    manager._model_pool_download_bytes_charged = 4096

    pool = ModelPool(
        PoolParams(
            seat_count=2,
            ranker_enabled=True,
            min_dwell_minutes=1.0,
            zero_fulfillment_demotion_minutes=1.0,
            rotation_minutes=1000.0,
        ),
    )
    seat_time = time.monotonic()
    pool.tick(
        seat_time,
        ranked=[
            RankedCandidate(name="Deliberate", score=10.0, on_disk=True),
            RankedCandidate(name="AlbedoBase XL", score=5.0, on_disk=True),
        ],
        demand_is_stale=False,
    )
    # Keep Deliberate earning its seat while AlbedoBase XL goes without fulfillment and demotes to the bench.
    pool.on_pop_outcome(
        lane=PopLane.FIXED,
        advertised=frozenset({"Deliberate", "AlbedoBase XL"}),
        popped_model="Deliberate",
        now=seat_time + 190.0,
    )
    pool.tick(
        seat_time + 200.0,
        ranked=[RankedCandidate(name="Deliberate", score=10.0, on_disk=True)],
        demand_is_stale=False,
    )
    manager._model_pool = pool

    manager._job_popper._pool_last_routed_lane = PopLane.FIXED
    manager._job_popper._pool_last_fixed_seat_count = 1
    manager._process_map[7] = make_mock_process_info(
        7,
        model_name="Deliberate",
        state=HordeProcessState.WAITING_FOR_JOB,
        device_index=1,
    )
    manager._model_demand_poller.seed(DemandSnapshot(records={}, fetched_at=time.monotonic()))

    snapshot = manager._build_worker_state_snapshot()

    pool_snapshot = snapshot.model_pool
    assert pool_snapshot is not None
    assert pool_snapshot.enabled is True
    assert len(pool_snapshot.seats) == 2
    assert pool_snapshot.seats[0].model == "Deliberate"
    assert pool_snapshot.seats[0].source == "RANKER"
    assert pool_snapshot.seats[0].readiness is ModelPoolSeatReadiness.RESIDENT
    assert pool_snapshot.seats[0].resident_process_ids == [7]
    assert pool_snapshot.seats[0].resident_device_indices == [1]
    assert pool_snapshot.seats[1].model is None
    assert pool_snapshot.seats[1].readiness is ModelPoolSeatReadiness.EMPTY
    assert any(bench_row.model == "AlbedoBase XL" for bench_row in pool_snapshot.bench)
    assert pool_snapshot.current_lane == "FIXED"
    assert pool_snapshot.last_fixed_seat_count == 1
    assert pool_snapshot.demand_age_seconds is not None
    assert pool_snapshot.demand_age_seconds >= 0.0
    assert pool_snapshot.download_budget_gb == 5.0
    assert pool_snapshot.download_bytes_charged == 4096

    for seat_row in pool_snapshot.seats:
        assert seat_row.dwell_seconds is None or seat_row.dwell_seconds >= 0.0
        assert seat_row.last_fulfilled_age_seconds is None or seat_row.last_fulfilled_age_seconds >= 0.0
        assert seat_row.rescue_expires_in_seconds is None or seat_row.rescue_expires_in_seconds >= 0.0
    for bench_row in pool_snapshot.bench:
        assert bench_row.cooldown_remaining_seconds >= 0.0


def test_snapshot_projects_pool_lane_tally() -> None:
    """The cumulative per-lane pop/fulfillment tally the popper accrues is projected onto the snapshot.

    Drives the popper's real outcome-reporting path (the same call ``api_job_pop`` makes) with a fixed-lane
    fulfilled pop, a fixed-lane empty pop, and a free-lane empty pop, then reads the projected counts.
    """
    manager = make_testable_process_manager()
    manager.bridge_data.model_pool.enabled = True

    fixed = LaneDecision(
        lane=PopLane.FIXED,
        advertised=frozenset({"Deliberate"}),
        next_state=PoolLaneState(),
        reason="fixed",
    )
    free = LaneDecision(
        lane=PopLane.FREE,
        advertised=frozenset({"AlbedoBase XL"}),
        next_state=PoolLaneState(),
        reason="free",
    )
    manager._job_popper._report_pool_pop_outcome(fixed, popped_model="Deliberate")
    manager._job_popper._report_pool_pop_outcome(fixed, popped_model=None)
    manager._job_popper._report_pool_pop_outcome(free, popped_model=None)

    pool_snapshot = manager._build_worker_state_snapshot().model_pool

    assert pool_snapshot is not None
    assert pool_snapshot.fixed_pops == 2
    assert pool_snapshot.fixed_fulfilled == 1
    assert pool_snapshot.fixed_resident_hits == 0
    assert pool_snapshot.free_pops == 1
    assert pool_snapshot.free_fulfilled == 0
    assert pool_snapshot.free_resident_hits == 0


def test_publish_is_dirty_gated_with_a_floor() -> None:
    """Snapshots publish on signature change and at the floor, and are suppressed when unchanged."""
    manager = make_testable_process_manager()
    recorder = _Recorder()
    manager._supervisor = recorder  # type: ignore[assignment]
    manager._build_worker_state_snapshot = lambda: object()  # type: ignore[assignment,method-assign,return-value]
    manager._process_map[0] = make_mock_process_info(0, state=HordeProcessState.WAITING_FOR_JOB)

    # First publish: the signature changes from None to a value, so a frame goes out.
    manager._publish_supervisor_snapshot()
    assert recorder.count == 1

    # No change and within the floor: suppressed.
    manager._publish_supervisor_snapshot()
    assert recorder.count == 1

    # A state change makes the signature differ: it publishes on the next tick (~2 Hz responsiveness).
    manager._process_map[0].last_process_state = HordeProcessState.INFERENCE_STARTING
    manager._publish_supervisor_snapshot()
    assert recorder.count == 2

    # Still no further change, still within the floor: suppressed.
    manager._publish_supervisor_snapshot()
    assert recorder.count == 2

    # Simulate the floor elapsing: a heartbeat frame goes out even with no state change.
    manager._last_supervisor_publish_time -= manager._supervisor_publish_floor_interval + 1.0
    manager._publish_supervisor_snapshot()
    assert recorder.count == 3


def test_headless_publish_still_builds_the_snapshot_at_the_floor() -> None:
    """With no supervisor attached the snapshot is still built at the floor cadence, and never sent.

    The build is what records the periodic stats sample the exported stats file carries, so a headless run
    (harness, soak, a worker whose supervisor went away) keeps its offline duty spine.
    """
    manager = make_testable_process_manager()
    manager._supervisor = None
    builds = 0

    def _build() -> object:
        nonlocal builds
        builds += 1
        return object()

    manager._build_worker_state_snapshot = _build  # type: ignore[assignment,method-assign]
    manager._process_map[0] = make_mock_process_info(0, state=HordeProcessState.WAITING_FOR_JOB)

    manager._publish_supervisor_snapshot()
    assert builds == 1

    # Within the floor: no rebuild, and a state change alone does not force one (nothing is watching).
    manager._process_map[0].last_process_state = HordeProcessState.INFERENCE_STARTING
    manager._publish_supervisor_snapshot()
    assert builds == 1

    manager._last_supervisor_publish_time -= manager._supervisor_publish_floor_interval + 1.0
    manager._publish_supervisor_snapshot()
    assert builds == 2
