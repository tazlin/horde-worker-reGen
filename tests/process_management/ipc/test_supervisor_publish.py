"""Tests for the dirty-gated, floor-bounded supervisor snapshot publishing cadence.

Publishing must surface display-relevant change promptly (a changed state signature publishes on the
next tick) while staying quiet when nothing changes (a periodic floor still emits a heartbeat frame so
the TUI can tell a live worker from a hung one). The cadence is isolated here from the full snapshot
build, which is exercised elsewhere.
"""

from __future__ import annotations

import time

import pytest

from horde_worker_regen.process_management.config.worker_state import PopGate
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    RECENT_EVENTS_IN_SNAPSHOT,
    ModelPoolSeatReadiness,
    PopLivenessSnapshot,
    SupervisorCommand,
    SupervisorControlMessage,
    WorkerEventKind,
)
from horde_worker_regen.process_management.jobs.pool_lanes import LaneDecision, PoolLaneState
from horde_worker_regen.process_management.jobs.text_generation_coordinator import TextJobInFlight
from horde_worker_regen.process_management.scheduling.model_demand_poller import DemandSnapshot
from horde_worker_regen.process_management.scheduling.model_pool import (
    ModelPool,
    PoolParams,
    PopLane,
    RankedCandidate,
)
from horde_worker_regen.process_management.scheduling.workload_kind import WorkloadKind
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


def test_the_event_ring_reaches_the_snapshot_oldest_first() -> None:
    """The dashboards read the worker's transitions off the snapshot, in the order they happened.

    Oldest first because the feed that renders it accumulates downward and a reconnecting frontend walks
    the ring comparing sequences; a reversed ring would make the first event it sees the newest one.
    """
    manager = make_testable_process_manager()
    manager._run_metrics.record_event(WorkerEventKind.JOB_POPPED, model="Deliberate", job_id="job-1")
    manager._run_metrics.record_event(WorkerEventKind.PRELOAD_STARTED, model="Deliberate", process_id=2)

    snapshot = manager._build_worker_state_snapshot()

    assert [event.kind for event in snapshot.recent_events] == [
        WorkerEventKind.JOB_POPPED,
        WorkerEventKind.PRELOAD_STARTED,
    ]
    assert [event.sequence for event in snapshot.recent_events] == [1, 2]
    assert snapshot.recent_events[1].payload.process_id == 2


def test_the_event_ring_on_the_snapshot_is_bounded() -> None:
    """Every snapshot carries the whole ring, so the ring's own bound is the payload's bound."""
    manager = make_testable_process_manager()
    for _ in range(RECENT_EVENTS_IN_SNAPSHOT * 2):
        manager._run_metrics.record_event(WorkerEventKind.DOWNLOAD_FINISHED, model="Deliberate")

    snapshot = manager._build_worker_state_snapshot()

    assert len(snapshot.recent_events) == RECENT_EVENTS_IN_SNAPSHOT
    assert snapshot.recent_events[-1].sequence == RECENT_EVENTS_IN_SNAPSHOT * 2


def test_a_worker_that_did_nothing_yet_carries_an_empty_ring() -> None:
    """The field is additive: a worker with no transitions yet reports the snapshot it always did."""
    manager = make_testable_process_manager()

    assert manager._build_worker_state_snapshot().recent_events == []


def test_an_operator_pause_and_resume_are_recorded_once_each() -> None:
    """A frontend may re-send a pause over an already-paused worker, and the ring records the posture.

    The local pause is the operator's, not the horde's, so it carries no workload: that is what tells a
    reader (and the feed's wording) which of the two held the worker's pops back.
    """
    manager = make_testable_process_manager()

    for _ in range(3):
        manager._apply_supervisor_command(SupervisorControlMessage(command=SupervisorCommand.PAUSE))
    for _ in range(3):
        manager._apply_supervisor_command(SupervisorControlMessage(command=SupervisorCommand.RESUME))

    events = manager._run_metrics.recent_events()
    assert [event.kind for event in events] == [WorkerEventKind.MAINTENANCE_ON, WorkerEventKind.MAINTENANCE_OFF]
    assert [event.workload for event in events] == [None, None]


def test_text_work_reaches_the_whole_worker_counters() -> None:
    """The header, the hero, the Stats tab and the native page read these, and a scribe's work is the worker's.

    ``total_num_completed_jobs`` counts terminal jobs with faults included, so the text term is submits plus
    fault reports; otherwise a faulted text job would be counted as faulted and never as done.
    """
    manager = make_testable_process_manager(scribe=True)
    coordinator = manager._text_coordinator
    assert coordinator is not None
    coordinator.num_jobs_submitted = 4
    coordinator.num_jobs_faulted = 1
    coordinator.num_jobs_popped = 6
    coordinator._in_flight["in-hand"] = TextJobInFlight(
        job_id="in-hand",
        payload={},
        time_popped=time.time(),
        model_name="koboldcpp/a-model",
    )
    coordinator._last_pop_time = time.time()

    snapshot = manager._build_worker_state_snapshot()

    assert snapshot.num_jobs_submitted == 5
    assert snapshot.num_jobs_faulted == 1
    assert snapshot.jobs_in_progress == 1
    assert snapshot.num_jobs_popped == 6, "popped is the cumulative count, not the work in hand"
    assert snapshot.jobs_in_hand == 1
    assert snapshot.seconds_since_last_pop is not None
    assert snapshot.seconds_since_last_pop < 60.0
    assert snapshot.latest_stats_sample is not None
    assert snapshot.latest_stats_sample.jobs_submitted == 5
    # The per-workload split is what it was: the headline is a sum, not a replacement.
    assert (snapshot.text_total_submitted, snapshot.text_total_faulted, snapshot.text_jobs_in_flight) == (4, 1, 1)


def test_a_worker_not_serving_text_counts_no_text_terms() -> None:
    """The text flow is registered on every worker and idle unless `scribe` is on, so every text term is zero."""
    manager = make_testable_process_manager()

    snapshot = manager._build_worker_state_snapshot()

    counters = (
        snapshot.num_jobs_submitted,
        snapshot.num_jobs_faulted,
        snapshot.num_jobs_popped,
        snapshot.jobs_in_progress,
    )
    assert counters == (0, 0, 0, 0)
    assert snapshot.seconds_since_last_pop is None


def test_alchemy_work_reaches_the_whole_worker_counters() -> None:
    """An alchemist-only worker's header read the image tracker's zero for the same reason a scribe's did.

    Alchemy forms never enter the image job tracker (it is read for gating and never written), so the
    coordinator's own counters are the only place its work exists.
    """
    manager = make_testable_process_manager(alchemist=True)
    coordinator = manager._alchemy_coordinator
    coordinator.num_forms_submitted = 7
    coordinator.num_forms_faulted = 2
    coordinator._last_pop_time = time.time()

    snapshot = manager._build_worker_state_snapshot()

    assert snapshot.num_jobs_submitted == 9
    assert snapshot.num_jobs_faulted == 2
    assert snapshot.seconds_since_last_pop is not None
    assert snapshot.seconds_since_last_pop < 60.0
    # The per-workload split is untouched, as it is for text.
    assert (snapshot.alchemy_total_submitted, snapshot.alchemy_total_faulted) == (7, 2)


def test_the_held_pop_gate_reaches_the_snapshot_with_when_it_engaged() -> None:
    """The gate holding the image popper and its engagement time travel on the wire, and clear together."""
    manager = make_testable_process_manager()
    engaged_at = time.time() - 3600.0
    manager._state.last_pop_gate = str(PopGate.NO_SAFETY_PROCESS)
    manager._state.last_pop_gate_since = engaged_at

    held = manager._build_worker_state_snapshot()
    assert held.pop_gate == "no_safety_process"
    assert held.pop_gate_since == engaged_at

    manager._state.last_pop_gate = None
    manager._state.last_pop_gate_since = time.time()
    cleared = manager._build_worker_state_snapshot()
    assert cleared.pop_gate is None
    assert cleared.pop_gate_since is None


def test_the_pop_liveness_verdict_reaches_the_snapshot_with_the_line_it_logs() -> None:
    """The sentinel's verdict on a silent image intake travels on the wire, and clears once a pop concludes."""
    manager = make_testable_process_manager()
    now = time.time()
    manager._state.last_pop_gate = str(PopGate.NO_SAFETY_PROCESS)
    manager._state.last_pop_gate_since = now - 3600.0
    manager._state.last_pop_attempt_completed_at = now - 3600.0

    errored = manager._build_worker_state_snapshot().pop_liveness
    assert errored.level == "error"
    assert errored.silent_seconds is not None and errored.silent_seconds >= 3600.0
    assert errored.detail is not None
    assert errored.detail.startswith("Pop liveness: no pop attempt has reached the horde for 36")
    assert "pops are held at gate 'no_safety_process' (held 36" in errored.detail

    manager._state.last_pop_attempt_completed_at = time.time() - 90.0
    assert manager._build_worker_state_snapshot().pop_liveness.level == "warn"

    manager._state.last_pop_attempt_completed_at = time.time()
    assert manager._build_worker_state_snapshot().pop_liveness == PopLivenessSnapshot()


def test_an_image_intake_the_worker_does_not_serve_publishes_no_concern() -> None:
    """The sentinel's exemption for a deselected image role reaches the wire, so no frontend re-applies it."""
    manager = make_testable_process_manager()
    manager._state.last_pop_gate = str(PopGate.IMAGE_GENERATION_NOT_SERVED)
    manager._state.last_pop_gate_since = time.time() - 3600.0
    manager._state.last_pop_attempt_completed_at = time.time() - 3600.0

    assert manager._build_worker_state_snapshot().pop_liveness == PopLivenessSnapshot()


def test_each_flow_reports_its_own_last_pop() -> None:
    """A text pop a second ago leaves the image flow's own figure at an hour, and the headline at the youngest."""
    manager = make_testable_process_manager(scribe=True)
    text = manager._text_coordinator
    assert text is not None
    now = time.time()
    manager._state.last_job_pop_time = now - 3600.0
    text._last_pop_time = now - 1.0

    snapshot = manager._build_worker_state_snapshot()

    per_flow = snapshot.seconds_since_last_pop_per_workload
    assert set(per_flow) == {WorkloadKind.IMAGE_GENERATION, WorkloadKind.TEXT_GENERATION}
    assert per_flow[WorkloadKind.IMAGE_GENERATION] >= 3600.0
    assert per_flow[WorkloadKind.TEXT_GENERATION] < 60.0
    assert snapshot.seconds_since_last_pop == min(per_flow.values())


def test_a_held_safety_gate_says_why_results_wait_instead_of_safety_checking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no safety process the Now line names the missing process, never a safety check that is not running."""
    manager = make_testable_process_manager()
    manager._state.last_pop_gate = str(PopGate.NO_SAFETY_PROCESS)
    manager._state.last_pop_gate_since = time.time() - 3600.0
    monkeypatch.setattr(type(manager._job_tracker), "safety_backlog_depth", property(lambda _tracker: 9))

    intent = manager._build_orchestration_intent()

    assert intent.summary == "9 results waiting: no safety process is running."
    assert "Safety checking" not in intent.summary
    assert intent.next_action == "Waiting for recovery to start a safety process."
    assert intent.why is not None and intent.why.startswith("No image jobs taken for 1h")


def test_the_run_record_counts_every_flows_delivered_work() -> None:
    """A session is judged productive by what it delivered, and a scribe delivers none of it as image jobs."""
    manager = make_testable_process_manager(scribe=True, alchemist=True)
    text = manager._text_coordinator
    assert text is not None
    text.num_jobs_submitted = 3
    text.num_jobs_faulted = 1
    manager._alchemy_coordinator.num_forms_submitted = 2

    record = manager.build_run_record()

    assert record.jobs_submitted == 6
    assert record.jobs_faulted == 1
