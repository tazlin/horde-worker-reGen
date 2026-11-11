"""Tests for the dirty-gated, floor-bounded supervisor snapshot publishing cadence.

Publishing must surface display-relevant change promptly (a changed state signature publishes on the
next tick) while staying quiet when nothing changes (a periodic floor still emits a heartbeat frame so
the TUI can tell a live worker from a hung one). The cadence is isolated here from the full snapshot
build, which is exercised elsewhere.
"""

from __future__ import annotations

import time

import pytest

import horde_worker_regen.process_management.process_manager as process_manager_module
from horde_worker_regen.process_management.config.worker_state import PopGate
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    EARNING_ROWS_IN_SNAPSHOT,
    RECENT_EVENTS_IN_SNAPSHOT,
    ModelPoolSeatReadiness,
    PopLivenessSnapshot,
    PreloadAdmissionSnapshot,
    StartupContextReading,
    SupervisorCommand,
    SupervisorControlMessage,
    WorkerEventKind,
)
from horde_worker_regen.process_management.jobs.pool_lanes import LaneDecision, PoolLaneState
from horde_worker_regen.process_management.jobs.text_generation_coordinator import TextJobInFlight
from horde_worker_regen.process_management.process_manager import (
    HordeWorkerProcessManager,
    SystemResources,
    startup_context_readings,
)
from horde_worker_regen.process_management.resources.device_info import TorchDeviceInfo, TorchDeviceMap
from horde_worker_regen.process_management.scheduling.card_earning import (
    CARD_EARNING_BUCKET_SECONDS,
    CardEarningAccumulator,
    CardTick,
)
from horde_worker_regen.process_management.scheduling.governance.preload_admission import AdmissionDecision
from horde_worker_regen.process_management.scheduling.governance.whole_card import (
    POP_CLAIM_VALUE_MIN_JOBS,
    PopClaimValueVerdict,
    two_sided_95_band_width,
)
from horde_worker_regen.process_management.scheduling.model_demand_poller import DemandSnapshot
from horde_worker_regen.process_management.scheduling.model_pool import (
    ModelPool,
    PoolParams,
    PopLane,
    RankedCandidate,
)
from horde_worker_regen.process_management.scheduling.workload_kind import WorkloadKind
from horde_worker_regen.utils.accelerator_probe import (
    ContextReadingVerdict,
    ProbedAccelerator,
    SelectedContextReading,
    select_context_readings,
)
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)


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
    """Every snapshot includes the whole ring, so the ring's own bound is the payload's bound."""
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


async def test_the_published_preload_decision_is_the_heads() -> None:
    """The published preload decision is the head's, although a later job's decision was recorded after it."""
    manager = make_testable_process_manager()
    head = await track_popped_job_async(manager._job_tracker, make_job_pop_response("head-model"))
    later = await track_popped_job_async(manager._job_tracker, make_job_pop_response("later-model"))
    scheduler = manager._inference_scheduler
    scheduler._record_preload_admission(AdmissionDecision.DEFER_BUDGET, job=head, reason="the head does not fit")
    scheduler._record_preload_admission(AdmissionDecision.ADMIT, job=later, reason="fits on the other card")

    preload = manager._build_worker_state_snapshot().scheduling_governance.preload

    assert preload.model == "head-model"
    assert preload.decision == str(AdmissionDecision.DEFER_BUDGET)
    assert preload.reason == "the head does not fit"


def test_no_published_preload_decision_when_the_head_was_not_decided() -> None:
    """With an empty queue the published preload decision is empty, whatever the record for an earlier job holds."""
    manager = make_testable_process_manager()
    manager._inference_scheduler._record_preload_admission(
        AdmissionDecision.DEFER_BUDGET,
        job=make_job_pop_response("departed-model"),
        reason="a job no longer queued",
    )

    assert manager._build_worker_state_snapshot().scheduling_governance.preload == PreloadAdmissionSnapshot()


def _snapshot_fingerprint(manager: HordeWorkerProcessManager) -> str:
    return manager._build_worker_state_snapshot().config.config_fingerprint


def test_the_snapshot_fingerprint_is_the_live_settings_fingerprint() -> None:
    """The dashboard and the earning store read one value."""
    manager = make_testable_process_manager()

    fingerprint = _snapshot_fingerprint(manager)

    assert fingerprint
    assert fingerprint == manager._capacity_workload_fingerprint()


def test_a_concurrency_change_changes_the_fingerprint() -> None:
    """A supervisor concurrency command moves the live cap between reloads, and that counts as a change."""
    manager = make_testable_process_manager(max_threads=2)
    before = _snapshot_fingerprint(manager)

    manager._apply_set_concurrency(target_threads=1, target_processes=None)

    assert manager.max_concurrent_inference_processes == 1
    assert _snapshot_fingerprint(manager) != before


def test_a_workload_setting_changes_the_fingerprint() -> None:
    """A listed workload flag is part of the fingerprint."""
    manager = make_testable_process_manager()
    before = _snapshot_fingerprint(manager)

    manager._runtime_config.bridge_data.allow_controlnet = not manager.bridge_data.allow_controlnet

    assert _snapshot_fingerprint(manager) != before


def test_a_setting_outside_the_list_does_not_change_the_fingerprint() -> None:
    """The NSFW flag is not a capacity or workload setting."""
    manager = make_testable_process_manager()
    before = _snapshot_fingerprint(manager)

    manager._runtime_config.bridge_data.nsfw = not manager.bridge_data.nsfw

    assert _snapshot_fingerprint(manager) == before


def test_suspended_lora_work_does_not_change_the_fingerprint() -> None:
    """A full LoRA disk turns off LoRA advertising without a settings change, so the earning window stays."""
    manager = make_testable_process_manager(allow_lora=True)
    before = manager._build_worker_state_snapshot().config

    manager._state.lora_disk_exhausted = True

    after = manager._build_worker_state_snapshot().config
    assert before.effective_allow_lora is True
    assert after.effective_allow_lora is False
    assert after.config_fingerprint == before.config_fingerprint


EARNING_BUCKET_START = 1000.0 * CARD_EARNING_BUCKET_SECONDS
"""A tick time at a bucket boundary, so each worked bucket stays inside one bucket."""


def _earning_accumulator() -> CardEarningAccumulator:
    return CardEarningAccumulator(max_tick_seconds=1000.0, clock=lambda: EARNING_BUCKET_START)


def _two_bucket_earning_window() -> CardEarningAccumulator:
    """Card 0 over two buckets, with model A at 1.75 kudos per second and a standard error of 0.375.

    Bucket one: A and B share 20 s, A paid 10 and B paid 5. Bucket two: five lanes share 50 s, A has 30 s and B
    20 s, A paid 60 and B paid 30. Card 0 then idles 10 s in bucket two, so its wall is 80 s and 70 s are
    attributed. The rest of A's card has 10 s paid 5 and 30 s paid 30, a rate of 35 / 40 = 0.875, residuals of
    -3.75 and 3.75, and a standard error of sqrt(2 x 28.125) / 40 = 0.1875. A whole-card pop claim for A stands
    through bucket two's 50 s of jobs.
    """
    accumulator = _earning_accumulator()
    second_bucket = EARNING_BUCKET_START + CARD_EARNING_BUCKET_SECONDS
    accumulator.observe(EARNING_BUCKET_START, {})
    accumulator.observe(EARNING_BUCKET_START + 20.0, {0: CardTick(in_flight_models=("A", "B"))})
    accumulator.credit_finished_job(device_index=0, model="A", kudos=10.0)
    accumulator.credit_finished_job(device_index=0, model="B", kudos=5.0)
    accumulator.observe(second_bucket, {})
    accumulator.observe(
        second_bucket + 50.0,
        {0: CardTick(in_flight_models=("A", "A", "A", "B", "B"), claimed_model="A")},
    )
    accumulator.credit_finished_job(device_index=0, model="A", kudos=60.0)
    accumulator.credit_finished_job(device_index=0, model="B", kudos=30.0)
    accumulator.observe(second_bucket + 60.0, {0: CardTick()})
    return accumulator


class TestCardEarningSnapshot:
    """The snapshot includes the earning figures ready to show, computed on the worker."""

    def test_an_empty_window_gives_empty_lists(self) -> None:
        """With an empty window the rows and cards are empty, and the start and restart are None."""
        manager = make_testable_process_manager()
        manager._card_earning = _earning_accumulator()

        earning = manager._build_worker_state_snapshot().card_earning

        assert earning.rows == []
        assert earning.rows_omitted == 0
        assert earning.cards == []
        assert earning.window_started_at is None
        assert earning.last_restart is None

    def test_the_worked_window_gives_its_rows_cards_and_start(self) -> None:
        """Rows are ranked by attributed seconds, and the card row splits the wall."""
        earning = HordeWorkerProcessManager._card_earning_snapshot(_two_bucket_earning_window())

        assert earning.verdict_min_jobs == POP_CLAIM_VALUE_MIN_JOBS
        assert [(row.device_index, row.model) for row in earning.rows] == [(0, "A"), (0, "B")]
        row_a = earning.rows[0]
        assert row_a.model_kudos_per_second == pytest.approx(1.75)
        assert row_a.rest_kudos_per_second == pytest.approx(0.875)
        assert row_a.jobs == 2
        assert row_a.model_buckets == 2
        assert row_a.rest_buckets == 2
        assert row_a.attributed_card_seconds == pytest.approx(40.0)
        assert row_a.claim_seconds == pytest.approx(50.0)
        assert row_a.kudos_gap_to_rest == pytest.approx(-35.0)
        assert row_a.verdict == PopClaimValueVerdict.UNMEASURED.value
        assert earning.rows[1].claim_seconds == 0.0
        assert len(earning.cards) == 1
        assert earning.cards[0].wall_seconds == pytest.approx(80.0)
        assert earning.cards[0].attributed_seconds == pytest.approx(70.0)
        assert earning.cards[0].unattributed_seconds == pytest.approx(10.0)
        assert earning.window_started_at == EARNING_BUCKET_START

    def test_band_ends_are_the_rate_plus_or_minus_the_width_in_standard_errors(self) -> None:
        """Two buckets give Student's t for one degree of freedom as the width."""
        row_a = HordeWorkerProcessManager._card_earning_snapshot(_two_bucket_earning_window()).rows[0]
        width = two_sided_95_band_width(2)

        assert row_a.model_band_low == pytest.approx(1.75 - width * 0.375)
        assert row_a.model_band_high == pytest.approx(1.75 + width * 0.375)
        assert row_a.rest_band_low == pytest.approx(0.875 - width * 0.1875)
        assert row_a.rest_band_high == pytest.approx(0.875 + width * 0.1875)

    def test_a_rate_without_a_standard_error_has_no_band(self) -> None:
        """One bucket does not give a spread, so the band ends are None while the rate is present."""
        accumulator = _earning_accumulator()
        accumulator.observe(EARNING_BUCKET_START, {})
        accumulator.observe(EARNING_BUCKET_START + 10.0, {0: CardTick(in_flight_models=("A",))})
        accumulator.credit_finished_job(device_index=0, model="A", kudos=5.0)

        row = HordeWorkerProcessManager._card_earning_snapshot(accumulator).rows[0]

        assert row.model_kudos_per_second == pytest.approx(0.5)
        assert row.model_band_low is None
        assert row.model_band_high is None
        assert row.rest_kudos_per_second is None
        assert row.rest_band_low is None

    def test_the_rows_are_bounded_keeping_the_most_attributed_seconds(self) -> None:
        """Past the bound the snapshot keeps the pairs with the most card time and counts the rest."""
        accumulator = _earning_accumulator()
        pair_count = EARNING_ROWS_IN_SNAPSHOT + 5
        now = EARNING_BUCKET_START
        accumulator.observe(now, {})
        for model_number in range(pair_count):
            now += 1.0 + model_number
            accumulator.observe(now, {0: CardTick(in_flight_models=(f"model-{model_number:02d}",))})

        earning = HordeWorkerProcessManager._card_earning_snapshot(accumulator)

        assert len(earning.rows) == EARNING_ROWS_IN_SNAPSHOT
        assert earning.rows_omitted == 5
        assert earning.rows[0].model == f"model-{pair_count - 1:02d}"
        assert {row.model for row in earning.rows} == {f"model-{number:02d}" for number in range(5, pair_count)}

    def test_a_restart_during_a_run_names_the_changed_setting(self) -> None:
        """A supervisor concurrency command moves the live cap, and the next control-tick comparison names it."""
        manager = make_testable_process_manager(max_threads=2)
        manager._card_earning.restart_window_if_changed(manager._capacity_workload_settings())
        assert manager._build_worker_state_snapshot().card_earning.last_restart is None

        manager._apply_set_concurrency(target_threads=1, target_processes=None)
        manager._restart_card_earning_window_if_changed()

        restart = manager._build_worker_state_snapshot().card_earning.last_restart
        assert restart is not None
        assert restart.changed_settings == ["max_threads"]

    def test_the_figures_are_reused_until_the_refresh_interval_passes(self) -> None:
        """The window read is too slow for every snapshot, so a recent reading is reused and an old one is not."""
        manager = make_testable_process_manager()
        manager._card_earning = _two_bucket_earning_window()
        first = manager._build_worker_state_snapshot().card_earning
        manager._card_earning.credit_finished_job(device_index=0, model="A", kudos=5.0)

        reused = manager._build_worker_state_snapshot().card_earning
        manager._card_earning_snapshot_built_at -= manager._CARD_EARNING_SNAPSHOT_REFRESH_SECONDS
        refreshed = manager._build_worker_state_snapshot().card_earning

        assert reused.rows[0].jobs == first.rows[0].jobs
        assert refreshed.rows[0].jobs == first.rows[0].jobs + 1


def _probe_reading(
    index: int, *, first_mb: int, marginal_mb: int, baseline_mb: int | None = 3600
) -> ProbedAccelerator:
    """Return one card's probe reading built from its two figures, with the raw readings behind them."""
    after_context_mb = (baseline_mb or 0) + first_mb
    return ProbedAccelerator(
        index=index,
        name=f"GPU{index}",
        total_vram_mb=24564,
        device_baseline_mb=baseline_mb,
        context_device_used_mb=after_context_mb,
        after_holder_device_used_mb=after_context_mb + marginal_mb if marginal_mb > 0 else None,
        runtime_overhead_mb=first_mb,
        marginal_overhead_mb=marginal_mb,
    )


def _resources_with_readings(runs: list[list[ProbedAccelerator]]) -> SystemResources:
    """Return system resources whose cards and context selection come from canned probe runs."""
    selections = select_context_readings(runs)
    device_map = TorchDeviceMap(
        root={
            selection.accelerator.index: TorchDeviceInfo(
                device_name=selection.accelerator.name,
                device_index=selection.accelerator.index,
                total_memory=selection.accelerator.total_vram_mb * 1024 * 1024,
            )
            for selection in selections
        },
    )
    return SystemResources(total_ram_bytes=32 * 1024**3, device_map=device_map, context_readings=selections)


class TestStartupContextReadings:
    """The snapshot includes each card's startup context measurement, built once for the session."""

    def test_a_consistent_card_gives_its_figures_and_raw_readings(self) -> None:
        """An agreeing first run gives the chosen figures, the three readings, one run and no run list."""
        manager = make_testable_process_manager(
            system_resources=_resources_with_readings([[_probe_reading(0, first_mb=487, marginal_mb=488)]]),
        )
        manager._runtime_config.bridge_data.vram_per_process_overhead_mb = 0

        (reading,) = manager._build_worker_state_snapshot().startup_context_readings

        assert reading.device_index == 0
        assert reading.verdict == ContextReadingVerdict.CONSISTENT.value
        assert (reading.chosen.first_context_mb, reading.chosen.marginal_mb) == (487, 488)
        assert (reading.chosen.baseline_mb, reading.chosen.after_context_mb, reading.chosen.after_holder_mb) == (
            3600,
            4087,
            4575,
        )
        assert (reading.probe_runs, reading.chosen_run) == (1, 1)
        assert reading.first_context_override_mb is None
        assert reading.runs == []

    def test_an_inconsistent_card_lists_every_run(self) -> None:
        """No agreeing run gives the median first-context run as chosen, and every run's figures in order."""
        manager = make_testable_process_manager(
            system_resources=_resources_with_readings(
                [
                    [_probe_reading(0, first_mb=1161, marginal_mb=426)],
                    [_probe_reading(0, first_mb=900, marginal_mb=300)],
                    [_probe_reading(0, first_mb=1000, marginal_mb=200)],
                ],
            ),
        )
        manager._runtime_config.bridge_data.vram_per_process_overhead_mb = 0

        (reading,) = manager._build_worker_state_snapshot().startup_context_readings

        assert reading.verdict == ContextReadingVerdict.INCONSISTENT.value
        assert (reading.chosen.first_context_mb, reading.chosen.marginal_mb) == (1000, 200)
        assert (reading.probe_runs, reading.chosen_run) == (3, 3)
        assert [(run.first_context_mb, run.marginal_mb) for run in reading.runs] == [
            (1161, 426),
            (900, 300),
            (1000, 200),
        ]

    def test_an_unchecked_card_keeps_its_first_run(self) -> None:
        """An unmeasured marginal leaves the first run unchecked, with no holder reading and no run list."""
        manager = make_testable_process_manager(
            system_resources=_resources_with_readings([[_probe_reading(0, first_mb=1161, marginal_mb=0)]]),
        )
        manager._runtime_config.bridge_data.vram_per_process_overhead_mb = 0

        (reading,) = manager._build_worker_state_snapshot().startup_context_readings

        assert reading.verdict == ContextReadingVerdict.UNCHECKED.value
        assert reading.chosen.first_context_mb == 1161
        assert reading.chosen.after_holder_mb is None
        assert reading.runs == []

    def test_the_override_setting_is_reported_and_follows_a_reload(self) -> None:
        """A positive ``vram_per_process_overhead_mb`` is reported, and a changed value rebuilds the readings."""
        manager = make_testable_process_manager(
            system_resources=_resources_with_readings([[_probe_reading(0, first_mb=487, marginal_mb=488)]]),
        )
        bridge_data = manager._runtime_config.bridge_data
        bridge_data.vram_per_process_overhead_mb = 600

        (with_override,) = manager._build_worker_state_snapshot().startup_context_readings
        bridge_data.vram_per_process_overhead_mb = 0
        (without_override,) = manager._build_worker_state_snapshot().startup_context_readings

        assert with_override.first_context_override_mb == 600
        assert without_override.first_context_override_mb is None

    def test_the_readings_are_built_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The measurement is fixed for the session, so later snapshots do not build it again."""
        manager = make_testable_process_manager(
            system_resources=_resources_with_readings([[_probe_reading(0, first_mb=487, marginal_mb=488)]]),
        )
        manager._runtime_config.bridge_data.vram_per_process_overhead_mb = 0
        builds: list[int] = []

        def _counting_build(
            selections: list[SelectedContextReading],
            *,
            first_context_override_mb: int | None,
        ) -> list[StartupContextReading]:
            builds.append(len(builds))
            return startup_context_readings(selections, first_context_override_mb=first_context_override_mb)

        monkeypatch.setattr(process_manager_module, "startup_context_readings", _counting_build)

        first = manager._build_worker_state_snapshot().startup_context_readings
        second = manager._build_worker_state_snapshot().startup_context_readings

        assert len(builds) == 1
        assert second == first

    def test_resources_without_a_probe_give_no_readings(self) -> None:
        """Injected resources with no selection give an empty list."""
        manager = make_testable_process_manager()

        assert manager._build_worker_state_snapshot().startup_context_readings == []

    def test_two_cards_are_listed_by_device_index(self) -> None:
        """Each card has its own reading, in device order."""
        selections = select_context_readings(
            [[_probe_reading(1, first_mb=470, marginal_mb=480), _probe_reading(0, first_mb=487, marginal_mb=488)]],
        )
        readings = startup_context_readings(selections, first_context_override_mb=None)

        assert [reading.device_index for reading in readings] == [0, 1]
