"""Load simulations for the dedicated post-processing lane's dispatch policy.

These tests drive the real orchestrator, job tracker, message dispatcher, recovery coordinator, and
shared reserve ledger through synthetic traffic on a virtual clock: no GPU, no child processes, no
real sleeping. The lane's per-chain durations and VRAM peaks come from the measured envelope in
:mod:`horde_worker_regen.process_management.simulation.pp_load`, so the dispatch *policy* is exercised
against realistic costs independent of the seed-based estimator's accuracy.

Invariants:

- **Conservation**: every job queued for post-processing is always accounted for (pending, being
  post-processed, or terminal). A job silently vanishing is the worst possible outcome because its
  finished inference is forfeited without any log or fault.
- **Bounded patience**: a job must reach a terminal outcome (post-processed or no-image fault)
  within a patience window even when its chain never fits the card. Parking a finished generation
  forever forfeits kudos and risks server-side timeouts.
- **No head-of-line blocking**: a chain too large for the card must not starve fittable jobs queued
  behind it.
- **Bounded reclaim requests**: deferral must not fire a VRAM-reclaim request every scheduling tick.

The dispatch policy satisfies all four: a queue scan dispatches the first fittable job ahead of an
unfittable head, an aging window submits a no-image fault once a job has been unfittable past the
admission patience, and the idle-VRAM reclaim is issued once per starvation episode rather than per tick.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest
from horde_sdk.ai_horde_api import GENERATION_STATE

from horde_worker_regen.process_management.ipc.messages import (
    HordeImageResult,
    HordePostProcessControlMessage,
    HordePostProcessResultMessage,
    HordeProcessState,
)
from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.resources.vram_arbiter import DeviceVramState, MeasuredVramSnapshot
from horde_worker_regen.process_management.simulation.pp_load import (
    AVERAGE_CARD,
    CONTENTION_SLOWDOWN_FACTOR,
    JOB_SHAPES,
    THRASH_GUARD_BAND_MB,
    THRASH_SLOWDOWN_FACTOR,
    ArrivalPattern,
    PostProcessJobClass,
    PostProcessLoadModel,
    PostProcessLoadScenario,
    canned_scenarios,
)
from horde_worker_regen.process_management.workers import post_process_orchestrator as post_process_orchestrator_module
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
)


@pytest.fixture(autouse=True)
def _measured_estimator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the orchestrator the measured cost model in place of the seed-based estimator.

    The scenarios probe the dispatch *policy*; estimator accuracy is validated separately (in
    hordelib, which owns the seeds). Patching in the measured envelope means a failure here is a
    policy failure, not a seed artifact.
    """
    model = PostProcessLoadModel()
    monkeypatch.setattr(
        post_process_orchestrator_module,
        "predict_job_post_processing_vram_mb",
        model.estimate_for_job,
    )


PATIENCE_S = 150.0
"""The longest a queued post-processing job may take to reach a terminal outcome in these scenarios.

Comfortably above the largest uncontended chain duration (about 30s) plus one orphan-recovery grace
(90s), so only genuine starvation or an unbounded thrash spiral violates it.
"""

_LANE_PROCESS_ID = 7


@dataclass
class _JobRecord:
    """One simulated job's lifecycle bookkeeping."""

    job_info: HordeJobInfo
    job_class: PostProcessJobClass
    queued_at_s: float
    terminal_at_s: float | None = None


@dataclass
class _InFlight:
    """The lane's current simulated execution."""

    job_id: object
    completes_at_s: float
    thrashed: bool


@dataclass
class _SimOutcome:
    """Aggregated results of one scenario run."""

    records: list[_JobRecord]
    reclaim_requests: int
    thrashed_dispatches: int
    end_time_s: float
    pending_at_end: int
    being_at_end: int

    @property
    def terminal(self) -> list[_JobRecord]:
        return [r for r in self.records if r.terminal_at_s is not None]

    @property
    def starved(self) -> list[_JobRecord]:
        return [r for r in self.records if r.terminal_at_s is None or r.terminal_at_s - r.queued_at_s > PATIENCE_S]


class LaneLoadSimulator:
    """Steps the real post-processing orchestration stack through a scenario on a virtual clock."""

    def __init__(self, scenario: PostProcessLoadScenario, *, vram_reserve_mb: int = 2048) -> None:
        """Wire a testable process manager with a mock lane and the scenario's card as VRAM reporter.

        Args:
            scenario: The traffic scenario to run.
            vram_reserve_mb: The configured VRAM reserve the admission gate adds to each estimate.
        """
        self.scenario = scenario
        self.model = PostProcessLoadModel()
        self._vram_reserve_mb = float(vram_reserve_mb)
        self.now_s = 0.0
        self.reclaim_requests = 0
        self.thrashed_dispatches = 0
        self.records: dict[object, _JobRecord] = {}
        self.in_flight: _InFlight | None = None
        self._pending_enqueues: list[HordeJobInfo] = []

        self.pm = make_testable_process_manager(
            enable_vram_budget=True,
            vram_reserve_mb=vram_reserve_mb,
        )
        self.lane = make_mock_process_info(
            _LANE_PROCESS_ID,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
            process_type=HordeProcessType.POST_PROCESS,
        )
        self.pm._process_map.clear()
        self.pm._process_map.update({_LANE_PROCESS_ID: self.lane})
        self.pm._recovery_coordinator._clock = lambda: self.now_s

        # The scenario card is the only VRAM reporter; deferral never finds idle inference VRAM to
        # reclaim in these runs, so a True return never masks the reclaim-request count.
        orchestrator = self.pm._post_process_orchestrator
        orchestrator._request_vram_reclaim = self._count_reclaim
        # Drive the admission-patience aging window off the virtual clock, as the recovery coordinator is.
        orchestrator._clock = lambda: self.now_s

    def _count_reclaim(self, *_args: object) -> bool:
        self.reclaim_requests += 1
        return False

    def _publish_card_state(self) -> None:
        free_mb = self.scenario.card.free_at(self.now_s)
        total_mb = float(self.scenario.card.total_vram_mb)
        self.lane.total_vram_mb = int(total_mb)
        self.lane.vram_usage_mb = int(total_mb - free_mb)
        # The lane's memory admission is the VRAM arbiter's; freeze a cycle whose measured floor reproduces the
        # card's occupancy. Folding the configured reserve into the baseline (with a zeroed noise buffer) makes
        # the arbiter's admission boundary "chain peak + reserve <= free", the exact figure the lane's headroom
        # decision uses, so the scenario's fittable/unfittable classification carries through the flipped gate.
        state = DeviceVramState(
            total_vram_mb=total_mb,
            baseline_mb=self._vram_reserve_mb,
            committed_vram_mb=total_mb - free_mb,
            planned_unmaterialized_mb=0.0,
            committed_is_stale=False,
            noise_buffer_mb=0.0,
        )
        self.pm._vram_arbiter.begin_cycle(MeasuredVramSnapshot(devices={self.lane.device_index or 0: state}))

    def _enqueue(self, job_class: PostProcessJobClass) -> None:
        shape = JOB_SHAPES[job_class]
        job = make_job_pop_response(
            width=shape.width,
            height=shape.height,
            n_iter=shape.n_iter,
            post_processing=list(shape.post_processing),
        )
        job_info = HordeJobInfo(
            sdk_api_job_info=job,
            job_image_results=[HordeImageResult(image_bytes=b"raw")] * max(1, shape.n_iter),
            state=GENERATION_STATE.ok,
            censored=False,
            time_popped=time.time(),
        )
        self.records[job.id_] = _JobRecord(job_info=job_info, job_class=job_class, queued_at_s=self.now_s)
        self._pending_enqueues.append(job_info)

    def _on_lane_dispatch(self, message: HordePostProcessControlMessage) -> None:
        record = self.records[message.job_id]
        shape = JOB_SHAPES[record.job_class]
        megapixels = (shape.width * shape.height) / 1_000_000
        cost = self.model.chain_cost(list(shape.post_processing), megapixels, max(1, shape.n_iter))
        free_mb = self.scenario.card.free_at(self.now_s)
        thrashed = (free_mb - cost.peak_vram_mb) < THRASH_GUARD_BAND_MB
        slowdown = THRASH_SLOWDOWN_FACTOR if thrashed else CONTENTION_SLOWDOWN_FACTOR
        if thrashed:
            self.thrashed_dispatches += 1
        self.in_flight = _InFlight(
            job_id=message.job_id,
            completes_at_s=self.now_s + cost.wall_s * slowdown,
            thrashed=thrashed,
        )

    async def _deliver_result(self) -> None:
        assert self.in_flight is not None
        job_record = self.records[self.in_flight.job_id]
        num_images = len(job_record.job_info.job_image_results or [])
        message = HordePostProcessResultMessage(
            process_id=_LANE_PROCESS_ID,
            process_launch_identifier=0,
            info="simulated post-processing result",
            time_elapsed=1.0,
            job_id=self.in_flight.job_id,
            job_image_results=[HordeImageResult(image_bytes=b"post-processed")] * max(1, num_images),
            state=GENERATION_STATE.ok,
        )
        self.in_flight = None
        await self.pm._message_dispatcher._handle_post_process_result(message)
        self.pm._process_map.on_process_state_change(_LANE_PROCESS_ID, HordeProcessState.WAITING_FOR_JOB)

    def _reconcile_in_flight_with_tracker(self) -> None:
        """Mirror an orphan requeue into the simulated lane: the reaped execution is abandoned.

        When the recovery coordinator requeues a job the parent no longer considers it being
        post-processed; live recovery replaces the lane process, so the simulated in-flight execution
        is dropped and the lane returns to accepting work.
        """
        if self.in_flight is None:
            return
        being_ids = {info.sdk_api_job_info.id_ for info in self.pm._job_tracker.jobs_being_post_processed}
        if self.in_flight.job_id not in being_ids:
            self.in_flight = None
            self.pm._process_map.on_process_state_change(_LANE_PROCESS_ID, HordeProcessState.WAITING_FOR_JOB)

    def _note_terminals(self) -> None:
        tracker = self.pm._job_tracker
        terminal_ids = {info.sdk_api_job_info.id_ for info in tracker.jobs_pending_safety_check}
        terminal_ids |= {info.sdk_api_job_info.id_ for info in tracker.jobs_pending_submit}
        for job_id, record in self.records.items():
            if record.terminal_at_s is None and job_id in terminal_ids:
                record.terminal_at_s = self.now_s

    async def run(self) -> _SimOutcome:
        """Run the scenario to its duration (plus a drain tail) and return the aggregated outcome."""
        arrivals = list(self.scenario.job_sequence)
        next_arrival_s = 0.0
        drain_deadline_s = self.scenario.duration_s + PATIENCE_S

        dt = 1.0
        while self.now_s < drain_deadline_s:
            self.now_s += dt
            self._publish_card_state()

            while arrivals and self.now_s >= next_arrival_s and self.now_s <= self.scenario.duration_s:
                for _ in range(min(self.scenario.arrivals.burst_size, len(arrivals))):
                    self._enqueue(arrivals.pop(0))
                next_arrival_s = self.now_s + self.scenario.arrivals.interval_s
            for job_info in self._pending_enqueues:
                await self.pm._job_tracker.queue_for_post_processing(job_info)
            self._pending_enqueues.clear()

            if self.in_flight is not None and self.now_s >= self.in_flight.completes_at_s:
                await self._deliver_result()

            send_calls_before = self.lane.pipe_connection.send.call_count
            await self.pm.start_post_processing()
            if self.lane.pipe_connection.send.call_count > send_calls_before:
                sent = self.lane.pipe_connection.send.call_args.args[0]
                assert isinstance(sent, HordePostProcessControlMessage)
                self._on_lane_dispatch(sent)

            if int(self.now_s) % 5 == 0:
                await self.pm._recovery_coordinator.reconcile_orphaned_post_process_jobs()
                self._reconcile_in_flight_with_tracker()

            self._note_terminals()

            if (
                not arrivals
                and not self._pending_enqueues
                and all(r.terminal_at_s is not None for r in self.records.values())
            ):
                break

        tracker = self.pm._job_tracker
        return _SimOutcome(
            records=list(self.records.values()),
            reclaim_requests=self.reclaim_requests,
            thrashed_dispatches=self.thrashed_dispatches,
            end_time_s=self.now_s,
            pending_at_end=len(tracker.jobs_pending_post_processing),
            being_at_end=len(tracker.jobs_being_post_processed),
        )


_SCENARIOS = {s.name: s for s in canned_scenarios()}

# Scenarios containing traffic the card structurally cannot host: at least one chain's estimated peak
# plus the fixed VRAM reserve exceeds the card's steady free figure. Under the dispatch policy these
# jobs must not park; they age out to a no-image fault within patience.
_SCENARIOS_WITH_UNFITTABLE_TRAFFIC = {
    "average_typical",
    "average_heavy",
    "low_end_light",
    "low_end_typical",
}


class TestScenarioConservation:
    """No traffic pattern may ever lose a job: queued equals pending plus in-flight plus terminal."""

    @pytest.mark.parametrize("name", sorted(_SCENARIOS))
    async def test_every_job_is_accounted_for(self, name: str) -> None:
        """Every queued job is visible in exactly one place when the scenario ends."""
        simulator = LaneLoadSimulator(_SCENARIOS[name])
        outcome = await simulator.run()

        accounted = len(outcome.terminal) + outcome.pending_at_end + outcome.being_at_end
        assert accounted == len(outcome.records), (
            f"{len(outcome.records) - accounted} job(s) vanished without a terminal outcome, a pending "
            f"queue entry, or an in-flight execution in scenario {name}"
        )


class TestScenarioPatience:
    """Every queued job reaches a terminal outcome within the patience window."""

    @pytest.mark.parametrize("name", sorted(set(_SCENARIOS) - _SCENARIOS_WITH_UNFITTABLE_TRAFFIC))
    async def test_no_starvation_where_traffic_fits(self, name: str) -> None:
        """Traffic the card can host is fully served within patience."""
        simulator = LaneLoadSimulator(_SCENARIOS[name])
        outcome = await simulator.run()

        starved = outcome.starved
        assert not starved, (
            f"{len(starved)} job(s) exceeded the {PATIENCE_S:.0f}s patience window in scenario {name}: "
            + ", ".join(f"{r.job_class} queued at {r.queued_at_s:.0f}s" for r in starved[:5])
        )

    @pytest.mark.parametrize("name", sorted(_SCENARIOS_WITH_UNFITTABLE_TRAFFIC))
    async def test_unfittable_traffic_faults_without_images_within_patience(self, name: str) -> None:
        """Traffic the card cannot host must still terminate as no-image faults within patience."""
        simulator = LaneLoadSimulator(_SCENARIOS[name])
        outcome = await simulator.run()

        assert not outcome.starved, (
            f"{len(outcome.starved)} job(s) were parked past the patience window in scenario {name} "
            "instead of faulting without images"
        )
        assert any(r.job_info.state == GENERATION_STATE.faulted for r in outcome.terminal), (
            f"scenario {name} contains unfittable PP traffic but did not produce a faulted submit"
        )


class TestHeadOfLineBlocking:
    """A chain too large for the card must not starve fittable jobs queued behind it."""

    def _hol_scenario(self) -> PostProcessLoadScenario:
        return PostProcessLoadScenario(
            name="hol_probe",
            card=AVERAGE_CARD,
            arrivals=ArrivalPattern(interval_s=1.0),
            job_sequence=[
                PostProcessJobClass.X4_FACEFIX_LARGE,
                PostProcessJobClass.X2_ONLY,
                PostProcessJobClass.X2_ONLY,
                PostProcessJobClass.STRIP_BACKGROUND,
            ],
            duration_s=240.0,
        )

    async def test_cheap_jobs_bypass_an_unfittable_head(self) -> None:
        """Cheap chains queued behind an unfittable head still complete within patience."""
        simulator = LaneLoadSimulator(self._hol_scenario())
        outcome = await simulator.run()

        cheap = [r for r in outcome.records if r.job_class is not PostProcessJobClass.X4_FACEFIX_LARGE]
        cheap_starved = [r for r in cheap if r.terminal_at_s is None or r.terminal_at_s - r.queued_at_s > PATIENCE_S]
        assert not cheap_starved, f"{len(cheap_starved)} cheap job(s) were starved behind an unfittable head chain"


class TestReclaimRequestChurn:
    """Deferral must not convert into an unbounded stream of VRAM-reclaim requests."""

    async def test_reclaim_requests_are_bounded_per_starvation_episode(self) -> None:
        """A single unfittable head causes a bounded number of reclaim requests, not one per tick."""
        simulator = LaneLoadSimulator(
            PostProcessLoadScenario(
                name="reclaim_probe",
                card=AVERAGE_CARD,
                arrivals=ArrivalPattern(interval_s=1.0),
                job_sequence=[PostProcessJobClass.X4_FACEFIX_LARGE],
                duration_s=120.0,
            ),
        )
        outcome = await simulator.run()

        assert outcome.reclaim_requests <= 5, (
            f"{outcome.reclaim_requests} reclaim requests were issued for a single starved chain"
        )


class TestThroughputSanity:
    """With ample headroom the lane serves the whole workload at its measured serial pace."""

    async def test_high_end_typical_completes_serially(self) -> None:
        """On a high-end card every typical-mix job completes, none thrash, and the lane keeps pace."""
        simulator = LaneLoadSimulator(_SCENARIOS["high_end_typical"])
        outcome = await simulator.run()

        assert len(outcome.terminal) == len(outcome.records)
        assert outcome.thrashed_dispatches == 0
        waits = [r.terminal_at_s - r.queued_at_s for r in outcome.terminal if r.terminal_at_s is not None]
        assert max(waits) <= 90.0, f"worst-case wait {max(waits):.0f}s on an uncontended high-end card"
