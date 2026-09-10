"""Behavioural tests for the shared per-job lifecycle model over synthetic worker logs.

The line builders here are the single home for the worker's job-shaped log formats in tests; the detector
tests import them so a reworded emit is fixed in one place. Each builder reproduces the literal line the
registry records, so a builder that drifts from the worker fails ``test_log_signatures.py`` first.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from horde_worker_regen.analysis.bundle import LogBundle
from horde_worker_regen.analysis.correlate import build_session_context
from horde_worker_regen.analysis.job_lifecycle import (
    JobLifecycleModel,
    LaneRole,
    build_job_lifecycle,
    job_lifecycle_for,
    percentile,
)
from horde_worker_regen.analysis.log_ingest import read_records
from horde_worker_regen.analysis.sessions import segment_sessions

_DAY = "2026-09-09"
_STARTUP = "Setting up logger for main process"


def startup_line(ts: str = "01:00:00.000") -> str:
    """The main-process logger-setup line that opens a session (the segmentation boundary)."""
    return f"{_DAY} {ts} | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}"


def _line(ts: str, level: str, location: str, message: str) -> str:
    """One loguru-formatted orchestrator record."""
    return f"{_DAY} {ts} | {level:<8} | {location} - {message}"


def popped_job(ts: str, *, job_id: str, model: str, emps: int = 35, batch: int = 1) -> str:
    """job_popper.api_job_pop: a job accepted from the horde."""
    return _line(
        ts,
        "INFO",
        "horde_worker_regen.process_management.jobs.job_popper:api_job_pop:2378",
        f"Popped job {job_id} ({emps} eMPS) (model: {model}, batch: {batch}, loras: False, post_processing: False)",
    )


def job_queue(ts: str, *entries: tuple[str, str]) -> str:
    """job_popper._enqueue_popped_job: the pending queue after a pop."""
    body = ", ".join(f"<{job_id[:8]}: {model}>" for job_id, model in entries)
    return _line(
        ts,
        "INFO",
        "horde_worker_regen.process_management.jobs.job_popper:_enqueue_popped_job:1864",
        f"Job queue: {body}",
    )


def inference_dispatched(ts: str, *, job_id: str, process: int) -> str:
    """inference_scheduler.start_inference: a job seated on an inference lane."""
    return _line(
        ts,
        "INFO",
        "horde_worker_regen.process_management.scheduling.inference_scheduler:start_inference:10400",
        f"Starting inference for job {job_id[:8]} on process {process}",
    )


def batch_ids(ts: str, *job_ids: str) -> str:
    """inference_scheduler._log_job_dispatch_details: every horde job in the dispatched batch."""
    return _line(
        ts,
        "DEBUG",
        "horde_worker_regen.process_management.scheduling.inference_scheduler:_log_job_dispatch_details:9404",
        f"All Batch IDs: [{', '.join(job_ids)}]",
    )


def inference_finished(ts: str, *, job_id: str, model: str, process: int, seconds: float) -> str:
    """message_dispatcher._handle_inference_result: a lane reporting a completed generation."""
    return _line(
        ts,
        "INFO",
        "horde_worker_regen.process_management.ipc.message_dispatcher:_handle_inference_result:1668",
        f"Inference finished for job {job_id[:8]} ({model}) on process {process}. It took {seconds} "
        f"seconds, finishing at 3.42 iterations per second and reported 1 faults.",
    )


def safety_checked(ts: str, *, job_id: str, seconds: float) -> str:
    """message_dispatcher._handle_safety_result: the safety stage clearing a job's images."""
    return _line(
        ts,
        "DEBUG",
        "horde_worker_regen.process_management.ipc.message_dispatcher:_handle_safety_result:1974",
        f"Job {job_id} had 0 images censored and took {seconds} seconds to check safety",
    )


def submitted_generation(ts: str, *, job_id: str, model: str, popped_ago: float, generate: float) -> str:
    """job_submitter.submit_single_generation: a result accepted by the horde."""
    return _line(
        ts,
        "SUCCESS",
        "horde_worker_regen.process_management.jobs.job_submitter:submit_single_generation:428",
        f"Submitted generation {job_id[:8]} (model: {model}) for 22.62 kudos. Job popped {popped_ago} "
        f"seconds ago and took {generate} to generate. (3.46 kudos/second for the whole batch. 0.4 or "
        f"greater is ideal)",
    )


def job_faulted(ts: str, *, job_id: str, process: int) -> str:
    """message_dispatcher._handle_faulted_inference_result: a lane-side fault naming its job."""
    return _line(
        ts,
        "WARNING",
        "horde_worker_regen.process_management.ipc.message_dispatcher:_handle_faulted_inference_result:892",
        f"Job {job_id} faulted on process {process} (RuntimeError: Pipeline failed to run).",
    )


def fault_reported(ts: str, *, job_id: str) -> str:
    """job_submitter.submit_single_generation: the terminal fault the worker reports to the horde."""
    return _line(
        ts,
        "ERROR",
        "horde_worker_regen.process_management.jobs.job_submitter:submit_single_generation:449",
        f"{job_id} faulted. Reported fault to the horde. Job popped 168.92 seconds ago and took 0.00 to generate.",
    )


def line_skip(ts: str, *, job_id: str, reason: str, process: int, displaced: str) -> str:
    """inference_scheduler.start_inference: a job seated ahead of the queue head."""
    return _line(
        ts,
        "INFO",
        "horde_worker_regen.process_management.scheduling.inference_scheduler:start_inference:10370",
        f"Job {job_id} skipped the line ({reason}) and will run on process {process} ahead of job "
        f"{displaced}: the displaced job's process is busy sampling its own model.",
    )


def inference_lane_started(ts: str, *, process: int, device: int) -> str:
    """process_lifecycle._start_inference_process: an inference child spawned onto a card."""
    return _line(
        ts,
        "INFO",
        "horde_worker_regen.process_management.lifecycle.process_lifecycle:_start_inference_process:2492",
        f"Starting inference process on PID {process} (device {device})",
    )


def post_process_lane_started(ts: str, *, process: int, device: int) -> str:
    """process_lifecycle.start_post_process_processes: the post-processing child spawned onto a card."""
    return _line(
        ts,
        "INFO",
        "horde_worker_regen.process_management.lifecycle.process_lifecycle:start_post_process_processes:1345",
        f"Started post-process process (id: {process}, device_index: {device})",
    )


def utilities_lane_started(ts: str, *, process: int, device: int) -> str:
    """process_lifecycle.start_utilities_processes: the image-utilities child spawned onto a card."""
    return _line(
        ts,
        "INFO",
        "horde_worker_regen.process_management.lifecycle.process_lifecycle:start_utilities_processes:1614",
        f"Started image utilities process (id: {process}, device_index: {device})",
    )


def safety_lane_started(ts: str, *, process: int = 0) -> str:
    """process_lifecycle.start_safety_processes: the safety child (its line carries no card)."""
    return _line(
        ts,
        "INFO",
        "horde_worker_regen.process_management.lifecycle.process_lifecycle:start_safety_processes:1248",
        f"Started safety process (id: {process})",
    )


def auxiliary_lane_safety_sharing(ts: str, *, lanes: str = "post-processing", device: int = 1) -> str:
    """process_lifecycle._note_auxiliary_lane_safety_sharing: safety came back to a pinned lane's card."""
    return _line(
        ts,
        "INFO",
        "horde_worker_regen.process_management.lifecycle.process_lifecycle:_note_auxiliary_lane_safety_sharing:1363",
        f"The safety process and the {lanes} lane(s) now share device {device}; the lane(s) stay on the "
        f"card they were placed on.",
    )


def driving_cards(ts: str, *, cards: int, processes_per_card: int = 1) -> str:
    """process_manager._build_card_runtimes: the multi-card boot summary and the intake budget."""
    per_card = ", ".join(
        f"card {index}: {processes_per_card} process(es), intake {processes_per_card}" for index in range(cards)
    )
    return _line(
        ts,
        "INFO",
        "horde_worker_regen.process_management.process_manager:_build_card_runtimes:2197",
        f"Driving {cards} cards, each with its own inference process pool; max_threads/queue_size apply "
        f"per card. Worker-wide intake budget: {cards * processes_per_card} job(s) held at once "
        f"({per_card}).",
    )


def model_preloading(ts: str, *, model: str, process: int) -> str:
    """inference_scheduler._send_preload: weights requested onto a lane."""
    return _line(
        ts,
        "DEBUG",
        "horde_worker_regen.process_management.scheduling.inference_scheduler:_send_preload:5656",
        f"Preloading model {model} on process {process}",
    )


def model_unloaded(ts: str, *, model: str, process: int) -> str:
    """message_dispatcher._handle_model_state_change: a lane dropping its resident model."""
    return _line(
        ts,
        "INFO",
        "horde_worker_regen.process_management.ipc.message_dispatcher:_handle_model_state_change:1558",
        f"Process {process} unloaded model {model}",
    )


def preload_cleared(ts: str, *, model: str, process: int) -> str:
    """inference_scheduler.preload_models: weights fetched for a job the scheduler then seated elsewhere."""
    return _line(
        ts,
        "DEBUG",
        "horde_worker_regen.process_management.scheduling.inference_scheduler:preload_models:7227",
        f"Clearing preloaded model {model} from process {process} as it is no longer needed",
    )


def status_block(
    ts: str,
    *,
    lanes: list[tuple[int, str, str | None]],
    aux: list[tuple[int, str, str]] | None = None,
    pending: list[tuple[str, str]] | None = None,
) -> list[str]:
    """status_reporter.print_status: one whole status print.

    ``lanes`` are ``(process id, state, model)`` inference lanes (a None model renders "No model loaded");
    ``aux`` are ``(process id, role, state)`` auxiliary lanes; ``pending`` is the queue listing.
    """
    location = "horde_worker_regen.reporting.status_reporter:print_status:180"
    lines = [_line(ts, "INFO", "horde_worker_regen.reporting.status_reporter:print_status:172", "^" * 80)]
    for process, state, model in lanes:
        model_part = f"<u>{model}</u> stable_diffusion_xl)" if model is not None else "No model loaded"
        lines.append(
            _line(
                ts,
                "INFO",
                location,
                f"  Process {process} ({state}) ({model_part}) <fg #7b7d7d>[last message: 0.0 secs ago: "
                f"START_INFERENCE heartbeat delta: 0.13]</>",
            ),
        )
    for process, role, state in aux or []:
        lines.append(_line(ts, "INFO", location, f"  Process {process}: ({role}) {state} "))
    body = ", ".join(f"<{job_id[:8]}: {model}>" for job_id, model in pending or [])
    lines.append(
        _line(ts, "INFO", "horde_worker_regen.reporting.status_reporter:_print_job_info:380", f"  Jobs: {body}"),
    )
    return lines


def ipc_drain(ts: str, *, process: int, message_type: str = "HordeInferenceResultMessage") -> str:
    """message_dispatcher._dispatch_buffered_message: one drained child message."""
    return _line(
        ts,
        "DEBUG",
        "horde_worker_regen.process_management.ipc.message_dispatcher:_dispatch_buffered_message:632",
        f"Received {message_type} from process {process}: 3.42 iterations per second",
    )


def lifecycle_from(bridge_log: str, tmp_path: Path) -> JobLifecycleModel:
    """Write a synthetic bridge log and return its single session's lifecycle model."""
    (tmp_path / "bridge.log").write_text(bridge_log, encoding="utf-8")
    bundle = LogBundle.from_path(tmp_path)
    session = segment_sessions(bundle.orchestrator_records())[0]
    return job_lifecycle_for(build_session_context(session, bundle))


def bridge(*lines: object) -> str:
    """A single-session bridge log: the startup boundary followed by the given lines (or line lists)."""
    flat: list[str] = [startup_line()]
    for entry in lines:
        if isinstance(entry, list):
            flat.extend(entry)
        else:
            flat.append(str(entry))
    return "\n".join(flat)


_JOB_A = "aaaaaaaa-1111-4111-8111-111111111111"
_JOB_B = "bbbbbbbb-2222-4222-8222-222222222222"
_JOB_C = "cccccccc-3333-4333-8333-333333333333"


class TestWaitSegments:
    """Splitting a job's wall clock into pop->dispatch, generation, and finished->submit."""

    def test_all_three_segments_are_measured(self, tmp_path: Path) -> None:
        """A job with every lifecycle line yields all three wait segments from the timestamps."""
        model = lifecycle_from(
            bridge(
                popped_job("01:00:00.000", job_id=_JOB_A, model="Deliberate"),
                inference_dispatched("01:00:30.000", job_id=_JOB_A, process=3),
                inference_finished("01:00:45.000", job_id=_JOB_A, model="Deliberate", process=3, seconds=15.0),
                safety_checked("01:00:47.000", job_id=_JOB_A, seconds=1.5),
                submitted_generation(
                    "01:00:50.000", job_id=_JOB_A, model="Deliberate", popped_ago=50.0, generate=15.0
                ),
            ),
            tmp_path,
        )
        job = model.jobs[_JOB_A[:8]]
        assert job.pre_inference_seconds == 30.0
        assert job.inference_seconds == 15.0
        assert job.post_inference_seconds == 5.0
        assert job.safety_seconds == 1.5
        assert job.model == "Deliberate"
        assert job.process_id == 3

    def test_pop_is_reconstructed_when_the_capture_begins_mid_run(self, tmp_path: Path) -> None:
        """A submit line states how long ago the pop was, which recovers a pop that is off the log."""
        model = lifecycle_from(
            bridge(
                submitted_generation(
                    "01:05:00.000", job_id=_JOB_A, model="Deliberate", popped_ago=90.0, generate=12.0
                ),
            ),
            tmp_path,
        )
        job = model.jobs[_JOB_A[:8]]
        assert job.popped_at == datetime(2026, 9, 9, 1, 3, 30)
        assert job.generation_seconds == 12.0
        # Without dispatch or finish lines the wait cannot be located, and the model says so by returning
        # None rather than by guessing.
        assert job.pre_inference_seconds is None
        assert job.post_inference_seconds is None
        assert model.has_lifecycle_lines is False

    def test_a_job_that_faults_before_dispatch_is_marked(self, tmp_path: Path) -> None:
        """A job faulted with no dispatch line is recorded as faulted and as never having been seated."""
        model = lifecycle_from(
            bridge(
                popped_job("01:00:00.000", job_id=_JOB_A, model="Deliberate"),
                fault_reported("01:00:20.000", job_id=_JOB_A),
            ),
            tmp_path,
        )
        job = model.jobs[_JOB_A[:8]]
        assert job.faulted is True
        assert job.faulted_before_dispatch is True
        assert job.pre_inference_seconds is None

    def test_batched_jobs_inherit_the_lead_job_timings(self, tmp_path: Path) -> None:
        """Only the lead job is named in the dispatch line; the batch line joins the rest to it."""
        model = lifecycle_from(
            bridge(
                popped_job("01:00:00.000", job_id=_JOB_A, model="Deliberate", batch=2),
                popped_job("01:00:01.000", job_id=_JOB_B, model="Deliberate", batch=2),
                inference_dispatched("01:00:10.000", job_id=_JOB_A, process=3),
                batch_ids("01:00:10.000", _JOB_A, _JOB_B),
                inference_finished("01:00:25.000", job_id=_JOB_A, model="Deliberate", process=3, seconds=15.0),
            ),
            tmp_path,
        )
        member = model.jobs[_JOB_B[:8]]
        assert member.batch_lead_job_id == _JOB_A[:8]
        assert member.process_id == 3
        assert member.pre_inference_seconds == 9.0
        # A batched inference is one act of seating, however many horde jobs rode on it.
        assert model.dispatch_count == 1


class TestPercentiles:
    """Percentile behaviour on the tiny samples a crash bundle actually contains."""

    def test_empty_sample_has_no_percentile(self) -> None:
        """No jobs means no number, not a zero."""
        assert percentile([], 0.9) is None

    def test_single_sample_is_its_own_percentile(self) -> None:
        """One job's value is every percentile of the session."""
        assert percentile([7.0], 0.9) == 7.0
        assert percentile([7.0], 0.5) == 7.0

    def test_two_samples_report_an_observed_value(self) -> None:
        """Nearest-rank keeps a two-job p90 to a value a job actually exhibited."""
        assert percentile([2.0, 8.0], 0.9) == 8.0
        assert percentile([2.0, 8.0], 0.1) == 2.0


class TestLanePlacement:
    """The process-to-card map, its history across re-spawns, and its migrations."""

    def test_roles_and_cards_are_mapped(self, tmp_path: Path) -> None:
        """Each spawn line places its lane on a card; the safety lane's line carries none."""
        model = lifecycle_from(
            bridge(
                inference_lane_started("01:00:00.000", process=3, device=0),
                inference_lane_started("01:00:01.000", process=4, device=1),
                post_process_lane_started("01:00:02.000", process=1, device=1),
                utilities_lane_started("01:00:03.000", process=2, device=1),
                safety_lane_started("01:00:04.000"),
            ),
            tmp_path,
        )
        placements = model.current_placements()
        assert placements[3].device_index == 0
        assert placements[3].role is LaneRole.INFERENCE
        assert placements[1].role is LaneRole.POST_PROCESS
        assert placements[0].device_index is None
        assert sorted(model.card_occupancy()) == [0, 1]

    def test_a_respawn_onto_another_card_is_a_migration_and_the_latest_wins(self, tmp_path: Path) -> None:
        """The map tracks the newest placement while the history keeps the move that produced it."""
        model = lifecycle_from(
            bridge(
                post_process_lane_started("01:00:00.000", process=1, device=1),
                post_process_lane_started("01:30:00.000", process=1, device=0),
            ),
            tmp_path,
        )
        assert model.current_placements()[1].device_index == 0
        assert len(model.lane_history) == 2
        migrations = model.lane_migrations()
        assert len(migrations) == 1
        assert (migrations[0].from_device_index, migrations[0].to_device_index) == (1, 0)
        assert migrations[0].timestamp == datetime(2026, 9, 9, 1, 30, 0)

    def test_a_job_takes_the_card_its_lane_held_at_dispatch(self, tmp_path: Path) -> None:
        """A job dispatched before a lane moved is attributed to the card the lane held then."""
        model = lifecycle_from(
            bridge(
                inference_lane_started("01:00:00.000", process=3, device=0),
                popped_job("01:00:05.000", job_id=_JOB_A, model="Deliberate"),
                inference_dispatched("01:00:06.000", job_id=_JOB_A, process=3),
                inference_lane_started("01:10:00.000", process=3, device=5),
                popped_job("01:10:05.000", job_id=_JOB_B, model="Deliberate"),
                inference_dispatched("01:10:06.000", job_id=_JOB_B, process=3),
            ),
            tmp_path,
        )
        assert model.jobs[_JOB_A[:8]].device_index == 0
        assert model.jobs[_JOB_B[:8]].device_index == 5


class TestWorkerShape:
    """Card count and intake budget, and what the model says when the boot line is missing."""

    def test_driving_line_gives_card_count_and_intake(self, tmp_path: Path) -> None:
        """The multi-card boot summary yields the card count, the budget, and each card's share."""
        model = lifecycle_from(bridge(driving_cards("01:00:00.000", cards=4, processes_per_card=2)), tmp_path)
        assert model.card_count == 4
        assert model.intake_budget == 8
        assert model.card_intake[2].process_count == 2

    def test_missing_driving_line_degrades_rather_than_raising(self, tmp_path: Path) -> None:
        """A single-card (or older) log has no card count, and every derived view still works."""
        model = lifecycle_from(
            bridge(
                popped_job("01:00:00.000", job_id=_JOB_A, model="Deliberate"),
                inference_dispatched("01:00:10.000", job_id=_JOB_A, process=3),
            ),
            tmp_path,
        )
        assert model.card_count is None
        assert model.intake_budget is None
        assert model.sampling_concurrency().mean_cards_busy is None
        assert model.status_census().snapshots == 0


class TestStatusSnapshots:
    """Parsing the ~20s status print into lane states and the pending queue."""

    def test_lane_states_models_and_queue_are_parsed(self, tmp_path: Path) -> None:
        """A status print yields each lane's role, state and model, and the queue listing."""
        model = lifecycle_from(
            bridge(
                status_block(
                    "01:00:00.000",
                    lanes=[
                        (3, "72% of 40 steps using k_euler_a", "Deliberate"),
                        (4, "WAITING_FOR_JOB", "AlbedoBase XL 3.1"),
                        (5, "WAITING_FOR_JOB", None),
                        (6, "PROCESS_STARTING", None),
                    ],
                    aux=[(0, "SAFETY", "WAITING_FOR_JOB"), (1, "POST_PROCESS", "POST_PROCESSING")],
                    pending=[(_JOB_A, "Deliberate"), (_JOB_B, "Flux.1-Schnell fp8 (Compact)")],
                ),
            ),
            tmp_path,
        )
        assert len(model.status_snapshots) == 1
        snapshot = model.status_snapshots[0]
        lanes = {lane.process_id: lane for lane in snapshot.lanes}
        assert lanes[3].sampling is True
        assert lanes[3].model == "Deliberate"
        assert lanes[4].idle is True
        assert lanes[5].model is None
        assert lanes[6].state == "PROCESS_STARTING"
        assert lanes[6].idle is False
        assert lanes[0].role is LaneRole.SAFETY
        assert lanes[1].state == "POST_PROCESSING"
        assert [pending.model for pending in snapshot.pending_jobs] == [
            "Deliberate",
            "Flux.1-Schnell fp8 (Compact)",
        ]
        assert len(snapshot.inference_lanes) == 4

    def test_an_empty_queue_still_produces_a_snapshot(self, tmp_path: Path) -> None:
        """A status print with nothing queued is a data point about idleness, not a parse failure."""
        model = lifecycle_from(
            bridge(status_block("01:00:00.000", lanes=[(3, "WAITING_FOR_JOB", None)], pending=[])),
            tmp_path,
        )
        assert model.status_snapshots[0].pending_jobs == ()

    def test_a_truncated_status_print_is_still_captured(self, tmp_path: Path) -> None:
        """A print whose Jobs line is missing (log cut mid-print) yields the lanes it did carry."""
        lines = status_block("01:00:00.000", lanes=[(3, "WAITING_FOR_JOB", "Deliberate")], pending=[])
        model = lifecycle_from(bridge(lines[:-1]), tmp_path)
        assert len(model.status_snapshots) == 1
        assert model.status_snapshots[0].pending_jobs == ()

    def test_seatable_pending_counts_only_free_cards(self, tmp_path: Path) -> None:
        """A pending job is seatable when its model sits on an idle lane whose card runs nothing.

        The lane holding "Deliberate" is idle but shares card 0 with a lane that is sampling, so the job
        for it is not seatable; the "AlbedoBase XL 3.1" lane on card 1 is free, so its job is.
        """
        model = lifecycle_from(
            bridge(
                inference_lane_started("01:00:00.000", process=3, device=0),
                inference_lane_started("01:00:00.000", process=4, device=0),
                inference_lane_started("01:00:00.000", process=5, device=1),
                status_block(
                    "01:00:10.000",
                    lanes=[
                        (3, "72% of 40 steps using k_euler_a", "Nova Anime XL"),
                        (4, "WAITING_FOR_JOB", "Deliberate"),
                        (5, "WAITING_FOR_JOB", "AlbedoBase XL 3.1"),
                    ],
                    pending=[(_JOB_A, "Deliberate"), (_JOB_B, "AlbedoBase XL 3.1")],
                ),
            ),
            tmp_path,
        )
        census = model.status_census()
        assert census.median_seatable_pending == 1
        assert census.median_idle_lanes == 2
        # The head's model is resident on an idle lane, but that lane's card is busy sampling.
        assert census.head_states.resident_but_blocked == 1


class TestConcurrencyAndCensus:
    """The time-weighted sampling-concurrency sweep and the line-skip / model-movement censuses."""

    def test_concurrency_counts_distinct_cards_not_lanes(self, tmp_path: Path) -> None:
        """Two jobs overlapping on one card is one busy card; on two cards it is two."""
        model = lifecycle_from(
            bridge(
                inference_lane_started("01:00:00.000", process=3, device=0),
                inference_lane_started("01:00:00.000", process=4, device=0),
                inference_dispatched("01:00:10.000", job_id=_JOB_A, process=3),
                inference_finished("01:00:20.000", job_id=_JOB_A, model="Deliberate", process=3, seconds=10.0),
                inference_dispatched("01:00:10.000", job_id=_JOB_B, process=4),
                inference_finished("01:00:20.000", job_id=_JOB_B, model="Deliberate", process=4, seconds=10.0),
            ),
            tmp_path,
        )
        profile = model.sampling_concurrency()
        assert profile.mean_cards_busy == 1.0
        assert profile.observed_seconds == 10.0

    def test_line_skips_are_censused_by_reason(self, tmp_path: Path) -> None:
        """The reason token is opaque, so a new scheduler reason still lands in the census."""
        model = lifecycle_from(
            bridge(
                line_skip("01:00:00.000", job_id=_JOB_A, reason="diversity", process=3, displaced=_JOB_C),
                line_skip("01:00:01.000", job_id=_JOB_B, reason="cross_card", process=4, displaced=_JOB_C),
                line_skip("01:00:02.000", job_id=_JOB_C, reason="cross_card", process=5, displaced=_JOB_A),
            ),
            tmp_path,
        )
        assert model.line_skip_census() == {"diversity": 1, "cross_card": 2}
        assert model.jobs[_JOB_A[:8]].line_skipped is True

    def test_model_movement_is_counted(self, tmp_path: Path) -> None:
        """Preloads, unloads and cleared preloads are counted separately, not lumped together."""
        model = lifecycle_from(
            bridge(
                model_preloading("01:00:00.000", model="Deliberate", process=3),
                model_preloading("01:00:01.000", model="Nova Anime XL", process=4),
                model_unloaded("01:00:02.000", model="Deliberate", process=3),
                preload_cleared("01:00:03.000", model="Nova Anime XL", process=4),
            ),
            tmp_path,
        )
        assert model.model_movement.preloads == 2
        assert model.model_movement.unloads == 1
        assert model.model_movement.cleared_preloads == 1

    def test_ipc_gaps_are_measured_against_the_status_cadence(self, tmp_path: Path) -> None:
        """The stall threshold comes from the session's own status cadence, not from a constant."""
        lines: list[str] = []
        for minute in range(4):
            lines.extend(status_block(f"01:0{minute}:00.000", lanes=[(3, "WAITING_FOR_JOB", None)], pending=[]))
        lines.append(ipc_drain("01:00:00.000", process=3))
        lines.append(ipc_drain("01:03:30.000", process=3))
        model = lifecycle_from(bridge(lines), tmp_path)
        assert model.status_cadence_seconds() == 60.0
        drain = model.ipc_drain_profile()
        assert drain.threshold_seconds == 90.0
        assert len(drain.gaps_over_threshold) == 1
        assert drain.gaps_over_threshold[0][1] == 210.0


class TestModelIsCachedAndSafe:
    """The parse is shared, and it never raises on a session it cannot make sense of."""

    def test_the_model_is_parsed_once_per_context(self, tmp_path: Path) -> None:
        """A second call returns the same object rather than reparsing the session."""
        (tmp_path / "bridge.log").write_text(bridge(popped_job("01:00:00.000", job_id=_JOB_A, model="X")), "utf-8")
        bundle = LogBundle.from_path(tmp_path)
        context = build_session_context(segment_sessions(bundle.orchestrator_records())[0], bundle)
        assert job_lifecycle_for(context) is job_lifecycle_for(context)

    def test_an_empty_session_yields_an_empty_model(self) -> None:
        """No records at all is an empty model, not an exception."""
        model = build_job_lifecycle([])
        assert model.jobs == {}
        assert model.wait_segments()[0].count == 0
        assert model.sampling_concurrency().observed_seconds == 0.0
        assert model.ipc_drain_profile().drains == 0

    def test_unrelated_records_are_ignored(self, tmp_path: Path) -> None:
        """Lines the model knows nothing about pass through without disturbing the parse."""
        text = bridge(
            "2026-09-09 01:00:00.000 | INFO | some.module:some_function:1 - an unrelated line",
            popped_job("01:00:01.000", job_id=_JOB_A, model="Deliberate"),
        )
        (tmp_path / "bridge.log").write_text(text, encoding="utf-8")
        records = list(read_records(tmp_path / "bridge.log"))
        model = build_job_lifecycle(records)
        assert set(model.jobs) == {_JOB_A[:8]}
