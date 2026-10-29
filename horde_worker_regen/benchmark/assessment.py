"""Assess capabilities by sending a limited number of jobs to one normal worker."""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path, PureWindowsPath
from statistics import median
from typing import IO, Protocol

from horde_sdk.generation_parameters.alchemy.consts import KNOWN_ALCHEMY_FORMS
from loguru import logger
from ruamel.yaml import YAML

from horde_worker_regen.analysis.system_info import rotational_storage
from horde_worker_regen.benchmark.assessment_policy import (
    AssessmentCase,
    assessment_priority,
    combined_case,
    installed_models_for_tiers,
    measured_recommendation,
    model_guidance,
    next_boundary,
    representative_cases,
)
from horde_worker_regen.benchmark.capabilities.capability import Capability, CapabilityKind
from horde_worker_regen.benchmark.capabilities.catalog import CatalogOptions, build_capability_catalog
from horde_worker_regen.benchmark.capabilities.report_render import (
    describe_decision,
    describe_jobs_per_minute,
    describe_lane_throughput,
    describe_sampling_rates,
    describe_wall_throughput,
    render_markdown,
)
from horde_worker_regen.benchmark.capabilities.result import (
    AssessmentBudget,
    CapabilityProbeResult,
    CapabilityReport,
    CapabilityVerdict,
    EvidenceScope,
    EvidenceStatus,
    MachineInfo,
    ModelLoadingSummary,
    OperatingProfile,
    ReferenceRateBasis,
    ReferenceSamplingRate,
    ServicePlacement,
    WorkloadEvidence,
)
from horde_worker_regen.benchmark.enums import BenchTier
from horde_worker_regen.benchmark.local_horde import LocalHorde
from horde_worker_regen.benchmark.progress_channel import (
    AssessmentJobProgress,
    LevelFinished,
    LevelPlanRow,
    LevelProgress,
    LevelStarted,
    NullProgressSink,
    ProgressSink,
    RampFinished,
    RampPlanned,
    RampStarted,
    RampStarting,
    SuggestionDecisionRow,
)
from horde_worker_regen.process_management.ipc.messages import SamplingWindow
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    SupervisorCommand,
    SupervisorControlMessage,
    WorkerStateSnapshot,
)
from horde_worker_regen.process_management.models.feature_readiness import GatedFeature
from horde_worker_regen.process_management.resources.run_metrics import JobMetricsRecord
from horde_worker_regen.process_management.scheduling.clearance_lease import CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS
from horde_worker_regen.process_management.scheduling.workload_flow import WorkloadKind


class AssessmentWorker(Protocol):
    """Worker operations used by the assessment controller and CPU-only test doubles."""

    async def start(self) -> None:
        """Launch the worker once."""

    async def configure(self, case: AssessmentCase, deadline: float) -> str | None:
        """Configure a drained worker, returning an explicit refusal when capacity cannot be applied."""

    async def close(self) -> None:
        """Stop the owned worker and its descendants."""

    def alive(self) -> bool:
        """Whether the worker can still make progress."""


def _card_total_mb(service: LocalHorde, device: int) -> float | None:
    """Return the card's total VRAM from the worker's snapshot, or None before the worker has reported it."""
    snapshot = service.snapshot
    if snapshot is None:
        return None
    for card in snapshot.per_card:
        if card.device_index == device and card.total_vram_mb is not None:
            return float(card.total_vram_mb)
    return None


def _probe_storage(worker: AssessmentWorker) -> dict[str, bool | None]:
    """Return storage types for the model directories, or an empty mapping for a test double."""
    if not isinstance(worker, OrdinaryAssessmentWorker):
        return {}
    return {directory: rotational_storage(Path(directory)) for directory in worker.model_directories}


def observed_sampling_concurrency(records: list[JobMetricsRecord], device_index: int) -> int:
    """Count overlapping child-stamped sampling windows on the named card, excluding touching endpoints."""
    edges: list[tuple[float, int]] = []
    for record in records:
        window = record.sampling_window
        if record.device_index == device_index and window is not None and window.last_step > window.first_step:
            edges.extend(((window.first_step, 1), (window.last_step, -1)))
    active = peak = 0
    for _, delta in sorted(edges):
        active += delta
        peak = max(peak, active)
    return peak


SINGLE_LANE_THROUGHPUT = 1.0
"""The lane-equivalent throughput of one lane sampling at its reference rate.

With overlapping sampling windows, a value below this threshold means the lanes sampled more slowly together
than one lane alone. Partial weight unloading can cause this. Such a result cannot confirm a thread count,
even when total job throughput improved."""

SINGLE_LANE_WALL_GAIN = 1.0
"""The reference throughput ratio for comparing multiple lanes with one lane."""

THREAD_COUNT_REASON_CODES = frozenset(
    {"sampling_not_overlapped", "throughput_below_single_lane", "throughput_unmeasured", "wall_gain_below_noise"}
)
"""The reason codes of a check whose thread count was not confirmed while its other limits ran cleanly."""

_FINALIZED_STAGE = "FINALIZED"
"""The ``stage_timestamps`` key the parent stamps when a job's record is finalized after submit."""


@dataclass(frozen=True)
class ClearanceWait:
    """The time one image job's child blocked on the clearance lease, as the child measured it."""

    seconds: float
    timed_out: bool
    record: JobMetricsRecord

    @property
    def reached_timeout(self) -> bool:
        """Whether the job sampled without a grant or waited at least the lease-acquire timeout.

        A child held that long samples unpriced, so the job's completion does not show that the card can serve
        the settings.
        """
        return self.timed_out or self.seconds >= CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS


def clearance_waits(records: list[JobMetricsRecord]) -> list[ClearanceWait]:
    """Return the clearance wait of each image job whose child blocked on the lease or went without a grant."""
    waits: list[ClearanceWait] = []
    for record in records:
        window = record.sampling_window
        if record.workload is not WorkloadKind.IMAGE_GENERATION or window is None:
            continue
        if window.lease_wait_seconds <= 0 and not window.lease_timed_out:
            continue
        waits.append(ClearanceWait(seconds=window.lease_wait_seconds, timed_out=window.lease_timed_out, record=record))
    return waits


def describe_clearance_timeout(wait: ClearanceWait) -> str:
    """Return the reason a check whose job sampled only after the clearance timeout is not confirmed."""
    record = wait.record
    size = f"{record.width}x{record.height}" if record.width is not None and record.height is not None else None
    shape = ", ".join(part for part in (record.model_name, size, f"batch {record.batch_count}") if part is not None)
    return (
        f"The worker could not price a job ({shape}) and sampled it only after the clearance timeout. "
        f"It waited {wait.seconds:.0f} s for the clearance lease, against a "
        f"{CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS:.0f} s timeout. The settings were not confirmed"
    )


@dataclass(frozen=True)
class PhaseWallThroughput:
    """Measured job throughput during a concurrency check."""

    megapixelsteps_per_second: float
    jobs_per_minute: float


def _job_wall_spans(records: list[JobMetricsRecord]) -> list[tuple[float, float]] | None:
    """Return each image job's start and end on one clock, or None when the jobs do not share one.

    Pop and finalization are wall-clock stamps and sampling windows are on the host's monotonic clock, so the
    choice is made for the whole phase. The pop-to-finalize span includes loads, encodes, safety and submit,
    so the comparison includes work done by lanes while another lane samples.
    """
    image_records = [record for record in records if record.workload is WorkloadKind.IMAGE_GENERATION]
    if not image_records:
        return None
    if all(record.time_popped is not None and _FINALIZED_STAGE in record.stage_timestamps for record in image_records):
        return [
            (record.time_popped, record.stage_timestamps[_FINALIZED_STAGE])
            for record in image_records
            if record.time_popped is not None
        ]
    if all(record.sampling_window is not None for record in image_records):
        return [
            (record.sampling_window.first_step, record.sampling_window.last_step)
            for record in image_records
            if record.sampling_window is not None
        ]
    return None


def phase_wall_throughput(records: list[JobMetricsRecord]) -> PhaseWallThroughput | None:
    """Return the megapixelsteps and jobs a phase's image jobs delivered per unit of wall time.

    The span runs from the earliest pop to the latest finalization, falling back to the sampling window edges
    when any job lacks either stamp.

    Returns:
        The figures, or None when a job has neither stamps nor a window, the span is empty, or the jobs report
        no megapixelsteps.
    """
    spans = _job_wall_spans(records)
    if spans is None:
        return None
    wall_seconds = max(end for _, end in spans) - min(start for start, _ in spans)
    megapixelsteps = sum(
        record.megapixelsteps for record in records if record.workload is WorkloadKind.IMAGE_GENERATION
    )
    if wall_seconds <= 0 or megapixelsteps <= 0:
        return None
    return PhaseWallThroughput(
        megapixelsteps_per_second=megapixelsteps / wall_seconds, jobs_per_minute=len(spans) * 60 / wall_seconds
    )


def wall_noise_margin(records: list[JobMetricsRecord]) -> float:
    """Return the relative median absolute deviation of the jobs' start-to-end times, 0 when unmeasured.

    The median deviation rather than the range keeps one slow job, such as the first load of a model, from
    widening the margin a thread count's gain has to clear.
    """
    spans = _job_wall_spans(records)
    if spans is None:
        return 0.0
    durations = [end - start for start, end in spans]
    centre = median(durations)
    if centre <= 0:
        return 0.0
    return median(abs(duration - centre) for duration in durations) / centre


def _sampling_rate(record: JobMetricsRecord) -> float | None:
    """Return the job's overall sampling rate, or None when the child did not report a positive one."""
    phase = record.phase_metrics
    if phase is None or phase.sampling is None or phase.sampling.iterations_per_second <= 0:
        return None
    return phase.sampling.iterations_per_second


def _windows_overlap(first: SamplingWindow, second: SamplingWindow) -> bool:
    """Whether two sampling windows share time, excluding touching endpoints as the overlap count does."""
    return first.first_step < second.last_step and second.first_step < first.last_step


def _union_seconds(windows: list[SamplingWindow]) -> float:
    """Return the length of the time covered by at least one window."""
    total = 0.0
    span: tuple[float, float] | None = None
    for window in sorted(windows, key=lambda item: item.first_step):
        if span is not None and window.first_step <= span[1]:
            span = (span[0], max(span[1], window.last_step))
            continue
        if span is not None:
            total += span[1] - span[0]
        span = (window.first_step, window.last_step)
    if span is not None:
        total += span[1] - span[0]
    return total


def model_sampling_rates(records: list[JobMetricsRecord]) -> dict[str, float]:
    """Return the median sampling rate of each model's image jobs, in the order the models first appear."""
    rates: dict[str, list[float]] = {}
    for record in records:
        rate = _sampling_rate(record)
        if record.workload is WorkloadKind.IMAGE_GENERATION and record.model_name is not None and rate is not None:
            rates.setdefault(record.model_name, []).append(rate)
    return {model_name: median(values) for model_name, values in rates.items()}


ShapeKey = tuple[int | None, int | None, int, bool, str | None, tuple[str, ...]]


def job_shape(record: JobMetricsRecord) -> ShapeKey:
    """Return the workload shape that decides a job's sampling rate, so references compare like with like."""
    return (
        record.width,
        record.height,
        record.batch_count,
        record.hires_fix,
        record.control_type,
        tuple(record.post_processing),
    )


def reference_sampling_rates(
    run_records: list[JobMetricsRecord],
    device_index: int,
    model_names: list[str],
    *,
    shapes: Mapping[str, set[ShapeKey]] | None = None,
) -> dict[str, ReferenceSamplingRate]:
    """Return each named model's single-lane sampling rate on the card from the run's records.

    A job whose window does not overlap another window on the card sampled alone, so the median of those jobs is
    the model's single-lane rate. With ``shapes``, only jobs of that model at one of the given shapes count,
    since a larger image or a bigger batch samples at a different rate for reasons other than the lane
    count. A model with no such job takes the fastest rate the run observed for it, which can only
    understate the lane count. A model with no rated job on the card is omitted. The relative spread is the
    median absolute deviation over the median, so one slow job among the single-lane samples does not
    widen it.
    """
    on_card = [record for record in run_records if record.device_index == device_index]
    windows = [record.sampling_window for record in on_card if record.sampling_window is not None]
    references: dict[str, ReferenceSamplingRate] = {}
    for model_name in model_names:
        alone: list[float] = []
        observed: list[float] = []
        for record in on_card:
            if record.workload is not WorkloadKind.IMAGE_GENERATION or record.model_name != model_name:
                continue
            if shapes is not None and model_name in shapes and job_shape(record) not in shapes[model_name]:
                continue
            rate = _sampling_rate(record)
            if rate is None:
                continue
            observed.append(rate)
            window = record.sampling_window
            # The list includes the current job's window, so a job that sampled alone has one match.
            if window is not None and sum(_windows_overlap(window, other) for other in windows) == 1:
                alone.append(rate)
        if alone:
            centre = median(alone)
            deviation = median(abs(rate - centre) for rate in alone)
            references[model_name] = ReferenceSamplingRate(
                iterations_per_second=centre,
                basis=ReferenceRateBasis.SINGLE_LANE,
                relative_spread=deviation / centre if centre > 0 else 0.0,
            )
        elif observed:
            references[model_name] = ReferenceSamplingRate(
                iterations_per_second=max(observed), basis=ReferenceRateBasis.RUN_MAXIMUM
            )
    return references


def lane_equivalent_throughput(
    check_records: list[JobMetricsRecord], reference_rates: dict[str, float]
) -> float | None:
    """Return how many single lanes' worth of sampling the image jobs delivered while any of them sampled.

    Each job's steps divided by its model's single-lane rate are the seconds one lane alone would have
    spent on them. Divide the sum by the time covered by the sampling windows. The result is 1.0 for one
    lane at its reference rate. Excluding idle gaps keeps dispatch delays out of the sampling measurement.

    Returns:
        The figure, or None when a job lacks a sampling window, a rate or sampling seconds, or its model has
        no reference rate.
    """
    image_records = [record for record in check_records if record.workload is WorkloadKind.IMAGE_GENERATION]
    if not image_records:
        return None
    single_lane_seconds = 0.0
    windows: list[SamplingWindow] = []
    for record in image_records:
        rate = _sampling_rate(record)
        reference = reference_rates.get(record.model_name) if record.model_name is not None else None
        if (
            record.sampling_window is None
            or rate is None
            or record.sampling_seconds is None
            or reference is None
            or reference <= 0
        ):
            return None
        single_lane_seconds += rate * record.sampling_seconds / reference
        windows.append(record.sampling_window)
    busy_seconds = _union_seconds(windows)
    return single_lane_seconds / busy_seconds if busy_seconds > 0 else None


WORKER_CONTACT_TIMEOUT_SECONDS = 30.0
"""The contact timeout allows for delays between the relay's once-per-second liveness frames."""


def worker_unresponsive(service: LocalHorde) -> bool:
    """Return whether a worker that once reported has stopped sending liveness and state updates."""
    return (
        service.last_contact is not None and time.monotonic() - service.last_contact > WORKER_CONTACT_TIMEOUT_SECONDS
    )


class AssessmentProgressWatch:
    """Bound inactivity without treating routine heartbeats as completed work."""

    def __init__(self, service: LocalHorde, timeout: float) -> None:
        """Use the configured worker timeout for unchanged work and a separate contact limit."""
        self.service = service
        self.timeout = timeout
        self.changed_at = time.monotonic()
        self._observed: set[tuple[object, ...]] = set()
        self.reason = ""

    def stalled(self) -> bool:
        """Return whether communication or meaningful work progress has stopped."""
        now = time.monotonic()
        if worker_unresponsive(self.service):
            self.reason = f"The worker has not sent an update for {WORKER_CONTACT_TIMEOUT_SECONDS:.0f} seconds"
            return True
        state = self.service.snapshot
        signature: tuple[object, ...] = ()
        if state is not None:
            signature = (
                state.config_revision,
                state.supervisor_paused,
                tuple(
                    (
                        p.last_process_state,
                        p.current_job_id,
                        p.last_current_step,
                        p.heartbeats_inference_steps,
                        p.loaded_horde_model_name,
                    )
                    for p in state.processes
                ),
                tuple(
                    (key, len(work.submitted), len(work.uploaded), work.issued)
                    for key, work in self.service.work.items()
                ),
            )
        if signature not in self._observed:
            self._observed.add(signature)
            self.changed_at = now
        if now - self.changed_at > self.timeout:
            self.reason = f"Work has not progressed for {self.timeout:.0f} seconds; see the worker logs"
            return True
        return False


def worker_progress_timeout(config: dict[str, object]) -> float:
    """Allow unchanged work for the longest configured worker loading or inference timeout."""
    values = [60.0]
    for key, default in (("process_timeout", 300), ("preload_timeout", 300), ("inference_step_timeout", 60)):
        value = config.get(key, default)
        values.append(float(value) if isinstance(value, int | float) else float(default))
    return max(values)


def model_loading_summaries(records: list[JobMetricsRecord]) -> list[ModelLoadingSummary]:
    """Aggregate observed loader events without counting missing measurements as cache hits."""
    models: dict[str, ModelLoadingSummary] = {}
    seen: set[tuple[str, float, float]] = set()
    for record in sorted(records, key=lambda item: item.time_popped or 0):
        metrics = record.phase_metrics
        if metrics is None:
            continue
        for load in metrics.model_loads:
            identity = (load.model_name, load.timestamp, load.duration_seconds)
            if load.phase != "disk_to_ram" or identity in seen:
                continue
            seen.add(identity)
            if load.model_name not in models:
                models[load.model_name] = ModelLoadingSummary(
                    model_name=load.model_name,
                    first_seconds=load.duration_seconds,
                )
            else:
                models[load.model_name].repeat_seconds.append(load.duration_seconds)
        if record.model_name in models and record.sampling_seconds is not None:
            models[record.model_name].sampling_seconds.append(record.sampling_seconds)
    return list(models.values())


def _single_lane_joint_jobs(queue_size: int) -> int:
    """Return the jobs a one-lane joint offers, enough to fill its queue once when the queue is deeper than one."""
    return queue_size + 1 if queue_size > 1 else 1


def _single_thread_follow_up(case: AssessmentCase) -> AssessmentCase:
    """Return a multi-lane joint workload at one thread, so its batch, resolution and queue size can confirm alone.

    The capability magnitude carries the thread count, so the follow-up's evidence id differs from the joint's
    and the report keeps both.
    """
    return replace(
        case,
        capability=case.capability.model_copy(update={"magnitude": 1}),
        profile=case.profile.model_copy(
            update={
                "threads": 1,
                "lease_slots": None,
                "companion_model": None,
                "work_jobs": _single_lane_joint_jobs(case.profile.queue_size),
            }
        ),
        notes=(
            *case.notes,
            f"Repeats the combined settings at one thread because the run at {case.profile.threads} threads "
            "did not confirm them.",
        ),
    )


def _confirmation_work_count(case: AssessmentCase) -> int:
    """Return the jobs a combined confirmation may offer, counting a multi-lane joint's single-thread follow-up."""
    follow_up = _single_thread_follow_up(case).check_work_count if case.profile.threads > 1 else 0
    return case.check_work_count + follow_up


def derived_work_budget(
    cases: list[AssessmentCase],
    *,
    validate: bool,
    options: CatalogOptions,
    targeted: bool = False,
    installed_models: int = 1,
) -> int:
    """Reserve representative work, two capacity checks per axis and one follow-up for overlap.

    A concurrency check offers two jobs per thread at one lane and again at its thread count, and its overlap
    follow-up repeats only the second run. The combined confirmation runs only its thread count, and a multi-lane
    joint keeps room to repeat at one thread.
    """
    anchors = [case for case in cases if case.capability.kind is CapabilityKind.BASELINE]
    total = sum(case.profile.threads * 2 * 2 if case.profile.threads > 1 else case.work_count for case in cases)
    for anchor in anchors:
        threads = max(
            (
                case.profile.threads
                for case in cases
                if case.profile.device_index == anchor.profile.device_index
                and case.profile.tier is anchor.profile.tier
            ),
            default=1,
        )
        confirmation_jobs = threads * 2 if threads > 1 else 1
        queue_size = max(
            (
                case.profile.queue_size
                for case in cases
                if case.profile.device_index == anchor.profile.device_index
                and case.profile.tier is anchor.profile.tier
            ),
            default=1,
        )
        if queue_size > 1:
            confirmation_jobs = max(confirmation_jobs, threads + queue_size)
        multi_lane = threads > 1
        if not targeted:
            for kind in (CapabilityKind.BATCH, CapabilityKind.THREADS, CapabilityKind.RESOLUTION):
                if kind in options.excluded_kinds:
                    continue
                if kind is CapabilityKind.RESOLUTION:
                    if not options.include_features:
                        continue
                elif not options.include_concurrency:
                    continue
                for magnitude in (2, 4):
                    if kind is CapabilityKind.THREADS:
                        # Each lane needs a distinct installed model. Do not reserve jobs for thread counts
                        # that the available models cannot exercise.
                        if magnitude > installed_models:
                            continue
                        jobs = magnitude * 2
                        total += jobs * 3
                        confirmation_jobs = max(confirmation_jobs, jobs)
                        multi_lane = True
                    else:
                        total += 1
        if validate:
            total += confirmation_jobs + (_single_lane_joint_jobs(queue_size) if multi_lane else 0)
    # Explicit thread probes still need their optional follow-up and two jobs per thread.
    if targeted:
        total += sum(case.profile.threads * 2 for case in cases if case.profile.threads > 1)
    return max(1, total)


def unavailable_feature(case: AssessmentCase, snapshot: WorkerStateSnapshot) -> str | None:
    """Explain why a feature is unavailable before sending a job the worker cannot accept."""
    required: set[GatedFeature] = set()
    if case.profile.control_type:
        required.add(GatedFeature.CONTROLNET)
    if case.profile.workflow == "qr_code":
        required.add(GatedFeature.SDXL_CONTROLNET)
    if case.profile.post_processing:
        required.add(GatedFeature.POST_PROCESSING)
    readiness = snapshot.feature_readiness
    if readiness is not None:
        for feature in readiness.gated:
            if feature.feature in required and not feature.offered:
                return f"{feature.label} is withheld: {feature.detail}; prepare this feature before assessment"
    return None


def ready_inference_lanes(state: WorkerStateSnapshot, device_index: int) -> int:
    """Count live inference processes on the card that can take a job now."""
    return sum(
        process.device_index == device_index
        and process.process_type == "INFERENCE"
        and process.is_alive
        and process.last_process_state
        in {"WAITING_FOR_JOB", "PRELOADED_MODEL", "INFERENCE_COMPLETE", "ALCHEMY_COMPLETE"}
        for process in state.processes
    )


class OrdinaryAssessmentWorker:
    """Launch normal startup with an isolated configuration and the standard supervisor channel."""

    def __init__(self, service: LocalHorde, directory: Path, config: dict[str, object]) -> None:
        """Keep the source config separate from the operator's persistent configuration."""
        self._service = service
        self._directory = directory.resolve()
        self._config = config
        cache_root = os.environ.get("AIWORKER_CACHE_HOME") or config.get("cache_home")
        if not isinstance(cache_root, str) or not cache_root:
            cache_root = config.get("models_folder_parent")
        if not isinstance(cache_root, str) or not cache_root:
            data_root = os.environ.get("HORDE_WORKER_DATA_DIR")
            cache_root = str(Path(data_root) / "models") if data_root else "models"
        self._cache_root = str(Path(cache_root).resolve())
        extra_roots = os.environ.get("AIWORKER_EXTRA_MODEL_DIRECTORIES")
        self._extra_roots = (
            os.pathsep.join(str(Path(root).resolve()) for root in extra_roots.split(os.pathsep) if root)
            if extra_roots
            else None
        )
        self._process: asyncio.subprocess.Process | None = None
        self._console: IO[bytes] | None = None
        self._applied_config: dict[str, object] | None = None

    @property
    def model_directories(self) -> list[str]:
        """Return the resolved cache root and extra model directories this worker loads weights from."""
        if self._extra_roots is not None:
            # The environment list takes precedence over the config's, as the worker's presence helper reads it.
            extra = [root for root in self._extra_roots.split(os.pathsep) if root]
        else:
            configured = self._config.get("extra_model_directories")
            extra = [root for root in configured if isinstance(root, str)] if isinstance(configured, list) else []
        return list(dict.fromkeys([self._cache_root, *extra]))

    def _write_config(self, config: dict[str, object]) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        temporary = self._directory / "bridgeData.yaml.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            YAML().dump(config, stream)
        temporary.replace(self._directory / "bridgeData.yaml")

    async def start(self) -> None:
        """Launch one fresh interpreter with the local URL set before SDK imports."""
        self._write_config(self._config)
        env = dict(os.environ)
        env.update(
            {
                "AI_HORDE_URL": f"{self._service.url}/api/",
                "AIWORKER_CACHE_HOME": self._cache_root,
                "HORDE_WORKER_RUN_ROOT": str(self._directory),
                "HORDE_BENCHMARK_SERVICE_URL": self._service.url,
                "HORDE_BENCHMARK_SERVICE_TOKEN": self._service.token,
            }
        )
        env.pop("AI_HORDE_DEV_URL", None)
        if self._extra_roots is not None:
            env["AIWORKER_EXTRA_MODEL_DIRECTORIES"] = self._extra_roots
        # An inherited env-form config must not override the isolated YAML or point the worker at
        # production. Cache/reference/download environment variables remain shared with preparation.
        env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[2]), env.get("PYTHONPATH", "")])
        self._console = (self._directory / "startup.log").open("wb")
        self._process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-m",
            "horde_worker_regen.benchmark.worker_client",
            cwd=self._directory,
            env=env,
            stdout=self._console,
            stderr=asyncio.subprocess.STDOUT,
        )

    def alive(self) -> bool:
        """Whether the owned interpreter is still running."""
        return self._process is not None and self._process.returncode is None

    async def _wait(self, predicate: Callable[[WorkerStateSnapshot], bool], deadline: float) -> bool:
        watch = AssessmentProgressWatch(self._service, self._progress_timeout())
        while time.monotonic() < deadline and self.alive():
            snapshot = self._service.snapshot
            if snapshot is not None and predicate(snapshot):
                return True
            if watch.stalled():
                self._service.phase = watch.reason
                return False
            await asyncio.sleep(0.1)
        self._service.phase = "The time limit was reached" if self.alive() else "The worker exited"
        return False

    def _progress_timeout(self) -> float:
        return worker_progress_timeout(self._config)

    def _case_config(self, case: AssessmentCase, snapshot: WorkerStateSnapshot) -> dict[str, object]:
        profile = case.profile
        config = dict(self._config)
        config.update(
            {
                "max_threads": profile.threads,
                "queue_size": profile.queue_size,
                "high_performance_mode": profile.high_performance_mode,
                "moderate_performance_mode": profile.moderate_performance_mode,
                "max_batch": profile.batch,
                "max_power": max(1, math.ceil(profile.width * profile.height / (8 * 64 * 64))),
                "allow_controlnet": bool(profile.control_type or profile.workflow),
                "allow_sdxl_controlnet": profile.workflow == "qr_code",
                "allow_post_processing": bool(profile.post_processing),
                "allow_lora": bool(profile.lora_names),
                "allow_img2img": bool(profile.control_type or profile.workflow),
                "allow_painting": False,
                "alchemy_caption_enabled": "caption" in profile.alchemy_forms,
                "alchemy_allow_concurrent": bool(profile.alchemy_forms),
                "forms": required_alchemy_config_forms(self._config, profile.alchemy_forms),
            }
        )
        if profile.lease_slots is not None:
            config.update({"gpu_sampling_lease_enabled": True, "gpu_sampling_lease_slots": profile.lease_slots})
        original_overrides = self._config.get("gpu_overrides", {})
        card_policies = (
            {str(index): policy for index, policy in original_overrides.items() if isinstance(policy, dict)}
            if isinstance(original_overrides, dict)
            else {}
        )
        config["gpu_overrides"] = {
            card.device_index: {
                **card_policies.get(str(card.device_index), {}),
                "models_to_load": (
                    [profile.model_name, *([profile.companion_model] if profile.companion_model else [])]
                    if card.device_index == profile.device_index
                    else []
                ),
                "max_threads": profile.threads if card.device_index == profile.device_index else 1,
                "queue_size": profile.queue_size,
                "high_performance_mode": profile.high_performance_mode,
                "moderate_performance_mode": profile.moderate_performance_mode,
                "max_batch": profile.batch,
                "max_power": config["max_power"],
                "allow_controlnet": config["allow_controlnet"],
                "allow_sdxl_controlnet": config["allow_sdxl_controlnet"],
                "allow_post_processing": config["allow_post_processing"],
                "allow_lora": config["allow_lora"],
                "allow_img2img": config["allow_img2img"],
                "allow_painting": False,
            }
            for card in snapshot.per_card
        }
        return config

    async def configure(self, case: AssessmentCase, deadline: float) -> str | None:
        """Pause, drain, apply the workload config and acknowledge normal card resizing.

        A check whose config equals the last one applied, with no issued work still unfinalized, keeps the
        running worker and only confirms its lanes are ready. On a real card the boundary protocol costs tens
        of seconds against a few seconds of sampling, and repeating it would change nothing.
        """
        service = self._service
        profile = case.profile
        previous = self._applied_config
        # Cleared before any wait so a refusal, a failed wait or a cancellation by the caller's watch leaves the
        # next check on the full protocol.
        self._applied_config = None
        snapshot = service.snapshot
        if (
            previous is not None
            and snapshot is not None
            and not any(work.issued and not work.finalized for work in service.work.values())
            and self._case_config(case, snapshot) == previous
        ):
            refusal = unavailable_feature(case, snapshot)
            if refusal is not None:
                return refusal
            service.phase = f"Confirming {profile.threads} inference processes"
            if not await self._wait(
                lambda state: ready_inference_lanes(state, profile.device_index) >= profile.threads, deadline
            ):
                return service.phase + ". The requested inference processes were not ready"
            self._applied_config = previous
            service.phase = "Waiting for the worker to take test jobs"
            return None
        service.phase = "Waiting for current jobs to finish"
        service.command(SupervisorControlMessage(command=SupervisorCommand.PAUSE))
        if not await self._wait(lambda state: state.supervisor_paused and state.jobs_in_hand == 0, deadline):
            return service.phase + ". Current work did not finish before applying test settings"
        snapshot = service.snapshot
        assert snapshot is not None
        revision = snapshot.config_revision
        config = self._case_config(case, snapshot)
        self._write_config(config)
        service.phase = "Applying test settings"
        service.command(SupervisorControlMessage(command=SupervisorCommand.RELOAD_CONFIG))
        if not await self._wait(lambda state: state.config_revision > revision, deadline):
            return service.phase + ". The worker did not confirm the new settings"
        snapshot = service.snapshot
        assert snapshot is not None
        refusal = unavailable_feature(case, snapshot)
        if refusal is not None:
            return refusal
        while time.monotonic() < deadline:
            service.phase = f"Preparing {profile.threads} inference processes"
            request_id = uuid.uuid4().hex
            service.command(
                SupervisorControlMessage(
                    command=SupervisorCommand.RESIZE_CAPACITY,
                    device_index=profile.device_index,
                    target_threads=profile.threads,
                    request_id=request_id,
                )
            )
            if not await self._wait(
                lambda state, expected=request_id: (
                    state.capacity_change is not None and state.capacity_change.request_id == expected
                ),
                deadline,
            ):
                return service.phase + ". The worker did not confirm the capacity change"
            snapshot = service.snapshot
            assert snapshot is not None and snapshot.capacity_change is not None
            if snapshot.capacity_change.accepted:
                if not await self._wait(
                    lambda state: ready_inference_lanes(state, profile.device_index) >= profile.threads, deadline
                ):
                    return service.phase + ". The requested inference processes did not become ready"
                service.command(SupervisorControlMessage(command=SupervisorCommand.RESUME))
                self._applied_config = config
                service.phase = "Waiting for the worker to take test jobs"
                return None
            if not snapshot.capacity_change.retryable:
                return snapshot.capacity_change.reason
            await asyncio.sleep(0.2)
        return "The time limit was reached while preparing inference processes"

    async def close(self) -> None:
        """Drain through normal shutdown, then reap owned descendants if the deadline expires."""
        process = self._process
        if process is not None and process.returncode is None:
            self._service.withdraw_unissued()
            self._service.command(SupervisorControlMessage(command=SupervisorCommand.SHUTDOWN))
            try:
                await asyncio.wait_for(process.wait(), timeout=30)
            except TimeoutError:
                from horde_worker_regen.process_management.lifecycle.owned_process_registry import kill_process_tree

                await asyncio.to_thread(kill_process_tree, process.pid, grace_seconds=2)
                await process.wait()
        if process is not None:
            from horde_worker_regen.process_management.lifecycle.owned_process_registry import OwnedProcessRegistry

            registry = OwnedProcessRegistry(self._directory / ".horde_worker_regen" / "owned_pids.json")
            await asyncio.to_thread(registry.reap_orphans_from_previous_run, kill_tree=True)
        if self._console is not None:
            self._console.close()


def _add_optional(first: float | None, second: float | None) -> float | None:
    """Sum two optional durations, unknown only when both are."""
    if first is None and second is None:
        return None
    return (first or 0.0) + (second or 0.0)


def required_alchemy_config_forms(config: dict[str, object], forms: list[str]) -> list[str]:
    """Include selected service forms in the ordinary worker's configured offers."""
    original = config.get("forms", [])
    configured = {str(item) for item in original} if isinstance(original, list) else set()
    direct = {str(item.value) for item in KNOWN_ALCHEMY_FORMS}
    for form in forms:
        configured.add(form if form in direct else "post-process")
    return sorted(configured)


def assessment_config(
    models: dict[BenchTier, str],
    source: Path,
    *,
    alchemy_forms: list[str] | None = None,
) -> dict[str, object]:
    """Copy ordinary runtime policies while isolating identity, offered work and durable state."""
    raw = YAML(typ="safe").load(source) if source.exists() else {}
    config: dict[str, object] = dict(raw or {})
    if "allow_inpainting" in config:
        config.setdefault("allow_painting", config.pop("allow_inpainting"))
    original_overrides = config.get("gpu_overrides", {})
    overrides = original_overrides if isinstance(original_overrides, dict) else {}
    for policy in overrides.values():
        if isinstance(policy, dict) and "allow_inpainting" in policy:
            policy.setdefault("allow_painting", policy.pop("allow_inpainting"))
    config.update(
        {
            "api_key": "b" * 22,
            "dreamer_name": "local-capability-assessment",
            "models_to_load": sorted(set(models.values())),
            "max_threads": 1,
            "queue_size": 1,
            "gpu_overrides": {
                index: {
                    **policy,
                    "max_threads": 1,
                    "queue_size": 1,
                    "models_to_load": sorted(set(models.values())),
                }
                for index, policy in overrides.items()
                if isinstance(policy, dict)
            },
            "dynamic_models": False,
            "stats_export_enabled": True,
            "remove_maintenance_on_init": False,
            "dry_run_skip_api": False,
            "dry_run_skip_inference": False,
            "dry_run_skip_safety": False,
            "dry_run_skip_post_processing": False,
            "dreamer": True,
            "scribe": False,
            "alchemist": True,
            "alchemy_caption_enabled": "caption" in (alchemy_forms or []),
            "forms": required_alchemy_config_forms(config, alchemy_forms or []),
        }
    )
    # The copied YAML can contain relative model/cache paths. The worker's isolated working
    # directory must still resolve the same weights as the operator's original config.
    for key in ("cache_home", "models_folder_parent"):
        value = config.get(key)
        if isinstance(value, str):
            config[key] = str((source.resolve().parent / value).resolve())
    extra_directories = config.get("extra_model_directories")
    if isinstance(extra_directories, list):
        config["extra_model_directories"] = [
            str((source.resolve().parent / value).resolve()) for value in extra_directories if isinstance(value, str)
        ]
    custom_models = config.get("custom_models")
    if isinstance(custom_models, list):
        for model in custom_models:
            if isinstance(model, dict) and isinstance(model.get("filepath"), str):
                model["filepath"] = str((source.resolve().parent / model["filepath"]).resolve())
    config.pop("horde_url", None)
    return config


@dataclass(frozen=True)
class _PhaseRun:
    """Jobs offered during a check and the reason the run stopped."""

    ids: list[str]
    configure_seconds: float
    refusal: str | None = None
    stop_reason: str = ""
    unavailable_forms: bool = False


class OperatorAssessment:
    """Run baseline and capacity checks within a shared budget, reserving work for combined confirmation."""

    def __init__(
        self,
        *,
        catalog_options: CatalogOptions,
        out_dir: Path,
        seconds: float | None = None,
        max_jobs: int | None = None,
        validate: bool = True,
        only_probe: str | None = None,
        progress_sink: ProgressSink | None = None,
        models: dict[BenchTier, str] | None = None,
        worker_factory: Callable[[LocalHorde, Path, dict[str, object]], AssessmentWorker] = OrdinaryAssessmentWorker,
        per_card: bool = False,
    ) -> None:
        """Configure an assessment without starting a worker or importing torch."""
        if (seconds is not None and seconds <= 0) or (max_jobs is not None and max_jobs < 1):
            raise ValueError("assessment seconds and job budget must be positive")
        self.options = catalog_options
        if only_probe:
            probe = next(
                (item for item in build_capability_catalog(catalog_options) if item.capability.slug == only_probe),
                None,
            )
            if probe is None:
                raise ValueError(f"Unknown or unavailable capability: {only_probe}")
            self.options = catalog_options.model_copy(update={"tiers": [probe.capability.tier]})
        self.out_dir = out_dir.resolve()
        self.budget = AssessmentBudget(seconds=seconds, max_jobs=max_jobs or 1)
        self._explicit_jobs = max_jobs
        self.per_card = per_card
        self.validate = validate
        self.only_probe = only_probe
        self.progress = progress_sink or NullProgressSink()
        self._models = models
        self._worker_factory = worker_factory
        self._records: dict[str, JobMetricsRecord] = {}
        self._offsets: dict[Path, int] = {}
        self._planned: dict[str, AssessmentCase] = {}
        self._source_config: dict[str, object] = {}
        self._operator_config: dict[str, object] = {}
        self._installed: dict[BenchTier, str] = {}

    def _plan(self, case: AssessmentCase) -> None:
        self._planned[case.case_id] = case
        self.progress.emit(
            RampPlanned(
                run_id=self.out_dir.name,
                rows=[
                    LevelPlanRow(
                        level_id=item.case_id,
                        stage="assessment",
                        tier=str(item.profile.tier),
                        features=[self._description(item)],
                    )
                    for item in self._planned.values()
                ],
            )
        )

    @staticmethod
    def _description(case: AssessmentCase) -> str:
        profile = case.profile
        labels = {
            CapabilityKind.BASELINE: "Baseline",
            CapabilityKind.BATCH: "Multiple images per job",
            CapabilityKind.THREADS: "Concurrent sampling",
            CapabilityKind.RESOLUTION: "Larger images",
            CapabilityKind.QUEUE_SIZE: "Queued jobs",
            CapabilityKind.HIRES_FIX: "Two-pass generation",
            CapabilityKind.POST_PROCESSING: "Post-processing",
            CapabilityKind.CONTROLNET: "ControlNet",
            CapabilityKind.QR_CODE: "QR code",
            CapabilityKind.ALCHEMY_CLIP: "Captioning",
            CapabilityKind.ALCHEMY_GRAPH: "Image tools",
            CapabilityKind.ALCHEMY_CONCURRENT: "Image tools during generation",
            CapabilityKind.LORA_DOWNLOAD: "LoRA download",
        }
        check = "Combined settings" if case.combined else labels.get(case.capability.kind, case.capability.label)
        models = profile.model_name + (f" with {profile.companion_model}" if profile.companion_model else "")
        return (
            f"{models}: {check}, {profile.width}x{profile.height}, "
            f"batch {profile.batch}, {profile.threads} sampling threads"
        )

    def _mode_profile(self, case: AssessmentCase) -> AssessmentCase:
        config = self._source_config
        overrides = config.get("gpu_overrides", {})
        if isinstance(overrides, dict):
            for index, values in overrides.items():
                if str(index) == str(case.profile.device_index) and isinstance(values, dict):
                    config = {**config, **values}
        return AssessmentCase(
            case.capability,
            case.profile.model_copy(
                update={
                    "high_performance_mode": config.get("high_performance_mode") is True,
                    "moderate_performance_mode": config.get("moderate_performance_mode") is True
                    and config.get("high_performance_mode") is not True,
                }
            ),
            case.combined,
        )

    def _read_records(self) -> None:
        for path in (self.out_dir / "worker" / ".horde_worker_regen" / "stats").glob("*.jsonl"):
            # Files are bounded and local. The final line can be in flight while the writer flushes.
            with path.open(encoding="utf-8") as stream:
                stream.seek(self._offsets.get(path, 0))
                while line := stream.readline():
                    if not line.endswith("\n"):
                        break
                    self._offsets[path] = stream.tell()
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as error:
                        logger.warning(f"Invalid completed stats record in {path.name}: {error}")
                        continue
                    if event.get("event") == "job_completed":
                        record = JobMetricsRecord.model_validate(event["job"])
                        self._records[record.job_id] = record

    async def run_async(self) -> CapabilityReport:
        """Run one normal worker, persist partial evidence on budget exhaustion and close every owned child."""
        started = time.monotonic()
        deadline = started + self.budget.seconds if self.budget.seconds is not None else math.inf
        source = Path("bridgeData.yaml")
        raw = YAML(typ="safe").load(source) if source.exists() else {}
        self._operator_config = dict(raw) if isinstance(raw, dict) else {}
        models = self._models
        if models is None:
            # Every installed family is resolved so a concurrency check can lend a lane to a model from a
            # family the operator did not select; only selected families get their own checks.
            models = installed_models_for_tiers(list(BenchTier))
        self._installed = dict(models)
        models = {tier: name for tier, name in models.items() if tier in self.options.tiers}
        self.out_dir.mkdir(parents=True, exist_ok=True)
        evidence: list[WorkloadEvidence] = []
        async with LocalHorde() as service:
            forms = sorted(
                {
                    form.form
                    for probe in build_capability_catalog(self.options)
                    if not self.only_probe or probe.capability.slug == self.only_probe
                    for form in probe.scenario.alchemy_forms[:1]
                }
            )
            self._source_config = assessment_config(models, source, alchemy_forms=forms)
            worker = self._worker_factory(service, self.out_dir / "worker", self._source_config)
            boot_stop = ""
            try:
                if models:
                    await worker.start()
                    boot_watch = AssessmentProgressWatch(service, self._idle_timeout())
                    next_startup_update = started
                    while service.snapshot is None and worker.alive() and time.monotonic() < deadline:
                        if boot_watch.stalled():
                            boot_stop = boot_watch.reason
                            break
                        if time.monotonic() >= next_startup_update:
                            self.progress.emit(
                                RampStarting(
                                    run_id=self.out_dir.name,
                                    process_mode="real",
                                    phase="Starting the worker. Waiting for GPU information "
                                    f"({time.monotonic() - started:.0f}s)",
                                )
                            )
                            next_startup_update = time.monotonic() + 1
                        await asyncio.sleep(0.1)
                snapshot = service.snapshot
                devices = [card.device_index for card in snapshot.per_card] if snapshot is not None else []
                # MachineInfo carries one VRAM figure, so it is filled only when every card reports the same total.
                totals = {card.total_vram_mb for card in snapshot.per_card} if snapshot is not None else set()
                machine = MachineInfo(
                    gpu_name=", ".join(card.device_name or str(card.device_index) for card in snapshot.per_card)
                    if snapshot is not None
                    else None,
                    total_vram_mb=next(iter(totals)) if len(totals) == 1 else None,
                    storage=await asyncio.to_thread(_probe_storage, worker),
                )
                cases = representative_cases(self.options, models, devices, only_probe=self.only_probe)
                if self.only_probe and devices:
                    matching = [case for case in cases if case.capability.slug == self.only_probe]
                    if not matching:
                        raise ValueError(f"Unknown or unavailable capability: {self.only_probe}")
                    selected = [case for case in cases if case.capability.kind is CapabilityKind.BASELINE] + matching
                    cases = list({case.case_id: case for case in selected}.values())
                cases = [self._mode_profile(case) for case in cases]
                if self._explicit_jobs is None:
                    self.budget.max_jobs = derived_work_budget(
                        cases,
                        validate=self.validate,
                        options=self.options,
                        targeted=bool(self.only_probe),
                        installed_models=len(self._installed),
                    )
                for case in cases:
                    self._plan(case)
                self.progress.emit(
                    RampStarted(
                        run_id=self.out_dir.name,
                        num_levels=len(cases),
                        tiers=[str(tier) for tier in self.options.tiers],
                        process_mode="real",
                        gpu_name=machine.gpu_name,
                        work_limit=self.budget.max_jobs,
                        time_limit_seconds=self.budget.seconds,
                    )
                )
                await self._run_cases(service, worker, cases, evidence, deadline)
                for item in evidence:
                    if item.status is EvidenceStatus.PREDICTED:
                        self._plan(AssessmentCase(item.capability, item.profile))
                        self._finish(item)
                self.budget.stop_reason = (
                    "No eligible installed checkpoints for the selected families. "
                    "Check exclusions or download models first"
                    if not models
                    else self._worker_exit_reason()
                    if not worker.alive()
                    else "The worker stopped responding. Inspect worker/logs/bridge.log"
                    if worker_unresponsive(service)
                    else "The worker did not report a usable card. " + (boot_stop or "The time limit was reached")
                    if not devices
                    else "wall-clock budget exhausted"
                    if time.monotonic() >= deadline
                    else "job budget exhausted"
                    if any(item.reason_code in {"work_limit", "confirmation_limit"} for item in evidence)
                    else "Selected checks finished. Estimates remain untested"
                )
            finally:
                await worker.close()
                self._read_records()
        self.budget.elapsed_seconds = time.monotonic() - started
        report = CapabilityReport(
            run_id=self.out_dir.name,
            machine=machine,
            operator_assessment=True,
            evidence=evidence,
            budget=self.budget,
            suggested_bridge_data=measured_recommendation(evidence, per_card=self.per_card),
            model_loading=model_loading_summaries(list(self._records.values())),
            model_guidance=model_guidance(evidence, machine, self.options, installed_models=len(self._installed)),
            probes=[
                CapabilityProbeResult(
                    capability=item.capability,
                    verdict=CapabilityVerdict.PROVEN
                    if item.status is EvidenceStatus.MEASURED_SUCCESS
                    else CapabilityVerdict.DISPROVEN
                    if item.status is EvidenceStatus.MEASURED_FAILURE
                    else CapabilityVerdict.SKIPPED,
                    reasons=[item.reason],
                )
                for item in evidence
            ],
        )
        report.storage_notes = self._storage_notes(report)
        (self.out_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        (self.out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
        decisions = [
            SuggestionDecisionRow(
                setting=item.setting,
                value_text=describe_decision(item)[0],
                basis=item.basis.value,
                basis_label=describe_decision(item)[1],
                detail=item.detail,
            )
            for item in report.suggested_bridge_data.decisions
        ]
        self.progress.emit(
            RampFinished(
                run_id=report.run_id,
                suggestion_confirmed=report.suggested_bridge_data.safe_to_apply,
                levels_passed=sum(item.passed for item in report.probes),
                levels_total=len(report.probes),
                report_path=str(self.out_dir / "report.md"),
                suggested_bridge_data_yaml=report.suggested_bridge_data.as_yaml_block(),
                suggestion_decisions=decisions,
                stop_reason=self.budget.stop_reason,
                model_guidance=report.model_guidance,
                storage_notes=report.storage_notes,
                model_loading_summary=[
                    f"{item.model_name}: first observed load {item.first_seconds:.2f}s. "
                    + (
                        f"Repeat loads {min(item.repeat_seconds):.2f} to {max(item.repeat_seconds):.2f}s. "
                        if item.repeat_seconds
                        else "Repeat loads not reported. "
                    )
                    + f"{len(item.load_seconds)} loads measured."
                    for item in report.model_loading
                ],
                run_jobs_completed=len(self._records),
            )
        )
        return report

    def _idle_timeout(self) -> float:
        return worker_progress_timeout(self._source_config)

    def _worker_exit_reason(self) -> str:
        """Report the log paths and last error line when worker startup fails."""
        reason = "The worker exited. Inspect worker/logs/bridge_main_startup.log and bridge.log"
        for name in ("bridge_main_startup.log", "bridge.log"):
            path = self.out_dir / "worker" / "logs" / name
            if not path.is_file():
                continue
            tail = ""
            with suppress(OSError), path.open("rb") as stream:
                # Only the tail is read; bridge.log can be large and the last error is what matters.
                stream.seek(max(0, path.stat().st_size - 65536))
                tail = stream.read().decode("utf-8", errors="replace")
            errors = [
                line.strip()
                for line in tail.splitlines()
                if any(token in line for token in ("ERROR", "CRITICAL", "Traceback"))
            ]
            if errors:
                return f"{reason}. Last error in {name}: {errors[-1][-300:]}"
        return reason

    def _storage_notes(self, report: CapabilityReport) -> list[str]:
        notes = [
            "Model loading includes deserialization and may read from the OS cache. "
            "Physical disk speed was not measured."
        ]
        rotational = sorted(path for path, is_rotational in report.machine.storage.items() if is_rotational)
        solid_state = bool(report.machine.storage) and all(
            is_rotational is False for is_rotational in report.machine.storage.values()
        )
        if solid_state:
            notes.append("The models are on solid-state storage.")
        from horde_worker_regen.process_management.scheduling.host_commit import page_file_drive_roots

        # Page file roots are Windows drive roots, so the model paths are read as Windows paths on every host.
        # Off Windows the root set is empty and the note is not added.
        page_file_drives = page_file_drive_roots()
        model_drives = {
            PureWindowsPath(path).drive.lower() + "\\"
            for path in report.machine.storage
            if page_file_drives and PureWindowsPath(path).drive
        }
        for drive in sorted(model_drives & page_file_drives):
            notes.append(
                f"A page file shares drive {drive[:2].upper()} with the models. A system-managed page file grows "
                "into that drive's free space under load, so low-disk warnings there can be page-file growth."
            )
        if not report.model_loading:
            notes.append("Model loading time was not reported.")
        for item in report.model_loading:
            if item.sampling_seconds and median(item.load_seconds) >= median(item.sampling_seconds):
                notes.append(
                    f"Loading {item.model_name} took as long as sampling or longer. "
                    "Changing models often can reduce throughput."
                )
        successful = [
            item.profile
            for item in report.evidence
            if item.combined_validation and item.status is EvidenceStatus.MEASURED_SUCCESS
        ]
        for profile in successful:
            from horde_worker_regen.process_management.process_manager import resolve_card_concurrency

            # Count the operator's configured models because the assessment leaves that list unchanged.
            # The checkpoints loaded for measurement may be only a subset.
            model_count = self._operator_model_count(profile.device_index)
            if model_count is None:
                continue
            concurrency = resolve_card_concurrency(
                max_threads=profile.threads,
                queue_size=profile.queue_size,
                num_models_to_load=model_count,
                gpu_sampling_lease_enabled=self._operator_config.get("gpu_sampling_lease_enabled") is True,
                gpu_sampling_lease_slots=self._configured_lease_slots(),
                max_threads_ceiling=profile.threads,
            )
            process_count = concurrency.target_process_count
            if solid_state or model_count <= process_count:
                continue
            counts = (
                f"Card {profile.device_index} has {model_count} configured models and "
                f"{process_count} inference processes at these settings."
            )
            if rotational:
                resident = f"{process_count} model stays" if process_count == 1 else f"{process_count} models stay"
                notes.append(
                    f"{counts} The models are on rotational storage ({', '.join(rotational)}). "
                    f"{resident} resident, and each change to another model reloads it from disk."
                )
            else:
                notes.append(
                    f"{counts} The storage type was not determined. "
                    "If the models are on rotational storage, each change to another model reloads it from disk."
                )
        return list(dict.fromkeys(notes))

    def _operator_model_count(self, device_index: int) -> int | None:
        """Count the models the operator's configuration resolves to for a card, meta rules expanded."""
        from horde_worker_regen.bridge_data.data_model import reGenBridgeData
        from horde_worker_regen.bridge_data.load_config import BridgeDataLoader
        from horde_worker_regen.reference_helper import ensure_offline_reference_manager

        config = dict(self._operator_config)
        overrides = config.get("gpu_overrides")
        if isinstance(overrides, dict):
            for index, values in overrides.items():
                if str(index) == str(device_index) and isinstance(values, dict):
                    config.update(values)
        models = config.get("models_to_load")
        if not isinstance(models, list) or not models:
            return None
        bridge = reGenBridgeData.model_validate(
            {
                "models_to_load": models,
                "models_to_skip": config.get("models_to_skip") or [],
                "load_large_models": config.get("load_large_models", False),
                "only_models_on_disk": False,
            }
        )
        return len(
            BridgeDataLoader._resolve_meta_instructions(
                bridge,
                ensure_offline_reference_manager(),
                bounded_beta_read=True,
            ),
        )

    async def _run_cases(
        self,
        service: LocalHorde,
        worker: AssessmentWorker,
        cases: list[AssessmentCase],
        evidence: list[WorkloadEvidence],
        deadline: float,
    ) -> None:
        anchors = {(case.profile.device_index, case.profile.tier) for case in cases}
        explored: dict[tuple[int, BenchTier, CapabilityKind], int] = {}
        while cases:
            cases.sort(
                key=lambda item: (
                    item.capability.kind is not CapabilityKind.BASELINE,
                    item.profile.device_index,
                    str(item.profile.tier),
                    assessment_priority(item),
                )
            )
            case = cases.pop(0)
            if case.profile.threads > 1:
                companion = self._companion_model(case, evidence)
                if companion is None:
                    self._plan(case)
                    item = self._unknown(
                        case,
                        "Not applicable: with one offered model the worker keeps every job on the lane that holds "
                        "it. Install a second model to test concurrent sampling",
                    ).model_copy(update={"reason_code": "not_applicable"})
                    evidence.append(item)
                    self._finish(item)
                    continue
                # One sampler can finish more jobs per minute when other lanes prepare the next job.
                # Preserve an existing lease slot count so the check measures the operator's settings.
                lease_slots = None if self._source_config.get("gpu_sampling_lease_enabled") is True else 1
                case = AssessmentCase(
                    case.capability,
                    case.profile.model_copy(
                        update={
                            "work_jobs": case.profile.threads * 2,
                            "companion_model": companion,
                            "lease_slots": lease_slots,
                        }
                    ),
                    case.combined,
                )
            self._plan(case)
            required = case.check_work_count
            confirmations = [
                combined_case(evidence, device=device, tier=tier, card_total_mb=_card_total_mb(service, device))
                for device, tier in anchors
            ]
            reserve = (
                sum(_confirmation_work_count(item) if item is not None else 1 for item in confirmations)
                if self.validate
                else 0
            )
            baseline_passed = any(
                item.capability.kind is CapabilityKind.BASELINE
                and item.status is EvidenceStatus.MEASURED_SUCCESS
                and item.profile.device_index == case.profile.device_index
                and item.profile.tier is case.profile.tier
                for item in evidence
            )
            can_run = case.capability.kind is CapabilityKind.BASELINE or baseline_passed
            room = self.budget.jobs_offered + required <= self.budget.max_jobs - reserve
            confirmation_time = (self.budget.seconds or 0) * 0.2 if self.validate else 0
            time_room = time.monotonic() < deadline - confirmation_time
            responsive = worker.alive() and not worker_unresponsive(service)
            if not can_run or not room or not time_room or not responsive:
                if not any(item.evidence_id == case.case_id for item in evidence):
                    code, reason = (
                        ("prerequisite", "Not tested: the baseline did not pass")
                        if not can_run
                        else ("work_limit", "Not tested: the remaining job allowance is reserved for confirmation")
                        if not room
                        else ("time_limit", "Not tested: the remaining time is reserved for confirmation")
                        if not time_room
                        else ("worker_exited", "Not tested: the worker exited")
                        if not worker.alive()
                        else ("worker_unresponsive", "Not tested: the worker stopped responding")
                    )
                    item = self._unknown(case, reason).model_copy(update={"reason_code": code})
                    evidence.append(item)
                    self._finish(item)
                continue
            probe_deadline = deadline - confirmation_time
            if self.budget.seconds is not None:
                previous = [
                    item.elapsed_seconds
                    for item in evidence
                    if item.elapsed_seconds > 0 and item.status is EvidenceStatus.MEASURED_SUCCESS
                ]
                needed = min(previous, default=1.0)
                if probe_deadline - time.monotonic() < needed:
                    result = self._unknown(
                        case, "Not tested: too little time remains to prepare this check"
                    ).model_copy(update={"reason_code": "time_limit"})
                    evidence.append(result)
                    self._finish(result)
                    continue
            result = await self._execute(service, worker, case, probe_deadline, reserved_jobs=reserve)
            evidence[:] = [item for item in evidence if item.evidence_id != result.evidence_id]
            evidence.append(result)
            boundary = next_boundary(
                result, include_concurrency=self.options.include_concurrency and not self.only_probe
            )
            key = (case.profile.device_index, case.profile.tier, case.capability.kind)
            explored[key] = explored.get(key, 0) + 1
            if case.capability.kind is not CapabilityKind.BASELINE and explored[key] >= 2:
                boundary = None
            if boundary is not None and boundary.capability.kind in self.options.excluded_kinds:
                boundary = None
            # Dispatch keeps one model on one lane, so sampling overlap cannot exceed the installed-model count.
            if (
                boundary is not None
                and boundary.capability.kind is CapabilityKind.THREADS
                and boundary.profile.threads > len(self._installed)
            ):
                boundary = None
            if boundary is not None:
                # Larger batches require more memory. Stop growing after a measured card approaches
                # its capacity. Record untested settings separately from measured failures.
                snapshot = service.snapshot
                card = (
                    next((item for item in snapshot.per_card if item.device_index == case.profile.device_index), None)
                    if snapshot
                    else None
                )
                if (
                    result.device_peak_mb
                    and card
                    and card.total_vram_mb
                    and result.device_peak_mb * 1.5 >= card.total_vram_mb
                ):
                    evidence.append(
                        WorkloadEvidence(
                            evidence_id=boundary.case_id,
                            capability=boundary.capability,
                            profile=boundary.profile,
                            status=EvidenceStatus.PREDICTED,
                            predicted_from=[result.evidence_id],
                            reason="Larger settings may need most of the card's memory. "
                            "Offloading can change this estimate",
                        )
                    )
                else:
                    evidence.append(
                        WorkloadEvidence(
                            evidence_id=boundary.case_id,
                            capability=boundary.capability,
                            profile=boundary.profile,
                            status=EvidenceStatus.PREDICTED,
                            predicted_from=[result.evidence_id],
                            reason="extrapolated from the preceding successful workload; not directly confirmed",
                        )
                    )
                    boundary = AssessmentCase(
                        boundary.capability, boundary.profile.model_copy(update={"work_jobs": None})
                    )
                    cases.append(boundary)
            if (
                case.capability.kind is CapabilityKind.BASELINE
                and baseline_passed is False
                and result.status is EvidenceStatus.MEASURED_SUCCESS
                and self.options.include_concurrency
                and CapabilityKind.THREADS not in self.options.excluded_kinds
                and not self.only_probe
            ):
                cases.append(
                    AssessmentCase(
                        Capability(tier=case.profile.tier, kind=CapabilityKind.THREADS, magnitude=2),
                        case.profile.model_copy(update={"threads": 2}),
                    )
                )
            if (
                case.capability.kind is CapabilityKind.BASELINE
                and result.status is EvidenceStatus.MEASURED_SUCCESS
                and self.options.include_features
                and CapabilityKind.RESOLUTION not in self.options.excluded_kinds
                and not self.only_probe
            ):
                cases.append(
                    AssessmentCase(
                        Capability(
                            tier=case.profile.tier,
                            kind=CapabilityKind.RESOLUTION,
                            magnitude=(math.isqrt(case.profile.width**2 * 2) // 64) * 64,
                        ),
                        case.profile.model_copy(
                            update={
                                "width": (math.isqrt(case.profile.width**2 * 2) // 64) * 64,
                                "height": (math.isqrt(case.profile.height**2 * 2) // 64) * 64,
                            }
                        ),
                    )
                )
        if not self.validate:
            return
        confirmations = []
        for device, tier in sorted(anchors):
            case = combined_case(evidence, device=device, tier=tier, card_total_mb=_card_total_mb(service, device))
            if case is None:
                continue
            if case.profile.threads > 1:
                case = replace(case, profile=case.profile.model_copy(update={"work_jobs": case.profile.threads * 2}))
            confirmations.append(case)
        for confirmation_index, case in enumerate(confirmations):
            reserve = sum(_confirmation_work_count(pending) for pending in confirmations[confirmation_index + 1 :])
            follow_up = _single_thread_follow_up(case) if case.profile.threads > 1 else None
            result = await self._confirm(
                service,
                worker,
                case,
                deadline,
                reserved_jobs=reserve + (follow_up.check_work_count if follow_up is not None else 0),
            )
            evidence.append(result)
            # A joint refused only for its thread count still carries batch, resolution and queue limits that
            # one lane can confirm. A follow-up is at one thread, so it is never followed up.
            if (
                follow_up is None
                or result.status is EvidenceStatus.MEASURED_SUCCESS
                or result.reason_code not in THREAD_COUNT_REASON_CODES
            ):
                continue
            evidence.append(await self._confirm(service, worker, follow_up, deadline, reserved_jobs=reserve))

    async def _confirm(
        self,
        service: LocalHorde,
        worker: AssessmentWorker,
        case: AssessmentCase,
        deadline: float,
        *,
        reserved_jobs: int,
    ) -> WorkloadEvidence:
        """Run one combined confirmation, or record why the worker, time or work limit does not allow it."""
        self._plan(case)
        if not worker.alive() or worker_unresponsive(service):
            code, reason = (
                ("worker_exited", "Not tested: the worker exited")
                if not worker.alive()
                else ("worker_unresponsive", "Not tested: the worker stopped responding")
            )
            item = self._unknown(case, reason).model_copy(update={"reason_code": code})
            self._finish(item)
            return item
        if time.monotonic() >= deadline or self.budget.jobs_offered + case.check_work_count > self.budget.max_jobs:
            item = self._unknown(
                case, "Not tested: the remaining work or time limit does not allow confirmation"
            ).model_copy(update={"reason_code": "confirmation_limit"})
            self._finish(item)
            return item
        return await self._execute(service, worker, case, deadline, reserved_jobs=reserved_jobs)

    def _configured_lease_slots(self) -> int | None:
        value = self._source_config.get("gpu_sampling_lease_slots")
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def _expected_sampling_concurrency(self, case: AssessmentCase) -> int:
        """Resolve how many lanes can denoise at once under the check's lease settings.

        The worker resolves the lease slot count from the same inputs, so the check measures the overlap the
        configuration permits rather than the thread count alone. A profile's explicit slot count replaces the
        operator's lease settings.
        """
        from horde_worker_regen.process_management.process_manager import resolve_card_concurrency

        profile = case.profile
        lease_enabled = (
            profile.lease_slots is not None or self._source_config.get("gpu_sampling_lease_enabled") is True
        )
        if profile.threads == 1 or not lease_enabled:
            return profile.threads
        concurrency = resolve_card_concurrency(
            max_threads=profile.threads,
            queue_size=profile.queue_size,
            num_models_to_load=2 if profile.companion_model else 1,
            gpu_sampling_lease_enabled=True,
            gpu_sampling_lease_slots=profile.lease_slots
            if profile.lease_slots is not None
            else self._configured_lease_slots(),
            max_threads_ceiling=profile.threads,
        )
        return max(1, min(profile.threads, concurrency.gpu_sampling_lease_slots))

    def _companion_model(self, case: AssessmentCase, evidence: list[WorkloadEvidence]) -> str | None:
        """Pick a second installed model for the other lane, preferring one whose baseline passed here."""
        others = sorted({name for name in self._installed.values() if name != case.profile.model_name})
        if not others:
            return None
        passed = {
            item.profile.model_name
            for item in evidence
            if item.profile.device_index == case.profile.device_index
            and item.capability.kind is CapabilityKind.BASELINE
            and item.status is EvidenceStatus.MEASURED_SUCCESS
        }
        return next((name for name in others if name in passed), others[0])

    @staticmethod
    def _unknown(case: AssessmentCase, reason: str) -> WorkloadEvidence:
        return WorkloadEvidence(
            evidence_id=case.case_id,
            capability=case.capability,
            profile=case.profile,
            status=EvidenceStatus.UNKNOWN,
            reason=reason,
            scope=EvidenceScope.SHARED_WORKER if case.profile.alchemy_forms else EvidenceScope.CARD,
            combined_validation=case.combined,
            notes=list(case.notes),
        )

    def _finish(self, result: WorkloadEvidence) -> None:
        labels = {
            EvidenceStatus.MEASURED_SUCCESS: "passed",
            EvidenceStatus.MEASURED_FAILURE: "failed",
            EvidenceStatus.PREDICTED: "estimated",
            EvidenceStatus.UNKNOWN: "inconclusive" if result.attempted else "not tested",
        }
        if result.reason_code == "unavailable":
            labels[EvidenceStatus.UNKNOWN] = "unavailable"
        if result.reason_code == "not_applicable":
            labels[EvidenceStatus.UNKNOWN] = "not applicable"
        reasons = [result.reason]
        for description in (describe_wall_throughput(result), describe_lane_throughput(result)):
            if description is not None:
                reasons.append(description)
        self.progress.emit(
            LevelFinished(
                level_id=result.evidence_id,
                outcome=labels[result.status],
                reasons=reasons,
                evidence_status=result.status.value,
                attempted=result.attempted,
                reason_code=result.reason_code,
                jobs_completed=result.jobs_completed,
                jobs_expected=len(result.job_ids) if result.job_ids else None,
                jobs_faulted=result.jobs_faulted,
                elapsed_seconds=result.elapsed_seconds,
                its_p50=result.sampling_its,
                vram_used_high_water_mb=result.device_peak_mb,
            )
        )

    def _live_progress(
        self,
        service: LocalHorde,
        case: AssessmentCase,
        ids: list[str],
        started: float,
        *,
        jobs_expected: int | None = None,
        earlier_ids: list[str] | None = None,
    ) -> None:
        self._read_records()
        snapshot = service.snapshot
        processes = [p for p in snapshot.processes if p.current_job_id in ids and p.is_busy] if snapshot else []
        active = [
            AssessmentJobProgress(
                job_id=p.current_job_id or "",
                process_id=p.process_id,
                state=p.last_process_state.replace("_", " ").lower(),
                current_step=p.last_current_step,
                total_steps=p.last_total_steps,
                iterations_per_second=p.last_iterations_per_second
                if p.last_heartbeat_type == "INFERENCE_STEP"
                else None,
            )
            for p in processes
        ]
        rates = [p.iterations_per_second for p in active if p.iterations_per_second is not None]
        records = [self._records[key] for key in ids if key in self._records]
        completed_rates = [
            r.phase_metrics.sampling.iterations_per_second
            for r in records
            if r.phase_metrics is not None
            and r.phase_metrics.sampling is not None
            and r.phase_metrics.sampling.iterations_per_second is not None
        ]
        card = (
            next((c for c in snapshot.per_card if c.device_index == case.profile.device_index), None)
            if snapshot
            else None
        )
        used = (
            round(card.total_vram_mb - card.free_vram_mb)
            if card is not None and card.total_vram_mb is not None and card.free_vram_mb is not None
            else None
        )
        sample = snapshot.latest_stats_sample if snapshot else None
        busy = (
            sample.gpu_busy_fraction * 100
            if sample and sample.gpu_busy_fraction is not None and snapshot is not None and len(snapshot.per_card) == 1
            else None
        )
        done = sum(key in self._records and service.work[key].finalized for key in ids)
        faults = sum(service.work[key].faulted for key in ids)
        loading = [
            load.duration_seconds
            for record in records
            if record.phase_metrics is not None
            for load in record.phase_metrics.model_loads
            if load.phase == "disk_to_ram"
        ]
        phase = service.phase
        if ids:
            unissued_forms = [key for key in ids[case.image_work_count :] if not service.work[key].issued]
            if unissued_forms:
                phase = f"Waiting for the worker to take {', '.join(case.profile.alchemy_forms)}"
            elif active:
                phase = f"Running {case.profile.model_name}"
            elif any(service.work[key].issued for key in ids) and done < len(ids):
                phase = "Waiting for safety checks, upload or final records"
            elif not active and done < len(ids) and snapshot is not None:
                phase = snapshot.orchestration_intent.summary or "Waiting for the worker to take test jobs"
        earlier = earlier_ids or []
        self.progress.emit(
            LevelProgress(
                level_id=case.case_id,
                jobs_expected=jobs_expected if jobs_expected is not None else case.work_count,
                jobs_completed=done + sum(key in self._records and service.work[key].finalized for key in earlier),
                jobs_faulted=faults + sum(service.work[key].faulted for key in earlier),
                jobs_waiting=sum(not service.work[key].issued for key in ids),
                active_jobs=active,
                iterations_per_second=median(rates) if rates else None,
                recent_iterations_per_second=median(completed_rates) if completed_rates else None,
                vram_used_mb=used,
                gpu_busy_percent=busy,
                elapsed_seconds=time.monotonic() - started,
                phase=phase,
                process_summary=", ".join(
                    f"{p.process_id}: {p.last_process_state.lower().replace('_', ' ')}" for p in snapshot.processes
                )
                if snapshot
                else "",
                num_process_recoveries=snapshot.num_process_recoveries if snapshot else 0,
                worker_update_age=max(0, time.time() - snapshot.timestamp) if snapshot else None,
                model_loading_seconds=sum(loading) if loading else None,
                run_jobs_completed=len(self._records),
            )
        )

    def _finished_count(self, service: LocalHorde, ids: list[str]) -> int:
        """Count the jobs with a submit receipt and a finalized record."""
        return sum(key in self._records and service.work[key].finalized for key in ids)

    def _finished_cleanly(self, service: LocalHorde, ids: list[str]) -> bool:
        """Whether every job finalized with a record and none faulted."""
        return bool(ids) and all(
            key in self._records
            and service.work[key].finalized
            and not service.work[key].faulted
            and not self._records[key].faulted
            for key in ids
        )

    def _sampling_seconds_total(self, ids: list[str]) -> float | None:
        """Sum the image jobs' reported sampling seconds, None when no job reported one."""
        sampled = [
            self._records[key].sampling_seconds
            for key in ids
            if key in self._records
            and self._records[key].workload is WorkloadKind.IMAGE_GENERATION
            and self._records[key].sampling_seconds is not None
        ]
        return sum(value for value in sampled if value is not None) if sampled else None

    @staticmethod
    def _thread_count_shortfall(
        profile: OperatingProfile,
        *,
        concurrency: int,
        throughput: float | None,
        check_rates: dict[str, float],
        references: dict[str, ReferenceSamplingRate],
        wall: PhaseWallThroughput | None,
        single_wall: PhaseWallThroughput | None,
        wall_gain_required: float | None,
    ) -> tuple[str, str] | None:
        """Return the reason code and reason a concurrency check's thread count is not confirmed, or None.

        Without a required gain only the thrashing gate applies, as for a combined confirmation whose thread
        count its thread check already ranked.
        """
        jobs = describe_jobs_per_minute(
            profile,
            jobs_per_minute=wall.jobs_per_minute if wall is not None else None,
            single_lane_jobs_per_minute=single_wall.jobs_per_minute if single_wall is not None else None,
        )
        if concurrency >= 2 and throughput is not None and throughput < SINGLE_LANE_THROUGHPUT:
            return (
                "throughput_below_single_lane",
                f"Jobs finished and {concurrency} sampling threads overlapped, but together they sampled "
                f"{throughput:.2f} times one lane's rate: {describe_sampling_rates(check_rates, references)}. "
                f"Partly unloaded weights can cause this slowdown. {jobs}. "
                "The thread count was not confirmed",
            )
        if wall_gain_required is None:
            return None
        if wall is None or single_wall is None:
            return (
                "throughput_unmeasured",
                "Jobs finished, but their records did not include the timestamps or megapixelsteps needed "
                f"to compare {profile.threads} lanes with one lane. The thread count was not confirmed",
            )
        gain = wall.megapixelsteps_per_second / single_wall.megapixelsteps_per_second
        if gain <= SINGLE_LANE_WALL_GAIN or gain < wall_gain_required:
            return (
                "wall_gain_below_noise",
                f"Jobs finished. {jobs}. The throughput ratio was {gain:.2f}; confirmation requires an improvement "
                f"beyond the single-lane variation (minimum ratio {wall_gain_required:.2f}). "
                "The thread count was not confirmed",
            )
        return None

    async def _run_phase(
        self,
        service: LocalHorde,
        worker: AssessmentWorker,
        level: AssessmentCase,
        phase: AssessmentCase,
        deadline: float,
        started: float,
        *,
        jobs_expected: int,
        earlier_ids: list[str],
    ) -> _PhaseRun:
        """Apply one phase's settings, offer its jobs and wait for their final records or a stop.

        Progress is reported under the level's check, counting the jobs of earlier phases as finished work.
        """
        configure_started = time.monotonic()
        configuration = asyncio.create_task(worker.configure(phase, deadline))
        configuration_watch = AssessmentProgressWatch(service, self._idle_timeout())
        refusal: str | None = None
        try:
            while not configuration.done():
                self._live_progress(service, level, [], started, jobs_expected=jobs_expected, earlier_ids=earlier_ids)
                if configuration_watch.stalled():
                    refusal = configuration_watch.reason
                    break
                await asyncio.wait({configuration}, timeout=1)
            if configuration.done():
                refusal = await configuration
        finally:
            if not configuration.done():
                configuration.cancel()
                with suppress(asyncio.CancelledError):
                    await configuration
        configure_seconds = time.monotonic() - configure_started
        if refusal is not None:
            return _PhaseRun(ids=[], configure_seconds=configure_seconds, refusal=refusal)
        scenario = phase.scenario()
        initial_form_offers = service.forms_offer_count
        ids = service.enqueue(scenario.expand_image_jobs(), scenario.expand_alchemy_forms())
        self.budget.jobs_offered += len(ids)
        next_progress = started
        watch = AssessmentProgressWatch(service, self._idle_timeout())
        stop_reason = ""
        unavailable_forms = False
        while worker.alive() and time.monotonic() < deadline:
            self._read_records()
            receipts_done = all(service.work[identifier].finalized for identifier in ids)
            # A receipt precedes parent-side finalization. Wait for the exported terminal record
            # too, preserving the same end-to-end accounting used by ordinary worker telemetry.
            if receipts_done and all(identifier in self._records for identifier in ids):
                break
            if phase.profile.alchemy_forms and service.forms_offer_count >= initial_form_offers + 2:
                missing = [
                    form for form in phase.profile.alchemy_forms if form not in (service.offered_forms or set())
                ]
                unissued = [key for key in ids[phase.image_work_count :] if not service.work[key].issued]
                if missing and unissued:
                    stop_reason = (
                        f"The worker is not offering {', '.join(missing)}. "
                        "Check the required model and serving process before rerunning this check"
                    )
                    unavailable_forms = True
                    break
            if watch.stalled():
                stop_reason = watch.reason
                break
            now = time.monotonic()
            if now >= next_progress:
                self._live_progress(service, level, ids, started, jobs_expected=jobs_expected, earlier_ids=earlier_ids)
                next_progress = now + 1
            await asyncio.sleep(0.1)
        self._live_progress(service, level, ids, started, jobs_expected=jobs_expected, earlier_ids=earlier_ids)
        return _PhaseRun(
            ids=ids, configure_seconds=configure_seconds, stop_reason=stop_reason, unavailable_forms=unavailable_forms
        )

    async def _execute(
        self,
        service: LocalHorde,
        worker: AssessmentWorker,
        case: AssessmentCase,
        deadline: float,
        *,
        retry_overlap: bool = True,
        emit_finished: bool = True,
        reserved_jobs: int = 0,
        single_lane_ids: list[str] | None = None,
    ) -> WorkloadEvidence:
        """Run one check and judge its evidence.

        A concurrency check runs its jobs at one lane under the operator's lease settings first, then at its
        thread count under the profile's, and ranks the two by megapixelsteps per second of wall time. The
        single-lane run is also the sampling-rate reference for the lane-equivalent diagnostic. An overlap
        follow-up passes ``single_lane_ids`` so only the thread-count run repeats.
        """
        if case.profile.threads > 1 and retry_overlap:
            # Double the baseline steps so sampling lasts beyond pop and dispatch delays on warm lanes.
            # Covering a full model load would make these checks too long. The follow-up reuses this count.
            maximum = self._source_config.get("max_steps", 150)
            case = replace(
                case,
                profile=case.profile.model_copy(
                    update={"steps": min(int(maximum) if isinstance(maximum, int) else 150, case.profile.steps * 2)}
                ),
            )
        self._plan(case)
        concurrent = case.profile.threads > 1
        # A combined confirmation proves the joint profile under the configuration its thread check ranked, so
        # repeating the single-lane reference would double the work without testing another configuration.
        runs_single_lane = concurrent and single_lane_ids is None and not case.combined
        reference_ids = list(single_lane_ids) if single_lane_ids is not None else []
        jobs_expected = case.work_count * 2 if runs_single_lane else case.work_count
        self.progress.emit(
            LevelStarted(
                level_id=case.case_id,
                stage="assessment",
                jobs_expected=jobs_expected,
                description=self._description(case),
                tier=str(case.profile.tier),
                axis=str(case.capability.kind),
            )
        )
        started = time.monotonic()
        snapshot = service.snapshot
        recoveries = snapshot.num_process_recoveries if snapshot else 0
        self.progress.emit(
            LevelProgress(
                level_id=case.case_id,
                jobs_expected=jobs_expected,
                phase="Preparing test settings and inference processes",
            )
        )
        configure_seconds: float | None = None
        refusal: str | None = None
        stop_reason = ""
        unavailable_forms = False
        check_ids: list[str] = []
        run_check = True
        if runs_single_lane:
            single_lane_case = AssessmentCase(
                case.capability, case.profile.model_copy(update={"threads": 1, "lease_slots": None}), case.combined
            )
            single_lane = await self._run_phase(
                service, worker, case, single_lane_case, deadline, started, jobs_expected=jobs_expected, earlier_ids=[]
            )
            reference_ids = single_lane.ids
            configure_seconds = single_lane.configure_seconds
            refusal, stop_reason = single_lane.refusal, single_lane.stop_reason
            unavailable_forms = single_lane.unavailable_forms
            # A reference that did not finish cleanly cannot rank the thread count, so its jobs are not offered.
            run_check = refusal is None and not stop_reason and self._finished_cleanly(service, reference_ids)
        if run_check:
            check = await self._run_phase(
                service,
                worker,
                case,
                case,
                deadline,
                started,
                jobs_expected=jobs_expected,
                earlier_ids=reference_ids if runs_single_lane else [],
            )
            check_ids = check.ids
            configure_seconds = _add_optional(configure_seconds, check.configure_seconds)
            refusal, stop_reason, unavailable_forms = check.refusal, check.stop_reason, check.unavailable_forms
        ids = reference_ids + check_ids
        if refusal is not None:
            result = self._unknown(case, refusal).model_copy(
                update={
                    "attempted": "withheld" not in refusal,
                    "reason_code": "unavailable" if "withheld" in refusal else "configuration",
                    "elapsed_seconds": time.monotonic() - started,
                    "configure_seconds": configure_seconds,
                    "job_ids": ids,
                    "jobs_completed": self._finished_count(service, ids),
                }
            )
        else:
            records = [self._records[identifier] for identifier in ids if identifier in self._records]
            image_records = [record for record in records if record.workload is WorkloadKind.IMAGE_GENERATION]
            check_records = [self._records[identifier] for identifier in check_ids if identifier in self._records]
            check_image_records = [
                record for record in check_records if record.workload is WorkloadKind.IMAGE_GENERATION
            ]
            reference_image_records = [
                self._records[identifier]
                for identifier in reference_ids
                if identifier in self._records and self._records[identifier].workload is WorkloadKind.IMAGE_GENERATION
            ]
            retry_provenance_complete = bool(image_records) and all(
                record.inference_attempts is not None
                and record.resource_failures is not None
                and record.degraded_retry_used is not None
                and record.host_commit_refusals is not None
                for record in image_records
            )
            inference_failures = sum(record.inference_attempts or 0 for record in image_records)
            resource_failures = sum(record.resource_failures or 0 for record in image_records)
            degraded_retries = sum(bool(record.degraded_retry_used) for record in image_records)
            host_refusals = sum(record.host_commit_refusals or 0 for record in image_records)
            retried = bool(inference_failures or resource_failures or degraded_retries or host_refusals)
            shared_service = bool(case.profile.alchemy_forms)
            placements = [
                ServicePlacement(
                    job_id=record.job_id, form=record.model_name, reported_device_index=record.device_index
                )
                for record in records
                if record.workload is WorkloadKind.ALCHEMY
            ]
            concurrency = observed_sampling_concurrency(check_records, case.profile.device_index)
            expected_concurrency = self._expected_sampling_concurrency(case)
            concurrency_confirmed = expected_concurrency == 1 or concurrency >= expected_concurrency
            check_rates: dict[str, float] = {}
            references: dict[str, ReferenceSamplingRate] = {}
            throughput: float | None = None
            wall: PhaseWallThroughput | None = None
            single_wall: PhaseWallThroughput | None = None
            wall_gain: float | None = None
            wall_gain_required: float | None = None
            if concurrent:
                check_rates = model_sampling_rates(check_image_records)
                shapes: dict[str, set[ShapeKey]] = {}
                for record in check_image_records:
                    if record.model_name is not None:
                        shapes.setdefault(record.model_name, set()).add(job_shape(record))
                # A combined confirmation has no single-lane run of its own. Its references come from the run's
                # records at the same shapes, where the thread check's single-lane jobs sampled alone.
                references = reference_sampling_rates(
                    list(self._records.values()) if case.combined else reference_image_records,
                    case.profile.device_index,
                    list(check_rates),
                    shapes=shapes,
                )
                throughput = lane_equivalent_throughput(
                    check_image_records, {name: item.iterations_per_second for name, item in references.items()}
                )
                if not case.combined:
                    wall = phase_wall_throughput(check_image_records)
                    single_wall = phase_wall_throughput(reference_image_records)
                    wall_gain_required = SINGLE_LANE_WALL_GAIN + wall_noise_margin(reference_image_records)
                    if wall is not None and single_wall is not None:
                        wall_gain = wall.megapixelsteps_per_second / single_wall.megapixelsteps_per_second
            snapshot = service.snapshot
            recovered = max(0, snapshot.num_process_recoveries - recoveries) if snapshot else 0
            complete = len(records) == len(ids) and all(service.work[identifier].finalized for identifier in ids)
            # The thread-count run is not offered after a faulted or unfinished single-lane run, and a check that
            # did not run it says nothing about overlap or throughput.
            checked = complete and bool(check_ids)
            lease_waits = clearance_waits(image_records)
            longest_lease_wait = max(lease_waits, key=lambda wait: wait.seconds, default=None)
            late_sampling = (
                max(
                    (wait for wait in lease_waits if wait.reached_timeout), key=lambda wait: wait.seconds, default=None
                )
                if checked
                else None
            )
            shortfall = (
                self._thread_count_shortfall(
                    case.profile,
                    concurrency=concurrency,
                    throughput=throughput,
                    check_rates=check_rates,
                    references=references,
                    wall=wall,
                    single_wall=single_wall,
                    wall_gain_required=wall_gain_required,
                )
                if checked and concurrent
                else None
            )
            faulted = any(record.faulted for record in records) or any(
                service.work[identifier].faulted for identifier in ids
            )
            attributed = all(
                record.device_index == case.profile.device_index
                for record in records
                if record.workload is WorkloadKind.IMAGE_GENERATION
            )
            peaks = [
                record.phase_metrics.vram_device_used_high_water_mb
                for record in records
                if record.device_index == case.profile.device_index
                and record.phase_metrics is not None
                and record.phase_metrics.vram_device_used_high_water_mb is not None
            ]
            ram = [
                record.phase_metrics.ram_used_high_water_mb
                for record in records
                if record.phase_metrics is not None and record.phase_metrics.ram_used_high_water_mb is not None
            ]
            status = (
                EvidenceStatus.MEASURED_FAILURE
                if faulted
                else EvidenceStatus.MEASURED_SUCCESS
                if checked
                and attributed
                and not recovered
                and not retried
                and retry_provenance_complete
                and peaks
                and late_sampling is None
                and concurrency_confirmed
                and shortfall is None
                else EvidenceStatus.UNKNOWN
            )
            missing_details: list[str] = []
            if len(records) != len(ids):
                missing_details.append(
                    f"final records for {len(ids) - len(records)} of {len(ids)} jobs were not reported"
                )
            unfinished_jobs = sum(not service.work[identifier].finalized for identifier in ids)
            if unfinished_jobs:
                missing_details.append(f"{unfinished_jobs} of {len(ids)} jobs have not submitted results")
            if not check_ids:
                missing_details.append("the requested workload was not run")
            unknown_cards = sum(record.device_index is None for record in image_records)
            if unknown_cards:
                missing_details.append(
                    f"the execution card was not reported for {unknown_cards} "
                    f"image job{'s' if unknown_cards != 1 else ''}"
                )
            other_cards = sum(
                record.device_index is not None and record.device_index != case.profile.device_index
                for record in image_records
            )
            if other_cards:
                missing_details.append(
                    f"{other_cards} image job{'s' if other_cards != 1 else ''} ran on a different card"
                )
            if not peaks:
                missing_details.append("device peak memory was not reported for the selected card")
            incomplete_reason = "Settings could not be confirmed: " + "; ".join(missing_details) + "."
            reason = (
                "At least one job faulted."
                if faulted
                else "A worker process was replaced during the check. This run could not confirm the settings."
                if recovered
                else (
                    f"Jobs completed after inference recovery: {inference_failures} failed attempts, "
                    f"{resource_failures} resource failures, {degraded_retries} degraded retries, "
                    f"{host_refusals} host commit refusals. This run could not confirm the settings."
                )
                if complete and retried
                else (
                    "Jobs completed, but image retry history was not reported. "
                    "This run could not confirm the settings."
                )
                if complete and not retry_provenance_complete
                else describe_clearance_timeout(late_sampling)
                if late_sampling is not None
                else (
                    f"Jobs finished, but at most {concurrency} of {expected_concurrency} sampling threads overlapped. "
                    "Concurrent sampling was not confirmed"
                )
                if checked and not concurrency_confirmed
                else shortfall[1]
                if shortfall is not None
                else (
                    f"Jobs finished, passed safety checks and were submitted on the selected card. The sampling "
                    f"lease permits {expected_concurrency} concurrent denoise loops and the other "
                    f"{case.profile.threads - expected_concurrency} lanes staged"
                )
                if status is EvidenceStatus.MEASURED_SUCCESS
                and not shared_service
                and expected_concurrency < case.profile.threads
                else (
                    "companion image completed on the selected card and service forms completed "
                    "through shared-worker placement. "
                    "The reported lane for each form is listed separately. The execution device was not reported"
                    if shared_service
                    else "Jobs finished, passed safety checks and were submitted on the selected card"
                )
                if status is EvidenceStatus.MEASURED_SUCCESS
                else stop_reason or incomplete_reason
            )
            image_rates = [
                record.phase_metrics.sampling.iterations_per_second
                for record in image_records
                if record.phase_metrics is not None
                and record.phase_metrics.sampling is not None
                and record.phase_metrics.sampling.iterations_per_second is not None
            ]
            result = WorkloadEvidence(
                evidence_id=case.case_id,
                capability=case.capability,
                profile=case.profile,
                status=status,
                reason=reason,
                scope=EvidenceScope.SHARED_WORKER if shared_service else EvidenceScope.CARD,
                service_placements=placements,
                job_ids=ids,
                elapsed_seconds=time.monotonic() - started,
                configure_seconds=configure_seconds,
                sampling_seconds=self._sampling_seconds_total(ids),
                clearance_wait_seconds=longest_lease_wait.seconds if longest_lease_wait is not None else None,
                recoveries=recovered,
                inference_failures=inference_failures,
                resource_failures=resource_failures,
                degraded_retries=degraded_retries,
                host_commit_refusals=host_refusals,
                retry_provenance_complete=retry_provenance_complete,
                observed_sampling_concurrency=concurrency,
                expected_sampling_concurrency=expected_concurrency,
                lane_equivalent_throughput=throughput,
                wall_megapixelsteps_per_second=wall.megapixelsteps_per_second if wall is not None else None,
                single_lane_wall_megapixelsteps_per_second=single_wall.megapixelsteps_per_second
                if single_wall is not None
                else None,
                wall_gain=wall_gain,
                wall_gain_required=wall_gain_required,
                jobs_per_minute=wall.jobs_per_minute if wall is not None else None,
                single_lane_jobs_per_minute=single_wall.jobs_per_minute if single_wall is not None else None,
                model_sampling_rates=check_rates,
                model_reference_rates=references,
                device_peak_mb=max(peaks) if peaks else None,
                ram_peak_mb=max(ram) if ram else None,
                combined_validation=case.combined,
                notes=list(case.notes),
                attempted=True,
                reason_code="unavailable"
                if unavailable_forms
                else "clearance_timeout"
                if late_sampling is not None
                else "sampling_not_overlapped"
                if checked and not concurrency_confirmed
                else shortfall[0]
                if shortfall is not None
                else "worker_stalled"
                if stop_reason
                else "measured"
                if status is EvidenceStatus.MEASURED_SUCCESS
                else "incomplete",
                jobs_completed=self._finished_count(service, ids),
                jobs_faulted=sum(service.work[key].faulted for key in ids),
                sampling_its=median(image_rates) if image_rates else None,
            )
            if not complete:
                service.withdraw_unissued()
        if (
            retry_overlap
            and result.reason_code == "sampling_not_overlapped"
            and result.retry_provenance_complete
            and not (
                result.inference_failures
                or result.resource_failures
                or result.degraded_retries
                or result.host_commit_refusals
                or result.recoveries
                or result.jobs_faulted
            )
            and self.budget.jobs_offered + case.work_count <= self.budget.max_jobs - reserved_jobs
            and time.monotonic() < deadline
        ):
            repeated = await self._execute(
                service,
                worker,
                case,
                deadline,
                retry_overlap=False,
                emit_finished=False,
                reserved_jobs=reserved_jobs,
                single_lane_ids=reference_ids if concurrent else None,
            )
            merged_ids = list(dict.fromkeys(result.job_ids + repeated.job_ids))
            result = repeated.model_copy(
                update={
                    "job_ids": merged_ids,
                    "elapsed_seconds": result.elapsed_seconds + repeated.elapsed_seconds,
                    "configure_seconds": _add_optional(result.configure_seconds, repeated.configure_seconds),
                    "sampling_seconds": self._sampling_seconds_total(merged_ids),
                    "jobs_completed": self._finished_count(service, merged_ids),
                    "jobs_faulted": sum(service.work[key].faulted for key in merged_ids),
                }
            )
        if emit_finished:
            self._finish(result)
        logger.info(f"Capability evidence: {result.model_dump_json()}")
        return result
