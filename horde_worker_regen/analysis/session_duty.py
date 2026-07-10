"""Stats-backed offline GPU duty-cycle session analysis.

The live worker exports typed JSONL events under ``.horde_worker_regen/stats``. This module treats
those files as the structured spine for an offline report: ``stats_sample`` events describe the worker
state over time, while ``job_completed`` events provide phase metrics for the work that finished in the
same session. Log parsing remains a fallback in :mod:`horde_worker_regen.analysis.duty_log_report`.
"""

from __future__ import annotations

import gzip
import json
import re
import statistics
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import IO, Any, cast

from horde_worker_regen.stats_operations import default_stats_dir

_STATS_SESSION_RE = re.compile(r"^stats-v(?P<version>.+)-(?P<stamp>\d{8}-\d{6})-(?P<index>\d+)\.jsonl(?:\.gz)?$")
_DEFAULT_TARGET_DUTY_PERCENT = 90.0
_DEFAULT_BUSY_THRESHOLD = 0.10
_MAX_INFERRED_INTERVAL_SECONDS = 300.0


class DutyLossKind(StrEnum):
    """Maintainer-facing buckets for GPU duty-cycle loss attribution."""

    DEMAND_LIMITED = "demand_limited"
    SCHEDULER_WAIT = "scheduler_wait"
    MODEL_LOAD = "model_load"
    MODEL_UNLOAD = "model_unload"
    VRAM_TRANSFER = "vram_transfer"
    AUX_DOWNLOAD = "aux_download"
    SAFETY = "safety"
    SUBMIT = "submit"
    POST_PROCESSING = "post_processing"
    PROCESS_RECOVERY = "process_recovery"
    LOCAL_PAUSE = "local_pause"
    API_BACKOFF = "api_backoff"
    UNKNOWN = "unknown"


@dataclass
class DutyLossBucket:
    """Aggregated wall-clock-equivalent loss for one duty attribution bucket."""

    kind: DutyLossKind
    idle_seconds: float = 0.0
    partial_utilization_seconds: float = 0.0
    idle_samples: int = 0
    partial_samples: int = 0
    evidence: list[str] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        """Total loss seconds in this bucket."""
        return self.idle_seconds + self.partial_utilization_seconds


@dataclass
class DutyWindowBreakdown:
    """Attribution for one interval between adjacent stats samples."""

    start_timestamp: float
    end_timestamp: float
    duration_seconds: float
    gpu_duty_percent: float | None
    gpu_busy_fraction: float | None
    idle_kind: DutyLossKind
    partial_kind: DutyLossKind
    idle_seconds: float
    partial_utilization_seconds: float
    evidence: str = ""


@dataclass
class InferenceQueueWaitModelBreakdown:
    """Per-model contribution to popped-job wait before inference starts."""

    model_name: str
    jobs: int
    total_seconds: float
    median_seconds: float
    p90_seconds: float
    max_seconds: float
    sampling_seconds: float
    wait_to_sampling_ratio: float | None


@dataclass
class InferenceQueueWaitSummary:
    """Aggregate popped-job wait before inference starts."""

    jobs: int
    total_seconds: float
    median_seconds: float
    p90_seconds: float
    max_seconds: float
    sampling_seconds: float
    wait_to_sampling_ratio: float | None
    top_models: list[InferenceQueueWaitModelBreakdown] = field(default_factory=list)


@dataclass
class InferenceDispatchGapSummary:
    """Sampled intervals where inference work was queued but no inference job was active."""

    queued_seconds: float
    no_active_inference_seconds: float
    no_active_inference_fraction: float | None
    top_states: dict[str, float] = field(default_factory=dict)


@dataclass
class SessionDutyReport:
    """A complete stats-backed duty-cycle analysis for one worker stats session."""

    session_id: str
    source_files: list[str]
    start_timestamp: float | None
    end_timestamp: float | None
    sample_count: int
    completed_jobs: int
    mean_gpu_duty_percent: float | None
    mean_gpu_busy_fraction: float | None
    busy_fraction_percent: float | None
    target_duty_percent: float
    buckets: list[DutyLossBucket]
    windows: list[DutyWindowBreakdown]
    per_job_phase_medians: dict[str, float]
    per_job_phase_totals: dict[str, float]
    inference_queue_wait: InferenceQueueWaitSummary | None
    inference_dispatch_gap: InferenceDispatchGapSummary | None
    model_breakdown: dict[str, int]
    baseline_breakdown: dict[str, int]
    churn_per_hour: dict[str, float]
    churn_per_completed_job: dict[str, float]
    operator_recommendations: list[str]
    maintainer_notes: list[str]
    unknown_event_count: int = 0
    slot_duty_seconds: dict[str, float] = field(default_factory=dict)
    """Slot-seconds per slot-duty bucket over the session (difference of the worker's cumulative totals).

    Capacity-normalized: the shares sum to ~100% of ``capacity x wall``, so ``sampling`` is the
    productive share and every other bucket names what an empty slot was waiting on. Empty for stats
    files written before the worker recorded slot duty."""
    slot_duty_capacity: int | None = None
    """Configured concurrent-inference slot count the slot-duty totals are normalized against."""
    concurrency_occupancy: dict[str, float] = field(default_factory=dict)
    """Seconds spent at each concurrent in-flight job count (key = the count as text), from adjacent
    stats samples. The direct read of how much of the configured thread capacity actually ran."""

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation with enum values, not enum objects."""
        payload = asdict(self)
        for bucket in cast(list[dict[str, object]], payload["buckets"]):
            bucket["kind"] = str(bucket["kind"])
        for window in cast(list[dict[str, object]], payload["windows"]):
            window["idle_kind"] = str(window["idle_kind"])
            window["partial_kind"] = str(window["partial_kind"])
        return payload


def analyze_stats_sessions(
    stats_dir: Path | None = None,
    *,
    last: bool = False,
    target_duty_percent: float = _DEFAULT_TARGET_DUTY_PERCENT,
    busy_threshold: float = _DEFAULT_BUSY_THRESHOLD,
) -> list[SessionDutyReport]:
    """Analyze retained stats JSONL sessions from ``stats_dir``.

    Args:
        stats_dir: Directory containing ``stats-v*.jsonl`` and/or ``stats-v*.jsonl.gz`` files. Defaults to
            ``.horde_worker_regen/stats`` in the current working directory.
        last: Return only the newest stats session.
        target_duty_percent: Duty target used to compute partial-utilization shortfall.
        busy_threshold: Samples below this busy fraction count as near-idle for sample counters.

    Returns:
        One report per stats session, in chronological file order.
    """
    groups = discover_stats_sessions(stats_dir or default_stats_dir())
    if last and groups:
        groups = groups[-1:]
    return [
        analyze_stats_files(
            session_id=session_id,
            paths=paths,
            target_duty_percent=target_duty_percent,
            busy_threshold=busy_threshold,
        )
        for session_id, paths in groups
    ]


def discover_stats_sessions(stats_dir: Path) -> list[tuple[str, list[Path]]]:
    """Return stats files grouped by filename session stamp and sorted by rotation index."""
    if not stats_dir.exists():
        return []
    grouped: dict[str, list[Path]] = {}
    loose_index = 0
    for path in sorted(_iter_stats_files(stats_dir), key=_stats_file_order_key):
        match = _STATS_SESSION_RE.match(path.name)
        if match is None:
            session_id = f"loose-{loose_index:03d}-{path.stem}"
            loose_index += 1
        else:
            session_id = f"{match.group('stamp')} v{match.group('version')}"
        grouped.setdefault(session_id, []).append(path)
    return sorted(grouped.items(), key=lambda item: _stats_file_order_key(item[1][0]))


def analyze_stats_files(
    *,
    session_id: str,
    paths: list[Path],
    target_duty_percent: float = _DEFAULT_TARGET_DUTY_PERCENT,
    busy_threshold: float = _DEFAULT_BUSY_THRESHOLD,
) -> SessionDutyReport:
    """Analyze a single stats session from one or more rotated files."""
    samples: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    unknown_event_count = 0
    for payload in _read_stats_payloads(paths):
        event = payload.get("event")
        if event == "stats_sample" and isinstance(payload.get("sample"), dict):
            samples.append(cast(dict[str, Any], payload["sample"]))
        elif event == "job_completed" and isinstance(payload.get("job"), dict):
            job = cast(dict[str, Any], payload["job"])
            if isinstance(payload.get("baseline"), str):
                job["baseline"] = payload["baseline"]
            jobs.append(job)
        else:
            unknown_event_count += 1

    samples.sort(key=lambda sample: _float_or_none(sample.get("timestamp")) or 0.0)
    jobs.sort(key=lambda job: _job_finalized_timestamp(job) or 0.0)
    # Drop the cold-boot warm-up: samples before the first inference began are one-time model-load time,
    # not inter-job inefficiency, and counting them only depresses the duty headline with noise. Mirrors the
    # live worker's first-inference cutoff so an offline report and the live log agree on the same span.
    first_inference_ts = _first_inference_timestamp(jobs)
    if first_inference_ts is not None:
        samples = [
            sample for sample in samples if (_float_or_none(sample.get("timestamp")) or 0.0) >= first_inference_ts
        ]
    phase_totals = _phase_totals(jobs)
    phase_medians = _phase_medians(jobs)
    dominant_partial = _dominant_partial_kind(phase_totals)
    windows, buckets = _build_windows_and_buckets(
        samples,
        dominant_partial=dominant_partial,
        target_duty_percent=target_duty_percent,
        busy_threshold=busy_threshold,
    )
    duration_hours = _duration_hours(samples)
    completed_jobs = sum(1 for job in jobs if not bool(job.get("faulted")) and not bool(job.get("is_alchemy")))
    churn_delta = _counter_delta(samples, "churn_counts")
    slot_duty_delta = _counter_delta(samples, "slot_duty_totals")
    slot_duty_capacity = _last_int_or_none(samples, "slot_duty_capacity")

    report = SessionDutyReport(
        session_id=session_id,
        source_files=[str(path) for path in paths],
        start_timestamp=_float_or_none(samples[0].get("timestamp")) if samples else None,
        end_timestamp=_float_or_none(samples[-1].get("timestamp")) if samples else None,
        sample_count=len(samples),
        completed_jobs=completed_jobs,
        mean_gpu_duty_percent=_mean(_float_or_none(sample.get("gpu_duty_percent")) for sample in samples),
        mean_gpu_busy_fraction=_mean(_float_or_none(sample.get("gpu_busy_fraction")) for sample in samples),
        busy_fraction_percent=None,
        target_duty_percent=target_duty_percent,
        buckets=sorted(buckets.values(), key=lambda bucket: bucket.total_seconds, reverse=True),
        windows=windows,
        per_job_phase_medians=phase_medians,
        per_job_phase_totals=phase_totals,
        inference_queue_wait=_inference_queue_wait_summary(jobs),
        inference_dispatch_gap=_inference_dispatch_gap_summary(samples),
        model_breakdown=_count_by(jobs, "model_name"),
        baseline_breakdown=_count_by(jobs, "baseline"),
        churn_per_hour={kind: count / duration_hours for kind, count in churn_delta.items()} if duration_hours else {},
        churn_per_completed_job={kind: count / completed_jobs for kind, count in churn_delta.items()}
        if completed_jobs
        else {},
        operator_recommendations=[],
        maintainer_notes=[],
        unknown_event_count=unknown_event_count,
        slot_duty_seconds=slot_duty_delta,
        slot_duty_capacity=slot_duty_capacity,
        concurrency_occupancy=_concurrency_occupancy(samples),
    )
    report.busy_fraction_percent = (
        report.mean_gpu_busy_fraction * 100.0 if report.mean_gpu_busy_fraction is not None else None
    )
    report.operator_recommendations = _operator_recommendations(report)
    report.maintainer_notes = _maintainer_notes(report)
    return report


def render_session_duty_report(reports: list[SessionDutyReport]) -> str:
    """Render stats-backed duty reports for terminal output."""
    if not reports:
        return "No stats-backed duty-cycle sessions found. Falling back to log parsing may still work."

    out: list[str] = []
    for report in reports:
        start = _format_ts(report.start_timestamp)
        end = _format_ts(report.end_timestamp, time_only=True)
        out.append(f"== Stats session {report.session_id} | {start}-{end} | {report.sample_count} samples ==")
        duty = "?" if report.mean_gpu_duty_percent is None else f"{report.mean_gpu_duty_percent:.0f}%"
        busy = "?" if report.busy_fraction_percent is None else f"{report.busy_fraction_percent:.0f}%"
        out.append(f"   duty: mean {duty}  busy {busy}  completed jobs {report.completed_jobs}")
        if report.concurrency_occupancy:
            occupancy_total = sum(report.concurrency_occupancy.values())
            if occupancy_total > 0:
                shares = "  ".join(
                    f"{count}x {seconds / occupancy_total:.0%}"
                    for count, seconds in sorted(report.concurrency_occupancy.items(), key=lambda kv: kv[0])
                )
                capacity = f" (capacity {report.slot_duty_capacity})" if report.slot_duty_capacity else ""
                out.append(f"   concurrency occupancy{capacity}: {shares}")
        if report.slot_duty_seconds:
            slot_total = sum(report.slot_duty_seconds.values())
            if slot_total > 0:
                slots = "  ".join(
                    f"{bucket} {seconds / slot_total:.0%} ({seconds:.0f}s)"
                    for bucket, seconds in sorted(report.slot_duty_seconds.items(), key=lambda kv: -kv[1])[:6]
                )
                out.append(f"   slot duty: {slots}")
        top = [bucket for bucket in report.buckets if bucket.total_seconds > 0.0][:5]
        if top:
            rendered = "  ".join(
                f"{bucket.kind.value} {bucket.total_seconds:.0f}s"
                f" (idle {bucket.idle_seconds:.0f}s, partial {bucket.partial_utilization_seconds:.0f}s)"
                for bucket in top
            )
            out.append(f"   top loss buckets: {rendered}")
        else:
            out.append("   top loss buckets: none attributed")
        if report.per_job_phase_medians:
            phases = "  ".join(
                f"{phase} {seconds:.1f}s/job"
                for phase, seconds in sorted(
                    report.per_job_phase_medians.items(), key=lambda item: item[1], reverse=True
                )[:5]
            )
            out.append(f"   phase medians: {phases}")
        if report.inference_queue_wait is not None:
            wait = report.inference_queue_wait
            ratio = "?" if wait.wait_to_sampling_ratio is None else f"{wait.wait_to_sampling_ratio:.2f}x"
            out.append(
                "   inference queue wait: "
                f"total {wait.total_seconds:.0f}s ({wait.total_seconds / max(wait.jobs, 1):.1f}s/job; "
                f"median {wait.median_seconds:.1f}s, p90 {wait.p90_seconds:.1f}s, max {wait.max_seconds:.1f}s; "
                f"{ratio} sampling time; overlaps active inference)"
            )
            if wait.top_models:
                models = "  ".join(
                    f"{model.model_name} {model.total_seconds:.0f}s/{model.jobs}j" for model in wait.top_models[:3]
                )
                out.append(f"   queue wait by model: {models}")
        if report.inference_dispatch_gap is not None:
            gap = report.inference_dispatch_gap
            fraction = (
                "?" if gap.no_active_inference_fraction is None else f"{gap.no_active_inference_fraction * 100.0:.0f}%"
            )
            out.append(
                "   inference dispatch gap: "
                f"{gap.no_active_inference_seconds:.0f}s queued with no active inference "
                f"({fraction} of queued sample time)"
            )
            if gap.top_states:
                states = "  ".join(f"{state} {seconds:.0f}s" for state, seconds in list(gap.top_states.items())[:3])
                out.append(f"   dispatch gap states: {states}")
        if report.churn_per_completed_job:
            churn = "  ".join(f"{kind} {value:.2f}/job" for kind, value in report.churn_per_completed_job.items())
            out.append(f"   churn: {churn}")
        if report.operator_recommendations:
            out.append("   operator view: " + " ".join(report.operator_recommendations[:3]))
        if report.maintainer_notes:
            out.append("   maintainer view: " + " ".join(report.maintainer_notes[:3]))
        out.append("")
    return "\n".join(out).rstrip()


def _iter_stats_files(stats_dir: Path) -> Iterable[Path]:
    yield from stats_dir.glob("stats-v*.jsonl")
    yield from stats_dir.glob("stats-v*.jsonl.gz")


def _stats_file_order_key(path: Path) -> tuple[float, str]:
    match = _STATS_SESSION_RE.match(path.name)
    if match is not None:
        return (float(match.group("stamp").replace("-", "")), f"{int(match.group('index')):06d}")
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (mtime, path.name)


def _read_stats_payloads(paths: list[Path]) -> Iterable[dict[str, Any]]:
    for path in sorted(paths, key=_stats_file_order_key):
        compressed = path.name.endswith(".gz")
        try:
            with _open_text(path, compressed=compressed) as handle:
                for line in handle:
                    try:
                        payload = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(payload, dict):
                        yield cast(dict[str, Any], payload)
        except OSError:
            continue


def _open_text(path: Path, *, compressed: bool) -> IO[str]:
    if compressed:
        return cast(IO[str], gzip.open(path, "rt", encoding="utf-8"))
    return path.open("rt", encoding="utf-8")


def _build_windows_and_buckets(
    samples: list[dict[str, Any]],
    *,
    dominant_partial: DutyLossKind,
    target_duty_percent: float,
    busy_threshold: float,
) -> tuple[list[DutyWindowBreakdown], dict[DutyLossKind, DutyLossBucket]]:
    windows: list[DutyWindowBreakdown] = []
    buckets = {kind: DutyLossBucket(kind=kind) for kind in DutyLossKind}
    intervals = _sample_intervals(samples)
    previous_sample: dict[str, Any] | None = None
    for sample, duration in zip(samples, intervals, strict=False):
        timestamp = _float_or_none(sample.get("timestamp"))
        if timestamp is None or duration <= 0.0:
            continue
        duty = _float_or_none(sample.get("gpu_duty_percent"))
        busy_fraction = _float_or_none(sample.get("gpu_busy_fraction"))
        if busy_fraction is None:
            busy_fraction = 1.0 if (duty is not None and duty >= target_duty_percent) else 0.0
        busy_fraction = max(0.0, min(1.0, busy_fraction))
        idle_seconds = duration * max(0.0, 1.0 - busy_fraction)
        partial_seconds = 0.0
        if duty is not None and busy_fraction > 0.0 and duty < target_duty_percent:
            partial_seconds = duration * busy_fraction * ((target_duty_percent - duty) / target_duty_percent)
        idle_kind = _idle_kind(sample, previous_sample)
        partial_kind = _partial_kind(sample, fallback=dominant_partial)
        evidence = _sample_evidence(sample)
        windows.append(
            DutyWindowBreakdown(
                start_timestamp=timestamp,
                end_timestamp=timestamp + duration,
                duration_seconds=duration,
                gpu_duty_percent=duty,
                gpu_busy_fraction=busy_fraction,
                idle_kind=idle_kind,
                partial_kind=partial_kind,
                idle_seconds=idle_seconds,
                partial_utilization_seconds=partial_seconds,
                evidence=evidence,
            ),
        )
        if idle_seconds > 0.0:
            bucket = buckets[idle_kind]
            bucket.idle_seconds += idle_seconds
            if busy_fraction < busy_threshold:
                bucket.idle_samples += 1
            _add_evidence(bucket, evidence)
        if partial_seconds > 0.0:
            bucket = buckets[partial_kind]
            bucket.partial_utilization_seconds += partial_seconds
            bucket.partial_samples += 1
            _add_evidence(bucket, evidence)
        previous_sample = sample
    return windows, buckets


def _sample_intervals(samples: list[dict[str, Any]]) -> list[float]:
    if not samples:
        return []
    timestamps = [_float_or_none(sample.get("timestamp")) for sample in samples]
    deltas = [
        next_ts - ts
        for ts, next_ts in zip(timestamps, timestamps[1:], strict=False)
        if ts is not None and next_ts is not None and 0.0 < next_ts - ts <= _MAX_INFERRED_INTERVAL_SECONDS
    ]
    fallback = statistics.median(deltas) if deltas else 1.0
    intervals: list[float] = []
    for index, ts in enumerate(timestamps):
        if ts is None:
            intervals.append(0.0)
            continue
        if index + 1 < len(timestamps) and timestamps[index + 1] is not None:
            delta = cast(float, timestamps[index + 1]) - ts
            intervals.append(delta if 0.0 < delta <= _MAX_INFERRED_INTERVAL_SECONDS else fallback)
        else:
            intervals.append(fallback)
    return intervals


def _idle_kind(sample: dict[str, Any], previous_sample: dict[str, Any] | None) -> DutyLossKind:
    if (
        bool(sample.get("self_throttle_paused"))
        or bool(sample.get("last_pop_maintenance_mode"))
        or bool(sample.get("worker_details_maintenance"))
        or bool(sample.get("in_error_backoff"))
    ):
        return DutyLossKind.API_BACKOFF
    if bool(sample.get("supervisor_paused")) or bool(sample.get("maintenance_mode")):
        return DutyLossKind.LOCAL_PAUSE
    if _no_jobs_delta_hint(sample, previous_sample):
        return DutyLossKind.DEMAND_LIMITED
    if _counter_increased(sample, previous_sample, "num_process_recoveries"):
        return DutyLossKind.PROCESS_RECOVERY
    if _int_value(sample.get("jobs_pending_submit")) > 0:
        return DutyLossKind.SUBMIT
    if _int_value(sample.get("jobs_pending_safety_check")) + _int_value(sample.get("jobs_being_safety_checked")) > 0:
        return DutyLossKind.SAFETY
    if _int_value(sample.get("jobs_pending_inference")) > 0 or _int_value(sample.get("pending_megapixelsteps")) > 0:
        return DutyLossKind.SCHEDULER_WAIT
    if _int_value(sample.get("alchemy_forms_pending")) + _int_value(sample.get("alchemy_forms_in_flight")) > 0:
        return DutyLossKind.AUX_DOWNLOAD
    state = _state_text(sample)
    if "MODEL" in state or "PRELOAD" in state:
        return DutyLossKind.MODEL_LOAD
    return DutyLossKind.UNKNOWN


def _partial_kind(sample: dict[str, Any], *, fallback: DutyLossKind) -> DutyLossKind:
    state = _state_text(sample)
    if _int_value(sample.get("jobs_pending_submit")) > 0:
        return DutyLossKind.SUBMIT
    if _int_value(sample.get("jobs_pending_safety_check")) + _int_value(sample.get("jobs_being_safety_checked")) > 0:
        return DutyLossKind.SAFETY
    if "INFERENCE_POST_PROCESSING" in state or "POST_PROCESS" in state or "UPSCALE" in state or "FACE" in state:
        return DutyLossKind.POST_PROCESSING
    if "PRELOADING_MODEL" in state or "MODEL" in state or "PRELOAD" in state:
        return DutyLossKind.MODEL_LOAD
    # The one-time RAM->VRAM staging/prompt-encode window is INFERENCE_PRIMED (INFERENCE_STARTING now
    # means the denoise loop is actually running, which is not a duty loss). Attribute the transfer gap
    # to the primed window.
    if "INFERENCE_PRIMED" in state or "VRAM" in state:
        return DutyLossKind.VRAM_TRANSFER
    if "UNLOAD" in state or "EVICT" in state:
        return DutyLossKind.MODEL_UNLOAD
    if _int_value(sample.get("jobs_in_progress")) > 0:
        return fallback
    return DutyLossKind.UNKNOWN


def _state_text(sample: dict[str, Any]) -> str:
    parts = [
        str(sample.get("process_state_summary") or ""),
        str(sample.get("orchestration_intent_summary") or ""),
        str(sample.get("orchestration_next_action") or ""),
        str(sample.get("orchestration_why") or ""),
        str(sample.get("orchestration_raw_gate") or ""),
    ]
    return " ".join(parts).upper()


def _no_jobs_delta_hint(sample: dict[str, Any], previous_sample: dict[str, Any] | None) -> bool:
    if bool(sample.get("last_pop_no_jobs_available")):
        return True
    skipped = sample.get("last_pop_skipped_reasons")
    if isinstance(skipped, dict) and skipped:
        return True
    current = _float_or_none(sample.get("time_spent_no_jobs_available"))
    previous = _float_or_none(previous_sample.get("time_spent_no_jobs_available")) if previous_sample else None
    if current is None:
        return False
    return current > (previous or 0.0)


def _counter_increased(sample: dict[str, Any], previous_sample: dict[str, Any] | None, key: str) -> bool:
    current = _float_or_none(sample.get(key))
    previous = _float_or_none(previous_sample.get(key)) if previous_sample else None
    return current is not None and current > (previous or 0.0)


def _inference_dispatch_gap_summary(samples: list[dict[str, Any]]) -> InferenceDispatchGapSummary | None:
    queued_seconds = 0.0
    no_active_inference_seconds = 0.0
    state_seconds: dict[str, float] = {}
    for sample, duration in zip(samples, _sample_intervals(samples), strict=False):
        if duration <= 0.0 or _int_value(sample.get("jobs_pending_inference")) <= 0:
            continue
        queued_seconds += duration
        if _int_value(sample.get("jobs_in_progress")) > 0:
            continue
        no_active_inference_seconds += duration
        state = _dispatch_gap_state(sample)
        state_seconds[state] = state_seconds.get(state, 0.0) + duration
    if queued_seconds <= 0.0:
        return None
    top_states = dict(sorted(state_seconds.items(), key=lambda item: item[1], reverse=True)[:5])
    return InferenceDispatchGapSummary(
        queued_seconds=queued_seconds,
        no_active_inference_seconds=no_active_inference_seconds,
        no_active_inference_fraction=no_active_inference_seconds / queued_seconds,
        top_states=top_states,
    )


def _dispatch_gap_state(sample: dict[str, Any]) -> str:
    state = str(sample.get("process_state_summary") or "").strip()
    if state:
        return state[:80]
    intent = str(sample.get("orchestration_intent_summary") or "").strip()
    if intent:
        return intent[:80]
    action = str(sample.get("orchestration_next_action") or "").strip()
    if action:
        return action[:80]
    return "unknown"


def _inference_queue_wait_summary(jobs: list[dict[str, Any]]) -> InferenceQueueWaitSummary | None:
    wait_samples: list[float] = []
    sampling_seconds = 0.0
    by_model: dict[str, list[dict[str, float]]] = {}
    for job in jobs:
        if bool(job.get("faulted")) or bool(job.get("is_alchemy")):
            continue
        queue_wait = _float_or_none(job.get("queue_wait_seconds"))
        if queue_wait is None:
            continue
        queue_wait = max(0.0, queue_wait)
        sampling = _job_sampling_seconds(job)
        wait_samples.append(queue_wait)
        sampling_seconds += sampling
        model_name = str(job.get("model_name") or "unknown")
        by_model.setdefault(model_name, []).append({"wait": queue_wait, "sampling": sampling})
    if not wait_samples:
        return None

    top_models: list[InferenceQueueWaitModelBreakdown] = []
    for model_name, entries in by_model.items():
        waits = [entry["wait"] for entry in entries]
        model_sampling = sum(entry["sampling"] for entry in entries)
        total = sum(waits)
        top_models.append(
            InferenceQueueWaitModelBreakdown(
                model_name=model_name,
                jobs=len(entries),
                total_seconds=total,
                median_seconds=_percentile(waits, 0.5),
                p90_seconds=_percentile(waits, 0.9),
                max_seconds=max(waits),
                sampling_seconds=model_sampling,
                wait_to_sampling_ratio=(total / model_sampling) if model_sampling > 0 else None,
            ),
        )
    top_models.sort(key=lambda item: item.total_seconds, reverse=True)
    total_wait = sum(wait_samples)
    return InferenceQueueWaitSummary(
        jobs=len(wait_samples),
        total_seconds=total_wait,
        median_seconds=_percentile(wait_samples, 0.5),
        p90_seconds=_percentile(wait_samples, 0.9),
        max_seconds=max(wait_samples),
        sampling_seconds=sampling_seconds,
        wait_to_sampling_ratio=(total_wait / sampling_seconds) if sampling_seconds > 0 else None,
        top_models=top_models[:5],
    )


def _job_sampling_seconds(job: dict[str, Any]) -> float:
    metrics = job.get("phase_metrics")
    if not isinstance(metrics, dict):
        return 0.0
    sampling = metrics.get("sampling")
    if not isinstance(sampling, dict):
        return 0.0
    return max(0.0, _float_or_none(sampling.get("duration_seconds")) or 0.0)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


def _phase_totals(jobs: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for job in jobs:
        for phase, seconds in _job_phase_seconds(job).items():
            totals[phase] = totals.get(phase, 0.0) + seconds
    return dict(sorted(totals.items(), key=lambda item: item[1], reverse=True))


def _phase_medians(jobs: list[dict[str, Any]]) -> dict[str, float]:
    samples: dict[str, list[float]] = {}
    for job in jobs:
        for phase, seconds in _job_phase_seconds(job).items():
            samples.setdefault(phase, []).append(seconds)
    return {
        phase: statistics.median(values)
        for phase, values in sorted(samples.items(), key=lambda item: statistics.median(item[1]), reverse=True)
    }


def _job_phase_seconds(job: dict[str, Any]) -> dict[str, float]:
    phases: dict[str, float] = {}
    safety = _float_or_none(job.get("safety_seconds"))
    if safety is not None:
        phases["safety"] = max(0.0, safety)
    stamps = job.get("stage_timestamps")
    if isinstance(stamps, dict):
        submit_ready = _float_or_none(stamps.get("PENDING_SUBMIT"))
        finalized = _float_or_none(stamps.get("FINALIZED"))
        if submit_ready is not None and finalized is not None:
            phases["submit"] = max(0.0, finalized - submit_ready)
    metrics = job.get("phase_metrics")
    if not isinstance(metrics, dict):
        return phases
    model_loads = metrics.get("model_loads")
    if isinstance(model_loads, list):
        for load in model_loads:
            if not isinstance(load, dict):
                continue
            seconds = _float_or_none(load.get("duration_seconds")) or 0.0
            phase = load.get("phase")
            if phase == "disk_to_ram":
                phases["model_load"] = phases.get("model_load", 0.0) + seconds
            elif phase == "ram_to_vram":
                phases["vram_transfer"] = phases.get("vram_transfer", 0.0) + seconds
    sampling = metrics.get("sampling")
    if isinstance(sampling, dict):
        seconds = _float_or_none(sampling.get("duration_seconds"))
        if seconds is not None:
            phases["sampling"] = max(0.0, seconds)
    phase_seconds = metrics.get("phase_seconds")
    if isinstance(phase_seconds, dict):
        _add_phase_seconds(phases, phase_seconds, "model_unload", DutyLossKind.MODEL_UNLOAD.value)
        _add_phase_seconds(phases, phase_seconds, "vae_decode", "vae_decode")
        for key in ("clip_encode", "vae_encode"):
            _add_phase_seconds(phases, phase_seconds, key, "encode")
        for key in ("pipeline_setup", "pipeline_validate", "pipeline_finalize"):
            _add_phase_seconds(phases, phase_seconds, key, "graph_overhead")
    return phases


def _add_phase_seconds(phases: dict[str, float], phase_seconds: dict[str, Any], source: str, target: str) -> None:
    seconds = _float_or_none(phase_seconds.get(source))
    if seconds is not None:
        phases[target] = phases.get(target, 0.0) + max(0.0, seconds)


def _dominant_partial_kind(phase_totals: dict[str, float]) -> DutyLossKind:
    mapping = {
        "model_load": DutyLossKind.MODEL_LOAD,
        "model_unload": DutyLossKind.MODEL_UNLOAD,
        "vram_transfer": DutyLossKind.VRAM_TRANSFER,
        "safety": DutyLossKind.SAFETY,
        "submit": DutyLossKind.SUBMIT,
        "graph_overhead": DutyLossKind.POST_PROCESSING,
        "encode": DutyLossKind.POST_PROCESSING,
        "vae_decode": DutyLossKind.POST_PROCESSING,
    }
    for phase in phase_totals:
        kind = mapping.get(phase)
        if kind is not None:
            return kind
    return DutyLossKind.UNKNOWN


def _last_int_or_none(samples: list[dict[str, Any]], key: str) -> int | None:
    """The last sample's integer value for ``key``, or None when absent/invalid (pre-field stats files)."""
    for sample in reversed(samples):
        value = sample.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _concurrency_occupancy(samples: list[dict[str, Any]]) -> dict[str, float]:
    """Seconds spent at each concurrent in-flight job count, from adjacent stats samples.

    Each inter-sample interval is attributed to the earlier sample's ``jobs_in_progress`` reading, the
    same convention the loss windows use. This is the summary the slot-duty buckets explain: occupancy
    says how much of the configured capacity ran; the buckets say why the rest did not.
    """
    intervals = _sample_intervals(samples)
    occupancy: dict[str, float] = {}
    for sample, interval in zip(samples, intervals, strict=False):
        in_progress = sample.get("jobs_in_progress")
        if not isinstance(in_progress, int) or isinstance(in_progress, bool) or in_progress < 0:
            continue
        key = str(in_progress)
        occupancy[key] = occupancy.get(key, 0.0) + interval
    return occupancy


def _counter_delta(samples: list[dict[str, Any]], key: str) -> dict[str, float]:
    first = samples[0].get(key) if samples else None
    last = samples[-1].get(key) if samples else None
    if not isinstance(first, dict) or not isinstance(last, dict):
        return {}
    out: dict[str, float] = {}
    for name, value in last.items():
        if isinstance(value, int | float):
            start = first.get(name, 0)
            start_value = float(start) if isinstance(start, int | float) else 0.0
            out[str(name)] = max(0.0, float(value) - start_value)
    return out


def _duration_hours(samples: list[dict[str, Any]]) -> float | None:
    if len(samples) < 2:
        return None
    start = _float_or_none(samples[0].get("timestamp"))
    end = _float_or_none(samples[-1].get("timestamp"))
    if start is None or end is None or end <= start:
        return None
    return (end - start) / 3600.0


def _count_by(jobs: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in jobs:
        value = job.get(key) or "unknown"
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def _operator_recommendations(report: SessionDutyReport) -> list[str]:
    recs: list[str] = []
    bucket_map = {bucket.kind: bucket for bucket in report.buckets}
    demand = bucket_map.get(DutyLossKind.DEMAND_LIMITED)
    scheduler = bucket_map.get(DutyLossKind.SCHEDULER_WAIT)
    model_load = bucket_map.get(DutyLossKind.MODEL_LOAD)
    local_pause = bucket_map.get(DutyLossKind.LOCAL_PAUSE)
    if demand is not None and demand.total_seconds > 0:
        recs.append(
            "The session spent measurable time demand-limited; adding more local tuning cannot create Horde jobs."
        )
    if scheduler is not None and scheduler.total_seconds > 0:
        recs.append(
            "Pending work existed while the GPU was not busy; check queue size, concurrency, and residency settings."
        )
    if model_load is not None and model_load.total_seconds > 0:
        recs.append(
            "Model load/transfer dominated loss; reduce model churn or use a smaller served model set "
            "for this VRAM size."
        )
    if local_pause is not None and local_pause.total_seconds > 0:
        recs.append("Local pause or maintenance time is counted separately from worker inefficiency.")
    if (
        not recs
        and report.mean_gpu_duty_percent is not None
        and report.mean_gpu_duty_percent >= report.target_duty_percent
    ):
        recs.append("Duty met the configured target; no duty-cycle tuning action is indicated from these stats.")
    return recs


def _maintainer_notes(report: SessionDutyReport) -> list[str]:
    notes: list[str] = []
    unknown = next((bucket for bucket in report.buckets if bucket.kind == DutyLossKind.UNKNOWN), None)
    if unknown is not None and unknown.total_seconds > 0:
        notes.append(
            "Unknown/unattributed time remains; inspect sample state/intent coverage around the evidence timestamps."
        )
    recovery = next((bucket for bucket in report.buckets if bucket.kind == DutyLossKind.PROCESS_RECOVERY), None)
    if recovery is not None and recovery.total_seconds > 0:
        notes.append("Process recovery coincided with duty loss; correlate with action ledger and crash diagnostics.")
    if report.unknown_event_count:
        notes.append(f"Ignored {report.unknown_event_count} unknown stats event(s); preserve them when downsampling.")
    return notes


def _sample_evidence(sample: dict[str, Any]) -> str:
    timestamp = _float_or_none(sample.get("timestamp"))
    stamp = _format_ts(timestamp) if timestamp is not None else "?"
    state = str(sample.get("process_state_summary") or "").strip()
    gate = str(sample.get("orchestration_raw_gate") or sample.get("orchestration_why") or "").strip()
    if state and gate:
        return f"{stamp}: {state}; {gate}"
    if state:
        return f"{stamp}: {state}"
    if gate:
        return f"{stamp}: {gate}"
    return stamp


def _add_evidence(bucket: DutyLossBucket, evidence: str) -> None:
    if evidence and evidence not in bucket.evidence and len(bucket.evidence) < 5:
        bucket.evidence.append(evidence)


def _job_finalized_timestamp(job: dict[str, Any]) -> float | None:
    stamps = job.get("stage_timestamps")
    if isinstance(stamps, dict):
        return _float_or_none(stamps.get("FINALIZED"))
    return None


def _first_inference_timestamp(jobs: list[dict[str, Any]]) -> float | None:
    """Earliest moment any job entered ``INFERENCE_IN_PROGRESS``, or None when no job reached inference.

    The cutoff before which warm-up samples are excluded from the duty analysis, matching the live worker's
    first-inference boundary so cold-boot model loading is never charged against the duty cycle.
    """
    starts = [
        started
        for job in jobs
        if isinstance(job.get("stage_timestamps"), dict)
        and (started := _float_or_none(job["stage_timestamps"].get("INFERENCE_IN_PROGRESS"))) is not None
    ]
    return min(starts) if starts else None


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return statistics.mean(clean) if clean else None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _int_value(value: object) -> int:
    return int(value) if isinstance(value, int) else 0


def _format_ts(timestamp: float | None, *, time_only: bool = False) -> str:
    if timestamp is None:
        return "?"
    fmt = "%H:%M" if time_only else "%Y-%m-%d %H:%M"
    return datetime.fromtimestamp(timestamp).strftime(fmt)


__all__ = [
    "DutyLossBucket",
    "DutyLossKind",
    "DutyWindowBreakdown",
    "SessionDutyReport",
    "analyze_stats_files",
    "analyze_stats_sessions",
    "discover_stats_sessions",
    "render_session_duty_report",
]
