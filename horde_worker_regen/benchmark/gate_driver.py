"""Runs the disagg gate mixes through the harness and scores an A/B comparison.

This is the measurement gate for a disaggregation-optimization campaign: it drives a named payload mix
(:mod:`horde_worker_regen.benchmark.disagg_mixes`) through the in-repo harness under two bridge-data
configurations, scores each run's kudos/hour with the server-parity scorer, and reports the mechanism
metrics (per-stage disk->RAM reload counts, stage latency percentiles, the device paging-cliff count)
that explain a throughput delta.

Order effects and slow drift are cancelled by running the two variants in ABBA order per rung on an
identical seed, so both variants meet byte-identical work and the difference reflects the configuration,
not the arrival of a heavier job in one arm.

**Torch-free by construction.** The worker parent must never load torch, so this module imports the
harness, the run-metrics types, and the kudos scorer lazily inside the functions that run, never at
module import time; the engine version is read from distribution metadata (``horde-engine``) without
importing hordelib.

**Paging-edge signal.** The device paging cliff is surfaced from the run-metrics
``governor_saturation_events`` field: the device-free governor's transitions into its SATURATED band,
which is where the reclaim ladder runs because the WDDM driver has begun demoting VRAM to system memory.
It is a real, device-truth counter and is zero on a fake-mode run by construction (no device governor),
so a nonzero value only appears in a real-mode rung.

**Kudos checkpoint.** Scoring needs the server's kudos checkpoint, supplied out of band exactly as
:mod:`horde_worker_regen.analysis.kudos_score` expects it: the checkpoint path comes from the CLI
``--kudos-ckpt`` argument or the ``AI_HORDE_KUDOS_MODEL_CKPT`` environment variable. When scoring is
required but no checkpoint resolves, the run fails with a clear message rather than reporting a silent
``None``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from loguru import logger
from pydantic import BaseModel, Field

from horde_worker_regen.benchmark.disagg_mixes import (
    DEFAULT_GATE_LORA_POOL,
    DisaggGateMix,
    build_disagg_gate_scenario,
)

if TYPE_CHECKING:
    from horde_worker_regen.process_management.resources.run_metrics import JobMetricsRecord, RunMetricsSnapshot

GateProcessMode = Literal["fake", "real"]

ScalarValue = str | int | float | bool | None
FlatScalarMap = dict[str, ScalarValue]
"""A flat, JSON-scalar-only mapping used for bridge-data overrides and env overrides in a report."""

DEFAULT_RUNGS_SECONDS: tuple[float, ...] = (90.0, 300.0, 600.0, 1200.0)
"""The default rung ladder: a short warm-up rung through progressively longer sustained rungs."""

_KUDOS_CKPT_ENV = "AI_HORDE_KUDOS_MODEL_CKPT"
"""The environment variable the kudos checkpoint path is read from when no explicit path is given."""

_LOCKFILE_NAME = "disagg_gate.lock"
_REPORT_NAME = "disagg_gate_report.json"
_WHOLE_JOB_STAGE_KEY = "whole_job"
"""The mechanism-metric key for untagged monolithic job records (no single pipeline stage owns them)."""

_ENGINE_DISTRIBUTION = "horde-engine"


class GateError(RuntimeError):
    """Base class for gate-driver precondition failures."""


class GatePreconditionError(GateError):
    """A real-mode rung was refused because the box is not in a safe state to measure on."""


class GateScoringUnavailableError(GateError):
    """Kudos scoring was required but no checkpoint could be resolved."""


class GateVariant(BaseModel):
    """One arm of an A/B comparison: a label plus the overrides that define its configuration."""

    label: str
    bridge_overrides: FlatScalarMap = Field(default_factory=dict)
    """Bridge-data fields applied on top of the harness defaults for this arm."""
    env_overrides: dict[str, str] = Field(default_factory=dict)
    """Process environment variables set for the duration of this arm's run and restored afterwards."""


class GateRunResult(BaseModel):
    """The scored outcome and mechanism metrics of one gate rung run of one variant."""

    mix: DisaggGateMix
    rung_seconds: float
    seed: int
    variant_label: str
    order_index: int
    """Position of this run within its rung's ABBA sequence (0..3), so order effects stay visible."""
    mode: GateProcessMode
    bridge_overrides: FlatScalarMap = Field(default_factory=dict)
    env_overrides: dict[str, str] = Field(default_factory=dict)

    kudos_per_hour: float | None = None
    """Server-parity kudos/hour, or None when scoring was unavailable and not required."""
    kudos_required: bool = False

    jobs_completed: int = 0
    faults: int = 0
    timed_out: bool = False
    exit_reason: str = ""

    disk_to_ram_events_by_stage: dict[str, int] = Field(default_factory=dict)
    """Count of disk->RAM model loads keyed by pipeline stage tag, plus ``whole_job`` for monolithic records."""
    disk_to_ram_seconds_by_stage: dict[str, float] = Field(default_factory=dict)
    """Seconds spent in those disk->RAM loads, keyed the same way."""
    stage_latency_p50_seconds: dict[str, float] = Field(default_factory=dict)
    """Median measured compute wall-time per stage (sampling plus named GPU phases), keyed as above.

    Derived only from durations the records already carry; a stage with no timed record is omitted rather
    than reported as zero."""
    stage_latency_p95_seconds: dict[str, float] = Field(default_factory=dict)
    """The p95 counterpart to :attr:`stage_latency_p50_seconds`."""

    paging_cliff_events: int | None = None
    """Device-free governor SATURATED-band crossings (the WDDM paging cliff); None only if unmeasured."""

    hordelib_version: str | None = None
    job_records_path: str | None = None
    """Path to the JSONL dump of this run's job records, rescorable by the kudos_score CLI."""
    report_path: str | None = None
    """Path to the ladder's aggregate JSON report (set when this result is part of a ladder)."""


@dataclass
class GateRunConfig:
    """Inputs for a single gate rung run of a single variant."""

    mix: DisaggGateMix
    rung_seconds: float
    seed: int
    variant: GateVariant
    output_dir: Path
    mode: GateProcessMode = "fake"
    model_pool: Sequence[str] | None = None
    img2img_fraction: float = 0.2
    lora_fraction: float = 0.35
    jobs_per_minute_estimate: float = 12.0
    lora_pool: Sequence[str] | None = None
    lora_pool_is_version: bool = False
    kudos_ckpt_path: Path | None = None
    require_kudos: bool = False
    timeout_seconds: float | None = None


def _resolve_kudos_ckpt(explicit: Path | None) -> Path | None:
    """Resolve the kudos checkpoint from the explicit path or the environment, or None if neither is a file."""
    candidate = explicit
    if candidate is None:
        raw = os.environ.get(_KUDOS_CKPT_ENV)
        candidate = Path(raw) if raw else None
    if candidate is not None and candidate.is_file():
        return candidate
    return None


def _pytest_is_running() -> bool:
    """Whether any pytest process is currently running on the box (best-effort, via psutil if present)."""
    try:
        import psutil
    except ImportError:
        return False

    pytest_token = re.compile(r"(?:^|[\\/])(?:pytest|py\.test)(?:$|[\s.])")
    own_pid = os.getpid()
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        with contextlib.suppress(Exception):
            if process.info["pid"] == own_pid:
                continue
            name = process.info.get("name") or ""
            cmdline = " ".join(process.info.get("cmdline") or [])
            if pytest_token.search(name) or pytest_token.search(cmdline):
                return True
    return False


@contextlib.contextmanager
def _temporary_env(overrides: Mapping[str, str]) -> Iterator[None]:
    """Apply env overrides for the duration of the block, restoring the prior values (or absence) after."""
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for key, prior in previous.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


def _default_timeout_seconds(config: GateRunConfig) -> float:
    """A generous hard-timeout for a rung, covering drain (soak) or a cold real-mode job list."""
    if config.mix is DisaggGateMix.CHURN_SEEDED_RANDOM:
        # The harness soak treats this as the hard cap during the load phase; leave room for the drain
        # and a real-mode cold start on top of the sustained window.
        return config.rung_seconds + 60.0 + 300.0
    return max(600.0, config.rung_seconds * 3.0 + 120.0)


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Linear-interpolated percentile of a non-empty sample (nearest-rank when it lands on an index)."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 4)
    rank = (percentile / 100.0) * (len(ordered) - 1)
    low_index = math.floor(rank)
    high_index = math.ceil(rank)
    if low_index == high_index:
        return round(ordered[low_index], 4)
    fraction = rank - low_index
    return round(ordered[low_index] * (1.0 - fraction) + ordered[high_index] * fraction, 4)


def _stage_record_compute_seconds(record: JobMetricsRecord) -> float | None:
    """The compute wall-time a stage record measured (sampling plus named GPU phases), or None if untimed.

    Model-load time is excluded here because it is reported separately as the disk->RAM breakdown; this is
    the stage's own work, summed from the durations the record already carries. None when the record holds
    no timed phase at all, so an untimed record does not masquerade as a zero-latency stage.
    """
    metrics = record.phase_metrics
    if metrics is None:
        return None
    total = 0.0
    saw_timing = False
    if metrics.sampling is not None:
        total += metrics.sampling.duration_seconds
        saw_timing = True
    for phase_seconds in metrics.phase_seconds.values():
        total += phase_seconds
        saw_timing = True
    return total if saw_timing else None


def _disk_to_ram_by_stage(snapshot: RunMetricsSnapshot) -> tuple[dict[str, int], dict[str, float]]:
    """Count and time the disk->RAM model loads, grouped by the stage that incurred them.

    Stage-tagged records attribute their loads to their pipeline stage; untagged whole-job (monolithic)
    records attribute theirs to ``whole_job``. Every stage key is pre-seeded to zero so an A/B pair always
    presents the same key set (a stage that ran with no reload and a stage that did not run both read as
    zero reloads, which is the true statement the count makes).
    """
    from horde_worker_regen.process_management.ipc.messages import PipelineStageTag

    counts: dict[str, int] = {stage.value: 0 for stage in PipelineStageTag}
    counts[_WHOLE_JOB_STAGE_KEY] = 0
    seconds: dict[str, float] = dict.fromkeys(counts, 0.0)

    def _accumulate(record: JobMetricsRecord, key: str) -> None:
        metrics = record.phase_metrics
        if metrics is None:
            return
        for event in metrics.model_loads:
            if event.phase != "disk_to_ram":
                continue
            counts[key] += 1
            seconds[key] = round(seconds[key] + event.duration_seconds, 4)

    for record in snapshot.stage_metrics:
        key = record.stage.value if record.stage is not None else _WHOLE_JOB_STAGE_KEY
        _accumulate(record, key)
    for record in snapshot.jobs:
        _accumulate(record, _WHOLE_JOB_STAGE_KEY)
    return counts, seconds


def _stage_latency_percentiles(snapshot: RunMetricsSnapshot) -> tuple[dict[str, float], dict[str, float]]:
    """Derive p50/p95 stage latency from the durations the records already carry (omitting untimed stages)."""
    samples: dict[str, list[float]] = {}

    def _add(key: str, value: float | None) -> None:
        if value is not None:
            samples.setdefault(key, []).append(value)

    for record in snapshot.stage_metrics:
        key = record.stage.value if record.stage is not None else _WHOLE_JOB_STAGE_KEY
        _add(key, _stage_record_compute_seconds(record))
    for record in snapshot.jobs:
        if record.is_alchemy:
            continue
        _add(_WHOLE_JOB_STAGE_KEY, record.e2e_seconds)

    p50 = {key: _percentile(values, 50.0) for key, values in samples.items() if values}
    p95 = {key: _percentile(values, 95.0) for key, values in samples.items() if values}
    return p50, p95


def _score_kudos(
    snapshot: RunMetricsSnapshot,
    *,
    ckpt_path: Path | None,
    require_kudos: bool,
) -> float | None:
    """Score the run's image jobs, or return None when scoring is unavailable and not required."""
    if ckpt_path is None:
        if require_kudos:
            raise GateScoringUnavailableError(
                "kudos scoring was required but no checkpoint resolved; pass --kudos-ckpt or set "
                f"{_KUDOS_CKPT_ENV} to the server kudos checkpoint",
            )
        return None
    from horde_worker_regen.analysis.kudos_score import KudosModelScorer, score_session

    scorer = KudosModelScorer(ckpt_path)
    report = score_session(list(snapshot.jobs), scorer)
    return report.kudos_per_hour


def _hordelib_version() -> str | None:
    """Read the pinned engine version from distribution metadata without importing hordelib."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(_ENGINE_DISTRIBUTION)
    except PackageNotFoundError:
        return None


def _safe_filename_fragment(text: str) -> str:
    """A filesystem-safe fragment of a label for artifact filenames."""
    return re.sub(r"[^0-9A-Za-z._-]+", "-", text).strip("-") or "run"


def _write_job_records(snapshot: RunMetricsSnapshot, config: GateRunConfig, order_index: int) -> str:
    """Dump this run's job records (whole-job then stage records) as JSONL for later rescoring/analysis.

    Stage records ride in the same file so the artifact is self-sufficient for mechanism analysis: under
    disaggregation the whole-job records carry no model-load events (each stage reports its own), so a
    jobs-only dump would silently look load-free.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    label = _safe_filename_fragment(config.variant.label)
    filename = f"{config.mix.value}_{int(config.rung_seconds)}s_{label}_{order_index}.jobs.jsonl"
    path = config.output_dir / filename
    with path.open("w", encoding="utf-8") as records_file:
        for record in snapshot.jobs:
            records_file.write(record.model_dump_json() + "\n")
        for stage_record in snapshot.stage_metrics:
            records_file.write(stage_record.model_dump_json() + "\n")
    return str(path)


def run_gate_rung(config: GateRunConfig, *, order_index: int = 0) -> GateRunResult:
    """Build the mix, run it through the harness, score it, and derive the mechanism metrics.

    Args:
        config: The rung/variant inputs.
        order_index: This run's position within its rung's ABBA sequence, recorded on the result.

    Returns:
        The scored, mechanism-annotated result.

    Raises:
        GatePreconditionError: A real-mode rung was requested while a pytest process is running.
        GateScoringUnavailableError: Scoring was required but no checkpoint resolved.
    """
    if config.mode == "real" and _pytest_is_running():
        raise GatePreconditionError(
            "refusing a real-mode gate rung while a pytest process is running on this box",
        )

    from horde_worker_regen.harness import HarnessConfig, run_harness

    lora_pool = config.lora_pool if config.lora_pool is not None else DEFAULT_GATE_LORA_POOL
    scenario = build_disagg_gate_scenario(
        config.mix,
        rung_seconds=config.rung_seconds,
        seed=config.seed,
        model_pool=config.model_pool,
        img2img_fraction=config.img2img_fraction,
        lora_fraction=config.lora_fraction,
        jobs_per_minute_estimate=config.jobs_per_minute_estimate,
        lora_pool=lora_pool,
        lora_pool_is_version=config.lora_pool_is_version,
    )
    timeout_seconds = (
        config.timeout_seconds if config.timeout_seconds is not None else _default_timeout_seconds(config)
    )
    harness_config = HarnessConfig.from_scenario(
        scenario,
        process_mode=config.mode,
        timeout_seconds=timeout_seconds,
        bridge_data_overrides=dict(config.variant.bridge_overrides),
    )

    ckpt_path = _resolve_kudos_ckpt(config.kudos_ckpt_path)
    with _temporary_env(config.variant.env_overrides):
        harness_result = run_harness(harness_config)

    if getattr(harness_result, "boot_failed_no_progress", False):
        # A run that booted no working children has nothing to score; merging a 0.0 row into the report
        # would silently poison an A/B comparison.
        raise GateError(
            f"harness boot failure: no jobs progressed (exit_reason={harness_result.exit_reason!r}); "
            "the run is not scoreable",
        )

    snapshot = harness_result.metrics
    if snapshot is None:
        # The harness always attaches a snapshot; guard so scoring/derivation never dereferences None.
        raise GateError("harness returned no run-metrics snapshot")

    disk_to_ram_counts, disk_to_ram_seconds = _disk_to_ram_by_stage(snapshot)
    p50, p95 = _stage_latency_percentiles(snapshot)
    kudos_per_hour = _score_kudos(snapshot, ckpt_path=ckpt_path, require_kudos=config.require_kudos)
    job_records_path = _write_job_records(snapshot, config, order_index)

    return GateRunResult(
        mix=config.mix,
        rung_seconds=config.rung_seconds,
        seed=config.seed,
        variant_label=config.variant.label,
        order_index=order_index,
        mode=config.mode,
        bridge_overrides=dict(config.variant.bridge_overrides),
        env_overrides=dict(config.variant.env_overrides),
        kudos_per_hour=kudos_per_hour,
        kudos_required=config.require_kudos,
        jobs_completed=harness_result.num_jobs_completed,
        faults=harness_result.num_jobs_faulted,
        timed_out=harness_result.timed_out,
        exit_reason=harness_result.exit_reason,
        disk_to_ram_events_by_stage=disk_to_ram_counts,
        disk_to_ram_seconds_by_stage=disk_to_ram_seconds,
        stage_latency_p50_seconds=p50,
        stage_latency_p95_seconds=p95,
        paging_cliff_events=snapshot.governor_saturation_events,
        hordelib_version=_hordelib_version(),
        job_records_path=job_records_path,
    )


@contextlib.contextmanager
def _gate_lock(output_dir: Path) -> Iterator[Path]:
    """Hold an exclusive gate lockfile in the output dir, refusing to start if one already exists."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lockfile = output_dir / _LOCKFILE_NAME
    if lockfile.exists():
        raise GatePreconditionError(
            f"a gate run is already in progress (lockfile {lockfile} exists); remove it if it is stale",
        )
    lockfile.write_text(str(os.getpid()), encoding="utf-8")
    try:
        yield lockfile
    finally:
        lockfile.unlink(missing_ok=True)


def _abba_variant_order(variant_a: GateVariant, variant_b: GateVariant) -> tuple[GateVariant, ...]:
    """The ABBA counterbalance order for one rung, so run-order drift cancels between the two arms."""
    return (variant_a, variant_b, variant_b, variant_a)


def run_gate_ladder(
    *,
    mix: DisaggGateMix,
    output_dir: Path,
    variant_a: GateVariant,
    variant_b: GateVariant,
    seed: int,
    rungs: Sequence[float] = DEFAULT_RUNGS_SECONDS,
    mode: GateProcessMode = "fake",
    model_pool: Sequence[str] | None = None,
    img2img_fraction: float = 0.2,
    lora_fraction: float = 0.35,
    jobs_per_minute_estimate: float = 12.0,
    lora_pool: Sequence[str] | None = None,
    lora_pool_is_version: bool = False,
    kudos_ckpt_path: Path | None = None,
    require_kudos: bool = False,
) -> list[GateRunResult]:
    """Run the full A/B ladder (ABBA per rung on identical seeds) and write the JSON report.

    Every run in a rung meets identical work (same seed, same rung), and the two arms alternate in ABBA
    order so a linear drift over the rung cancels between them. The aggregate report is written to
    ``output_dir`` and a compact table is logged.

    Returns:
        Every run's result, in execution order, each carrying the report path.
    """
    with _gate_lock(output_dir):
        report_path = output_dir / _REPORT_NAME
        results: list[GateRunResult] = []

        def _persist_results() -> None:
            # Rewritten after every run so a crash mid-ladder never loses the completed runs' derived
            # metrics (the per-run jobs JSONL only carries raw records, not the scored/derived result).
            for result in results:
                result.report_path = str(report_path)
            report_path.write_text(
                json.dumps([result.model_dump(mode="json") for result in results], indent=2),
                encoding="utf-8",
            )

        for rung_seconds in rungs:
            for order_index, variant in enumerate(_abba_variant_order(variant_a, variant_b)):
                run_config = GateRunConfig(
                    mix=mix,
                    rung_seconds=rung_seconds,
                    seed=seed,
                    variant=variant,
                    output_dir=output_dir,
                    mode=mode,
                    model_pool=model_pool,
                    img2img_fraction=img2img_fraction,
                    lora_fraction=lora_fraction,
                    jobs_per_minute_estimate=jobs_per_minute_estimate,
                    lora_pool=lora_pool,
                    lora_pool_is_version=lora_pool_is_version,
                    kudos_ckpt_path=kudos_ckpt_path,
                    require_kudos=require_kudos,
                )
                results.append(run_gate_rung(run_config, order_index=order_index))
                _persist_results()

        for line in _format_results_table(results):
            logger.info(line)
        logger.info(f"Disagg gate report written to {report_path}")
        return results


def run_single_gate_order(
    *,
    mix: DisaggGateMix,
    output_dir: Path,
    variant_a: GateVariant,
    variant_b: GateVariant,
    seed: int,
    rung_seconds: float,
    order_index: int,
    mode: GateProcessMode = "fake",
    model_pool: Sequence[str] | None = None,
    img2img_fraction: float = 0.2,
    lora_fraction: float = 0.35,
    jobs_per_minute_estimate: float = 12.0,
    lora_pool: Sequence[str] | None = None,
    lora_pool_is_version: bool = False,
    kudos_ckpt_path: Path | None = None,
    require_kudos: bool = False,
) -> GateRunResult:
    """Run exactly one ABBA slot of one rung and merge its result into the on-disk report.

    Process-isolation escape hatch: a full in-process ladder shares one interpreter across four
    harness lifecycles, so a defect in any run's teardown can take the remaining runs with it.
    Running slot-per-process contains such a failure to its own run; the report file accumulates
    across invocations (an existing entry for the same rung and slot is replaced, so retries are safe).

    Returns:
        The slot's result, carrying the merged report path.

    Raises:
        ValueError: ``order_index`` is not one of the ABBA slots 0-3.
    """
    if not 0 <= order_index <= 3:
        raise ValueError("order_index must be one of the ABBA slots 0-3")
    variant = _abba_variant_order(variant_a, variant_b)[order_index]
    with _gate_lock(output_dir):
        run_config = GateRunConfig(
            mix=mix,
            rung_seconds=rung_seconds,
            seed=seed,
            variant=variant,
            output_dir=output_dir,
            mode=mode,
            model_pool=model_pool,
            img2img_fraction=img2img_fraction,
            lora_fraction=lora_fraction,
            jobs_per_minute_estimate=jobs_per_minute_estimate,
            lora_pool=lora_pool,
            lora_pool_is_version=lora_pool_is_version,
            kudos_ckpt_path=kudos_ckpt_path,
            require_kudos=require_kudos,
        )
        result = run_gate_rung(run_config, order_index=order_index)

        report_path = output_dir / _REPORT_NAME
        merged: list[dict[str, object]] = []
        if report_path.exists():
            merged = json.loads(report_path.read_text(encoding="utf-8"))
        merged = [
            entry
            for entry in merged
            if not (entry.get("order_index") == order_index and entry.get("rung_seconds") == rung_seconds)
        ]
        result.report_path = str(report_path)
        merged.append(result.model_dump(mode="json"))
        merged.sort(key=lambda entry: (entry["rung_seconds"], entry["order_index"]))
        report_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        logger.info(f"Disagg gate report merged at {report_path}")
        return result


def _format_results_table(results: Iterable[GateRunResult]) -> list[str]:
    """Render the results as compact fixed-width rows for the log/stdout."""
    header = f"{'rung':>6} {'ord':>3} {'variant':<12} {'jobs':>5} {'flt':>4} {'kudos/hr':>9} {'d2r':>5} {'page':>5}"
    rows = [header, "-" * len(header)]
    for result in results:
        kudos = "n/a" if result.kudos_per_hour is None else f"{result.kudos_per_hour:.1f}"
        paging = "n/a" if result.paging_cliff_events is None else str(result.paging_cliff_events)
        disk_to_ram_total = sum(result.disk_to_ram_events_by_stage.values())
        rows.append(
            f"{int(result.rung_seconds):>6} {result.order_index:>3} {result.variant_label[:12]:<12} "
            f"{result.jobs_completed:>5} {result.faults:>4} {kudos:>9} {disk_to_ram_total:>5} {paging:>5}",
        )
    return rows


def _coerce_scalar(raw: str) -> ScalarValue:
    """Conservatively parse an override value into a bool, None, int, float, or str."""
    text = raw.strip()
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    with contextlib.suppress(ValueError):
        return int(text)
    with contextlib.suppress(ValueError):
        return float(text)
    return text


def _parse_overrides(pairs: Iterable[str]) -> FlatScalarMap:
    """Parse ``key=value`` override strings into a typed flat map."""
    overrides: FlatScalarMap = {}
    for pair in pairs:
        if "=" not in pair:
            raise argparse.ArgumentTypeError(f"override {pair!r} must be in key=value form")
        key, _, raw = pair.partition("=")
        overrides[key.strip()] = _coerce_scalar(raw)
    return overrides


def _parse_env_overrides(pairs: Iterable[str]) -> dict[str, str]:
    """Parse ``KEY=value`` environment override strings; values stay strings (the environment's type)."""
    env_overrides: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise argparse.ArgumentTypeError(f"env override {pair!r} must be in KEY=value form")
        key, _, raw = pair.partition("=")
        env_overrides[key.strip()] = raw
    return env_overrides


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the disagg A/B gate ladder through the harness.")
    parser.add_argument("--mix", required=True, choices=[mix.value for mix in DisaggGateMix], help="Payload mix")
    parser.add_argument(
        "--rung",
        action="append",
        type=float,
        default=None,
        help="Rung duration in seconds (repeatable); defaults to the 90/300/600/1200 ladder",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the mix's deterministic construction")
    parser.add_argument("--mode", choices=("fake", "real"), default="fake", help="Harness process mode")
    parser.add_argument("--a-override", action="append", default=[], help="Variant A bridge override (key=value)")
    parser.add_argument("--b-override", action="append", default=[], help="Variant B bridge override (key=value)")
    parser.add_argument("--a-env", action="append", default=[], help="Variant A environment override (KEY=value)")
    parser.add_argument("--b-env", action="append", default=[], help="Variant B environment override (KEY=value)")
    parser.add_argument(
        "--single-order",
        type=int,
        default=None,
        help=(
            "Run exactly one ABBA slot (0-3) of the single configured rung and exit; the report merges "
            "across invocations. Isolates each run in its own process."
        ),
    )
    parser.add_argument("--a-label", default="A", help="Label for variant A")
    parser.add_argument("--b-label", default="B", help="Label for variant B")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for the report and artifacts")
    parser.add_argument("--kudos-ckpt", type=Path, default=None, help="Path to the server kudos checkpoint")
    parser.add_argument("--require-kudos", action="store_true", help="Fail if kudos scoring cannot be performed")
    parser.add_argument("--model-pool", action="append", default=None, help="Override model (repeatable)")
    parser.add_argument("--img2img-fraction", type=float, default=0.2, help="Share of jobs that are img2img")
    parser.add_argument("--lora-fraction", type=float, default=0.35, help="Share of jobs carrying a LoRA")
    parser.add_argument(
        "--lora-name",
        action="append",
        default=None,
        help="LoRA reference for the pool (repeatable); real mode should name LoRAs already on disk",
    )
    parser.add_argument(
        "--lora-is-version",
        action="store_true",
        help="Treat every --lora-name as a CivitAI version id (exact, cache-resolvable) rather than a model name",
    )
    parser.add_argument(
        "--jobs-per-minute",
        type=float,
        default=12.0,
        help="Expected completion rate used only to size the fixed job lists",
    )
    return parser


def _main() -> int:
    """CLI entry point: parse args, run the ladder, and print the report path."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    variant_a = GateVariant(
        label=args.a_label,
        bridge_overrides=_parse_overrides(args.a_override),
        env_overrides=_parse_env_overrides(args.a_env),
    )
    variant_b = GateVariant(
        label=args.b_label,
        bridge_overrides=_parse_overrides(args.b_override),
        env_overrides=_parse_env_overrides(args.b_env),
    )
    rungs = tuple(args.rung) if args.rung else DEFAULT_RUNGS_SECONDS

    if args.single_order is not None:
        if len(rungs) != 1:
            parser.error("--single-order requires exactly one --rung")
        single_result = run_single_gate_order(
            mix=DisaggGateMix(args.mix),
            output_dir=args.out,
            variant_a=variant_a,
            variant_b=variant_b,
            seed=args.seed,
            rung_seconds=rungs[0],
            order_index=args.single_order,
            mode=args.mode,
            model_pool=args.model_pool,
            img2img_fraction=args.img2img_fraction,
            lora_fraction=args.lora_fraction,
            jobs_per_minute_estimate=args.jobs_per_minute,
            lora_pool=args.lora_name,
            lora_pool_is_version=args.lora_is_version,
            kudos_ckpt_path=args.kudos_ckpt,
            require_kudos=args.require_kudos,
        )
        for line in _format_results_table([single_result]):
            print(line)
        print(f"Report: {args.out / _REPORT_NAME}")
        return 0

    results = run_gate_ladder(
        mix=DisaggGateMix(args.mix),
        output_dir=args.out,
        variant_a=variant_a,
        variant_b=variant_b,
        seed=args.seed,
        rungs=rungs,
        mode=args.mode,
        model_pool=args.model_pool,
        img2img_fraction=args.img2img_fraction,
        lora_fraction=args.lora_fraction,
        jobs_per_minute_estimate=args.jobs_per_minute,
        lora_pool=args.lora_name,
        lora_pool_is_version=args.lora_is_version,
        kudos_ckpt_path=args.kudos_ckpt,
        require_kudos=args.require_kudos,
    )

    for line in _format_results_table(results):
        print(line)
    print(f"Report: {args.out / _REPORT_NAME}")
    return 0


__all__ = [
    "DEFAULT_RUNGS_SECONDS",
    "GateError",
    "GatePreconditionError",
    "GateRunConfig",
    "GateRunResult",
    "GateScoringUnavailableError",
    "GateVariant",
    "run_gate_ladder",
    "run_single_gate_order",
    "run_gate_rung",
]


if __name__ == "__main__":
    sys.exit(_main())
