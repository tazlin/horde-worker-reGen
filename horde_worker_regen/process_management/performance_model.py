"""Expected-time-to-complete model: what a dispatched job *should* cost, so "slow" is measurable.

The worker has no way to tell a genuinely-stuck job from one that is merely large until it has a
reference for how fast a job of a given shape ought to sample. This module supplies that reference as
expected sampling iterations-per-second (it/s) keyed by a coarse :class:`JobSignature` (baseline +
resolution/steps/batch buckets + controlnet/hires flags). Two sources feed it:

- **Benchmark seed:** a prior ``report.json`` records each model tier's stage-A reference it/s
  (``BenchmarkReport.tier_baselines_its``). Each maps to the exact baseline signature that level ran
  (native resolution, default steps, batch 1, no features), so the seed only answers for baseline-like
  jobs and never over-promises a rate for a feature-laden one.
- **Self-calibration:** every completed image job contributes its observed sampling it/s to a bounded
  rolling window for its signature. Once a signature has enough samples its learned median takes over
  from the seed, and signatures the benchmark never covered become answerable too.

Nothing is enforced until a signature has a seed or enough samples, so a cold start raises no false
alarms. The learned table persists to ``.horde_worker_regen/perf_model.json`` so calibration survives
restarts. We use :meth:`PerformanceModel.expected_sampling_seconds` elsewhere to grade a running job;
this module only measures, it does not act.

The module is deliberately dependency-light: the benchmark/hordelib chains are imported lazily inside
:func:`load_seed_its_by_signature` so the model itself can be constructed early and in tests.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import enum
import json
import os
import statistics
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

    from horde_worker_regen.process_management.job_models import HordeJobInfo
    from horde_worker_regen.process_management.job_tracker import TrackedJob
    from horde_worker_regen.process_management.messages import HordeJobMetricsMessage

PERF_MODEL_SCHEMA_VERSION = 1
"""Bumped when the persisted schema changes incompatibly; an older file is discarded on read."""

PERF_MODEL_FILENAME = "perf_model.json"

BENCHMARK_BASELINE_STEPS = 30
"""Steps the stage-A baseline levels sample at (mirrors ``CannedImageJobSpec.steps``). Used to build the
exact signature a benchmark tier baseline was measured against when seeding from a report."""

_HIRES_FIX_ITERATION_MULTIPLIER = 2.0
"""Hires-fix runs a second sampling pass, so a hires job samples roughly twice the nominal steps. The
sampler's it/s already blends both passes, so its signature is calibrated separately (see
``has_hires_fix`` in the key) and its iteration count is scaled by this factor for the time estimate."""

_DEFAULT_MIN_SAMPLES = 5
"""How many observations a signature needs before its learned median is trusted over the seed/None."""

_DEFAULT_MAX_SAMPLES_PER_SIGNATURE = 50
"""Rolling-window size per signature; old samples age out so the model tracks recent hardware reality."""

_PERSIST_EVERY_N_OBSERVATIONS = 10
"""Throttle disk writes: persist after this many new observations (plus an explicit save on shutdown)."""


class ResolutionBucket(enum.StrEnum):
    """Coarse output-size band (by megapixels), since sampling cost scales with pixel count."""

    TINY = "<=0.3MP"
    SMALL = "<=0.8MP"
    MEDIUM = "<=1.3MP"
    LARGE = "<=2.5MP"
    HUGE = ">2.5MP"


class StepsBucket(enum.StrEnum):
    """Coarse sampling-steps band."""

    VERY_LOW = "<=10"
    LOW = "<=20"
    MEDIUM = "<=35"
    HIGH = "<=60"
    VERY_HIGH = ">60"


class BatchBucket(enum.StrEnum):
    """Coarse batch-size (``n_iter``) band; batching lowers per-step it/s, so it is calibrated apart."""

    SINGLE = "1"
    PAIR = "2"
    SMALL = "3-4"
    LARGE = "5+"


def _resolution_bucket(width: int, height: int) -> ResolutionBucket:
    """Band an output size by total megapixels."""
    megapixels = (max(0, width) * max(0, height)) / 1_000_000.0
    if megapixels <= 0.3:
        return ResolutionBucket.TINY
    if megapixels <= 0.8:
        return ResolutionBucket.SMALL
    if megapixels <= 1.3:
        return ResolutionBucket.MEDIUM
    if megapixels <= 2.5:
        return ResolutionBucket.LARGE
    return ResolutionBucket.HUGE


def _steps_bucket(steps: int) -> StepsBucket:
    """Band a sampling-steps count."""
    if steps <= 10:
        return StepsBucket.VERY_LOW
    if steps <= 20:
        return StepsBucket.LOW
    if steps <= 35:
        return StepsBucket.MEDIUM
    if steps <= 60:
        return StepsBucket.HIGH
    return StepsBucket.VERY_HIGH


def _batch_bucket(n_iter: int) -> BatchBucket:
    """Band a batch size (``n_iter``)."""
    if n_iter <= 1:
        return BatchBucket.SINGLE
    if n_iter == 2:
        return BatchBucket.PAIR
    if n_iter <= 4:
        return BatchBucket.SMALL
    return BatchBucket.LARGE


@dataclasses.dataclass(frozen=True)
class JobSignature:
    """A coarse fingerprint of a job's inference shape, used to look up an expected sampling rate.

    ``total_sampling_iterations`` is the job's actual iteration count (steps, doubled for hires-fix); it
    is excluded from equality/hashing (``compare=False``) and from :attr:`key` so two jobs in the same
    band but with slightly different step counts share one calibration bucket while still each computing
    their own expected seconds.
    """

    baseline: str
    resolution_bucket: ResolutionBucket
    steps_bucket: StepsBucket
    batch_bucket: BatchBucket
    has_controlnet: bool
    has_hires_fix: bool
    total_sampling_iterations: int = dataclasses.field(default=0, compare=False)

    @property
    def key(self) -> str:
        """A stable string key for the calibration/seed tables (the iteration count is intentionally out)."""
        controlnet = "cn" if self.has_controlnet else "nocn"
        hires = "hires" if self.has_hires_fix else "nohires"
        return (
            f"{self.baseline}|{self.resolution_bucket.value}|{self.steps_bucket.value}|"
            f"{self.batch_bucket.value}|{controlnet}|{hires}"
        )

    @property
    def is_baseline_like(self) -> bool:
        """Whether this matches the conservative shape a benchmark tier baseline is measured at."""
        return self.batch_bucket == BatchBucket.SINGLE and not self.has_controlnet and not self.has_hires_fix


def baseline_signature(*, baseline: str, resolution: int) -> JobSignature:
    """Build the exact stage-A baseline signature for a tier (native square resolution, default steps)."""
    return JobSignature(
        baseline=baseline,
        resolution_bucket=_resolution_bucket(resolution, resolution),
        steps_bucket=_steps_bucket(BENCHMARK_BASELINE_STEPS),
        batch_bucket=_batch_bucket(1),
        has_controlnet=False,
        has_hires_fix=False,
        total_sampling_iterations=BENCHMARK_BASELINE_STEPS,
    )


def signature_from_job(job: ImageGenerateJobPopResponse, baseline: str | None) -> JobSignature | None:
    """Derive a :class:`JobSignature` from a job pop response, or ``None`` if it cannot be characterized.

    Returns ``None`` when the baseline is unknown or the job declares no sampling steps (e.g. a malformed
    payload), since neither a calibration bucket nor an expected time would be meaningful.
    """
    if baseline is None:
        return None

    payload = job.payload
    steps = payload.ddim_steps if payload.ddim_steps is not None else 0
    if steps <= 0:
        return None

    width = payload.width if payload.width is not None else 0
    height = payload.height if payload.height is not None else 0
    n_iter = payload.n_iter if payload.n_iter is not None else 1
    has_controlnet = payload.control_type is not None
    has_hires_fix = bool(payload.hires_fix)

    total_iterations = round(steps * _HIRES_FIX_ITERATION_MULTIPLIER) if has_hires_fix else steps

    return JobSignature(
        baseline=str(baseline),
        resolution_bucket=_resolution_bucket(width, height),
        steps_bucket=_steps_bucket(steps),
        batch_bucket=_batch_bucket(n_iter),
        has_controlnet=has_controlnet,
        has_hires_fix=has_hires_fix,
        total_sampling_iterations=total_iterations,
    )


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text to ``path`` atomically: a temp file in the same directory, then ``os.replace``."""
    handle, temp_path_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    temp_path = Path(temp_path_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as temp_file:
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            temp_path.unlink()
        raise


def load_seed_its_by_signature(results_dir: Path | str) -> dict[str, float]:
    """Build a seed ``{signature key: it/s}`` table from a benchmark ``report.json``, or ``{}`` if absent.

    Each tier's reference it/s is mapped to the exact baseline signature that the tier's stage-A level
    ran at, so the seed only answers for baseline-like jobs. Reads never raise: a missing, unreadable, or
    schema-mismatched report yields an empty seed rather than blocking worker startup. The benchmark
    import chain is loaded lazily here so this module stays import-light.
    """
    from horde_worker_regen.benchmark.ladder import _TIER_BASELINES, _TIER_RESOLUTIONS
    from horde_worker_regen.benchmark.report import BenchmarkReport

    report_path = Path(results_dir) / "report.json"
    if not report_path.exists():
        return {}

    try:
        report = BenchmarkReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as read_error:
        logger.debug(f"Could not read benchmark report at {report_path} ({read_error}); not seeding perf model.")
        return {}

    seed: dict[str, float] = {}
    for tier, its_p50 in report.tier_baselines_its.items():
        baseline = _TIER_BASELINES.get(tier)
        resolution = _TIER_RESOLUTIONS.get(tier)
        if baseline is None or resolution is None or its_p50 <= 0:
            continue
        seed[baseline_signature(baseline=baseline, resolution=resolution).key] = its_p50

    return seed


class PerformanceModel:
    """Tracks expected sampling it/s per :class:`JobSignature`, seeded by benchmark and self-calibrating.

    Lifecycle in the worker: constructed with an optional benchmark seed and persistence path; fed every
    child per-job metrics message (:meth:`on_job_metrics`, which caches the observed it/s) and every job
    finalization (:meth:`on_job_finalized`, which pairs that it/s with the job's signature and learns
    from it). Consumers ask :meth:`expected_sampling_seconds` / :meth:`expected_its`.

    Thread safety: not internally synchronized; like the rest of the process-manager collaborators it is
    only touched from the main control loop's thread.
    """

    def __init__(
        self,
        *,
        seed_its_by_signature: Mapping[str, float] | None = None,
        path: Path | None = None,
        baseline_resolver: Callable[[str], str | None] | None = None,
        min_samples: int = _DEFAULT_MIN_SAMPLES,
        max_samples_per_signature: int = _DEFAULT_MAX_SAMPLES_PER_SIGNATURE,
    ) -> None:
        """Initialize the model.

        Args:
            seed_its_by_signature: Benchmark-derived ``{signature key: it/s}`` reference rates, used until a
                signature has enough self-calibration samples.
            path: Where to persist the learned table; ``None`` keeps the model purely in memory (tests).
            baseline_resolver: Maps a model name to its baseline string (e.g. the model reference lookup),
                so :meth:`on_job_finalized` can build a signature. When ``None``, finalized jobs are not
                learned from (the model still serves seeded answers).
            min_samples: Observations a signature needs before its learned median is trusted.
            max_samples_per_signature: Rolling-window size per signature.
        """
        self._seed_its: dict[str, float] = dict(seed_its_by_signature or {})
        self._path = path
        self._baseline_resolver = baseline_resolver
        self._min_samples = max(1, min_samples)
        self._max_samples = max(1, max_samples_per_signature)

        self._calibration_samples: dict[str, collections.deque[float]] = {}
        self._its_by_job_id: dict[str, float] = {}
        self._observations_since_save = 0
        self._file_disabled = False

        self._load()

    # region queries

    def expected_its(self, signature: JobSignature) -> float | None:
        """The expected sampling it/s for this signature, or ``None`` when nothing yet answers for it.

        Prefers the self-calibrated median once a signature has at least ``min_samples`` observations;
        otherwise falls back to the benchmark seed (which only carries baseline-like signatures).
        """
        samples = self._calibration_samples.get(signature.key)
        if samples is not None and len(samples) >= self._min_samples:
            return statistics.median(samples)
        return self._seed_its.get(signature.key)

    def expected_sampling_seconds(self, signature: JobSignature) -> float | None:
        """Expected wall seconds to sample this job, or ``None`` when no rate is known yet.

        ``signature.total_sampling_iterations / expected_its``. This is sampling time only: model load,
        conditioning, VAE, safety and submit are deliberately excluded because they are highly variable
        and not what "the GPU is sampling too slowly" means.
        """
        its = self.expected_its(signature)
        if its is None or its <= 0:
            return None
        return signature.total_sampling_iterations / its

    def sample_count(self, signature: JobSignature) -> int:
        """How many self-calibration samples this signature currently holds."""
        samples = self._calibration_samples.get(signature.key)
        return len(samples) if samples is not None else 0

    # endregion

    # region feeds

    def observe(self, signature: JobSignature, observed_its: float) -> None:
        """Fold one observed sampling it/s into the rolling window for ``signature``."""
        if observed_its <= 0:
            return
        samples = self._calibration_samples.get(signature.key)
        if samples is None:
            samples = collections.deque(maxlen=self._max_samples)
            self._calibration_samples[signature.key] = samples
        samples.append(observed_its)

        self._observations_since_save += 1
        if self._observations_since_save >= _PERSIST_EVERY_N_OBSERVATIONS:
            self.save()

    def on_job_metrics(self, message: HordeJobMetricsMessage) -> None:
        """Cache the sampling it/s from a child's per-job metrics message (image jobs only).

        Image-job metrics arrive before the job finalizes, so the rate is held by job id until
        :meth:`on_job_finalized` can pair it with the job's signature. Alchemy forms have no sampling
        signature and are ignored.
        """
        if message.is_alchemy:
            return
        sampling = message.phase_metrics.sampling
        if sampling is not None and sampling.iterations_per_second > 0:
            self._its_by_job_id[message.job_id] = sampling.iterations_per_second

    def on_job_finalized(self, tracked: TrackedJob, completed_job_info: HordeJobInfo) -> None:
        """Learn from a finalized image job: pair its cached sampling it/s with its signature.

        Logs the prediction-vs-actual it/s at debug (the model's prior expectation, computed before this
        sample is folded in) so a slow job is visible in the logs even before we act on it.
        """
        job_id = str(tracked.job_id)
        observed_its = self._its_by_job_id.pop(job_id, None)
        if observed_its is None:
            return

        job = tracked.sdk_api_job_info
        baseline = self._baseline_resolver(job.model) if self._baseline_resolver and job.model is not None else None
        signature = signature_from_job(job, baseline)
        if signature is None:
            return

        expected_its = self.expected_its(signature)
        if expected_its is not None and expected_its > 0:
            ratio = observed_its / expected_its
            logger.debug(
                f"Job {job_id[:8]} sampled at {observed_its:.2f} it/s vs expected {expected_its:.2f} "
                f"({ratio:.0%}) for signature {signature.key}",
            )

        self.observe(signature, observed_its)

    def forget_job(self, job_id: str) -> None:
        """Drop any cached it/s for a job that will not finalize (faulted/abandoned), bounding the cache."""
        self._its_by_job_id.pop(str(job_id), None)

    # endregion

    # region persistence

    def save(self) -> None:
        """Persist the learned calibration table atomically. No-op without a path; never raises."""
        self._observations_since_save = 0
        if self._path is None or self._file_disabled:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": PERF_MODEL_SCHEMA_VERSION,
                "samples": {key: list(values) for key, values in self._calibration_samples.items()},
            }
            _atomic_write_text(self._path, json.dumps(payload))
        except OSError as write_error:
            # A perf-model write must never take the worker down; degrade to in-memory and stop retrying.
            logger.debug(f"Could not persist perf model to {self._path} ({write_error}); continuing in memory.")
            self._file_disabled = True

    def _load(self) -> None:
        """Load a previously persisted calibration table, tolerating a missing or corrupt file."""
        if self._path is None or not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as read_error:
            logger.debug(f"Could not read perf model at {self._path} ({read_error}); starting uncalibrated.")
            return

        if not isinstance(raw, dict) or raw.get("schema_version") != PERF_MODEL_SCHEMA_VERSION:
            return
        samples = raw.get("samples")
        if not isinstance(samples, dict):
            return

        for key, values in samples.items():
            if not isinstance(key, str) or not isinstance(values, list):
                continue
            floats = [float(value) for value in values if isinstance(value, (int, float))]
            if floats:
                self._calibration_samples[key] = collections.deque(
                    floats[-self._max_samples :],
                    maxlen=self._max_samples,
                )

    # endregion
