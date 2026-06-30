"""Attribute a probe's wall-clock to warmup versus actual inference.

A single capability probe that boots its own worker spends most of its wall-clock *not* generating:
the process spawns, imports torch, initialises the inference engine, then cold-loads a checkpoint
before the first pixel is sampled. On a reused (warm) worker that cost is paid once and amortised; on
a per-probe cold boot it is paid every time, which is why an isolated probe can read as minutes of
wall-clock at a low GPU-core duty cycle. This module turns the timestamps the harness already records
into an explicit "where did the time go" split so that cost is measured rather than guessed at.

It is pure and torch-free: it reads only the absolute :class:`JobMetricsRecord` stage timestamps and
phase metrics plus the run's start epoch and elapsed wall, so it is safe to import anywhere the rest
of the capability engine is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from horde_worker_regen.process_management.resources.run_metrics import JobMetricsRecord

_INFERENCE_START_STAGE = "INFERENCE_IN_PROGRESS"
"""Stage timestamp marking the moment a job's GPU work began (the end of pre-inference warmup)."""

_FINALIZED_STAGE = "FINALIZED"
"""Stage timestamp marking a job's completion (submission), used to bound the productive window."""

# GPU-touching phases, mirroring ``duty_cycle.GPU_BUSY_PHASES``: the model's VRAM upload plus the
# sampling, VAE-decode and prompt/image-encode passes are the wall-clock actually spent computing.
_ENCODE_PHASE_KEYS = ("clip_encode", "vae_encode")
_VAE_DECODE_PHASE_KEY = "vae_decode"


def _gpu_active_seconds_for_job(job: JobMetricsRecord) -> float:
    """Seconds this job spent doing GPU work (VRAM load, sampling, VAE decode, prompt/image encode)."""
    pm = job.phase_metrics
    if pm is None:
        return 0.0
    vram_load = sum(load.duration_seconds for load in pm.model_loads if load.phase == "ram_to_vram")
    sampling = pm.sampling.duration_seconds if pm.sampling is not None else 0.0
    phase_seconds = pm.phase_seconds or {}
    vae = phase_seconds.get(_VAE_DECODE_PHASE_KEY, 0.0)
    encode = sum(phase_seconds.get(key, 0.0) for key in _ENCODE_PHASE_KEYS)
    return vram_load + sampling + vae + encode


def _cold_model_load_seconds(first_job: JobMetricsRecord) -> float | None:
    """The first job's one-time model load (disk->RAM plus RAM->VRAM), or None without phase metrics."""
    pm = first_job.phase_metrics
    if pm is None:
        return None
    return sum(load.duration_seconds for load in pm.model_loads)


class ProbeTiming(BaseModel):
    """Where one probe's wall-clock went, split into warmup, productive inference, and teardown.

    The three wall segments (:attr:`startup_seconds`, :attr:`active_window_seconds`,
    :attr:`teardown_seconds`) are derived from coarse stage timestamps and sum to roughly
    :attr:`total_seconds`; :attr:`gpu_active_seconds` is the finer measure of time genuinely spent
    computing, summed from per-job phase metrics. A segment is ``None`` when the data needed to bound
    it is absent (no completed image job, or a driver that did not record the run-start epoch).
    """

    total_seconds: float
    """The probe's whole wall-clock (the harness elapsed time)."""
    startup_seconds: float | None = None
    """Run start to the first job's inference: process spawn plus engine init (warmup the worker boot pays)."""
    active_window_seconds: float | None = None
    """First job's inference start to the last job's completion: the window in which work is produced."""
    teardown_seconds: float | None = None
    """Last job's completion to the end of the run: shutdown and drain."""
    cold_model_load_seconds: float | None = None
    """The first job's one-time checkpoint load (disk->RAM->VRAM): warmup paid inside the active window."""
    gpu_active_seconds: float | None = None
    """Summed per-job GPU work (VRAM load, sampling, VAE, encode): time actually doing inference."""
    jobs_completed: int = 0
    """How many non-alchemy image jobs the window produced, so a rate reads in context."""

    @property
    def gpu_active_fraction(self) -> float | None:
        """Share of the whole wall-clock spent doing GPU work, or None when it cannot be measured.

        This is the headline that explains a low observed duty cycle: a per-probe cold boot can leave
        only a small fraction of the run actually computing, the rest being startup and one-time load.
        """
        if self.gpu_active_seconds is None or self.total_seconds <= 0:
            return None
        return self.gpu_active_seconds / self.total_seconds

    def summary(self) -> str:
        """A one-line breakdown for logs and reports, e.g. the warmup-versus-inference headline."""
        fraction = self.gpu_active_fraction
        head = f"total {self.total_seconds:.0f}s"
        if fraction is not None and self.gpu_active_seconds is not None:
            head += (
                f" ({fraction * 100:.0f}% inference: {self.gpu_active_seconds:.0f}s over {self.jobs_completed} jobs)"
            )

        segments: list[str] = []
        if self.startup_seconds is not None:
            segments.append(f"startup {self.startup_seconds:.0f}s")
        if self.active_window_seconds is not None:
            segments.append(f"window {self.active_window_seconds:.0f}s")
        if self.teardown_seconds is not None:
            segments.append(f"teardown {self.teardown_seconds:.0f}s")
        if self.cold_model_load_seconds is not None:
            segments.append(f"cold model-load {self.cold_model_load_seconds:.0f}s")

        return f"{head}: {', '.join(segments)}" if segments else head


def probe_timing(
    *,
    started_at_epoch: float,
    elapsed_seconds: float,
    jobs: list[JobMetricsRecord],
) -> ProbeTiming:
    """Build the warmup/inference/teardown split for one probe run.

    ``started_at_epoch`` is the harness run-start epoch (0.0 if the driver did not record it, in which
    case the startup/teardown segments are reported as unknown); ``elapsed_seconds`` is the whole wall;
    ``jobs`` is the run's per-job metrics. Only non-alchemy image jobs with stage timestamps bound the
    window, since they are what the inference processes spend their wall-clock on.
    """
    timed_jobs = [job for job in jobs if not job.is_alchemy and _INFERENCE_START_STAGE in (job.stage_timestamps or {})]
    timed_jobs.sort(key=lambda job: job.stage_timestamps[_INFERENCE_START_STAGE])

    gpu_active = sum(_gpu_active_seconds_for_job(job) for job in jobs if not job.is_alchemy)
    has_phase_metrics = any(job.phase_metrics is not None for job in jobs if not job.is_alchemy)

    if not timed_jobs:
        return ProbeTiming(
            total_seconds=elapsed_seconds,
            gpu_active_seconds=gpu_active if has_phase_metrics else None,
            jobs_completed=0,
        )

    first_inference = timed_jobs[0].stage_timestamps[_INFERENCE_START_STAGE]
    finalized_stamps = [
        job.stage_timestamps[_FINALIZED_STAGE] for job in timed_jobs if _FINALIZED_STAGE in job.stage_timestamps
    ]
    last_finalized = max(finalized_stamps) if finalized_stamps else None

    known_start = started_at_epoch > 0.0
    startup = max(0.0, first_inference - started_at_epoch) if known_start else None
    active_window = max(0.0, last_finalized - first_inference) if last_finalized is not None else None
    teardown = (
        max(0.0, elapsed_seconds - (last_finalized - started_at_epoch))
        if (known_start and last_finalized is not None)
        else None
    )

    return ProbeTiming(
        total_seconds=elapsed_seconds,
        startup_seconds=startup,
        active_window_seconds=active_window,
        teardown_seconds=teardown,
        cold_model_load_seconds=_cold_model_load_seconds(timed_jobs[0]),
        gpu_active_seconds=gpu_active if has_phase_metrics else None,
        jobs_completed=len(finalized_stamps),
    )


__all__ = ["ProbeTiming", "probe_timing"]
