"""A worker-owned VRAM budget so concurrent inference processes do not over-commit one device.

The worker spawns several inference processes that each load models into the *same* GPU
independently. Without a shared accountant, nothing stops their combined resident footprint from
exceeding device VRAM, which is the multi-process over-commit that produced the observed live OOM
(several processes resident, ~277 MiB free, death during tiled VAE decode).

This module predicts a job's peak VRAM from hordelib's per-job burden estimate
(:func:`hordelib.api.estimate_job_burden`, the same estimate the benchmark pre-flight trusts) and
compares it against the device's *measured* free VRAM plus a reserve for transient spikes. The
prediction is intentionally the conservative hordelib estimate rather than a learned per-job
measurement: on a shared device the only measurement available (per-process VRAM high-water) is
device-wide and so reflects *every* resident model, not the marginal cost of one job, so feeding it
back would massively over-throttle a multi-model worker. Refining the prediction with a true
*marginal* per-job measurement is a hordelib-side follow-up.

The shape mirrors :class:`~horde_worker_regen.process_management.alchemy_popper.AlchemyHeadroomEstimator`,
which already gates graph-alchemy forms the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from loguru import logger

if TYPE_CHECKING:
    from hordelib.feature_impact import FEATURE_KIND, BurdenEstimate


@dataclass(frozen=True)
class BudgetVerdict:
    """Represents the outcome of a single resource-budget check, with enough detail to log a reason.

    Shared by the VRAM and RAM budgets; ``predicted_mb`` and ``available_mb`` carry whichever resource
    the producing budget measures (free device VRAM, or available system RAM).
    """

    fits: bool
    """Whether the job's predicted cost plus the reserve fits the measured available resource."""
    predicted_mb: float | None
    """The job's predicted cost (MB) for this resource, or None when no estimate could be produced."""
    available_mb: float | None
    """The measured available resource (MB) at check time, or None when no telemetry exists yet."""
    reserve_mb: float
    """The reserve (MB) required on top of the prediction."""

    def reason(self) -> str:
        """Return a short human-readable explanation, for logging an admit/defer decision."""
        if self.available_mb is None:
            return "no telemetry yet (cold start); admitted"
        if self.predicted_mb is None:
            return f"no burden estimate; admitted on {self.available_mb:.0f} MB available"
        verb = "fits" if self.fits else "does NOT fit"
        return (
            f"job needs ~{self.predicted_mb:.0f} MB + {self.reserve_mb:.0f} MB reserve "
            f"vs {self.available_mb:.0f} MB available: {verb}"
        )


def _job_feature_kinds(job: ImageGenerateJobPopResponse) -> list[FEATURE_KIND]:
    """Return the hordelib ``FEATURE_KIND`` values a live job's payload implies, for burden estimation.

    Exactness is not required: the baseline term dominates the estimate and the feature deltas only
    refine it, so a missed feature errs slightly low and an unknown extra is simply ignored by the
    registry. ``FEATURE_KIND`` is imported lazily so the parent process does not eagerly pull hordelib.
    """
    from horde_sdk.generation_parameters import KNOWN_FACEFIXERS, KNOWN_UPSCALERS
    from hordelib.feature_impact import FEATURE_KIND

    payload = job.payload
    features: list[FEATURE_KIND] = []

    if payload.loras:
        features.append(FEATURE_KIND.lora)
    if payload.tis:
        features.append(FEATURE_KIND.ti)
    if payload.control_type:
        features.append(FEATURE_KIND.controlnet)
    if payload.hires_fix:
        features.append(FEATURE_KIND.hires_fix)
    if job.source_image is not None:
        features.append(FEATURE_KIND.img2img)

    post_processing = payload.post_processing or []
    upscaler_values = {u.value for u in KNOWN_UPSCALERS}
    facefix_values = {u.value for u in KNOWN_FACEFIXERS}
    if any(pp in upscaler_values for pp in post_processing):
        features.append(FEATURE_KIND.post_processing_upscale)
    if any(pp in facefix_values for pp in post_processing):
        features.append(FEATURE_KIND.post_processing_facefix)

    return features


def predict_job_vram_mb(job: ImageGenerateJobPopResponse, baseline: str | None) -> float | None:
    """Return a job's predicted peak VRAM (MB) via hordelib's burden estimate, or None when unavailable.

    Never raises: a missing baseline falls back to hordelib's heavy seed, and any unexpected error
    yields None so the caller treats the cost as unknown (and admits) rather than crashing the
    scheduling cycle.
    """
    burden = _estimate_job_burden(job, baseline)
    return None if burden is None else float(burden.vram_mb)


def predict_job_ram_mb(job: ImageGenerateJobPopResponse, baseline: str | None) -> float | None:
    """Return a job's predicted system-RAM cost (MB) via hordelib's burden estimate, or None.

    The RAM analogue of :func:`predict_job_vram_mb`; used by the RAM budget to keep resident-in-RAM
    weights from forcing the OS to page. Never raises (see :func:`predict_job_vram_mb`).
    """
    burden = _estimate_job_burden(job, baseline)
    return None if burden is None else float(burden.ram_mb)


def _estimate_job_burden(job: ImageGenerateJobPopResponse, baseline: str | None) -> BurdenEstimate | None:
    """Return hordelib's ``BurdenEstimate`` for a job, or None when the estimate cannot be produced.

    Imported lazily so the parent process never eagerly pulls hordelib. Never raises: any error is
    logged at debug and yields None so the scheduling cycle is never crashed by a bad estimate.
    """
    try:
        from hordelib.api import estimate_job_burden

        return estimate_job_burden(
            baseline=str(baseline) if baseline is not None else "",
            width=job.payload.width,
            height=job.payload.height,
            batch=max(1, job.payload.n_iter),
            features=_job_feature_kinds(job),
        )
    except Exception as e:
        logger.debug(f"Job burden estimate failed for job {getattr(job, 'id_', None)}: {type(e).__name__} {e}")
        return None


class VramBudget:
    """Decides whether the device's measured free VRAM can absorb another job's predicted peak.

    Stateless beyond its configured reserve: the device-wide free figure already reflects every
    resident model across all processes, so the budget needs no per-model bookkeeping of its own.
    """

    def __init__(self, *, reserve_mb: float) -> None:
        """Initialize with the reserve (MB) to keep free on top of any job's predicted peak."""
        self._reserve_mb = reserve_mb

    @property
    def reserve_mb(self) -> float:
        """The reserve (MB) kept free on top of a job's predicted peak."""
        return self._reserve_mb

    def set_reserve_mb(self, reserve_mb: float) -> None:
        """Update the reserve (MB); honored live on config reload."""
        self._reserve_mb = reserve_mb

    def check_job(
        self,
        job: ImageGenerateJobPopResponse,
        baseline: str | None,
        free_vram_mb: float | None,
    ) -> BudgetVerdict:
        """Return the budget verdict for admitting ``job`` given the measured free VRAM.

        Admits (fits=True) when telemetry is absent (cold start) or no estimate is available, so the
        budget never wedges a worker that has not yet reported VRAM; otherwise requires
        ``free >= predicted + reserve``.
        """
        if free_vram_mb is None:
            return BudgetVerdict(fits=True, predicted_mb=None, available_mb=None, reserve_mb=self._reserve_mb)

        predicted = predict_job_vram_mb(job, baseline)
        if predicted is None:
            return BudgetVerdict(
                fits=True,
                predicted_mb=None,
                available_mb=free_vram_mb,
                reserve_mb=self._reserve_mb,
            )

        fits = free_vram_mb >= predicted + self._reserve_mb
        return BudgetVerdict(
            fits=fits,
            predicted_mb=predicted,
            available_mb=free_vram_mb,
            reserve_mb=self._reserve_mb,
        )


class RamBudget:
    """Decides whether measured available system RAM can absorb another job's predicted RAM cost.

    The RAM analogue of :class:`VramBudget`. ``high_memory_mode`` keeps model weights resident in
    system RAM as well as VRAM; with several processes that can exhaust RAM and force the OS to page
    to disk (observed in the live run as a worker paging out under load), which collapses throughput.
    The available-RAM figure is system-wide (it already reflects every process), so like the VRAM
    budget this needs no per-process bookkeeping.
    """

    def __init__(self, *, reserve_mb: float) -> None:
        """Initialize with the reserve (MB) to keep available on top of any job's predicted RAM cost."""
        self._reserve_mb = reserve_mb

    @property
    def reserve_mb(self) -> float:
        """The reserve (MB) kept available on top of a job's predicted RAM cost."""
        return self._reserve_mb

    def set_reserve_mb(self, reserve_mb: float) -> None:
        """Update the reserve (MB); honored live on config reload."""
        self._reserve_mb = reserve_mb

    def check_job(
        self,
        job: ImageGenerateJobPopResponse,
        baseline: str | None,
        available_ram_mb: float | None,
    ) -> BudgetVerdict:
        """Return the budget verdict for admitting ``job`` given the measured available system RAM.

        Admits (fits=True) when no measurement or estimate is available, so the budget never wedges a
        worker; otherwise requires ``available >= predicted + reserve``.
        """
        if available_ram_mb is None:
            return BudgetVerdict(fits=True, predicted_mb=None, available_mb=None, reserve_mb=self._reserve_mb)

        predicted = predict_job_ram_mb(job, baseline)
        if predicted is None:
            return BudgetVerdict(
                fits=True,
                predicted_mb=None,
                available_mb=available_ram_mb,
                reserve_mb=self._reserve_mb,
            )

        fits = available_ram_mb >= predicted + self._reserve_mb
        return BudgetVerdict(
            fits=fits,
            predicted_mb=predicted,
            available_mb=available_ram_mb,
            reserve_mb=self._reserve_mb,
        )
