"""A worker-owned VRAM budget so concurrent inference processes do not over-commit one device.

The worker spawns several inference processes that each load models into the *same* GPU
independently. Without a shared accountant, nothing stops their combined resident footprint from
exceeding device VRAM: a multi-process over-commit that drives the device out of memory once a
transient spike (a tiled VAE decode, say) lands with several models resident and little free VRAM left.

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

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from loguru import logger

if TYPE_CHECKING:
    from hordelib.feature_impact import FEATURE_KIND, BurdenEstimate

    from horde_worker_regen.bridge_data.data_model import reGenBridgeData
    from horde_worker_regen.process_management.job_tracker import JobTracker


def is_model_locally_unservable(
    *,
    overbudget_fault_count: int,
    last_overbudget_fault_time: float | None,
    threshold: int,
    cooldown_seconds: float,
    now: float | None = None,
) -> bool:
    """Return whether a model's over-budget fault streak marks it locally unservable (held back).

    Pure admission policy over raw counters: a model whose consecutive over-budget *terminal* faults reach
    ``threshold`` is held back until ``cooldown_seconds`` have elapsed since its last such fault. A model
    the device genuinely cannot run faults every attempt no matter how it is isolated, so holding it back
    is what stops the worker dropping jobs faster than the horde server tolerates. ``threshold <= 0``
    disables the breaker. Prefer :func:`is_model_locally_unservable_for` at call sites that have the
    config and tracker in hand; this primitive is the table-testable core.
    """
    if threshold <= 0 or overbudget_fault_count < threshold:
        return False
    if last_overbudget_fault_time is None:
        return False
    current = time.time() if now is None else now
    return (current - last_overbudget_fault_time) < cooldown_seconds


def is_model_locally_unservable_for(
    bridge_data: reGenBridgeData,
    job_tracker: JobTracker,
    model: str | None,
    *,
    now: float | None = None,
) -> bool:
    """Return whether ``model`` is currently held back as locally unservable, per config and fault history.

    The single policy point shared by the scheduler's best-effort-admit gate and the popper's model
    selection, so popping and admitting never disagree on which models are held back. Reads the configured
    breaker thresholds from ``bridge_data`` and the per-model fault streak from ``job_tracker``. Tolerant of
    partially-mocked config: a non-integer threshold (or one ``<= 0``) disables the breaker.
    """
    if model is None:
        return False
    threshold = bridge_data.unservable_model_fault_threshold
    if not isinstance(threshold, int) or isinstance(threshold, bool):
        return False
    cooldown = bridge_data.unservable_model_cooldown_seconds
    cooldown_is_numeric = isinstance(cooldown, (int, float)) and not isinstance(cooldown, bool)
    cooldown_seconds = float(cooldown) if cooldown_is_numeric else 0.0
    return is_model_locally_unservable(
        overbudget_fault_count=job_tracker.get_model_overbudget_fault_count(model),
        last_overbudget_fault_time=job_tracker.model_last_overbudget_fault_time(model),
        threshold=threshold,
        cooldown_seconds=cooldown_seconds,
        now=now,
    )


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


@dataclass(frozen=True)
class StreamForecast:
    """Forecast of whether loading a model will make ComfyUI stream its weights from host RAM.

    ComfyUI keeps a model's weights resident only while free VRAM stays above its inference reserve
    (``minimum_inference_memory()``, mirrored torch-free by
    :func:`hordelib.vram_planning.compute_inference_reserve_mb`). Below that reserve the model's weights
    and/or per-step activations overflow to host RAM and stream to the device every sampling step, which
    collapses the step rate; on a memory-constrained device a heavy model loaded alongside other resident
    models is easily driven into this state, and the slowdown can be mistaken for a hang. The overflow can
    come from ComfyUI's own weight offloading *or* from the GPU driver's system-memory fallback once free
    VRAM nears zero (the latter is invisible to ComfyUI's own offload accounting), so this forecast reasons
    purely about measured/derivable free VRAM rather than any backend offload counter.

    A subtlety this forecast exists to capture: each sibling inference process carries a fixed runtime
    (CUDA) context of roughly a gigabyte that is *not* reclaimed by evicting its model, only by stopping
    the process. That context may not have materialised at the instant a load is admitted (idle processes
    allocate lazily), so the instantaneous free-VRAM measurement can read deceptively high and then collapse
    once the siblings touch the device. The forecast therefore compares the model against three capacities:
    free *now*, free after sibling *models* are evicted (their contexts remaining), and free with the card
    to itself (siblings stopped). That lets the scheduler choose the least-disruptive remedy that actually
    works: load as-is, evict sibling models, stop sibling processes, or tolerate an unavoidable streamed run
    under the over-budget step grace.

    All fields are MB. ``free_if_alone_mb`` is the free VRAM achievable with sole residency (every other
    process stopped so only this process's context remains): ``total`` minus one process's overhead.
    ``free_after_model_evict_mb`` is the free VRAM achievable while the sibling processes stay alive but
    hold no model (every process's context resident): ``total`` minus all processes' overhead.
    """

    weights_mb: float | None
    """Resident weight footprint ComfyUI compares against its budget, or None when it cannot be estimated."""
    reserve_mb: float
    """The free-VRAM headroom that must remain after the weights load to avoid streaming."""
    free_now_mb: float | None
    """Measured device-wide free VRAM now, or None at cold start (before any process has reported)."""
    free_if_alone_mb: float | None
    """Free VRAM achievable with sole residency (siblings stopped), or None when total VRAM is unknown."""
    free_after_model_evict_mb: float | None
    """Free VRAM achievable with siblings alive but model-free (all contexts resident), or None when total
    VRAM is unknown. Sits between ``free_now_mb`` and ``free_if_alone_mb``: lower than alone (sibling
    contexts still cost VRAM) but it is the best a model can get without stopping any process."""
    total_vram_mb: float | None = None
    """Device total VRAM (MB), or None when unknown. Kept so the forecast can size a partial teardown."""
    per_process_overhead_mb: float = 0.0
    """Per-process runtime/context VRAM (MB). Kept so the forecast can size a partial teardown."""
    base_reserve_mb: float | None = None
    """The bounded inference-reserve floor (ComfyUI ``minimum_inference_memory`` / the configured floor).

    Sizes the decisions about the persistent *weight* footprint (``fits_alone``, ``streams_unavoidably``,
    and the weight-headroom gate), as distinct from the activation-inclusive ``reserve_mb`` that sizes the
    co-resident streaming check and the teardown *depth*. Folding the conservative, batch-and-resolution
    scaled activation peak into every fit decision conflates a transient activation spike with the persistent
    footprint: it can flip a moderate-weight model into claiming the whole card, or push a model whose weights
    fit alone marginally past free-if-alone so it falsely reads as streaming-unavoidable. Keeping the weight
    decisions on this bounded floor holds them independent of the activation estimate. None falls back to
    ``reserve_mb`` so a directly-constructed forecast keeps its prior single-reserve behavior."""

    @property
    def known(self) -> bool:
        """Whether enough is known (weight estimate and a current measurement) to forecast at all."""
        return self.weights_mb is not None and self.free_now_mb is not None

    @property
    def _effective_base_reserve(self) -> float:
        """The bounded weight-footprint reserve, falling back to ``reserve_mb`` when unset."""
        return self.base_reserve_mb if self.base_reserve_mb is not None else self.reserve_mb

    def _fits(self, free_mb: float | None, reserve_mb: float) -> bool:
        """Whether the weights plus ``reserve_mb`` fit within ``free_mb`` (None capacity admits)."""
        if not self.known or free_mb is None:
            return True
        assert self.weights_mb is not None
        return (free_mb - self.weights_mb) >= reserve_mb

    def _fits_peak(self, free_mb: float | None) -> bool:
        """Whether the weights plus the activation-inclusive reserve fit (the co-resident streaming test)."""
        return self._fits(free_mb, self.reserve_mb)

    def _fits_weights(self, free_mb: float | None) -> bool:
        """Whether the persistent weight footprint plus the bounded floor fits (the residency test)."""
        return self._fits(free_mb, self._effective_base_reserve)

    @property
    def _weights_dominant(self) -> bool:
        """Whether the model is too heavy to co-reside with even one sibling context (a whole-card model).

        Tearing down sibling *processes* only reclaims their fixed ~1GB CUDA contexts, so it helps a model
        whose own weights-plus-activations leave no room for another context, not one whose large estimate is
        a transient activation spike. The gate is keyed on the activation-inclusive peak against the ceiling
        of sole-occupancy-plus-one-sibling-context (``total - 2*overhead``): a model that cannot fit even
        there genuinely needs the card. When total VRAM or the per-process overhead is unknown (a
        directly-constructed forecast) it defaults True, preserving the prior, more eager behavior.
        """
        if self.total_vram_mb is None or self.per_process_overhead_mb <= 0:
            return True
        return not self._fits_peak(self.total_vram_mb - 2 * self.per_process_overhead_mb)

    @property
    def fits_coresident(self) -> bool:
        """True when the model loads without streaming and without evicting or stopping anything.

        Requires headroom against *both* the instantaneous free measurement and the structural floor once
        every process's context has materialised (``free_after_model_evict_mb``). The structural check is
        what stops a deceptively-high instantaneous reading (idle siblings whose contexts have not yet
        allocated) from admitting a heavy model that then collapses free VRAM as those contexts appear.
        Unknown cost or a cold start admits (True) so the forecast never blocks a load on a guess.
        """
        return self._fits_peak(self.free_now_mb) and self._fits_peak(self.free_after_model_evict_mb)

    @property
    def fits_after_model_evict(self) -> bool:
        """True when evicting sibling *models* (their contexts remaining) leaves enough headroom."""
        return self._fits_peak(self.free_after_model_evict_mb)

    @property
    def fits_alone(self) -> bool:
        """True when sole residency (siblings stopped) fits the persistent weight footprint.

        Keyed on the bounded weight reserve, not the activation-inclusive peak: a model whose *weights* fit
        the card alone is servable via whole-card residency (it loads with the bounded inference reserve free
        and runs, slowly at high resolution, under the over-budget step grace). Sizing this on the conservative
        activation peak instead can push a model whose weights fit alone marginally past free-if-alone, so it
        falsely reads as streaming-unavoidable and skips the clean whole-card path.
        """
        return self._fits_weights(self.free_if_alone_mb)

    @property
    def needs_exclusive_residency(self) -> bool:
        """Streams co-resident, fits alone, and is weight-dominant: the scheduler should give it the whole card.

        The ``_weights_dominant`` gate is what distinguishes a genuine whole-card model (Flux: heavy weights
        leave no room for a sibling context) from a moderate-weight model with a large transient activation
        estimate (SDXL at a big batch/resolution: it can co-reside, so it must reclaim a sibling *model*
        through the normal budget path, not claim the device and tear down sibling *processes*).
        """
        return self.known and not self.fits_coresident and self.fits_alone and self._weights_dominant

    @property
    def requires_sibling_teardown(self) -> bool:
        """The model cannot fit even after sibling models are evicted, fits alone, and is weight-dominant.

        When True, dropping the siblings' resident models is not enough: their fixed per-process contexts
        themselves over-commit the device, so the scheduler must *stop* idle sibling processes to reclaim
        that VRAM (a context is only freed by the process exiting, never by emptying the allocator cache).
        Gated on ``_weights_dominant`` so only a genuinely card-filling model triggers a process teardown.
        """
        return self.known and not self.fits_after_model_evict and self.fits_alone and self._weights_dominant

    @property
    def streams_unavoidably(self) -> bool:
        """Streams even with the whole device to itself (e.g. fp16 weights on a too-small card).

        Weight-based (via ``fits_alone``): only a model whose persistent weights overflow the card alone is
        truly unservable. A heavy-activation model whose weights fit alone is not unavoidable, it just needs
        sole residency.
        """
        return self.known and not self.fits_alone

    def max_resident_processes(self) -> int | None:
        """Largest number of co-resident process contexts that still fits the weights plus the reserve.

        Lets a teardown stop only as many sibling processes as the model actually needs gone, rather than
        always collapsing to sole residency. Returns None when it cannot be sized (unknown weights/total, or
        no per-process overhead to reason about), and at least 1 otherwise (the loading process must survive).
        """
        if self.weights_mb is None or self.total_vram_mb is None or self.per_process_overhead_mb <= 0:
            return None
        budget = self.total_vram_mb - self.weights_mb - self.reserve_mb
        if budget <= 0:
            return 1
        return max(1, int(budget // self.per_process_overhead_mb))

    def reason(self) -> str:
        """Return a short human-readable explanation, for logging a residency decision."""
        if self.weights_mb is None:
            return "no weight estimate; treated as co-resident"
        if self.free_now_mb is None:
            return f"no telemetry yet (cold start); weights ~{self.weights_mb:.0f} MB"
        if self.fits_coresident:
            return (
                f"weights ~{self.weights_mb:.0f} MB + {self.reserve_mb:.0f} MB reserve fit "
                f"{self.free_now_mb:.0f} MB free: co-resident"
            )
        after_evict = f"{self.free_after_model_evict_mb:.0f}" if self.free_after_model_evict_mb is not None else "?"
        alone = f"{self.free_if_alone_mb:.0f}" if self.free_if_alone_mb is not None else "?"
        if not self.fits_alone:
            verdict = "weights overflow the card alone: streams unavoidably"
        elif self.requires_sibling_teardown:
            verdict = "weight-dominant: needs idle sibling processes stopped (their contexts over-commit the card)"
        elif self.needs_exclusive_residency:
            verdict = "weight-dominant: needs sole residency (evict sibling models)"
        else:
            verdict = "activation peak only (not weight-dominant): co-resident after evicting a sibling model"
        return (
            f"weights ~{self.weights_mb:.0f} MB + {self.reserve_mb:.0f} MB reserve exceed "
            f"{self.free_now_mb:.0f} MB free (after model evict: {after_evict} MB, alone: {alone} MB): {verdict}"
        )


@dataclass(frozen=True)
class WholeCardResidencyState:
    """Read-only view of the scheduler's whole-card exclusive-residency posture, for the status snapshot.

    Decouples the worker-side runtime read (assembled by :class:`InferenceScheduler`) from the wire/UI
    model, so the serialization layer and the TUI never reach into scheduler internals. ``possible``
    describes whether the feature *could* engage under the current config and process topology (the
    operator heads-up); the remaining fields carry the live detail of a residency that is currently held.
    All MB figures are floats; the snapshot builder rounds them for the wire.
    """

    possible: bool = False
    """Config + topology mean a heavy model could claim the whole card (so a teardown is not a surprise)."""
    enabled: bool = False
    """The ``whole_card_exclusive_residency`` config flag."""
    safety_off_gpu_enabled: bool = False
    """Whether a whole-card job would also move the safety process off-GPU (config + safety on-GPU)."""
    cooldown_seconds: float = 0.0
    """Configured seconds a residency is held after its last heavy job drains."""
    per_process_overhead_mb: float = 0.0
    """Per-process CUDA-context VRAM the forecast assumes (configured override, else measured)."""
    total_vram_mb: float | None = None
    """Device total VRAM (MB), or None before any process has reported."""

    active: bool = False
    """A whole-card residency is currently held."""
    model: str | None = None
    """The model holding sole residency, when active."""
    phase: str = ""
    """``establishing`` | ``holding`` | ``restoring`` while active; empty otherwise."""
    safety_paused: bool = False
    """The safety process is currently paused off-GPU for this residency."""
    processes_now: int = 0
    """Loaded inference processes right now (after any teardown)."""
    processes_target: int = 0
    """Inference processes the residency targets (the forecast's max-resident count)."""
    processes_max: int = 0
    """The normal inference-process ceiling, so the paused count is ``processes_max - processes_now``."""
    cooldown_remaining_seconds: float | None = None
    """Seconds left before the residency restores after its jobs drained, or None when not holding."""

    weights_mb: float | None = None
    """Resident weight footprint of the residency model (the establishing forecast), for the detail view."""
    reserve_mb: float | None = None
    """Free-VRAM headroom the forecast required (activation working set), for the detail view."""
    free_now_mb: float | None = None
    """Measured device-wide free VRAM at establishment, for the detail view."""
    free_if_alone_mb: float | None = None
    """Free VRAM achievable with sole residency, for the detail view."""
    max_resident_processes: int | None = None
    """The forecast's largest co-resident process count that still avoids streaming."""


def predict_job_weight_mb(job: ImageGenerateJobPopResponse, baseline: str | None) -> float | None:
    """Return a job's resident weight footprint (MB), or None when it cannot be estimated.

    Uses hordelib's per-baseline weight seed (:meth:`BaselineBurden.resident_weight_estimate_mb`), which is
    the figure ComfyUI compares against its weight budget when deciding to keep weights resident or stream
    them, distinct from :func:`predict_job_vram_mb`'s activation-inclusive steady estimate. Weights do not
    scale with resolution or batch (activations do), so there is no per-megapixel term here. Imported from
    the torch-free ``feature_impact`` submodule, not the ``hordelib.api`` facade, so the orchestrator stays
    torch-free. Never raises.
    """
    if baseline is None:
        return None
    try:
        from hordelib.feature_impact import get_baseline_burden

        entry = get_baseline_burden(str(baseline))
        if entry is None:
            return None
        return float(entry.resident_weight_estimate_mb())
    except Exception as e:
        logger.debug(f"Job weight estimate failed for {baseline!r}: {type(e).__name__} {e}")
        return None


def effective_inference_reserve_mb(
    total_vram_mb: float | None,
    configured_floor_mb: float,
    *,
    reserve_vram_gb: float | None = None,
) -> float:
    """Return the free-VRAM headroom to preserve to avoid weight streaming.

    The hard minimum is ComfyUI's own inference reserve (``minimum_inference_memory()``), mirrored torch-free
    via :func:`hordelib.vram_planning.compute_inference_reserve_mb` so the worker's forecast lines up with the
    backend's actual streaming threshold; the configured ``vram_reserve_mb`` is honored as an additional floor
    (a safety margin on top). Falls back to the configured floor when total VRAM is unknown (cold start).
    """
    if total_vram_mb is None or total_vram_mb <= 0:
        return float(configured_floor_mb)
    try:
        from hordelib.vram_planning import compute_inference_reserve_mb

        comfy_reserve = compute_inference_reserve_mb(int(total_vram_mb), reserve_vram_gb=reserve_vram_gb)
    except Exception as e:
        logger.debug(f"Inference-reserve lookup failed for {total_vram_mb} MB: {type(e).__name__} {e}")
        return float(configured_floor_mb)
    return float(max(comfy_reserve, configured_floor_mb))


def forecast_weight_streaming(
    job: ImageGenerateJobPopResponse,
    baseline: str | None,
    *,
    free_now_mb: float | None,
    total_vram_mb: float | None,
    per_process_overhead_mb: float,
    num_inference_processes: int,
    configured_reserve_floor_mb: float,
    reserve_vram_gb: float | None = None,
    num_extra_resident_contexts: int = 0,
) -> StreamForecast:
    """Return a :class:`StreamForecast` for loading ``job``'s model given the device's measured state.

    Two derived capacities let the forecast pick the least-disruptive remedy. ``free_if_alone`` is
    ``total - per_process_overhead`` (sole residency: every sibling process stopped, only this process's
    context left). ``free_after_model_evict`` is ``total - num_inference_processes * per_process_overhead``
    (siblings alive but holding no model, so all contexts remain): the best a model can reach without
    stopping a process. Comparing the model against both tells apart one that only streams because of
    co-resident sibling *models* (curable by eviction) from one whose siblings' fixed *contexts* over-commit
    the card (needs the processes stopped) from one that streams even alone (unavoidable). ``num_inference_
    processes`` is the count of loaded inference processes (including the one that will load this model);
    values below 1 are treated as 1.

    ``num_extra_resident_contexts`` is the count of *non-inference* processes that also hold a CUDA context
    on the card (the safety process when ``safety_on_gpu`` is set). Their contexts are real device-wide
    commitments that a stopping of idle *inference* siblings cannot reclaim, so they are subtracted from
    both achievable-free figures; sole residency for a heavy model therefore implies moving them off the
    GPU too. Never raises.
    """
    weights_mb = predict_job_weight_mb(job, baseline)
    base_reserve_mb = effective_inference_reserve_mb(
        total_vram_mb,
        configured_reserve_floor_mb,
        reserve_vram_gb=reserve_vram_gb,
    )
    # The reserve must cover the model's *activation working set*, not a flat constant. ComfyUI's own
    # minimum_inference_memory (and the configured floor) is sized for an SD1.5/SDXL step; a heavy or
    # high-resolution model (Flux at 1024^2) needs several GB of attention activations per step, far above
    # that. Underestimating it is exactly what lets the forecast judge a model "fits" when its sampling step
    # then drives free VRAM to zero and the driver spills activations to host RAM. The activation-inclusive
    # peak (predict_job_vram_mb, which scales with resolution and batch) minus the resident weights is the
    # real per-step headroom to keep free, so fold it into the reserve.
    peak_mb = predict_job_vram_mb(job, baseline)
    activation_working_set_mb = 0.0
    if peak_mb is not None and weights_mb is not None:
        activation_working_set_mb = max(0.0, peak_mb - weights_mb)
    reserve_mb = max(base_reserve_mb, activation_working_set_mb)
    overhead = max(0.0, per_process_overhead_mb)
    process_count = max(1, num_inference_processes)
    extra_contexts = max(0, num_extra_resident_contexts)
    if total_vram_mb is None or total_vram_mb <= 0:
        free_if_alone_mb = None
        free_after_model_evict_mb = None
    else:
        # free_if_alone is the absolute ceiling: the model's own process is the only context on the card,
        # which for whole-card residency means safety is moved off-GPU too. So the extra (safety) context is
        # NOT charged here -- a model only "streams unavoidably" when it overflows even that ceiling.
        free_if_alone_mb = max(0.0, float(total_vram_mb) - overhead)
        # free_after_model_evict is the current reality with every process's context materialised, including
        # the safety-on-GPU context, since stopping idle *inference* siblings cannot reclaim it.
        free_after_model_evict_mb = max(0.0, float(total_vram_mb) - overhead * (process_count + extra_contexts))
    return StreamForecast(
        weights_mb=weights_mb,
        reserve_mb=reserve_mb,
        base_reserve_mb=base_reserve_mb,
        free_now_mb=free_now_mb,
        free_if_alone_mb=free_if_alone_mb,
        free_after_model_evict_mb=free_after_model_evict_mb,
        total_vram_mb=float(total_vram_mb) if total_vram_mb is not None and total_vram_mb > 0 else None,
        per_process_overhead_mb=overhead,
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

    The prediction is the larger of the steady-state sampling burden (``BurdenEstimate.vram_mb``) and the
    baseline's *load peak* (:func:`_baseline_load_peak_mb`). The steady figure reflects resident sampling
    cost, but a combined checkpoint transiently co-resides its text encoder and diffusion weights while
    loading, and it was that load peak (not steady sampling) that drove device free VRAM to near zero in the
    previous versions and triggered the weight-spill thrash. Taking the max stops the budget admitting a
    heavy model into a load the device cannot actually hold.

    Never raises: a missing baseline falls back to hordelib's heavy seed, and any unexpected error
    yields None so the caller treats the cost as unknown (and admits) rather than crashing the
    scheduling cycle.
    """
    burden = _estimate_job_burden(job, baseline)
    steady_mb = None if burden is None else float(burden.vram_mb)
    load_peak_mb = _baseline_load_peak_mb(baseline)
    candidates = [value for value in (steady_mb, load_peak_mb) if value is not None]
    return max(candidates) if candidates else None


def _baseline_load_peak_mb(baseline: str | None) -> float | None:
    """Return hordelib's recommended-VRAM load peak (MB) for ``baseline``, or None when unavailable.

    Sourced from ``hordelib.feature_impact.get_baseline_burden`` (``min_recommended_vram_mb``), the
    recommended free-VRAM headroom for a baseline's transient load peak. Imported from the torch-free
    ``feature_impact`` submodule, NOT the ``hordelib.api`` facade: the facade drags torch (~500MB) into
    whatever imports it, and this runs in the torch-free orchestrator. Never raises: any error yields
    None so the steady estimate stands.
    """
    if baseline is None:
        return None
    try:
        from hordelib.feature_impact import get_baseline_burden

        entry = get_baseline_burden(str(baseline))
        if entry is None:
            return None
        return float(entry.min_recommended_vram_mb)
    except Exception as e:
        logger.debug(f"Baseline load-peak lookup failed for {baseline!r}: {type(e).__name__} {e}")
        return None


def predict_job_ram_mb(job: ImageGenerateJobPopResponse, baseline: str | None) -> float | None:
    """Return a job's predicted system-RAM cost (MB) via hordelib's burden estimate, or None.

    The RAM analogue of :func:`predict_job_vram_mb`; used by the RAM budget to keep resident-in-RAM
    weights from forcing the OS to page. Never raises (see :func:`predict_job_vram_mb`).
    """
    burden = _estimate_job_burden(job, baseline)
    return None if burden is None else float(burden.ram_mb)


def _estimate_job_burden(job: ImageGenerateJobPopResponse, baseline: str | None) -> BurdenEstimate | None:
    """Return hordelib's ``BurdenEstimate`` for a job, or None when the estimate cannot be produced.

    Imported from the torch-free ``hordelib.feature_impact`` submodule (not the ``hordelib.api`` facade,
    which would drag torch into the orchestrator). Never raises: any error is logged at debug and yields
    None so the scheduling cycle is never crashed by a bad estimate.
    """
    try:
        from hordelib.feature_impact import estimate_job_burden

        return estimate_job_burden(
            baseline=baseline if baseline is not None else "",
            width=job.payload.width,
            height=job.payload.height,
            batch=max(1, job.payload.n_iter),
            features=_job_feature_kinds(job),
        )
    except Exception as e:
        logger.debug(f"Job burden estimate failed for job {job.id_}: {type(e).__name__} {e}")
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
    to disk, which collapses throughput.
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
