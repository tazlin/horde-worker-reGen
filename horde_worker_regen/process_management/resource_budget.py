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
    """VRAM (MB) of the *first/sole* process's runtime context: the one-time CUDA runtime/kernel allocation
    plus one context. This is what a single fresh process measures and what survives at sole residency, so it
    sizes ``free_if_alone``. It is NOT the cost of each additional process: on one device the CUDA runtime is
    loaded once and shared, so every sibling beyond the first costs only ``marginal_process_overhead_mb``."""
    marginal_process_overhead_mb: float | None = None
    """VRAM (MB) each *additional* sibling process's context costs once the first process has paid the shared
    one-time CUDA runtime cost (measured ~10x smaller than ``per_process_overhead_mb`` on a 24GB CUDA card).

    Sizes ``free_after_model_evict`` (and the partial-teardown depth) as ``per_process_overhead + (contexts-1)
    * marginal`` rather than ``contexts * per_process_overhead``. The latter multiplies the one-time runtime
    cost by the process count, manufacturing a multi-GB phantom shortfall that flips a co-residable model into
    falsely demanding sibling-process teardown. None (or a non-positive value) falls back to
    ``per_process_overhead_mb`` so a directly-constructed forecast keeps its prior, conservative behavior."""
    wants_whole_card: bool = False
    """The baseline is declared to want sole residency regardless of how its weight estimate happens to fit.

    Some baselines (Cascade/Flux/Qwen/Z-Image: the ``EXTRA_LARGE`` tier) are known to behave badly sharing the
    card even when a conservative weight seed reads as comfortably co-resident, so the scheduler treats them as
    whole-card models on intent rather than waiting for the weight estimate to cross a knife-edge VRAM
    threshold. This only *biases toward* sole residency (see :attr:`needs_exclusive_residency`); it never makes
    an un-loadable model loadable, since the ``fits_alone`` guard still applies."""

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

    @property
    def _effective_marginal_overhead_mb(self) -> float:
        """The per-additional-context VRAM cost, falling back to the (larger) first-context overhead when unset.

        Defaulting to ``per_process_overhead_mb`` reproduces the prior ``contexts * per_process_overhead``
        sizing exactly, so a directly-constructed forecast that does not supply a marginal keeps its old,
        conservative behavior.
        """
        marginal = self.marginal_process_overhead_mb
        if marginal is not None and marginal > 0:
            return marginal
        return self.per_process_overhead_mb

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

        Tearing down sibling *processes* only reclaims their CUDA contexts, so it helps a model whose own
        weights-plus-activations leave no room for another context, not one whose large estimate is a transient
        activation spike. The gate is keyed on the activation-inclusive peak against the ceiling of
        sole-occupancy-plus-one-sibling-context (``total - per_process_overhead - marginal``: the loader's full
        first context plus one additional sibling context): a model that cannot fit even there genuinely needs
        the card. When total VRAM or the per-process overhead is unknown (a directly-constructed forecast) it
        defaults True, preserving the prior, more eager behavior.
        """
        if self.total_vram_mb is None or self.per_process_overhead_mb <= 0:
            return True
        self_plus_one_sibling = (
            self.total_vram_mb - self.per_process_overhead_mb - self._effective_marginal_overhead_mb
        )
        return not self._fits_peak(self_plus_one_sibling)

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
    def fits_weights_now(self) -> bool:
        """True when the persistent weights + bounded floor fit the *measured* free VRAM right now.

        The load-safety gate for a whole-card terminal admit: ``fits_alone`` proves the weights *can* fit at
        sole residency (against the structural ``free_if_alone`` ceiling), but a teardown frees the siblings'
        VRAM asynchronously, so the measurement can still read low for a tick or two after their processes
        stop. This is the same weight-headroom test against the live ``free_now``, so it only reads True once
        the device has actually drained enough that loading the weights will not itself fault. Unknown cost or
        a cold start (no measurement) reads False here, deferring the terminal admit until a real reading.
        """
        return self.known and self._fits_weights(self.free_now_mb)

    @property
    def needs_exclusive_residency(self) -> bool:
        """Streams co-resident, fits alone, and is weight-dominant: the scheduler should give it the whole card.

        The ``_weights_dominant`` gate is what distinguishes a genuine whole-card model (Flux: heavy weights
        leave no room for a sibling context) from a moderate-weight model with a large transient activation
        estimate (SDXL at a big batch/resolution: it can co-reside, so it must reclaim a sibling *model*
        through the normal budget path, not claim the device and tear down sibling *processes*).

        A baseline flagged ``wants_whole_card`` takes the same sole-residency path even when its (conservative)
        weight seed reads as co-resident: the tier classification asserts it never shares well in practice, so
        intent wins over a weight estimate that merely happens to fit. ``fits_alone`` still gates it, so a
        model that genuinely cannot be served alone is never forced down this path.
        """
        if not (self.known and self.fits_alone):
            return False
        if self.wants_whole_card:
            return True
        return not self.fits_coresident and self._weights_dominant

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
    def needs_process_count_reduction(self) -> bool:
        """Return True when the model fits alone but not after sibling models are evicted.

        A moderate-weight model whose bounded weights are squeezed off the card by the *live* sibling
        process contexts, yet which fits once the process count is reduced: the over-commit is cured by
        stopping idle sibling processes, distinct from the sole residency a weight-dominant model needs.

        This catches an over-commit the weight-dominant gate misses: four ~4 GB CUDA contexts on a 24 GB card
        leave under 4 GB free even with every sibling model evicted -- below an SDXL checkpoint's ~4.9 GB of
        weights -- so the model literally cannot load until a sibling *process* stops. ``_weights_dominant``
        (and therefore
        ``needs_exclusive_residency`` / ``requires_sibling_teardown``) miss it: their self-plus-one-sibling
        ceiling judges the moderate weights "not card-filling", so no teardown is triggered and the head is
        deferred until the starvation backstop force-admits it into an OOM.

        Keyed on the *bounded weight* footprint (``_fits_weights``), not the activation-inclusive peak, so a
        transient activation spike whose weights still fit after model eviction is left co-resident rather than
        needlessly cutting concurrency. Topology-aware through ``free_after_model_evict_mb`` (the live-context
        floor). Excludes the weight-dominant sole-residency case (``needs_exclusive_residency``) and
        ``streams_unavoidably`` (no teardown helps, via ``fits_alone``): when True the remedy is reducing the
        process count to ``max_resident_processes``, where the model co-resides.
        """
        return (
            self.known
            and not self._fits_weights(self.free_after_model_evict_mb)
            and self.fits_alone
            and not self.needs_exclusive_residency
        )

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
        always collapsing to sole residency. The loader's own context costs the full ``per_process_overhead``
        (it pays the one-time CUDA runtime cost); each additional co-resident context costs only the
        ``marginal``. Returns None when it cannot be sized (unknown weights/total, or no per-process overhead to
        reason about), and at least 1 otherwise (the loading process must survive).

        A ``wants_whole_card`` baseline collapses straight to 1: the tier declares it never shares the card, so
        the teardown target is sole residency regardless of how many contexts the weight seed would otherwise
        leave room for (a too-low seed must not let a sibling context creep back onto the card).
        """
        if self.wants_whole_card and self.known:
            return 1
        if self.weights_mb is None or self.total_vram_mb is None or self.per_process_overhead_mb <= 0:
            return None
        budget = self.total_vram_mb - self.weights_mb - self.reserve_mb
        if budget <= self.per_process_overhead_mb:
            return 1
        additional = int((budget - self.per_process_overhead_mb) // self._effective_marginal_overhead_mb)
        return max(1, 1 + additional)

    def reason(self) -> str:
        """Return a short human-readable explanation, for logging a residency decision."""
        if self.weights_mb is None:
            return "no weight estimate; treated as co-resident"
        if self.free_now_mb is None:
            return f"no telemetry yet (cold start); weights ~{self.weights_mb:.0f} MB"
        # A whole-card-intent baseline takes sole residency even when its weight seed fits co-resident, so report
        # that intent rather than the misleading "co-resident" the raw fit would otherwise print.
        if self.fits_coresident and not self.needs_exclusive_residency:
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
        elif self.needs_exclusive_residency and self.wants_whole_card and self.fits_coresident:
            verdict = "whole-card baseline: sole residency on intent (evict sibling models)"
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
    post_processing_reserve_mb: float = 0.0,
    marginal_process_overhead_mb: float | None = None,
    wants_whole_card: bool = False,
) -> StreamForecast:
    """Return a :class:`StreamForecast` for loading ``job``'s model given the device's measured state.

    Two derived capacities let the forecast pick the least-disruptive remedy. ``free_if_alone`` is
    ``total - per_process_overhead`` (sole residency: every sibling process stopped, only this process's
    context left). ``free_after_model_evict`` is ``total - per_process_overhead - (contexts - 1) * marginal``
    (siblings alive but holding no model, so all contexts remain): the best a model can reach without
    stopping a process. Comparing the model against both tells apart one that only streams because of
    co-resident sibling *models* (curable by eviction) from one whose siblings' fixed *contexts* over-commit
    the card (needs the processes stopped) from one that streams even alone (unavoidable). ``num_inference_
    processes`` is the count of loaded inference processes (including the one that will load this model);
    values below 1 are treated as 1.

    ``per_process_overhead_mb`` is the cost of the *first/sole* context (the one-time, device-wide CUDA
    runtime/kernel allocation plus one context), the figure a single fresh process measures. Every *additional*
    sibling context costs only ``marginal_process_overhead_mb`` because the runtime is loaded once per device
    and shared. Sizing ``free_after_model_evict`` as ``contexts * per_process_overhead`` (the old behavior,
    preserved when ``marginal`` is None) multiplies that one-time cost by the process count and manufactures a
    multi-GB phantom shortfall, which flips a co-residable model into falsely demanding a sibling-process
    teardown. ``free_if_alone`` keeps the full first-context overhead (the surviving process still pays the
    one-time cost).

    ``num_extra_resident_contexts`` is the count of *non-inference* processes that also hold a CUDA context
    on the card (the safety process when ``safety_on_gpu`` is set). Their contexts are real device-wide
    commitments that a stopping of idle *inference* siblings cannot reclaim, so they are subtracted (at the
    marginal cost, the runtime being already paid) from the achievable-free figure; sole residency for a heavy
    model therefore implies moving them off the GPU too.

    ``wants_whole_card`` flags a baseline the caller has classified as a sole-residency model on intent (the
    ``EXTRA_LARGE`` tier: Cascade/Flux/Qwen/Z-Image), so a conservative weight seed that happens to fit
    co-resident does not stop it claiming the card. It only biases the residency verdict; ``fits_alone`` still
    governs whether sole residency is even achievable. Never raises.
    """
    weights_mb = predict_job_weight_mb(job, baseline)
    base_reserve_mb = effective_inference_reserve_mb(
        total_vram_mb,
        configured_reserve_floor_mb,
        reserve_vram_gb=reserve_vram_gb,
    )
    # The reserve must cover the model's *sampling-phase activation working set*, not a flat constant.
    # ComfyUI's own minimum_inference_memory (and the configured floor) is sized for an SD1.5/SDXL step; a
    # heavy or high-resolution model (Flux at 1024^2) needs several GB of attention activations per sampling
    # step, far above that. Underestimating it is exactly what lets the forecast judge a model "fits" when its
    # sampling step then drives free VRAM to zero and the driver spills activations to host RAM. The
    # sampling-phase peak (predict_job_sampling_vram_mb, which scales with resolution and batch) minus the
    # resident weights is the real per-step headroom to keep free, so fold it into the reserve. The
    # post-processing activation (upscaler/face-fixer) is deliberately excluded here: it runs *after* sampling
    # on the already-resident model, temporally disjoint from weight residency, so charging it against the
    # sampling footprint conflates a transient, output-scaled spike with the persistent weights and can flip a
    # moderate model (an SDXL job that merely requests a 4x upscaler) into falsely reading as weight-dominant
    # and claiming the whole card. The post-processing phase is reserved separately, when imminent, via
    # ``post_processing_reserve_mb`` below.
    peak_mb = predict_job_sampling_vram_mb(job, baseline)
    activation_working_set_mb = 0.0
    if peak_mb is not None and weights_mb is not None:
        activation_working_set_mb = max(0.0, peak_mb - weights_mb)
    # ``post_processing_reserve_mb`` is the imminent post-processing peak of in-flight jobs whose inference
    # slots have already been released (so the measurement still reads high) but whose upscalers/face-fixers
    # are about to allocate. Folding it into the activation-inclusive reserve (never the bounded weight floor
    # ``base_reserve_mb``, since it is a transient activation peak and not this model's persistent weights)
    # makes the co-residency and weight-dominant tests forward-looking: a heavy model will escalate to
    # evicting a sibling model or claiming the card rather than co-residing into VRAM that is about to be
    # reclaimed. Because ``activation_working_set_mb`` is now sampling-only, this term carries the whole
    # post-processing contribution: the loading model's own post-proc is reserved by the scheduler's committed
    # reserve once it reaches that phase, not pre-charged here against its sampling-phase residency.
    reserve_mb = max(base_reserve_mb, activation_working_set_mb) + max(0.0, post_processing_reserve_mb)
    overhead = max(0.0, per_process_overhead_mb)
    # The first context pays the one-time CUDA runtime cost; each additional context costs only the marginal.
    # Default the marginal to the full overhead so an unsupplied marginal reproduces the old contexts*overhead.
    marginal = (
        overhead
        if marginal_process_overhead_mb is None or marginal_process_overhead_mb <= 0
        else float(
            marginal_process_overhead_mb,
        )
    )
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
        # the safety-on-GPU context, since stopping idle *inference* siblings cannot reclaim it. The loading
        # process's context costs the full first-context overhead (it pays the shared one-time runtime cost);
        # every other inference and safety context costs only the marginal.
        additional_contexts = (process_count - 1) + extra_contexts
        free_after_model_evict_mb = max(0.0, float(total_vram_mb) - overhead - marginal * additional_contexts)
    return StreamForecast(
        weights_mb=weights_mb,
        reserve_mb=reserve_mb,
        base_reserve_mb=base_reserve_mb,
        free_now_mb=free_now_mb,
        free_if_alone_mb=free_if_alone_mb,
        free_after_model_evict_mb=free_after_model_evict_mb,
        total_vram_mb=float(total_vram_mb) if total_vram_mb is not None and total_vram_mb > 0 else None,
        per_process_overhead_mb=overhead,
        marginal_process_overhead_mb=marginal,
        wants_whole_card=wants_whole_card,
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


def _job_upscale_factor(job: ImageGenerateJobPopResponse) -> float:
    """Return the largest linear upscale factor the job requests, or 1.0 when it requests none.

    The post-processing activation peak scales with the *upscaled output* megapixels, so the factor (x2 vs
    x4) is the dominant per-model signal for that phase. The factor ROM and resolution live in hordelib
    (the single source of truth for upscaler facts); both it and ``KNOWN_UPSCALERS`` are torch-free and
    imported lazily so the orchestrator never eagerly pulls hordelib. Never raises: any error yields 1.0
    (no enlargement) so the estimate falls back to generation resolution rather than crashing.
    """
    try:
        from horde_sdk.generation_parameters import KNOWN_UPSCALERS
        from hordelib.pipeline.constants import max_upscale_factor

        post_processing = job.payload.post_processing or []
        upscaler_values = {u.value for u in KNOWN_UPSCALERS}
        upscalers = [pp for pp in post_processing if pp in upscaler_values]
        return float(max_upscale_factor(upscalers))
    except Exception as e:
        logger.debug(f"Upscale-factor resolution failed for job {job.id_}: {type(e).__name__} {e}")
        return 1.0


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


def predict_job_sampling_vram_mb(job: ImageGenerateJobPopResponse, baseline: str | None) -> float | None:
    """Return a job's predicted *sampling-phase* peak VRAM (MB), or None when unavailable.

    This is the peak that governs weight *residency* and *preload admission*: the resident weights plus the
    per-step sampling activation that must stay in VRAM together while the model samples, taken as the larger
    of the sampling-phase burden (``BurdenEstimate.vram_sampling_mb``) and the baseline's transient *load
    peak* (a combined checkpoint co-resides its text encoder and diffusion weights while loading).

    It deliberately excludes the post-processing activation (upscaler/face-fixer). That peak runs *after*
    sampling on the already-loaded model (often after the inference slot is released for overlap, sometimes
    after the weights are evicted), so it is temporally disjoint from weight residency. Folding it into a
    residency or preload decision conflates a transient, output-scaled spike with the persistent weight
    footprint and can flip a moderate model (an SDXL job that merely requests a 4x upscaler) into falsely
    claiming the whole card. The post-processing phase is budgeted separately, when it is imminent, via
    :func:`predict_job_post_processing_vram_mb` and the scheduler's committed post-processing reserve.

    Falls back to the combined steady estimate (as :func:`predict_job_vram_mb`) when the pinned hordelib
    predates the phase-split, so an older engine keeps its prior, more conservative behavior. Never raises.
    """
    burden = _estimate_job_burden(job, baseline)
    if burden is None:
        return _baseline_load_peak_mb(baseline)
    try:
        sampling_mb: float | None = float(burden.vram_sampling_mb)
    except AttributeError:
        # Older hordelib without the phase-split: the sampling-only figure is unknown, so fall back to the
        # combined steady estimate (conservative) rather than losing the residency signal entirely.
        sampling_mb = float(burden.vram_mb)
    load_peak_mb = _baseline_load_peak_mb(baseline)
    candidates = [value for value in (sampling_mb, load_peak_mb) if value is not None]
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


def predict_job_post_processing_vram_mb(job: ImageGenerateJobPopResponse, baseline: str | None) -> float | None:
    """Return a job's predicted *post-processing-phase* VRAM peak (MB), or None when unavailable.

    This is the marginal upscaler/face-fixer cost that lands *after* sampling, once the inference slot has
    already been released for the next job. The scheduler reserves it against concurrent dispatch so a
    freshly-released slot is not handed VRAM an in-flight job's upscaler is about to claim. Returns 0.0 for
    a job with no post-processing features (so the reserve self-scales away when nothing is post-processing),
    and None when no estimate is available. Never raises (see :func:`predict_job_vram_mb`).
    """
    burden = _estimate_job_burden(job, baseline)
    if burden is None:
        return None
    try:
        post_processing_mb = burden.vram_post_processing_mb
    except AttributeError:
        # An older pinned hordelib predates the phase-split; the post-proc peak is then unknown and the
        # caller reserves nothing rather than crashing the scheduling cycle.
        return None
    return float(post_processing_mb)


def _estimate_job_burden(job: ImageGenerateJobPopResponse, baseline: str | None) -> BurdenEstimate | None:
    """Return hordelib's ``BurdenEstimate`` for a job, or None when the estimate cannot be produced.

    Imported from the torch-free ``hordelib.feature_impact`` submodule (not the ``hordelib.api`` facade,
    which would drag torch into the orchestrator). Never raises: any error is logged at debug and yields
    None so the scheduling cycle is never crashed by a bad estimate.
    """
    try:
        from hordelib.feature_impact import estimate_job_burden

        common_kwargs = {
            "baseline": baseline if baseline is not None else "",
            "width": job.payload.width,
            "height": job.payload.height,
            "batch": max(1, job.payload.n_iter),
            "features": _job_feature_kinds(job),
        }
        try:
            return estimate_job_burden(**common_kwargs, post_processing_upscale_factor=_job_upscale_factor(job))
        except TypeError:
            # An older pinned hordelib predates the upscale-factor parameter. Fall back to the coarse
            # (generation-resolution) estimate rather than losing the whole budget on a version mismatch.
            return estimate_job_burden(**common_kwargs)
    except Exception as e:
        logger.debug(f"Job burden estimate failed for job {job.id_}: {type(e).__name__} {e}")
        return None


class CommittedReserveLedger:
    """A single accounting of VRAM/RAM committed by in-flight work across every workload flow.

    The measured device-wide free-VRAM (and system available-RAM) figure already reflects every
    *realised* allocation, but it lags work that has been admitted and is about to allocate: a job
    whose inference slot was released for overlap before its upscaler allocates, or an alchemy form
    just dispatched to a child process. Each flow registers that not-yet-realised cost here so every
    admission gate subtracts the same combined figure and two flows cannot independently admit
    against the same free VRAM (the over-commit the image and alchemy gates used to each cause alone).

    Entries are namespaced by ``(flow, unit)`` so a flow can refresh or drop only its own holds. The
    ledger is pure accounting: it does not know why a reserve exists, mirroring how :class:`VramBudget`
    keeps no per-model bookkeeping. All callers run on the single event-loop thread, so no locking.

    The hold is intentionally conservative: a reserve is held until the flow releases it, so while an
    admitted unit's allocation is being realised its cost is briefly counted twice (once in the now-lower
    measured free figure, once here). Erring toward deferral is the safe direction; it prevents the
    over-commit at the cost of occasionally under-using VRAM under heavy concurrent load. Fairness
    (which flow wins a contended lane) is decided above this layer, not by it.
    """

    def __init__(self) -> None:
        """Initialize an empty ledger."""
        self._vram_mb: dict[tuple[str, str], float] = {}
        self._ram_mb: dict[tuple[str, str], float] = {}

    def set(self, flow: str, unit: str, *, vram_mb: float = 0.0, ram_mb: float = 0.0) -> None:
        """Register (or refresh) the committed VRAM/RAM for one unit of work.

        Non-positive figures are stored as zero rather than dropped, so a unit that is still in flight
        but momentarily estimated at zero keeps its slot in the flow namespace.
        """
        self._vram_mb[(flow, unit)] = max(0.0, vram_mb)
        self._ram_mb[(flow, unit)] = max(0.0, ram_mb)

    def release(self, flow: str, unit: str) -> None:
        """Drop the reserve for one unit of work (idempotent)."""
        self._vram_mb.pop((flow, unit), None)
        self._ram_mb.pop((flow, unit), None)

    def replace_flow(
        self,
        flow: str,
        *,
        vram_mb_by_unit: dict[str, float],
        ram_mb_by_unit: dict[str, float] | None = None,
    ) -> None:
        """Atomically replace every entry for ``flow`` with the given per-unit costs.

        Reconciling the whole namespace each cycle (rather than tracking individual add/release events)
        makes the ledger self-healing: a unit whose result message was lost when its process died simply
        stops appearing in the next reconcile and its reserve is dropped, so no stale hold leaks.
        """
        ram_mb_by_unit = ram_mb_by_unit or {}
        self._vram_mb = {k: v for k, v in self._vram_mb.items() if k[0] != flow}
        self._ram_mb = {k: v for k, v in self._ram_mb.items() if k[0] != flow}
        for unit, vram_mb in vram_mb_by_unit.items():
            self._vram_mb[(flow, unit)] = max(0.0, vram_mb)
        for unit, ram_mb in ram_mb_by_unit.items():
            self._ram_mb[(flow, unit)] = max(0.0, ram_mb)

    def total_vram_mb(self) -> float:
        """Return the combined committed VRAM (MB) across all flows."""
        return sum(self._vram_mb.values())

    def total_ram_mb(self) -> float:
        """Return the combined committed RAM (MB) across all flows."""
        return sum(self._ram_mb.values())


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
        committed_reserve_mb: float = 0.0,
    ) -> BudgetVerdict:
        """Return the budget verdict for admitting ``job`` given the measured free VRAM.

        Admits (fits=True) when telemetry is absent (cold start) or no estimate is available, so the
        budget never wedges a worker that has not yet reported VRAM; otherwise requires
        ``free - committed_reserve >= predicted + reserve``.

        ``predicted`` is the *sampling-phase* peak (:func:`predict_job_sampling_vram_mb`), the cost of
        loading the weights and sampling them. A job's post-processing peak (its upscaler/face-fixer) is
        not charged here: it runs after sampling, once the slot is released, and is reserved at that point
        through ``committed_reserve_mb`` instead. Gating the preload on the combined lifetime peak would
        double-charge an output-scaled upscale spike against the load decision and needlessly defer (or
        misroute through the over-budget path) an ordinary job that merely requests an upscaler.

        ``committed_reserve_mb`` is VRAM already spoken for by in-flight jobs whose cost is not yet
        reflected in the measured free figure: chiefly the post-processing peak of a job whose inference
        slot has been released (for overlap) but whose upscaler/face-fixer has not yet allocated. Holding
        it back here is what stops the freed slot from being handed VRAM the in-flight job is about to
        claim. It defaults to 0.0, so callers that do not track it (and the unit tests) keep the prior
        instantaneous behavior.
        """
        if free_vram_mb is None:
            return BudgetVerdict(fits=True, predicted_mb=None, available_mb=None, reserve_mb=self._reserve_mb)

        effective_free_mb = free_vram_mb - committed_reserve_mb
        predicted = predict_job_sampling_vram_mb(job, baseline)
        if predicted is None:
            return BudgetVerdict(
                fits=True,
                predicted_mb=None,
                available_mb=effective_free_mb,
                reserve_mb=self._reserve_mb,
            )

        fits = effective_free_mb >= predicted + self._reserve_mb
        return BudgetVerdict(
            fits=fits,
            predicted_mb=predicted,
            available_mb=effective_free_mb,
            reserve_mb=self._reserve_mb,
        )


class RamBudget:
    """Decides whether measured available system RAM can absorb another job's predicted RAM cost.

    The RAM analogue of :class:`VramBudget`. The worker keeps recently-used model weights resident in
    system RAM for fast reload; with several processes that can exhaust RAM and force the OS to page to
    disk, which collapses throughput.
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
        committed_reserve_mb: float = 0.0,
    ) -> BudgetVerdict:
        """Return the budget verdict for admitting ``job`` given the measured available system RAM.

        Admits (fits=True) when no measurement or estimate is available, so the budget never wedges a
        worker; otherwise requires ``available - committed_reserve >= predicted + reserve``.

        ``committed_reserve_mb`` is RAM already spoken for by in-flight work whose cost is not yet
        reflected in the measured available figure (the RAM analogue of the VRAM committed reserve;
        chiefly other flows' just-admitted weight loads). It defaults to 0.0 so callers that do not
        track it keep the prior instantaneous behavior.
        """
        if available_ram_mb is None:
            return BudgetVerdict(fits=True, predicted_mb=None, available_mb=None, reserve_mb=self._reserve_mb)

        effective_available_mb = available_ram_mb - committed_reserve_mb
        predicted = predict_job_ram_mb(job, baseline)
        if predicted is None:
            return BudgetVerdict(
                fits=True,
                predicted_mb=None,
                available_mb=effective_available_mb,
                reserve_mb=self._reserve_mb,
            )

        fits = effective_available_mb >= predicted + self._reserve_mb
        return BudgetVerdict(
            fits=fits,
            predicted_mb=predicted,
            available_mb=effective_available_mb,
            reserve_mb=self._reserve_mb,
        )
