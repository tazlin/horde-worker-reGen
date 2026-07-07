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

The shape mirrors :class:`~horde_worker_regen.process_management.jobs.alchemy_popper.AlchemyHeadroomEstimator`,
which already gates graph-alchemy forms the same way.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from hordelib.feature_impact import FEATURE_KIND, BurdenEstimate

    from horde_worker_regen.bridge_data.data_model import reGenBridgeData
    from horde_worker_regen.process_management.jobs.job_tracker import JobTracker


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
    device_index: int | None = None,
    now: float | None = None,
) -> bool:
    """Return whether ``model`` is currently held back as locally unservable, per config and fault history.

    The single policy point shared by the scheduler's best-effort-admit gate and the popper's model
    selection, so popping and admitting never disagree on which models are held back. Reads the configured
    breaker thresholds from ``bridge_data`` and the per-model fault streak from ``job_tracker``. Tolerant of
    partially-mocked config: a non-integer threshold (or one ``<= 0``) disables the breaker.

    Args:
        bridge_data: The worker config carrying the breaker threshold and cooldown.
        job_tracker: The job tracker holding the per-(model, card) fault streak.
        model: The model to test, or None (never unservable).
        device_index: The card to test the streak on. With None (single-GPU, or a worker-wide query) the
            worst streak across every card is used, so the single-GPU reading is unchanged.
        now: Optional time override for the cooldown comparison.
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
        overbudget_fault_count=job_tracker.get_model_overbudget_fault_count(model, device_index=device_index),
        last_overbudget_fault_time=job_tracker.model_last_overbudget_fault_time(model, device_index=device_index),
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


_WHOLE_CARD_WARRANT_FRACTION = 0.4
"""Share of total VRAM a model's persistent footprint (weights + bounded floor) must reach before the
disruptive whole-card residency machinery is justified for it. Below this the model regains ample room by
evicting a sibling *model*, so its sibling process *contexts* are never the binding constraint; routing it
onto the whole-card path can only come from an over-counted per-context overhead. Hardware-relative by
design: an SDXL checkpoint (~5GB) is well under this fraction of a 24GB card (co-resides) but over it on a
small card (where it genuinely contends), which is exactly when a teardown is and is not appropriate."""


_SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB = 384.0
"""Conservative per-additional-context marginal VRAM (MB) seeded when no measurement is available.

The device VRAM a torch/CUDA process holds decomposes into four terms that must be accounted separately:

1. device baseline: OS/desktop/other applications' VRAM, shared, attributable to no worker process;
2. per-process marginal fixed overhead: the CUDA context plus import-time allocations (~200-300MB on a
   24GB CUDA card), which persists after a model unloads and is reclaimed only by the process exiting;
3. unloadable model weights; and
4. transient per-job activation peaks.

Only term (2) is what an *additional* sibling context costs. The one-time, device-wide CUDA runtime
allocation is paid once by the first/sole context and shared, so it must never be re-charged per extra
context. When neither the startup probe nor a clean idle-residency reading has measured the marginal,
the forecast seeds it with this constant rather than reusing ``per_process_overhead_mb`` (the
first/sole-context figure, roughly 1300MB, which bundles the device baseline and the one-time runtime
and would therefore double-count the baseline against every extra context, manufacturing a multi-GB
phantom shortfall that wedges high-VRAM workers). The value is set deliberately a little above the
measured range (roughly 200-300MB) so an unmeasured host errs toward reserving slightly more than one
extra context truly needs, never less. ``per_process_overhead_mb`` is still charged exactly once per
device (it sizes ``free_if_alone``); this seed only ever prices the *additional* contexts."""


_WIN32_CONTEXT_CONSTANT_MB = 243.0
"""Fixed CUDA-context VRAM (MB) a torch process holds on Windows/WDDM, excluding its allocator reservation.

Established by a cross-platform probe (std 0, fork vs spawn identical): a child's device footprint is exactly
``context_constant + torch.cuda.memory_reserved()``. Used as the per-process context charge in the committed-
VRAM ledger when no measured marginal is available. Windows reads higher than Linux (WDDM's driver model)."""

_LINUX_CONTEXT_CONSTANT_MB = 144.0
"""Fixed CUDA-context VRAM (MB) a torch process holds on native Linux, excluding its allocator reservation.

The Linux counterpart to :data:`_WIN32_CONTEXT_CONSTANT_MB`, established by the same probe (std 0)."""


def platform_context_constant_mb(
    measured_marginal_mb: float | None = None,
    *,
    platform: str | None = None,
) -> float:
    """Return the per-process CUDA-context VRAM charge (MB) for the committed-VRAM ledger.

    Resolution order: a *measured* per-additional-context marginal (the probe's second-context delta or the
    idle-residency derivation, resolved by :class:`ContextOverheadModel`) wins when available (> 0); otherwise
    the platform-specific probed seed (:data:`_WIN32_CONTEXT_CONSTANT_MB` on Windows,
    :data:`_LINUX_CONTEXT_CONSTANT_MB` on Linux); otherwise the generic
    :data:`_SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB` for an unknown platform. This is the context term the ledger
    adds to each live GPU process's ``process_reserved_mb`` to get its full device footprint.

    Args:
        measured_marginal_mb: A measured per-additional-context marginal (MB), or None/<=0 when unmeasured.
        platform: The platform string to resolve seeds against; defaults to :data:`sys.platform`.
    """
    if measured_marginal_mb is not None and measured_marginal_mb > 0:
        return float(measured_marginal_mb)
    resolved_platform = sys.platform if platform is None else platform
    if resolved_platform == "win32":
        return _WIN32_CONTEXT_CONSTANT_MB
    if resolved_platform.startswith("linux"):
        return _LINUX_CONTEXT_CONSTANT_MB
    return _SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB


_CORESIDENT_SIBLING_MODEL_FLOOR_MB = 5000.0
"""Free VRAM (MB), beyond a model's own weights + bounded floor at sole residency, that must remain for the
card to count as having room for *another full model* to co-reside. Sized to a representative full checkpoint
(an SDXL-class ~5GB model). A whole-card-intent (EXTRA_LARGE) model yields its sole-residency intent only on a
card this roomy: there, it can co-reside with a sibling model and its "never shares well" contract is upheld by
the concurrency overlap gate (no co-*sampling*) rather than by reserving the device. Below this floor (a heavy
model on a tight card, e.g. a ~10GB model on 16GB where ~3GB is left), co-residing a second model would
thrash, so intent still claims the card. This counts room for a *model*, not for an empty CUDA *context* (which
``max_resident_processes`` measures and which can read high even when no real model fits)."""


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
    """Core (diffusion) resident weight footprint ComfyUI compares against its budget, or None when it
    cannot be estimated. Streaming and fits-alone judgments key on this figure: support components can
    time-share a card the full set does not fit, at per-phase swap cost."""
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
    footprint_mb: float | None = None
    """Full per-job resident weight footprint: core weights plus the support components (text encoders,
    VAE) the engine force-loads over each job. Sibling-room and card-dominance judgments key on this
    figure; a multi-component checkpoint judged by its core weights alone reads as co-residable on a
    card where its own components then evict each other all job long. None falls back to
    ``weights_mb`` so a directly-constructed forecast keeps its prior single-figure behavior."""
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
    falsely demanding sibling-process teardown. None (or a non-positive value) falls back to the seeded
    :data:`_SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB` (a conservative bound on one additional context, well below
    the first-context overhead), so an unmeasured host never re-charges the one-time runtime cost per extra
    context."""
    wants_whole_card: bool = False
    """The baseline is declared to want sole residency regardless of how its weight estimate happens to fit.

    Some baselines (Cascade/Flux/Qwen/Z-Image: the ``EXTRA_LARGE`` tier) are known to behave badly sharing the
    card even when a conservative weight seed reads as comfortably co-resident, so the scheduler treats them as
    whole-card models on intent rather than waiting for the weight estimate to cross a knife-edge VRAM
    threshold. This only *biases toward* sole residency (see :attr:`needs_exclusive_residency`); it never makes
    an un-loadable model loadable, since the ``fits_alone`` guard still applies."""

    base_reserve_mb: float | None = None
    """The weight-fit floor: ComfyUI's own inference-reserve streaming threshold (``minimum_inference_memory``).

    Sizes the decisions about the persistent *weight* footprint (``fits_alone``, ``streams_unavoidably``,
    and the weight-headroom gate), as distinct from the activation-inclusive ``reserve_mb`` that sizes the
    co-resident streaming check and the teardown *depth*. This is deliberately the ComfyUI threshold alone,
    *not* the operator's configured ``vram_reserve_mb``: that configured figure is a co-residency/sampling
    safety margin (folded into ``reserve_mb``), and enforcing it as a load-feasibility floor makes a model
    whose weights fit the drained card (Flux ~11.5 GB on 16 GB) read as streaming-unavoidable so its
    whole-card dispatch gate never converges. Folding the conservative, batch-and-resolution scaled activation
    peak into every fit decision conflates a transient activation spike with the persistent footprint: it can
    flip a moderate-weight model into claiming the whole card, or push a model whose weights fit alone
    marginally past free-if-alone so it falsely reads as streaming-unavoidable. Keeping the weight decisions on
    this bounded threshold holds them independent of both the activation estimate and the operator margin. None
    falls back to ``reserve_mb`` so a directly-constructed forecast keeps its prior single-reserve behavior."""

    @property
    def known(self) -> bool:
        """Whether enough is known (weight estimate and a current measurement) to forecast at all."""
        return self.weights_mb is not None and self.free_now_mb is not None

    @property
    def _effective_base_reserve(self) -> float:
        """The bounded weight-footprint reserve, falling back to ``reserve_mb`` when unset."""
        return self.base_reserve_mb if self.base_reserve_mb is not None else self.reserve_mb

    @property
    def _effective_footprint_mb(self) -> float | None:
        """The full resident footprint for room/dominance judgments, falling back to the core weights."""
        return self.footprint_mb if self.footprint_mb is not None else self.weights_mb

    @property
    def _effective_marginal_overhead_mb(self) -> float:
        """The per-additional-context VRAM cost, seeding a conservative constant when none was measured.

        A measured marginal (the probe's second-context delta or the idle-floor derivation, resolved
        upstream and passed in as ``marginal_process_overhead_mb``) wins. When none was measured the
        fallback is the seeded :data:`_SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB`, NOT ``per_process_overhead_mb``:
        the first/sole-context figure bundles the device baseline and the one-time CUDA runtime, so charging
        it per additional context double-counts the baseline and over-states the contexts' combined cost.
        ``per_process_overhead_mb`` is charged once per device (it sizes ``free_if_alone``); every additional
        context is priced by this marginal.
        """
        marginal = self.marginal_process_overhead_mb
        if marginal is not None and marginal > 0:
            return marginal
        return _SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB

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
    def _persistent_weights_dominant(self) -> bool:
        """Whether the *persistent* weights leave no room for even one sibling context (a sole-residency model).

        The persistent-footprint counterpart to :attr:`_weights_dominant`: it decides whether a (non-whole-card-
        intent) model needs *sole* residency, so stopping siblings down to one process is the only fit. Keyed on
        ``_fits_weights`` (weights + bounded floor) against the same sole-occupancy-plus-one-sibling-context
        ceiling, so a transient activation spike never forces a process teardown: a moderate-weight model whose
        weights *do* fit beside a context is instead left to co-reside (evicting a sibling model) or, if the live
        contexts over-commit its weights, to a *partial* reduction via :attr:`needs_process_count_reduction`
        (which keeps the surviving contexts rather than collapsing to one). Only a model whose persistent weights
        genuinely fill the card reads dominant here. Defaults True when the card cannot be sized, matching
        :attr:`_weights_dominant`.
        """
        if self.total_vram_mb is None or self.per_process_overhead_mb <= 0:
            return True
        self_plus_one_sibling = (
            self.total_vram_mb - self.per_process_overhead_mb - self._effective_marginal_overhead_mb
        )
        return not self._fits_weights(self_plus_one_sibling)

    @property
    def _has_room_for_coresident_model(self) -> bool:
        """Whether sole-residency free VRAM holds another full model beside these weights (a genuinely roomy card).

        Distinguishes "many empty contexts fit" (what :meth:`max_resident_processes` counts, a context is cheap)
        from "a real sibling *model* fits", which is what decides whether a whole-card-intent model can co-reside
        without thrashing. Measured as the free VRAM left at sole residency after this model's weights and bounded
        floor, against :data:`_CORESIDENT_SIBLING_MODEL_FLOOR_MB`. False when unsized, so intent is preserved on a
        card whose room cannot be established (the conservative direction).
        """
        footprint_mb = self._effective_footprint_mb
        if self.free_if_alone_mb is None or footprint_mb is None:
            return False
        return (
            self.free_if_alone_mb - footprint_mb - self._effective_base_reserve
        ) >= _CORESIDENT_SIBLING_MODEL_FLOOR_MB

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
        """Streams co-resident, fits alone, and cannot usefully share the card: give it the whole device.

        The grant rests on two different "room" questions, one per branch, neither keyed on the transient
        activation peak:

        - **Whole-card-intent models** (``wants_whole_card``, the EXTRA_LARGE tier) prefer the card to
          themselves, so they reserve it *unless* there is room for another full model to co-reside
          (:attr:`_has_room_for_coresident_model`). On a genuinely roomy card (Flux fp8 on 24 GB) they co-reside
          like any other model; the "never shares well" contract is then upheld by the concurrency overlap gate
          (no co-*sampling*), not by reserving the device, which where the budget shows ample room only churns it.
        - **Ordinary models** co-reside by default and need *sole* residency only when their persistent weights
          are too heavy to fit beside even one sibling context (:attr:`_persistent_weights_dominant`). A
          moderate-weight model with a large transient activation estimate (a 4.9 GB SDXL at a big batch) never
          trips this: its weights leave ample room, so the spike is absorbed by evicting a sibling *model* and
          sampling under the over-budget step grace. If the live contexts over-commit the weights, the *partial*
          :attr:`needs_process_count_reduction` (which keeps the surviving contexts rather than collapsing to one)
          covers that instead.

        ``fits_alone`` still gates the whole property, so a model that cannot be served alone is never forced down
        this path.
        """
        if not (self.known and self.fits_alone):
            return False
        if self.wants_whole_card and not self._has_room_for_coresident_model:
            return True
        return not self.fits_coresident and self._persistent_weights_dominant

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
        leave under 4 GB free even with every sibling model evicted, below an SDXL checkpoint's ~4.9 GB of
        weights, so the model literally cannot load until a sibling *process* stops. ``_weights_dominant``
        (and therefore
        ``needs_exclusive_residency`` / ``requires_sibling_teardown``) miss it: their self-plus-one-sibling
        ceiling judges the moderate weights "not card-filling", so no teardown is triggered and the head is
        deferred until the old starvation backstop admitted it into an OOM.

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
    def is_card_demanding(self) -> bool:
        """Whether the model's persistent footprint is a large enough share of the card to justify reserving it.

        The whole-card residency machinery (reserve the device, move safety off-GPU, stop sibling contexts,
        hold through a cooldown) is disruptive and only pays off for a model whose weights genuinely dominate
        the card. A model whose weights-plus-bounded-floor occupy only a small fraction of total VRAM always
        regains ample room by evicting a sibling *model*, so its sibling *contexts* are never the binding
        constraint; a teardown demand for it can only come from an over-counted per-context overhead. A
        ``wants_whole_card`` baseline short-circuits True (the tier asserts it never shares well). This is the
        teardown *warrant* (is a teardown demand trustworthy), kept deliberately conservative: it only matters
        once a teardown is actually demanded (``needs_exclusive_residency`` / the context-reduction path), which a
        roomy EXTRA_LARGE model never reaches because it co-resides, so leaving this eager never over-grants, it
        only avoids declining a genuine context-reduction. Conservatively True when the footprint cannot be sized,
        preserving the prior, more eager behavior. See :data:`_WHOLE_CARD_WARRANT_FRACTION`; the share is taken
        against total VRAM so the verdict is hardware-relative.
        """
        if self.wants_whole_card:
            return True
        footprint_mb = self._effective_footprint_mb
        if footprint_mb is None or self.total_vram_mb is None or self.total_vram_mb <= 0:
            return True
        return (footprint_mb + self._effective_base_reserve) >= self.total_vram_mb * _WHOLE_CARD_WARRANT_FRACTION

    @property
    def admit_requires_isolation(self) -> bool:
        """Whether an over-budget classified admit of this model must run with the device to itself.

        Isolation protects a heavy checkpoint from a concurrent sibling load pushing its weights into
        host-RAM streaming; that hazard needs both a footprint that dominates the card
        (:attr:`is_card_demanding`) *and* a card too small to host a sibling model beside it. A
        card-dominating model on a genuinely roomy card (:attr:`_has_room_for_coresident_model`)
        co-resides safely: its no-co-*sampling* contract is upheld by the concurrency overlap gate, so
        reserving the device for its admit only freezes the sibling lane through the model's multi-GB
        load. Both judgments charge the full resident footprint (core weights plus force-loaded support
        components); a Flux fp8 whose trio occupies ~16 GB has no sibling room on a 24 GB card even
        though its core weights alone would suggest otherwise. Unsized forecasts read as isolating
        (room reads False, dominance reads True), keeping the conservative direction wherever the card
        cannot be measured.
        """
        if self._has_room_for_coresident_model:
            return False
        return self.is_card_demanding

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

        A ``wants_whole_card`` baseline is sized by this same budget arithmetic rather than collapsing straight
        to 1. Its intent is that it never *co-samples* (enforced separately by the concurrency overlap gate),
        not that every sibling *context* must be torn down: a context that holds no model and is not sampling
        costs only its (cheap) per-context VRAM, so on a card whose VRAM genuinely holds the weights-plus-reserve
        alongside one or more idle sibling contexts (a Flux fp8 checkpoint (~11.5GB) on a 24GB card), keeping
        those contexts avoids a full teardown-and-respawn every time the heavy head cycles (the residency churn
        that capped throughput). On a card with no such room (the same fp8 weights on a 16GB card) the budget
        returns 1, so the behavior stays hardware-relative. Only when the footprint cannot be sized at all does a
        whole-card-intent model still collapse to 1 (conservative); an ordinary model is then unsizable (None).
        """
        footprint_mb = self._effective_footprint_mb
        if footprint_mb is None or self.total_vram_mb is None or self.per_process_overhead_mb <= 0:
            return 1 if (self.wants_whole_card and self.known) else None
        # Sized on the full footprint: while the job runs, its support components share the card with
        # the core weights, so contexts kept beyond that set must fit beside all of it.
        budget = self.total_vram_mb - footprint_mb - self.reserve_mb
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
    """Return a job's core (diffusion) resident weight footprint (MB), or None when unestimable.

    Uses hordelib's per-baseline weight seed (:meth:`BaselineBurden.resident_weight_estimate_mb`), which is
    the figure ComfyUI compares against its weight budget when deciding to keep weights resident or stream
    them, distinct from :func:`predict_job_vram_mb`'s activation-inclusive steady estimate. Weights do not
    scale with resolution or batch (activations do), so there is no per-megapixel term here. This is the
    *core* figure: support components (text encoders, VAE) time-share a constrained card via per-phase
    swaps, so streaming and fits-alone judgments key on the core weights; sibling-room judgments use
    :func:`predict_job_footprint_mb` instead. Imported from the torch-free ``feature_impact`` submodule,
    not the ``hordelib.api`` facade, so the orchestrator stays torch-free. Never raises.
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


def predict_job_footprint_mb(job: ImageGenerateJobPopResponse, baseline: str | None) -> float | None:
    """Return a job's full resident weight footprint (MB): core weights plus support components.

    Uses hordelib's :meth:`BaselineBurden.resident_footprint_estimate_mb`: the core diffusion weights
    plus the text encoders and VAE the engine force-loads onto the device over the course of every job.
    Sibling-room and isolation verdicts must charge this whole set; a multi-component checkpoint judged
    by its core weights alone is under-counted by the full text-encoder size, and room granted against
    that smaller figure is room that does not exist (the components then evict each other all job long).
    Streaming and fits-alone judgments keep using :func:`predict_job_weight_mb`: the components can
    time-share a card the full set does not fit, at per-phase swap cost. Never raises.
    """
    if baseline is None:
        return None
    try:
        from hordelib.feature_impact import get_baseline_burden

        entry = get_baseline_burden(str(baseline))
        if entry is None:
            return None
        return float(entry.resident_footprint_estimate_mb())
    except Exception as e:
        logger.debug(f"Job footprint estimate failed for {baseline!r}: {type(e).__name__} {e}")
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
        return configured_floor_mb
    try:
        from hordelib.vram_planning import compute_inference_reserve_mb

        comfy_reserve = compute_inference_reserve_mb(int(total_vram_mb), reserve_vram_gb=reserve_vram_gb)
    except Exception as e:
        logger.debug(f"Inference-reserve lookup failed for {total_vram_mb} MB: {type(e).__name__} {e}")
        return configured_floor_mb
    return max(comfy_reserve, configured_floor_mb)


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
    committed_reserve_mb: float = 0.0,
    marginal_process_overhead_mb: float | None = None,
    wants_whole_card: bool = False,
    disaggregated: bool = False,
    disaggregation_sibling_charge_mb: float = 0.0,
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
    and shared. Sizing ``free_after_model_evict`` as ``contexts * per_process_overhead`` multiplies that
    one-time cost by the process count and manufactures a multi-GB phantom shortfall, which flips a
    co-residable model into falsely demanding a sibling-process teardown. When the marginal was not measured,
    the seeded :data:`_SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB` prices the additional contexts (a conservative
    bound well below the first-context cost), never the full overhead. ``free_if_alone`` keeps the full
    first-context overhead (the surviving process still pays the one-time cost).

    ``num_extra_resident_contexts`` is the count of *non-inference* processes that also hold a CUDA context
    on the card (the safety process when ``safety_on_gpu`` is set). Their contexts are real device-wide
    commitments that a stopping of idle *inference* siblings cannot reclaim, so they are subtracted (at the
    marginal cost, the runtime being already paid) from the achievable-free figure; sole residency for a heavy
    model therefore implies moving them off the GPU too.

    ``wants_whole_card`` flags a baseline the caller has classified as a sole-residency model on intent (the
    ``EXTRA_LARGE`` tier: Cascade/Flux/Qwen/Z-Image), so a conservative weight seed that happens to fit
    co-resident does not stop it claiming the card. It only biases the residency verdict; ``fits_alone`` still
    governs whether sole residency is even achievable. Never raises.

    ``disaggregated`` marks a job that runs on the pipeline-disaggregation path: its sampler process holds
    only the UNet (core diffusion weights plus sampling activation), not the support weights or the VAE
    decode spike the whole-job estimate bakes in. Both the persistent-footprint and activation-inclusive
    charges then key on :func:`predict_job_sampler_only_vram_mb` (the ~6.6GB SDXL sampler figure that keeps
    two samplers co-resident on a 16GB card, where the ~16GB whole-job charge collapses them to one).
    ``disaggregation_sibling_charge_mb`` is the image lane's concurrent VAE-decode spike (the caller passes
    :func:`effective_post_process_vram_quota_mb`), charged against the siblings-present achievable-free
    figure so a second sampler is not admitted into VRAM the lane's decode is about to claim.
    """
    if disaggregated:
        # The sampler holds only the UNet: core weights plus sampling activation, with the text encoders,
        # VAE, and decode spike running in the encode service and image lane. That combined sampler figure
        # is the persistent footprint AND the peak here, so it drives both the weight-fit and co-resident
        # tests without the whole-job support/decode weight that would collapse two samplers into one.
        sampler_charge_mb = predict_job_sampler_only_vram_mb(job, baseline)
        weights_mb = sampler_charge_mb
        footprint_mb = sampler_charge_mb
    else:
        weights_mb = predict_job_weight_mb(job, baseline)
        footprint_mb = predict_job_footprint_mb(job, baseline)
    # The weight-fit floor is ComfyUI's *own* streaming threshold (``minimum_inference_memory``), NOT the
    # operator's configured ``vram_reserve_mb``. That configured figure is a sampling / co-residency safety
    # margin (how much headroom to keep free while a model samples beside siblings), not a statement about
    # whether a model can physically load at all. Folding it into the weight-fit floor conflates the two: on
    # a 16 GB card a 4096 MB reserve makes an 11.5 GB Flux checkpoint read as needing 15.6 GB free to "fit
    # alone" when a fully drained card only reaches ~15.1 GB, so its whole-card dispatch gate can never
    # converge and the head wedges. Load feasibility (``fits_alone`` / ``fits_weights_now`` /
    # ``streams_unavoidably``) therefore keys on the streaming threshold alone; the configured margin is
    # applied to the activation-inclusive ``reserve_mb`` below, where it governs co-residency and teardown
    # depth (its proper role) without ever making a model that fits the drained card unservable.
    base_reserve_mb = effective_inference_reserve_mb(
        total_vram_mb,
        0.0,
        reserve_vram_gb=reserve_vram_gb,
    )
    configured_floor_mb = max(0.0, configured_reserve_floor_mb)
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
    # and claiming the whole card. Post-processing runs on the dedicated lane, whose resident context is
    # charged via ``num_extra_resident_contexts``.
    # A disaggregated sampler's activation is already folded into its sampler-only charge (the peak is that
    # charge), so the whole-job sampling peak is not read for it: doing so would re-add the support/decode
    # activation the sampler process never holds.
    peak_mb = weights_mb if disaggregated else predict_job_sampling_vram_mb(job, baseline)
    activation_working_set_mb = 0.0
    if peak_mb is not None and weights_mb is not None:
        activation_working_set_mb = max(0.0, peak_mb - weights_mb)
    # ``committed_reserve_mb`` is the VRAM in-flight concurrent work (alchemy forms today) has committed
    # but the measurement may not yet reflect. Folding it into the activation-inclusive reserve (never the
    # bounded weight floor ``base_reserve_mb``, since it is transient and not this model's persistent
    # weights) keeps the co-residency and weight-dominant tests forward-looking: a heavy model escalates to
    # evicting a sibling model or claiming the card rather than co-residing into VRAM that is about to be
    # reclaimed. The configured operator margin joins the activation working set here (not the weight floor):
    # it is the co-residency headroom the operator wants preserved while a model samples beside siblings.
    reserve_mb = max(base_reserve_mb, activation_working_set_mb, configured_floor_mb) + max(0.0, committed_reserve_mb)
    overhead = max(0.0, per_process_overhead_mb)
    # The first context pays the one-time, device-wide CUDA runtime cost; each additional context costs only
    # the marginal. When the marginal was not measured (probe or idle-floor both absent upstream), seed it
    # with the conservative per-additional-context constant rather than the full first-context overhead: the
    # latter re-charges the one-time runtime and the device baseline against every extra context, a phantom
    # multi-GB shortfall. ``overhead`` is still charged once (it sizes free_if_alone below).
    marginal = (
        _SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB
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
        # NOT charged here; a model only "streams unavoidably" when it overflows even that ceiling.
        free_if_alone_mb = max(0.0, float(total_vram_mb) - overhead)
        # free_after_model_evict is the current reality with every process's context materialised, including
        # the safety-on-GPU context, since stopping idle *inference* siblings cannot reclaim it. The loading
        # process's context costs the full first-context overhead (it pays the shared one-time runtime cost);
        # every other inference and safety context costs only the marginal.
        additional_contexts = (process_count - 1) + extra_contexts
        # Under disaggregation the image lane VAE-decodes the previous job while this one samples, so its
        # decode spike is a real, concurrent device commitment idle inference siblings cannot reclaim. Charge
        # it here (against the siblings-present figure the co-resident test reads) so a second sampler is not
        # admitted into VRAM the lane's decode is about to take. Left at 0 off the disaggregation path.
        lane_spike_mb = max(0.0, disaggregation_sibling_charge_mb) if disaggregated else 0.0
        free_after_model_evict_mb = max(
            0.0,
            float(total_vram_mb) - overhead - marginal * additional_contexts - lane_spike_mb,
        )
    return StreamForecast(
        weights_mb=weights_mb,
        footprint_mb=footprint_mb,
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


def predict_job_sampler_only_vram_mb(job: ImageGenerateJobPopResponse, baseline: str | None) -> float | None:
    """Return a job's predicted VRAM (MB) for a UNet-only (disaggregated) sampler process, or None.

    Under pipeline disaggregation the text encoders and VAE run in other processes, so a sampler holds
    only the core diffusion weights plus sampling activation, not the support weights or the decode spike
    baked into the whole-job ``vram_sampling_mb``. This charges ``BurdenEstimate.vram_sampler_only_mb``,
    which is what keeps two disaggregated samplers co-resident where the whole-job charge collapses them
    (measured: an SDXL sampler pins ~6.6GB, two fit a 16GB card; a whole SDXL job pins ~16GB, two do not).

    Falls back to the whole-job sampling figure when the pinned hordelib predates the split (never below
    the full charge), so an older engine keeps its conservative behavior. Never raises.
    """
    burden = _estimate_job_burden(job, baseline)
    if burden is None:
        return _baseline_load_peak_mb(baseline)
    sampler_only_mb = getattr(burden, "vram_sampler_only_mb", 0) or 0
    if sampler_only_mb <= 0:
        # Older hordelib without the disaggregation split: fall back to the whole-job sampling charge.
        return predict_job_sampling_vram_mb(job, baseline)
    return float(sampler_only_mb)


def predict_job_decode_spike_mb(job: ImageGenerateJobPopResponse, baseline: str | None) -> float | None:
    """Return the disaggregated image lane's VAE tiled-decode activation spike (MB), or None when unavailable.

    Under pipeline disaggregation the image lane VAE-decodes the previous job's latent while a sampler runs,
    so the coresidency verdict must reserve that *bounded* concurrent decode activation, not the lane's whole
    allocator-guard quota. Charging the full quota (up to ~8GB on a 16GB card) against coresidency denies a
    second sampler the card physically holds: two SDXL samplers (~6.2GB each) plus a ~2.5GB decode spike were
    measured co-resident at ~14.9GB, whereas two samplers plus the ~8GB quota over-commit the card and
    collapse the pipeline back to one sampler, which is exactly the collapse disaggregation exists to prevent.

    Sourced from ``BurdenEstimate.vram_decode_spike_mb`` via the same access idiom as
    :func:`predict_job_sampler_only_vram_mb` reads ``vram_sampler_only_mb`` (``getattr`` with a default, so a
    pinned hordelib that predates the field stays statically analyzable and does not fault). Returns None when
    the field is absent or non-positive, so the caller falls back to the conservative full-quota lane charge:
    an older engine is then safe, just not optimally packed. Never raises.
    """
    burden = _estimate_job_burden(job, baseline)
    if burden is None:
        return None
    decode_spike_mb = getattr(burden, "vram_decode_spike_mb", None)
    if decode_spike_mb is None or decode_spike_mb <= 0:
        return None
    return float(decode_spike_mb)


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


@dataclass
class _PlannedReserve:
    """A planned (admitted, not-yet-materialised) VRAM charge that decays as its target's reservation fills.

    Held separately from the ledger's flat committed entries because it carries the extra state the
    double-count guard needs: which process the charge will land on, that process's measured allocator
    reservation at admit time, and a high-water mark of the growth observed since, so the outstanding planned
    share can be computed against the target's current reservation (see
    :meth:`CommittedReserveLedger.effective_planned_vram_mb`).

    Consumption is monotonic: the outstanding share is measured against the greatest reserved-growth ever
    seen for this entry, never the instantaneous growth. Once the load has materialised, a later collapse of
    the target's reservation (an eviction that frees the VRAM back to the card) cannot resurrect the charge,
    because a materialised anchor's job is already done.
    """

    vram_mb: float
    """The planned VRAM charge (MB) at admit time."""
    target_process_id: int
    """The process the charge will materialise on."""
    reserved_at_admit_mb: float
    """The target's measured allocator reservation (MB) when the charge was admitted."""
    materialized_watermark_mb: float = 0.0
    """The greatest reserved-growth (MB) past admit ever observed for this entry; ratchets up, never down."""


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
        self._planned: dict[tuple[str, str], _PlannedReserve] = {}

    def set(self, flow: str, unit: str, *, vram_mb: float = 0.0, ram_mb: float = 0.0) -> None:
        """Register (or refresh) the committed VRAM/RAM for one unit of work.

        Non-positive figures are stored as zero rather than dropped, so a unit that is still in flight
        but momentarily estimated at zero keeps its slot in the flow namespace.
        """
        self._vram_mb[(flow, unit)] = max(0.0, vram_mb)
        self._ram_mb[(flow, unit)] = max(0.0, ram_mb)

    def set_planned(
        self,
        flow: str,
        unit: str,
        *,
        vram_mb: float,
        target_process_id: int,
        reserved_at_admit_mb: float,
    ) -> None:
        """Register a *planned* (admitted but not-yet-materialised) VRAM charge for the ledger-driven identity.

        Distinct from :meth:`set`: a planned charge is a load the admission identity has admitted whose
        allocation the measured committed floor does not yet reflect, and it *decays* as the target process's
        own measured allocator reservation grows past what it held when the charge was admitted (see
        :meth:`effective_planned_vram_mb`). This decay is the double-count guard that lets the measured floor
        and the planned overlay be summed without charging the same load twice: while the load materialises it
        is briefly counted once here and once in the (now higher) measured reservation, and the planned share
        shrinks to zero exactly as the measured share fills in. Consumption is monotonic, so once the load has
        materialised the charge stays consumed even if the target's reservation is later evicted back down;
        re-registering the same unit (a genuinely new admission) resets the watermark and charges in full again.

        Args:
            flow: The workload flow namespace (so a flow can refresh or drop only its own planned holds).
            unit: The unit-of-work key within the flow.
            vram_mb: The planned charge (MB); stored as zero when non-positive.
            target_process_id: The process the charge will materialise on, whose measured reservation decays it.
            reserved_at_admit_mb: The target's measured allocator reservation (MB) at admit time; growth beyond
                this figure is treated as the planned charge materialising.
        """
        self._planned[(flow, unit)] = _PlannedReserve(
            vram_mb=max(0.0, vram_mb),
            target_process_id=target_process_id,
            reserved_at_admit_mb=max(0.0, reserved_at_admit_mb),
        )

    def effective_planned_vram_mb(self, process_reserved_by_pid: dict[int, float]) -> float:
        """Return the combined planned VRAM (MB) still outstanding, each entry decayed by what has materialised.

        For each planned entry the materialised amount is the high-water mark of
        ``max(0, current_reserved - reserved_at_admit)`` seen for its target process, and the entry's
        outstanding charge is ``max(0, planned - materialised)``. Summing these gives the planned overlay the
        admission identity adds on top of the measured committed floor without double-counting a load that is
        partway into VRAM.

        Consumption is monotonic: each call ratchets the entry's watermark up by any newly-observed growth and
        never lowers it. This is what stops a materialised-then-evicted anchor from resurrecting: once the
        target's reservation has grown to cover the charge, a later collapse of that reservation (an eviction
        that returns the VRAM to the card) leaves the watermark high, so the charge stays consumed. A load that
        is genuinely still in flight (its reservation has not yet grown) keeps its full charge, preserving the
        same-cycle double-admit guard the overlay exists for.

        Args:
            process_reserved_by_pid: Current measured allocator reservation (MB) keyed by process id; a target
                absent from the map is treated as holding zero (nothing has materialised for it yet).
        """
        total = 0.0
        for entry in self._planned.values():
            current_reserved = process_reserved_by_pid.get(entry.target_process_id, 0.0)
            materialised = max(0.0, current_reserved - entry.reserved_at_admit_mb)
            if materialised > entry.materialized_watermark_mb:
                entry.materialized_watermark_mb = materialised
            total += max(0.0, entry.vram_mb - entry.materialized_watermark_mb)
        return total

    def planned_charge_for_unit(self, flow: str, unit: str, process_reserved_by_pid: dict[int, float]) -> float:
        """Return one planned unit's still-outstanding charge (MB), decayed by what has materialised, or zero.

        The single-entry counterpart of :meth:`effective_planned_vram_mb`, used to net a request's own planned
        charge out of the admission identity so a re-ask cannot double-count itself: the caller subtracts this
        from the overlay for the request that targets ``unit``. Read-only, so it never advances the entry's
        materialisation watermark (the per-cycle :meth:`effective_planned_vram_mb` owns that ratchet); it reports
        the outstanding charge against the greater of the recorded watermark and the target's current growth.

        Args:
            flow: The workload flow namespace the unit lives under.
            unit: The unit-of-work key within the flow.
            process_reserved_by_pid: Current measured allocator reservation (MB) keyed by process id; a target
                absent from the map is treated as holding zero.
        """
        entry = self._planned.get((flow, unit))
        if entry is None:
            return 0.0
        current_reserved = process_reserved_by_pid.get(entry.target_process_id, 0.0)
        materialised = max(entry.materialized_watermark_mb, max(0.0, current_reserved - entry.reserved_at_admit_mb))
        return max(0.0, entry.vram_mb - materialised)

    def reconcile_planned(self, flow: str, live_units: Iterable[str]) -> None:
        """Prune ``flow``'s planned charges to only the units whose admission is still in flight (self-healing).

        The planned counterpart of :meth:`replace_flow`: each scheduling cycle the caller derives, from live
        process/model state, the set of units whose admitted load has not yet materialised, and passes it here.
        Any planned entry for ``flow`` whose unit is absent from that set is dropped, so a finished, faulted, or
        dead admission releases its charge purely by omission with no explicit release call (the same reason the
        flat reserve is reconciled rather than event-tracked: a lost result message cannot leak a stale hold).

        Surviving entries are kept *as-is*, preserving each one's admit-time reservation baseline so its
        per-target decay (see :meth:`effective_planned_vram_mb`) is never reset. Entries under other flows are
        untouched. The rebuild is idempotent: re-running it with the same live set is a no-op, so it is safe to
        drive from any per-cycle choke point that assembles the measured state (it may run several times a cycle).
        Creating an entry is the grant path's job (:meth:`set_planned`); this method only ever prunes, so a unit
        that appears live but was never granted simply carries no planned charge (a conservative under-count).
        """
        live = set(live_units)
        self._planned = {key: entry for key, entry in self._planned.items() if key[0] != flow or key[1] in live}

    def release(self, flow: str, unit: str) -> None:
        """Drop the reserve for one unit of work (idempotent), including any planned charge for it."""
        self._vram_mb.pop((flow, unit), None)
        self._ram_mb.pop((flow, unit), None)
        self._planned.pop((flow, unit), None)

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

    def total_vram_mb_excluding(self, flow: str) -> float:
        """Return the combined committed VRAM (MB) across every flow except ``flow``.

        Lets a per-card VRAM gate substitute that card's own share of a card-attributed flow (image
        post-processing) for the worker-wide aggregate, while still charging the flows that are not
        card-attributed (alchemy) against the card.
        """
        return sum(mb for (registered_flow, _unit), mb in self._vram_mb.items() if registered_flow != flow)

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
        *,
        disaggregated: bool = False,
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

        ``disaggregated`` marks a job that runs on the pipeline-disaggregation path: its sampler process
        holds only the UNet, so the charge is :func:`predict_job_sampler_only_vram_mb` (the ~6.6GB SDXL
        sampler figure) rather than the ~16GB whole-job sampling peak. This is what admits a second sampler
        beside a first on a card the whole-job charge would collapse to one.
        """
        if free_vram_mb is None:
            return BudgetVerdict(fits=True, predicted_mb=None, available_mb=None, reserve_mb=self._reserve_mb)

        effective_free_mb = free_vram_mb - committed_reserve_mb
        predicted = (
            predict_job_sampler_only_vram_mb(job, baseline)
            if disaggregated
            else predict_job_sampling_vram_mb(job, baseline)
        )
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


# The defaults a partially-mocked or older config falls back to, so the pressure check never crashes the
# scheduling cycle on a non-numeric attribute (mirrors how the unservable breaker tolerates a bad threshold).
# Kept in step with reGenBridgeData.ram_pressure_pause_percent's default: a resident inference process can
# allocate several GB in a single step, so the floor leaves ~15% of RAM free rather than only 10%.
_DEFAULT_RAM_PRESSURE_PAUSE_PERCENT = 85.0
_DEFAULT_RAM_PRESSURE_MIN_FREE_MB = 1024.0


def ram_pressure_floor_mb(
    total_ram_mb: float | None,
    *,
    pause_percent: float = _DEFAULT_RAM_PRESSURE_PAUSE_PERCENT,
    min_free_mb: float = _DEFAULT_RAM_PRESSURE_MIN_FREE_MB,
) -> float:
    """Return the absolute available-RAM danger floor (MB): below it the worker must degrade, not load.

    The floor is the *more conservative* (higher) of two readings, so each protects the regime the other
    misses: ``(100 - pause_percent)%`` of total RAM guards a large-RAM host (where a fixed MB floor would be
    a negligible sliver), and ``min_free_mb`` guards a small-RAM host (where the percentage can resolve to
    too few megabytes to load a model's weights safely). With the defaults (90%, 1024 MB) a 32 GB host
    degrades below ~3.2 GB free and an 8 GB host below 1 GB free. ``min_free_mb`` alone applies when total
    RAM is unknown.
    """
    if total_ram_mb is None or total_ram_mb <= 0:
        return float(min_free_mb)
    percent_floor = max(0.0, (100.0 - pause_percent)) / 100.0 * float(total_ram_mb)
    return max(percent_floor, float(min_free_mb))


@dataclass(frozen=True)
class RamPressureVerdict:
    """Whether the host is below its absolute system-RAM danger floor, with enough detail to log a reason.

    Distinct from :class:`BudgetVerdict`, which is a marginal per-job admission check: this is the
    whole-host floor that drives the degrade response (refuse new loads, shed idle processes, throttle
    pops) the OOM-kill spiral needs, independent of any one job's predicted cost.
    """

    under_pressure: bool
    """True when available RAM is below the danger floor (or, conservatively, when no reading exists)."""
    available_mb: float | None
    """Measured available system RAM (MB) at check time, or None when no telemetry exists."""
    floor_mb: float
    """The absolute available-RAM floor (MB) below which the worker degrades."""
    total_mb: float | None = None
    """Total system RAM (MB), or None when unknown."""

    def reason(self) -> str:
        """Return a short human-readable explanation, for logging a degrade/clear decision."""
        if self.available_mb is None:
            return f"no RAM telemetry; floor {self.floor_mb:.0f} MB"
        verb = "below" if self.under_pressure else "above"
        return f"available {self.available_mb:.0f} MB {verb} danger floor {self.floor_mb:.0f} MB"


def assess_ram_pressure(
    available_ram_mb: float | None,
    total_ram_mb: float | None,
    *,
    pause_percent: float = _DEFAULT_RAM_PRESSURE_PAUSE_PERCENT,
    min_free_mb: float = _DEFAULT_RAM_PRESSURE_MIN_FREE_MB,
) -> RamPressureVerdict:
    """Return whether the host is below its absolute system-RAM danger floor.

    Pure policy over measured readings: ``under_pressure`` is True when measured available RAM has fallen
    below :func:`ram_pressure_floor_mb`. A missing available reading yields ``under_pressure=False`` (no
    telemetry never *fabricates* pressure, so a worker that has not yet measured RAM is not wedged); a
    missing total only widens the floor to the absolute ``min_free_mb``. Never raises.
    """
    floor_mb = ram_pressure_floor_mb(total_ram_mb, pause_percent=pause_percent, min_free_mb=min_free_mb)
    under = available_ram_mb is not None and available_ram_mb < floor_mb
    return RamPressureVerdict(
        under_pressure=under,
        available_mb=available_ram_mb,
        floor_mb=floor_mb,
        total_mb=total_ram_mb,
    )
