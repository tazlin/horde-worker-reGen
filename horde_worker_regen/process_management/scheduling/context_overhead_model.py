"""Per-context VRAM overhead measurement model.

A torch/CUDA inference process holds a fixed VRAM cost (the one-time device-wide runtime plus one
context) before any model loads, and each additional sibling context adds a marginal cost. The streaming
forecast needs both figures to size the free VRAM achievable under sole residency and after evicting
sibling *models* without multiplying the one-time cost by the process count.

This model owns those measurements and the derivations over them; it performs no orchestration and holds
no collaborator references, so the scheduler feeds it plain numbers (free/used VRAM and process counts it
gathers from the process map) and reads back derived overhead. Keeping the numeric model standalone makes
its derivation rules directly unit-testable without a process map or a running pool.
"""

from __future__ import annotations

from horde_worker_regen.utils.config_coercion import config_number


class ContextOverheadModel:
    """Tracks measured per-process and marginal CUDA-context VRAM costs and derives forecast inputs.

    The per-process overhead is the first/sole context cost (it includes the one-time device-wide CUDA
    runtime allocation); the marginal overhead is the cost of each *additional* sibling context. Both are
    measured at startup by the manager's probe; the marginal can also be derived from an observed
    all-contexts idle residency when the probe could not measure it directly.
    """

    def __init__(self) -> None:
        """Initialize with no measurements; every figure reads as unset until the probe or an observation lands."""
        # Startup-measured per-process VRAM overhead: one torch/CUDA context, no model. The streaming
        # forecast subtracts it from total VRAM to estimate the free achievable under sole residency. 0
        # until measured (free-if-alone == total then). NB: this is the *first/sole* context cost (it
        # includes the one-time, device-wide CUDA runtime allocation), NOT the marginal cost of an
        # additional sibling context, which is derived below.
        self._per_process_overhead_mb: float = 0.0
        # Startup-measured *marginal* VRAM cost of each additional sibling context (the probe's second-
        # context delta). Hard data available from the first scheduling tick, so it sizes
        # free_after_model_evict correctly even in the startup window before any sibling reaches idle. 0
        # until measured (or unmeasurable), where the model falls back to the idle-residency derivation
        # and then to the conservative overhead-per-context sizing.
        self._marginal_overhead_mb: float = 0.0
        # Lowest device-wide *used* VRAM observed while every loaded inference process is idle with no model
        # resident (the clean all-contexts baseline, typically at startup). This is the true combined cost
        # of all process contexts, the one-time CUDA runtime plus one context each, so the marginal cost of
        # an additional context is (residency - per_process_overhead) / (count - 1). A runtime fallback for
        # the probe's direct marginal measurement: sizes free_after_model_evict from measurement instead of
        # multiplying the one-time cost by the process count. None until seen.
        self._idle_context_residency_mb: float | None = None
        self._idle_residency_process_count: int = 0
        # Highest device-wide *used* VRAM observed while every loaded inference process is idle with no model
        # resident: the floor reclaim can never get below. The clean baseline above keeps the *minimum* on
        # the assumption a model's cache returns to the device when it unloads; when that assumption fails
        # (the allocator/runtime retains multi-GB per context, as a real inference context does once it has
        # loaded a checkpoint), the *effective* floor is the maximum, not the minimum. A probe measured
        # against a minimal holder under-counts this, so once the effective floor is known it supersedes the
        # probe in deriving the per-context marginal; otherwise the forecast believes in reclaimable VRAM the
        # device never returns and routes every load into an evict-all admit. None until seen.
        self._effective_idle_used_mb: float | None = None
        self._effective_idle_process_count: int = 0

    def set_per_process_overhead_mb(self, overhead_mb: int | float) -> None:
        """Record the startup-measured per-process VRAM overhead (MB) for the streaming forecast."""
        coerced = config_number(overhead_mb)
        if coerced is not None and coerced >= 0:
            self._per_process_overhead_mb = coerced

    def set_marginal_overhead_mb(self, marginal_mb: int | float) -> None:
        """Record the startup-measured *marginal* per-additional-context VRAM cost (MB) from the probe.

        Hard data (the probe's second-context delta) available from the first scheduling tick, so it fixes
        the startup-window over-count without waiting for siblings to reach idle. 0 (or unmeasurable) leaves
        the model on its idle-residency fallback.
        """
        coerced = config_number(marginal_mb)
        if coerced is not None and coerced >= 0:
            self._marginal_overhead_mb = coerced

    def per_process_mb(self, *, config_override_mb: float | None) -> float:
        """Return the per-process VRAM overhead (MB) to assume: configured override, else measured, else 0.

        An explicit ``vram_per_process_overhead_mb`` config value (> 0) wins so operators can tune; otherwise
        the startup-measured figure is used. This is the *first/sole* context cost (it includes the one-time
        CUDA runtime allocation), used to size ``free_if_alone``; the per-additional-context cost is
        :meth:`marginal_mb`.

        Args:
            config_override_mb (float | None): The coerced ``vram_per_process_overhead_mb`` config value, or
                None when it is unset or non-numeric (the scheduler coerces it before passing it in).
        """
        if config_override_mb is not None and config_override_mb > 0:
            return config_override_mb
        return self._per_process_overhead_mb

    def observe_idle_residency(self, *, used_mb: float, idle_inference_process_count: int) -> None:
        """Record a device-wide used-VRAM reading taken while every inference process is idle and model-less.

        The reading is the true combined cost of all process contexts (the one-time CUDA runtime plus one
        context each), which the forecast needs to size ``free_after_model_evict`` without multiplying the
        one-time cost by the process count. The clean window is at startup, before any model loads; once a
        model has loaded this rarely holds again, so the minimum observed value is kept (later, cache-dirtied
        observations read higher and are ignored for the clean baseline). The *effective* floor keeps the
        worst (highest) reading per process count instead, since that VRAM provably never returns.

        The caller is responsible for confirming the clean precondition (all inference processes up, idle,
        and holding no model) and for computing ``used_mb``; this method only updates the cached figures.

        Args:
            used_mb (float): Device-wide used VRAM (total minus free) at the clean-baseline reading.
            idle_inference_process_count (int): Number of live inference processes at the reading.
        """
        if self._idle_context_residency_mb is None or used_mb < self._idle_context_residency_mb:
            self._idle_context_residency_mb = used_mb
            self._idle_residency_process_count = idle_inference_process_count
        # The effective floor is the *worst* (highest) fully-idle, fully-evicted reading: the VRAM reclaim
        # provably cannot return. Kept per the live context count so a later, fewer-process reading does not
        # mask an earlier over-commit.
        if (
            self._effective_idle_used_mb is None
            or idle_inference_process_count > self._effective_idle_process_count
            or (
                idle_inference_process_count == self._effective_idle_process_count
                and used_mb > self._effective_idle_used_mb
            )
        ):
            self._effective_idle_used_mb = used_mb
            self._effective_idle_process_count = idle_inference_process_count

    def marginal_mb(self, *, config_override_mb: float | None) -> float | None:
        """Return the per-additional-context VRAM cost (MB), or None to fall back to the first-context overhead.

        Prefers the probe's directly-measured second-context delta (hard data, available from the first tick,
        so it also covers the startup window where siblings have not yet reached idle). Failing that (the
        probe could not measure it on this backend), derives it from the measured all-contexts idle residency:
        ``residency = per_process_overhead + (count - 1) * marginal``, so ``marginal = (residency -
        per_process_overhead) / (count - 1)``. Returns None when neither is available: no probe delta, and no
        clean idle baseline (or only one process up, or a residency at/below the first-context overhead, an
        inconsistent reading), in which case the forecast conservatively reuses the first-context overhead per
        additional context.

        Args:
            config_override_mb (float | None): The coerced ``vram_per_process_overhead_mb`` config value (it
                feeds the per-process overhead the derivation subtracts), or None when unset.
        """
        per_process = self.per_process_mb(config_override_mb=config_override_mb)

        def _derive(residency: float | None, count: int) -> float | None:
            if residency is None or count < 2 or per_process <= 0 or residency <= per_process:
                return None
            return (residency - per_process) / (count - 1)

        # The measured *effective* floor is ground truth for what reclaim achieves: when a real inference
        # context retains more cache than the probe's minimal holder allocated, the probe under-counts the
        # marginal and the forecast over-counts reclaimable VRAM. Take the larger of the probe estimate and
        # the effective-floor derivation so the forecast never believes in headroom the device will not
        # return; the effective floor only rises above the probe once contexts genuinely over-commit (the
        # threads>1 regime), so a roomy card keeps the probe estimate unchanged.
        effective_derived = _derive(self._effective_idle_used_mb, self._effective_idle_process_count)
        probe = self._marginal_overhead_mb if self._marginal_overhead_mb > 0 else None
        candidates = [candidate for candidate in (probe, effective_derived) if candidate is not None]
        if candidates:
            return max(candidates)
        # No probe and no effective floor: fall back to the clean idle-residency derivation (startup path).
        return _derive(self._idle_context_residency_mb, self._idle_residency_process_count)
