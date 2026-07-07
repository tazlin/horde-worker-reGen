"""Per-context VRAM overhead measurement model.

The accounting contract this model exists to enforce: device VRAM used decomposes into four terms that
must never be conflated when admitting concurrent work onto one card.

1. **Device baseline** (shared, per-device): the VRAM the OS/desktop/other applications hold. It is
   attributable to no worker process and enters the arithmetic exactly *once*, at device level (it is
   already reflected in the measured device-wide free figure). It must never be charged per process.
2. **Per-process marginal fixed overhead** (per additional context): the CUDA context plus import-time
   allocations, roughly 200-300MB on a 24GB CUDA card. It persists after a model unloads and is purgeable
   only by the process exiting. The *first/sole* context additionally pays the one-time, device-wide CUDA
   runtime allocation; that one-time cost is paid once and shared, so ``per_process_overhead_mb`` (the
   first-context figure) sizes sole residency while each *additional* sibling context costs only the
   marginal above.
3. **Unloadable model weights** (per resident model): reclaimed by evicting the model.
4. **Transient per-job activation peaks** (per running job): present only while a job samples/decodes.

A torch/CUDA inference process therefore holds term (2) before any model loads, and the streaming forecast
needs both the first-context figure and the marginal to size the free VRAM achievable under sole residency
and after evicting sibling *models* without multiplying the one-time cost (or the device baseline) by the
process count. A device-wide *used* reading conflates (1) with (2)+(3)+(4), so it is only ever a source for
the free-VRAM computation, never a per-process charge: charging an idle process's device-wide reading as
its own overhead would fold the whole device baseline into a single context.

This model owns those measurements and the derivations over them; it performs no orchestration and holds
no collaborator references, so the scheduler feeds it plain numbers (free/used VRAM and process counts it
gathers from the process map) and reads back derived overhead. Keeping the numeric model standalone makes
its derivation rules directly unit-testable without a process map or a running pool.
"""

from __future__ import annotations

from dataclasses import dataclass

from horde_worker_regen.utils.config_coercion import config_number


@dataclass(frozen=True)
class MarginalOverheadBreakdown:
    """The inputs and the chosen value behind a per-additional-context marginal, for diagnostics.

    Exposes which signal won (the directly-probed delta or the idle-residency derivation) so the
    streaming-forecast log can show *why* the forecast sized ``free_after_model_evict`` the way it did, rather
    than only the resulting number.
    """

    probe_mb: float | None
    """The probe's directly-measured second-context delta (MB), or None when the backend could not measure it
    (e.g. the per-process VRAM view on Windows WDDM, or a probe failure)."""
    idle_floor_mb: float | None
    """The marginal derived from the (invalidation-corrected) measured idle residency (MB), or None when no
    usable idle reading exists."""
    chosen_mb: float | None
    """The measured marginal the forecast will use (MB), or None when nothing was measured.

    None does NOT mean the forecast charges the full first-context overhead per additional context: the
    forecast seeds a conservative per-additional-context constant instead (see
    ``resource_budget._SEEDED_MARGINAL_CONTEXT_OVERHEAD_MB``). It stays None here so measurement-gated policy
    (e.g. whether a whole-card teardown demand rests on a *measured* per-context cost) can tell an actual
    measurement from the seed."""
    source: str
    """Which signal produced ``chosen_mb``: ``probe``, ``idle_floor``, or ``seeded``.

    ``seeded`` means neither the probe nor an idle-residency reading measured the marginal, so the downstream
    forecast prices additional contexts with the seeded conservative constant rather than any measured value.
    It is named ``seeded`` (not ``unmeasured``) so the forecast log reflects honestly that a deliberate seed,
    not the first-context overhead, is what is charged per extra context."""


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
        # Lowest worker-attributable bare-context total (MB) observed while every loaded inference process is
        # idle with no model resident (the clean all-contexts baseline, typically at startup). The caller
        # computes the reading as truthful device-used minus the shared device baseline minus every GPU
        # tenant's byte-exact allocator reservation, so it contains ONLY the context costs of the live GPU
        # tenants: the one-time CUDA runtime plus one context each. The marginal cost of an additional
        # context is then (residency - per_process_overhead) / (count - 1). A runtime fallback for the
        # probe's direct marginal measurement: sizes free_after_model_evict from measurement instead of
        # multiplying the one-time cost by the process count. None until seen.
        self._idle_context_residency_mb: float | None = None
        self._idle_residency_context_count: int = 0
        # Highest bare-context total observed at a clean all-idle reading: the floor reclaim can never get
        # below. The clean baseline above keeps the *minimum* on the assumption a context's runtime-held VRAM
        # returns when its work ends; when that assumption fails (the runtime retains allocations the torch
        # allocator cannot see, so they survive the reservation subtraction), the *effective* floor is the
        # maximum, not the minimum. A probe measured against a minimal holder under-counts this, so once the
        # effective floor is known it supersedes the probe in deriving the per-context marginal; otherwise
        # the forecast believes in reclaimable VRAM the device never returns and routes every load into an
        # evict-all admit. The max can over-read on a transient spike (a reading taken before a freed
        # allocation actually returned), so observe_device_residency ratchets it back down when a later
        # reading proves the device runs below it: capture raises it to the worst clean reading,
        # invalidation lowers it toward the level the device actually sustains. None until seen.
        self._effective_idle_context_total_mb: float | None = None
        self._effective_idle_context_count: int = 0

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

    def observe_idle_residency(self, *, context_total_mb: float, context_count: int) -> None:
        """Record the bare-context total observed while every inference process is idle and model-less.

        The reading is the true combined cost of the live GPU tenants' contexts (the one-time CUDA runtime
        plus one context each), which the forecast needs to size ``free_after_model_evict`` without
        multiplying the one-time cost by the process count. The clean window is at startup, before any model
        loads; once a model has loaded this rarely holds again, so the minimum observed value is kept. The
        *effective* floor keeps the worst (highest) reading per context count instead, since that VRAM
        provably never returns.

        The caller is responsible for confirming the clean precondition (all inference processes up, idle,
        and holding no model) and for computing ``context_total_mb`` as truthful device-used minus the shared
        device baseline minus every GPU tenant's byte-exact allocator reservation, so the figure contains
        only context costs: never the device baseline, never resident weights. Attributing anything else here
        multiplies it into the per-context marginal and prices the whole card into phantom over-commit.

        Args:
            context_total_mb (float): Worker-attributable bare-context VRAM total (MB) at the reading.
            context_count (int): Number of live GPU-context-bearing worker processes at the reading.
        """
        if self._idle_context_residency_mb is None or context_total_mb < self._idle_context_residency_mb:
            self._idle_context_residency_mb = context_total_mb
            self._idle_residency_context_count = context_count
        # The effective floor is the *worst* (highest) fully-idle, fully-evicted reading: the VRAM reclaim
        # provably cannot return. Kept per the live context count so a later, fewer-process reading does not
        # mask an earlier over-commit.
        if (
            self._effective_idle_context_total_mb is None
            or context_count > self._effective_idle_context_count
            or (
                context_count == self._effective_idle_context_count
                and context_total_mb > self._effective_idle_context_total_mb
            )
        ):
            self._effective_idle_context_total_mb = context_total_mb
            self._effective_idle_context_count = context_count

    def marginal_mb(self, *, config_override_mb: float | None) -> float | None:
        """Return the per-additional-context VRAM cost (MB), or None to fall back to the first-context overhead.

        See :meth:`marginal_breakdown` for the full resolution rule; this returns only the chosen value.

        Args:
            config_override_mb (float | None): The coerced ``vram_per_process_overhead_mb`` config value (it
                feeds the per-process overhead the derivation subtracts), or None when unset.
        """
        return self.marginal_breakdown(config_override_mb=config_override_mb).chosen_mb

    def marginal_breakdown(self, *, config_override_mb: float | None) -> MarginalOverheadBreakdown:
        """Resolve the per-additional-context marginal and report the signals behind it.

        Prefers the larger of the probe's directly-measured second-context delta and the idle-residency
        derivation (``marginal = (residency - per_process_overhead) / (count - 1)``), so the forecast never
        under-counts reclaimable VRAM: a real inference context can retain more than the probe's minimal matmul
        holder allocated, and the measured idle floor catches that. The trustworthiness of a *high* idle floor
        is enforced upstream by :meth:`observe_device_residency`, which lowers a latched floor once the device
        proves it was a transient spike rather than sustained retention. So a floor that survives here is one
        the device has not contradicted, and it is allowed to supersede the probe; a transient over-read has
        already been corrected down before it reaches this point.

        Returns None for ``chosen_mb`` only when nothing is measurable (no probe delta and no usable idle
        reading), tagged ``source='seeded'``: the forecast then prices additional contexts with a conservative
        seeded constant (never the first-context overhead), so None here is a measurement absence, not a
        directive to charge the full overhead per context.

        Args:
            config_override_mb (float | None): The coerced ``vram_per_process_overhead_mb`` config value (it
                feeds the per-process overhead the derivation subtracts), or None when unset.
        """
        per_process = self.per_process_mb(config_override_mb=config_override_mb)

        def _derive(residency: float | None, count: int) -> float | None:
            if residency is None or count < 2 or per_process <= 0 or residency <= per_process:
                return None
            return (residency - per_process) / (count - 1)

        probe = self._marginal_overhead_mb if self._marginal_overhead_mb > 0 else None
        idle_floor = _derive(self._effective_idle_context_total_mb, self._effective_idle_context_count)
        if idle_floor is None:
            # No effective (worst-case) floor yet: fall back to the clean idle-residency derivation (startup).
            idle_floor = _derive(self._idle_context_residency_mb, self._idle_residency_context_count)

        candidates = [
            (value, source) for value, source in ((probe, "probe"), (idle_floor, "idle_floor")) if value is not None
        ]
        if not candidates:
            return MarginalOverheadBreakdown(probe, idle_floor, None, "seeded")
        chosen, source = max(candidates, key=lambda candidate: candidate[0])
        return MarginalOverheadBreakdown(probe, idle_floor, chosen, source)

    def observe_device_residency(self, *, context_total_mb: float, context_count: int) -> None:
        """Lower a latched effective idle floor once a later reading proves it was not sustained.

        The effective floor (:meth:`observe_idle_residency`) keeps the *worst* clean all-idle reading on the
        premise that the device retains that VRAM. A single transient spike, a reading taken before a freed
        runtime allocation actually returned, would otherwise pin the floor for the whole session and inflate
        the per-context marginal into a teardown-forcing phantom. Any later bare-context reading below the
        floor, taken with at least as many contexts live, disproves it: the device demonstrably runs below
        that level, so the VRAM the floor counted as unreclaimable was reclaimed.

        Unlike :meth:`observe_idle_residency` this does not require the clean all-idle precondition: the
        caller nets resident weights out via the byte-exact reservations, and any residual weight-adjacent
        VRAM only makes the reading (hence the correction) *conservative*. A reading still below the latched
        floor is therefore unambiguous proof the latch was too high. The floor only ratchets *down* here,
        toward the minimum the device has demonstrated, and never below zero.

        Args:
            context_total_mb (float): Current worker-attributable bare-context VRAM total (MB).
            context_count (int): Number of live GPU-context-bearing worker processes at the reading.
        """
        if self._effective_idle_context_total_mb is None:
            return
        if context_count < self._effective_idle_context_count:
            return
        if context_total_mb < self._effective_idle_context_total_mb:
            self._effective_idle_context_total_mb = max(0.0, context_total_mb)
