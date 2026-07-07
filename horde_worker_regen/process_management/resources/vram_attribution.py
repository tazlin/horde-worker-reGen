"""Observational reconciliation of the worker's committed-VRAM ledger against the device's true usage.

The worker's several inference/lane processes each commit device VRAM independently; the committed-VRAM
ledger (:meth:`ProcessMap.committed_vram_mb`) sums every live process's ``context_constant +
process_reserved_mb`` into the exact device memory attributable to the worker. This module reconciles that
ledger against a parent-side, device-wide *used* reading (NVML device-total-used, read torch-free from
outside the CUDA workload) plus a captured device baseline (the OS/desktop/other-apps VRAM the worker cannot
attribute to any of its processes):

    drift_mb = device_used_mb - (baseline_estimate_mb + committed_vram_mb)

A persistent positive drift means the device holds more VRAM than the worker's own ledger plus the baseline
account for: either an un-attributed allocation, a leak, or (the case this exists for) VRAM the driver has
already begun spilling to host RAM. On Windows/WDDM this ledger arithmetic is the ONLY early
overcommit/paging signal that exists: the driver never OOMs at the physical ceiling (allocations silently
demote to the system-backed shared segment), and both ``mem_get_info`` and core-utilization telemetry keep
reading healthy, so no probe or driver counter can see the overcommit coming; only this sum-vs-capacity
arithmetic can.

This layer is strictly observational: it measures, captures the baseline, and emits a single rate-limited
warning when drift persists. It does not feed admission or forecast decisions. When no device-wide used
source exists on a given path (no NVML, non-NVIDIA host) the reconciliation degrades to producing no drift
(the caller can still log committed-vs-capacity); it never raises.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

_DRIFT_WARN_THRESHOLD_MB = 1024.0
"""Positive drift (MB) above which the attribution is considered materially unexplained.

Sized to absorb ordinary measurement noise and the sub-GB slack between a snapshot ``process_reserved_mb``
and the device's rounded used figure, so only a genuine multi-hundred-MB-plus un-attributed commitment
(the leading edge of a WDDM spill) trips it."""

_DRIFT_CONSECUTIVE_OBSERVATIONS = 2
"""Consecutive over-threshold observations required before the warning fires, so a single transient spike
(a reading taken mid-load before the ledger caught up) does not warn."""

_DRIFT_WARN_INTERVAL_SECONDS = 60.0
"""Minimum seconds between drift warnings, so a sustained drift logs once a minute rather than every tick."""

_PHYSICAL_PRESSURE_CONSECUTIVE_OBSERVATIONS = 2
"""Consecutive physical-overcommit observations required before one pressure unload is issued.

Mirrors :data:`_DRIFT_CONSECUTIVE_OBSERVATIONS`: a single transient reading (a sampling peak caught mid-step
before the allocator settled) must not trigger an eviction, so the physical condition
``committed + baseline > total`` has to hold across this many consecutive fresh observations first."""

_LEDGER_PHANTOM_TOLERANCE_MB = 2048.0
"""Margin (MB) by which the committed ledger must exceed the device-used truth to be judged a phantom.

The committed ledger sums each lane's ``memory_reserved()``, a per-process figure that can detach upward from
device reality: an unloaded model's allocator cache the torch caching allocator has not returned, or a
reservation the WDDM driver already demoted to host RAM, both keep counting against ``committed`` while the
physical pages are free. Since the worker cannot hold more device VRAM than the device itself reports used,
``committed`` exceeding ``device_used`` by more than this margin is arithmetically impossible for a truthful
ledger and marks a phantom over-count. Sized well above ordinary sampling/report skew so a genuine, briefly
under-reported real over-commit is never misclassified as a phantom."""

_RECALIBRATE_INTERVAL_SECONDS = 30.0
"""Minimum seconds between committed-ledger recalibration signals, so a persistent phantom asks the lanes to
release their allocator cache and re-report periodically rather than every attribution tick while the fresh
reports are still settling back to device truth."""

_REPORT_STALENESS_SECONDS = 15.0
"""Report age (seconds) beyond which a ledger contributor is treated as an UNKNOWN, incomparable tenant.

Three times the child's 5 s memory-report cadence, so an ordinary skipped or delayed report never trips it,
but a process that has genuinely stopped reporting (a wedged or blocked child) is not silently trusted at its
last figure. When any committed-ledger contributor is this stale the reconciliation is skipped entirely
(drift uncomputable): staleness-aware reconciliation prevents both false drift alarms (warning on a device
anchor the ledger can no longer be compared to) and false confidence (declaring no drift while a tenant's true
footprint is unknown)."""


@dataclass(frozen=True)
class DriftObservation:
    """The outcome of one reconciliation: the drift and whether the caller should warn.

    ``drift_mb`` is None when the reconciliation could not be computed (no device-used reading yet, or no
    baseline captured), in which case ``should_warn`` is always False and the caller degrades to logging
    committed-vs-capacity only.
    """

    drift_mb: float | None
    """``device_used_mb - (baseline_estimate_mb + committed_vram_mb)``, or None when uncomputable."""
    device_used_mb: float | None
    """The device-wide used VRAM (MB) reconciled against, or None when no source was available."""
    baseline_estimate_mb: float | None
    """The captured device baseline (MB) at the quietest observed moment, or None until captured."""
    committed_vram_mb: float
    """The worker's committed-VRAM ledger sum (MB) at this observation."""
    consecutive_over_threshold: int
    """How many consecutive observations drift has now exceeded the threshold (0 when it did not)."""
    should_warn: bool
    """Whether the caller should emit the single rate-limited drift warning for this observation."""


@dataclass(frozen=True)
class PhysicalPressureObservation:
    """The outcome of one physical-overcommit check: whether the worker has physically over-committed the card.

    Distinct from :class:`DriftObservation` (which reconciles the ledger against a device-used anchor to warn):
    this reasons purely about the worker's own committed ledger against the physical ceiling, and drives a
    corrective *action* (one idle-model unload), so it carries its own streak and hysteresis state.

    The trigger is the *physical* over-commit ``committed_vram_mb + baseline_estimate_mb > total_vram_mb``, NOT
    an exceedance of the (lower) admission ceiling: legitimate transient sampling peaks routinely exceed the
    admission ceiling and must never trigger an eviction, whereas the worker's committed footprint plus the
    shared baseline exceeding the physical total means the card is genuinely over-subscribed and the driver is
    about to spill to host RAM.
    """

    over_physical_ceiling: bool
    """True when ``committed + baseline > total`` for this observation (0 when uncomputable/stale)."""
    consecutive_over_ceiling: int
    """How many consecutive fresh observations the physical over-commit has now held (0 when it did not)."""
    should_unload: bool
    """Whether the caller should issue one under-pressure idle-model unload for this observation.

    Only set for a *corroborated* over-commit (the device-used truth also shows the card near full, or no
    device-used reading was available); a phantom over-commit signals :attr:`should_recalibrate` instead."""
    ledger_phantom: bool
    """True when the committed over-commit is contradicted by the device-used truth (``committed`` exceeds
    ``device_used`` beyond the phantom tolerance): a bookkeeping over-count no eviction can cure."""
    should_recalibrate: bool
    """Whether the caller should recalibrate the committed ledger to device truth this observation.

    Set for a streak-confirmed, rate-limited phantom over-commit: the correct response is to have the idle
    lanes release their allocator cache and re-report so ``committed`` converges back to device reality, NOT
    to evict a model (there is nothing real to evict) and NOT to latch suppressed (which would wedge admission
    on the phantom forever)."""
    committed_vram_mb: float
    """The worker's committed-VRAM ledger sum (MB) at this observation."""
    device_used_mb: float | None
    """The device-wide used VRAM (MB) truth reconciled against, or None when no source was available."""
    baseline_estimate_mb: float | None
    """The captured device baseline (MB), or None until captured (then the check is uncomputable)."""
    total_vram_mb: float | None
    """Device total VRAM (MB), or None when unknown (then the check is uncomputable)."""


class VramAttributionReconciler:
    """Captures the device baseline and reconciles it, plus the committed ledger, against device-used VRAM.

    Holds only scalar state (the captured baseline, the consecutive-drift streak, the last-warn time); the
    caller supplies each observation's numbers (device-used, committed sum) and whether the worker currently
    holds any resident model, and receives a :class:`DriftObservation` describing the drift and whether to
    warn. Keeping the arithmetic and the streak/rate-limit policy here (rather than in the control loop) makes
    it directly unit-testable without a process map or NVML.
    """

    def __init__(
        self,
        *,
        drift_warn_threshold_mb: float = _DRIFT_WARN_THRESHOLD_MB,
        consecutive_observations: int = _DRIFT_CONSECUTIVE_OBSERVATIONS,
        warn_interval_seconds: float = _DRIFT_WARN_INTERVAL_SECONDS,
    ) -> None:
        """Initialize with no captured baseline and no drift streak.

        Args:
            drift_warn_threshold_mb: Positive drift (MB) above which an observation counts toward warning.
            consecutive_observations: Consecutive over-threshold observations required before warning.
            warn_interval_seconds: Minimum seconds between successive drift warnings.
        """
        self._drift_warn_threshold_mb = drift_warn_threshold_mb
        self._consecutive_observations = consecutive_observations
        self._warn_interval_seconds = warn_interval_seconds
        self._baseline_estimate_mb: float | None = None
        self._consecutive_over_threshold = 0
        self._pressure_consecutive_required = _PHYSICAL_PRESSURE_CONSECUTIVE_OBSERVATIONS
        self._pressure_consecutive_over_ceiling = 0
        # Once a pressure unload is issued, suppress re-issuing until the physical over-commit clears (measured
        # committed drops back below the ceiling), so a single sustained over-commit reclaims once, not every tick.
        self._pressure_suppressed = False
        # A phantom over-commit (committed exceeds device-used truth) drives recalibration, not eviction; it
        # carries its own confirming streak and rate-limit so a transient reading does not fire and a persistent
        # one recalibrates periodically rather than every tick.
        self._ledger_phantom_tolerance_mb = _LEDGER_PHANTOM_TOLERANCE_MB
        self._recalibrate_interval_seconds = _RECALIBRATE_INTERVAL_SECONDS
        self._phantom_consecutive = 0
        self._last_recalibrate_time: float | None = None
        # None until the first warning fires, so the rate-limit never suppresses the very first warning
        # (a fixed 0.0 would gate it out whenever the clock reads below the interval, e.g. under a test clock).
        self._last_warn_time: float | None = None

    @property
    def baseline_estimate_mb(self) -> float | None:
        """The captured device baseline (MB), or None until a quiet reading has been observed."""
        return self._baseline_estimate_mb

    def note_baseline(self, device_used_mb: float | None, *, any_model_resident: bool) -> None:
        """Capture the device baseline: the minimum device-used observed while no worker model is resident.

        The baseline is the shared device VRAM the worker cannot attribute to any of its processes (OS,
        desktop, other applications, and the fixed contexts of any GPU processes that have not yet reported
        an allocator reservation). Taken as the minimum device-used seen at a quiet moment (no worker model
        loaded) so it never absorbs the worker's own resident weights, which would then be double-subtracted
        from the drift. A reading taken while a model is resident is ignored for this purpose.

        Args:
            device_used_mb: The current device-wide used VRAM (MB), or None when no source is available.
            any_model_resident: Whether any worker process currently holds a resident model.
        """
        if device_used_mb is None or any_model_resident:
            return
        if self._baseline_estimate_mb is None or device_used_mb < self._baseline_estimate_mb:
            self._baseline_estimate_mb = device_used_mb

    def observe(
        self,
        *,
        device_used_mb: float | None,
        committed_vram_mb: float,
        committed_is_stale: bool = False,
        now: float | None = None,
    ) -> DriftObservation:
        """Reconcile the committed ledger and captured baseline against the device-used reading.

        Computes ``drift = device_used_mb - (baseline_estimate_mb + committed_vram_mb)``, advances the
        consecutive-over-threshold streak, and decides whether to warn (streak reached the required
        consecutive count AND the rate-limit interval has elapsed since the last warning). When the drift
        cannot be computed (no device-used reading, no baseline captured yet, or the committed ledger is
        stale) the streak resets and no warning is signalled: the caller degrades to logging
        committed-vs-capacity only.

        A stale committed ledger is treated exactly like a missing reading: one contributor whose report has
        aged out makes the whole ledger an UNKNOWN tenant that the device anchor cannot be compared to, so
        reconciling would risk both a false drift alarm and false confidence. The streak resets so a
        transient staleness window never carries a partial streak into the next comparable observation.

        Args:
            device_used_mb: The device-wide used VRAM (MB), or None when no source was available.
            committed_vram_mb: The worker's committed-VRAM ledger sum (MB) at this observation.
            committed_is_stale: True when a committed-ledger contributor's report has aged past the staleness
                bound, making the ledger incomparable to the device anchor for this observation.
            now: Optional time override (epoch seconds) for the rate-limit comparison.
        """
        current = time.time() if now is None else now
        if device_used_mb is None or self._baseline_estimate_mb is None or committed_is_stale:
            self._consecutive_over_threshold = 0
            return DriftObservation(
                drift_mb=None,
                device_used_mb=device_used_mb,
                baseline_estimate_mb=self._baseline_estimate_mb,
                committed_vram_mb=committed_vram_mb,
                consecutive_over_threshold=0,
                should_warn=False,
            )

        drift_mb = device_used_mb - (self._baseline_estimate_mb + committed_vram_mb)
        if drift_mb > self._drift_warn_threshold_mb:
            self._consecutive_over_threshold += 1
        else:
            self._consecutive_over_threshold = 0

        should_warn = False
        if self._consecutive_over_threshold >= self._consecutive_observations and (
            self._last_warn_time is None or (current - self._last_warn_time) >= self._warn_interval_seconds
        ):
            should_warn = True
            self._last_warn_time = current

        return DriftObservation(
            drift_mb=drift_mb,
            device_used_mb=device_used_mb,
            baseline_estimate_mb=self._baseline_estimate_mb,
            committed_vram_mb=committed_vram_mb,
            consecutive_over_threshold=self._consecutive_over_threshold,
            should_warn=should_warn,
        )

    def observe_physical_pressure(
        self,
        *,
        committed_vram_mb: float,
        total_vram_mb: float | None,
        device_used_mb: float | None = None,
        committed_is_stale: bool = False,
        now: float | None = None,
    ) -> PhysicalPressureObservation:
        """Decide whether the worker has physically over-committed the card, and how to relieve it.

        The trigger is ``committed_vram_mb + baseline_estimate_mb > total_vram_mb`` (the *physical* ceiling,
        NOT the lower admission ceiling): only a genuine over-subscription of device VRAM warrants evicting an
        idle resident model, so transient sampling peaks that exceed the admission ceiling but stay within the
        physical total never fire. The over-commit must hold across
        :data:`_PHYSICAL_PRESSURE_CONSECUTIVE_OBSERVATIONS` consecutive fresh observations (a single transient
        reading does not fire).

        The response branches on whether the device-used truth corroborates the ledger, because the committed
        ledger sums each lane's ``memory_reserved()`` and that per-process figure can detach *upward* from
        device reality (an unloaded model's allocator cache the caching allocator has not returned, or a
        reservation the WDDM driver already spilled to host RAM). The worker cannot hold more device VRAM than
        the device reports used, so:

        * When ``committed`` exceeds ``device_used`` beyond :data:`_LEDGER_PHANTOM_TOLERANCE_MB` the over-commit
          is a **phantom**: eviction cannot cure it (there is nothing real to evict) and latching suppressed
          would wedge admission on the phantom forever. This signals :attr:`should_recalibrate` (streak-confirmed
          and rate-limited) so the caller has the idle lanes release their allocator cache and re-report, driving
          ``committed`` back to device truth. It never latches suppressed.
        * Otherwise the over-commit is **corroborated** (or ``device_used`` is unavailable, so the ledger is
          trusted): this signals :attr:`should_unload` once per sustained over-commit, suppressed until the
          measured committed drops back below the ceiling, exactly as before.

        Uncomputable inputs (a stale committed ledger, no captured baseline, or an unknown total) reset every
        streak and signal no action: the check degrades safely exactly as the drift reconciliation does.

        Args:
            committed_vram_mb: The worker's committed-VRAM ledger sum (MB) at this observation.
            total_vram_mb: Device total VRAM (MB), or None when unknown.
            device_used_mb: The device-wide used VRAM (MB) truth (parent-side NVML), or None when unavailable.
            committed_is_stale: True when a ledger contributor's report has aged past the staleness bound.
            now: Optional time override (epoch seconds) for the recalibration rate-limit comparison.
        """
        current = time.time() if now is None else now
        baseline = self._baseline_estimate_mb
        if committed_is_stale or total_vram_mb is None or baseline is None:
            self._pressure_consecutive_over_ceiling = 0
            self._phantom_consecutive = 0
            return PhysicalPressureObservation(
                over_physical_ceiling=False,
                consecutive_over_ceiling=0,
                should_unload=False,
                ledger_phantom=False,
                should_recalibrate=False,
                committed_vram_mb=committed_vram_mb,
                device_used_mb=device_used_mb,
                baseline_estimate_mb=baseline,
                total_vram_mb=total_vram_mb,
            )

        over_ceiling = (committed_vram_mb + baseline) > total_vram_mb
        phantom = (
            over_ceiling
            and device_used_mb is not None
            and committed_vram_mb > device_used_mb + self._ledger_phantom_tolerance_mb
        )

        should_unload = False
        should_recalibrate = False

        if phantom:
            # The ledger claims an over-commit the device does not corroborate. Do not evict (nothing real is
            # held) and do not suppress: recalibrate the ledger back to device truth, rate-limited so the
            # freshly re-reported reservations get a chance to settle between signals.
            self._pressure_consecutive_over_ceiling = 0
            self._pressure_suppressed = False
            self._phantom_consecutive += 1
            if self._phantom_consecutive >= self._pressure_consecutive_required and (
                self._last_recalibrate_time is None
                or (current - self._last_recalibrate_time) >= self._recalibrate_interval_seconds
            ):
                should_recalibrate = True
                self._last_recalibrate_time = current
        elif over_ceiling:
            self._phantom_consecutive = 0
            self._pressure_consecutive_over_ceiling += 1
            if (
                self._pressure_consecutive_over_ceiling >= self._pressure_consecutive_required
                and not self._pressure_suppressed
            ):
                should_unload = True
                self._pressure_suppressed = True
        else:
            self._phantom_consecutive = 0
            self._pressure_consecutive_over_ceiling = 0
            # The physical over-commit has cleared, so a future over-commit is again eligible to unload.
            self._pressure_suppressed = False

        return PhysicalPressureObservation(
            over_physical_ceiling=over_ceiling,
            consecutive_over_ceiling=self._pressure_consecutive_over_ceiling,
            should_unload=should_unload,
            ledger_phantom=phantom,
            should_recalibrate=should_recalibrate,
            committed_vram_mb=committed_vram_mb,
            device_used_mb=device_used_mb,
            baseline_estimate_mb=baseline,
            total_vram_mb=total_vram_mb,
        )
