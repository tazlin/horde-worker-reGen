"""Whole-card exclusive-residency state and its pure queries.

A heavy model can claim a whole card to itself by stopping that card's idle sibling inference contexts
(a context's VRAM is only reclaimed when its process exits) and, on the card the safety process sits on,
moving safety off-GPU. This module owns the per-card residency records and every question that can be
answered from them alone: which cards hold a residency, which card holds a given model, what phase a
residency is in, whether an establish/restore grace window is active, and whether the bounded drain
backstop has elapsed. The scheduler keeps the transitions that touch live processes (establish, converge,
restore); it reads and writes residency state exclusively through
[`WholeCardResidencyLedger`][horde_worker_regen.process_management.scheduling.governance.whole_card.WholeCardResidencyLedger].

Also home to [`max_coresident_for_peak`]
[horde_worker_regen.process_management.scheduling.governance.whole_card.max_coresident_for_peak], the
pure sizing rule for how many live inference contexts a rejected peak can co-reside with.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto

from horde_worker_regen.process_management.resources.resource_budget import StreamForecast

__all__ = [
    "WholeCardPhase",
    "WholeCardResidency",
    "WholeCardResidencyLedger",
    "WholeCardResidencyMachine",
    "max_coresident_for_peak",
]


@dataclass
class WholeCardResidency:
    """Mutable whole-card exclusive-residency state for one card (the worker, on a single-GPU host).

    The ledger keys one of these per device index so two heavy models on different cards each hold their
    own residency independently. A single-GPU worker keeps exactly one instance under the ``None`` key,
    so its behaviour is identical to the pre-multi-GPU scalar fields.
    """

    model: str | None = None
    """The model holding (or being given) sole residency on this card; None when no residency is held."""
    forecast: StreamForecast | None = None
    """The streaming forecast that established this residency, cached for the status snapshot's hard numbers."""
    established_at: float = 0.0
    """When this residency was first established (stop siblings, cycle safety, load weights); 0.0 when none.

    The establishment intentionally holds the queue, which the recovery supervisor must not mistake for a
    structural wedge until the establish grace elapses."""
    cooldown_until: float = 0.0
    """Wall-clock time until which this residency is held even after its heavy job drains, so a burst of
    heavy jobs reuses one residency instead of each churning a teardown/restore + safety cycle."""
    restore_at: float = 0.0
    """When this residency was last restored (siblings respawned, safety cycled back on-GPU); 0.0 when none.
    The restore churn also briefly makes the queue unservable, so the wedge grace must cover it too."""


class WholeCardPhase(StrEnum):
    """The externally-visible phase of one card's whole-card residency."""

    NONE = auto()
    """No residency is held on the card."""
    ESTABLISHING = auto()
    """A residency is held and still inside its establish grace (teardown/load in progress)."""
    HOLDING = auto()
    """A residency is held past its establish grace (the heavy model owns the card)."""


class WholeCardResidencyLedger:
    """Owns the per-card whole-card residency records and their pure queries.

    Thread Safety:
        Owned and mutated exclusively by the scheduler's control loop; not safe for concurrent mutation.
    """

    def __init__(self) -> None:
        """Initialize an empty ledger (no card holds a residency)."""
        self._residencies: dict[int | None, WholeCardResidency] = {}

    def state_for(self, device_index: int | None) -> WholeCardResidency:
        """Return the (lazily-created) residency state for ``device_index``.

        ``None`` is the single-GPU / worker-wide key, so a single-GPU host keeps exactly one residency
        state.
        """
        state = self._residencies.get(device_index)
        if state is None:
            state = WholeCardResidency()
            self._residencies[device_index] = state
        return state

    def get(self, device_index: int | None) -> WholeCardResidency | None:
        """Return the residency state for ``device_index`` without creating one, or None when absent."""
        return self._residencies.get(device_index)

    def held(self) -> list[tuple[int | None, WholeCardResidency]]:
        """Return ``(device_index, state)`` for every card currently holding a residency (model set)."""
        return [(index, state) for index, state in self._residencies.items() if state.model is not None]

    def any_held(self) -> bool:
        """Return whether any card currently holds a whole-card residency."""
        return any(state.model is not None for state in self._residencies.values())

    def holder_for_model(self, model: str | None) -> tuple[bool, int | None]:
        """Return ``(found, device_index)`` for the card whose held residency is for ``model``.

        ``found`` distinguishes a genuine hit on the ``None`` (single-GPU / worker-wide) key from a miss,
        since ``None`` is itself a valid residency key.
        """
        if model is None:
            return (False, None)
        for device_index, state in self._residencies.items():
            if state.model == model:
                return (True, device_index)
        return (False, None)

    def record_grant(
        self,
        device_index: int | None,
        *,
        model: str | None,
        forecast: StreamForecast | None,
        cooldown_until: float,
        now: float,
        refresh_established: bool,
    ) -> WholeCardResidency:
        """Record a residency grant (an establishment or a RAM pre-stage) for ``device_index``.

        Sets the model, forecast, and cooldown; stamps ``established_at`` when ``refresh_established`` is
        set or the residency is fresh, so the recovery supervisor's grace window is measured from when the
        intentional hold began. Returns the updated state.
        """
        state = self.state_for(device_index)
        if refresh_established or state.established_at == 0.0:
            state.established_at = now
        state.model = model
        state.forecast = forecast
        state.cooldown_until = cooldown_until
        return state

    def phase(
        self,
        device_index: int | None,
        *,
        now: float,
        establish_grace_seconds: float,
    ) -> tuple[str | None, WholeCardPhase]:
        """Return ``(model, phase)`` for the residency held on ``device_index``.

        ``model`` is None (with phase ``NONE``) when the card holds no residency. Reads without creating:
        a card with no residency is left absent from the ledger.
        """
        state = self._residencies.get(device_index)
        if state is None or state.model is None:
            return None, WholeCardPhase.NONE
        establishing = state.established_at != 0.0 and (now - state.established_at) < establish_grace_seconds
        return state.model, (WholeCardPhase.ESTABLISHING if establishing else WholeCardPhase.HOLDING)

    def grace_active(
        self,
        *,
        now: float,
        establish_grace_seconds: float,
        restore_grace_seconds: float,
    ) -> bool:
        """Return whether any residency is establishing or restoring, so a held queue is intentional.

        Bounded by the two grace windows so a residency that genuinely never loads (or a restore that
        never completes) still trips the recovery supervisor.
        """
        for state in self._residencies.values():
            establishing = (
                state.model is not None
                and state.established_at != 0.0
                and (now - state.established_at) < establish_grace_seconds
            )
            restoring = state.restore_at != 0.0 and (now - state.restore_at) < restore_grace_seconds
            if establishing or restoring:
                return True
        return False

    def drain_backstop_elapsed(self, device_index: int | None, *, now: float, settle_seconds: float) -> bool:
        """Return whether the bounded drain-settle window has elapsed since this residency was established.

        The deterministic backstop for the dispatch gate: once a structurally-complete teardown has held
        for ``settle_seconds`` without the live free-VRAM reading confirming the drain, the head is
        admitted on the structural guarantee rather than parking forever.
        """
        state = self._residencies.get(device_index)
        if state is None or state.established_at == 0.0:
            return False
        return (now - state.established_at) >= settle_seconds


class WholeCardResidencyMachine(WholeCardResidencyLedger):
    """Whole-card residency state machine plus pure transition queries.

    The scheduler still executes side effects (process scale-down, safety cycling, VRAM eviction), but this
    class owns the multi-tick residency state and the policy questions that can be answered without touching
    live process objects. It extends :class:`WholeCardResidencyLedger` so existing adapter properties can be
    migrated incrementally without changing behavior.
    """

    def residency_demanded(
        self,
        forecast: StreamForecast,
        *,
        enabled: bool,
        is_head_blocker: bool,
    ) -> bool:
        """Return whether a job should enter the whole-card residency pipeline."""
        needs_teardown = forecast.needs_exclusive_residency or forecast.needs_process_count_reduction
        return enabled and needs_teardown and is_head_blocker

    def target_process_count(self, forecast: StreamForecast | None) -> int:
        """Return the live inference-process target for a held residency."""
        if forecast is None:
            return 1
        return forecast.max_resident_processes() or 1

    def teardown_complete(
        self,
        forecast: StreamForecast,
        *,
        loaded_process_count: int,
        safety_pause_required: bool,
        safety_paused: bool,
        post_process_pause_required: bool = False,
        post_process_cleared: bool = True,
        component_lane_pause_required: bool = False,
        component_lane_cleared: bool = True,
        weights_fit_live: bool,
        drain_backstop_elapsed: bool,
    ) -> bool:
        """Return whether a held residency has cleared enough room for the head to sample.

        The head must not be admitted until every VRAM consumer the residency displaces has actually vacated
        the card: the live inference-process count is at (or below) the forecast's target, safety is off-GPU if
        this residency needs it, and the dedicated post-processing and component lanes have left the card if
        they need to. A lane's context is only freed when its process exits, so ``post_process_cleared`` and
        ``component_lane_cleared`` are structural checks (the lane is gone), distinct from the pause merely
        having been requested; admitting the head while a lane's context is still resident is exactly what
        leaves too little room and streams the weights.
        """
        if loaded_process_count > self.target_process_count(forecast):
            return False
        if safety_pause_required and not safety_paused:
            return False
        if post_process_pause_required and not post_process_cleared:
            return False
        if component_lane_pause_required and not component_lane_cleared:
            return False
        if weights_fit_live:
            return True
        return forecast.fits_alone and drain_backstop_elapsed


def max_coresident_for_peak(
    *,
    total_vram_mb: float | None,
    per_process_overhead_mb: float,
    marginal_overhead_mb: float | None,
    peak_mb: float,
    reserve_mb: float,
) -> int | None:
    """Return the largest live inference-process count that still fits ``peak_mb`` plus ``reserve_mb``.

    The loader's first context costs the full one-time overhead; each additional co-resident context
    costs only the marginal (falling back to the full figure when unmeasured or zero). Returns None when
    the depth cannot be sized (no reported total VRAM, or non-positive overhead figures); never below
    one, since the job's own context always exists.
    """
    if total_vram_mb is None or per_process_overhead_mb <= 0:
        return None
    marginal = marginal_overhead_mb or per_process_overhead_mb
    if marginal <= 0:
        return None
    budget = total_vram_mb - peak_mb - reserve_mb
    if budget <= per_process_overhead_mb:
        return 1
    return max(1, 1 + int((budget - per_process_overhead_mb) // marginal))
