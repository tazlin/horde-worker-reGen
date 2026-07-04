"""Capacity-normalized wall-clock accounting for the worker's inference slots.

The GPU duty-cycle telemetry answers *whether* the device was busy; nothing answers *what the worker's
configured capacity was doing* over the same wall clock. A two-thread worker running one job is 100%
"busy" by the device's utilization counter while half its capacity idles, and the reason the second
slot stayed empty (an overlap headway, an exclusive admit, a deferred preload, or simply no queued
work) is scattered across throttled debug lines that only fire after multi-second parks.

This module owns that accounting as a pure accumulator: each scheduler tick contributes
``dt x capacity`` slot-seconds, split between ``SAMPLING`` (slots running a job) and exactly one
attribution bucket for the empty slots (what stopped the next dispatch, or ``NO_LOCAL_WORK`` when the
local queue had nothing to dispatch). Because every second of every slot lands in exactly one bucket,
the shares sum to 100% of capacity and "active vs idle vs gated" is a direct read, over any window, by
differencing two snapshots of the monotonically-growing totals. The accumulator holds no scheduler
references; the scheduler feeds it plain numbers, which keeps the arithmetic unit-testable.
"""

from __future__ import annotations

from strenum import StrEnum


class SlotDutyBucket(StrEnum):
    """Where one inference slot's wall clock went during one scheduler tick.

    ``SAMPLING`` is the productive bucket; every other bucket is an empty slot attributed to the gate
    (or supply state) that kept it empty. The empty-slot buckets mirror the dispatch-stall diagnosis
    branches (:meth:`InferenceScheduler._classify_dispatch_stall`) so the periodic attribution line,
    the stats stream, and the parked-head log text all name the same causes.
    """

    SAMPLING = "sampling"
    """The slot was running a dispatched job (inference in flight)."""
    NO_LOCAL_WORK = "no_local_work"
    """No queued job was waiting for the slot: supply-side (horde demand, pop governors, or empty queue)."""
    MODEL_LOADING = "model_loading"
    """The next job's model was preloading; the slot idles until the load lands."""
    PRELOAD_DEFERRED = "preload_deferred"
    """The next job's model was not resident and no preload had been admitted (a budget defer)."""
    WHOLE_CARD_RESERVED = "whole_card_reserved"
    """A whole-card residency held for a different model reserved the card the next job needed."""
    RESIDENT_SLOT_BUSY = "resident_slot_busy"
    """The next job's model was resident, but only on a process busy with other work."""
    KEEP_SINGLE_INFERENCE = "keep_single_inference"
    """The keep-single-inference guard held dispatch (batched job, ControlNet workflow, or the
    post-processing overlap rule)."""
    EXCLUSIVE_ISOLATION = "exclusive_isolation"
    """An exclusively-admitted over-budget job had the device; other dispatch was suppressed."""
    CONCURRENCY_CAP = "concurrency_cap"
    """The in-progress cap (as currently computed) was reached below the configured thread count."""
    OVERLAP_HEADWAY = "overlap_headway"
    """The overlap gate held the candidate until the in-flight work made enough progress."""
    WHOLE_CARD_CONVERGENCE = "whole_card_convergence"
    """A whole-card head waited for the pool to converge to sole residency."""
    UNEXPLAINED = "unexplained"
    """No gate claimed the empty slot: the scheduler-stall-shaped case worth reporting."""


class SlotDutyAccumulator:
    """Accumulates slot-seconds per :class:`SlotDutyBucket` from per-tick observations.

    ``observe`` is called once per scheduler tick with the live numbers; the elapsed time since the
    previous observation is attributed then. Totals only grow, so any window's breakdown is the
    difference of two snapshots; consumers (the periodic log line, the stats stream) difference their
    own anchors without this class tracking windows.
    """

    def __init__(self) -> None:
        """Start with empty totals; the first observation only anchors the clock."""
        self._totals: dict[str, float] = {}
        self._last_observation_time: float | None = None
        self._capacity: int = 0

    @property
    def capacity(self) -> int:
        """The configured slot count at the latest observation (0 until first observed)."""
        return self._capacity

    def observe(
        self,
        now: float,
        *,
        capacity: int,
        busy_slots: int,
        waiting_jobs: int,
        hold: SlotDutyBucket | None,
    ) -> None:
        """Attribute the wall clock since the previous observation across ``capacity`` slots.

        The interval is priced at this (closing) observation's state: with the control loop ticking
        sub-second, a state transition mis-prices at most one tick, which keeps the accumulator
        stateless about anything but totals and the clock.

        Args:
            now: Current monotonic-enough clock (``time.time()``); a non-advancing or backwards
                reading contributes nothing rather than corrupting totals.
            capacity: Configured concurrent-inference slot count this tick.
            busy_slots: Slots running a dispatched job (clamped into ``[0, capacity]``).
            waiting_jobs: Queued jobs not yet dispatched; 0 attributes every empty slot to
                ``NO_LOCAL_WORK`` regardless of ``hold``.
            hold: Why the next dispatch did not happen, when the scheduler knows; ``None`` with
                waiting work reads as ``UNEXPLAINED``.
        """
        previous = self._last_observation_time
        self._last_observation_time = now
        self._capacity = capacity
        if previous is None or now <= previous or capacity <= 0:
            return
        dt = now - previous

        busy = min(max(busy_slots, 0), capacity)
        if busy:
            self._totals[SlotDutyBucket.SAMPLING] = self._totals.get(SlotDutyBucket.SAMPLING, 0.0) + dt * busy

        empty = capacity - busy
        if empty <= 0:
            return
        if waiting_jobs <= 0:
            bucket: SlotDutyBucket = SlotDutyBucket.NO_LOCAL_WORK
        else:
            bucket = hold if hold is not None else SlotDutyBucket.UNEXPLAINED
        self._totals[bucket] = self._totals.get(bucket, 0.0) + dt * empty

    def totals(self) -> dict[str, float]:
        """A snapshot of cumulative slot-seconds per bucket (copies; safe to difference later)."""
        return dict(self._totals)

    @staticmethod
    def format_window(
        window_totals: dict[str, float],
        *,
        capacity: int,
        top_n: int = 5,
    ) -> str | None:
        """Render one window's slot-second breakdown as a compact, greppable attribution string.

        Shares are of the window's total slot-seconds (they sum to ~100% of capacity-time), largest
        first, ``SAMPLING`` always leading when present so the productive share reads at a glance.
        Returns None for an empty window.
        """
        total = sum(window_totals.values())
        if total <= 0:
            return None
        parts: list[str] = []
        sampling = window_totals.get(SlotDutyBucket.SAMPLING, 0.0)
        if sampling > 0:
            parts.append(f"sampling {sampling / total:.0%}")
        rest = sorted(
            ((k, v) for k, v in window_totals.items() if k != SlotDutyBucket.SAMPLING and v > 0),
            key=lambda kv: -kv[1],
        )
        parts.extend(f"{k} {v / total:.0%}" for k, v in rest[:top_n])
        return f"slot attribution (capacity {capacity}): " + ", ".join(parts)
