"""A single accounting of the pop/scheduling *governors* that gate when the worker takes work.

Many independent conditions can hold back a job pop or reshape it: a whole-card residency reserving the
device, the large-model switch throttle and re-entry cooldown, post-inference backpressure, a model held
back as locally unservable, the consecutive-failure pause, pop error-backoff, a LoRA download backoff, model
stickiness, the megapixelstep wait, and the worker's own self-throttle. Each lives in its own subsystem and
was, until now, only visible (if at all) as a one-off log line, so an operator watching the dashboard could
not tell *which* governor was holding the worker back or *for how long*.

This registry is the one place that turns those scattered conditions into an observable, comparable shape.
Each scheduling cycle the worker reports every governor's current state as a :class:`PopGovernorReading`; the
registry tracks, per governor, the *spell* it is currently in (when it started, why, how much longer it is
expected to last) plus session totals (how many times it has engaged and how long it has held in aggregate).
It emits a grep-friendly ``ENTER``/``EXIT`` log line at each spell boundary (the contract the log-triage and
duty-cycle tooling parse) and exposes a snapshot the TUI renders live.

The registry is pure beyond its tracking state and its injected logger: the caller supplies ``now`` and the
readings, so it has no clock or I/O of its own and is fully table-testable. All callers run on the single
event-loop thread, so no locking.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class PopGovernorReading:
    """One governor's current state, reported once per scheduling cycle.

    A reading is a *level* (is the governor engaged right now), not an edge: the registry derives the
    enter/exit edges by comparing successive readings, so a caller only has to answer "is this condition
    holding, why, and for how much longer" each cycle.
    """

    name: str
    """Stable machine key (snake_case), e.g. ``large_model_switch``. Identifies the governor across cycles."""
    label: str
    """Short human-friendly name for the dashboard, e.g. ``Large-model switch throttle``."""
    active: bool
    """Whether the governor is engaged (holding back or reshaping pops) right now."""
    reason: str | None = None
    """A short human-readable cause for the current engagement, or None."""
    expected_remaining_seconds: float | None = None
    """Best estimate of seconds until the governor releases, or None when it has no fixed timer (a condition
    that clears when the underlying state changes rather than on a clock)."""


@dataclass
class _GovernorSpell:
    """Mutable per-governor tracking: the current spell plus session aggregates."""

    label: str
    reason: str | None = None
    expected_remaining_seconds: float | None = None
    spell_started_at: float | None = None
    """When the current active spell began, or None when the governor is idle."""
    triggers: int = 0
    """How many spells (engagements) this governor has had this session."""
    closed_total_seconds: float = 0.0
    """Aggregate seconds of completed spells; the live spell is added on top at snapshot time."""

    @property
    def active(self) -> bool:
        """Whether a spell is currently open."""
        return self.spell_started_at is not None


@dataclass(frozen=True)
class GovernorSpellView:
    """An immutable view of one governor's tracked state, for projection onto the wire model."""

    name: str
    label: str
    active: bool
    reason: str | None
    expected_remaining_seconds: float | None
    current_spell_seconds: float
    triggers: int
    total_active_seconds: float
    fraction_of_session: float


def _format_seconds(seconds: float) -> str:
    """Format a duration compactly for a log line (e.g. ``45s``, ``3m12s``)."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, rem = divmod(int(seconds), 60)
    return f"{minutes}m{rem:02d}s"


class PopGovernorRegistry:
    """Tracks the live spell and session aggregates of every pop/scheduling governor.

    Fed one batch of :class:`PopGovernorReading` per scheduling cycle via :meth:`update`; emits ``ENTER`` /
    ``EXIT`` log lines at spell boundaries and exposes :meth:`views` for the status snapshot.
    """

    def __init__(self, *, log: Callable[[str], None] | None = None) -> None:
        """Initialize empty. ``log`` receives the grep-friendly boundary lines (defaults to a loguru info)."""
        self._spells: dict[str, _GovernorSpell] = {}
        self._log = log if log is not None else _default_log

    def update(self, readings: Iterable[PopGovernorReading], *, now: float) -> None:
        """Fold one cycle's readings in, opening/closing spells and logging the boundaries.

        A governor that reports ``active`` for the first time (or after being idle) opens a spell: its trigger
        count increments and an ``ENTER`` line is logged. One that was active and now reads inactive closes its
        spell: the elapsed time is banked into the session total and an ``EXIT`` line is logged. While a spell
        stays open the reason and expected-remaining are refreshed from the latest reading.
        """
        for reading in readings:
            spell = self._spells.get(reading.name)
            if spell is None:
                spell = _GovernorSpell(label=reading.label)
                self._spells[reading.name] = spell
            spell.label = reading.label

            if reading.active and not spell.active:
                spell.spell_started_at = now
                spell.triggers += 1
                spell.reason = reading.reason
                spell.expected_remaining_seconds = reading.expected_remaining_seconds
                self._log(self._enter_line(reading))
            elif reading.active and spell.active:
                spell.reason = reading.reason
                spell.expected_remaining_seconds = reading.expected_remaining_seconds
            elif (not reading.active) and spell.active:
                assert spell.spell_started_at is not None
                spell_seconds = max(0.0, now - spell.spell_started_at)
                spell.closed_total_seconds += spell_seconds
                self._log(self._exit_line(reading.name, spell, spell_seconds))
                spell.spell_started_at = None
                spell.reason = None
                spell.expected_remaining_seconds = None

    def views(self, *, now: float, session_elapsed_seconds: float) -> list[GovernorSpellView]:
        """Project every tracked governor onto an immutable view list, newest-active first.

        ``fraction_of_session`` is the governor's aggregate active time over the session so far (0 when the
        session length is unknown/zero). Governors that have never engaged are omitted: the snapshot shows only
        governors with history or a live spell, so the dashboard is not cluttered by always-idle ones.
        """
        views: list[GovernorSpellView] = []
        for name, spell in self._spells.items():
            current_spell_seconds = (now - spell.spell_started_at) if spell.active and spell.spell_started_at else 0.0
            total = spell.closed_total_seconds + current_spell_seconds
            if spell.triggers == 0 and not spell.active:
                continue
            fraction = (total / session_elapsed_seconds) if session_elapsed_seconds > 0 else 0.0
            views.append(
                GovernorSpellView(
                    name=name,
                    label=spell.label,
                    active=spell.active,
                    reason=spell.reason,
                    expected_remaining_seconds=spell.expected_remaining_seconds,
                    current_spell_seconds=current_spell_seconds,
                    triggers=spell.triggers,
                    total_active_seconds=total,
                    fraction_of_session=min(1.0, max(0.0, fraction)),
                ),
            )
        # Active governors first (longest-running first), then idle ones by total time.
        views.sort(key=lambda v: (not v.active, -v.current_spell_seconds, -v.total_active_seconds))
        return views

    def _enter_line(self, reading: PopGovernorReading) -> str:
        reason = f" ({reading.reason})" if reading.reason else ""
        eta = (
            f"; expected ~{_format_seconds(reading.expected_remaining_seconds)}"
            if reading.expected_remaining_seconds is not None
            else ""
        )
        return f"Pop governor ENTER: {reading.name}{reason}{eta}"

    def _exit_line(self, name: str, spell: _GovernorSpell, spell_seconds: float) -> str:
        total = spell.closed_total_seconds
        return (
            f"Pop governor EXIT: {name} after {_format_seconds(spell_seconds)} "
            f"({spell.triggers}x this session, {_format_seconds(total)} total)"
        )


def _default_log(line: str) -> None:
    """Default sink for boundary lines: a loguru info, imported lazily so the module stays import-light."""
    from loguru import logger

    logger.info(line)
