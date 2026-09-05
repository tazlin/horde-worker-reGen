"""Coalescing for diagnostics that a control loop would otherwise re-emit every tick.

A loop that re-evaluates the same condition several times a second produces a log line per tick if each
evaluation logs. The throttle keeps the lines that carry information: the first observation, every change in
the observed state, and a periodic restatement of an unchanged state with a count of the repeats it absorbed.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable

DIAGNOSTIC_REPEAT_SECONDS = 30.0
"""How long an unchanged diagnostic stays suppressed before it is restated with its repeat count."""

DIAGNOSTIC_MB_BUCKET = 256.0
"""Bucket width for memory figures folded into a throttle key, so measurement jitter is not a state change."""


class DiagnosticThrottle:
    """Decide, per named diagnostic, whether this observation should be emitted.

    Each diagnostic is tracked under its own name with the state key it last emitted, when, and how many
    unchanged repeats have been swallowed since.
    """

    def __init__(self, clock: Callable[[], float], *, repeat_seconds: float = DIAGNOSTIC_REPEAT_SECONDS) -> None:
        """Track diagnostics against ``clock`` and restate an unchanged one every ``repeat_seconds``."""
        self._clock = clock
        self._repeat_seconds = repeat_seconds
        self._state: dict[str, tuple[Hashable, float, int]] = {}

    def suppressed_count(self, name: str, state_key: Hashable) -> int | None:
        """Return how many repeats were suppressed when ``name`` should emit now, or ``None`` to stay quiet.

        The first observation emits with a count of zero. A changed ``state_key`` emits immediately, carrying
        the count of unchanged repeats swallowed before the change. An unchanged key emits again only once the
        repeat interval has elapsed.
        """
        now = self._clock()
        previous = self._state.get(name)
        if previous is None:
            self._state[name] = (state_key, now, 0)
            return 0

        previous_key, previous_emit, suppressed = previous
        if previous_key != state_key or (now - previous_emit) >= self._repeat_seconds:
            self._state[name] = (state_key, now, 0)
            return suppressed

        self._state[name] = (previous_key, previous_emit, suppressed + 1)
        return None

    def reset(self) -> None:
        """Forget every diagnostic so each next observation emits as a first observation."""
        self._state.clear()

    def forget(self, name: str) -> None:
        """Drop the record for ``name`` so its next observation emits as a first observation."""
        self._state.pop(name, None)


def suppressed_suffix(suppressed_count: int) -> str:
    """Render the trailing note for a diagnostic that absorbed unchanged repeats; empty when none were."""
    if suppressed_count <= 0:
        return ""
    return f" (suppressed {suppressed_count} unchanged repeats)"


def diagnostic_mb_bucket(value: float | None) -> int | None:
    """Bucket a memory figure for use in a throttle key."""
    if value is None:
        return None
    return round(value / DIAGNOSTIC_MB_BUCKET)
