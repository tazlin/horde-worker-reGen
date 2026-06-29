"""Exponential backoff that withholds LoRA job pops after repeated auxiliary-download stalls.

When a worker's ad-hoc LoRA downloads keep stalling long enough for the orchestrator to tear down and
replace the inference slot, popping more LoRA jobs only feeds the same failing download path: each job
sits minutes in the aux-download phase, faults, and churns a process while the GPU goes idle. This
backoff reacts to those teardowns by temporarily withholding LoRA support from job pops, doubling the
withholding window on each successive strike so a persistent upstream outage escalates from a brief
pause to a long one, then resetting once downloads have been healthy for a while.

The state is a plain container mutated only from the single-threaded main-process scheduling loop (the
job popper reads it, the process lifecycle writes a strike when it reaps a stuck aux-download slot), so
it needs no locking; the same convention the other ``WorkerState`` flags follow.
"""

from __future__ import annotations

import dataclasses

BASE_BACKOFF_SECONDS = 60.0
"""Withholding window applied on the first strike. Each further strike doubles it."""

BACKOFF_MULTIPLIER = 2.0
"""Factor the window grows by per consecutive strike."""

MAX_BACKOFF_SECONDS = 1800.0
"""Ceiling on the withholding window; a sustained outage holds here rather than growing unbounded."""

STRIKE_DECAY_SECONDS = 900.0
"""Quiet period after the last strike that returns the escalation to zero.

A new strike arriving after this much trouble-free time is treated as the start of a fresh incident
(window back to ``BASE_BACKOFF_SECONDS``) rather than a continuation of the previous escalation."""


@dataclasses.dataclass
class LoraDownloadBackoff:
    """Escalating, self-decaying suppression of LoRA pops driven by aux-download teardowns."""

    strikes: int = 0
    """Consecutive aux-download teardowns in the current incident; 0 when healthy."""

    suppressed_until: float = 0.0
    """Wall-clock time LoRA pops may resume; pops are withheld while ``now`` is below it."""

    last_strike_at: float = 0.0
    """Wall-clock time of the most recent strike, used to decide when an incident has ended."""

    def register_timeout(self, now: float) -> float:
        """Record an aux-download teardown and (re)arm the withholding window.

        Returns the window length (seconds) now in force, for logging.
        """
        # A strike well after the previous one is a new incident, not an escalation of the old one:
        # reset so a fresh stall starts from the base window rather than an inherited long one.
        if self.strikes > 0 and (now - self.last_strike_at) > STRIKE_DECAY_SECONDS:
            self.strikes = 0

        self.strikes += 1
        window = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * (BACKOFF_MULTIPLIER ** (self.strikes - 1)))
        self.suppressed_until = now + window
        self.last_strike_at = now
        return window

    def pops_suppressed(self, now: float) -> bool:
        """Whether LoRA pops are currently being withheld."""
        return now < self.suppressed_until

    def is_escalation_active(self, now: float) -> bool:
        """Whether an incident is currently in force (drives the fast-fault aux-download timeout).

        True while pops are suppressed or a strike is still recent (within the decay window). It
        self-expires by time, so a recovered incident reverts the fast-fault grace to normal without
        needing the strike count to be cleared; a later strike after the quiet period starts fresh.
        """
        if self.strikes <= 0:
            return False
        return self.pops_suppressed(now) or (now - self.last_strike_at) <= STRIKE_DECAY_SECONDS

    def remaining_seconds(self, now: float) -> float:
        """Seconds until LoRA pops may resume; 0 when not suppressed."""
        return max(0.0, self.suppressed_until - now)
