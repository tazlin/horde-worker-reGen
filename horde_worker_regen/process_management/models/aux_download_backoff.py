"""Exponential backoff that escalates after repeated ad-hoc auxiliary-download stalls.

When a worker's ad-hoc auxiliary downloads (LoRAs, textual inversions) keep stalling long enough for the
orchestrator to tear down and replace the inference slot, feeding more jobs into the same failing download
path only churns processes: each job sits minutes in the aux-download phase, faults, and burns a slot while
the GPU goes idle. This backoff reacts to those teardowns by doubling an escalation window on each successive
strike so a persistent upstream outage escalates from a brief pause to a long one, then resets once downloads
have been healthy for a while. While the window is active a fresh failure of the same class is classified as
terminal rather than retryable, so a job is not requeued into the same failing path.

One instance is held per auxiliary class (see ``WorkerState``). The LoRA instance additionally gates pop-time
LoRA advertising, because the pop request carries an ``allow_lora`` capability flag the worker can withhold.
The textual-inversion instance influences fault classification only: the pop request has no per-request
textual-inversion capability flag, so a textual-inversion window cannot suppress that traffic at the pop.

The state is a plain container mutated only from the single-threaded main-process scheduling loop (the job
popper reads it, the process lifecycle writes a strike when it reaps a stuck aux-download slot), so it needs
no locking; the same convention the other ``WorkerState`` flags follow.
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
class AuxDownloadBackoff:
    """Escalating, self-decaying reaction to ad-hoc auxiliary-download teardowns for one auxiliary class."""

    strikes: int = 0
    """Consecutive aux-download teardowns in the current incident; 0 when healthy."""

    suppressed_until: float = 0.0
    """Wall-clock time the active window ends; the escalation counts as in force while ``now`` is below it.

    For the LoRA instance this is also the time pop-time LoRA advertising may resume; for a class with no
    per-request capability flag it bounds fault classification only."""

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
        """Whether the active window still covers ``now`` (LoRA pop advertising withheld while true)."""
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
        """Seconds until the active window ends; 0 when not currently in force."""
        return max(0.0, self.suppressed_until - now)
