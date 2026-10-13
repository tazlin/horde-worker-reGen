"""The wait a safety check spends for room on its card before it runs.

An on-GPU safety process needs more device memory for the length of an evaluation than it holds at rest. A check
whose growth does not fit the card's measured room waits, and while it waits its booking keeps new work out of
that room, so the card drains to it as running jobs finish. This module owns how long each check has waited and
the session's counts. The scheduler decides the verdicts and owns the booking.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterable

SAFETY_CHECK_CPU_SECONDS_SEED = 3.0
"""What a safety check on the CPU is assumed to cost (seconds) until this worker has measured one.

A waiting check is sent to the CPU once it has waited as long as a CPU check takes, because past that point the
wait costs more than the fallback. The measured figure replaces this as soon as one check has run on the CPU."""


class SafetyCheckWindow(enum.Enum):
    """What the card offers the safety check at the head of the queue."""

    OPEN = "open"
    """The check's growth fits and is booked, or nothing prices it. Send the check to the device."""
    WAIT = "wait"
    """The growth does not fit yet. Its booking stands and the check stays queued."""
    CPU = "cpu"
    """The check waited as long as a CPU check takes. Send it to run on the CPU."""


class SafetyCheckWindowLedger:
    """How long each queued safety check has waited for room, and the session's window counts."""

    def __init__(self, clock: Callable[[], float]) -> None:
        """Track waits against ``clock``."""
        self._clock = clock
        self.waiting_since: dict[str, float] = {}
        """When each check was first found not to fit, keyed by job id."""
        self.opened_after_wait = 0
        """Checks that waited and then ran on the device."""
        self.sent_to_cpu = 0
        """Checks that waited out their patience and ran on the CPU."""
        self.waited_seconds = 0.0
        """Cumulative seconds finished checks spent waiting."""

    def waited(self, job_id: str) -> float:
        """Count ``job_id`` as waiting and return how long it has waited so far."""
        since = self.waiting_since.setdefault(job_id, self._clock())
        return max(0.0, self._clock() - since)

    def resolve(self, job_id: str, *, on_cpu: bool) -> float | None:
        """Close ``job_id``'s wait, returning its length, or None when it never waited."""
        since = self.waiting_since.pop(job_id, None)
        if since is None:
            return None
        waited = max(0.0, self._clock() - since)
        self.waited_seconds += waited
        if on_cpu:
            self.sent_to_cpu += 1
        else:
            self.opened_after_wait += 1
        return waited

    def prune(self, live_job_ids: Iterable[str]) -> None:
        """Forget waits for jobs that left the safety queue by another path."""
        live = set(live_job_ids)
        for job_id in [job_id for job_id in self.waiting_since if job_id not in live]:
            del self.waiting_since[job_id]
