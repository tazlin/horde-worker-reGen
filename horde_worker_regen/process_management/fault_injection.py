"""Typed fault-injection profiles for the fake worker processes.

The fakes in :mod:`fake_worker_processes` already speak the real pipe/queue protocol; a
:class:`FaultProfile` tells one of them to *misbehave* in a specific, reproducible way so the
orchestration layer's crash/hang/resource handling can be exercised without a GPU or a real
failure. Profiles are plain pydantic models so they pickle cleanly across the spawn boundary
(inject them with ``functools.partial`` of a module-level entry point in
``fake_worker_processes``).

The goal is diagnostic: drive the real process manager, scheduler, safety orchestrator and job
tracker through hangs, crashes, dropped heartbeats, slowness, resource exhaustion and malformed
messages, and assert that the worker recovers (job eventually completes-or-faults, the slot is
replaced, no semaphore is orphaned, the worker keeps running).
"""

from __future__ import annotations

import enum

from pydantic import BaseModel


class FaultKind(enum.StrEnum):
    """The shapes of misbehaviour a fake process can be told to exhibit.

    Used to label an active fault (in diagnostics, in the faulted-result ``info`` string the
    OOM path emits, and in tests) so a scenario can refer to a fault symbolically rather than by
    re-deriving it from the profile's individual fields.
    """

    CRASH_ON_START = "crash_on_start"
    HANG = "hang"
    CRASH = "crash"
    STALL_IN_PRELOAD = "stall_in_preload"
    DROP_HEARTBEATS = "drop_heartbeats"
    SLOW = "slow"
    OOM = "oom"
    CORRUPT_MESSAGE = "corrupt_message"


# Tag prefix the fake stamps onto a faulted result's ``info`` so the main process (and tests) can
# recognise an injected resource failure. Phase 4's failure classifier keys on this in fake mode.
FAULT_INFO_PREFIX = "injected-fault:"


class FaultProfile(BaseModel):
    """A reproducible misbehaviour script for a single fake worker process.

    Every field is independent, so a profile can combine several faults. The ``*_on_job_n`` fields
    are 1-based job ordinals counted by the process itself (the nth job it is asked to run). The
    default profile (all fields falsy / unit) makes the fake behave exactly like a normal fake.
    """

    crash_on_start: bool = False
    """Hard-exit during process init, before the first job (simulates an import or CUDA-init failure)."""

    hang_after_n_jobs: int | None = None
    """After completing this many jobs, the next job is accepted but never finishes and emits no
    further heartbeats (simulates a wedged inference loop that the watchdog must time out)."""

    crash_on_job_n: int | None = None
    """On this job ordinal, hard-exit mid-job via ``os._exit`` (simulates a segfault or an OS OOM-kill)."""

    stall_in_preload: bool = False
    """Enter ``PRELOADING_MODEL`` and never report ``PRELOADED_MODEL`` (simulates a stuck model load)."""

    drop_heartbeats: bool = False
    """Never emit ``INFERENCE_STEP`` heartbeats during a job, so mid-inference stall detection has no signal
    and must fall back to a coarser timeout."""

    slow_factor: float = 1.0
    """Multiplies the per-job delay; ``> 1`` makes jobs run slower than their expected time."""

    oom_on_job_n: int | None = None
    """On this job ordinal, report a faulted result tagged as an out-of-memory failure instead of images."""

    corrupt_on_job_n: int | None = None
    """On this job ordinal, emit a misrouted/garbage message before the real result, exercising the
    dispatcher's tolerance of malformed or mismatched messages."""

    def is_noop(self) -> bool:
        """Return True if this profile requests no misbehaviour at all."""
        return self == FaultProfile()

    def active_kinds(self) -> set[FaultKind]:
        """Return the set of fault kinds this profile would exhibit (for diagnostics and assertions)."""
        kinds: set[FaultKind] = set()
        if self.crash_on_start:
            kinds.add(FaultKind.CRASH_ON_START)
        if self.hang_after_n_jobs is not None:
            kinds.add(FaultKind.HANG)
        if self.crash_on_job_n is not None:
            kinds.add(FaultKind.CRASH)
        if self.stall_in_preload:
            kinds.add(FaultKind.STALL_IN_PRELOAD)
        if self.drop_heartbeats:
            kinds.add(FaultKind.DROP_HEARTBEATS)
        if self.slow_factor != 1.0:
            kinds.add(FaultKind.SLOW)
        if self.oom_on_job_n is not None:
            kinds.add(FaultKind.OOM)
        if self.corrupt_on_job_n is not None:
            kinds.add(FaultKind.CORRUPT_MESSAGE)
        return kinds
