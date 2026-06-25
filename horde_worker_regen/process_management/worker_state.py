"""Mutable shared state container for cross-cutting worker flags.

All sub-components that need to read or write these flags receive a
reference to the same WorkerState instance, eliminating callback
lambdas and cross-component property writes.
"""

from __future__ import annotations

import dataclasses
import time
from collections import deque

from horde_worker_regen.process_management.lora_download_backoff import LoraDownloadBackoff


@dataclasses.dataclass
class WorkerState:
    """Cross-cutting mutable flags shared by all process-management sub-components."""

    shutting_down: bool = False
    shutting_down_time: float = 0.0
    shut_down: bool = False

    last_job_pop_time: float = 0.0
    last_pop_no_jobs_available: bool = False
    last_pop_maintenance_mode: bool = False
    last_pop_skipped_reasons: dict[str, int] = dataclasses.field(default_factory=dict)
    """Why the last 'no job available' pop skipped work, per reason (models/nsfw/max_pixels/...).

    Surfaced to the TUI as a "why no work" breakdown so an operator can see a quiet worker is
    configured out of the available jobs (wrong models, too-low max_power, etc.) rather than idle.
    """

    supervisor_paused: bool = False
    """Local pause requested by a supervising frontend (TUI). Stops new job/alchemy pops; in-flight work finishes."""

    downloads_only_hold: bool = False
    """The worker is in a download-only posture (pre-fetch models without committing the GPU): the
    download process runs but inference/safety are held and no jobs are popped. Lifted by GO_LIVE. Kept
    separate from ``supervisor_paused`` so leaving the hold never clobbers an operator's manual pause."""

    self_throttle_paused: bool = False
    """Worker-initiated local pop-pause: the self-throttle backstop engaged because resource/OOM faults
    accumulated fast enough to risk the horde forcing the worker into maintenance. Stops new pops (in-flight
    work finishes) until the cooldown elapses. Kept separate from ``supervisor_paused`` so the worker's own
    throttle never clobbers (or is clobbered by) an operator's manual pause."""

    self_throttle_paused_until: float = 0.0
    """Wall-clock time the self-throttle pop-pause auto-resumes; 0 when not throttling."""

    lora_disk_exhausted: bool = False
    """The LoRA cache volume is below its free-space floor and eviction could not clear it.

    Set by the main loop's disk check after the ad-hoc cache has had a chance to evict to make room.
    While true the worker stops advertising LoRA support on job pops (see the job popper's
    ``_effective_allow_lora``) so it isn't handed jobs whose LoRAs it cannot download, and the TUI
    surfaces a prominent warning. Cleared automatically once free space recovers above the floor."""

    lora_download_backoff: LoraDownloadBackoff = dataclasses.field(default_factory=LoraDownloadBackoff)
    """Escalating suppression of LoRA pops after repeated ad-hoc download teardowns.

    The process lifecycle records a strike whenever it reaps an inference slot stuck downloading
    auxiliary models; while the resulting window is active the job popper stops advertising LoRA
    support (see ``_lora_disk_permits``) so the worker stops feeding jobs into a failing download
    path. Windows double per consecutive strike and reset after a healthy stretch."""

    consecutive_failed_jobs: int = 0
    too_many_consecutive_failed_jobs: bool = False
    too_many_consecutive_failed_jobs_time: float = 0.0
    # Must match CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS in pop_throttler.py;
    # this field is used by the status reporter for display only.
    too_many_consecutive_failed_jobs_wait_time: float = 180

    kudos_generated_this_session: float = 0.0
    kudos_events: deque[tuple[float, float]] = dataclasses.field(default_factory=deque)

    alchemy_forms_in_flight: int = 0
    """Alchemy forms currently anywhere in the pop->dispatch->submit pipeline.

    Maintained by the AlchemyCoordinator loop so other components (e.g. the job popper's
    no-jobs idle tracking) can treat active alchemy work as the worker being busy.
    """

    avg_safety_seconds: float = 0.0
    """Exponential moving average of the measured wall-clock per safety check (0 until first sample).

    The safety stage is a single (often CPU-bound) process downstream of inference, and nothing bounded
    its queue: when inference outruns safety the post-inference backlog grew until jobs aged past their
    horde ttl and were server-aborted as "too slow". The job popper reads this (with
    ``recent_job_ttl``) to apply post-inference backpressure -- stop popping while the safety backlog
    can no longer clear within the deadline -- so the pipeline self-limits to its slowest stage instead
    of spiralling into forced maintenance. See :meth:`record_safety_duration`."""

    recent_job_ttl: float | None = None
    """The most recent horde-supplied job ttl (seconds before the horde aborts a job as stale), or None.

    Captured on each successful pop; the popper uses it to size the post-inference backpressure budget to
    the actual deadline. Falls back to a conservative constant when the horde does not supply one."""

    def record_safety_duration(self, seconds: float) -> None:
        """Fold one measured safety-check wall-clock into the EMA used for post-inference backpressure."""
        if seconds <= 0:
            return
        alpha = 0.2
        if self.avg_safety_seconds <= 0:
            self.avg_safety_seconds = seconds
        else:
            self.avg_safety_seconds = (1 - alpha) * self.avg_safety_seconds + alpha * seconds

    def initiate_shutdown(self) -> None:
        """Mark the worker as shutting down (idempotent)."""
        if not self.shutting_down:
            self.shutting_down = True
            self.shutting_down_time = time.time()

    def last_pop_recently(self) -> bool:
        """Return True if a job was popped within the last 10 seconds."""
        return (time.time() - self.last_job_pop_time) < 10
