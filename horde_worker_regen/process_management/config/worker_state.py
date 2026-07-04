"""Mutable shared state container for cross-cutting worker flags.

All sub-components that need to read or write these flags receive a
reference to the same WorkerState instance, eliminating callback
lambdas and cross-component property writes.
"""

from __future__ import annotations

import dataclasses
import time
from collections import deque

from horde_worker_regen.process_management.models.lora_download_backoff import LoraDownloadBackoff


@dataclasses.dataclass
class WorkerState:
    """Cross-cutting mutable flags shared by all process-management sub-components."""

    shutting_down: bool = False
    shutting_down_time: float = 0.0
    shut_down: bool = False

    last_job_pop_time: float = 0.0
    last_pop_no_jobs_available: bool = False
    last_pop_maintenance_mode: bool = False
    server_maintenance_cleared_by_job_pop: bool = False
    """A real popped job proved the horde is sending work again, even if worker-details polling is stale."""
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

    ram_pressure_pop_hold: bool = False
    """Soft, pre-floor pop hold set while system RAM is *approaching* its danger floor (within the marginal
    RAM reserve of it) or while an over-ceiling process is being drained for reclaim. Distinct from the hard
    ``self_throttle_paused`` (which fires at/below the floor): this stops the popper starting a new job's ttl
    clock on work the degraded worker cannot promptly serve, so the job does not age past its ttl in-queue and
    get aborted by the horde as too slow. In-flight work is unaffected; cleared as soon as RAM recovers and no
    process is draining."""

    post_processing_disabled_by_breaker: bool = False
    """Session-latched: the post-processing fault breaker tripped on repeated unhostable post-processing peaks.

    A post-processing peak that cannot be hosted (a single-process worker on a tiny card, or a card a job
    over-commits) faults the job and, reaped, accumulates toward this breaker. While true the job popper stops
    advertising post-processing support (see the popper's ``pop_allow_post_processing``) so the worker is no
    longer handed upscale/face-fix jobs it cannot host, ending the fault->forced-maintenance spiral. The
    over-commit is structural, so this clears only on restart (auto-recovery would simply re-trip it) and is
    deliberately NOT cleared by a save-our-ship soft reset; the operator should downgrade settings."""

    post_processing_breaker_tripped_at: float = 0.0
    """Wall-clock time the post-processing breaker tripped; 0 when not tripped (for the operator advisory/TUI)."""

    post_processing_disabled_reason: str = ""
    """Operator-facing reason post-processing was session-disabled; empty when still enabled.

    The fault breaker and whole-card residency compatibility checks both use
    ``post_processing_disabled_by_breaker`` to stop advertising post-processing. This detail tells the TUI and
    logs which structural condition caused the session latch.
    """

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

    wants_line_skip_candidate: bool = False
    """An aux-model download has been blocking dispatch past the configured threshold with no suitable
    already-popped bypass job, so the line-skip in-progress cap may be bypassed to keep the GPU busy.
    Cleared by the line-skip cap gate as soon as no aux download exceeds the threshold."""

    gpu_torch_incompatible: bool = False
    """Session-latched: an inference child reported the installed PyTorch has no CUDA kernels for this GPU.

    Set from the child's ``TORCH_GPU_INCOMPATIBLE`` report (the child is the only process that touches
    torch; the parent learns of the mismatch through that torch-free signal). The wheel's compiled
    architectures do not include the device's compute capability, so every job would die at the first
    kernel launch. While true the job/alchemy poppers stop popping entirely (the GPU cannot serve any
    work) and the TUI surfaces the reason prominently. The mismatch is a build/hardware fact, so this
    never clears at runtime: it is fixed by reinstalling the matching backend (and restarting)."""

    gpu_torch_incompatible_reason: str = ""
    """Operator-facing explanation for ``gpu_torch_incompatible`` (the child's ``info`` string), relayed
    verbatim to the TUI. Empty until the flag latches."""

    torch_build_cpu_only: bool = False
    """Session-latched: an inference child reported the installed PyTorch is a CPU-only build.

    Set from the child's ``TORCH_BUILD_CPU_ONLY`` report. Unlike ``gpu_torch_incompatible`` nothing is
    broken: the build simply has no GPU backend, so image generation is disabled (CPU inference is
    impractically slow) while alchemy keeps running. While true the *image* job popper stops popping
    (the alchemy popper is unaffected). This is the runtime counterpart of the ``bin/backend`` 'cpu'
    sentinel: it makes a CPU torch build prevent image generation even when the sentinel was never set
    (e.g. a manual CPU install). A build fact, so it never clears at runtime; fixed by installing a GPU
    build (and restarting)."""

    torch_build_cpu_only_reason: str = ""
    """Operator-facing explanation for ``torch_build_cpu_only`` (the child's ``info`` string), relayed
    verbatim to the TUI. Empty until the flag latches."""

    consecutive_failed_jobs: int = 0
    too_many_consecutive_failed_jobs: bool = False
    too_many_consecutive_failed_jobs_time: float = 0.0
    # Must match CONSECUTIVE_FAILED_JOBS_WAIT_SECONDS in pop_throttler.py;
    # this field is used by the status reporter for display only.
    too_many_consecutive_failed_jobs_wait_time: float = 180

    kudos_generated_this_session: float = 0.0
    kudos_events: deque[tuple[float, float]] = dataclasses.field(default_factory=deque)

    first_kudos_event_time: float | None = None
    """Wall-clock of the first successful submit (image or alchemy), or None before any kudos land.

    The kudos/hr rate is undefined until this is set: a cold worker reads "warming up" rather than a
    misleadingly low number built on the process-start-to-first-job lead-in."""

    eligible_seconds_total: float = 0.0
    """Cumulative *productive* wall-seconds since :attr:`first_kudos_event_time`: time during which the
    worker held at least one job anywhere in its pipeline. This is the honest denominator for the kudos/hr
    rate.

    Time with an empty pipeline (server exhaustion, server maintenance, or an operator pause that has
    already drained the local queue) earns no kudos and is not counted, so the rate neither decays while
    idle nor charges the cold-start lead-in. A pause or maintenance while queued work is still draining *is*
    productive (kudos keep landing) and is counted, so the exclusion keys off actual pipeline occupancy
    rather than the pause/maintenance flags."""

    _eligible_last_tick_time: float | None = None
    """Internal marker: wall-clock of the last :meth:`tick_eligible_seconds` call, for delta accumulation."""

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
    ``recent_job_ttl``) to apply post-inference backpressure: stop popping while the safety backlog
    can no longer clear within the deadline, so the pipeline self-limits to its slowest stage instead
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

    def note_first_kudos_event(self, now: float) -> None:
        """Record the first successful submit, which starts the kudos/hr measurement window (idempotent)."""
        if self.first_kudos_event_time is None:
            self.first_kudos_event_time = now

    def tick_eligible_seconds(self, now: float, *, has_pipeline_work: bool, max_dt: float) -> None:
        """Advance the productive-time clock that denominates the kudos/hr rate; call once per control tick.

        Before the first submit the clock does not run (only the last-tick marker is kept fresh so the first
        productive interval is not overcounted). After it, wall-time since the previous tick is added only
        while the pipeline holds work, so idle, maintenance, and drained-pause stretches never inflate the
        denominator. ``max_dt`` clamps a stalled-loop gap so a long pause between ticks cannot dump a
        spurious burst of "productive" time into the total.
        """
        previous = self._eligible_last_tick_time
        self._eligible_last_tick_time = now
        if self.first_kudos_event_time is None or previous is None:
            return
        if has_pipeline_work:
            dt = now - previous
            if dt > 0:
                self.eligible_seconds_total += min(dt, max_dt)

    def initiate_shutdown(self) -> None:
        """Mark the worker as shutting down (idempotent)."""
        if not self.shutting_down:
            self.shutting_down = True
            self.shutting_down_time = time.time()

    def last_pop_recently(self) -> bool:
        """Return True if a job was popped within the last 10 seconds."""
        return (time.time() - self.last_job_pop_time) < 10
