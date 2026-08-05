"""The declared release path of every gate that can hold or defer the worker's work.

A wedge is a gate that engages and never releases while every backstop reads the situation as legitimate.
The gates themselves are correct in isolation: each one decides its own question well. What is missing is a
single statement, per gate, of *how work gets out again* and *within what*. Without it, a new hold can be
added correctly and still wedge the worker, because nothing ever asked whether its release condition is
reachable from the states that engage it.

This module is that statement. Every gate declares:

* which enumerable runtime surface names it (so the set of gates is closed, not a matter of reading code);
* what engages it;
* what releases it, as an event or a bound rather than a hope;
* the bound: seconds to resolution, or the named backstop that escalates if resolution does not come;
* where an engagement is observable at runtime, so an operator can confirm which gate is standing.

It is pure data. Nothing on a control-loop path consults it: the scheduler makes its decisions exactly as
before, and this file adds no per-tick work. The consumers are the guardrail test (which fails when a gate
appears on one of the runtime surfaces without an entry here) and the hostile per-gate suite (which attacks
each declaration: that the release is reachable, that no state composed from a request's own footprint can
defer it permanently, and that the bound holds under a fake clock).

Placement note: gates span job popping, preload admission, dispatch scheduling, whole-card governance and
process lifecycle. Filing this under any one of those packages would read as that subsystem owning the
others' release paths, which is the opposite of the point. It sits under ``liveness`` because liveness is
the property it exists to protect, and no subsystem owns it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "GATE_REGISTRY",
    "GateEntry",
    "GateKind",
    "GateSurface",
    "entry_for",
    "registered_keys",
]


class GateSurface(StrEnum):
    """The enumerable runtime surface a gate's identity comes from.

    Each surface is a closed set in production code (a ``StrEnum``, or an enum-keyed decision), so the
    guardrail can enumerate it and demand an entry per member. ``SCHEDULER_BUDGET`` is the exception and
    carries the gates that hold work without stamping a name anywhere a reader can enumerate; those are
    listed by hand and the guardrail cannot catch a new one, which is itself worth knowing.
    """

    PRELOAD_ADMISSION = "preload_admission"
    """A member of ``AdmissionDecision``: the preload pass's verdict on one pending job."""
    POP_GATE = "pop_gate"
    """A member of ``PopGate``: the condition that ended a job-pop cycle before it reached the horde."""
    WHOLE_CARD_GOVERNOR = "whole_card_governor"
    """A member of ``WholeCardGovernor``: a churn brake on opening a new whole-card residency."""
    DISPATCH_STALL = "dispatch_stall"
    """A member of ``SlotDutyBucket``: what an empty inference slot's wall clock was attributed to."""
    LANE_PAUSE = "lane_pause"
    """A member of ``PauseOwner``: which subsystem moved a GPU-bearing service process off its card."""
    POP_PAUSE = "pop_pause"
    """A member of ``PopPauseOwner``: which backstop armed the worker's self-throttle pop pause."""
    RECOVERY_PARK = "recovery_park"
    """A member of ``RecoveryParkReason``: which exhausted escalation parked the worker."""
    SCHEDULER_BUDGET = "scheduler_budget"
    """A scheduler-side budget that holds work without naming itself on an enumerable surface."""


class GateKind(StrEnum):
    """Whether an entry holds work or concludes it."""

    HOLD = "hold"
    """The gate defers work that is still expected to be served, so it must declare a release and a bound."""
    OUTCOME = "outcome"
    """The entry is a terminal verdict, not a hold: the work proceeds, is faulted, or is reissued. It still
    takes an entry, because it is a member of a surface the guardrail enumerates, and it declares why it
    cannot wedge."""


@dataclass(frozen=True)
class GateEntry:
    """One gate's declared identity, release path and bound."""

    key: str
    """The value the gate takes on its surface (the ``StrEnum`` member's value, or the hand-listed name)."""
    surface: GateSurface
    """Which enumerable runtime surface :attr:`key` belongs to."""
    kind: GateKind
    """Whether this defers work (and so owes a release path) or concludes it."""
    subsystem: str
    """The module that owns the decision, as an import path fragment."""
    engaged_by: str
    """The condition that puts the gate in force."""
    released_by: str
    """The event or bound that takes it back out of force. For an ``OUTCOME``, why it cannot hold work."""
    bound_seconds: float | None
    """Seconds from engagement to resolution or escalation, when a constant fixes it; None when the release
    is an event rather than a clock and the bound is carried by :attr:`backstop`."""
    bound_source: str
    """The constant or rule that fixes :attr:`bound_seconds`, or how the bound is derived when it is not a
    single constant. Empty when the gate has no clock of its own."""
    backstop: str
    """The named surface that escalates if the release does not come. Empty only when :attr:`bound_seconds`
    is set and the gate's own clock is the whole story."""
    observable_at: str
    """Where an engagement is visible at runtime: the stamp, decision record, or log surface to read."""


def registered_keys(surface: GateSurface) -> frozenset[str]:
    """Return every key registered against ``surface``."""
    return frozenset(entry.key for entry in GATE_REGISTRY if entry.surface is surface)


def entry_for(surface: GateSurface, key: str) -> GateEntry | None:
    """Return the entry registered for ``key`` on ``surface``, or None when it is unregistered."""
    for entry in GATE_REGISTRY:
        if entry.surface is surface and entry.key == key:
            return entry
    return None


_POP_GATE_WEDGE_BACKSTOP = (
    "WorkerRecoveryCoordinator.pop_gate_held_wedge_active (POP_GATE_HELD_WEDGE_SECONDS), with "
    "HordeWorkerProcessManager._check_pop_liveness disclosing the hold first"
)
"""The escalation that covers a pop gate which stands with no pop attempt reaching the horde."""

_FULL_QUEUE_BACKSTOP = (
    "HordeWorkerProcessManager._check_full_queue_liveness (POP_LIVENESS_FROZEN_QUEUE_SECONDS), then "
    + _POP_GATE_WEDGE_BACKSTOP
)
"""The escalation that separates a queue at its configured depth from one that has stopped moving."""

_DISPATCH_STALL_OBSERVABLE = (
    "InferenceScheduler._classify_dispatch_stall's bucket, accumulated by SlotDutyAccumulator and printed "
    "in the periodic slot-attribution line; the text half feeds the throttled parked-head log line"
)
"""Where a dispatch-stall attribution is readable at runtime."""

_PRELOAD_ADMISSION_OBSERVABLE = (
    "the AdmissionResult recorded as InferenceScheduler._last_preload_admission, quoted by the parked-head "
    "line when it names the head's model"
)
"""Where a preload-admission verdict is readable at runtime."""

_GIVE_UP_BACKSTOP = (
    "WorkerRecoveryCoordinator's give-up assessor, which faults and reissues accepted work the pool cannot "
    "serve rather than holding it past its ttl"
)
"""The escalation of last resort for accepted work that no gate will release."""


GATE_REGISTRY: tuple[GateEntry, ...] = (
    # region preload admission
    GateEntry(
        key="admit",
        surface=GateSurface.PRELOAD_ADMISSION,
        kind=GateKind.OUTCOME,
        subsystem="process_management.scheduling.governance.preload_admission",
        engaged_by="every admission stage cleared the job for a preload on the selected slot",
        released_by="not a hold: the preload is sent on this pass",
        bound_seconds=None,
        bound_source="",
        backstop="",
        observable_at=_PRELOAD_ADMISSION_OBSERVABLE,
    ),
    GateEntry(
        key="next_job",
        surface=GateSurface.PRELOAD_ADMISSION,
        kind=GateKind.HOLD,
        subsystem="process_management.scheduling.governance.preload_admission",
        engaged_by="this pending job needs nothing from the preload pass, or its own attempt is settled",
        released_by=(
            "the next preload pass re-asks for this job from its current state; the skip is per-pass and "
            "carries no latch into the next one"
        ),
        bound_seconds=None,
        bound_source="one scheduling cycle",
        backstop=_GIVE_UP_BACKSTOP,
        observable_at=_PRELOAD_ADMISSION_OBSERVABLE,
    ),
    GateEntry(
        key="stop_pass",
        surface=GateSurface.PRELOAD_ADMISSION,
        kind=GateKind.HOLD,
        subsystem="process_management.scheduling.governance.preload_admission",
        engaged_by="a condition that applies to the whole pass, so scanning further pending jobs is pointless",
        released_by=(
            "the next scheduling cycle opens a fresh pass; the pass-scoped stop holds no state across cycles"
        ),
        bound_seconds=None,
        bound_source="one scheduling cycle",
        backstop=_GIVE_UP_BACKSTOP,
        observable_at=_PRELOAD_ADMISSION_OBSERVABLE,
    ),
    GateEntry(
        key="quarantined",
        surface=GateSurface.PRELOAD_ADMISSION,
        kind=GateKind.OUTCOME,
        subsystem="process_management.scheduling.governance.preload_admission",
        engaged_by="the job's model is quarantined out of rotation",
        released_by="not a hold: the job is faulted for reissue rather than kept",
        bound_seconds=None,
        bound_source="",
        backstop="",
        observable_at=_PRELOAD_ADMISSION_OBSERVABLE,
    ),
    GateEntry(
        key="unserviceable",
        surface=GateSurface.PRELOAD_ADMISSION,
        kind=GateKind.OUTCOME,
        subsystem="process_management.scheduling.governance.preload_admission",
        engaged_by="the model's minimum device footprint fits no serving card on this host",
        released_by="not a hold: the job is faulted for reissue, since no amount of waiting makes it fit",
        bound_seconds=None,
        bound_source="",
        backstop="",
        observable_at=_PRELOAD_ADMISSION_OBSERVABLE,
    ),
    GateEntry(
        key="already_loaded",
        surface=GateSurface.PRELOAD_ADMISSION,
        kind=GateKind.OUTCOME,
        subsystem="process_management.scheduling.governance.preload_admission",
        engaged_by="the model is resident or already loading, so no preload is needed",
        released_by="not a hold: the job proceeds to dispatch on the residency it already has",
        bound_seconds=None,
        bound_source="",
        backstop="",
        observable_at=_PRELOAD_ADMISSION_OBSERVABLE,
    ),
    GateEntry(
        key="defer_ram_pressure",
        surface=GateSurface.PRELOAD_ADMISSION,
        kind=GateKind.HOLD,
        subsystem="process_management.scheduling.governance.ram_governor",
        engaged_by="host RAM is at its absolute floor, so another resident model would push the host into swap",
        released_by=(
            "system RAM returning above the floor, whether by the reclaim ladder freeing resident models, a "
            "draining process exiting, or foreign pressure easing"
        ),
        bound_seconds=None,
        bound_source="",
        backstop=(
            "the RAM reclaim ladder, which escalates from idle-VRAM eviction to cycling an allocator-stuck "
            "slot and finally admits the head best-effort rather than parking it"
        ),
        observable_at=_PRELOAD_ADMISSION_OBSERVABLE,
    ),
    GateEntry(
        key="defer_vram_growth_hold",
        surface=GateSurface.PRELOAD_ADMISSION,
        kind=GateKind.HOLD,
        subsystem="process_management.resources.vram_arbiter",
        engaged_by=(
            "device-level free VRAM on the target card is under the soft floor, so bringing another model to "
            "it would grow a footprint already near the paging cliff"
        ),
        released_by="the card's measured free VRAM recovering, by reclaim or by foreign pressure easing",
        bound_seconds=None,
        bound_source="",
        backstop=(
            "the reclaim ladder's measured-attempt escape hatch, which admits against device truth once its "
            "rungs are spent rather than deferring on the same reading indefinitely"
        ),
        observable_at=_PRELOAD_ADMISSION_OBSERVABLE,
    ),
    GateEntry(
        key="exclusive_in_progress",
        surface=GateSurface.PRELOAD_ADMISSION,
        kind=GateKind.HOLD,
        subsystem="process_management.scheduling.inference_scheduler",
        engaged_by="an exclusively-admitted over-budget job holds the card, suppressing unrelated staging",
        released_by="that job completing or faulting, which drops its exclusive claim",
        bound_seconds=None,
        bound_source="the in-flight job's own duration",
        backstop=(
            "the inference step-progress watchdog, which faults a sampler that stops making progress, so the "
            "exclusive claim cannot outlive a job that has stopped running"
        ),
        observable_at="the exclusive_isolation slot-duty bucket, plus JobTracker.is_admitted_exclusive",
    ),
    GateEntry(
        key="no_target",
        surface=GateSurface.PRELOAD_ADMISSION,
        kind=GateKind.HOLD,
        subsystem="process_management.scheduling.governance.preload_admission",
        engaged_by="every inference slot is busy, protected by affinity, draining, or holding a queued model",
        released_by=(
            "a slot becoming free, or the head-room selection freeing one: a starved head overrides the "
            "affinity and queued-model guards precisely so this cannot be permanent"
        ),
        bound_seconds=None,
        bound_source="",
        backstop=_GIVE_UP_BACKSTOP,
        observable_at=_PRELOAD_ADMISSION_OBSERVABLE,
    ),
    GateEntry(
        key="replace_process",
        surface=GateSurface.PRELOAD_ADMISSION,
        kind=GateKind.HOLD,
        subsystem="process_management.lifecycle.process_lifecycle",
        engaged_by="the chosen slot must be cycled before its model change can be attempted again",
        released_by="the replacement process coming up and accepting the preload on a later pass",
        bound_seconds=None,
        bound_source="",
        backstop=(
            "the crash-loop detector, which quarantines a slot whose replacements exceed "
            "CRASH_LOOP_MAX_REPLACEMENTS inside CRASH_LOOP_WINDOW_SECONDS rather than cycling it forever"
        ),
        observable_at="the action ledger's process replacement events, and the slot's launch identifier",
    ),
    GateEntry(
        key="defer_concurrency",
        surface=GateSurface.PRELOAD_ADMISSION,
        kind=GateKind.HOLD,
        subsystem="process_management.scheduling.governance.preload_admission",
        engaged_by=(
            "a model load is already in flight on this device, so a second would stack their disk-read and "
            "allocation spikes"
        ),
        released_by="the in-flight load finishing, faulting, or its process dying, any of which drops the count",
        bound_seconds=None,
        bound_source="the in-flight load's own duration",
        backstop=(
            "the preload-death watchdog, which clears a loading entry whose process is gone, so the count "
            "cannot be held by a load that will never conclude"
        ),
        observable_at=_PRELOAD_ADMISSION_OBSERVABLE,
    ),
    GateEntry(
        key="defer_budget",
        surface=GateSurface.PRELOAD_ADMISSION,
        kind=GateKind.HOLD,
        subsystem="process_management.resources.vram_arbiter",
        engaged_by="the resource budget or the reclamation ladder declined this preload against measured truth",
        released_by=(
            "room appearing on the card, whether by an in-flight job finishing, an eviction, or the ladder's "
            "own rungs freeing it"
        ),
        bound_seconds=None,
        bound_source="",
        backstop=(
            "the ladder's terminal rung, which admits the head best-effort once its attempts are exhausted, "
            "and then " + _GIVE_UP_BACKSTOP
        ),
        observable_at=_PRELOAD_ADMISSION_OBSERVABLE,
    ),
    GateEntry(
        key="prestage",
        surface=GateSurface.PRELOAD_ADMISSION,
        kind=GateKind.OUTCOME,
        subsystem="process_management.scheduling.governance.whole_card",
        engaged_by="a whole-card head should be brought into host RAM before it samples",
        released_by="not a hold: the pre-stage load is started on this pass",
        bound_seconds=None,
        bound_source="",
        backstop="",
        observable_at=_PRELOAD_ADMISSION_OBSERVABLE,
    ),
    # endregion
    # region pop gates
    GateEntry(
        key="intake_paused",
        surface=GateSurface.POP_GATE,
        kind=GateKind.HOLD,
        subsystem="process_management.config.worker_state",
        engaged_by=(
            "a worker-wide intake hold: shutdown, an operator or self pause, a download-only hold, or a "
            "terminal-recovery park"
        ),
        released_by="the flow that set the hold clearing it; each arming site owns its own release",
        bound_seconds=None,
        bound_source="",
        backstop=(
            "none, deliberately: this is the operator's own hold or a terminal state, and the pop-liveness "
            "sentinel stays silent for it rather than reporting an intended stop as a wedge"
        ),
        observable_at="WorkerState.workload_intake_paused, and the last_pop_gate stamp",
    ),
    GateEntry(
        key="ram_pressure",
        surface=GateSurface.POP_GATE,
        kind=GateKind.HOLD,
        subsystem="process_management.scheduling.inference_scheduler",
        engaged_by=(
            "host RAM is approaching its danger floor, or a process over the ceiling is draining, so a newly "
            "accepted job would age past its ttl before the degraded worker could serve it"
        ),
        released_by=(
            "RAM recovering above the release threshold with no process draining, which clears "
            "WorkerState.ram_pressure_pop_hold on the next scheduling cycle"
        ),
        bound_seconds=None,
        bound_source="",
        backstop=(
            "InferenceScheduler.governance_healthy_but_held, read by the recovery coordinator's healthy-hold "
            "watchdog: a hold standing over a measurably healthy host with nothing draining is a governance "
            "latch rather than pressure. Then " + _POP_GATE_WEDGE_BACKSTOP
        ),
        observable_at=(
            "WorkerState.ram_pressure_pop_hold, the last_pop_gate stamp, and the ram_pressure skipped-reason "
            "count; the reading behind it comes from InferenceScheduler._measured_available_ram_mb, which a "
            "test pins through set_available_ram_mb_provider"
        ),
    ),
    GateEntry(
        key="torch_unusable",
        surface=GateSurface.POP_GATE,
        kind=GateKind.HOLD,
        subsystem="process_management.jobs.job_popper",
        engaged_by="the installed torch build has no kernels for this GPU, or is CPU-only",
        released_by=(
            "nothing at runtime: this is a build or hardware fact and is deliberately sticky for the session, "
            "since every popped job would fail at its first kernel launch"
        ),
        bound_seconds=None,
        bound_source="",
        backstop=(
            "none by design; the condition is disclosed once at the transition and the remedy is reinstalling "
            "a matching torch build"
        ),
        observable_at="WorkerState.gpu_torch_incompatible / torch_build_cpu_only, and the last_pop_gate stamp",
    ),
    GateEntry(
        key="consecutive_failure_pause",
        surface=GateSurface.POP_GATE,
        kind=GateKind.HOLD,
        subsystem="process_management.jobs.job_popper",
        engaged_by="a run of consecutive job failures armed the failure backoff",
        released_by="the backoff deadline elapsing, which the popper re-checks each cycle",
        bound_seconds=None,
        bound_source="the configured consecutive-failure pause duration",
        backstop=_POP_GATE_WEDGE_BACKSTOP,
        observable_at=(
            "the consecutive_failure_pause governor spell in PopGovernorRegistry, and the last_pop_gate stamp"
        ),
    ),
    GateEntry(
        key="queue_full",
        surface=GateSurface.POP_GATE,
        kind=GateKind.HOLD,
        subsystem="process_management.jobs.job_popper",
        engaged_by="the local queue already holds its configured depth of accepted work",
        released_by=(
            "a queued job dispatching or completing, which drops the depth below the configured maximum; a "
            "full queue is a legitimate hold only while it moves"
        ),
        bound_seconds=None,
        bound_source="",
        backstop=_FULL_QUEUE_BACKSTOP,
        observable_at=(
            "the last_pop_gate stamp with last_pop_gate_since, against the job tracker's completion and inference- "
            "start counters"
        ),
    ),
    GateEntry(
        key="safety_backlog",
        surface=GateSurface.POP_GATE,
        kind=GateKind.HOLD,
        subsystem="process_management.jobs.job_popper",
        engaged_by="the post-inference safety stage is backed up at or past its self-tuned cap",
        released_by=(
            "the backlog draining to the release fraction of that cap; the latch is hysteretic, so it engages "
            "at the cap and releases lower, and it cannot re-engage at the same depth it just released at"
        ),
        bound_seconds=None,
        bound_source="",
        backstop=_POP_GATE_WEDGE_BACKSTOP,
        observable_at=(
            "the safety_backlog skipped-reason count, the periodic withholding warning, and the last_pop_gate stamp"
        ),
    ),
    GateEntry(
        key="submit_backlog",
        surface=GateSurface.POP_GATE,
        kind=GateKind.HOLD,
        subsystem="process_management.jobs.job_popper",
        engaged_by="the submit stage is backed up, so newly accepted work would age past its ttl awaiting delivery",
        released_by="the submit backlog draining, which the same hysteretic latch re-reads each cycle",
        bound_seconds=None,
        bound_source="",
        backstop=_POP_GATE_WEDGE_BACKSTOP,
        observable_at=(
            "the submit_backlog skipped-reason count, the latch's own engage/release lines, and the last_pop_gate "
            "stamp"
        ),
    ),
    GateEntry(
        key="warmup_first_job",
        surface=GateSurface.POP_GATE,
        kind=GateKind.HOLD,
        subsystem="process_management.jobs.job_popper",
        engaged_by="a job is queued and no job has completed this session, so the queue is held at one",
        released_by=(
            "the first job completing or faulting; the hold is on queueing ahead, never on serving the job "
            "already accepted, so the condition that clears it is the one already in flight"
        ),
        bound_seconds=None,
        bound_source="the first job's own duration",
        backstop=_GIVE_UP_BACKSTOP,
        observable_at="the last_pop_gate stamp, against JobTracker.total_num_completed_jobs",
    ),
    GateEntry(
        key="no_safety_process",
        surface=GateSurface.POP_GATE,
        kind=GateKind.HOLD,
        subsystem="process_management.lifecycle.process_lifecycle",
        engaged_by="no safety process is available to clear a generated result",
        released_by="the lifecycle manager starting or restoring a safety process",
        bound_seconds=None,
        bound_source="",
        backstop=(
            "the safety respawn ladder and its futility check, which escalate a safety pool that will not "
            "come up rather than leaving intake held on a process nothing is rebuilding"
        ),
        observable_at="the process map's safety slot state, and the last_pop_gate stamp",
    ),
    GateEntry(
        key="no_inference_process",
        surface=GateSurface.POP_GATE,
        kind=GateKind.HOLD,
        subsystem="process_management.lifecycle.process_lifecycle",
        engaged_by="no inference process is available to take work",
        released_by="an inference process becoming available, whether by finishing work or by being respawned",
        bound_seconds=None,
        bound_source="",
        backstop=("the recovery coordinator's pool escalation, and " + _POP_GATE_WEDGE_BACKSTOP),
        observable_at="ProcessMap.get_first_available_inference_process, and the last_pop_gate stamp",
    ),
    GateEntry(
        key="no_models_configured",
        surface=GateSurface.POP_GATE,
        kind=GateKind.HOLD,
        subsystem="process_management.jobs.job_popper",
        engaged_by="the configuration names no image models to load, so no offer can be composed",
        released_by="the operator configuring models, which a config reload picks up",
        bound_seconds=None,
        bound_source="",
        backstop=("none by design; it is a configuration error, disclosed at error level on every cycle it holds"),
        observable_at="the repeated configuration error line, and the last_pop_gate stamp",
    ),
    GateEntry(
        key="megapixelstep_wait",
        surface=GateSurface.POP_GATE,
        kind=GateKind.HOLD,
        subsystem="process_management.jobs.pop_throttler",
        engaged_by="the in-flight megapixelstep total is above the governor's threshold for accepting more work",
        released_by="in-flight work draining below the threshold as jobs complete",
        bound_seconds=None,
        bound_source="the in-flight work's own duration",
        backstop=(
            "the idle-fill breaker, which marks a pop urgent and bypasses this governor when a head has sat "
            "starved on an idle device, and then " + _POP_GATE_WEDGE_BACKSTOP
        ),
        observable_at="the megapixelstep_wait governor spell in PopGovernorRegistry, and the last_pop_gate stamp",
    ),
    GateEntry(
        key="pop_frequency_gate",
        surface=GateSurface.POP_GATE,
        kind=GateKind.HOLD,
        subsystem="process_management.jobs.pop_throttler",
        engaged_by="the inter-pop cadence has not elapsed since the previous attempt",
        released_by="the cadence elapsing, or the caller marking the pop urgent because a slot is starved",
        bound_seconds=None,
        bound_source="the configured inter-pop interval",
        backstop=_POP_GATE_WEDGE_BACKSTOP,
        observable_at="WorkerState.last_job_pop_time, and the last_pop_gate stamp",
    ),
    GateEntry(
        key="no_eligible_models",
        surface=GateSurface.POP_GATE,
        kind=GateKind.HOLD,
        subsystem="process_management.jobs.job_popper",
        engaged_by="model selection produced no model this worker can currently serve",
        released_by=(
            "the selection inputs changing: a model finishing download, a slot freeing, or the previous pop's "
            "no-jobs evidence lapsing"
        ),
        bound_seconds=None,
        bound_source="",
        backstop=_POP_GATE_WEDGE_BACKSTOP,
        observable_at=(
            "the unservable_model_holdback governor spell, the serviceability exclusion lines, and the last_pop_gate "
            "stamp"
        ),
    ),
    GateEntry(
        key="large_model_limits",
        surface=GateSurface.POP_GATE,
        kind=GateKind.HOLD,
        subsystem="process_management.jobs.job_popper",
        engaged_by="the large-model switch throttle and re-entry cooldown between them emptied this cycle's offer",
        released_by="the configured switch interval or re-entry cooldown elapsing",
        bound_seconds=None,
        bound_source="the configured large-model switch interval and re-entry cooldown",
        backstop=_POP_GATE_WEDGE_BACKSTOP,
        observable_at="the large_model_switch and large_model_reentry governor spells, and the last_pop_gate stamp",
    ),
    # endregion
    # region whole-card churn governors
    GateEntry(
        key="establish_rate",
        surface=GateSurface.WHOLE_CARD_GOVERNOR,
        kind=GateKind.HOLD,
        subsystem="process_management.scheduling.governance.whole_card",
        engaged_by=(
            "this card has already established as many whole-card residencies as the rolling window allows, "
            "which is the signature of a pricing oscillation rather than of demand"
        ),
        released_by=(
            "the oldest establishment aging out of the rolling window, at which point the next ask is admitted"
        ),
        bound_seconds=240.0,
        bound_source="whole_card._ESTABLISH_WINDOW_SECONDS",
        backstop=(
            "the governor defer dwell, past which the head stops asking for the card and ordinary measured "
            "admission decides, so a rate brake can never become an absolute park"
        ),
        observable_at="the throttled whole_card_governor_hold warning, which names the governor and its arithmetic",
    ),
    GateEntry(
        key="grace_budget",
        surface=GateSurface.WHOLE_CARD_GOVERNOR,
        kind=GateKind.HOLD,
        subsystem="process_management.scheduling.governance.whole_card",
        engaged_by=(
            "the grace windows this card has opened inside the rolling window exceed its allowance, so opening "
            "another would disarm the recovery supervisor for longer than a wedge could hide behind"
        ),
        released_by=(
            "enough charges aging out of the rolling window for the spend to fall back inside the allowance; "
            "only a physical establish or restore is charged, so reuse of a held residency never spends it"
        ),
        bound_seconds=1200.0,
        bound_source="whole_card._GRACE_BUDGET_WINDOW_SECONDS",
        backstop=(
            "the governor defer dwell, past which the head is served co-resident where the device reading "
            "allows rather than waiting on the allowance"
        ),
        observable_at=(
            "the throttled whole_card_governor_hold warning, which quotes the spend, the remaining allowance and the "
            "replenish wait"
        ),
    ),
    # endregion
    # region dispatch-stall attributions
    GateEntry(
        key="sampling",
        surface=GateSurface.DISPATCH_STALL,
        kind=GateKind.OUTCOME,
        subsystem="process_management.scheduling.slot_duty",
        engaged_by="the slot was running a dispatched job",
        released_by="not a hold: this is the productive bucket",
        bound_seconds=None,
        bound_source="",
        backstop="",
        observable_at=_DISPATCH_STALL_OBSERVABLE,
    ),
    GateEntry(
        key="no_local_work",
        surface=GateSurface.DISPATCH_STALL,
        kind=GateKind.OUTCOME,
        subsystem="process_management.scheduling.slot_duty",
        engaged_by="no queued job was waiting for the slot",
        released_by="not a hold: nothing is being held, since there is nothing to hold",
        bound_seconds=None,
        bound_source="",
        backstop="",
        observable_at=_DISPATCH_STALL_OBSERVABLE,
    ),
    GateEntry(
        key="model_loading",
        surface=GateSurface.DISPATCH_STALL,
        kind=GateKind.HOLD,
        subsystem="process_management.scheduling.inference_scheduler",
        engaged_by="the head's model is preloading, so the slot idles until the load lands",
        released_by="the load completing, faulting, or its process dying",
        bound_seconds=None,
        bound_source="the load's own duration",
        backstop=(
            "the preload-death watchdog and the aux-download deadline, which fault a load that will never "
            "conclude; the idle-fill breaker feeds a free sibling meanwhile"
        ),
        observable_at=_DISPATCH_STALL_OBSERVABLE,
    ),
    GateEntry(
        key="preload_deferred",
        surface=GateSurface.DISPATCH_STALL,
        kind=GateKind.HOLD,
        subsystem="process_management.scheduling.inference_scheduler",
        engaged_by="the head's model is not resident and its preload was declined, or none was attempted",
        released_by="whatever releases the admission decision recorded against it; this bucket quotes that verdict",
        bound_seconds=None,
        bound_source="",
        backstop="the backstop of the preload_admission entry the recorded decision names, then " + _GIVE_UP_BACKSTOP,
        observable_at=_DISPATCH_STALL_OBSERVABLE,
    ),
    GateEntry(
        key="whole_card_reserved",
        surface=GateSurface.DISPATCH_STALL,
        kind=GateKind.HOLD,
        subsystem="process_management.scheduling.governance.whole_card",
        engaged_by=(
            "a whole-card residency is held for a different model, so the card is reserved and its siblings are torn "
            "down"
        ),
        released_by="that residency draining and restoring, which respawns the siblings and frees the card",
        bound_seconds=None,
        bound_source="the residency cooldown plus the holder's own job duration",
        backstop=(
            "the residency min-hold floor bounds only early release; the drain backstop admits the holder's "
            "head on its structural guarantee, and the recovery supervisor escalates a residency that never "
            "completes once its granted grace window elapses"
        ),
        observable_at=_DISPATCH_STALL_OBSERVABLE,
    ),
    GateEntry(
        key="resident_slot_busy",
        surface=GateSurface.DISPATCH_STALL,
        kind=GateKind.HOLD,
        subsystem="process_management.scheduling.inference_scheduler",
        engaged_by="the head's model is resident only on a process busy with other work",
        released_by="that process finishing its job, or a second copy of the model being staged elsewhere",
        bound_seconds=None,
        bound_source="the busy job's own duration",
        backstop="the inference step-progress watchdog, which faults a sampler that stops making progress",
        observable_at=_DISPATCH_STALL_OBSERVABLE,
    ),
    GateEntry(
        key="keep_single_inference",
        surface=GateSurface.DISPATCH_STALL,
        kind=GateKind.HOLD,
        subsystem="process_management.lifecycle.process_map",
        engaged_by=(
            "a workflow that cannot coexist with any concurrent inference is live on a slot that can accept "
            "work, holding the worker to a single lane"
        ),
        released_by=(
            "that workflow's job leaving the slot it was last referenced on; the hold is derived from live "
            "process state every cycle and latches nothing"
        ),
        bound_seconds=None,
        bound_source="the holding job's own duration",
        backstop="the inference step-progress watchdog, then " + _GIVE_UP_BACKSTOP,
        observable_at=_DISPATCH_STALL_OBSERVABLE,
    ),
    GateEntry(
        key="exclusive_isolation",
        surface=GateSurface.DISPATCH_STALL,
        kind=GateKind.HOLD,
        subsystem="process_management.scheduling.inference_scheduler",
        engaged_by="an exclusively-admitted over-budget job holds the card to itself",
        released_by="that job completing or faulting, which drops its exclusive claim",
        bound_seconds=None,
        bound_source="the exclusive job's own duration",
        backstop="the inference step-progress watchdog, then " + _GIVE_UP_BACKSTOP,
        observable_at=_DISPATCH_STALL_OBSERVABLE,
    ),
    GateEntry(
        key="concurrency_cap",
        surface=GateSurface.DISPATCH_STALL,
        kind=GateKind.HOLD,
        subsystem="process_management.scheduling.inference_scheduler",
        engaged_by="the in-progress count is at the cap the scheduler currently computes",
        released_by="an in-flight job completing or faulting",
        bound_seconds=None,
        bound_source="the in-flight jobs' own durations",
        backstop="the inference step-progress watchdog, then " + _GIVE_UP_BACKSTOP,
        observable_at=_DISPATCH_STALL_OBSERVABLE,
    ),
    GateEntry(
        key="overlap_headway",
        surface=GateSurface.DISPATCH_STALL,
        kind=GateKind.HOLD,
        subsystem="process_management.scheduling.inference_scheduler",
        engaged_by=(
            "the in-flight job has not yet made size-appropriate sampling headway, so a newcomer joining now "
            "would stack two loads and activation peaks"
        ),
        released_by=(
            "the in-flight job passing its headway fraction, or finishing; headway is measured as progress "
            "through the running job, so it advances with every sampled step"
        ),
        bound_seconds=None,
        bound_source="a fraction of the in-flight job's steps",
        backstop=(
            "the running job finishing bounds the wait unconditionally, since the gate only holds while "
            "something else samples"
        ),
        observable_at=_DISPATCH_STALL_OBSERVABLE,
    ),
    GateEntry(
        key="whole_card_convergence",
        surface=GateSurface.DISPATCH_STALL,
        kind=GateKind.HOLD,
        subsystem="process_management.scheduling.governance.whole_card",
        engaged_by="a held residency's pre-staged head waits for the pool to collapse to sole residency",
        released_by=(
            "the live inference-process count reaching the residency's effective target with safety and the "
            "service lanes clear of the card"
        ),
        bound_seconds=None,
        bound_source="the drain settle window measured from structural completion",
        backstop=(
            "WholeCardResidencyLedger.drain_backstop_elapsed, which admits the head on the forecast's "
            "sole-residency guarantee once the structural teardown has held for the settle window"
        ),
        observable_at=_DISPATCH_STALL_OBSERVABLE,
    ),
    GateEntry(
        key="disagg_pin_wait",
        surface=GateSurface.DISPATCH_STALL,
        kind=GateKind.HOLD,
        subsystem="process_management.disaggregation",
        engaged_by="the head's model is resident only on a disaggregation-pinned sampler lane",
        released_by="the pin releasing when its disaggregated job completes, freeing the lane for the head",
        bound_seconds=None,
        bound_source="the pinned job's own duration",
        backstop=("the disaggregation lane strand and pin-wedge detectors, which release a pin whose owner is gone"),
        observable_at=_DISPATCH_STALL_OBSERVABLE,
    ),
    GateEntry(
        key="degraded_isolation_pending",
        surface=GateSurface.DISPATCH_STALL,
        kind=GateKind.HOLD,
        subsystem="process_management.scheduling.inference_scheduler",
        engaged_by="the head's next dispatch must run isolated and waits for the card to clear of other work",
        released_by="the card clearing as the other in-flight jobs finish",
        bound_seconds=None,
        bound_source="the other in-flight jobs' own durations",
        backstop="the inference step-progress watchdog, then " + _GIVE_UP_BACKSTOP,
        observable_at=_DISPATCH_STALL_OBSERVABLE,
    ),
    GateEntry(
        key="residency_reconciliation",
        surface=GateSurface.DISPATCH_STALL,
        kind=GateKind.HOLD,
        subsystem="process_management.scheduling.inference_scheduler",
        engaged_by=(
            "a resident head's materialisation would over-commit the card until an idle resident is evicted, "
            "so dispatch is held while the eviction runs"
        ),
        released_by="the eviction freeing room, which the gate re-reads on the next dispatch pass",
        bound_seconds=None,
        bound_source="the eviction's own duration, a few scheduling cycles",
        backstop=(
            "the dispatch hold ledger, which the scheduler clears when the head dispatches or the job leaves "
            "the queue, and then " + _GIVE_UP_BACKSTOP
        ),
        observable_at=_DISPATCH_STALL_OBSERVABLE,
    ),
    GateEntry(
        key="post_processing_defer",
        surface=GateSurface.DISPATCH_STALL,
        kind=GateKind.HOLD,
        subsystem="process_management.scheduling.inference_scheduler",
        engaged_by=(
            "an in-flight or imminent post-processing chain's committed VRAM and this job's sampling peak "
            "cannot share the card"
        ),
        released_by="the chain finishing and releasing the device",
        bound_seconds=None,
        bound_source="the post-processing chain's own duration",
        backstop=("the post-processing drain watchdog, which reclaims a lane whose chain has stopped progressing"),
        observable_at=_DISPATCH_STALL_OBSERVABLE,
    ),
    GateEntry(
        key="clearance_hold",
        surface=GateSurface.DISPATCH_STALL,
        kind=GateKind.HOLD,
        subsystem="process_management.scheduling.clearance_lease",
        engaged_by=(
            "a staged waiter's full materialisation does not fit measured device truth at the moment it would "
            "take the GPU"
        ),
        released_by="eviction making room, which grants the lease",
        bound_seconds=None,
        bound_source="CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS",
        backstop=(
            "the lease-acquire timeout, past which the waiter degrades into unpriced sampling rather than "
            "holding the slot"
        ),
        observable_at=_DISPATCH_STALL_OBSERVABLE,
    ),
    GateEntry(
        key="unexplained",
        surface=GateSurface.DISPATCH_STALL,
        kind=GateKind.HOLD,
        subsystem="process_management.scheduling.inference_scheduler",
        engaged_by=(
            "the head is resident on an idle process and no gate claimed the empty slot: the "
            "scheduler-stall-shaped case, and the one this registry exists to keep empty"
        ),
        released_by=(
            "nothing declared, which is the point: an engagement here means a hold exists that no entry in "
            "this registry describes"
        ),
        bound_seconds=None,
        bound_source="",
        backstop=_GIVE_UP_BACKSTOP,
        observable_at=_DISPATCH_STALL_OBSERVABLE,
    ),
    # endregion
    # region service-lane pauses
    GateEntry(
        key="whole_card",
        surface=GateSurface.LANE_PAUSE,
        kind=GateKind.HOLD,
        subsystem="process_management.lifecycle.process_lifecycle",
        engaged_by="a whole-card residency moved a GPU-bearing service process off its card to free the device",
        released_by=(
            "the residency restoring, which is the same initiator releasing its own pause; another subsystem's "
            "request arriving meanwhile does not take ownership and cannot clear it"
        ),
        bound_seconds=None,
        bound_source="the residency's own hold",
        backstop=(
            "the recovery supervisor, which escalates a residency that has not completed once its granted "
            "grace window elapses, and the safety reconciler, which restores the lane when every request and "
            "restore veto has cleared"
        ),
        observable_at="the lane pause/restore log lines, which name the initiating subsystem",
    ),
    GateEntry(
        key="reclaim_ladder",
        surface=GateSurface.LANE_PAUSE,
        kind=GateKind.HOLD,
        subsystem="process_management.lifecycle.process_lifecycle",
        engaged_by="the verified reclaim ladder moved a service process off-GPU to free measured room",
        released_by="the ladder's own restore once the pressure that justified the pause has eased",
        bound_seconds=None,
        bound_source="",
        backstop=(
            "the reclaim ladder's own termination: its rungs are finite and the terminal rung admits the head "
            "rather than holding the pause open for a reclaim that is not coming"
        ),
        observable_at="the lane pause/restore log lines, which name the initiating subsystem",
    ),
    GateEntry(
        key="runtime_safety_placement",
        surface=GateSurface.LANE_PAUSE,
        kind=GateKind.HOLD,
        subsystem="process_management.lifecycle.process_lifecycle",
        engaged_by="the runtime placement policy moved safety off-GPU because its context did not fit beside sampling",
        released_by="the placement policy re-evaluating once sampling frees the room its context needs",
        bound_seconds=None,
        bound_source="",
        backstop=(
            "the no_safety_process pop gate stops intake while safety is unavailable, so the pause cannot "
            "silently accumulate work it will not clear"
        ),
        observable_at="the lane pause/restore log lines, which name the initiating subsystem",
    ),
    # endregion
    # region self-throttle pop pause owners
    GateEntry(
        key="fault_throttle",
        surface=GateSurface.POP_PAUSE,
        kind=GateKind.HOLD,
        subsystem="process_management.config.worker_state",
        engaged_by=(
            "a fault backstop armed the self-throttle pause: either resource/OOM self-maintenance or the "
            "terminal fault-rate breaker"
        ),
        released_by="the standing pause deadline lapsing; no site shortens a deadline it does not own",
        bound_seconds=None,
        bound_source="the arming backstop's own pause duration",
        backstop=(
            "the deadline itself is the bound: the pause is a timer, not a condition, so it cannot outlive "
            "its own expiry"
        ),
        observable_at=(
            "WorkerState.self_throttle_paused_until with its owner and reason, and the self_throttle_pause governor "
            "spell"
        ),
    ),
    GateEntry(
        key="ram_pressure",
        surface=GateSurface.POP_PAUSE,
        kind=GateKind.HOLD,
        subsystem="process_management.config.worker_state",
        engaged_by="the host-RAM-pressure governor armed the pause while system RAM was under its danger floor",
        released_by="the standing pause deadline lapsing",
        bound_seconds=None,
        bound_source="the governor's own pause duration",
        backstop="the deadline itself is the bound",
        observable_at=(
            "WorkerState.self_throttle_paused_until with its owner and reason, and the self_throttle_pause governor "
            "spell"
        ),
    ),
    GateEntry(
        key="safety",
        surface=GateSurface.POP_PAUSE,
        kind=GateKind.HOLD,
        subsystem="process_management.config.worker_state",
        engaged_by="the safety soft-pause armed the pause because a generated result could not be safety-checked",
        released_by="the standing pause deadline lapsing",
        bound_seconds=None,
        bound_source="the soft-pause duration",
        backstop="the deadline itself is the bound",
        observable_at=(
            "WorkerState.self_throttle_paused_until with its owner and reason, and the self_throttle_pause governor "
            "spell"
        ),
    ),
    # endregion
    # region recovery parks
    GateEntry(
        key="runaway_recoveries",
        surface=GateSurface.RECOVERY_PARK,
        kind=GateKind.HOLD,
        subsystem="process_management.lifecycle.worker_recovery_coordinator",
        engaged_by="process recoveries breached the rolling-window ceiling, so rebuilding is not stabilising the pool",
        released_by=(
            "nothing at runtime, deliberately: the park is the terminal state of an exhausted escalation, and "
            "continuing to rebuild is what it exists to stop"
        ),
        bound_seconds=None,
        bound_source="",
        backstop=(
            "none by design; the park sets worker-wide intake off and is disclosed with its reason so an "
            "operator restarts rather than the worker cycling itself"
        ),
        observable_at="WorkerState.recovery_parked with its RecoveryParkReason, and the action ledger",
    ),
    GateEntry(
        key="unrecoverable_pool",
        surface=GateSurface.RECOVERY_PARK,
        kind=GateKind.HOLD,
        subsystem="process_management.lifecycle.worker_recovery_coordinator",
        engaged_by="the escalation spent its in-place remedies over a pool that still cannot serve accepted work",
        released_by="nothing at runtime; the remedies are exhausted by definition",
        bound_seconds=None,
        bound_source="",
        backstop=(
            "none by design; accepted work is faulted and reissued before the park, so the park holds no "
            "obligations open"
        ),
        observable_at="WorkerState.recovery_parked with its RecoveryParkReason, and the action ledger",
    ),
    # endregion
    # region scheduler budgets with no enumerable runtime key
    GateEntry(
        key="affinity_line_skip",
        surface=GateSurface.SCHEDULER_BUDGET,
        kind=GateKind.HOLD,
        subsystem="process_management.scheduling.dispatch_affinity",
        engaged_by=(
            "a resident-model job passes the FIFO head while the head's model is cold, holding the head out "
            "of its own queue position"
        ),
        released_by=(
            "either bound being reached: the committed skip count hitting the ceiling, or the wall-clock "
            "budget elapsing since the head's first skip. Both are keyed to the head's identity, so a "
            "different head starts a fresh window and one head's spent budget never carries to the next"
        ),
        bound_seconds=45.0,
        bound_source=(
            "dispatch_affinity._AFFINITY_BUDGET_MAX_SECONDS, the ceiling of the ttl-derived budget; the "
            "count ceiling is _AFFINITY_MAX_SKIPS and either bound alone ends the bypass"
        ),
        backstop=(
            "the bound is unconditional: no path raises it, and a max_skips of zero disables bypassing "
            "entirely, so the head always reclaims the slot it never lost"
        ),
        observable_at="the head-starvation diagnostic, which names the skip count and the span it covered",
    ),
    GateEntry(
        key="whole_card_min_hold",
        surface=GateSurface.SCHEDULER_BUDGET,
        kind=GateKind.HOLD,
        subsystem="process_management.scheduling.governance.whole_card",
        engaged_by=(
            "a different-model head asks to release a residency that was granted within the minimum hold, "
            "which would buy one lighter job at the price of two full pool rebuilds"
        ),
        released_by="the minimum hold elapsing from the grant",
        bound_seconds=90.0,
        bound_source="whole_card._MIN_HOLD_SECONDS",
        backstop=(
            "the floor is deliberately shorter than the establish grace window, so it can never be the thing "
            "keeping a residency alive past the point the recovery supervisor is watching"
        ),
        observable_at="WholeCardResidency.min_hold_until, and the residency status snapshot",
    ),
    # endregion
)
"""Every gate that can hold or defer work, with its declared release path and bound.

Ordered by surface for reading. The guardrail test enumerates each surface's runtime members and fails on
any member without an entry here, so the tuple cannot fall behind the code for the surfaces it can
enumerate. ``SCHEDULER_BUDGET`` entries are hand-listed and carry no such protection.
"""
