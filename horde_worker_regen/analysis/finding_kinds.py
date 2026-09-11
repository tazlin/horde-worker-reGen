"""The declared set of diagnoses ``horde-log diagnose`` can reach, and the shape of one.

A finding id is a compatibility surface: it keys the ``diagnose --json`` output, the dashboard's
Diagnostics tab, the ``see also:`` cross-references, and the catalogue in
``docs/reference/log_findings.md``. Declaring the set here rather than at each emit site is what makes a
cross-reference checkable and a catalogue entry enforceable.

Public surface:

- :class:`Severity`: how urgent a finding is, and the report's sort key.
- :class:`FindingKind`: one member per diagnosis; its value is the printed id.
- :class:`FindingSpec`: the per-kind constants (catalogue title, invariant remediation, default
  cross-reference).
- :data:`FINDING_SPECS`: the table, one entry per kind.
- :class:`Finding`: one emitted diagnosis, pairing a kind with the measurement that produced it.

This module deliberately imports nothing from the rest of the analysis package: the detectors, the
report renderers and the catalogue test all read the table, so it has to sit below all of them.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field


class Severity(enum.StrEnum):
    """How urgent a finding is; also its sort key (critical first).

    The reader sees an action word for each level (see :data:`SEVERITY_BADGE_WORDS`); the values here are
    the ids the JSON output and the catalogue carry.
    """

    CRITICAL = "critical"
    """The worker is losing work or will be paused by the horde."""
    WARNING = "warning"
    """Something is wrong or wasteful and the reader should look."""
    SUGGESTION = "suggestion"
    """Nothing is wrong; a change would likely earn more."""
    INFO = "info"
    """Context the reader may want; no action."""


SEVERITY_ORDER: Mapping[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.WARNING: 1,
    Severity.SUGGESTION: 2,
    Severity.INFO: 3,
}
"""The report's sort key: most urgent first."""

SEVERITY_BADGE_WORDS: Mapping[Severity, str] = {
    Severity.CRITICAL: "Fix now",
    Severity.WARNING: "Check",
    Severity.SUGGESTION: "Try",
    Severity.INFO: "Note",
}
"""What the badge says to the reader: what to do with the finding, not a level name to decode."""


class FindingKind(enum.StrEnum):
    """Every diagnosis a detector can emit, by the id it prints.

    The values are the ids operators, the JSON output and the catalogue already use, so they are fixed:
    renaming one is a breaking change to all three, not a refactor.
    """

    CRASH_ON_START_LOOP = "crash_on_start_loop"
    PRELOAD_KILLS_CHILD_LOOP = "preload_kills_child_loop"
    EMPTY_MODEL_POP_CASCADE = "empty_model_pop_cascade"
    DOOMED_POOL_NO_GIVEUP = "doomed_pool_no_giveup"
    GAVE_UP_CLEAN = "gave_up_clean"
    STUCK_INFERENCE_STEP = "stuck_inference_step"
    POST_PROCESSING_VRAM_STALL = "post_processing_vram_stall"
    ORPHAN_WEDGE = "orphan_wedge"

    OOM = "oom"
    SWALLOWED_OOM = "swallowed_oom"
    FILE_DESCRIPTOR_EXHAUSTION = "file_descriptor_exhaustion"
    SCHEDULER_STARVATION_WEDGE = "scheduler_starvation_wedge"
    UNSATISFIABLE_HEAD_STARVATION = "unsatisfiable_head_starvation"
    RESIDENCY_RECONCILIATION_HOLDS = "residency_reconciliation_holds"
    WHOLE_CARD_CONVERGENCE_WEDGE = "whole_card_convergence_wedge"
    WHOLE_CARD_NONHEAD_RESIDENCY_STARVATION = "whole_card_nonhead_residency_starvation"
    WHOLE_CARD_RESIDENCY_CHURN = "whole_card_residency_churn"
    WHOLE_CARD_POP_CLAIM_EPISODES = "whole_card_pop_claim_episodes"
    WHOLE_CARD_POP_CLAIM_MONOPOLY = "whole_card_pop_claim_monopoly"
    MODEL_CHURN = "model_churn"

    MULTI_CARD_DISPATCH_SERIALIZATION = "multi_card_dispatch_serialization"
    HEAD_DISPATCH_STALL = "head_dispatch_stall"
    SLOW_GENERATION_DROP_SPIRAL = "slow_generation_drop_spiral"
    POST_PROCESSING_DEFERRAL_STARVATION = "post_processing_deferral_starvation"
    SAFETY_STAGE_STALL = "safety_stage_stall"
    SAFETY_STAGE_CAPACITY = "safety_stage_capacity"
    POP_LIVENESS_FULL_QUEUE = "pop_liveness_full_queue"
    PARENT_LOOP_STALL = "parent_loop_stall"
    LANE_PLACEMENT = "lane_placement"

    FORCED_MAINTENANCE = "forced_maintenance"
    CONSECUTIVE_FAILURE_PAUSE = "consecutive_failure_pause"
    POP_API_ERROR_DOMINANCE = "pop_api_error_dominance"
    POP_GOVERNOR_DOMINANCE = "pop_governor_dominance"
    FAULTED_JOB_CENSUS = "faulted_job_census"
    MODEL_REFERENCE_SAMPLE_FAULT = "model_reference_sample_fault"

    SESSION_SUMMARY = "session_summary"


@dataclass(frozen=True)
class FindingSpec:
    """Represents everything about a diagnosis that does not depend on the session being read.

    The title is the catalogue name for the kind. A detector that words its headline differently per
    severity (a wedge versus the same gate merely parking a head) passes its own title at the emit site;
    the spec's stays the name the catalogue and the cross-references use.

    A finding shows two layers. The plain layer is the headline the detector writes per emit and the
    "Do this" line: ``action`` is the part of that line that holds for every emit of the kind. Where the
    fix genuinely depends on what was measured, it is empty and the detector supplies the whole text as an
    addendum; where both are present the rendered line is the invariant followed by the addendum.
    ``detail`` is the detail layer's prose: what is going on underneath, in the same voice but free to name
    the subsystem; the CLI prints it and the dashboard collapses it.

    ``see_also`` points at another diagnosis; ``reference_page`` points at prose. They are separate fields
    because they are checked differently: a cross-reference has to name a declared kind, a reference page
    has to be a file that exists, and a kind whose fix needs background has both.
    """

    kind: FindingKind
    title: str
    action: str = ""
    detail: str = ""
    see_also: FindingKind | None = None
    reference_page: str | None = None
    """A repo-relative path to the docs page that explains the subsystem behind this diagnosis.

    Set it where acting on the remediation needs background the finding cannot carry inline; a test
    asserts the page is on disk, so a moved or renamed page fails rather than printing a dead path."""


def _spec_table(*specs: FindingSpec) -> Mapping[FindingKind, FindingSpec]:
    """Index the specs by kind, refusing a duplicate or a kind declared twice."""
    table: dict[FindingKind, FindingSpec] = {}
    for spec in specs:
        if spec.kind in table:
            raise ValueError(f"duplicate FindingSpec for {spec.kind}")
        table[spec.kind] = spec
    return table


FINDING_SPECS: Mapping[FindingKind, FindingSpec] = _spec_table(
    # --- Startup and process lifecycle ---
    FindingSpec(
        kind=FindingKind.CRASH_ON_START_LOOP,
        title="Inference pool crashes on start",
    ),
    FindingSpec(
        kind=FindingKind.PRELOAD_KILLS_CHILD_LOOP,
        title="A model ends every slot it is loaded onto",
        see_also=FindingKind.EMPTY_MODEL_POP_CASCADE,
    ),
    FindingSpec(
        kind=FindingKind.EMPTY_MODEL_POP_CASCADE,
        title="Pops with no model name",
        see_also=FindingKind.PRELOAD_KILLS_CHILD_LOOP,
    ),
    FindingSpec(
        kind=FindingKind.DOOMED_POOL_NO_GIVEUP,
        title="Recovery storm never gave up",
        action=(
            "Pair with the crash-on-start root cause: the pool cannot recover, so it should give up "
            "fast. The give-up abort only fires when every slot is quarantined at the exact give-up "
            "tick, but a soft reset's transient un-quarantine (and a clean window longer than the "
            "recovery clean streak) keeps that from coinciding, so the worker spins. Make the abort "
            "latch 'was fully quarantined this episode' rather than sampling the instantaneous state."
        ),
        see_also=FindingKind.CRASH_ON_START_LOOP,
    ),
    FindingSpec(
        kind=FindingKind.GAVE_UP_CLEAN,
        title="Worker gave up on an unrecoverable pool",
        action="No worker action needed beyond fixing the underlying crash cause; the bail-out worked.",
    ),
    FindingSpec(
        kind=FindingKind.STUCK_INFERENCE_STEP,
        title="Inference wedged on a non-advancing step",
        action=(
            "Recovery worked, but the hang is upstream in ComfyUI/hordelib. The usual trigger is a "
            "corrupt or incompatible model+LoRA combination: e.g. an SD1.5 LoRA applied to an SDXL "
            "checkpoint produces a `ERROR lora ... shape ... is invalid` storm and then the pipeline "
            "hangs at the final step. Check the affected slot's bridge_<N>.log just before the reap for "
            "that shape-mismatch storm and exclude the offending LoRA/model pairing. If healthy jobs are "
            "being reaped, raise `inference_stuck_step_repeat_limit`."
        ),
    ),
    FindingSpec(
        kind=FindingKind.POST_PROCESSING_VRAM_STALL,
        title="Post-processing stalled on an over-committed card",
        reference_page="docs/explanation/performance_and_backpressure.md",
    ),
    FindingSpec(
        kind=FindingKind.ORPHAN_WEDGE,
        title="Orphaned in-progress jobs",
        action=(
            "Inspect the inference slots for hangs/OOM around these punts; a sustained storm should "
            "escalate to a soft reset (pool rebuild)."
        ),
    ),
    # --- Memory and residency ---
    FindingSpec(
        kind=FindingKind.OOM,
        title="GPU out-of-memory faults",
        action=(
            "Reduce concurrency/queue or enable a more conservative VRAM budget; if these recur under "
            "a budget that should fit, suspect over-admission of a heavy head (Flux fp8 / SDXL). The "
            "named co-residency and free-VRAM figures say which: several co-resident processes with "
            "near-zero free VRAM points at too many models admitted onto one card, not at the faulting "
            "model being individually oversized."
        ),
    ),
    FindingSpec(
        kind=FindingKind.SWALLOWED_OOM,
        title="Jobs faulted with 'no images produced'",
        action=(
            "Check VRAM headroom around these faults; if memory-bound, treat 'no images produced' as a "
            "resource failure so the self-throttle/breaker engages."
        ),
    ),
    FindingSpec(
        kind=FindingKind.FILE_DESCRIPTOR_EXHAUSTION,
        title="Inference process exhausted its file-descriptor limit (EMFILE)",
        action=(
            "Treat this as a descriptor leak, not memory pressure: reducing concurrency or the VRAM "
            "budget will not help. As an immediate stopgap, raise the worker's soft descriptor limit "
            "(ulimit -n, or LimitNOFILE= in the systemd unit) so a slow leak takes far longer to reach "
            "the ceiling. The real fix is to find what leaks descriptors in the inference child; these "
            "logs cannot pinpoint it because the worker emits no descriptor-headroom telemetry, so add "
            "RLIMIT_NOFILE headroom to the per-process status line (alongside the free-RAM/VRAM figures) "
            "so the next occurrence names the leaking growth. This fault is POSIX-specific (Windows has "
            "no RLIMIT_NOFILE and a far higher handle ceiling), so it is a concern for Linux hosts."
        ),
    ),
    FindingSpec(
        kind=FindingKind.SCHEDULER_STARVATION_WEDGE,
        title="Scheduler wedged on VRAM-budget over-deferral",
    ),
    FindingSpec(
        kind=FindingKind.UNSATISFIABLE_HEAD_STARVATION,
        title="Head-of-queue model persistently starved on an idle device",
        action=(
            "Confirm the named model actually fits this device (its resident weights plus activation "
            "working set against measured device-free VRAM); a head that never admits despite an idle, "
            "ample-VRAM device points at an over-conservative per-process overhead or an unsatisfiable "
            "budget for that model. Reduce process churn (unload_models_from_vram_often / "
            "high_performance_mode) so a settled baseline sizes the overhead, relax the VRAM budget, or "
            "drop the model if the device genuinely cannot host it. A give-up / pop-hold should bound the "
            "wait so the head cannot starve the queue indefinitely."
        ),
        see_also=FindingKind.SCHEDULER_STARVATION_WEDGE,
    ),
    FindingSpec(
        kind=FindingKind.RESIDENCY_RECONCILIATION_HOLDS,
        title="Dispatch held to reconcile residency (idle-VRAM eviction swap-churn)",
        see_also=FindingKind.HEAD_DISPATCH_STALL,
    ),
    FindingSpec(
        kind=FindingKind.WHOLE_CARD_CONVERGENCE_WEDGE,
        title="Whole-card residency cannot reach sole residency (a sibling or service lane pins the teardown)",
        action=(
            "Capture the surrounding scheduling logs and the process map: the stall line names what pinned "
            "the teardown (a sibling process and its queued model, or a lane). A recurrence points at the "
            "whole-card teardown failing to stop an eligible sibling or failing to order a lane pause. As an "
            "operational stopgap, reducing queue_size or avoiding a heavy whole-card model alongside a deep "
            "same-cycle queue lowers the odds of hitting this shape."
        ),
        see_also=FindingKind.HEAD_DISPATCH_STALL,
    ),
    FindingSpec(
        kind=FindingKind.WHOLE_CARD_NONHEAD_RESIDENCY_STARVATION,
        title="Whole-card residency held for a non-head model starved the queue head",
        action=(
            "The whole-card residency must only be granted to the head (next-to-dispatch) job; a deeper-queue "
            "heavy model should defer until it becomes the head rather than reserving the card. If this "
            "recurs, capture the residency establish/pre-stage lines and the queue order to confirm which "
            "model claimed the card while a different head was pending."
        ),
        see_also=FindingKind.SCHEDULER_STARVATION_WEDGE,
    ),
    FindingSpec(
        kind=FindingKind.WHOLE_CARD_RESIDENCY_CHURN,
        title="Whole-card residency reserved and restored repeatedly (reservation churn)",
        see_also=FindingKind.SCHEDULER_STARVATION_WEDGE,
    ),
    FindingSpec(
        kind=FindingKind.WHOLE_CARD_POP_CLAIM_EPISODES,
        title="Whole-card residency claimed the pop offer",
        action=(
            "Nothing to do if the claims match the heavy work this worker exists to serve. If they are "
            "long and frequent for a model the operator did not intend to specialise in, the levers are "
            "the served model set and whole_card_residency_max_hold_seconds, which caps how long one "
            "residency may own the intake."
        ),
        see_also=FindingKind.WHOLE_CARD_POP_CLAIM_MONOPOLY,
    ),
    FindingSpec(
        kind=FindingKind.WHOLE_CARD_POP_CLAIM_MONOPOLY,
        title="Whole-card pop claim held to its cap while other models' work waited",
        action=(
            "Decide whether this worker should specialise. If it should, the parked models do not belong in "
            "its served set and removing them ends the contention. If it should serve a mix, lower "
            "whole_card_residency_max_hold_seconds so a residency gives the intake back sooner, or take the "
            "claimed model out of the pool if it can only run with the whole card."
        ),
        see_also=FindingKind.WHOLE_CARD_POP_CLAIM_EPISODES,
    ),
    FindingSpec(
        kind=FindingKind.MODEL_CHURN,
        title="Model weights churning through the lanes",
        action=(
            "Each turn costs a load off disk and the eviction that made room for it, and the cleared "
            "preloads are that cost paid for nothing. Give retention a chance to pay off: check that "
            "unload_models_from_vram_often is not forcing an eviction after every job, that the model "
            "pool (or a narrower model list) keeps the offered set within what the lanes can hold, and "
            "that the queue is deep enough for same-model jobs to batch onto one lane. On a multi-card "
            "host, dispatch that will not spread across cards forces the same few lanes to serve every "
            "model in the queue, which shows up here as churn."
        ),
        see_also=FindingKind.MULTI_CARD_DISPATCH_SERIALIZATION,
    ),
    # --- Dispatch and throughput ---
    FindingSpec(
        kind=FindingKind.MULTI_CARD_DISPATCH_SERIALIZATION,
        title="Dispatch is not spreading across the cards",
        see_also=FindingKind.MODEL_CHURN,
    ),
    FindingSpec(
        kind=FindingKind.HEAD_DISPATCH_STALL,
        title="Head-of-queue job repeatedly parked",
    ),
    FindingSpec(
        kind=FindingKind.SLOW_GENERATION_DROP_SPIRAL,
        title="Slow generation is dropping jobs",
        see_also=FindingKind.FORCED_MAINTENANCE,
    ),
    FindingSpec(
        kind=FindingKind.POST_PROCESSING_DEFERRAL_STARVATION,
        title="Post-processing lane starved by its admission gate",
        action=(
            "Verify the admission inputs: device-free VRAM must reflect the lane's card, reservations must "
            "include only memory not yet materialized in that measurement, the proportional noise margin "
            "must be applied once, and the per-chain marginal candidate must match measured operation "
            "costs. A deferred drain head should spend ordinary idle cache/model reclaim first, then may "
            "temporarily borrow only a verified-idle service-lane context. It must still age out to a "
            "no-image fault after a bounded wait, and fittable jobs behind it must be allowed to pass. As a "
            "stopgap, lower resident VRAM on the lane's card or disable post-processing on this worker."
        ),
        see_also=FindingKind.POST_PROCESSING_VRAM_STALL,
        reference_page="docs/explanation/process_lanes_and_chaining.md",
    ),
    FindingSpec(
        kind=FindingKind.SAFETY_STAGE_STALL,
        title="Safety stage stalled (lost verdicts / backlog)",
    ),
    FindingSpec(
        kind=FindingKind.SAFETY_STAGE_CAPACITY,
        title="One safety process behind more finishes than it can check",
        action=(
            "Safety is one process for the whole worker, so its throughput is fixed while the fleet's "
            "finish rate scales with the cards. Give it more capacity: enable safety_on_gpu so a check "
            "costs a fraction of its CPU time, and raise the number of safety processes if the build "
            "supports it. Lowering max_power will not help, and neither will lowering queue_size: both "
            "shrink the work, not the per-check cost, and a slower fleet still hands every image to the "
            "same single checker."
        ),
        see_also=FindingKind.SAFETY_STAGE_STALL,
        reference_page="docs/explanation/process_lanes_and_chaining.md",
    ),
    FindingSpec(
        kind=FindingKind.POP_LIVENESS_FULL_QUEUE,
        title="Local job queue full and not draining (the worker served nothing)",
        see_also=FindingKind.WHOLE_CARD_RESIDENCY_CHURN,
    ),
    FindingSpec(
        kind=FindingKind.PARENT_LOOP_STALL,
        title="The parent loop stopped draining IPC",
        action=(
            "Find what blocked the loop rather than what it failed to do: a synchronous call on the "
            "asyncio thread (a model-reference read, a disk scan, a network call without a timeout) is "
            "the usual cause. The timestamps below bound each gap; `horde-log timeline` over that "
            "window shows what the parent was doing immediately before it went quiet."
        ),
        see_also=FindingKind.MULTI_CARD_DISPATCH_SERIALIZATION,
    ),
    FindingSpec(
        kind=FindingKind.LANE_PLACEMENT,
        title="Auxiliary lanes moved or stacked onto one card",
        action=(
            "Compare the per-card duty and free-VRAM readings for the card(s) named above against the "
            "rest of the fleet before reading a low duty there as a GPU problem. If the placement was "
            "not intended, pin the auxiliary lanes (safety_on_gpu and the post-processing/utilities "
            "device selection) so a re-spawn returns them to the card they were sized for."
        ),
        see_also=FindingKind.MULTI_CARD_DISPATCH_SERIALIZATION,
    ),
    # --- Pops, faults, and the horde ---
    FindingSpec(
        kind=FindingKind.FORCED_MAINTENANCE,
        title="Horde forced the worker into maintenance",
    ),
    FindingSpec(
        kind=FindingKind.CONSECUTIVE_FAILURE_PAUSE,
        title="Worker self-paused on consecutive faults",
        action=(
            "Find the fault source (the starvation-wedge / recovery / OOM findings); the pause clears on its "
            "own but will re-trigger until the faults stop."
        ),
    ),
    FindingSpec(
        kind=FindingKind.POP_API_ERROR_DOMINANCE,
        title="The horde repeatedly refused this worker's pops",
        see_also=FindingKind.POP_GOVERNOR_DOMINANCE,
    ),
    FindingSpec(
        kind=FindingKind.POP_GOVERNOR_DOMINANCE,
        title="A pop governor shaped much of the session",
        action=(
            "If throughput was lower than expected, this names where the time went. Whole-card residency or "
            "the large-model limiters point at the model mix and their configured durations "
            "(whole_card_residency_cooldown_seconds, large_model_switch_min_seconds, "
            "large_model_reentry_cooldown_seconds); backpressure points at a slow safety stage; the "
            "unservable holdback points at a model the device cannot run."
        ),
    ),
    FindingSpec(
        kind=FindingKind.FAULTED_JOB_CENSUS,
        title="Jobs faulted this session",
        action=(
            "Faulted jobs are reissued by the horde and counted against this worker; a sustained rate "
            "drives forced maintenance. Take the largest cause first: a give-up backstop count means the "
            "scheduler wedged and the recovery path drained the backlog rather than serving it, a "
            "safety-unrecoverable count means results were produced but could not be checked, and a "
            "process-fault count points at the slot that ran them."
        ),
        see_also=FindingKind.SCHEDULER_STARVATION_WEDGE,
    ),
    FindingSpec(
        kind=FindingKind.MODEL_REFERENCE_SAMPLE_FAULT,
        title="Sample stage faulted on an unreadable model reference",
        action=(
            "The model-reference cache refresh is racing an in-flight sample: the child re-reads a category "
            "while the cache is rewriting it. Hold the reference the job was admitted with for the life of "
            "the job (or make the refresh atomic from a reader's point of view) so a background refresh "
            "cannot fault work already running. Check the horde_model_reference cache path for the affected "
            "category and confirm it is readable and not being rewritten by a second process."
        ),
        see_also=FindingKind.FAULTED_JOB_CENSUS,
    ),
    # --- Session context ---
    FindingSpec(
        kind=FindingKind.SESSION_SUMMARY,
        title="Session summary",
    ),
)


@dataclass
class Finding:
    """Represents one diagnosis of a session: the headline, the evidence, and what to do about it.

    The kind carries everything constant about the diagnosis (see :data:`FINDING_SPECS`); the fields here
    carry what the session itself produced. ``headline`` is always written at the emit site because it
    carries the measurement: one sentence saying what happened, with the numbers and names.
    """

    kind: FindingKind
    severity: Severity
    headline: str
    title_override: str | None = None
    """A headline this emit words differently from the catalogue name, for a kind whose severities read as
    different incidents (a head parked behind a gate versus one with no gate at all)."""
    action_addendum: str | None = None
    """The part of the fix that depends on what was measured, appended to the spec's invariant advice."""
    evidence: list[str] = field(default_factory=list)
    see_also: FindingKind | None = None
    """The cross-reference for this emit; the spec's default is filled in when none is given."""

    def __post_init__(self) -> None:
        """Fall back to the kind's declared cross-reference when the emit site named none."""
        if self.see_also is None:
            self.see_also = self.spec.see_also

    @property
    def spec(self) -> FindingSpec:
        """The declared constants for this finding's kind."""
        return FINDING_SPECS[self.kind]

    @property
    def id(self) -> str:
        """The printed finding id: the handle the JSON output, the dashboard and the catalogue share."""
        return self.kind.value

    @property
    def title(self) -> str:
        """This emit's headline: its own wording where it has one, otherwise the catalogue name."""
        return self.title_override or self.spec.title

    @property
    def reference_page(self) -> str | None:
        """The docs page explaining this kind's subsystem, or None where the remediation stands alone."""
        return self.spec.reference_page

    @property
    def action(self) -> str:
        """The "Do this" line: the kind's invariant advice followed by whatever this emit added to it."""
        invariant = self.spec.action
        if not self.action_addendum:
            return invariant
        if not invariant:
            return self.action_addendum
        return f"{invariant} {self.action_addendum}"

    @property
    def detail(self) -> str:
        """The detail layer's prose for this kind, or an empty string where the plain layer says it all."""
        return self.spec.detail

    @property
    def badge(self) -> str:
        """The action word the reader sees for this finding's severity."""
        return SEVERITY_BADGE_WORDS[self.severity]
