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
        title="Image processes crash on start",
        detail=(
            "The worker cannot serve jobs until its image processes start. Each process is started again "
            "after a crash; after repeated crashes the worker stops using it, and when none are left it stops "
            "itself. The error is lifted from the process's own start-up log, since the parent only sees the "
            "exit code. When the error came from preparing the shared ComfyUI environment, the usual cause is "
            "several processes cloning custom nodes into one directory at once. A clone cut off partway leaves "
            "files no later start will overwrite."
        ),
    ),
    FindingSpec(
        kind=FindingKind.PRELOAD_KILLS_CHILD_LOOP,
        title="One model crashes every process that loads it",
        action="Remove the model from your model list, then download it again before adding it back.",
        detail=(
            "A process that dies while loading a model is replaced, and the same model is sent to the new "
            "one, so no single process shows a pattern. Grouping the deaths by model is what reveals it. A "
            "file that fails a size or hash check against the model reference is corrupt on disk. One that "
            "verifies cannot be loaded by this build, and has to stay out of rotation until that is resolved."
        ),
        see_also=FindingKind.EMPTY_MODEL_POP_CASCADE,
    ),
    FindingSpec(
        kind=FindingKind.EMPTY_MODEL_POP_CASCADE,
        title="The horde sent jobs with no model name",
        detail=(
            "A job offer names the model to run. When the name is blank, a current worker refuses the job "
            "and never loads a nameless model or counts the blank name against a real one. The rest of the "
            "model list keeps serving. Each refused offer is still a job this worker could not serve."
        ),
        see_also=FindingKind.PRELOAD_KILLS_CHILD_LOOP,
    ),
    FindingSpec(
        kind=FindingKind.DOOMED_POOL_NO_GIVEUP,
        title="The worker kept restarting processes that could not recover",
        action="Fix the crash named in the crash-on-start finding, then restart the worker.",
        detail=(
            "When every image process has crashed on start the worker is meant to stop itself so a "
            "supervisor can restart it. Here it kept rebuilding the processes instead. The stop only fires "
            "when every process is out of use at the same moment, and a rebuild briefly brings one back, so "
            "the moment never came. Until that is changed in the worker, the crash itself is the fix."
        ),
        see_also=FindingKind.CRASH_ON_START_LOOP,
    ),
    FindingSpec(
        kind=FindingKind.GAVE_UP_CLEAN,
        title="The worker stopped itself after its processes could not be restored",
        action="No action needed here. Fix the crash named in the crash-on-start finding.",
        detail=(
            "Stopping is the intended outcome when repeated rebuilds cannot restore a working image process: "
            "a supervisor or service manager can then restart the worker once the cause is fixed. This is "
            "not a hang."
        ),
    ),
    FindingSpec(
        kind=FindingKind.STUCK_INFERENCE_STEP,
        title="A job repeated one sampling step and never finished",
        action=(
            "Check the process log just before the restart for a LoRA shape error, and stop pairing that "
            "LoRA with that model. If healthy jobs are being restarted, raise `inference_stuck_step_repeat_limit`."
        ),
        detail=(
            "The process kept reporting the same step, usually the last one, while still sending heartbeats, "
            "so the silence timeout could not catch it. The stuck-step check restarted it on the repeat count "
            "instead. The usual trigger is a LoRA built for one model family applied to another, such as an "
            "SD1.5 LoRA on an SDXL model. The log then shows a burst of shape errors before the final step "
            "hangs. Each occurrence lost the job in flight and held the process's VRAM until the restart."
        ),
    ),
    FindingSpec(
        kind=FindingKind.POST_PROCESSING_VRAM_STALL,
        title="Post-processing ran out of room on the card",
        detail=(
            "An upscale or face fix peaks after sampling, and that peak competes with the image models and "
            "the other processes on the same card. When it does not fit, ComfyUI falls back to slower tiled "
            "or streamed execution or stalls outright. A process can only free its own memory, so making room "
            "is the worker's job. With the VRAM budget on, the post-processing peak is priced before it starts "
            "and idle models are unloaded to fit it. The `post_processing_fault_breaker_enabled` breaker turns "
            "post-processing off after repeated stalls, so the worker stops taking jobs it cannot finish, and "
            "turns it back on when free memory recovers."
        ),
        reference_page="docs/explanation/performance_and_backpressure.md",
    ),
    FindingSpec(
        kind=FindingKind.ORPHAN_WEDGE,
        title="Jobs were dropped because their process disappeared",
        action="Look at the process crash, hang or memory findings above this one and fix those first.",
        detail=(
            "A job in progress belongs to one image process. When that process is gone and nothing has "
            "picked the job up, the worker drops it so the queue can move. A run of these means something "
            "keeps taking processes out, most often a card that hangs on every job or runs out of memory."
        ),
    ),
    # --- Memory and residency ---
    FindingSpec(
        kind=FindingKind.OOM,
        title="The card ran out of memory",
        action=(
            "Lower `max_threads` or `queue_size`, or turn on the VRAM budget. If several processes were "
            "sharing the card with almost nothing free, fewer models at once is the fix, not a smaller model."
        ),
        detail=(
            "The allocator's own message names the model that faulted and the other processes holding "
            "memory on the card at that moment. Many processes with near-zero free memory means too many "
            "models were admitted onto one card. One process with plenty free elsewhere means that model "
            "alone is too large for the card."
        ),
    ),
    FindingSpec(
        kind=FindingKind.SWALLOWED_OOM,
        title="Jobs failed with no images and no reason given",
        action=(
            "Check free VRAM around these failures. If the card was full, treat them as out-of-memory faults "
            "and lower `max_threads` or `queue_size`."
        ),
        detail=(
            'ComfyUI can report an out-of-memory error as a plain "no images were produced" failure. The '
            "worker's own memory-failure protection keys on the real error, so it may never engage even "
            "though memory was the cause."
        ),
    ),
    FindingSpec(
        kind=FindingKind.FILE_DESCRIPTOR_EXHAUSTION,
        title="A process ran out of file handles",
        action=(
            'Raise the open-file limit as a stopgap, with "ulimit -n" or "LimitNOFILE" in the service '
            "unit. Lowering memory settings will not help."
        ),
        detail=(
            "The leak is in the worker and these logs cannot say where, so it is worth reporting. "
            "Once a process passes its open-file limit every file open is refused, so it fails every job "
            "while still sending heartbeats, and the silence check cannot see it. The model named is "
            "whatever was running when the limit was hit, not the cause. The fault text is the same plain "
            '"produced no results" an out-of-memory fault uses, which is why this is easy to misread. The '
            "worker reports no open-file headroom yet, so the growth cannot be traced from the log. Windows "
            "has a far higher handle ceiling, so this is a Linux concern."
        ),
    ),
    FindingSpec(
        kind=FindingKind.SCHEDULER_STARVATION_WEDGE,
        title="The VRAM budget held the next job back on an idle card",
        action=(
            "Relax the VRAM budget, or turn off `unload_models_from_vram_often` and `high_performance_mode` to "
            "reduce swapping. Either lets the next job in before the wait triggers a rebuild."
        ),
        detail=(
            "The budget refused to load the next job's model even though the card had plenty free. That "
            "usually happens when processes are cycling so fast that the budget never sees a settled baseline "
            "to size each process from, so it over-charges every load. When the wait runs long enough the "
            "worker rebuilds its processes and drops the queued jobs, which the horde counts as dropped."
        ),
    ),
    FindingSpec(
        kind=FindingKind.UNSATISFIABLE_HEAD_STARVATION,
        title="One model kept being held back on an idle card",
        action=(
            "Check that the named model fits this card with room to run. If it does, relax the VRAM budget "
            "or turn off `unload_models_from_vram_often` and `high_performance_mode`. If it does not, remove "
            "the model."
        ),
        detail=(
            "The same model kept reaching the front of the queue and being refused a load while the card sat "
            "idle with memory to spare. Either the budget charges each process more than it costs, or the "
            "model genuinely cannot fit beside what the card holds. Over-charging happens when processes "
            "cycle so fast that no settled baseline exists to size them from. A bounded wait is meant to give "
            "up on such a job so the queue moves; when nothing ever cleared it, the queue was stuck behind it."
        ),
        see_also=FindingKind.SCHEDULER_STARVATION_WEDGE,
    ),
    FindingSpec(
        kind=FindingKind.RESIDENCY_RECONCILIATION_HOLDS,
        title="The worker paused to make room before running the next job",
        detail=(
            "The next job's model is loaded, but its working set will not fit beside an idle neighbour's "
            "model. The worker unloads that neighbour first and waits for the room to appear. Each pause is a "
            "swap paid for holding more models than the card fits at peak. A few are normal. Many are a "
            "throughput cost, and a pause that never clears means the room never came and something else "
            "had to break the stall."
        ),
        see_also=FindingKind.HEAD_DISPATCH_STALL,
    ),
    FindingSpec(
        kind=FindingKind.WHOLE_CARD_CONVERGENCE_WEDGE,
        title="A model that needs the whole card could not get it",
        action=(
            "Capture the log around the stall and report it. As a stopgap, lower `queue_size`, or do not "
            "serve a whole-card model beside a deep queue of others."
        ),
        detail=(
            "A whole-card model, such as Flux, is loaded into a spare process while the worker clears the "
            "rest of the card for it. Here something never left: an idle process still holding a model that "
            "is queued behind it, or a service process the clearing waits on. The card never emptied, the "
            "job was refused every cycle, and in the end the worker rebuilt its processes. The clearing is "
            "meant to stop exactly that neighbour, so this shape means it did not engage."
        ),
        see_also=FindingKind.HEAD_DISPATCH_STALL,
    ),
    FindingSpec(
        kind=FindingKind.WHOLE_CARD_NONHEAD_RESIDENCY_STARVATION,
        title="The card was reserved for a job that was not next in line",
        action=(
            "Capture the log around the stall and report it. The worker should only reserve the card for the next job."
        ),
        detail=(
            "Reserving the whole card tears down the neighbouring processes. Granting that to a job further "
            "back in the queue removes the processes serving the jobs ahead of it. The next job then has "
            "nowhere to run while the card is held for a turn that has not come. The queue stops, and if "
            "nothing breaks the stall the worker rebuilds its processes and drops the backlog."
        ),
        see_also=FindingKind.SCHEDULER_STARVATION_WEDGE,
    ),
    FindingSpec(
        kind=FindingKind.WHOLE_CARD_RESIDENCY_CHURN,
        title="The whole card was reserved and released over and over",
        detail=(
            "Reserving the whole card reduces the running processes and moves the safety check off the GPU; "
            "releasing it reverses both. A few times a session is a deliberate hold. Many times is thrash. "
            "On a large card that usually means a model that does not need the whole card is being given it, "
            "since each extra process was charged too much memory. Each round trip costs a "
            "model reload and a safety restart, and paired with rebuilt processes or dropped jobs it is the "
            "reservation feeding a stall."
        ),
        see_also=FindingKind.SCHEDULER_STARVATION_WEDGE,
    ),
    FindingSpec(
        kind=FindingKind.WHOLE_CARD_POP_CLAIM_EPISODES,
        title="The worker asked the horde for one model only while it held the card",
        action=(
            "Nothing to do if that is the heavy work this worker is for. If the claims are long and frequent "
            "for a model you did not mean to specialise in, trim the model list or lower "
            "`whole_card_residency_max_hold_seconds`."
        ),
        detail=(
            "While a whole-card model holds the card, the worker offers only that model to the horde so the "
            "loaded weights keep earning. The claim ends when the horde runs out of that model's jobs or when "
            "the hold cap is reached."
        ),
        see_also=FindingKind.WHOLE_CARD_POP_CLAIM_MONOPOLY,
    ),
    FindingSpec(
        kind=FindingKind.WHOLE_CARD_POP_CLAIM_MONOPOLY,
        title="Other jobs waited while the worker kept asking for one model",
        action=(
            "Decide whether this worker should specialise. If yes, remove the waiting models from its list. "
            "If it should serve a mix, lower `whole_card_residency_max_hold_seconds`, or drop the model that "
            "needs the whole card."
        ),
        detail=(
            "A claim on the offer is meant to end on its own when the horde runs out of that model's jobs. "
            "One that runs to the hold cap was still narrowing the offer when the cap stopped it. Doing that "
            "repeatedly while other models' jobs sat waiting means accepted work aged behind an offer that "
            "would not take anything to relieve it."
        ),
        see_also=FindingKind.WHOLE_CARD_POP_CLAIM_EPISODES,
    ),
    FindingSpec(
        kind=FindingKind.MODEL_CHURN,
        title="Models were loaded and unloaded far more often than jobs ran",
        action=(
            "Turn off `unload_models_from_vram_often`. Serve fewer models, or use the model pool, so the "
            "offered set fits the card. Raise `queue_size` so jobs for one model can run back to back."
        ),
        detail=(
            "A process that keeps its model runs the next job for it at no cost. Every swap costs a load from "
            "disk and the unload that made room for it, and a load cleared before it ran is that cost paid "
            "for nothing. On a multi-card host, work that will not spread across the cards forces the same few "
            "processes to serve every model in the queue, which shows up here as churn."
        ),
        see_also=FindingKind.MULTI_CARD_DISPATCH_SERIALIZATION,
    ),
    # --- Dispatch and throughput ---
    FindingSpec(
        kind=FindingKind.MULTI_CARD_DISPATCH_SERIALIZATION,
        title="Work is not spreading across the cards",
        action=(
            "Run `horde-log jobs` for the per-job split and the busy-cards histogram, then report it with the "
            "log. Lowering `max_power` or `queue_size` only shrinks the queue, it does not spread it."
        ),
        detail=(
            "Jobs were waiting with their model already loaded on an idle process, while that process's card "
            "ran nothing. The loss is queue wait, not GPU speed. What serialises work across cards is the "
            "scheduler, not per-card settings. The usual causes are an ordering that will not start a job "
            "behind the next one on a free card, a whole-card reservation, or a per-card lease."
        ),
        see_also=FindingKind.MODEL_CHURN,
    ),
    FindingSpec(
        kind=FindingKind.HEAD_DISPATCH_STALL,
        title="The next job kept being held back",
        detail=(
            "The scheduler names the rule that held the next job each time: a concurrency cap, an overlap "
            "wait, or a load the VRAM budget put off. Sustained, that starves throughput even though nothing "
            "is stuck. A hold with no rule named is different: the model was loaded on an idle process and "
            "nothing was in the way, so the scheduler itself is at fault."
        ),
    ),
    FindingSpec(
        kind=FindingKind.SLOW_GENERATION_DROP_SPIRAL,
        title="Jobs took too long and were dropped",
        detail=(
            "The horde gives each job a deadline. A job that misses it is dropped and counted against the "
            "worker, and enough drops put the worker into maintenance. Where the time went matters. Waiting "
            "to start is a scheduling problem, and generation itself is a settings problem. Waiting after "
            "generation means a stage behind inference, usually the safety check, cannot keep up. The worker "
            "stops taking new jobs while the safety backlog cannot clear within the deadline, which bounds it."
        ),
        see_also=FindingKind.FORCED_MAINTENANCE,
    ),
    FindingSpec(
        kind=FindingKind.POST_PROCESSING_DEFERRAL_STARVATION,
        title="Post-processing waited for room that never came",
        action=(
            "Report it with the log around the wait. As a stopgap, serve fewer models on that card or turn off "
            "post-processing on this worker."
        ),
        detail=(
            "Post-processing runs on its own process and asks for room on the card before each chain. Here the "
            "answer stayed no for far longer than any chain could need. Either the room arithmetic was wrong "
            "for that card, or the job should have been given up on and let the ones behind it pass, and "
            "neither happened."
        ),
        see_also=FindingKind.POST_PROCESSING_VRAM_STALL,
        reference_page="docs/explanation/process_lanes_and_chaining.md",
    ),
    FindingSpec(
        kind=FindingKind.SAFETY_STAGE_STALL,
        title="The safety check fell behind or lost results",
        detail=(
            "Every finished image goes through one safety process before it is sent back. When that process "
            "loses a result or stops answering, finished work piles up behind it; when the pile is deep enough "
            "the jobs pass their deadline and are dropped."
        ),
    ),
    FindingSpec(
        kind=FindingKind.SAFETY_STAGE_CAPACITY,
        title="One safety process cannot keep up with the cards",
        action=(
            "Turn on `safety_on_gpu` so a check takes a fraction of the time. Lowering `max_power` or "
            "`queue_size` will not help: every image still goes through the same single checker."
        ),
        detail=(
            "Safety is one process for the whole worker, so its rate is fixed while the cards' finish rate "
            "grows with every card added. Results queue in front of it, and the wait between an image "
            "finishing and its safety result grows with the gap."
        ),
        see_also=FindingKind.SAFETY_STAGE_STALL,
        reference_page="docs/explanation/process_lanes_and_chaining.md",
    ),
    FindingSpec(
        kind=FindingKind.POP_LIVENESS_FULL_QUEUE,
        title="The queue was full and nothing was moving",
        detail=(
            "The worker holds a short queue of accepted jobs and asks the horde for more only when it has "
            "room. A full queue with nothing starting and nothing finishing is a stall, not the worker "
            "pacing itself. The log names what the next job was waiting for where the scheduler knows."
        ),
        see_also=FindingKind.WHOLE_CARD_RESIDENCY_CHURN,
    ),
    FindingSpec(
        kind=FindingKind.PARENT_LOOP_STALL,
        title="The main process stopped listening to its workers",
        action=(
            "Run `horde-log timeline` over the gap to see what the main process was doing just before it went "
            "quiet, then report it."
        ),
        detail=(
            "The main process reads messages from every worker process on one loop. A slow call on that "
            "loop blocks everything behind it, so nothing is started, finished or sent back until it returns. "
            "The usual culprits are a model reference read, a disk scan or a network call without a timeout. "
            "The gaps below are bounded by the timestamps around them."
        ),
        see_also=FindingKind.MULTI_CARD_DISPATCH_SERIALIZATION,
    ),
    FindingSpec(
        kind=FindingKind.LANE_PLACEMENT,
        title="Helper processes moved to a different card",
        action=(
            "Compare that card's duty and free VRAM with the rest before reading its low duty as a GPU "
            "problem. If the move was not intended, set `safety_on_gpu` and the post-processing card so a "
            "restart puts them back."
        ),
        detail=(
            "The safety, post-processing and utilities processes are placed on a card when they start. A "
            "restart can land one on a different card, or stack several on one, and that card's spare "
            "memory and duty change with them."
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
