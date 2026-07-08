"""Turn a correlated session into ranked, actionable findings: what went wrong and what to do.

Each detector recognizes one incident class from the signals the worker already emits (recovery
diagnostics, the action ledger, child tracebacks) and returns a :class:`Finding` with a plain-language
verdict, the evidence that supports it, and a remediation. This is the automated form of the manual log
archeology a maintainer would otherwise do: the crash-on-start detector lifts the child's exception
across the process boundary; the doomed-pool detector recognizes the save-our-ship loop that spins
without ever giving up.

Detectors are independent and registered in :data:`DETECTORS`, so a new incident class is one function
plus one list entry. They never raise: a detector that cannot make sense of a session returns no
findings rather than aborting the report.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from horde_worker_regen.utils.oom_signature import OOM_TEXT_RE

from .correlate import SessionContext, find_child_crash
from .governor_signatures import GOVERNOR_ENTER_RE, GOVERNOR_EXIT_RE, GOVERNOR_LABELS
from .log_ingest import LogRecord
from .sessions import SessionEndReason

# Signatures over orchestrator message text.
_QUARANTINE_RE = re.compile(r"quarantined \(crash on start")
_SOFT_RESET_RE = re.compile(r"Save-our-ship soft reset")
_POOLS_RECOVERED_RE = re.compile(r"pools recovered.*limp-by cleared")
_ABANDON_SHIP_RE = re.compile(r"abandoning ship|cannot restore a working process pool")
# The live worker reclaims-and-retries on this same fingerprint; keep the signature single-sourced.
_OOM_RE = OOM_TEXT_RE
# A faulted-inference result names its model and the failing node in a stable shape:
#   "... produced no results. Model: <name>. Error: <stage> (<NodeClass>): <underlying error>"
# Both the OOM and the file-descriptor detectors read the model off this to name the culprit, because the
# outer "Pipeline failed to run ... produced no results" wrapper is identical across unrelated root causes.
_FAULT_MODEL_RE = re.compile(r"Model: (?P<model>.+?)\. Error:")
_FAULTED_ON_PROCESS_RE = re.compile(r"faulted on process (?P<pid>\d+)")
# The CUDA allocator's own accounting from an OOM message: how little was free, and the sibling processes
# co-resident on the card. Several siblings each holding GiB with almost nothing free is the over-admission
# fingerprint (many models sharing one card), distinct from a single model that simply will not fit.
_OOM_FREE_VRAM_RE = re.compile(r"of which ([\d.]+) MiB is free")
_OOM_SIBLING_RE = re.compile(r"Process \d+ has ([\d.]+) GiB memory in use")
# The per-process file-descriptor ceiling (RLIMIT_NOFILE / EMFILE, errno 24). The kernel message is
# "Too many open files" (distinct from the system-wide ENFILE "... in system"); it arrives either as an
# os/psutil "[Errno 24] Too many open files: '<path>'" or as safetensors' "Too many open files (24)".
_FD_EXHAUSTION_RE = re.compile(r"Too many open files(?! in system)")
# The resource whose open() was refused, naming where the exhaustion bit: a /proc probe (psutil's
# free-RAM read, or the child's own /proc/<pid>/stat control-message read) or a checkpoint/LoRA .safetensors.
_FD_RESOURCE_RE = re.compile(r"Too many open files: '(?P<path>[^']+)'|open file <(?P<file>[^>]+)> in read-only mode")
_NO_IMAGES_RE = re.compile(r"no images were produced|no images produced")
# The in-progress orphan watchdog names itself in its punt line ("...(orphaned-job watchdog).") rather
# than using the words "orphaned in-progress", so the watchdog tag is the signature that actually
# matches the emitted text; the other alternatives stay for forward-compatibility and the ledger reason.
_ORPHAN_RE = re.compile(r"orphaned? in-progress|punt(?:ing|ed) (?:an? )?orphan|orphaned-job watchdog")
# The horde rejecting a pop because it forced the worker into maintenance, and the (server-supplied)
# reason it gives. "dropping too many jobs" is the worker's own fault and the actionable case; any other
# maintenance (operator-set, key issue) is informational.
_MAINTENANCE_POP_RE = re.compile(r"Failed to pop job \(Maintenance Mode\)")
_DROPPING_JOBS_RE = re.compile(r"dropping too many jobs")
# Save-our-ship faulting unservable backlog jobs (the "dropped jobs" the horde counts against the worker).
_GIVE_UP_RE = re.compile(r"gave up on (\d+) unservable job")
# The scheduler starving: the VRAM budget deferred the head-of-queue on an idle device, with the
# starvation duration and the free VRAM that proves the budget was over-conservative.
_FORCE_ADMIT_RE = re.compile(r"budget-deferred on an idle device for (\d+)s")
_DEVICE_FREE_VRAM_RE = re.compile(r"device_free_vram=(\d+)MB")
# The worker self-pausing pops after three consecutive faults.
_CONSECUTIVE_PAUSE_RE = re.compile(r"Too many consecutive failed jobs, pausing job pops")
# The horde aborting a generation server-side because the worker submitted it after the per-job deadline
# (the verbatim server message the submitter logs). Each such abort is a faulted job the horde counts
# against the worker, and a *sustained* run of them is the slow-generation death spiral that ends in
# forced maintenance, distinct from save-our-ship give-ups.
_SERVER_SLOW_ABORT_RE = re.compile(r"took too long to process and has been aborted")
# The worker-side corroboration: the inference grader flagging a job running N-times its expected
# sampling time, with the residency snapshot (free VRAM) that fingerprints an over-committed device.
_SLOWDOWN_GRADE_RE = re.compile(r"is ([\d.]+)x its expected sampling time")
# Each successful submit reports how long the job spent between pop and submit, and how long generation
# itself took. A large gap between the two means jobs aged in the pipeline (typically the single safety
# stage backing up), not in generation: a different cause, and fix, than a genuinely slow GPU.
_SUBMIT_LATENCY_RE = re.compile(r"Job popped ([\d.]+) seconds ago and took ([\d.]+) to generate")
# The wall-clock the safety stage took per check; a high average is the safety stage being the pipeline
# bottleneck (e.g. CPU safety with safety_on_gpu off).
_SAFETY_DURATION_RE = re.compile(r"took ([\d.]+) seconds to check safety")
# Safety-stage stall signals. A verdict that never returned strands a job in SAFETY_CHECKING; the worker
# now re-checks it (requeue), faults it with no image when the pipeline cannot check it (unrecoverable),
# soft-pauses pops while safety is unreliable, and throttles intake when the safety backlog is too deep.
# The dispatcher's "none was found" is the original lost-result signal that strands the job.
_SAFETY_REQUEUE_RE = re.compile(r"requeued it for a fresh safety check")
_SAFETY_UNRECOVERABLE_RE = re.compile(r"could not be safety-checked")
_SAFETY_SOFT_PAUSE_RE = re.compile(r"Soft-pausing job pops.*safety could not check a result")
_SAFETY_BACKPRESSURE_RE = re.compile(r"Withholding job pops: post-inference safety backlog (\d+) >= cap (\d+)")
_LOST_SAFETY_RESULT_RE = re.compile(r"Expected to find a completed job .* none was found")
# The scheduler explaining why a head-of-queue job is not dispatching despite pending work. The
# "no matching gate" variant is the scheduler-bug-shaped stall (model resident and idle, nothing blocking
# it, yet nothing dispatched).
_DISPATCH_STALL_RE = re.compile(r"Inference dispatch stalled: head ")
_DISPATCH_STALL_BUG_RE = re.compile(r"dispatch was withheld with no matching gate")
# The whole-card residency convergence deadlock: a heavy head is pre-staged and waiting for sole residency,
# but an idle sibling holds a model that is still queued behind it, so the scale-down guard protects that
# sibling from the teardown and the residency never collapses. The head is parked until the recovery
# supervisor soft-resets the pools. This is a distinct, nameable root cause (not a generic dispatch-path bug),
# so it gets its own detector; the phrase is the worker's _diagnose_dispatch_stall attribution for it.
_WHOLE_CARD_WEDGE_RE = re.compile(r"whole-card residency stuck: cannot reach sole residency")
# A whole-card residency granted to a model that is not the head of the queue: it reserves the card and tears
# its siblings down, so the actual head (a different model) cannot load and starves. Reads as a generic
# VRAM-budget defer (the card looks idle) unless attributed to the held non-head residency, so it gets its own
# detector keyed on the worker's _diagnose_dispatch_stall phrase for it.
_WHOLE_CARD_NONHEAD_RE = re.compile(r"whole-card residency is held for non-head model")
# A whole-card residency being established: the worker reserved the device for a model, tearing the process
# pool down to fewer contexts (and cycling safety off-GPU). One is routine; many in a session is reservation
# churn: the signature of models being driven onto the whole-card path that do not need it (on a high-VRAM
# card a model whose weights are a small fraction of total VRAM co-resides, so a teardown demand for it usually
# means the per-context overhead was over-counted). The phrase is the worker's establish-announce line.
_WHOLE_CARD_ESTABLISH_RE = re.compile(r"Whole-card residency: reserving the device for")
# The worker declining to reserve the card for a model whose teardown demand it does not trust (a card-light
# model on a host with no measured per-context cost). Surfaced as the positive counterpart: it confirms the
# trust gate is actively preventing reservation churn rather than the churn simply being absent.
_WHOLE_CARD_DECLINED_RE = re.compile(r"Declined a whole-card residency for")

# The stuck-step watchdog reaping a slot whose ComfyUI generation looped on one sampling step. The slot
# kept heart-beating (so the silence watchdog stayed blind), which is exactly why this needs its own
# detector rather than folding into a generic hang. The phrase is the worker's verbatim reap line.
_STUCK_STEP_RE = re.compile(r"stuck on a non-advancing sampling step|stuck-step watchdog")

# The post-processing-stage watchdog reaping a slot that went silent. Older workers reported this as
# INFERENCE_POST_PROCESSING; dedicated-lane workers report POST_PROCESS / POST_PROCESSING. The peak is still
# an upscaler/face-fixer allocation landing after sampling, concurrent with warm inference siblings.
_POST_PROCESSING_STALL_RE = re.compile(r"seems to be stuck post processing")
_DEDICATED_POST_PROCESS_RE = re.compile(
    r"\bPOST_PROCESS(?:\)|ING\b)|Post-processing (?:job|for job|finished)|dedicated post-process",
    re.IGNORECASE,
)
_LOW_VRAM_STREAM_RE = re.compile(
    r"Free VRAM: \d+ MB.*(?:below .*inference reserve|reclaimable torch cache|stream|til)",
    re.IGNORECASE,
)
# The feature-level circuit breaker disabling post-processing after a run of over-commit faults. Its trip
# line is the operator advisory: it confirms the spiral reached the self-protective latch (post-processing is
# now off until restart), so a session carrying it is escalated and the remediation points at the restart +
# downgrade. The phrase is the worker's verbatim breaker-trip line (process_manager).
_POST_PROCESSING_BREAKER_RE = re.compile(r"Post-processing fault breaker tripped")

# A median pop->submit latency this many times the median generation time means jobs are aging in the
# pipeline queue, not in generation (the post-inference safety-backlog signature).
_QUEUE_AGING_LATENCY_RATIO = 3.0

# A run of server-side slow-aborts at or above this is a spiral (the horde will force maintenance),
# not a stray slow job.
_SLOW_ABORT_SPIRAL_THRESHOLD = 3

# A whole-card residency established at or above this many times in a session is reservation churn (the
# process pool repeatedly torn down and rebuilt, safety cycled off/on the GPU), not a single deliberate hold.
_WHOLE_CARD_CHURN_THRESHOLD = 3

# A pool that flapped through at least this many soft resets is "stuck recovering", not a one-off blip.
_SOFT_RESET_FLAP_THRESHOLD = 2
# A recovery count at or above this is a storm worth surfacing on its own.
_RECOVERY_STORM_THRESHOLD = 5


class Severity(enum.StrEnum):
    """How urgent a finding is; also its sort key (critical first)."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


_SEVERITY_ORDER = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}


@dataclass
class Finding:
    """One diagnosis of a session: the verdict, the evidence, and what to do about it."""

    id: str
    severity: Severity
    title: str
    verdict: str
    remediation: str
    evidence: list[str] = field(default_factory=list)
    see_also: str | None = None


Detector = Callable[[SessionContext], "list[Finding]"]


def _matching(records: list[LogRecord], pattern: re.Pattern[str]) -> list[LogRecord]:
    """Orchestrator records whose message matches ``pattern``."""
    return [record for record in records if pattern.search(record.message)]


def _record_in_session(context: SessionContext, record: LogRecord) -> bool:
    """Return whether ``record`` belongs to the session's wall-clock window."""
    if record.timestamp is None:
        return False
    start, end = context.session.start_ts, context.session.end_ts
    if start is not None and record.timestamp < start:
        return False
    return not (end is not None and record.timestamp > end)


def _child_records_in_session(context: SessionContext) -> list[LogRecord]:
    """Return child loop records whose timestamps fall inside the diagnosed session."""
    records: list[LogRecord] = []
    for process_id in context.bundle.process_ids():
        records.extend(
            record for record in context.bundle.child_records(process_id) if _record_in_session(context, record)
        )
    return records


def _median(values: list[float]) -> float | None:
    """The median of ``values`` (robust to the warm-up/recalibration outliers in timing logs), or None."""
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _mean(values: list[float]) -> float:
    """The arithmetic mean of ``values`` (0.0 for an empty list)."""
    return sum(values) / len(values) if values else 0.0


def _evidence(record: LogRecord) -> str:
    """A one-line evidence reference: timestamp, source location, and a clipped message."""
    ts = record.timestamp.strftime("%H:%M:%S") if record.timestamp else "--:--:--"
    message = record.message if len(record.message) <= 140 else record.message[:137] + "..."
    return f"{ts}  {record.source_path.name}:{record.raw_lineno}  {message}"


def detect_crash_on_start_loop(context: SessionContext) -> list[Finding]:
    """Inference children that crash during startup, with the child's exception as the root cause."""
    crashing = [r for r in context.recoveries if r.last_state == "PROCESS_STARTING" and r.exitcode not in ("0", "")]
    if len(crashing) < _SOFT_RESET_FLAP_THRESHOLD:
        return []

    # Lift the actual exception from each affected slot's child startup log (the cross-process join).
    exceptions: dict[str, int] = {}
    evidence: list[str] = []
    for process_id in sorted({r.process_id for r in crashing}):
        first = next(r for r in crashing if r.process_id == process_id)
        crash = find_child_crash(context.bundle, process_id, first.timestamp, os_pid=first.os_pid)
        if crash is not None:
            exceptions[crash.exception] = exceptions.get(crash.exception, 0) + 1
            evidence.append(f"slot {process_id}: {_evidence(crash.record).split('  ', 1)[1]} -> {crash.exception}")

    cause = max(exceptions, key=lambda exc: exceptions[exc]) if exceptions else None
    verdict = (
        f"{len(crashing)} inference start(s) across {len({r.process_id for r in crashing})} slot(s) crashed before "
        "reaching readiness"
    )
    if cause is not None:
        verdict += f"; the child raised `{cause}`."
        remediation = (
            f"The inference subprocess fails during hordelib/ComfyUI init with `{cause}`. Fix that "
            "environment fault (e.g. reinstall a CUDA-enabled torch if it reports torch was not compiled "
            "with CUDA); the worker cannot serve until the children start."
        )
    else:
        verdict += " (no child traceback found to attribute)."
        remediation = (
            "Inspect the affected slot's bridge_inference_<N>_startup.log for the failing import/exception; "
            "the parent only sees a nonzero exit code."
        )
    return [
        Finding(
            id="crash_on_start_loop",
            severity=Severity.CRITICAL,
            title="Inference pool crashes on start",
            verdict=verdict,
            remediation=remediation,
            evidence=evidence[:6] or [_evidence(crashing[0].record)],
        ),
    ]


def detect_doomed_pool_no_giveup(context: SessionContext) -> list[Finding]:
    """The save-our-ship loop spun on an unrecoverable pool instead of giving up (the observed bug)."""
    session = context.session
    soft_resets = _matching(session.records, _SOFT_RESET_RE)
    recovered = _matching(session.records, _POOLS_RECOVERED_RE)
    quarantined = _matching(session.records, _QUARANTINE_RE)
    gave_up = bool(_matching(session.records, _ABANDON_SHIP_RE))

    # The defining symptom: the pool reached full quarantine (proved unrecoverable) yet the worker kept
    # going (a soft-reset recovery and/or a recovery storm) instead of abandoning ship. Soft-reset count
    # is supporting evidence, not a gate: the slow-restart variant of the bug flaps with as few as one
    # soft reset per episode because each episode closes on a long clean window.
    unrecoverable_seen = bool(quarantined)
    kept_going = bool(recovered) or session.peak_process_recoveries >= _RECOVERY_STORM_THRESHOLD
    if gave_up or not (unrecoverable_seen and kept_going):
        return []

    verdict = (
        f"The pool quarantined and was soft-reset {len(soft_resets)} time(s), recovering "
        f"{len(recovered)} time(s), and reached {session.peak_process_recoveries} process recoveries, but the "
        f"worker never abandoned ship (it ended via {session.end_reason}). A deterministically-doomed pool "
        "flapped between soft reset and re-crash instead of self-terminating."
    )
    return [
        Finding(
            id="doomed_pool_no_giveup",
            severity=Severity.CRITICAL,
            title="Recovery storm never gave up",
            verdict=verdict,
            remediation=(
                "Pair with the crash-on-start root cause: the pool cannot recover, so it should give up "
                "fast. The give-up abort only fires when every slot is quarantined at the exact give-up "
                "tick, but a soft reset's transient un-quarantine (and a clean window longer than the "
                "recovery clean streak) keeps that from coinciding, so the worker spins. Make the abort "
                "latch 'was fully quarantined this episode' rather than sampling the instantaneous state."
            ),
            evidence=[_evidence(r) for r in (soft_resets[:2] + recovered[:1] + quarantined[:1])],
            see_also="recovery_supervisor give-up phase mismatch",
        ),
    ]


def detect_gave_up_clean(context: SessionContext) -> list[Finding]:
    """The worker correctly abandoned ship on an unrecoverable pool (the healthy terminal path)."""
    abandon = _matching(context.session.records, _ABANDON_SHIP_RE)
    if not abandon:
        return []
    return [
        Finding(
            id="gave_up_clean",
            severity=Severity.INFO,
            title="Worker gave up on an unrecoverable pool",
            verdict=(
                "Save-our-ship abandoned ship and self-terminated after soft resets could not restore a "
                "working pool. This is the intended bail-out, not a hang; see the crash-on-start finding "
                "for why the pool was unrecoverable."
            ),
            remediation="No worker action needed beyond fixing the underlying crash cause; the bail-out worked.",
            evidence=[_evidence(abandon[0])],
        ),
    ]


def detect_stuck_inference_step(context: SessionContext) -> list[Finding]:
    """A slot wedged repeating one sampling step, which the silence-based hang watchdog cannot see."""
    stuck = _matching(context.session.records, _STUCK_STEP_RE)
    if not stuck:
        return []
    return [
        Finding(
            id="stuck_inference_step",
            severity=Severity.WARNING,
            title="Inference wedged on a non-advancing step",
            verdict=(
                f"{len(stuck)} time(s) an inference slot looped on a single sampling step (in practice the "
                "final step) and never returned a result, while still emitting heartbeats. The slot was not "
                "silent, so the per-step silence timeout could not catch it; the stuck-step watchdog reaped "
                "it on the child's non-advancing-repeat count instead. Each occurrence stranded the in-flight "
                "job and held the slot's VRAM until the reap."
            ),
            remediation=(
                "Recovery worked, but the hang is upstream in ComfyUI/hordelib. The usual trigger is a "
                "corrupt or incompatible model+LoRA combination: e.g. an SD1.5 LoRA applied to an SDXL "
                "checkpoint produces a `ERROR lora ... shape ... is invalid` storm and then the pipeline "
                "hangs at the final step. Check the affected slot's bridge_<N>.log just before the reap for "
                "that shape-mismatch storm and exclude the offending LoRA/model pairing. If healthy jobs are "
                "being reaped, raise `inference_stuck_step_repeat_limit`."
            ),
            evidence=[_evidence(r) for r in stuck[:4]],
        ),
    ]


def detect_post_processing_vram_stall(context: SessionContext) -> list[Finding]:
    """Post-processing overlapped with generation until ComfyUI fell back to low-VRAM streaming.

    The dedicated post-processing lane keeps the upscaler/face-fixer work out of the inference process, but
    it still allocates on the same GPU. A low-free-VRAM warning while that lane is active means the
    post-processing peak and concurrent inference residency were admitted against the same device headroom.
    """
    stalls = _matching(context.session.records, _POST_PROCESSING_STALL_RE)
    breaker_trips = _matching(context.session.records, _POST_PROCESSING_BREAKER_RE)
    child_records = _child_records_in_session(context)
    dedicated_activity = _matching(context.session.records, _DEDICATED_POST_PROCESS_RE) + _matching(
        child_records,
        _DEDICATED_POST_PROCESS_RE,
    )
    low_vram_warnings = _matching(child_records, _LOW_VRAM_STREAM_RE)
    if not stalls and not breaker_trips and not (dedicated_activity and low_vram_warnings):
        return []
    post_processing_recoveries = [
        r for r in context.recoveries if r.last_state in {"INFERENCE_POST_PROCESSING", "POST_PROCESSING"}
    ]
    dropped = _total_dropped_jobs(context.session.records)
    forced_maintenance = bool(_matching(context.session.records, _MAINTENANCE_POP_RE))
    escalated = dropped > 0 or forced_maintenance or bool(breaker_trips)
    verdict = (
        f"{len(stalls)} post-processing watchdog reap(s), {len(low_vram_warnings)} child low-free-VRAM "
        "warning(s), and dedicated post-processing activity were observed in the same session. The "
        "upscaler/face-fixer peak that lands after sampling was competing with inference models and CUDA "
        "contexts on the same card, pushing ComfyUI toward tiled/streaming execution instead of fast in-VRAM "
        "sampling. ComfyUI can only release this process's own cache; sibling process models and contexts are "
        "reclaimable only by the orchestrator."
    )
    if breaker_trips:
        verdict += (
            " The self-protective breaker tripped: post-processing is now disabled on this worker for the rest "
            "of the session (it kept being handed jobs it could not host)."
        )
    if dropped > 0 or forced_maintenance:
        verdict += f" It escalated: {dropped} backlog job(s) were faulted" + (
            " and the horde forced the worker into maintenance." if forced_maintenance else "."
        )
    return [
        Finding(
            id="post_processing_vram_stall",
            severity=Severity.CRITICAL if escalated else Severity.WARNING,
            title="Post-processing stalled on an over-committed card",
            verdict=verdict,
            remediation=(
                "Run with the VRAM budget enabled on a build where the dedicated post-processing lane "
                "participates in committed-reserve accounting and idle VRAM reclaim. As a stopgap, lower "
                "concurrency/queue or disable post-processing on this card; a 4x upscale needs several GB "
                "free at peak that a multi-context card may not spare. The "
                "`post_processing_fault_breaker_enabled` breaker disables post-processing automatically after "
                "repeated stalls so the worker stops feeding the forced-maintenance spiral"
                + (
                    "; it has already tripped here, so restart the worker after downgrading settings to "
                    "restore post-processing."
                    if breaker_trips
                    else "."
                )
            ),
            evidence=[_evidence(r) for r in (stalls[:4] + low_vram_warnings[:4] + breaker_trips[:1])]
            + [_evidence(r.record) for r in post_processing_recoveries[:2]],
            see_also="vram_ram_budget_subsystem",
        ),
    ]


_PP_DEFER_RE = re.compile(r"Deferring post-processing for job ([0-9a-f][0-9a-f-]{7,35})")
_PP_FINISHED_RE = re.compile(r"Post-processing finished for job")

# A handful of deferrals is healthy backpressure while a transient VRAM spike passes; the same job
# deferred this many times means its headroom condition is structurally unsatisfiable on this card.
_PP_DEFER_STARVATION_THRESHOLD = 30
_PP_DEFER_WARNING_THRESHOLD = 10


def detect_post_processing_deferral_starvation(context: SessionContext) -> list[Finding]:
    """The post-processing lane deferring the same job indefinitely instead of serving or faulting it.

    The lane's admission gate compares a job's estimated peak (plus the configured reserve) against the
    card's free VRAM after commitments. When that inequality can never be satisfied for the head of the
    pending queue, the head is deferred every scheduling tick, everything behind it waits, and the
    worker never reports the job faulted. The signature is one job accumulating deferral warnings by the
    tens to hundreds; when no lane completion lands after the storm begins, the whole lane is starved, not
    just the head.
    """
    defer_records: dict[str, list[LogRecord]] = {}
    for record in context.session.records:
        match = _PP_DEFER_RE.search(record.message)
        if match is not None:
            defer_records.setdefault(match.group(1), []).append(record)
    if not defer_records:
        return []

    worst_job, worst = max(defer_records.items(), key=lambda item: len(item[1]))
    if len(worst) < _PP_DEFER_WARNING_THRESHOLD:
        return []

    storm_start = next((r.timestamp for r in worst if r.timestamp is not None), None)
    completions_after = [
        r
        for r in _matching(context.session.records, _PP_FINISHED_RE)
        if storm_start is None or (r.timestamp is not None and r.timestamp >= storm_start)
    ]
    lane_fully_starved = not completions_after
    starved = len(worst) >= _PP_DEFER_STARVATION_THRESHOLD

    verdict = (
        f"Job {worst_job} was deferred by the post-processing admission gate {len(worst)} time(s) "
        f"({len(defer_records)} job(s) deferred in total). Its estimated peak plus the VRAM reserve never "
        "fit the card's free-after-commitments figure, and no bounded no-image fault ever released it, so its "
        "finished inference was held unsubmitted."
    )
    if lane_fully_starved:
        verdict += (
            " No post-processing completed after the deferrals began: the head of the queue starved the "
            "entire lane (head-of-line blocking)."
        )
    return [
        Finding(
            id="post_processing_deferral_starvation",
            severity=Severity.CRITICAL if starved and lane_fully_starved else Severity.WARNING,
            title="Post-processing lane starved by its admission gate",
            verdict=verdict,
            remediation=(
                "Verify the admission inputs: the free-VRAM figure must reflect the lane's card (not a stale "
                "minimum across children) and must not re-subtract commitments already realised in the "
                "measurement, and the per-chain peak estimate must match measured op costs. A deferred job "
                "must age out to a no-image fault after a bounded wait instead of parking forever, and "
                "fittable jobs behind an unfittable head must be dispatched ahead of it. As a stopgap, "
                "lower resident VRAM on the lane's card (fewer processes or models) or disable "
                "post-processing on this worker."
            ),
            evidence=[_evidence(r) for r in (worst[:2] + worst[-2:])],
            see_also="process_lanes_and_chaining",
        ),
    ]


def _distinct_ordered(values: list[str]) -> list[str]:
    """The distinct members of ``values`` in first-seen order (a small, stable, de-duplicated set)."""
    seen: dict[str, None] = {}
    for value in values:
        seen.setdefault(value, None)
    return list(seen)


def _faulting_models(records: list[LogRecord]) -> list[str]:
    """The distinct model names named across a set of faulted-inference records (the culprit models)."""
    models = [m.group("model") for r in records if (m := _FAULT_MODEL_RE.search(r.message))]
    return _distinct_ordered(models)


def _affected_slots(records: list[LogRecord]) -> list[int]:
    """The distinct inference slot numbers named by 'faulted on process N' across ``records``, sorted."""
    slots = {int(m.group("pid")) for r in records if (m := _FAULTED_ON_PROCESS_RE.search(r.message))}
    return sorted(slots)


def _clause_join(items: list[str]) -> str:
    """Join names as 'a', 'a and b', or 'a, b and c' for a readable inline clause."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def detect_oom(context: SessionContext) -> list[Finding]:
    """Out-of-memory faults (explicit CUDA OOM), naming the faulting model and the card's co-residency.

    The bare fault count does not tell a maintainer whether one model is simply too large for the card or
    many models were over-admitted onto it. The allocator's own message carries both: the model that
    faulted, and the sibling processes holding memory with almost nothing free. Naming them turns a plain
    count into a finding that identifies the faulting model and whether the card was over-committed (many
    processes co-resident with near-zero free VRAM), which points at the fix.
    """
    oom = _matching(context.session.records, _OOM_RE)
    if not oom:
        return []

    models = _faulting_models(oom)
    slots = _affected_slots(oom)
    free_vrams = [float(m.group(1)) for r in oom if (m := _OOM_FREE_VRAM_RE.search(r.message))]
    sibling_counts = [len(_OOM_SIBLING_RE.findall(r.message)) for r in oom]
    max_siblings = max(sibling_counts, default=0)

    verdict = f"{len(oom)} out-of-memory fault(s) during the session"
    if models:
        verdict += f", faulting {_clause_join([f'`{m}`' for m in models])}"
    if slots:
        verdict += f" on slot(s) {_clause_join([str(s) for s in slots])}"
    verdict += "."
    if free_vrams:
        verdict += f" The allocator reported as little as {min(free_vrams):.0f} MiB free at the fault"
        if max_siblings:
            # A sibling count here is (co-resident processes) - 1 (the message excludes the faulting
            # process's own line), so +1 to state the total sharing the card.
            verdict += (
                f", with {max_siblings + 1} processes co-resident on the card: the over-admission "
                "fingerprint (many models sharing one device), not a single model too large to fit"
            )
        verdict += "."

    return [
        Finding(
            id="oom",
            severity=Severity.CRITICAL,
            title="GPU out-of-memory faults",
            verdict=verdict,
            remediation=(
                "Reduce concurrency/queue or enable a more conservative VRAM budget; if these recur under "
                "a budget that should fit, suspect over-admission of a heavy head (Flux fp8 / SDXL). The "
                "named co-residency and free-VRAM figures say which: several co-resident processes with "
                "near-zero free VRAM points at too many models admitted onto one card, not at the faulting "
                "model being individually oversized."
            ),
            evidence=[_evidence(r) for r in oom[:4]],
        ),
    ]


def detect_file_descriptor_exhaustion(context: SessionContext) -> list[Finding]:
    """An inference process that ran its descriptor table into RLIMIT_NOFILE (EMFILE, errno 24).

    A descriptor leak in one inference child climbs until every ``open()`` is refused. The exhaustion then
    surfaces wherever a file is next opened, which is misleadingly far from the leak: psutil's free-RAM
    probe cannot read ``/proc/meminfo`` (it runs on every tqdm progress redraw during sampling), the
    checkpoint and LoRA ``.safetensors`` cannot be opened, and even the child's own control-message handler
    cannot read ``/proc/<pid>/stat``. From that point the process faults every job it is handed, yet it
    keeps heart-beating, so the silence-based hang watchdog stays blind; only the "failed to load model"
    recovery path eventually replaces the slot, after a long poisoned window of dropped jobs.

    This is a resource leak, not memory capacity, and it wears the same generic "Pipeline failed to run ...
    produced no results" wrapper as a CUDA OOM. Without its own detector it is read as an OOM and given the
    wrong remediation (reduce concurrency / VRAM budget), which does nothing for a descriptor leak. The
    faulting model named here is the job that happened to be running when the ceiling was hit, not the
    cause: the cause is whatever leaked descriptors, which these logs cannot pinpoint because the worker
    emits no descriptor-headroom telemetry (the actionable gap this finding calls out).
    """
    faults = _matching(context.session.records, _FD_EXHAUSTION_RE)
    if not faults:
        return []

    models = _faulting_models(faults)
    slots = _affected_slots(faults)
    resources = _distinct_ordered(
        [m.group("path") or m.group("file") for r in faults if (m := _FD_RESOURCE_RE.search(r.message))],
    )
    timestamps = [r.timestamp for r in faults if r.timestamp is not None]
    window_minutes = (timestamps[-1] - timestamps[0]).total_seconds() / 60 if len(timestamps) >= 2 else 0.0
    window_clause = f" over a {window_minutes:.0f}-minute poisoned window" if window_minutes >= 1 else ""

    # A recovery that replaced the affected slot after the leak (the "failed to load model" path), which is
    # the only thing that clears the poisoned process; naming it confirms the slot was eventually recycled.
    recovery = next(
        (r for r in context.recoveries if (not slots or r.process_id in slots) and "failed to load model" in r.reason),
        None,
    )

    slot_clause = f" on slot(s) {_clause_join([str(s) for s in slots])}" if slots else ""
    model_clause = f" serving {_clause_join([f'`{m}`' for m in models])}" if models else ""
    resource_clause = (
        f" The refused opens include {_clause_join([f'`{path}`' for path in resources[:4]])}." if resources else ""
    )
    recovery_clause = (
        f" The recovery supervisor eventually replaced the slot ({recovery.reason})."
        if recovery is not None
        else " No slot replacement was recorded, so the process may have stayed poisoned until the session ended."
    )

    return [
        Finding(
            id="file_descriptor_exhaustion",
            severity=Severity.CRITICAL,
            title="Inference process exhausted its file-descriptor limit (EMFILE)",
            verdict=(
                f"An inference process hit its per-process file-descriptor ceiling (errno 24, EMFILE) "
                f"{len(faults)} time(s){slot_clause}{model_clause}{window_clause}. Once over RLIMIT_NOFILE "
                f"every open() is refused, so the process faults every job while still heart-beating (the "
                f"silence watchdog cannot see it).{resource_clause}{recovery_clause} The named model is "
                "whatever was running when the ceiling was hit, not the cause: this is a descriptor leak, "
                "distinct from a CUDA OOM despite sharing the generic 'produced no results' fault text."
            ),
            remediation=(
                "Treat this as a descriptor leak, not memory pressure: reducing concurrency or the VRAM "
                "budget will not help. As an immediate stopgap, raise the worker's soft descriptor limit "
                "(ulimit -n, or LimitNOFILE= in the systemd unit) so a slow leak takes far longer to reach "
                "the ceiling. The real fix is to find what leaks descriptors in the inference child; these "
                "logs cannot pinpoint it because the worker emits no descriptor-headroom telemetry, so add "
                "RLIMIT_NOFILE headroom to the per-process status line (alongside the free-RAM/VRAM figures) "
                "so the next occurrence names the leaking growth. This fault is POSIX-specific (Windows has "
                "no RLIMIT_NOFILE and a far higher handle ceiling), so it is a concern for Linux hosts."
            ),
            evidence=[_evidence(r) for r in faults[:4]] + ([_evidence(recovery.record)] if recovery else []),
        ),
    ]


def detect_swallowed_oom(context: SessionContext) -> list[Finding]:
    """The 'no images were produced' classification gap (an OOM ComfyUI swallowed)."""
    no_images = _matching(context.session.records, _NO_IMAGES_RE)
    if not no_images:
        return []
    return [
        Finding(
            id="swallowed_oom",
            severity=Severity.WARNING,
            title="Jobs faulted with 'no images produced'",
            verdict=(
                f"{len(no_images)} job(s) faulted with a generic 'no images produced' message. ComfyUI can "
                "swallow a CUDA OOM into this generic failure, so the resource-failure breaker may never "
                "fire even though the cause was memory pressure."
            ),
            remediation=(
                "Check VRAM headroom around these faults; if memory-bound, treat 'no images produced' as a "
                "resource failure so the self-throttle/breaker engages."
            ),
            evidence=[_evidence(r) for r in no_images[:4]],
        ),
    ]


def detect_orphan_wedge(context: SessionContext) -> list[Finding]:
    """A storm of orphaned in-progress jobs (a flaky GPU stranding each inference)."""
    orphans = _matching(context.session.records, _ORPHAN_RE)
    if len(orphans) < _RECOVERY_STORM_THRESHOLD:
        return []
    return [
        Finding(
            id="orphan_wedge",
            severity=Severity.WARNING,
            title="Orphaned in-progress jobs",
            verdict=(
                f"{len(orphans)} job(s) were punted as orphaned in-progress with no owning live slot. A "
                "recurring orphan storm means something upstream keeps stranding jobs (often a GPU that "
                "hangs each inference)."
            ),
            remediation=(
                "Inspect the inference slots for hangs/OOM around these punts; a sustained storm should "
                "escalate to a soft reset (pool rebuild)."
            ),
            evidence=[_evidence(r) for r in orphans[:4]],
        ),
    ]


def _total_dropped_jobs(records: list[LogRecord]) -> int:
    """Sum the jobs save-our-ship faulted across every give-up in the session (the 'dropped' count)."""
    total = 0
    for record in _matching(records, _GIVE_UP_RE):
        match = _GIVE_UP_RE.search(record.message)
        if match is not None:
            total += int(match.group(1))
    return total


def _count_server_slow_aborts(records: list[LogRecord]) -> int:
    """Count generations the horde aborted server-side for being too slow (also 'dropped' jobs)."""
    return len(_matching(records, _SERVER_SLOW_ABORT_RE))


def _describe_drops(records: list[LogRecord]) -> str:
    """A clause naming what dropped jobs the worker produced in the lead-up to forced maintenance.

    The horde forces maintenance for *dropped* jobs, and a worker can drop them two distinct ways: by
    faulting unservable backlog jobs itself (save-our-ship give-up) or by submitting generations so late
    that the horde aborts them as too slow. The clause names whichever actually happened so the operator
    is pointed at the right upstream cause instead of a generic "investigate the faults".
    """
    giveups = _total_dropped_jobs(records)
    aborts = _count_server_slow_aborts(records)
    if giveups and aborts:
        return (
            f" The worker faulted {giveups} backlog job(s) via save-our-ship give-up and the horde aborted "
            f"{aborts} generation(s) as too slow just before."
        )
    if aborts:
        return (
            f" The horde aborted {aborts} generation(s) as too slow ('took too long to process') just "
            "before; the worker was generating slower than the horde's per-job deadline."
        )
    if giveups:
        return f" The worker faulted {giveups} backlog job(s) via save-our-ship give-up just before."
    return " investigate which jobs the worker faulted in the lead-up."


def detect_forced_maintenance(context: SessionContext) -> list[Finding]:
    """The horde forcing the worker into maintenance (the incident headline the operator actually sees).

    Maintenance is a *symptom*: the server steps in after the worker drops too many jobs. So the finding
    names the local drops as the cause rather than treating the maintenance flag as the thing to clear,
    and stays informational for maintenance the worker did not cause (operator-set, key issues).
    """
    maintenance = _matching(context.session.records, _MAINTENANCE_POP_RE)
    if not maintenance:
        return []

    giveups = _matching(context.session.records, _GIVE_UP_RE)
    slow_aborts = _matching(context.session.records, _SERVER_SLOW_ABORT_RE)
    forced_for_drops = any(_DROPPING_JOBS_RE.search(record.full_text) for record in maintenance)
    if forced_for_drops:
        drop_clause = _describe_drops(context.session.records)
        # Point the operator at whichever upstream finding actually applies: a slow-generation spiral and a
        # scheduler wedge produce the same maintenance symptom but call for opposite fixes.
        see_also = "slow_generation_drop_spiral" if slow_aborts else "scheduler_starvation_wedge"
        return [
            Finding(
                id="forced_maintenance",
                severity=Severity.CRITICAL,
                title="Horde forced the worker into maintenance",
                verdict=(
                    f"The horde rejected {len(maintenance)} pop(s) with forced maintenance because the worker "
                    f"dropped too many jobs.{drop_clause} Maintenance is the server's response to those drops, "
                    "not the underlying fault."
                ),
                remediation=(
                    "Fix what is dropping jobs (see the slow-generation / starvation-wedge / recovery findings) "
                    "rather than just clearing maintenance; it will re-trigger. If the worker is generating too "
                    "slowly, reduce max_power, max_threads, queue_size, or max_batch (and put models on an SSD); "
                    "if the cause is a self-inflicted scheduler wedge, reduce churn "
                    "(unload_models_from_vram_often / high_performance_mode)."
                ),
                evidence=[_evidence(r) for r in (maintenance[:1] + giveups[:1] + slow_aborts[:1])],
                see_also=see_also,
            ),
        ]
    return [
        Finding(
            id="forced_maintenance",
            severity=Severity.INFO,
            title="Worker was in maintenance mode",
            verdict=(
                f"The horde rejected {len(maintenance)} pop(s) with maintenance mode, but not for dropped jobs "
                "(likely operator-set or an API-key/credentials issue)."
            ),
            remediation=(
                "If unexpected, unpause the worker in the horde UI and confirm the API key is set; otherwise no "
                "action is needed."
            ),
            evidence=[_evidence(maintenance[0])],
        ),
    ]


def detect_scheduler_starvation_wedge(context: SessionContext) -> list[Finding]:
    """An over-conservative VRAM budget deferring head-of-queue jobs on an idle device (the root cause).

    The budget refused to admit a head-of-queue model on a device with ample free VRAM, so the queue
    deadlocked with idle processes and the recovery supervisor soft-reset the pools and faulted the
    backlog. A lone force-admit that broke the wedge without escalating is a near-miss (warning); a
    force-admit that still ended in a soft reset and faulted jobs is the self-inflicted wedge (critical).
    """
    starved = _matching(context.session.records, _FORCE_ADMIT_RE)
    if not starved:
        return []

    durations = [int(m.group(1)) for r in starved if (m := _FORCE_ADMIT_RE.search(r.message))]
    free_vrams = [int(m.group(1)) for r in starved if (m := _DEVICE_FREE_VRAM_RE.search(r.message))]
    max_starved = max(durations) if durations else 0
    free_hint = f" with as much as {max(free_vrams)} MB free VRAM on the device" if free_vrams else ""

    soft_resets = _matching(context.session.records, _SOFT_RESET_RE)
    dropped = _total_dropped_jobs(context.session.records)
    escalated = bool(soft_resets) or dropped > 0

    if escalated:
        return [
            Finding(
                id="scheduler_starvation_wedge",
                severity=Severity.CRITICAL,
                title="Scheduler wedged on VRAM-budget over-deferral",
                verdict=(
                    f"The VRAM budget deferred head-of-queue job(s) on an idle device for up to {max_starved}s"
                    f"{free_hint}, far more headroom than the head needed. The starved queue deadlocked, the "
                    f"recovery supervisor soft-reset the pools {len(soft_resets)} time(s) and faulted {dropped} "
                    "backlog job(s). Those faults are what the horde counts as dropped jobs."
                ),
                remediation=(
                    "The budget was over-conservative for this device (free VRAM was ample), most often because "
                    "rapid idle-process cycling left no settled baseline to size per-process overhead from. "
                    "Reduce churn (unload_models_from_vram_often / high_performance_mode) or relax the VRAM "
                    "budget so the head admits before the starvation timer trips the supervisor."
                ),
                evidence=[_evidence(r) for r in (starved[:2] + soft_resets[:1])],
                see_also="forced_maintenance",
            ),
        ]
    return [
        Finding(
            id="scheduler_starvation_wedge",
            severity=Severity.WARNING,
            title="Head-of-queue budget starvation (recovered)",
            verdict=(
                f"The VRAM budget deferred head-of-queue job(s) on an idle device for up to {max_starved}s"
                f"{free_hint}, but force-admit broke the wedge before it escalated to a soft reset. A near-miss: "
                "the budget is close to starving the scheduler on this device."
            ),
            remediation=(
                "Watch for recurrence under load; if it escalates to soft resets and faulted jobs, treat it as a "
                "wedge (reduce process churn or relax the VRAM budget)."
            ),
            evidence=[_evidence(r) for r in starved[:3]],
        ),
    ]


def detect_slow_generation_drop_spiral(context: SessionContext) -> list[Finding]:
    """The horde aborting generations as too slow, the drop mechanism behind a slow-worker maintenance.

    This is the root cause the starvation-wedge detector does not cover: the worker is not wedged, it is
    simply generating slower than the horde's per-job deadline, so the server aborts each late submission
    ("took too long to process") and faults it. A sustained run of these aborts is what the horde counts
    as dropped jobs and answers with forced maintenance. The worker-side grader corroborates with the
    slowdown ratio and the free-VRAM snapshot that fingerprints an over-committed device; a handful of
    isolated aborts is a warning, a sustained spiral (or one that already drew maintenance) is critical.
    """
    aborts = _matching(context.session.records, _SERVER_SLOW_ABORT_RE)
    if not aborts:
        return []

    slowdowns = _matching(context.session.records, _SLOWDOWN_GRADE_RE)
    ratios = [float(m.group(1)) for r in slowdowns if (m := _SLOWDOWN_GRADE_RE.search(r.message))]
    free_vrams = [int(m.group(1)) for r in slowdowns if (m := _DEVICE_FREE_VRAM_RE.search(r.message))]
    timestamps = [r.timestamp for r in aborts if r.timestamp is not None]
    span_minutes = (timestamps[-1] - timestamps[0]).total_seconds() / 60 if len(timestamps) >= 2 else 0.0
    span_clause = f" over {span_minutes:.0f} min" if span_minutes >= 1 else ""

    maintenance = _matching(context.session.records, _MAINTENANCE_POP_RE)
    forced_for_drops = any(_DROPPING_JOBS_RE.search(record.full_text) for record in maintenance)
    spiral = forced_for_drops or len(aborts) >= _SLOW_ABORT_SPIRAL_THRESHOLD
    severity = Severity.CRITICAL if spiral else Severity.WARNING

    # Decide whether jobs aged in the pipeline queue (fast generation, long pop->submit latency) or in
    # generation itself (slow GPU). The two share the "too slow" abort but call for opposite fixes, so the
    # detector measures the submitted jobs' own latency-vs-generation breakdown rather than guessing.
    latencies = [
        (float(m.group(1)), float(m.group(2)))
        for r in _matching(context.session.records, _SUBMIT_LATENCY_RE)
        if (m := _SUBMIT_LATENCY_RE.search(r.message))
    ]
    safety_times = [
        float(m.group(1))
        for r in _matching(context.session.records, _SAFETY_DURATION_RE)
        if (m := _SAFETY_DURATION_RE.search(r.message))
    ]
    median_latency = _median([lat for lat, _ in latencies])
    median_gen = _median([gen for _, gen in latencies])
    queue_aging = (
        median_latency is not None
        and median_gen is not None
        and median_gen > 0
        and median_latency >= median_gen * _QUEUE_AGING_LATENCY_RATIO
    )

    base_verdict = (
        f"The horde aborted {len(aborts)} generation(s){span_clause} as too slow ('took too long to "
        f"process and has been aborted'); each counts against the worker as a dropped job."
    )
    tail = "" if spiral else " Isolated so far, but a sustained run will draw horde-forced maintenance."

    if queue_aging:
        assert median_latency is not None and median_gen is not None
        safety_clause = f" The safety stage averaged {_mean(safety_times):.1f}s per check." if safety_times else ""
        safety_evidence = _matching(context.session.records, _SAFETY_DURATION_RE)[:1]
        return [
            Finding(
                id="slow_generation_drop_spiral",
                severity=severity,
                title="Jobs aging in the pipeline queue (not slow generation)",
                verdict=(
                    f"{base_verdict} Generation itself was fast (median {median_gen:.0f}s) but jobs waited a "
                    f"median {median_latency:.0f}s from pop to submit: they aged in the post-inference queue, "
                    f"not in generation.{safety_clause} A downstream stage (typically the single, often "
                    f"CPU-bound, safety process) is slower than inference, so its backlog grows until jobs "
                    f"exceed their ttl.{tail}"
                ),
                remediation=(
                    "This is a pipeline-balance problem, not a too-aggressive GPU config, so lowering "
                    "max_power will not help. The worker now applies post-inference backpressure (it stops "
                    "popping while the safety backlog cannot clear within the job ttl), which bounds this; if "
                    "it persists, speed up the bottleneck stage (e.g. enable safety_on_gpu so safety is not "
                    "CPU-bound, or add safety capacity) so throughput is not capped below inference."
                ),
                evidence=[_evidence(r) for r in (aborts[:2] + safety_evidence)],
                see_also="forced_maintenance",
            ),
        ]

    slow_clause = ""
    if ratios:
        slow_clause = f" The worker graded inference up to {max(ratios):.1f}x its expected sampling time"
        if free_vrams:
            slow_clause += f" with as little as {min(free_vrams)} MB free VRAM (an over-committed device)"
        slow_clause += "."
    return [
        Finding(
            id="slow_generation_drop_spiral",
            severity=severity,
            title="Slow generation is dropping jobs" if spiral else "Generations aborted as too slow",
            verdict=base_verdict + slow_clause + tail,
            remediation=(
                "The worker cannot finish jobs within the horde's deadline. Reduce max_power (smaller "
                "resolution / fewer steps), max_threads, queue_size, and/or max_batch so each job completes "
                "in time; put models on an SSD and free VRAM/RAM so the device is not over-committed. This is "
                "the upstream cause of any forced maintenance; clearing maintenance without slowing the "
                "intake will just re-trigger it."
            ),
            evidence=[_evidence(r) for r in (aborts[:2] + slowdowns[:2])],
            see_also="forced_maintenance",
        ),
    ]


def detect_consecutive_failure_pause(context: SessionContext) -> list[Finding]:
    """The worker self-pausing job pops after three consecutive faults (a downstream symptom)."""
    pauses = _matching(context.session.records, _CONSECUTIVE_PAUSE_RE)
    if not pauses:
        return []
    return [
        Finding(
            id="consecutive_failure_pause",
            severity=Severity.WARNING,
            title="Worker self-paused on consecutive faults",
            verdict=(
                f"The worker paused job pops {len(pauses)} time(s) after three consecutive faulted jobs. This is "
                "the worker protecting itself, downstream of whatever kept faulting jobs."
            ),
            remediation=(
                "Find the fault source (the starvation-wedge / recovery / OOM findings); the pause clears on its "
                "own but will re-trigger until the faults stop."
            ),
            evidence=[_evidence(r) for r in pauses[:3]],
        ),
    ]


def detect_safety_stage_stall(context: SessionContext) -> list[Finding]:
    """The safety stage stranding jobs whose verdict never returned (a forced-maintenance cause).

    A job sent to safety whose result is lost is invisible to the orchestrator and sits in SAFETY_CHECKING
    forever; the backlog pins pipeline slots and, with the queue unable to drain, latches the wedge that
    ends in dropped jobs. The worker now recovers it (re-check), or (when safety cannot be relied on),
    faults it with no image and soft-pauses pops. This surfaces that recovery so a maintenance episode is
    attributed to the *downstream safety stall* rather than to inference. Backpressure alone (the worker
    correctly throttling intake to a slow safety stage) is the benign, lower-severity case.
    """
    records = context.session.records
    requeues = _matching(records, _SAFETY_REQUEUE_RE)
    unrecoverable = _matching(records, _SAFETY_UNRECOVERABLE_RE)
    soft_pauses = _matching(records, _SAFETY_SOFT_PAUSE_RE)
    lost_results = _matching(records, _LOST_SAFETY_RESULT_RE)
    backpressure = _matching(records, _SAFETY_BACKPRESSURE_RE)

    if not (requeues or unrecoverable or soft_pauses or lost_results or backpressure):
        return []

    # Escalation (a job faulted with no image, or pops soft-paused) means the safety pipeline could not be
    # relied on and jobs were dropped: critical. Re-checks / lost results / pure backpressure recovered or
    # throttled without dropping: a warning that the safety stage is the bottleneck.
    escalated = bool(unrecoverable or soft_pauses)
    severity = Severity.CRITICAL if escalated else Severity.WARNING

    detail_bits: list[str] = []
    if lost_results:
        detail_bits.append(f"{len(lost_results)} safety result(s) never returned")
    if requeues:
        detail_bits.append(f"{len(requeues)} job(s) re-checked")
    if unrecoverable:
        detail_bits.append(f"{len(unrecoverable)} faulted with no image")
    if soft_pauses:
        detail_bits.append(f"{len(soft_pauses)} soft-pause(s)")
    if backpressure:
        detail_bits.append(f"{len(backpressure)} pop-throttle(s) on backlog")
    detail = "; ".join(detail_bits)

    if escalated:
        verdict = (
            f"The safety stage stranded jobs whose verdict never returned ({detail}). The worker faulted the "
            "unservable ones with no image and soft-paused pops, but those faults count as dropped jobs and can "
            "draw horde-forced maintenance. A lost safety verdict (a cycled/replaced safety process, or a dropped "
            "result message) is the root cause."
        )
        remediation = (
            "Stabilise the safety process: with safety_on_gpu set, frequent whole-card residency cycling or "
            "unload_models_from_vram_often can churn it; check the bridge_safety_*.log for crashes. The worker "
            "re-checks and (only as a last resort) faults with no image, so no unchecked image is ever submitted."
        )
    else:
        verdict = (
            f"The safety stage backed up or briefly lost a verdict ({detail}); the worker recovered (re-check) or "
            "throttled intake without dropping jobs. The safety stage is the pipeline bottleneck."
        )
        remediation = (
            "If it recurs under load, speed up safety (enable safety_on_gpu, or reduce post-processing) so the "
            "backlog does not grow; no action is needed for an isolated occurrence."
        )

    evidence = unrecoverable[:1] + soft_pauses[:1] + lost_results[:1] + requeues[:1] + backpressure[:1]
    return [
        Finding(
            id="safety_stage_stall",
            severity=severity,
            title="Safety stage stalled (lost verdicts / backlog)",
            verdict=verdict,
            remediation=remediation,
            evidence=[_evidence(r) for r in evidence[:4]],
            see_also="forced_maintenance" if escalated else None,
        ),
    ]


def detect_whole_card_convergence_wedge(context: SessionContext) -> list[Finding]:
    """A whole-card head parked because the residency cannot collapse to sole residency.

    The wedge fingerprint: a heavy whole-card head (e.g. Flux fp8) is pre-staged into a spare process, but an
    idle sibling still holds a model that is queued *behind* the head, and the live process count never reaches
    the forecast target, so the pre-staged head is deferred every tick until the recovery supervisor
    soft-resets the pools (faulting the head and forcing process recoveries). The whole-card convergence
    teardown is supposed to stop exactly that idle sibling (sparing only the head's holder), so reaching this
    state means the convergence shrink did not engage for this process/queue shape. It is distinct from a
    generic dispatch-path stall, so it is named explicitly in the worker's dispatch-stall log and detected on
    its own here.
    """
    wedges = _matching(context.session.records, _WHOLE_CARD_WEDGE_RE)
    if not wedges:
        return []
    return [
        Finding(
            id="whole_card_convergence_wedge",
            severity=Severity.CRITICAL,
            title="Whole-card residency cannot reach sole residency (queued-model sibling pins the teardown)",
            verdict=(
                f"A pre-staged whole-card head was parked {len(wedges)} time(s) because an idle sibling process "
                "still holds a model queued behind it and was not torn down, so the residency never collapsed "
                "to sole residency and the head was deferred until the recovery supervisor soft-reset the pools "
                "(faulting the head and forcing process recoveries). The whole-card convergence is meant to "
                "stop that sibling (sparing only the head's holder), so this indicates the convergence shrink "
                "did not engage for this process/queue shape."
            ),
            remediation=(
                "Capture the surrounding scheduling logs and the process map: confirm the pre-staged head's "
                "holder is identified (its loaded model name) and that the idle sibling is genuinely idle (not "
                "busy). A recurrence points at the whole-card teardown failing to stop an eligible sibling. As "
                "an operational stopgap, reducing queue_size or avoiding a heavy whole-card model alongside a "
                "deep same-cycle queue lowers the odds of hitting this shape."
            ),
            evidence=[_evidence(r) for r in wedges[:3]],
            see_also="head_dispatch_stall",
        ),
    ]


def detect_whole_card_nonhead_residency_starvation(context: SessionContext) -> list[Finding]:
    """A whole-card residency held for a non-head model, starving the actual head of the queue.

    The whole-card residency reserves the entire device and tears its sibling processes down. Granting it to
    a model that is not the head of the queue collapses the very processes serving the lighter heads ahead of
    it, so the real head has no resident process and cannot load while the card is reserved for a job whose
    turn has not come (held until that job drains). The head parks, the queue deadlocks, and the recovery
    supervisor soft-resets and faults the backlog. The whole-card residency is meant to be granted only to the
    head, so a firing here means a non-head model claimed the card, distinct from a genuine VRAM-budget
    over-deferral, which this would otherwise be mistaken for (the card looks idle with ample free VRAM).
    """
    starvations = _matching(context.session.records, _WHOLE_CARD_NONHEAD_RE)
    if not starvations:
        return []
    soft_resets = _matching(context.session.records, _SOFT_RESET_RE)
    dropped = _total_dropped_jobs(context.session.records)
    escalated = bool(soft_resets) or dropped > 0
    return [
        Finding(
            id="whole_card_nonhead_residency_starvation",
            severity=Severity.CRITICAL if escalated else Severity.WARNING,
            title="Whole-card residency held for a non-head model starved the queue head",
            verdict=(
                f"The head of the queue was parked {len(starvations)} time(s) because a whole-card residency was "
                "held for a different (non-head) model, which reserved the card and tore down the processes "
                "serving the head. "
                + (
                    f"The starved queue deadlocked, the recovery supervisor soft-reset the pools "
                    f"{len(soft_resets)} time(s) and faulted {dropped} backlog job(s)."
                    if escalated
                    else "Force-admit or a drain broke it before it escalated to a soft reset."
                )
            ),
            remediation=(
                "The whole-card residency must only be granted to the head (next-to-dispatch) job; a deeper-queue "
                "heavy model should defer until it becomes the head rather than reserving the card. If this "
                "recurs, capture the residency establish/pre-stage lines and the queue order to confirm which "
                "model claimed the card while a different head was pending."
            ),
            evidence=[_evidence(r) for r in (starvations[:2] + soft_resets[:1])],
            see_also="scheduler_starvation_wedge",
        ),
    ]


def detect_whole_card_residency_churn(context: SessionContext) -> list[Finding]:
    """The whole card was reserved repeatedly in a session: reservation churn, not a deliberate hold.

    Establishing a whole-card residency reserves the device, reduces the live process count, and cycles the
    safety process off the GPU; restoring it reverses all three. Doing that a handful of times in a session is
    thrash, and on a high-VRAM card it usually means a model that does not need the card is being driven onto
    the whole-card path: a model whose weights are a small fraction of total VRAM co-resides comfortably, so a
    teardown demand for it points at the per-context overhead being over-counted (the per-additional-context
    cost was not measured, so the one-time runtime cost is charged against every context, collapsing the
    structural free-VRAM floor). The churn alone caps throughput (reload + safety cycling per swap); paired
    with soft resets or dropped jobs it is the reservation feeding a starvation wedge.
    """
    establishes = _matching(context.session.records, _WHOLE_CARD_ESTABLISH_RE)
    if len(establishes) < _WHOLE_CARD_CHURN_THRESHOLD:
        return []
    soft_resets = _matching(context.session.records, _SOFT_RESET_RE)
    dropped = _total_dropped_jobs(context.session.records)
    declined = _matching(context.session.records, _WHOLE_CARD_DECLINED_RE)
    escalated = bool(soft_resets) or dropped > 0
    return [
        Finding(
            id="whole_card_residency_churn",
            severity=Severity.CRITICAL if escalated else Severity.WARNING,
            title="Whole-card residency reserved and restored repeatedly (reservation churn)",
            verdict=(
                f"The whole card was reserved {len(establishes)} time(s) this session, each time reducing the "
                "live process count and cycling safety off the GPU, then restoring them. Sustained reservation "
                "churn is the signature of a model being given the card that does not need it; on a high-VRAM "
                "card a model whose weights are a small fraction of total VRAM co-resides, so a teardown demand "
                "for it usually means the per-context overhead was over-counted (an unmeasured marginal). "
                + (
                    f"It escalated: the recovery supervisor soft-reset the pools {len(soft_resets)} time(s) and "
                    f"faulted {dropped} backlog job(s)."
                    if escalated
                    else "It did not escalate to a soft reset here, but the reload + safety cycling caps throughput."
                )
                + (
                    f" The trust gate declined {len(declined)} further reservation(s), so it is actively damping "
                    "the churn."
                    if declined
                    else ""
                )
            ),
            remediation=(
                "Confirm the reserved models are genuinely card-filling. If they are not (a small-weight model "
                "on a large card), make sure the per-additional-context VRAM cost is measured (the probe's "
                "second-context delta or a clean all-idle baseline), so the structural free-VRAM floor is not "
                "the one-time runtime cost multiplied by the process count. A correctly measured marginal lets "
                "such a model co-reside instead of reserving the card."
            ),
            evidence=[_evidence(r) for r in (establishes[:2] + soft_resets[:1])],
            see_also="scheduler_starvation_wedge",
        ),
    ]


def detect_head_dispatch_stall(context: SessionContext) -> list[Finding]:
    """A head-of-queue job that did not dispatch despite pending work and an idle, model-resident process.

    The scheduler returns ``None`` silently from several gates, so a stuck queue with idle processes used to
    leave no record of *why* the head was parked. The new dispatch-stall log names the blocking gate; the
    "no matching gate" variant is the genuinely anomalous case (the head's model is resident and idle, no
    gate is holding it, yet nothing dispatched) and is reported as critical, the rest as a warning. The
    whole-card convergence wedge (:func:`detect_whole_card_convergence_wedge`) and the non-head residency
    starvation (:func:`detect_whole_card_nonhead_residency_starvation`) have their own detectors, so their
    lines are excluded here to avoid double-reporting the same stall as a generic warning.
    """
    excluded = _matching(context.session.records, _WHOLE_CARD_WEDGE_RE) + _matching(
        context.session.records,
        _WHOLE_CARD_NONHEAD_RE,
    )
    stalls = [r for r in _matching(context.session.records, _DISPATCH_STALL_RE) if r not in excluded]
    if not stalls:
        return []

    bug_stalls = _matching(context.session.records, _DISPATCH_STALL_BUG_RE)
    if bug_stalls:
        return [
            Finding(
                id="head_dispatch_stall",
                severity=Severity.CRITICAL,
                title="Head-of-queue job not dispatching (no blocking gate)",
                verdict=(
                    f"The scheduler reported a parked head {len(bug_stalls)} time(s) whose model was resident on "
                    "an idle process with no gate holding it, yet nothing dispatched. That is a scheduler stall "
                    "(not a budget or concurrency decision) and can wedge the queue into dropped jobs."
                ),
                remediation=(
                    "Capture the surrounding scheduling logs and process map: a model-resident, idle-process head "
                    "that will not dispatch points to a dispatch-path bug (e.g. an eviction that clears the head's "
                    "resident model just before dispatch under unload_models_from_vram_often). Reduce churn as a "
                    "stopgap."
                ),
                evidence=[_evidence(r) for r in bug_stalls[:3]],
                see_also="scheduler_starvation_wedge",
            ),
        ]
    return [
        Finding(
            id="head_dispatch_stall",
            severity=Severity.WARNING,
            title="Head-of-queue job repeatedly parked",
            verdict=(
                f"The head of the queue was parked (not dispatching) {len(stalls)} time(s), each explained by a "
                "known gate (concurrency cap, overlap headway, keep-single-inference, or a deferred preload). "
                "Sustained, this starves throughput even though it is not a hard wedge."
            ),
            remediation=(
                "If throughput is low, the named gate is the lever: review max_threads / batch settings, the "
                "overlap-headway behaviour, or the VRAM budget that is deferring the preload."
            ),
            evidence=[_evidence(r) for r in stalls[:3]],
        ),
    ]


_GOVERNOR_DOMINANCE_FRACTION = 0.25
"""A governor engaged for at least this share of the session is worth surfacing as a throughput shaper."""
_GOVERNOR_DOMINANCE_MIN_SECONDS = 60.0
"""...but only when its absolute engaged time is non-trivial, so a short session does not flag on noise."""


def _governor_engaged_seconds(session: object) -> dict[str, float]:
    """Reconstruct each pop-governor's total engaged seconds from its ENTER/EXIT spell boundaries.

    Pairs each ENTER with the next EXIT for the same governor; a spell still open at the last record counts
    to the session's end. Keyed by the governor's machine name. Returns an empty mapping when no boundary
    lines are present (an older worker, or one with no governor ever engaging).
    """
    records = session.records  # type: ignore[attr-defined]
    end_ts = session.end_ts  # type: ignore[attr-defined]
    engaged: dict[str, float] = {}
    open_since: dict[str, object] = {}
    for record in records:
        if record.timestamp is None:
            continue
        enter = GOVERNOR_ENTER_RE.search(record.message)
        if enter:
            open_since[enter.group("name")] = record.timestamp
            continue
        exit_match = GOVERNOR_EXIT_RE.search(record.message)
        if exit_match:
            name = exit_match.group("name")
            started = open_since.pop(name, None)
            if started is not None:
                engaged[name] = engaged.get(name, 0.0) + max(0.0, (record.timestamp - started).total_seconds())
    for name, started in open_since.items():
        if end_ts is not None:
            engaged[name] = engaged.get(name, 0.0) + max(0.0, (end_ts - started).total_seconds())
    return engaged


def detect_pop_governor_dominance(context: SessionContext) -> list[Finding]:
    """A pop/scheduling governor that held the worker back for a large share of the session.

    Governors (whole-card residency, the large-model switch/re-entry limiters, backpressure, the unservable
    holdback, the various pauses) are each legitimate, but one consuming a big fraction of the session is a
    throughput-shaping signal worth surfacing: it points at the lever (a model mix, a config duration, a slow
    safety stage) the operator can act on, without itself being a fault.
    """
    session = context.session
    duration = session.duration_seconds
    if duration is None or duration <= 0:
        return []
    engaged = _governor_engaged_seconds(session)
    dominant = sorted(
        (
            (name, seconds)
            for name, seconds in engaged.items()
            if seconds >= _GOVERNOR_DOMINANCE_MIN_SECONDS and (seconds / duration) >= _GOVERNOR_DOMINANCE_FRACTION
        ),
        key=lambda item: -item[1],
    )
    if not dominant:
        return []
    phrases = [
        f"{GOVERNOR_LABELS.get(name, name)} ({seconds / duration * 100:.0f}% of the session, {seconds / 60:.1f} min)"
        for name, seconds in dominant
    ]
    return [
        Finding(
            id="pop_governor_dominance",
            severity=Severity.INFO,
            title="A pop governor shaped much of the session",
            verdict=(
                "The worker spent a large share of the session with a pop/scheduling governor engaged: "
                + "; ".join(phrases)
                + ". This is not a fault, but it is the dominant lever on throughput for this session."
            ),
            remediation=(
                "If throughput was lower than expected, this names where the time went. Whole-card residency or "
                "the large-model limiters point at the model mix and their configured durations "
                "(whole_card_residency_cooldown_seconds, large_model_switch_min_seconds, "
                "large_model_reentry_cooldown_seconds); backpressure points at a slow safety stage; the "
                "unservable holdback points at a model the device cannot run."
            ),
            evidence=[_evidence(r) for r in _matching(session.records, GOVERNOR_ENTER_RE)[:3]],
        ),
    ]


def detect_session_summary(context: SessionContext) -> list[Finding]:
    """An always-present rollup: how the session ended and its recovery/fault headline numbers."""
    session = context.session
    duration = session.duration_seconds
    duration_text = f"{duration / 60:.1f} min" if duration is not None else "unknown duration"
    severity = Severity.WARNING if session.end_reason is SessionEndReason.KILLED_OR_CRASHED else Severity.INFO
    return [
        Finding(
            id="session_summary",
            severity=severity,
            title="Session summary",
            verdict=(
                f"Ended via {session.end_reason} after {duration_text}; peak process recoveries "
                f"{session.peak_process_recoveries}; {len(context.recoveries)} recovery diagnostic(s); "
                f"version v{session.version or '?'}, {session.num_models or '?'} models, "
                f"{session.max_threads or '?'} thread(s)."
            ),
            remediation="",
        ),
    ]


DETECTORS: list[Detector] = [
    detect_crash_on_start_loop,
    detect_doomed_pool_no_giveup,
    detect_gave_up_clean,
    detect_forced_maintenance,
    detect_scheduler_starvation_wedge,
    detect_slow_generation_drop_spiral,
    detect_safety_stage_stall,
    detect_whole_card_convergence_wedge,
    detect_whole_card_nonhead_residency_starvation,
    detect_whole_card_residency_churn,
    detect_head_dispatch_stall,
    detect_consecutive_failure_pause,
    detect_pop_governor_dominance,
    detect_stuck_inference_step,
    detect_post_processing_vram_stall,
    detect_post_processing_deferral_starvation,
    detect_oom,
    detect_file_descriptor_exhaustion,
    detect_swallowed_oom,
    detect_orphan_wedge,
    detect_session_summary,
]


def run_detectors(context: SessionContext) -> list[Finding]:
    """Run all detectors over a session and return their findings, most-severe first."""
    findings: list[Finding] = []
    for detector in DETECTORS:
        try:
            findings.extend(detector(context))
        except Exception:  # noqa: BLE001 - a broken detector must not sink the whole report.
            continue
    findings.sort(key=lambda finding: _SEVERITY_ORDER[finding.severity])
    return findings
