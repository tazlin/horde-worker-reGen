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

import bisect
import enum
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from horde_worker_regen.utils.oom_signature import OOM_TEXT_RE

from .correlate import RecoveryDiagnostic, SessionContext, find_child_crash
from .governor_signatures import GOVERNOR_ENTER_RE, GOVERNOR_EXIT_RE, GOVERNOR_LABELS
from .log_ingest import LogRecord
from .sessions import SessionEndReason

# Signatures over orchestrator message text.
_QUARANTINE_RE = re.compile(r"quarantined \(crash on start")
_SOFT_RESET_RE = re.compile(r"Save-our-ship soft reset")

_WEDGE_ESCALATION_RE = re.compile(r"(Queue deadlock detected|Deadlock detected|Save-our-ship)")
"""The worker declaring the queue stopped, or its recovery ladder acting on that declaration.

A hold that self-clears is ordinary packing; a hold whose own window contains one of these lines did not
self-clear, whatever its per-hold text says, because the worker had already escalated past it."""
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
# The same slot-side fault, keyed on the job it names. Producer:
# ``message_dispatcher._handle_faulted_inference_result``.
_FAULTED_ON_PROCESS_JOB_RE = re.compile(r"Job (?P<job>[0-9a-fA-F-]{8,}) faulted on process (?P<pid>\d+)")
# The terminal per-job fault the worker reports to the horde. Producer:
# ``job_submitter.submit_single_generation``. This is the surface the horde's own faulted counter follows,
# so it is what a census must agree with; the slot-side line above is the same job seen earlier.
_FAULT_REPORTED_RE = re.compile(r"(?P<job>[0-9a-fA-F-]{8,}) faulted\. Reported fault to the horde\.")
# The pop line that binds a job id to the model it was popped for. Producer: ``job_popper.api_job_pop``.
_POPPED_JOB_RE = re.compile(r"Popped job (?P<job>\S+) .*?\(model: (?P<model>.+?), batch:")
# The safety watchdog faulting a named job it could not check. Producer:
# ``worker_recovery_coordinator`` (the id-anchored form of :data:`_SAFETY_UNRECOVERABLE_RE`).
_SAFETY_UNRECOVERABLE_JOB_RE = re.compile(r"Job (?P<job>\S+) could not be safety-checked")
# The retry suffix on a slot-side fault. Producer: the same dispatcher handler, which words a bounded
# retry as "requeued for another attempt" (or a degraded, isolated one). An attempt that came back for a
# retry is not a job the horde lost, so a census that counts it over-reports what the session dropped.
_FAULT_REQUEUED_RE = re.compile(r"requeued for (?:a degraded, isolated|another) attempt")
# A disaggregated pipeline stage faulting a named job. Producer:
# ``inference_process._run_sample_stage``, which runs only on the disaggregated path, so the phrase is
# specific to it rather than to monolithic inference.
_STAGE_FAULT_RE = re.compile(r"(?P<stage>\w[\w -]*) stage faulted for job (?P<job>\S+?):")
# The model-reference read that failed under an in-flight sample, and the cache-staleness lines that
# attribute it. Producers: ``horde_model_reference`` (raised into the stage fault above) and the replica
# backend's ``needs_refresh`` disclosure.
_MODEL_REFERENCE_UNREADABLE_RE = re.compile(
    r"Model reference for category (?P<category>\S+) not found or could not be parsed",
)
_MODEL_REFERENCE_STALE_RE = re.compile(r"needs refresh|cache is stale")
# The worker's most severe liveness signal: the local queue full and motionless. Producer:
# ``process_manager._full_queue_frozen_line``, logged at ERROR on a repeat clock.
_POP_LIVENESS_FROZEN_RE = re.compile(r"Pop liveness: the local job queue has been full and not draining")
_POP_LIVENESS_FROZEN_FIELDS_RE = re.compile(
    r"Pop liveness: the local job queue has been full and not draining for (?P<seconds>\d+)s "
    r"\((?P<waiting>\d+) accepted job\(s\) waiting, head model '(?P<model>[^']*)'\)",
)
# The whole-card residency governor's ENTER parenthetical names the model holding the card. Producer:
# ``PopGovernorRegistry`` via the scheduler's residency spell.
_RESIDENCY_GOVERNOR_MODEL_RE = re.compile(r"Pop governor ENTER: whole_card_residency \((?P<model>.+?) holds the card")
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
# The dispatch residency-reconciliation hold: the scheduler is holding a resident-idle head while it evicts
# an idle sibling's VRAM so the head's on-device materialisation fits the card. This is a benign, self-
# clearing swap-churn wait, not a scheduler bug, so it is excluded from the gate-less-stall detector and
# surfaced on its own as a GPU-uptime duty cost. The model and parked seconds ride the stall wrapper.
_DISPATCH_STALL_RECONCILE_RE = re.compile(r"held to reconcile residency")
_DISPATCH_STALL_FIELDS_RE = re.compile(
    r"Inference dispatch stalled: head \S+ \((?P<model>.+?)\) has been parked (?P<parked>\d+)s:",
)
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
# The same line's own figures. Producer: ``inference_scheduler._establish_whole_card_residency``, whose
# f-string words the counts as "(inference processes {current} -> {after} of {max}, target {target})"; the
# residency snapshot appended after it carries ``device_free_vram=<N>MB`` (:data:`_DEVICE_FREE_VRAM_RE`).
# Reading them is what separates a reservation that reduced the pool from one whose target was already met.
_WHOLE_CARD_ESTABLISH_FIELDS_RE = re.compile(
    r"Whole-card residency: reserving the device for (?P<model>.+?) \(inference processes "
    r"(?P<current>\d+) -> (?P<after>\d+) of (?P<total>\d+), target (?P<target>\d+)\)",
)
# The establishment-time forecast the dispatch path discloses. Producer:
# ``inference_scheduler._log_stream_forecast``. The bracketed measurement block and the trailing decision
# flags have grown fields over time, so each figure is matched independently and simply stays absent in an
# older capture rather than failing the whole parse.
_STREAM_FORECAST_RE = re.compile(r"Stream forecast for (?P<model>.+?): ")
_FORECAST_MARGINAL_RE = re.compile(r"marginal/ctx=(?P<marginal>[\d?]+)MB\(src=(?P<source>\w+)")
_FORECAST_UNRECLAIMABLE_RE = re.compile(r"unreclaimable=(?P<unreclaimable>[\d.]+)MB")
_FORECAST_REDUCTION_RE = re.compile(r"needs_process_count_reduction=(?P<reduction>True|False)")
# A pop that arrived with no model name, and the two worker generations' treatment of it. The older
# generation had no boundary check, so the blank identity travelled: it was preloaded as a literal empty
# name (``inference_scheduler._send_preload``), ended the slot it was sent to (the recovery reason from
# ``process_lifecycle``), was then counted and quarantined as if it were a model
# (``record_model_incident``), after which every later job for it was refused
# (``_attempt_preload_for_job``). Each of those lines carries the empty name verbatim, which is what makes
# the blank identity recognizable at all.
_EMPTY_MODEL_POP_RE = re.compile(r"Popped job (?P<job>\S+) .*\(model: , ")
_BLANK_PRELOAD_RE = re.compile(r"Preloading model {2}on process (?P<pid>\d+)")
_BLANK_MODEL_QUARANTINE_RE = re.compile(r"Model {2}caused (?P<count>\d+) (?P<kind>\w+) incident\(s\)")
_BLANK_QUARANTINE_SKIP_RE = re.compile(r"Skipping preload of quarantined model ;")
# The newer generation contains the same input at the boundary and never lets it reach a slot. Producers:
# ``job_popper._reject_malformed_pop``, ``inference_process._preload_model``, and
# ``process_lifecycle.record_model_incident``. A capture carrying these instead of the cascade above shows
# the containment working, so the count is a rate rather than a fault.
_MALFORMED_POP_REJECTED_RE = re.compile(
    r"Popped job (?P<job>\S+) carries no model name \(got .*?\); returning it to the horde",
)
_BLANK_PRELOAD_REFUSED_RE = re.compile(r"Refusing to preload a blank model name")
_BLANK_INCIDENT_REFUSED_RE = re.compile(r"incident reported against a blank model name")
# A slot replaced because loading a model ended it, naming the model. Producer: ``process_lifecycle``'s
# recovery reason. The name is captured (and may be empty) so repeated deaths can be grouped by model: a
# checkpoint that kills every slot it touches is a loop no per-slot breaker catches.
# Greedy to the reason's closing paren: model names carry their own parentheses ("AlbedoBase XL (SDXL)"),
# which a non-greedy match truncates at the first one.
_LOAD_FAILURE_MODEL_RE = re.compile(r"inference process replaced \(failed to load model (?P<model>.*)\)$")
# The horde rejecting a pop, quoting its own message. Producer:
# ``job_popper._handle_pop_error_response``. The message is what distinguishes an operator-fixable
# rejection (an account limit) from a transient server fault, and it is otherwise never surfaced.
_POP_API_ERROR_RE = re.compile(r"Failed to pop job \(API Error\): message='(?P<message>[^']*)'")
_POP_API_ERROR_CODE_RE = re.compile(r"rc='(?P<code>[^']*)'")
# The parent's own safety-placement actuator, and the consequence of using it. Producers:
# ``process_lifecycle.pause_safety_on_gpu`` / ``restore_safety_on_gpu`` (which name the initiating
# subsystem rather than a fixed phrase) and ``message_dispatcher._classify_retired_launch_message``. A
# verdict dropped because its launch was retired is the mechanism by which a placement cycle strands a job,
# so the pair together is an attribution rather than a guess.
_SAFETY_PLACEMENT_CYCLE_RE = re.compile(
    r"(?P<owner>[\w -]+): (?:moving the safety process off-GPU|restoring the safety process to the GPU)",
)
_RETIRED_SAFETY_RESULT_RE = re.compile(r"Ignoring result message from retired safety process")
# A child that died installing or repairing the shared ComfyUI environment rather than importing the
# inference stack. Concurrent cold starts clone into one environment directory, and a half-written clone
# leaves the tree in a state a later checkout refuses; the fix is the directory, never the torch install.
_GIT_ENVIRONMENT_FAILURE_RE = re.compile(
    r"GitCommandError|git clone .* failed|Untracked working tree file|unable to checkout working tree",
)
# The worker declining to reserve the card for a model whose teardown demand it does not trust (a card-light
# model on a host with no measured per-context cost). Surfaced as the positive counterpart: it confirms the
# trust gate is actively preventing reservation churn rather than the churn simply being absent.
_WHOLE_CARD_DECLINED_RE = re.compile(r"Declined a whole-card residency for")
# A whole-card residency claiming the worker's pop offer: while the claim stands the worker advertises only
# the resident model, so nothing else can arrive. The claim is invisible in the job stream (it changes what
# the horde is asked for, not what the worker does with what arrives), so the edges are the only record of it.
# Both phrases are the scheduler's verbatim engage/release disclosures.
_POP_CLAIM_ENGAGED_RE = re.compile(r"Whole-card pop claim engaged for (?P<model>.+?): advertising that model alone")
_POP_CLAIM_RELEASED_RE = re.compile(
    r"Whole-card pop claim released for (?P<model>.+?): (?P<release>[^;]+); advertising the full pool again",
)
_POP_CLAIM_CAP_RELEASE = "the maximum hold elapsed"
"""The release phrase for the maximum hold: the claim outlasted its window rather than ending on its own.

The other two ends are self-correcting (the demand dried up, or the residency let go), so only this one says
the claim was still narrowing the offer when the cap had to stop it."""

# The stuck-step watchdog reaping a slot whose ComfyUI generation looped on one sampling step. The slot
# kept heart-beating (so the silence watchdog stayed blind), which is exactly why this needs its own
# detector rather than folding into a generic hang. The phrase is the worker's verbatim reap line.
_STUCK_STEP_RE = re.compile(r"stuck on a non-advancing sampling step|stuck-step watchdog")

# The post-processing-stage watchdog reaping a slot that went silent. Older workers reported this as
# INFERENCE_POST_PROCESSING; dedicated-lane workers report POST_PROCESS / POST_PROCESSING. The peak is still
# an upscaler/face-fixer allocation landing after sampling, concurrent with warm inference siblings.
_POST_PROCESSING_STALL_RE = re.compile(r"seems to be stuck post processing")
_DEDICATED_POST_PROCESS_RE = re.compile(
    r"(?:last_process_state=HordeProcessState\.(?:INFERENCE_POST_PROCESSING|POST_PROCESSING)\b|"
    r"Post-processing (?:job|for job|finished for job) [0-9a-fA-F]{8})",
    re.IGNORECASE,
)
"""Proof that the dedicated post-processing lane actually ran a job.

The lane's stage transition and its result both reach the parent through the message dispatcher, which
prefixes them with its own ``Received <Message> from process N:`` wrapper, so a marker anchored to the
start of the message matched only the completion summary and missed every job that was still running.
Requiring a job id instead of a position keeps the marker off the prose that merely mentions
post-processing (per-lane RAM accounting, the model downloader's category tag), which names no job and
proves no lane activity."""
# The routine device-wide free-VRAM readout hordelib emits on *every* log_free_ram call. Its
# "reclaimable torch cache" note is present on almost all of them, so matching that note counted the whole
# session's readouts as warnings; the leading "Free VRAM: N MB" value is the only signal of an actually low
# reading. The genuinely-alarming below-reserve streaming warning is a distinct, throttled WARNING line with
# no colon after "Free VRAM" (see :data:`_LOW_VRAM_RESERVE_WARN_RE`).
_LOW_VRAM_READOUT_RE = re.compile(r"Free VRAM: (?P<free_mb>\d+) MB")
# hordelib's throttled warning that measured free VRAM fell below the inference working-set reserve, so a
# sampling step must stream activations over the bus and run several times slower. This is the alarming
# signal (it fires only under the reserve, already rate-limited), so it counts directly.
_LOW_VRAM_RESERVE_WARN_RE = re.compile(r"Free VRAM \d+ MB is below the \d+ MB inference reserve")
# A free-VRAM readout at or below this counts as a genuine low-VRAM dip. Chosen against the 16GB cards this
# lane runs on: their steady-state readouts sit near 9-10GB free, and a sampling working set needs a few GB,
# so a reading under 4GB free is approaching the driver's streaming cliff rather than routine headroom. This
# is a reporting contract, not a scheduling threshold: it only decides which readouts the overlap detector
# treats as evidence.
_LOW_FREE_VRAM_MB_THRESHOLD = 4096
# The feature-level circuit breaker disabling post-processing after a run of over-commit faults. Its trip
# line is the operator advisory: it confirms the spiral reached the self-protective latch (post-processing is
# now off until restart), so a session carrying it is escalated and the remediation points at the restart +
# downgrade. The phrase is the worker's verbatim breaker-trip line (process_manager).
_POST_PROCESSING_BREAKER_RE = re.compile(r"Post-processing fault breaker tripped")
# The parent's measured WDDM demand-paging verdict: worker child allocations demoted to system memory. This is
# the direct proof that co-resident work drove the device past its real headroom, so it corroborates a
# post-processing/inference overlap as a genuine stall rather than admitted co-residency. The phrase is the
# inference scheduler's verbatim rising-edge line (note_wddm_paging).
_WDDM_PAGING_RE = re.compile(r"WDDM demand-paging detected on worker processes")

# A median pop->submit latency this many times the median generation time means jobs are aging in the
# pipeline queue, not in generation (the post-inference safety-backlog signature).
_QUEUE_AGING_LATENCY_RATIO = 3.0

# A run of server-side slow-aborts at or above this is a spiral (the horde will force maintenance),
# not a stray slow job.
_SLOW_ABORT_SPIRAL_THRESHOLD = 3

# A whole-card residency established at or above this many times in a session is reservation churn (the
# process pool repeatedly torn down and rebuilt, safety cycled off/on the GPU), not a single deliberate hold.
_WHOLE_CARD_CHURN_THRESHOLD = 3

# Claims that had to be ended by the maximum hold, with other models' heads parked behind them, at or above
# this count is a pattern rather than one long burst: the resident model is repeatedly holding the offer for
# its full window while the queue carries work it will not let in.
_POP_CLAIM_CAP_MONOPOLY_THRESHOLD = 2

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


def _low_free_vram_records(records: list[LogRecord]) -> list[LogRecord]:
    """Records reporting a genuinely low free-VRAM condition, not the routine full-headroom readout flood.

    A routine readout counts only when its parsed ``Free VRAM: N MB`` value is under
    :data:`_LOW_FREE_VRAM_MB_THRESHOLD`; the below-inference-reserve streaming warning always counts (it fires
    only under the reserve). Order-preserving over the input.
    """
    low: list[LogRecord] = []
    for record in records:
        readout = _LOW_VRAM_READOUT_RE.search(record.message)
        if readout is not None:
            if int(readout.group("free_mb")) < _LOW_FREE_VRAM_MB_THRESHOLD:
                low.append(record)
            continue
        if _LOW_VRAM_RESERVE_WARN_RE.search(record.message):
            low.append(record)
    return low


def _window_key(record: LogRecord) -> datetime:
    """Time key for windowing records: a missing timestamp sorts first, as ``read_records`` orders it."""
    return record.timestamp or datetime.min


def _records_in_window(
    records: list[LogRecord],
    start: datetime | None,
    end: datetime | None,
) -> list[LogRecord]:
    """Slice time-ordered ``records`` to those whose timestamp falls within ``[start, end]`` inclusive.

    ``records`` must be ordered as ``read_records`` yields them (timestamp-less records first, then ascending
    timestamps); the live incremental reader preserves that order for a single-process log. A record with no
    timestamp is never inside a session window, and either bound is open when None. This is a binary-search
    equivalent of filtering each record by ``start <= ts <= end``, so its cost is logarithmic in the cached
    history rather than linear: on a long-running session the child logs grow without bound, and a per-pass
    full scan of them dominated the live watch.
    """
    if start is None:
        # Skip the timestamp-less prefix (it keys as datetime.min and is never inside a window).
        lo = bisect.bisect_right(records, datetime.min, key=_window_key)
    else:
        lo = bisect.bisect_left(records, start, key=_window_key)
    hi = len(records) if end is None else bisect.bisect_right(records, end, key=_window_key)
    return records[lo:hi]


def _child_records_in_session(context: SessionContext) -> list[LogRecord]:
    """Return child loop records whose timestamps fall inside the diagnosed session (windowed per slot)."""
    start, end = context.session.start_ts, context.session.end_ts
    records: list[LogRecord] = []
    for process_id in context.bundle.process_ids():
        records.extend(_records_in_window(context.bundle.child_records(process_id), start, end))
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
    crash_texts: list[str] = []
    for process_id in sorted({r.process_id for r in crashing}):
        first = next(r for r in crashing if r.process_id == process_id)
        crash = find_child_crash(context.bundle, process_id, first.timestamp, os_pid=first.os_pid)
        if crash is not None:
            exceptions[crash.exception] = exceptions.get(crash.exception, 0) + 1
            evidence.append(f"slot {process_id}: {_evidence(crash.record).split('  ', 1)[1]} -> {crash.exception}")
            crash_texts.append(crash.record.full_text)

    cause = max(exceptions, key=lambda exc: exceptions[exc]) if exceptions else None
    git_failure = any(_GIT_ENVIRONMENT_FAILURE_RE.search(text) for text in crash_texts)
    verdict = (
        f"{len(crashing)} inference start(s) across {len({r.process_id for r in crashing})} slot(s) crashed before "
        "reaching readiness"
    )
    if cause is not None and git_failure:
        verdict += f"; the child raised `{cause}` while preparing the shared ComfyUI environment."
        remediation = (
            "The child died installing or repairing the shared ComfyUI environment, not importing the "
            "inference stack, so nothing about this worker's Python packages is implicated. Several children "
            "cold-starting at once clone custom nodes into the same environment directory; a clone "
            "interrupted partway leaves files a later checkout will not overwrite, and every child after it "
            "fails the same way. Clear the environment directory named in the failure and let one start "
            "rebuild it. Newer hordelib serialises this preparation and repairs a half-written tree itself, "
            "so an upgrade prevents the recurrence."
        )
    elif cause is not None:
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
    """Post-processing overlapped with generation, distinguishing an admitted co-residency from a real stall.

    The dedicated post-processing lane keeps the upscaler/face-fixer work out of the inference process, but
    it still allocates on the same GPU. The scheduler admits sampling and post-processing co-residency when
    measured device truth affords it, so a low-free-VRAM reading while that lane is active is a tight-but-
    healthy overlap by default, not an over-commit. Bare co-occurrence therefore surfaces as an
    informational, audit-only finding.

    It escalates to a warning only when a corroborating signal confirms the overlap actually cost the
    device its headroom: a post-processing watchdog reap in the window, a parent WDDM demand-paging verdict
    (worker allocations demoted to system memory), or a child free-VRAM reading below the configured
    inference reserve (not merely low). Dropped jobs, forced maintenance, or a tripped fault breaker escalate
    it further to critical.

    "Low" is a genuine dip, not the routine readout flood: the child logs a free-VRAM readout on every
    heartbeat, so a reading counts only when its parsed free VRAM is under
    :data:`_LOW_FREE_VRAM_MB_THRESHOLD` (or it is the throttled below-inference-reserve streaming warning,
    which is alarming by construction and is itself a corroborating signal). The verdict's count is of those
    genuine readings, not of every readout.
    """
    stalls = _matching(context.session.records, _POST_PROCESSING_STALL_RE)
    breaker_trips = _matching(context.session.records, _POST_PROCESSING_BREAKER_RE)
    child_records = _child_records_in_session(context)
    dedicated_activity = _matching(context.session.records, _DEDICATED_POST_PROCESS_RE) + _matching(
        child_records,
        _DEDICATED_POST_PROCESS_RE,
    )
    low_vram_warnings = _low_free_vram_records(child_records)
    if not stalls and not breaker_trips and not (dedicated_activity and low_vram_warnings):
        return []
    post_processing_recoveries = [
        r for r in context.recoveries if r.last_state in {"INFERENCE_POST_PROCESSING", "POST_PROCESSING"}
    ]
    dropped = _total_dropped_jobs(context.session.records)
    forced_maintenance = bool(_matching(context.session.records, _MAINTENANCE_POP_RE))
    escalated = dropped > 0 or forced_maintenance or bool(breaker_trips)

    # Corroboration that the overlap was a genuine stall rather than admitted co-residency: a watchdog reap,
    # the parent's measured WDDM paging verdict, or a reading below the inference reserve (streaming fallback).
    below_reserve = [r for r in child_records if _LOW_VRAM_RESERVE_WARN_RE.search(r.message)]
    wddm_paging = _matching(context.session.records, _WDDM_PAGING_RE)
    corroborated = bool(stalls) or bool(wddm_paging) or bool(below_reserve)
    if not escalated and not corroborated:
        return [
            Finding(
                id="post_processing_vram_stall",
                severity=Severity.INFO,
                title="Post-processing co-resident with sampling (admitted)",
                verdict=(
                    f"Dedicated post-processing activity coincided with {len(low_vram_warnings)} child "
                    "low-free-VRAM reading(s), but nothing corroborated a stall: no post-processing watchdog "
                    "reap, no WDDM demand-paging, and no reading below the inference reserve. The scheduler "
                    "admits sampling and post-processing co-residency when measured device truth affords it, "
                    "so this is admitted co-residency operating as intended, not a stall. Recorded for audit."
                ),
                remediation=(
                    "No action needed: this is admitted co-residency, not an over-commit. If throughput "
                    "actually degrades, look for a corroborating signal (a post-processing watchdog reap, a "
                    "WDDM demand-paging verdict, or a below-inference-reserve streaming warning), which "
                    "escalates this finding to a warning."
                ),
                evidence=[_evidence(r) for r in (dedicated_activity[:2] + low_vram_warnings[:2])],
                see_also="vram_ram_budget_subsystem",
            ),
        ]
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
                    "; it has already tripped here. It re-enables on its own once the card's measured free VRAM "
                    "recovers above the post-processing peak; downgrade settings (or restart) if the card cannot "
                    "reach that headroom."
                    if breaker_trips
                    else "."
                )
            ),
            evidence=[_evidence(r) for r in (stalls[:4] + low_vram_warnings[:4] + breaker_trips[:1])]
            + [_evidence(r.record) for r in post_processing_recoveries[:2]],
            see_also="vram_ram_budget_subsystem",
        ),
    ]


# Residency-reconciliation holds are benign at low volume; sustained, they are a real swap-churn duty cost.
# A rate past this many holds per hour, or a cumulative parked share past this fraction of the session, is
# worth a warning rather than the informational default.
_RECONCILE_HOLD_RATE_WARNING_PER_HOUR = 30.0
_RECONCILE_HOLD_PARKED_FRACTION_WARNING = 0.05


_PP_DEFER_RE = re.compile(r"Deferring post-processing for job ([0-9a-f][0-9a-f-]{7,35})")
_PP_FINISHED_RE = re.compile(r"Post-processing finished for job")

# A handful of deferrals is healthy backpressure while a transient VRAM spike passes; the same job
# deferred this many times means its headroom condition is structurally unsatisfiable on this card.
_PP_DEFER_STARVATION_THRESHOLD = 30
_PP_DEFER_WARNING_THRESHOLD = 10


def detect_post_processing_deferral_starvation(context: SessionContext) -> list[Finding]:
    """The post-processing lane deferring the same job indefinitely instead of serving or faulting it.

    The lane's admission gate compares a job's marginal post-processing candidate against truthful device-free
    VRAM after outstanding reservations and the admission noise margin. Older logs described a different
    free-after-commitments figure, but both formats share the stable ``Deferring post-processing`` prefix this
    detector keys on. When the measured fit can never be satisfied for the head, the head is deferred every
    scheduling tick, everything behind it waits, and the worker never reports the job faulted. The signature is
    one job accumulating deferral warnings by the tens to hundreds; when no lane completion lands after the
    storm begins, the whole lane is starved, not just the head.
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
        f"({len(defer_records)} job(s) deferred in total). Its marginal post-processing candidate never fit "
        "the card's measured device room, and no bounded no-image fault ever released it, so its finished "
        "inference was held unsubmitted."
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
                "Verify the admission inputs: device-free VRAM must reflect the lane's card, reservations must "
                "include only memory not yet materialized in that measurement, the proportional noise margin "
                "must be applied once, and the per-chain marginal candidate must match measured operation "
                "costs. A deferred drain head should spend ordinary idle cache/model reclaim first, then may "
                "temporarily borrow only a verified-idle service-lane context. It must still age out to a "
                "no-image fault after a bounded wait, and fittable jobs behind it must be allowed to pass. As a "
                "stopgap, lower resident VRAM on the lane's card or disable post-processing on this worker."
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


_HEAD_STARVATION_MODEL_RE = re.compile(
    r"Head-of-queue (?P<model>.+?) deferred (?P<seconds>\d+)s >= \d+s with no verified progress",
)
_HEAD_STARVATION_AVAILABLE_RE = re.compile(r"device-free (?P<free>\d+)")
_HEAD_STARVATION_IDLE_WINDOW_SECONDS = 120.0
"""Repeated head starvation whose device-idle span reaches this reads as a persistent, not transient, stall."""
_HEAD_STARVATION_MIN_DIAGNOSTICS = 2
"""At least this many starvation diagnostics for one model before it reads as repeated rather than a one-off."""


def detect_unsatisfiable_head_starvation(context: SessionContext) -> list[Finding]:
    """One head-of-queue model deferred on an idle device across a long window with no corrective action.

    Distinct from :func:`detect_scheduler_starvation_wedge`, which keys on a wedge that already escalated to a
    soft reset: this recognizes the *persistent, unsatisfiable* case, where the same head model is
    deferred with no verified progress over and over while the device stays idle, and no corrective action (a
    save-our-ship give-up or a consecutive-failure pop-hold) ever clears it. The head is effectively
    unschedulable and the queue is silently wedged behind it. The model name and the device-idle arithmetic
    are read straight off the worker's force-admit diagnostic; the critical case is the one nothing resolved.
    """
    per_model: dict[str, list[tuple[LogRecord, int]]] = {}
    for record in context.session.records:
        match = _HEAD_STARVATION_MODEL_RE.search(record.message)
        if match is not None:
            per_model.setdefault(match.group("model"), []).append((record, int(match.group("seconds"))))
    if not per_model:
        return []

    model, entries = max(per_model.items(), key=lambda item: len(item[1]))
    if len(entries) < _HEAD_STARVATION_MIN_DIAGNOSTICS:
        return []

    timestamps = [record.timestamp for record, _ in entries if record.timestamp is not None]
    span_seconds = (max(timestamps) - min(timestamps)).total_seconds() if len(timestamps) >= 2 else 0.0
    max_starved = max(seconds for _, seconds in entries)
    # The device-idle window is whichever the log makes larger: the reported starvation duration on a single
    # diagnostic, or the wall-clock span across repeated ones (the same head kept starving the whole time).
    idle_window = max(span_seconds, float(max_starved))
    if idle_window < _HEAD_STARVATION_IDLE_WINDOW_SECONDS:
        return []

    free_vrams = [
        int(m.group("free")) for record, _ in entries if (m := _HEAD_STARVATION_AVAILABLE_RE.search(record.message))
    ]
    free_hint = f", with as much as {max(free_vrams)} MB free VRAM on the device" if free_vrams else ""

    storm_start = min(timestamps) if timestamps else None
    corrective = _matching(context.session.records, _GIVE_UP_RE) + _matching(
        context.session.records,
        _CONSECUTIVE_PAUSE_RE,
    )
    resolved = any(
        storm_start is None or (record.timestamp is not None and record.timestamp >= storm_start)
        for record in corrective
    )

    verdict = (
        f"Head-of-queue `{model}` was deferred with no verified progress {len(entries)} time(s) over "
        f"{idle_window:.0f}s (up to {max_starved}s starved{free_hint}): far more headroom and time than the "
        "head needed. The same model kept reaching the head and being deferred on an idle device."
    )
    if resolved:
        verdict += " The worker eventually gave up on the backlog or paused pops, so it did not stay silently wedged."
    else:
        verdict += (
            " No give-up or pop-hold ever cleared it within the window: the head is effectively unschedulable "
            "and the queue is silently wedged behind it."
        )
    return [
        Finding(
            id="unsatisfiable_head_starvation",
            severity=Severity.WARNING if resolved else Severity.CRITICAL,
            title="Head-of-queue model persistently starved on an idle device",
            verdict=verdict,
            remediation=(
                "Confirm the named model actually fits this device (its resident weights plus activation "
                "working set against measured device-free VRAM); a head that never admits despite an idle, "
                "ample-VRAM device points at an over-conservative per-process overhead or an unsatisfiable "
                "budget for that model. Reduce process churn (unload_models_from_vram_often / "
                "high_performance_mode) so a settled baseline sizes the overhead, relax the VRAM budget, or "
                "drop the model if the device genuinely cannot host it. A give-up / pop-hold should bound the "
                "wait so the head cannot starve the queue indefinitely."
            ),
            evidence=[_evidence(record) for record, _ in (entries[:2] + entries[-1:])],
            see_also="scheduler_starvation_wedge",
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


def _asserted_safety_chain(*, cycle_count: int, owners: list[str], retired_count: int) -> str:
    """State the parent-ordered safety cycle as the cause, from the actuator lines that record it.

    The generic wording offers a menu (a crash, a residency cycle, a dropped message) because usually
    nothing in the log says which. When the parent's own pause/restore lines are present they name the
    subsystem that ordered the move, and a verdict dropped on a retired launch is the exact mechanism that
    turns such a move into a stranded job, so the two together are an attribution rather than a candidate.
    """
    parts: list[str] = []
    if cycle_count:
        who = _clause_join(owners) if owners else "the parent"
        parts.append(
            f"{who} moved the safety process off and back onto the GPU {cycle_count} time(s) this session",
        )
    if retired_count:
        parts.append(
            f"{retired_count} verdict(s) arrived from a launch that move had already retired and were dropped",
        )
    return (
        "This was not an unexplained safety crash: " + _clause_join(parts) + ". Each replacement invalidates "
        "the launch that owns the in-flight checks, so the jobs waiting on them have no verdict to receive."
    )


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

    placement_cycles = _matching(records, _SAFETY_PLACEMENT_CYCLE_RE)
    retired_drops = _matching(records, _RETIRED_SAFETY_RESULT_RE)
    owners = _distinct_ordered(
        [m.group("owner").strip() for r in placement_cycles if (m := _SAFETY_PLACEMENT_CYCLE_RE.search(r.message))],
    )

    if escalated:
        verdict = (
            f"The safety stage stranded jobs whose verdict never returned ({detail}). The worker faulted the "
            "unservable ones with no image and soft-paused pops, but those faults count as dropped jobs and can "
            "draw horde-forced maintenance. "
        )
        if placement_cycles or retired_drops:
            # The parent's own actuator is in the log, so the chain is readable end to end and there is no
            # reason to hand back a list of candidate causes.
            verdict += _asserted_safety_chain(
                cycle_count=len(placement_cycles),
                owners=owners,
                retired_count=len(retired_drops),
            )
            remediation = (
                "The cycle is the worker's own, so the fix is upstream of safety: stop the subsystem named "
                "above from moving safety off the GPU as often as it does, or make it hand the in-flight "
                "checks over rather than retiring the launch that owns them. A verdict dropped on a retired "
                "launch is unrecoverable for that job by construction, so reducing the cycling rate is what "
                "reduces the drops."
            )
        else:
            verdict += (
                "A lost safety verdict (a cycled/replaced safety process, or a dropped result message) is the "
                "root cause."
            )
            remediation = (
                "Stabilise the safety process: with safety_on_gpu set, frequent whole-card residency cycling or "
                "unload_models_from_vram_often can churn it; check the bridge_safety_*.log for crashes. The worker "
                "re-checks and (only as a last resort) faults with no image, so no unchecked image is ever "
                "submitted."
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

    evidence = (
        placement_cycles[:1]
        + retired_drops[:1]
        + unrecoverable[:1]
        + soft_pauses[:1]
        + lost_results[:1]
        + requeues[:1]
        + backpressure[:1]
    )
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


@dataclass(frozen=True)
class _WholeCardClaim:
    """One reservation's own arithmetic, as the establish line states it."""

    model: str
    current: int
    after: int
    total: int
    target: int
    free_mb: int | None

    @property
    def demands_no_reduction(self) -> bool:
        """Whether the claim's target was already satisfied by the live process count.

        A residency is granted to make room by stopping siblings. A target at or above the live count asks
        for no sibling to stop, so the reservation reserves the card and returns nothing: the deficit that
        justified it cannot be closed by the action it authorized.
        """
        return self.target >= self.current

    @property
    def counts_text(self) -> str:
        """The claim's process-count arithmetic as the establish line words it."""
        return f"{self.current} -> {self.after} of {self.total}, target {self.target}"


def _whole_card_claims(records: list[LogRecord]) -> list[_WholeCardClaim]:
    """Parse the figures out of each whole-card establish line, skipping any that state none."""
    claims: list[_WholeCardClaim] = []
    for record in records:
        fields = _WHOLE_CARD_ESTABLISH_FIELDS_RE.search(record.message)
        if fields is None:
            continue
        free = _DEVICE_FREE_VRAM_RE.search(record.message)
        claims.append(
            _WholeCardClaim(
                model=fields.group("model"),
                current=int(fields.group("current")),
                after=int(fields.group("after")),
                total=int(fields.group("total")),
                target=int(fields.group("target")),
                free_mb=int(free.group(1)) if free is not None else None,
            ),
        )
    return claims


def _claim_figures_text(claims: list[_WholeCardClaim], *, count: int) -> str:
    """The process-count and free-VRAM figures the reservations carried, or a note that they carried none."""
    if not claims:
        return f"None of the {count} reservation(s) stated its process-count arithmetic."
    counts = _distinct_ordered([claim.counts_text for claim in claims])
    text = f"Its stated arithmetic across {len(claims)} reservation(s): {_clause_join(counts)}."
    frees = [claim.free_mb for claim in claims if claim.free_mb is not None]
    if frees:
        text += (
            f" Device free VRAM at reservation ranged {min(frees)}-{max(frees)}MB."
            if min(frees) != max(frees)
            else f" Device free VRAM at reservation was {frees[0]}MB."
        )
    return text


@dataclass(frozen=True)
class _StreamForecast:
    """The establishment-time forecast disclosure: what the scheduler priced the claim against."""

    model: str
    marginal_mb: float | None
    marginal_source: str | None
    unreclaimable_mb: float | None
    needs_reduction: bool | None

    @property
    def marginal_is_seeded(self) -> bool:
        """Whether the per-additional-context cost came from the seeded constant (nothing measurable)."""
        return self.marginal_source is not None and self.marginal_source.startswith("seed")

    def describe(self) -> str:
        """The cause figures, worded so the marginal's provenance is visible rather than assumed."""
        parts: list[str] = []
        if self.marginal_mb is not None and self.marginal_source is not None:
            provenance = "seeded, not measured" if self.marginal_is_seeded else f"measured from {self.marginal_source}"
            parts.append(f"per-context marginal {self.marginal_mb:.0f}MB ({provenance})")
        elif self.marginal_source is not None:
            parts.append(f"per-context marginal unavailable (src={self.marginal_source})")
        if self.unreclaimable_mb is not None:
            parts.append(f"{self.unreclaimable_mb:.0f}MB charged as unreclaimable")
        if self.needs_reduction is not None:
            parts.append(f"needs_process_count_reduction={self.needs_reduction}")
        if not parts:
            return ""
        return f"The forecast for {self.model} priced it with {_clause_join(parts)}."


def _last_stream_forecast(records: list[LogRecord]) -> _StreamForecast | None:
    """The most recent stream-forecast disclosure in the session, or None when the capture carries none.

    Older captures predate the ``unreclaimable`` and ``needs_process_count_reduction`` fields, so each is
    parsed independently and stays None when absent rather than disqualifying the whole disclosure.
    """
    for record in reversed(records):
        header = _STREAM_FORECAST_RE.search(record.message)
        if header is None:
            continue
        marginal = _FORECAST_MARGINAL_RE.search(record.message)
        unreclaimable = _FORECAST_UNRECLAIMABLE_RE.search(record.message)
        reduction = _FORECAST_REDUCTION_RE.search(record.message)
        chosen = marginal.group("marginal") if marginal is not None else None
        return _StreamForecast(
            model=header.group("model"),
            marginal_mb=float(chosen) if chosen is not None and chosen.isdigit() else None,
            marginal_source=marginal.group("source") if marginal is not None else None,
            unreclaimable_mb=float(unreclaimable.group("unreclaimable")) if unreclaimable is not None else None,
            needs_reduction=reduction.group("reduction") == "True" if reduction is not None else None,
        )
    return None


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

    claims = _whole_card_claims(establishes)
    forecast = _last_stream_forecast(context.session.records)
    incoherent = [claim for claim in claims if claim.demands_no_reduction]
    all_incoherent = bool(claims) and len(incoherent) == len(claims)
    unmeasured_marginal = forecast is not None and forecast.marginal_is_seeded

    headline = (
        (
            f"Every one of the {len(claims)} reservation(s) that stated its arithmetic demanded no reduction: "
            f"the target process count was at or above the live count (e.g. {incoherent[0].counts_text}), so the "
            "claim reserved the card without the teardown it was granted for. The claim is structurally "
            "incoherent, not merely over-eager. "
        )
        if all_incoherent
        else (
            f"The whole card was reserved {len(establishes)} time(s) this session, each time reducing the live "
            "process count and cycling safety off the GPU, then restoring them. Sustained reservation churn is "
            "the signature of a model being given the card that does not need it; on a high-VRAM card a model "
            "whose weights are a small fraction of total VRAM co-resides, so a teardown demand for it usually "
            "means the per-context overhead was over-counted"
            + (" (an unmeasured marginal). " if unmeasured_marginal else ". ")
        )
    )
    return [
        Finding(
            id="whole_card_residency_churn",
            severity=Severity.CRITICAL if escalated or all_incoherent else Severity.WARNING,
            title="Whole-card residency reserved and restored repeatedly (reservation churn)",
            verdict=(
                headline
                + _claim_figures_text(claims, count=len(establishes))
                + (f" {forecast.describe()}" if forecast is not None else "")
                + (
                    f" It escalated: the recovery supervisor soft-reset the pools {len(soft_resets)} time(s) and "
                    f"faulted {dropped} backlog job(s)."
                    if escalated
                    else " It did not escalate to a soft reset here, but the reload + safety cycling caps throughput."
                )
                + (
                    f" The trust gate declined {len(declined)} further reservation(s), so it is actively damping "
                    "the churn."
                    if declined
                    else ""
                )
            ),
            remediation=_churn_remediation(
                all_incoherent=all_incoherent,
                unmeasured_marginal=unmeasured_marginal,
                forecast=forecast,
            ),
            evidence=[_evidence(r) for r in (establishes[:2] + soft_resets[:1])],
            see_also="scheduler_starvation_wedge",
        ),
    ]


def _churn_remediation(
    *,
    all_incoherent: bool,
    unmeasured_marginal: bool,
    forecast: _StreamForecast | None,
) -> str:
    """The fix keyed to what the parsed figures actually show, rather than to one assumed cause.

    Blaming an unmeasured per-additional-context marginal is only true when the forecast says the marginal
    is seeded (nothing measurable); with a probe or idle-floor source it is measured, and asserting
    otherwise sends an operator after a number that is already correct.
    """
    if all_incoherent:
        return (
            "The target process count is at or above the live count, so the reservation cannot free anything "
            "by tearing siblings down: fix the target the forecast derives before tuning any VRAM figure. "
            "Check how the residency target is computed against the live process count, and whether the "
            "deficit driving it is made of charges a teardown cannot return"
            + (
                f" (the forecast attributes {forecast.unreclaimable_mb:.0f}MB as unreclaimable)."
                if forecast is not None and forecast.unreclaimable_mb is not None
                else "."
            )
        )
    if unmeasured_marginal:
        return (
            "The forecast is pricing additional contexts from a seeded constant, so nothing was measured. "
            "Get the per-additional-context VRAM cost measured (the probe's second-context delta or a clean "
            "all-idle baseline), so the structural free-VRAM floor is not the one-time runtime cost "
            "multiplied by the process count. A correctly measured marginal lets a small-weight model "
            "co-reside instead of reserving the card."
        )
    return (
        "The per-additional-context cost is measured, so the reservations are not the unmeasured-marginal "
        "case: confirm the reserved models are genuinely card-filling for this card, and check what the "
        "forecast charges as unreclaimable, since that deduction is what a teardown cannot give back and so "
        "is what keeps demanding the card."
    )


@dataclass(frozen=True)
class _PopClaimEpisode:
    """One span during which a whole-card residency held the worker's pop offer to a single model."""

    model: str
    engaged: LogRecord
    released: LogRecord | None
    release: str | None
    """The phrase naming which of the claim's ends fired, or None for a claim still standing at the end."""

    @property
    def seconds(self) -> float | None:
        """How long the claim stood, or None when either edge carried no timestamp."""
        start = self.engaged.timestamp
        end = self.released.timestamp if self.released is not None else None
        if start is None or end is None:
            return None
        return max(0.0, (end - start).total_seconds())


def _pop_claim_episodes(records: list[LogRecord]) -> list[_PopClaimEpisode]:
    """Pair the pop claim's engage and release disclosures into episodes, in order.

    Both edges are stated once per episode, so pairing each engage with the next release reconstructs the
    spans. A claim still standing at the last record is kept with no release: an episode the session ended
    inside is exactly the one an operator asking why the offer is narrow wants to see.
    """
    episodes: list[_PopClaimEpisode] = []
    open_engaged: LogRecord | None = None
    open_model = ""
    for record in records:
        engaged = _POP_CLAIM_ENGAGED_RE.search(record.message)
        if engaged is not None:
            open_engaged = record
            open_model = engaged.group("model")
            continue
        released = _POP_CLAIM_RELEASED_RE.search(record.message)
        if released is None or open_engaged is None:
            continue
        episodes.append(
            _PopClaimEpisode(
                model=open_model or released.group("model"),
                engaged=open_engaged,
                released=record,
                release=released.group("release").strip(),
            ),
        )
        open_engaged = None
        open_model = ""
    if open_engaged is not None:
        episodes.append(_PopClaimEpisode(model=open_model, engaged=open_engaged, released=None, release=None))
    return episodes


def _foreign_heads_parked_during(records: list[LogRecord], episode: _PopClaimEpisode) -> dict[str, int]:
    """Return how often each other model's head was parked while ``episode``'s claim stood.

    A parked head for a model the claim does not advertise is work the worker already accepted and cannot
    make progress on, sitting behind an offer that will not let its model's siblings in. Counted per model so
    the finding can say which work the claim was squeezing. Empty when the episode's edges are untimestamped,
    since nothing can then be placed inside it.
    """
    end = episode.released.timestamp if episode.released is not None else None
    parked: dict[str, int] = {}
    for record in _records_in_window(records, episode.engaged.timestamp, end):
        fields = _DISPATCH_STALL_FIELDS_RE.search(record.message)
        if fields is None or fields.group("model") == episode.model:
            continue
        parked[fields.group("model")] = parked.get(fields.group("model"), 0) + 1
    return parked


def detect_whole_card_pop_claim_episodes(context: SessionContext) -> list[Finding]:
    """The spans in which a whole-card residency held the pop offer to its own model.

    A claim changes what the worker asks the horde for, not what it does with what arrives, so it leaves no
    trace in the job stream: a session where one model served everything looks the same whether that was the
    demand or the claim. Reporting the episodes makes the difference visible, and the release reasons say how
    each one ended, which is what distinguishes a burst that finished (the demand dried up) from one the
    maximum hold had to stop.
    """
    episodes = _pop_claim_episodes(context.session.records)
    if not episodes:
        return []

    per_model: dict[str, int] = {}
    per_release: dict[str, int] = {}
    durations: list[float] = []
    for episode in episodes:
        per_model[episode.model] = per_model.get(episode.model, 0) + 1
        per_release[episode.release or "still standing at the end of the session"] = (
            per_release.get(episode.release or "still standing at the end of the session", 0) + 1
        )
        seconds = episode.seconds
        if seconds is not None:
            durations.append(seconds)

    model_breakdown = ", ".join(
        f"{model} x{count}" for model, count in sorted(per_model.items(), key=lambda kv: -kv[1])
    )
    release_breakdown = ", ".join(
        f"{release} x{count}" for release, count in sorted(per_release.items(), key=lambda kv: -kv[1])
    )
    held_text = (
        f" They held the offer for {sum(durations):.0f}s in total (longest {max(durations):.0f}s)."
        if durations
        else ""
    )
    return [
        Finding(
            id="whole_card_pop_claim_episodes",
            severity=Severity.INFO,
            title="Whole-card residency claimed the pop offer",
            verdict=(
                f"A whole-card residency narrowed the worker's advertised model set to its own model "
                f"{len(episodes)} time(s) this session. Per model: {model_breakdown}. How each ended: "
                f"{release_breakdown}.{held_text} While a claim stands the horde is only asked for that model, "
                "which is what keeps the resident weights on the card."
            ),
            remediation=(
                "Nothing to do if the claims match the heavy work this worker exists to serve. If they are "
                "long and frequent for a model the operator did not intend to specialise in, the levers are "
                "the served model set and whole_card_residency_max_hold_seconds, which caps how long one "
                "residency may own the intake."
            ),
            evidence=[_evidence(episode.engaged) for episode in episodes[:3]],
            see_also="whole_card_pop_claim_monopoly",
        ),
    ]


def detect_whole_card_pop_claim_monopoly(context: SessionContext) -> list[Finding]:
    """Claims repeatedly ended by the maximum hold while other models' heads were parked behind them.

    The claim's self-correcting end is the run of empty pops: when the resident model's demand dries up the
    claim releases itself. A claim that instead runs to the cap was still narrowing the offer when the cap
    stopped it, and doing that repeatedly while heads of other models sit parked is a mixed queue being
    squeezed: the accepted work cannot progress and the offer will not admit anything that would relieve it.
    One such episode is a legitimately long burst; a pattern of them is the residency outstaying the demand
    that justified it.
    """
    episodes = _pop_claim_episodes(context.session.records)
    squeezing = [
        (episode, _foreign_heads_parked_during(context.session.records, episode))
        for episode in episodes
        if episode.release == _POP_CLAIM_CAP_RELEASE
    ]
    squeezing = [(episode, parked) for episode, parked in squeezing if parked]
    if len(squeezing) < _POP_CLAIM_CAP_MONOPOLY_THRESHOLD:
        return []

    starved: dict[str, int] = {}
    for _episode, parked in squeezing:
        for model, count in parked.items():
            starved[model] = starved.get(model, 0) + count
    starved_breakdown = ", ".join(
        f"{model} x{count}" for model, count in sorted(starved.items(), key=lambda kv: -kv[1])
    )
    claimants = sorted({episode.model for episode, _parked in squeezing})
    return [
        Finding(
            id="whole_card_pop_claim_monopoly",
            severity=Severity.WARNING,
            title="Whole-card pop claim held to its cap while other models' work waited",
            verdict=(
                f"{len(squeezing)} pop claim(s) ran to the maximum hold rather than releasing on their own, and "
                f"each did so with another model's head parked behind it. Claimed by: {', '.join(claimants)}. "
                f"Heads parked meanwhile: {starved_breakdown}. A claim that has to be ended by its cap was still "
                "asking the horde for one model only while the queue held work it would not admit, so accepted "
                "jobs aged behind a narrowed offer."
            ),
            remediation=(
                "Decide whether this worker should specialise. If it should, the parked models do not belong in "
                "its served set and removing them ends the contention. If it should serve a mix, lower "
                "whole_card_residency_max_hold_seconds so a residency gives the intake back sooner, or take the "
                "claimed model out of the pool if it can only run with the whole card."
            ),
            evidence=[_evidence(episode.engaged) for episode, _parked in squeezing[:3]],
            see_also="whole_card_pop_claim_episodes",
        ),
    ]


def detect_head_dispatch_stall(context: SessionContext) -> list[Finding]:
    """A head-of-queue job that did not dispatch despite pending work and an idle, model-resident process.

    The scheduler returns ``None`` silently from several gates, so a stuck queue with idle processes used to
    leave no record of *why* the head was parked. The new dispatch-stall log names the blocking gate; the
    "no matching gate" variant is the genuinely anomalous case (the head's model is resident and idle, no
    gate is holding it, yet nothing dispatched) and is reported as critical, the rest as a warning. The
    whole-card convergence wedge (:func:`detect_whole_card_convergence_wedge`), the non-head residency
    starvation (:func:`detect_whole_card_nonhead_residency_starvation`), and the residency-reconciliation
    holds (:func:`detect_residency_reconciliation_holds`) have their own detectors, so their lines are
    excluded here to avoid double-reporting the same stall as a generic warning. A reconcile hold in
    particular is a benign swap-churn wait, not a gate-less scheduler stall, and must never read as the bug.
    """
    excluded = (
        _matching(context.session.records, _WHOLE_CARD_WEDGE_RE)
        + _matching(context.session.records, _WHOLE_CARD_NONHEAD_RE)
        + _matching(context.session.records, _DISPATCH_STALL_RECONCILE_RE)
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


def detect_residency_reconciliation_holds(context: SessionContext) -> list[Finding]:
    """The dispatch residency-reconciliation gate holding a resident head while it evicts idle VRAM.

    When a resident-idle head's on-device materialisation would over-commit the card, the scheduler holds its
    dispatch for a few ticks while it evicts an idle sibling's VRAM, then dispatches once the room is freed.
    This is not a scheduler bug: it is the swap-churn cost of packing more models onto one card than fit at
    peak. Reported so that cost is visible and not mistaken for a wedge. Benign (informational) at low volume;
    a warning when the holds are frequent (past :data:`_RECONCILE_HOLD_RATE_WARNING_PER_HOUR`) or when their
    cumulative parked time is a meaningful share of the session (past
    :data:`_RECONCILE_HOLD_PARKED_FRACTION_WARNING`). The per-hold parked seconds are read off the stall
    wrapper; because the parked-head log is throttled, the cumulative figure is a sample of the true parked
    time, not an exact total.
    """
    holds = [
        r
        for r in _matching(context.session.records, _DISPATCH_STALL_RE)
        if _DISPATCH_STALL_RECONCILE_RE.search(r.message)
    ]
    if not holds:
        return []

    per_model: dict[str, int] = {}
    parked_seconds_total = 0.0
    for record in holds:
        fields = _DISPATCH_STALL_FIELDS_RE.search(record.message)
        if fields is None:
            continue
        per_model[fields.group("model")] = per_model.get(fields.group("model"), 0) + 1
        parked_seconds_total += float(fields.group("parked"))

    duration = context.session.duration_seconds
    holds_per_hour = (len(holds) / duration * 3600.0) if duration else 0.0
    parked_fraction = (parked_seconds_total / duration) if duration else 0.0
    escalated = (
        holds_per_hour > _RECONCILE_HOLD_RATE_WARNING_PER_HOUR
        or parked_fraction > _RECONCILE_HOLD_PARKED_FRACTION_WARNING
    )
    # The benign reading rests entirely on each hold self-clearing once an eviction frees room. A deadlock
    # declaration or a recovery remedy inside the span the holds cover says the opposite happened: the room
    # never came and the worker escalated. Reporting that as low-cost duty noise sends a reader looking for a
    # throughput tweak while the queue is stopped.
    hold_span = [record.timestamp for record in holds if record.timestamp is not None]
    wedge_lines = (
        [
            record
            for record in _matching(context.session.records, _WEDGE_ESCALATION_RE)
            if record.timestamp is not None and min(hold_span) <= record.timestamp <= max(hold_span)
        ]
        if hold_span
        else []
    )
    wedged = bool(wedge_lines)
    escalated = escalated or wedged

    model_breakdown = ", ".join(
        f"{model} x{count}" for model, count in sorted(per_model.items(), key=lambda kv: -kv[1])
    )
    return [
        Finding(
            id="residency_reconciliation_holds",
            severity=Severity.WARNING if escalated else Severity.INFO,
            title="Dispatch held to reconcile residency (idle-VRAM eviction swap-churn)",
            verdict=(
                f"The scheduler held a resident head's dispatch to reconcile residency {len(holds)} time(s) "
                f"(~{holds_per_hour:.0f}/hour), evicting an idle sibling's VRAM so the head's materialisation "
                f"fit the card before it committed. Per model: {model_breakdown}. Roughly {parked_seconds_total:.0f}s "
                f"of head parking was observed across these holds"
                + (f" (~{parked_fraction:.0%} of the session)." if duration else ".")
                + (
                    " These holds did not self-clear: the worker declared the queue deadlocked (or ran a "
                    f"recovery remedy) {len(wedge_lines)} time(s) inside the same span, so the room the hold "
                    "was waiting for never arrived and something else had to break the stall."
                    if wedged
                    else " This is the swap-churn cost of packing more models onto the card than fit at peak, "
                    "not a scheduler wedge: each hold self-clears once the eviction frees room."
                    + (
                        " Sustained at this volume it is a real throughput and GPU-uptime drag."
                        if escalated
                        else " At this volume it is a benign, low-cost duty note."
                    )
                )
            ),
            remediation=(
                "Treat this as a wedge rather than churn: find what held the card across the span (an idle "
                "lane's component tenancy and a slot parked on a preload are both holders no job boundary "
                "returns) and confirm the hold actually issued a reclaim against it."
                if wedged
                else "If this volume is high, reduce the co-resident model pressure that forces the "
                "evictions: lower the served model set or concurrency for this VRAM size, or confirm the "
                "per-context VRAM cost is measured so the card is not over-packed at peak. A handful of "
                "holds is normal headroom management and needs no action."
            ),
            evidence=[_evidence(r) for r in holds[:3]],
            see_also="head_dispatch_stall",
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


@dataclass(frozen=True)
class _FaultedJob:
    """One job the worker reported faulted, with the cause read off the lines around it."""

    job_id: str
    record: LogRecord
    model: str | None
    cause: str

    def describe(self) -> str:
        """A census line: when, which job, which model, and what caused it."""
        ts = self.record.timestamp.strftime("%H:%M:%S") if self.record.timestamp else "--:--:--"
        return f"{ts}  {self.job_id[:8]}  {self.model or 'unknown model'}  [{self.cause}]"


_MODEL_REFERENCE_STALE_WINDOW_SECONDS = 60.0
"""How near a model-reference fault a cache-staleness reading has to be to be evidence about it.

The staleness disclosure fires on every category file the backend re-checks, so it is present throughout
a session; only the readings bracketing the fault attribute it to a refresh in flight."""


_GIVE_UP_ATTRIBUTION_SECONDS = 30.0
"""How long after a give-up backstop a fault report is still that backstop's doing.

The backstop faults its jobs and the submitter reports each one in the same burst, so the window only has
to cover the submits themselves; wider, and an unrelated later fault would be misattributed to it."""


def _faulted_job_models(records: list[LogRecord]) -> dict[str, str]:
    """Map job id to the model it was popped for, from the pop lines that state both."""
    models: dict[str, str] = {}
    for record in records:
        popped = _POPPED_JOB_RE.search(record.message)
        if popped is not None:
            models[popped.group("job")] = popped.group("model")
    return models


def _classify_fault(
    job_id: str,
    record: LogRecord,
    *,
    id_anchored: dict[str, str],
    give_up_times: list[datetime],
) -> str:
    """Name the cause of one fault, preferring evidence that names the job over evidence that only co-occurs."""
    anchored = id_anchored.get(job_id)
    if anchored is not None:
        return anchored
    timestamp = record.timestamp
    if timestamp is not None and any(
        0 <= (timestamp - given_up).total_seconds() <= _GIVE_UP_ATTRIBUTION_SECONDS for given_up in give_up_times
    ):
        return "give-up backstop"
    return "other"


def _faulted_jobs(context: SessionContext) -> list[_FaultedJob]:
    """Every job faulted this session, deduplicated across the two surfaces that report a fault.

    A job can be seen faulting on the slot that ran it and again as the terminal report to the horde; both
    name the same job, so the census keys on the job id and counts it once.
    """
    parent = context.session.records
    child = _child_records_in_session(context)
    models = _faulted_job_models(parent)
    give_up_times = [record.timestamp for record in _matching(parent, _GIVE_UP_RE) if record.timestamp is not None]

    id_anchored: dict[str, str] = {}
    for record in parent:
        # A job popped with no model name is unservable before anything touches it, whichever generation
        # of the worker handled it, so it is never a model or a device failure however it faults later.
        malformed = _EMPTY_MODEL_POP_RE.search(record.message) or _MALFORMED_POP_REJECTED_RE.search(record.message)
        if malformed is not None:
            id_anchored[malformed.group("job")] = "malformed pop (no model name)"
    for record in child + parent:
        stage = _STAGE_FAULT_RE.search(record.message)
        if stage is not None:
            id_anchored.setdefault(stage.group("job"), "disaggregation stage fault")
        safety = _SAFETY_UNRECOVERABLE_JOB_RE.search(record.message)
        if safety is not None:
            id_anchored[safety.group("job")] = "safety-unrecoverable"

    faults: dict[str, _FaultedJob] = {}
    for record in parent:
        slot_fault = _FAULTED_ON_PROCESS_JOB_RE.search(record.message)
        requeued = slot_fault is not None and _FAULT_REQUEUED_RE.search(record.message) is not None
        if slot_fault is not None:
            job_id = slot_fault.group("job")
            model = _FAULT_MODEL_RE.search(record.message)
            if not requeued:
                id_anchored.setdefault(job_id, "process fault")
            if model is not None:
                models.setdefault(job_id, model.group("model"))
        reported = _FAULT_REPORTED_RE.search(record.message)
        if reported is None and requeued:
            # An attempt that was handed back for a retry has not cost the horde a job; only the attempt
            # that exhausts the retry policy, or the terminal report, is a faulted job.
            continue
        job_id = reported.group("job") if reported is not None else (slot_fault and slot_fault.group("job"))
        if job_id is None:
            continue
        # The terminal report is the authoritative record of a fault, so it supersedes the slot-side one.
        if job_id in faults and reported is None:
            continue
        faults[job_id] = _FaultedJob(
            job_id=job_id,
            record=record,
            model=models.get(job_id),
            cause=_classify_fault(job_id, record, id_anchored=id_anchored, give_up_times=give_up_times),
        )
    return sorted(faults.values(), key=lambda fault: _window_key(fault.record))


def detect_faulted_job_census(context: SessionContext) -> list[Finding]:
    """Every job faulted this session, enumerated with its model and a cause.

    A faulted job is work the horde reissues and counts against the worker, yet each fault otherwise shows
    up only as an aside inside whichever detector recognized its cause, and jobs faulted for a cause no
    detector covers show up nowhere. A flat census answers "how much did this session drop, and to what"
    directly, and the per-cause counts say which of the drop mechanisms actually fired.
    """
    faults = _faulted_jobs(context)
    if not faults:
        return []
    counts: dict[str, int] = {}
    for fault in faults:
        counts[fault.cause] = counts.get(fault.cause, 0) + 1
    breakdown = _clause_join([f"{count} {cause}" for cause, count in sorted(counts.items(), key=lambda kv: -kv[1])])
    models = _distinct_ordered([fault.model for fault in faults if fault.model is not None])
    return [
        Finding(
            id="faulted_job_census",
            severity=Severity.WARNING,
            title="Jobs faulted this session",
            verdict=(
                f"{len(faults)} job(s) faulted this session, by cause: {breakdown}."
                + (f" Across model(s): {_clause_join(models)}." if models else "")
            ),
            remediation=(
                "Faulted jobs are reissued by the horde and counted against this worker; a sustained rate "
                "drives forced maintenance. Take the largest cause first: a give-up backstop count means the "
                "scheduler wedged and the recovery path drained the backlog rather than serving it, a "
                "safety-unrecoverable count means results were produced but could not be checked, and a "
                "process-fault count points at the slot that ran them."
            ),
            evidence=[fault.describe() for fault in faults[:8]],
            see_also="scheduler_starvation_wedge",
        ),
    ]


def _open_governor_at(records: list[LogRecord], moment: datetime, name: str) -> LogRecord | None:
    """The ENTER record of governor ``name``'s spell if it was still open at ``moment``, else None."""
    opened: LogRecord | None = None
    for record in records:
        if record.timestamp is None or record.timestamp > moment:
            break
        enter = GOVERNOR_ENTER_RE.search(record.message)
        if enter is not None and enter.group("name") == name:
            opened = record
            continue
        closed = GOVERNOR_EXIT_RE.search(record.message)
        if closed is not None and closed.group("name") == name:
            opened = None
    return opened


def detect_pop_liveness_full_queue(context: SessionContext) -> list[Finding]:
    """The local job queue was full and stopped moving: the worker accepted work and served none of it.

    Every other stall signal describes a part of the pipeline holding back; this one says the whole of it
    stopped. The queue being full legitimately holds pops back, so a frozen full queue is indistinguishable
    from a healthy worker at its configured depth on the pop side alone, and the worker only separates them
    by movement: nothing dispatched and nothing completed for the whole span. That makes it the most severe
    liveness signal the worker emits, and the one whose absence from a report is most misleading.
    """
    frozen = _matching(context.session.records, _POP_LIVENESS_FROZEN_RE)
    if not frozen:
        return []
    worst = 0
    heads: list[str] = []
    for record in frozen:
        fields = _POP_LIVENESS_FROZEN_FIELDS_RE.search(record.message)
        if fields is None:
            continue
        worst = max(worst, int(fields.group("seconds")))
        heads.append(fields.group("model"))
    holders = [
        _open_governor_at(context.session.records, record.timestamp, "whole_card_residency")
        for record in frozen
        if record.timestamp is not None
    ]
    residency_held = any(holder is not None for holder in holders)
    held_models = _distinct_ordered(
        [
            model.group("model")
            for holder in holders
            if holder is not None and (model := _RESIDENCY_GOVERNOR_MODEL_RE.search(holder.message)) is not None
        ],
    )
    return [
        Finding(
            id="pop_liveness_full_queue",
            severity=Severity.CRITICAL,
            title="Local job queue full and not draining (the worker served nothing)",
            verdict=(
                f"The local queue was reported full and motionless {len(frozen)} time(s), for as long as "
                f"{worst}s with nothing dispatched and nothing completed"
                + (f", head model {_clause_join(_distinct_ordered(heads))}" if heads else "")
                + ". Accepted work sat while the worker asked the horde for none, so this is total stall, not "
                "backpressure."
                + (
                    " A whole-card residency governor spell was open across the freeze, so the card was "
                    "reserved for one model while the head waited for another"
                    + (f" ({_clause_join(held_models)})." if held_models else ".")
                    if residency_held
                    else ""
                )
            ),
            remediation=(
                "Find what the head was waiting for over that span: the disclosure names the scheduler's own "
                "block reason where it has one."
                + (
                    " A residency held for a non-head model is the usual holder, and it must release or "
                    "downgrade rather than outlast the queue."
                    if residency_held
                    else " No whole-card residency was open across the freeze, so the holder is something the "
                    "card is carrying between jobs: an idle lane's component tenancy, a slot parked on a "
                    "preload nothing dispatched, or weights a child reported freeing and did not."
                )
                + " Until then the accepted jobs age toward their deadline and are faulted, which the horde "
                "counts against this worker."
            ),
            evidence=[_evidence(record) for record in frozen[:3]],
            see_also="whole_card_residency_churn",
        ),
    ]


def detect_model_reference_sample_fault(context: SessionContext) -> list[Finding]:
    """A sample stage faulted because the model reference was unreadable while the job was in flight.

    The reference is loaded once and refreshed in the background when its cached category files change on
    disk. A refresh that lands mid-sample leaves the child asking for a category the cache is rewriting, and
    the job faults on a data-availability error that has nothing to do with the model, the card, or the
    prompt. It reads as a generic stage fault, so it needs its own signature to be attributable at all.
    """
    # The fault lands in a slot log and the staleness disclosures can come from either side, so both are
    # merged into one time-ordered stream before any windowing.
    records = sorted(_child_records_in_session(context) + context.session.records, key=_window_key)
    faults = [
        record
        for record in records
        if _STAGE_FAULT_RE.search(record.message) and _MODEL_REFERENCE_UNREADABLE_RE.search(record.message)
    ]
    if not faults:
        return []
    # The staleness disclosure is routine and constant across a session, so only the readings that bracket
    # a fault say anything about it; counting them all would put a five-figure number next to one fault.
    stale = [
        record
        for fault in faults
        if fault.timestamp is not None
        for record in _records_in_window(
            records,
            fault.timestamp - timedelta(seconds=_MODEL_REFERENCE_STALE_WINDOW_SECONDS),
            fault.timestamp + timedelta(seconds=_MODEL_REFERENCE_STALE_WINDOW_SECONDS),
        )
        if _MODEL_REFERENCE_STALE_RE.search(record.message)
    ]
    categories = _distinct_ordered(
        [m.group("category") for record in faults if (m := _MODEL_REFERENCE_UNREADABLE_RE.search(record.message))],
    )
    jobs = _distinct_ordered([m.group("job") for record in faults if (m := _STAGE_FAULT_RE.search(record.message))])
    return [
        Finding(
            id="model_reference_sample_fault",
            severity=Severity.WARNING,
            title="Sample stage faulted on an unreadable model reference",
            verdict=(
                f"{len(faults)} sample stage(s) faulted because the {_clause_join(categories)} model reference "
                f"could not be read, affecting job(s) {_clause_join([job[:8] for job in jobs[:4]])}. The attempt "
                "was lost to a data-availability error, not to anything about the model or the device (a retry "
                "can still save the job, at the cost of the work already done)."
                + (
                    f" {len(stale)} cache-staleness line(s) sit alongside them, so a reference refresh was in "
                    "flight while the sample ran."
                    if stale
                    else ""
                )
            ),
            remediation=(
                "The model-reference cache refresh is racing an in-flight sample: the child re-reads a category "
                "while the cache is rewriting it. Hold the reference the job was admitted with for the life of "
                "the job (or make the refresh atomic from a reader's point of view) so a background refresh "
                "cannot fault work already running. Check the horde_model_reference cache path for the affected "
                "category and confirm it is readable and not being rewritten by a second process."
            ),
            evidence=[_evidence(record) for record in faults[:3]],
            see_also="faulted_job_census",
        ),
    ]


def detect_empty_model_pop_cascade(context: SessionContext) -> list[Finding]:
    """A pop that named no model, and how far into the worker it travelled.

    An empty model name identifies nothing any worker can load, so the job is unservable whatever happens
    to it. What decides the cost is where it is stopped. Without a boundary check the blank identity is
    dispatched like any other model: it is preloaded, ends the slot it is sent to, and is then counted as a
    model in its own right until it crosses the quarantine threshold, after which every later job for it is
    refused too. The pool churn and the poisoned quarantine both follow from one malformed response, which
    is why this needs naming as a single cause rather than as a recovery storm of unknown origin.

    Both worker generations are recognized, because the containment changes the reading entirely: with the
    boundary check in place the same input costs one reissued job and nothing else, so the count is a rate
    to watch rather than a fault to fix.
    """
    records = context.session.records
    empty_pops = _matching(records, _EMPTY_MODEL_POP_RE)
    blank_preloads = _matching(records, _BLANK_PRELOAD_RE)
    blank_quarantines = _matching(records, _BLANK_MODEL_QUARANTINE_RE)
    quarantine_skips = _matching(records, _BLANK_QUARANTINE_SKIP_RE)
    blank_deaths = [
        recovery
        for recovery in context.recoveries
        if (match := _LOAD_FAILURE_MODEL_RE.search(recovery.reason)) is not None and not match.group("model").strip()
    ]

    rejected = _matching(records, _MALFORMED_POP_REJECTED_RE)
    refused_preloads = _matching(records, _BLANK_PRELOAD_REFUSED_RE) + _matching(
        _child_records_in_session(context),
        _BLANK_PRELOAD_REFUSED_RE,
    )
    refused_incidents = _matching(records, _BLANK_INCIDENT_REFUSED_RE)

    uncontained = bool(blank_preloads or blank_deaths or blank_quarantines or quarantine_skips)
    if not uncontained and not (rejected or refused_preloads or refused_incidents):
        return []

    if uncontained:
        verdict = (
            f"{len(empty_pops)} pop(s) arrived with no model name and were queued as if the empty name were a "
            f"model: it was preloaded {len(blank_preloads)} time(s) and cost {len(blank_deaths)} child "
            "death(s), each one a slot ending on a load it could never complete."
        )
        if blank_quarantines or quarantine_skips:
            verdict += (
                f" The empty name then crossed the per-model incident threshold and was quarantined "
                f"{len(blank_quarantines)} time(s), after which {len(quarantine_skips)} further job(s) were "
                "refused against it. The quarantine set is holding an identity no job can ever satisfy and no "
                "load can ever clear."
            )
        remediation = (
            "Nothing about the models on this host is at fault: the worker accepted a malformed pop response "
            "and spent the pool on it. Upgrade the worker; newer versions contain and fault these at the pop "
            "boundary, handing the job straight back for reissue without preloading it, without ending a "
            "slot, and without counting it against any model. Until then the pool churn and the poisoned "
            "quarantine will recur for every such pop."
        )
        severity = Severity.CRITICAL
        evidence = (
            empty_pops[:1] + blank_preloads[:1] + blank_quarantines[:1] + quarantine_skips[:1],
            [r.record for r in blank_deaths[:1]],
        )
        evidence_lines = [_evidence(r) for r in evidence[0]] + [_evidence(r) for r in evidence[1]]
    else:
        verdict = (
            f"{len(rejected)} malformed pop(s) carrying no model name were rejected at the pop boundary and "
            "handed back for reissue, so the cascade is contained: nothing was preloaded and no slot was "
            "spent on them."
        )
        if refused_preloads:
            verdict += (
                f" A blank preload still reached a child {len(refused_preloads)} time(s), which refused it and "
                "stayed available rather than ending."
            )
        if refused_incidents:
            verdict += (
                f" {len(refused_incidents)} incident(s) reported against the blank name were refused, keeping "
                "it out of the quarantine set."
            )
        remediation = (
            "No worker-side fix is needed; the containment is doing its job. The rate is the thing to watch: "
            "each rejection is a job this worker was offered and could not serve, so a sustained rate is lost "
            "throughput and is worth raising with the horde as malformed pop responses."
        )
        severity = Severity.WARNING
        evidence_lines = [_evidence(r) for r in (rejected[:2] + refused_preloads[:1] + refused_incidents[:1])]

    return [
        Finding(
            id="empty_model_pop_cascade",
            severity=severity,
            title="Pops with no model name",
            verdict=verdict,
            remediation=remediation,
            evidence=evidence_lines,
            see_also="preload_kills_child_loop",
        ),
    ]


_PRELOAD_KILL_LOOP_THRESHOLD = 3
"""Slot deaths attributed to loading one model before it is a loop rather than a coincidence.

Matches the worker's own per-model incident threshold: at this count the worker itself concludes the model
is at fault and quarantines it, so a report that stayed quieter would be less informed than the worker."""


def detect_preload_kills_child_loop(context: SessionContext) -> list[Finding]:
    """One model that ends every slot it is preloaded onto, across the session.

    Slot deaths are ordinarily read per slot, and a model re-dispatched round-robin therefore looks like
    unrelated churn across a healthy-looking pool. Grouping the deaths by the model named in each
    replacement is what makes the pattern visible: the constant is the checkpoint, not the slot. The blank
    identity produces the same shape for a different reason and has its own finding, so it is excluded here
    rather than reported twice.
    """
    by_model: dict[str, list[RecoveryDiagnostic]] = {}
    for recovery in context.recoveries:
        match = _LOAD_FAILURE_MODEL_RE.search(recovery.reason)
        if match is None:
            continue
        model = match.group("model").strip()
        if not model:
            continue
        by_model.setdefault(model, []).append(recovery)

    looping = sorted(
        ((model, deaths) for model, deaths in by_model.items() if len(deaths) >= _PRELOAD_KILL_LOOP_THRESHOLD),
        key=lambda item: -len(item[1]),
    )
    if not looping:
        return []

    model, deaths = looping[0]
    slots = sorted({recovery.process_id for recovery in deaths})
    others = [f"{name} x{len(rest)}" for name, rest in looping[1:]]
    return [
        Finding(
            id="preload_kills_child_loop",
            severity=Severity.CRITICAL,
            title="A model ends every slot it is loaded onto",
            verdict=(
                f"Loading {model} ended the inference child {len(deaths)} time(s) across slot(s) "
                f"{_clause_join([str(slot) for slot in slots])}. The failing element is the model, not any one "
                "slot: it is re-dispatched to a fresh slot each time, so no per-slot breaker sees a pattern "
                "while the pool is rebuilt around it." + (f" Also seen for {_clause_join(others)}." if others else "")
            ),
            remediation=(
                f"Take {model} out of the served model set and re-download it; a checkpoint that ends the "
                "process during load is normally truncated or corrupt on disk, which a size or hash check "
                "against the model reference will show. If the file verifies, the load path cannot host it on "
                "this build and the model still has to come out of rotation until that is resolved. Each "
                "attempt costs a child and the jobs that child was holding."
            ),
            evidence=[_evidence(recovery.record) for recovery in deaths[:3]],
            see_also="empty_model_pop_cascade",
        ),
    ]


_POP_API_ERROR_DOMINANCE_THRESHOLD = 5
"""Identical pop rejections before the message is worth naming rather than counting as API noise."""
_POP_API_ERROR_DOMINANCE_SHARE = 0.6
"""The share of a session's pop errors one message must hold to be called the cause of the intake loss."""
# Rejections the horde will keep sending until the operator changes something. Everything else (rate
# limits, gateway and server faults) clears without intervention, and telling an operator to act on those
# sends them looking for a fault on their own machine that is not there.
_OPERATOR_FIXABLE_POP_ERRORS = re.compile(
    r"untrusted users can only have|maintenance mode|invalid api key|wrong credentials|"
    r"worker .*is not allowed|account .*suspend",
    re.IGNORECASE,
)


def detect_pop_api_error_dominance(context: SessionContext) -> list[Finding]:
    """One pop rejection repeating long enough to be the reason the worker had no work.

    A worker that is being refused work looks, from every internal signal, like a worker with nothing to
    do: the queue is empty, the governors report the intake pause they applied in response, and no stage
    ever misbehaves. The horde says why in the rejection itself, and that message is the only place the
    reason exists. Repeated verbatim it is a standing condition rather than a blip, and whether an operator
    can do anything about it is decided entirely by what it says.
    """
    errors = _matching(context.session.records, _POP_API_ERROR_RE)
    if len(errors) < _POP_API_ERROR_DOMINANCE_THRESHOLD:
        return []

    by_message: dict[str, list[LogRecord]] = {}
    for record in errors:
        match = _POP_API_ERROR_RE.search(record.message)
        if match is not None:
            by_message.setdefault(match.group("message"), []).append(record)
    if not by_message:
        return []

    message, occurrences = max(by_message.items(), key=lambda item: len(item[1]))
    if len(occurrences) < _POP_API_ERROR_DOMINANCE_THRESHOLD:
        return []
    if len(occurrences) / len(errors) < _POP_API_ERROR_DOMINANCE_SHARE:
        return []

    code_match = _POP_API_ERROR_CODE_RE.search(occurrences[0].message)
    code = code_match.group("code") if code_match is not None else None
    first, last = occurrences[0].timestamp, occurrences[-1].timestamp
    span = (last - first).total_seconds() if first is not None and last is not None else None
    span_text = f" over {span / 60:.0f} minute(s)" if span is not None and span >= 60 else ""
    operator_fixable = bool(_OPERATOR_FIXABLE_POP_ERRORS.search(message))
    others = len(by_message) - 1

    return [
        Finding(
            id="pop_api_error_dominance",
            severity=Severity.WARNING,
            title="The horde repeatedly refused this worker's pops",
            verdict=(
                f"The horde rejected {len(occurrences)} pop(s){span_text} with the same message: "
                f'"{message}"'
                + (f" (rc={code})." if code else ".")
                + " While that stands the worker is not being offered work, so an idle worker and a quiet job "
                "stream are the expected consequence rather than a local fault."
                + (f" {others} other pop error message(s) also occurred." if others > 0 else "")
            ),
            remediation=(
                (
                    "This rejection will not clear on its own: it is a condition on the account or the worker "
                    "registration, not a server fault, so the worker will keep being refused until it is "
                    "changed. Act on what the message says (worker count or naming against the account, the "
                    "API key, or a maintenance flag), then confirm pops resume."
                )
                if operator_fixable
                else (
                    "This reads as a transient server-side or rate-limit rejection, which normally clears "
                    "without intervention; the worker already backs off and retries. If it persists across "
                    "restarts, check the horde's status before changing anything locally."
                )
            ),
            evidence=[_evidence(record) for record in (occurrences[:1] + occurrences[-1:])],
            see_also="pop_governor_dominance",
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
                # A parent log rotates by size mid-run, so which archives were read decides the span every
                # figure above is measured over. Naming them keeps that attributable.
                + (
                    f" Rotation: {context.bundle.rotation_stitch.describe()}."
                    if context.bundle.rotation_stitch is not None
                    else ""
                )
            ),
            remediation="",
        ),
    ]


DETECTORS: list[Detector] = [
    detect_crash_on_start_loop,
    detect_empty_model_pop_cascade,
    detect_preload_kills_child_loop,
    detect_doomed_pool_no_giveup,
    detect_gave_up_clean,
    detect_forced_maintenance,
    detect_scheduler_starvation_wedge,
    detect_unsatisfiable_head_starvation,
    detect_slow_generation_drop_spiral,
    detect_safety_stage_stall,
    detect_whole_card_convergence_wedge,
    detect_whole_card_nonhead_residency_starvation,
    detect_whole_card_residency_churn,
    detect_whole_card_pop_claim_monopoly,
    detect_whole_card_pop_claim_episodes,
    detect_pop_liveness_full_queue,
    detect_head_dispatch_stall,
    detect_residency_reconciliation_holds,
    detect_faulted_job_census,
    detect_model_reference_sample_fault,
    detect_consecutive_failure_pause,
    detect_pop_governor_dominance,
    detect_pop_api_error_dominance,
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
