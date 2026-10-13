"""Turn a correlated session into ranked, actionable findings: what went wrong and what to do.

Each detector recognizes one incident class from the signals the worker already emits (recovery
diagnostics, the action ledger, child tracebacks) and returns a :class:`Finding` with a plain-language
verdict, the evidence that supports it, and a remediation. This is the automated form of the manual log
archeology a maintainer would otherwise do: the crash-on-start detector lifts the child's exception
across the process boundary; the doomed-pool detector recognizes the save-our-ship loop that spins
without ever giving up.

Detectors are independent and registered in :data:`DETECTORS`, so a new incident class is a
:class:`~horde_worker_regen.analysis.finding_kinds.FindingKind` and its spec, one function, and one list
entry. They never raise: a detector that cannot make sense of a session returns no findings rather than
aborting the report.

What a detector writes at the emit site is what the session itself produced: the verdict, the evidence,
the severity, and any part of the fix that depends on the measurement. The id, the catalogue title, the
invariant remediation and the default cross-reference are declared once in
:mod:`~horde_worker_regen.analysis.finding_kinds`.
"""

from __future__ import annotations

import bisect
import enum
import json
import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import yaml

from horde_worker_regen.process_management.workers.safety_process import SafetyEvaluationStage
from horde_worker_regen.utils.oom_signature import OOM_TEXT_RE

from .correlate import RecoveryDiagnostic, SessionContext, find_child_crash
from .finding_kinds import SEVERITY_ORDER, Finding, FindingKind, Severity
from .governor_signatures import GOVERNOR_ENTER_RE, GOVERNOR_EXIT_RE, GOVERNOR_PLAIN_PHRASES
from .job_lifecycle import JobLifecycleModel, LaneRole, job_lifecycle_for, percentile
from .log_ingest import LogRecord
from .log_signatures import pattern_for
from .sessions import SessionEndReason

# Signatures over orchestrator message text.
_QUARANTINE_RE = pattern_for("quarantine")
_SOFT_RESET_RE = pattern_for("soft_reset")

_WEDGE_ESCALATION_RE = re.compile(r"(Queue deadlock detected|Deadlock detected|Save-our-ship)")
"""The worker declaring the queue stopped, or its recovery ladder acting on that declaration.

A hold that self-clears is ordinary packing; a hold whose own window contains one of these lines did not
self-clear, whatever its per-hold text says, because the worker had already escalated past it."""
_POOLS_RECOVERED_RE = pattern_for("pools_recovered")
_ABANDON_SHIP_RE = pattern_for("abandon_ship")
# The live worker reclaims-and-retries on this same fingerprint; keep the signature single-sourced.
_OOM_RE = OOM_TEXT_RE
# A faulted-inference result names its model and the failing node in a stable shape:
#   "... produced no results. Model: <name>. Error: <stage> (<NodeClass>): <underlying error>"
# Both the OOM and the file-descriptor detectors read the model off this to name the culprit, because the
# outer "Pipeline failed to run ... produced no results" wrapper is identical across unrelated root causes.
_FAULT_MODEL_RE = re.compile(r"Model: (?P<model>.+?)\. Error:")
_FAULTED_ON_PROCESS_RE = pattern_for("faulted_on_process")
# The same slot-side fault, keyed on the job it names. Producer:
# ``message_dispatcher._handle_faulted_inference_result``.
_FAULTED_ON_PROCESS_JOB_RE = pattern_for("faulted_on_process_job")
# The terminal per-job fault the worker reports to the horde. Producer:
# ``job_submitter.submit_single_generation``. This is the surface the horde's own faulted counter follows,
# so it is what a census must agree with; the slot-side line above is the same job seen earlier.
_FAULT_REPORTED_RE = pattern_for("fault_reported")
# The pop line that binds a job id to the model it was popped for. Producer: ``job_popper.api_job_pop``.
_POPPED_JOB_RE = pattern_for("popped_job_model")
# The safety watchdog faulting a named job it could not check. Producer:
# ``worker_recovery_coordinator`` (the id-anchored form of :data:`_SAFETY_UNRECOVERABLE_RE`).
_SAFETY_UNRECOVERABLE_JOB_RE = pattern_for("safety_unrecoverable_job")
# The retry suffix on a slot-side fault. Producer: the same dispatcher handler, which words a bounded
# retry as "requeued for another attempt" (or a degraded, isolated one). An attempt that came back for a
# retry is not a job the horde lost, so a census that counts it over-reports what the session dropped.
_FAULT_REQUEUED_RE = pattern_for("fault_requeued_suffix")
# A disaggregated pipeline stage faulting a named job. Producer:
# ``inference_process._run_sample_stage``, which runs only on the disaggregated path, so the phrase is
# specific to it rather than to monolithic inference.
_STAGE_FAULT_RE = pattern_for("stage_fault")
# The model-reference read that failed under an in-flight sample, and the cache-staleness lines that
# attribute it. Producers: ``horde_model_reference`` (raised into the stage fault above) and the replica
# backend's ``needs_refresh`` disclosure.
_MODEL_REFERENCE_UNREADABLE_RE = pattern_for("model_reference_unreadable")
_MODEL_REFERENCE_STALE_RE = pattern_for("model_reference_stale")
# The worker's most severe liveness signal: the local queue full and motionless. Producer:
# ``process_manager._full_queue_frozen_line``, logged at ERROR on a repeat clock.
_POP_LIVENESS_FROZEN_RE = pattern_for("pop_liveness_frozen")
_POP_LIVENESS_FROZEN_FIELDS_RE = pattern_for("pop_liveness_frozen_fields")
# The whole-card residency governor's ENTER parenthetical names the model holding the card. Producer:
# ``PopGovernorRegistry`` via the scheduler's residency spell.
_RESIDENCY_GOVERNOR_MODEL_RE = pattern_for("residency_governor_model")
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
# Windows commit-charge exhaustion ("the paging file is too small"), raised while a process maps a checkpoint
# or loads a DLL. It reaches the logs inside a traceback, so it is matched against a record's full text.
_PAGEFILE_EXHAUSTED_RE = pattern_for("pagefile_exhausted")
_ACCELERATOR_UNUSABLE_RE = pattern_for("accelerator_unusable")
_TORCH_DRIVER_ASSERT_RE = pattern_for("torch_driver_assert")
_NVML_DRIVER_MISMATCH_RE = pattern_for("nvml_driver_mismatch")
_DEVICE_READING_MISSING_RE = pattern_for("device_reading_missing_defer")
_NO_IMAGES_RE = pattern_for("no_images")
# The in-progress orphan watchdog names itself in its punt line ("...(orphaned-job watchdog).") rather
# than using the words "orphaned in-progress", so the watchdog tag is the signature that actually
# matches the emitted text; the other alternatives stay for forward-compatibility and the ledger reason.
_ORPHAN_RE = pattern_for("orphaned_job_punt")
# The lane fault rules flagging an inference process for replacement: at once on a CUDA runtime error, or after a
# streak of faults across several models. Producer: ``process_lifecycle.record_inference_lane_outcome``.
_LANE_CUDA_CONTEXT_FAULT_RE = pattern_for("inference_lane_cuda_context_fault")
_LANE_FAULT_STREAK_RE = pattern_for("inference_lane_fault_streak")
_LANE_FAULT_REASON_MARKERS: tuple[str, ...] = ("(CUDA context fault)", "(fault streak:")
"""Recovery-reason fragments of a lane fault rule replacement. A plain crash or hang reads "(crashed or hung)".

Producer: ``process_lifecycle._replace_inference_process``, which wraps the rule's reason in the recovery reason."""
# Any inference slot quarantine, crash loop or crash on start. Producer:
# ``process_lifecycle._quarantine_inference_slot``.
_SLOT_QUARANTINED_RE = pattern_for("inference_slot_quarantined")
# The RAM governor pausing intake under its danger floor. The commit-bound form means the host ran out of commit.
# Producer: ``executor.execute_governance_actions``.
_RAM_DANGER_PAUSE_RE = pattern_for("ram_danger_floor_pause")
# The parent dropping its supervisor channel after it closed. Producer: ``process_manager._drop_closed_supervisor``.
_SUPERVISOR_CHANNEL_DROPPED_RE = pattern_for("supervisor_channel_dropped")
# A lane reporting a completed generation, which attributes a successful job to its process. Producer:
# ``message_dispatcher._handle_inference_result``.
_INFERENCE_FINISHED_RE = pattern_for("inference_finished")
# The CUDA runtime error torch raises for a failed CUDA call, which shares the "out of memory" wording with the
# allocator's own error but can leave the whole CUDA context unusable.
_CUDA_RUNTIME_OOM_RE = pattern_for("cuda_runtime_out_of_memory")
# The horde rejecting a pop because it forced the worker into maintenance, and the (server-supplied)
# reason it gives. "dropping too many jobs" is the worker's own fault and the actionable case; any other
# maintenance (operator-set, key issue) is informational.
_MAINTENANCE_POP_RE = pattern_for("maintenance_pop")
_DROPPING_JOBS_RE = pattern_for("dropping_jobs_reason")
# Save-our-ship faulting unservable backlog jobs (the "dropped jobs" the horde counts against the worker).
_GIVE_UP_RE = pattern_for("give_up")
# The scheduler starving: the VRAM budget deferred the head-of-queue on an idle device, with the
# starvation duration and the free VRAM that proves the budget was over-conservative.
_HEAD_STARVATION_MODEL_RE = pattern_for("head_starvation_model")
_HEAD_STARVATION_AVAILABLE_RE = pattern_for("head_starvation_available")
_DEVICE_FREE_VRAM_RE = re.compile(r"device_free_vram=(\d+)MB")
# The worker self-pausing pops after three consecutive faults.
_CONSECUTIVE_PAUSE_RE = pattern_for("consecutive_pause")
# A text backend refusing every job as busy while it generates nothing. A current worker writes one line
# per episode as it holds its text jobs; an older one faulted each job after a fixed number of busy answers,
# which shows only as busy re-offers and fault reports with no submitted text job between them.
_TEXT_BACKEND_WEDGED_RE = pattern_for("text_backend_wedged")
_TEXT_BACKEND_RELAUNCH_BOUND_RE = pattern_for("text_backend_wedge_relaunch_bound")
_TEXT_BUSY_REOFFER_RE = pattern_for("text_backend_busy_reoffer")
_TEXT_BUSY_FAULT_LEGACY_RE = pattern_for("text_backend_busy_fault_legacy")
_TEXT_JOB_FAULT_REPORTED_RE = pattern_for("text_job_fault_reported")
_TEXT_JOB_SUBMITTED_RE = pattern_for("text_job_submitted")
_TEXT_BUSY_FAULT_RUN_MIN_JOBS = 5
"""Busy-refused text jobs faulted in a row, with no text job submitted between them, that make a stuck backend.

A backend that is merely full finishes one of the jobs it holds within a generation deadline, and that
submit ends the run, so a run this long is a backend that took nothing at all."""
_TEXT_BUSY_FAULT_RUN_MIN_SPAN = timedelta(minutes=10)
"""How long such a run must last, so a burst of refusals while one long generation finishes never qualifies."""
# The horde aborting a generation server-side because the worker submitted it after the per-job deadline
# (the verbatim server message the submitter logs). Each such abort is a faulted job the horde counts
# against the worker, and a *sustained* run of them is the slow-generation death spiral that ends in
# forced maintenance, distinct from save-our-ship give-ups.
_SERVER_SLOW_ABORT_RE = pattern_for("server_slow_abort")
# The worker-side corroboration: the inference grader flagging a job running N-times its expected
# sampling time, with the residency snapshot (free VRAM) that fingerprints an over-committed device.
_SLOWDOWN_GRADE_RE = pattern_for("slowdown_grade")
# The wall-clock the safety stage took per check; a high average is the safety stage being the pipeline
# bottleneck (e.g. CPU safety with safety_on_gpu off).
_SAFETY_DURATION_RE = pattern_for("safety_check_duration")
# Safety-stage stall signals. A verdict that never returned strands a job in SAFETY_CHECKING; the worker
# now re-checks it (requeue), faults it with no image when the pipeline cannot check it (unrecoverable),
# soft-pauses pops while safety is unreliable, and throttles intake when the safety backlog is too deep.
# The dispatcher's "none was found" is the original lost-result signal that strands the job.
_SAFETY_REQUEUE_RE = pattern_for("safety_requeue")
_SAFETY_UNRECOVERABLE_RE = pattern_for("safety_unrecoverable")
_SAFETY_SOFT_PAUSE_RE = pattern_for("safety_soft_pause")
_SAFETY_BACKPRESSURE_RE = pattern_for("safety_backpressure")
_LOST_SAFETY_RESULT_RE = pattern_for("lost_safety_result")
# The scheduler explaining why a head-of-queue job is not dispatching despite pending work. The
# "no matching gate" variant is the scheduler-bug-shaped stall (model resident and idle, nothing blocking
# it, yet nothing dispatched).
_DISPATCH_STALL_RE = pattern_for("dispatch_stall")
_DISPATCH_STALL_BUG_RE = pattern_for("dispatch_stall_no_gate")
# The dispatch residency-reconciliation hold: the scheduler is holding a resident-idle head while it evicts
# an idle sibling's VRAM so the head's on-device materialisation fits the card. This is a benign, self-
# clearing swap-churn wait, not a scheduler bug, so it is excluded from the gate-less-stall detector and
# surfaced on its own as a GPU-uptime duty cost. The model and parked seconds ride the stall wrapper.
_DISPATCH_STALL_RECONCILE_RE = pattern_for("dispatch_stall_reconcile")
_DISPATCH_STALL_FIELDS_RE = pattern_for("dispatch_stall_fields")
# The whole-card residency convergence deadlock: a heavy head is pre-staged and waiting for sole residency,
# but an idle sibling holds a model that is still queued behind it, so the scale-down guard protects that
# sibling from the teardown and the residency never collapses. The head is parked until the recovery
# supervisor soft-resets the pools. This is a distinct, nameable root cause (not a generic dispatch-path bug),
# so it gets its own detector; the phrase is the worker's _diagnose_dispatch_stall attribution for it.
_WHOLE_CARD_WEDGE_RE = pattern_for("whole_card_wedge")
# A whole-card residency granted to a model that is not the head of the queue: it reserves the card and tears
# its siblings down, so the actual head (a different model) cannot load and starves. Reads as a generic
# VRAM-budget defer (the card looks idle) unless attributed to the held non-head residency, so it gets its own
# detector keyed on the worker's _diagnose_dispatch_stall phrase for it.
_WHOLE_CARD_NONHEAD_RE = pattern_for("whole_card_nonhead")
# A whole-card residency being established: the worker reserved the device for a model, tearing the process
# pool down to fewer contexts (and cycling safety off-GPU). One is routine; many in a session is reservation
# churn: the signature of models being driven onto the whole-card path that do not need it (on a high-VRAM
# card a model whose weights are a small fraction of total VRAM co-resides, so a teardown demand for it usually
# means the per-context overhead was over-counted). The phrase is the worker's establish-announce line.
_WHOLE_CARD_ESTABLISH_RE = pattern_for("whole_card_establish")
# The same line's own figures. Producer: ``inference_scheduler._establish_whole_card_residency``, whose
# f-string words the counts as "(inference processes {current} -> {after} of {max}, target {target})"; the
# residency snapshot appended after it carries ``device_free_vram=<N>MB`` (:data:`_DEVICE_FREE_VRAM_RE`).
# Reading them is what separates a reservation that reduced the pool from one whose target was already met.
_WHOLE_CARD_ESTABLISH_FIELDS_RE = pattern_for("whole_card_establish_fields")
# The establishment-time forecast the dispatch path discloses. Producer:
# ``inference_scheduler._log_stream_forecast``. The bracketed measurement block and the trailing decision
# flags have grown fields over time, so each figure is matched independently and simply stays absent in an
# older capture rather than failing the whole parse.
_STREAM_FORECAST_RE = pattern_for("stream_forecast")
_FORECAST_MARGINAL_RE = pattern_for("forecast_marginal")
_FORECAST_UNRECLAIMABLE_RE = pattern_for("forecast_unreclaimable")
_FORECAST_REDUCTION_RE = pattern_for("forecast_reduction")
# A pop that arrived with no model name, and the two worker generations' treatment of it. The older
# generation had no boundary check, so the blank identity travelled: it was preloaded as a literal empty
# name (``inference_scheduler._send_preload``), ended the slot it was sent to (the recovery reason from
# ``process_lifecycle``), was then counted and quarantined as if it were a model
# (``record_model_incident``), after which every later job for it was refused
# (``_attempt_preload_for_job``). Each of those lines carries the empty name verbatim, which is what makes
# the blank identity recognizable at all.
_EMPTY_MODEL_POP_RE = pattern_for("empty_model_pop")
_BLANK_PRELOAD_RE = pattern_for("blank_preload")
_BLANK_MODEL_QUARANTINE_RE = pattern_for("blank_model_quarantine")
_BLANK_QUARANTINE_SKIP_RE = pattern_for("blank_quarantine_skip")
# The newer generation contains the same input at the boundary and never lets it reach a slot. Producers:
# ``job_popper._reject_malformed_pop``, ``inference_process._preload_model``, and
# ``process_lifecycle.record_model_incident``. A capture carrying these instead of the cascade above shows
# the containment working, so the count is a rate rather than a fault.
_MALFORMED_POP_REJECTED_RE = pattern_for("malformed_pop_rejected")
_BLANK_PRELOAD_REFUSED_RE = pattern_for("blank_preload_refused")
_BLANK_INCIDENT_REFUSED_RE = pattern_for("blank_incident_refused")
# A slot replaced because loading a model ended it, naming the model. Producer: ``process_lifecycle``'s
# recovery reason. The name is captured (and may be empty) so repeated deaths can be grouped by model: a
# checkpoint that kills every slot it touches is a loop no per-slot breaker catches.
# Greedy to the reason's closing paren: model names carry their own parentheses ("AlbedoBase XL (SDXL)"),
# which a non-greedy match truncates at the first one.
_LOAD_FAILURE_MODEL_RE = pattern_for("load_failure_model")
# The horde rejecting a pop, quoting its own message. Producer:
# ``job_popper._handle_pop_error_response``. The message is what distinguishes an operator-fixable
# rejection (an account limit) from a transient server fault, and it is otherwise never surfaced.
_POP_API_ERROR_RE = pattern_for("pop_api_error")
_POP_API_ERROR_CODE_RE = pattern_for("pop_api_error_code")
# The parent's own safety-placement actuator, and the consequence of using it. Producers:
# ``process_lifecycle.pause_safety_on_gpu`` / ``restore_safety_on_gpu`` (which name the initiating
# subsystem rather than a fixed phrase) and ``message_dispatcher._classify_retired_launch_message``. A
# verdict dropped because its launch was retired is the mechanism by which a placement cycle strands a job,
# so the pair together is an attribution rather than a guess.
_SAFETY_PLACEMENT_CYCLE_RE = pattern_for("safety_placement_cycle")
_RETIRED_SAFETY_RESULT_RE = pattern_for("retired_safety_result")
# A safety check that found the card full and was rerun on the CPU, and the two residency commands that ran
# out of device memory. Producers: ``safety_process._evaluate_images_on_cpu``, ``demote_safety_weights`` and
# ``promote_safety_weights``, all written to the safety process's log. The parent's per-job safety line
# carries the job id and the duration of every check of the session, which the reruns are compared against.
_SAFETY_CPU_FALLBACK_RE = pattern_for("safety_check_cpu_fallback")
_SAFETY_WEIGHTS_RELEASE_OOM_RE = pattern_for("safety_weights_release_out_of_memory")
_SAFETY_WEIGHTS_PROMOTION_OOM_RE = pattern_for("safety_weights_promotion_out_of_memory")
_SAFETY_CHECKED_RE = pattern_for("safety_checked")
# A deferred GPU start that gave up on its card. Producers: ``process_lifecycle._escalate_safety_start_off_gpu``
# (safety moves to the CPU), ``_retire_stalled_inference_start`` (the inference slot leaves the card's plan)
# and ``_restore_retired_inference_slot`` (it rejoins the plan). Each gives-up line has a stalled-wait and a
# structural variant; the structural one states the card's achievable room, which is the operator's number.
_SAFETY_START_ESCALATED_RE = pattern_for("safety_start_escalated_to_cpu")
_SAFETY_START_ESCALATED_STRUCTURAL_RE = pattern_for("safety_start_escalated_structural")
_INFERENCE_START_RETIRED_RE = pattern_for("inference_start_retired")
_INFERENCE_START_RETIRED_STRUCTURAL_RE = pattern_for("inference_start_retired_structural")
_INFERENCE_SLOT_RESTORED_RE = pattern_for("inference_slot_restored")
# The text backend the worker launches, charged against its card. A structural shortfall on a card it holds
# is fixed by moving the backend, which the advice can then name. Producer:
# ``inference_scheduler._note_managed_tenant_charge``.
_MANAGED_TEXT_BACKEND_CHARGED_RE = pattern_for("managed_text_backend_charged")
_MANAGED_TEXT_BACKEND_RELEASED_RE = pattern_for("managed_text_backend_released")
# A child that died installing or repairing the shared ComfyUI environment rather than importing the
# inference stack. Concurrent cold starts clone into one environment directory, and a half-written clone
# leaves the tree in a state a later checkout refuses; the fix is the directory, never the torch install.
_GIT_ENVIRONMENT_FAILURE_RE = re.compile(
    r"GitCommandError|git clone .* failed|Untracked working tree file|unable to checkout working tree",
)
# The worker declining to reserve the card for a model whose teardown demand it does not trust (a card-light
# model on a host with no measured per-context cost). Surfaced as the positive counterpart: it confirms the
# trust gate is actively preventing reservation churn rather than the churn simply being absent.
_WHOLE_CARD_DECLINED_RE = pattern_for("whole_card_declined")
# A whole-card residency claiming the worker's pop offer: while the claim stands the worker advertises only
# the resident model, so nothing else can arrive. The claim is invisible in the job stream (it changes what
# the horde is asked for, not what the worker does with what arrives), so the edges are the only record of it.
# Both phrases are the scheduler's verbatim engage/release disclosures.
_POP_CLAIM_ENGAGED_RE = pattern_for("pop_claim_engaged")
_POP_CLAIM_RELEASED_RE = pattern_for("pop_claim_released")
_POP_CLAIM_CAP_RELEASE = "the maximum hold elapsed"
"""The release phrase for the maximum hold: the claim outlasted its window rather than ending on its own.

The other two ends are self-correcting (the demand dried up, or the residency let go), so only this one says
the claim was still narrowing the offer when the cap had to stop it."""

# The stuck-step watchdog reaping a slot whose ComfyUI generation looped on one sampling step. The slot
# kept heart-beating (so the silence watchdog stayed blind), which is exactly why this needs its own
# detector rather than folding into a generic hang. The phrase is the worker's verbatim reap line.
_STUCK_STEP_RE = pattern_for("stuck_step")

# The post-processing-stage watchdog reaping a slot that went silent. Older workers reported this as
# INFERENCE_POST_PROCESSING; dedicated-lane workers report POST_PROCESS / POST_PROCESSING. The peak is still
# an upscaler/face-fixer allocation landing after sampling, concurrent with warm inference siblings.
_POST_PROCESSING_STALL_RE = pattern_for("post_processing_stall")
_DEDICATED_POST_PROCESS_RE = pattern_for("dedicated_post_process_activity")
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
_LOW_VRAM_READOUT_RE = pattern_for("low_vram_readout")
# hordelib's throttled warning that measured free VRAM fell below the inference working-set reserve, so a
# sampling step must stream activations over the bus and run several times slower. This is the alarming
# signal (it fires only under the reserve, already rate-limited), so it counts directly.
_LOW_VRAM_RESERVE_WARN_RE = pattern_for("low_vram_reserve_warning")
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
_POST_PROCESSING_BREAKER_RE = pattern_for("post_processing_breaker_tripped")
# The parent's measured WDDM demand-paging verdict: worker child allocations demoted to system memory. This is
# the direct proof that co-resident work drove the device past its real headroom, so it corroborates a
# post-processing/inference overlap as a genuine stall rather than admitted co-residency. The phrase is the
# inference scheduler's verbatim rising-edge line (note_wddm_paging).
_WDDM_PAGING_RE = pattern_for("wddm_paging")
# The pop offer's edge lines. A current worker logs every change of the wire value with the reason it went off.
# A worker up to 18.8.1 logged only its free-VRAM headroom gate, which is the one reason it had.
_OFFER_WITHHELD_RE = pattern_for("post_processing_offer_withheld")
_OFFER_RESTORED_RE = pattern_for("post_processing_offer_restored")
_OFFER_FIRST_OFFERED_RE = pattern_for("post_processing_offer_first_offered")
_LEGACY_ADVERTISING_CLOSED_RES = (
    pattern_for("post_processing_advertising_withheld_legacy"),
    pattern_for("post_processing_advertising_holding_legacy"),
)
_LEGACY_ADVERTISING_RESTORED_RE = pattern_for("post_processing_advertising_restored_legacy")
_OFFER_WITHHELD_WARNING_FRACTION = 0.25
_OFFER_WITHHELD_CRITICAL_FRACTION = 0.75
_OFFER_WITHHELD_LONGEST_SECONDS = 900.0
_OFFER_EVIDENCE_GAP_SECONDS = 300.0
"""A silence between consecutive parent records longer than this leaves the offer's state unknown through it.

The parent prints a status block every 20 s or so, so minutes of silence mean the log was not being written: a
capture hole or a parent-loop stall. That time counts toward neither side and shortens any stretch spanning it.
"""

# A median pop->submit latency this many times the median generation time means jobs are aging somewhere
# other than generation. Which stage that is comes from the lifecycle split, never from this ratio alone.
_QUEUE_AGING_LATENCY_RATIO = 3.0

# Fractions of the fleet's card count that bound how much sampling parallelism a multi-card worker has
# lost. Averaging under half the cards is a loss worth reporting; under a third is the collapse to a
# handful of cards that leaves the rest of the fleet idle behind a full queue.
_MULTI_CARD_CONCURRENCY_WARNING_FRACTION = 0.5
_MULTI_CARD_CONCURRENCY_CRITICAL_FRACTION = 1 / 3
# How full the pending queue has to have sat, as a fraction of the worker's own intake budget, before an
# idle fleet is the scheduler's doing rather than the horde simply having no work to offer.
_MULTI_CARD_QUEUE_FULL_FRACTION = 0.75
# Below this many measurable in-flight jobs the concurrency sweep is describing a warm-up, not a session.
_MULTI_CARD_MIN_MEASURED_JOBS = 10

# Model movement per dispatch. A lane that keeps its model serves consecutive jobs of it for free, so a
# preload for every third job means the resident set turns over faster than the work does. A preload that
# is cleared before it runs is pure waste, and even a few percent of dispatches is worth naming.
_MODEL_CHURN_PRELOAD_RATIO = 1 / 3
_MODEL_CHURN_CLEARED_RATIO = 0.05
_MODEL_CHURN_MIN_DISPATCHES = 20

# An IPC silence this many times the derived stall threshold is not jitter around the threshold but the
# parent loop having stopped for a substantial fraction of a status period.
_PARENT_LOOP_STALL_CRITICAL_MULTIPLIER = 3.0
# Below this many drains the log is not carrying the parent's IPC debug stream at all, so its gaps say
# nothing about the loop.
_PARENT_LOOP_MIN_DRAINS = 20

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

# Out-of-memory faults single out one process when at least this share of them, and this many, came from it.
_OOM_CONCENTRATION_FRACTION = 0.8
_OOM_CONCENTRATION_MIN_FAULTS = 3
# The out-of-memory headline names this many models and counts the rest, so a session failing every model it
# serves still reads as one sentence.
_OOM_HEADLINE_MODELS = 3
# hordelib's line for a load that proceeded beside other resident models with no forced eviction, and the
# inference process's job start that bounds the lookback to the faulting job. The finding names this many such
# faults in its evidence and counts the rest.
_SAMPLING_BESIDE_RESIDENTS_RE = pattern_for("sampling_beside_resident_models")
_CHILD_JOB_STARTED_RE = pattern_for("inference_child_job_started")
_OOM_RESIDENT_EVIDENCE_FAULTS = 3
# A host commit exhaustion this close before a CUDA error or a dropped channel is taken as its likely cause.
_HOST_COMMIT_LOOKBACK = timedelta(seconds=60)
# A fault streak with at least this many resource faults, whose process fails again within the refault window
# after its replacement, points to card pressure on a healthy process.
_MISFIRE_MIN_RESOURCE_FAULTS = 2
_REFAULT_WINDOW = timedelta(seconds=600)


_RAM_HOLD_RE = pattern_for("host_ram_pop_hold")
_RAM_DEFER_RE = pattern_for("ram_preload_defer")
_RAM_CEILING_RE = pattern_for("ram_ceiling_reclaim")
_RAM_CREEP_RE = pattern_for("ram_creep_cycle")

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


def _count_text(count: int, singular: str, plural: str | None = None) -> str:
    """Return a count with its noun in the matching number: "1 job", "3 jobs"."""
    noun = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {noun}"


def _times_text(count: int) -> str:
    """Return how often something happened as a reader would say it: "once", "3 times"."""
    return "once" if count == 1 else f"{count} times"


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
            evidence.append(f"process {process_id}: {_evidence(crash.record).split('  ', 1)[1]} -> {crash.exception}")
            crash_texts.append(crash.record.full_text)

    cause = max(exceptions, key=lambda exc: exceptions[exc]) if exceptions else None
    git_failure = any(_GIT_ENVIRONMENT_FAILURE_RE.search(text) for text in crash_texts)
    starts = len(crashing)
    processes = _count_text(len({r.process_id for r in crashing}), "process", "processes")
    if cause is not None and git_failure:
        verdict = (
            f'{starts} image process starts across {processes} crashed with "{cause}" while '
            "preparing the shared ComfyUI environment."
        )
        remediation = (
            "Delete the environment directory shown in the error and start the worker again so one process "
            "rebuilds it. Your Python packages are not the problem. Updating hordelib prevents a repeat, since "
            "newer versions prepare that directory one process at a time and repair a half-written one."
        )
    elif cause is not None:
        verdict = f'{starts} image process starts across {processes} crashed with "{cause}".'
        remediation = (
            f'Fix the environment fault behind "{cause}". If it says torch was not compiled with CUDA, '
            "reinstall a CUDA-enabled torch."
        )
    else:
        verdict = (
            f"{starts} image process starts across {processes} crashed before they were ready, and "
            "no start-up log shows the error."
        )
        remediation = "Open the process's start-up log in the logs folder and look for the failing import or error."
    return [
        Finding(
            kind=FindingKind.CRASH_ON_START_LOOP,
            severity=Severity.CRITICAL,
            headline=verdict,
            action_addendum=remediation,
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
        f"The worker rebuilt its crashed image processes {_times_text(len(soft_resets))} and restarted a process "
        f"{_times_text(session.peak_process_recoveries)} without ever stopping itself, and the session ended as "
        f'"{session.end_reason}".'
    )
    return [
        Finding(
            kind=FindingKind.DOOMED_POOL_NO_GIVEUP,
            severity=Severity.CRITICAL,
            headline=verdict,
            evidence=[_evidence(r) for r in (soft_resets[:2] + recovered[:1] + quarantined[:1])],
        ),
    ]


def detect_gave_up_clean(context: SessionContext) -> list[Finding]:
    """The worker correctly abandoned ship on an unrecoverable pool (the healthy terminal path)."""
    abandon = _matching(context.session.records, _ABANDON_SHIP_RE)
    if not abandon:
        return []
    return [
        Finding(
            kind=FindingKind.GAVE_UP_CLEAN,
            severity=Severity.INFO,
            headline="The worker stopped itself after rebuilding its image processes could not restore a working one.",
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
            kind=FindingKind.STUCK_INFERENCE_STEP,
            severity=Severity.WARNING,
            headline=(
                "An image process repeated one sampling step without finishing and was restarted "
                f"{_times_text(len(stuck))}, losing the job it was running{' each time' if len(stuck) > 1 else ''}."
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
    # Maintenance the owner set is refused with the same line, so the horde's reason is what tells the two apart.
    forced_maintenance = any(
        _DROPPING_JOBS_RE.search(record.full_text)
        for record in _matching(context.session.records, _MAINTENANCE_POP_RE)
    )
    escalated = dropped > 0 or forced_maintenance or bool(breaker_trips)

    # Corroboration that the overlap was a genuine stall rather than admitted co-residency: a watchdog reap,
    # the parent's measured WDDM paging verdict, or a reading below the inference reserve (streaming fallback).
    below_reserve = [r for r in child_records if _LOW_VRAM_RESERVE_WARN_RE.search(r.message)]
    wddm_paging = _matching(context.session.records, _WDDM_PAGING_RE)
    corroborated = bool(stalls) or bool(wddm_paging) or bool(below_reserve)
    if not escalated and not corroborated:
        return [
            Finding(
                kind=FindingKind.POST_PROCESSING_VRAM_STALL,
                severity=Severity.INFO,
                title_override="Post-processing shared the card and nothing stalled",
                headline=(
                    f"Post-processing ran while free VRAM was low {_times_text(len(low_vram_warnings))}, and "
                    "nothing stalled."
                ),
                action_addendum="No action needed.",
                evidence=[_evidence(r) for r in (dedicated_activity[:2] + low_vram_warnings[:2])],
            ),
        ]
    outcome = ""
    if dropped > 0 or forced_maintenance:
        outcome = f", {_count_text(dropped, 'job')} {'was' if dropped == 1 else 'were'} dropped" + (
            " and the horde put the worker into maintenance" if forced_maintenance else ""
        )
    verdict = (
        f"Post-processing stalled {_times_text(len(stalls))} while free VRAM was low "
        f"{_times_text(len(low_vram_warnings))}{outcome}."
    )
    if breaker_trips:
        action = (
            "Post-processing is off for now and comes back on its own once a fault window passes without "
            "another stall. If the stalls return, turn on the VRAM budget, or lower `max_threads` or "
            "`queue_size`."
        )
    else:
        action = (
            "Turn on the VRAM budget. As a stopgap, lower `max_threads` or `queue_size`, or turn off "
            "post-processing on this card."
        )
    return [
        Finding(
            kind=FindingKind.POST_PROCESSING_VRAM_STALL,
            severity=Severity.CRITICAL if escalated else Severity.WARNING,
            headline=verdict,
            action_addendum=action,
            evidence=[_evidence(r) for r in (stalls[:4] + low_vram_warnings[:4] + breaker_trips[:1])]
            + [_evidence(r.record) for r in post_processing_recoveries[:2]],
        ),
    ]


class _OfferWithheldReason(enum.StrEnum):
    """Why a pop went out with post-processing off, worded as the finding's evidence names it."""

    LEGACY_HEADROOM_GATE = "a dip in free VRAM, on a worker at 18.8.1 or older"
    FAULT_BREAKER = "repeated post-processing faults (the fault breaker)"
    VRAM_ARBITER = "no card with room for the post-processing peak"
    LANE_PAUSED = "post-processing paused to free memory"
    MODELS_NOT_ON_DISK = "post-processing models not on disk"
    JOBS_IN_HAND = "too many post-processing jobs in hand"
    CONFIGURED_OFF = "post-processing off in the config"
    OTHER = "a reason this report does not recognise"


# Fragments of the reasons ``job_popper`` writes into the withheld edge. Matched by fragment because some carry
# a measured tail (the breaker's trip reason, the commitment limit). Time under ``CONFIGURED_OFF`` is the
# operator's choice, so it counts toward neither side of the withheld fraction.
_OFFER_REASON_FRAGMENTS: tuple[tuple[str, _OfferWithheldReason], ...] = (
    ("the fault breaker latch", _OfferWithheldReason.FAULT_BREAKER),
    ("the VRAM arbiter rules", _OfferWithheldReason.VRAM_ARBITER),
    ("lane is paused", _OfferWithheldReason.LANE_PAUSED),
    ("not on disk", _OfferWithheldReason.MODELS_NOT_ON_DISK),
    ("jobs are already in hand", _OfferWithheldReason.JOBS_IN_HAND),
    ("is off in the config", _OfferWithheldReason.CONFIGURED_OFF),
    ("does not offer post-processing", _OfferWithheldReason.CONFIGURED_OFF),
)

_OFFER_WITHHELD_ACTIONS: dict[_OfferWithheldReason, str] = {
    _OfferWithheldReason.LEGACY_HEADROOM_GATE: (
        "Update the worker. Versions up to 18.8.1 turned post-processing off on every free-VRAM dip."
    ),
    _OfferWithheldReason.FAULT_BREAKER: (
        "Repeated post-processing faults turned it off. Fix the cause given in the post-processing finding."
    ),
    _OfferWithheldReason.VRAM_ARBITER: (
        "No card had room for the post-processing peak beside the loaded models. Serve fewer models on that card."
    ),
    _OfferWithheldReason.LANE_PAUSED: (
        "The worker paused post-processing to make room for image generation. Serve fewer models on that "
        "card if this repeats."
    ),
    _OfferWithheldReason.MODELS_NOT_ON_DISK: (
        "The post-processing models were still downloading. Let the download finish, and check for download "
        "errors if it never does."
    ),
    _OfferWithheldReason.JOBS_IN_HAND: "No action needed. The worker was busy with post-processing work.",
    _OfferWithheldReason.CONFIGURED_OFF: "No action needed.",
    _OfferWithheldReason.OTHER: "Read the reason in the evidence and follow the finding it points to.",
}


@dataclass(frozen=True)
class _OfferEdge:
    """One logged change of the pop offer's post-processing value."""

    record: LogRecord
    timestamp: datetime
    withheld_reason: _OfferWithheldReason | None
    """Why the offer went off at this line, or None where it came back on."""


@dataclass(frozen=True)
class _OfferWithheldStretch:
    """One unbroken stretch with post-processing off for a counted reason."""

    start: datetime
    end: datetime
    reason: _OfferWithheldReason
    """The reason the stretch opened with; a later reason change inside it does not split it."""
    opened_by: LogRecord
    closed_by: LogRecord | None
    """The edge that turned the offer back on, or None where the stretch ran to the session end."""
    seconds: float
    """The stretch's recorded length: its span less any log silence inside it."""


@dataclass
class _OfferTimeline:
    """The offer state over a session, from its first edge to the session end."""

    offered_seconds: float = 0.0
    silent_seconds: float = 0.0
    """Log silence after the first edge, counted toward neither side."""
    withheld_seconds: dict[_OfferWithheldReason, float] = field(default_factory=dict)
    closures: dict[_OfferWithheldReason, int] = field(default_factory=dict)
    """Each transition into a counted reason, including a change of reason while already off."""
    stretches: list[_OfferWithheldStretch] = field(default_factory=list)


def _classify_offer_reason(reason: str) -> _OfferWithheldReason:
    """Map the reason a withheld edge carries to its kind."""
    for fragment, kind in _OFFER_REASON_FRAGMENTS:
        if fragment in reason:
            return kind
    return _OfferWithheldReason.OTHER


def _offer_edge(record: LogRecord) -> _OfferEdge | None:
    """Read one record as a change of the post-processing offer, current or legacy, or None."""
    if record.timestamp is None:
        return None
    withheld = _OFFER_WITHHELD_RE.search(record.message)
    if withheld is not None:
        return _OfferEdge(record, record.timestamp, _classify_offer_reason(withheld.group("reason")))
    if any(pattern.search(record.message) for pattern in _LEGACY_ADVERTISING_CLOSED_RES):
        return _OfferEdge(record, record.timestamp, _OfferWithheldReason.LEGACY_HEADROOM_GATE)
    if (
        _OFFER_RESTORED_RE.search(record.message)
        or _OFFER_FIRST_OFFERED_RE.search(record.message)
        or _LEGACY_ADVERTISING_RESTORED_RE.search(record.message)
    ):
        return _OfferEdge(record, record.timestamp, None)
    return None


def _counts_as_withheld(reason: _OfferWithheldReason | None) -> bool:
    """Whether time in this state counts toward the withheld side of the fraction."""
    return reason is not None and reason is not _OfferWithheldReason.CONFIGURED_OFF


def _log_silences(records: list[LogRecord]) -> list[tuple[datetime, datetime]]:
    """The spans between consecutive timestamped records longer than :data:`_OFFER_EVIDENCE_GAP_SECONDS`."""
    timestamps = [record.timestamp for record in records if record.timestamp is not None]
    return [
        (earlier, later)
        for earlier, later in zip(timestamps, timestamps[1:], strict=False)
        if (later - earlier).total_seconds() > _OFFER_EVIDENCE_GAP_SECONDS
    ]


def _recorded_seconds(silences: list[tuple[datetime, datetime]], start: datetime, end: datetime) -> float:
    """The length of ``[start, end]`` less the part of it the log was silent through."""
    silent = sum(
        max(0.0, (min(end, silence_end) - max(start, silence_start)).total_seconds())
        for silence_start, silence_end in silences
    )
    return max(0.0, (end - start).total_seconds() - silent)


def _offer_timeline(
    edges: list[_OfferEdge],
    session_end: datetime,
    silences: list[tuple[datetime, datetime]],
) -> _OfferTimeline:
    """Walk the edges in order and total the recorded time on each side, up to ``session_end``.

    Time before the first edge and log silences after it are unknown and not counted. A repeated edge for the
    state already in force (a legacy gate's hold line while it was already closed) changes nothing.
    """
    timeline = _OfferTimeline()
    offered: bool | None = None
    reason: _OfferWithheldReason | None = None
    since: datetime | None = None
    stretch_open: _OfferEdge | None = None

    def accrue(until: datetime) -> None:
        if since is None or until <= since:
            return
        seconds = _recorded_seconds(silences, since, until)
        timeline.silent_seconds += (until - since).total_seconds() - seconds
        if offered:
            timeline.offered_seconds += seconds
        elif _counts_as_withheld(reason) and reason is not None:
            timeline.withheld_seconds[reason] = timeline.withheld_seconds.get(reason, 0.0) + seconds

    for edge in edges:
        accrue(edge.timestamp)
        since = edge.timestamp
        was_withheld = _counts_as_withheld(reason) and not offered
        now_withheld = _counts_as_withheld(edge.withheld_reason)
        if now_withheld and edge.withheld_reason is not None and (not was_withheld or edge.withheld_reason != reason):
            timeline.closures[edge.withheld_reason] = timeline.closures.get(edge.withheld_reason, 0) + 1
        if now_withheld and not was_withheld:
            stretch_open = edge
        elif was_withheld and not now_withheld and stretch_open is not None:
            timeline.stretches.append(_offer_stretch(stretch_open, edge.timestamp, edge.record, silences))
            stretch_open = None
        offered = edge.withheld_reason is None
        reason = edge.withheld_reason
    accrue(session_end)
    if stretch_open is not None:
        stretch_end = max(session_end, stretch_open.timestamp)
        timeline.stretches.append(_offer_stretch(stretch_open, stretch_end, None, silences))
    return timeline


def _offer_stretch(
    opened: _OfferEdge,
    end: datetime,
    closed_by: LogRecord | None,
    silences: list[tuple[datetime, datetime]],
) -> _OfferWithheldStretch:
    """Build a withheld stretch from the edge that opened it, measured over its recorded time."""
    if opened.withheld_reason is None:
        raise ValueError("a withheld stretch opens on a withheld edge")
    return _OfferWithheldStretch(
        start=opened.timestamp,
        end=end,
        reason=opened.withheld_reason,
        opened_by=opened.record,
        closed_by=closed_by,
        seconds=_recorded_seconds(silences, opened.timestamp, end),
    )


def detect_post_processing_offer_withheld(context: SessionContext) -> list[Finding]:
    """Post-processing configured on but sent off on the wire for a large share of the session.

    The offer's state is rebuilt from its edge lines alone: a current worker's offer-changed line, or the
    headroom gate's advertising lines from a worker up to 18.8.1. The state is unknown until the first edge,
    and that time is left out, so a capture that begins mid-run is not read as either side. A log silence
    after it is left out the same way (:data:`_OFFER_EVIDENCE_GAP_SECONDS`). Fires on the
    withheld share or on one long stretch, since a short session can hide a long outage behind a low share.
    The advice follows the reason that held the offer off longest.
    """
    start, end = context.session.start_ts, context.session.end_ts
    records = _records_in_window(context.session.records, start, end)
    edges = [edge for edge in (_offer_edge(record) for record in records) if edge is not None]
    if not edges:
        return []
    session_end = end or max(record.timestamp for record in records if record.timestamp is not None)
    timeline = _offer_timeline(edges, session_end, _log_silences(records))
    withheld_seconds = sum(timeline.withheld_seconds.values())
    observed_seconds = withheld_seconds + timeline.offered_seconds
    if withheld_seconds <= 0 or not timeline.stretches:
        return []
    fraction = withheld_seconds / observed_seconds
    longest = max(timeline.stretches, key=lambda stretch: stretch.seconds)
    if fraction < _OFFER_WITHHELD_WARNING_FRACTION and longest.seconds < _OFFER_WITHHELD_LONGEST_SECONDS:
        return []

    dominant = max(timeline.withheld_seconds, key=lambda kind: timeline.withheld_seconds[kind])
    before_first_edge = (edges[0].timestamp - start).total_seconds() if start is not None else 0.0
    unknown_seconds = before_first_edge + timeline.silent_seconds
    closed_line = (
        f"longest stretch ended: {_evidence(longest.closed_by)}"
        if longest.closed_by is not None
        else f"longest stretch ran to the session end at {_clock_text(longest.end)}"
    )
    reason_lines = [
        f"{kind.value}: turned off {_times_text(timeline.closures.get(kind, 0))}, {seconds / 60:.0f} min in total"
        for kind, seconds in sorted(timeline.withheld_seconds.items(), key=lambda item: item[1], reverse=True)
    ]
    return [
        Finding(
            kind=FindingKind.POST_PROCESSING_OFFER_WITHHELD,
            severity=Severity.CRITICAL if fraction >= _OFFER_WITHHELD_CRITICAL_FRACTION else Severity.WARNING,
            headline=(
                f"Post-processing was off for {round(fraction * 100)} percent of the session, the longest stretch "
                f"{_span_text(longest.seconds)}."
            ),
            action_addendum=_OFFER_WITHHELD_ACTIONS[dominant],
            evidence=[
                f"Off for {withheld_seconds / 60:.0f} min and on for {timeline.offered_seconds / 60:.0f} min. "
                f"{unknown_seconds / 60:.0f} min were left out, because they came before the first logged change "
                "or while the log was silent.",
                f"longest stretch began ({longest.reason.value}): {_evidence(longest.opened_by)}",
                closed_line,
                *reason_lines,
            ],
        ),
    ]


_IDLE_FILL_ARMED_RE = pattern_for("idle_fill_armed")
_IDLE_FILL_DISARMED_RE = pattern_for("idle_fill_disarmed")
_IDLE_FILL_OFFER_RE = pattern_for("idle_fill_offer")
_ADVERTISED_MODELS_RE = pattern_for("advertised_models")
_NO_JOB_AVAILABLE_RE = pattern_for("no_job_available")
_MODELS_SKIP_RE = pattern_for("models_skip_count")

# The breaker arms after a head job waits `idle_fill_threshold_seconds` (5 by default) and disarms as soon as
# the head dispatches, so a healthy arm lasts seconds. One that outlives this many minutes kept the offer
# narrowed through many pop cycles.
_IDLE_FILL_STUCK_SECONDS = 180.0

# An offer under this share of the session's widest is a narrowed one. The ladder's rungs offer a handful of
# models out of a full catalogue; a worker serving two or three models never narrows by this much.
_NARROW_OFFER_FRACTION = 0.5


@dataclass(frozen=True)
class _NarrowOffer:
    """A stretch the pop offer stayed narrowed, read from the advertised-models lines alone."""

    opened: LogRecord
    closed: LogRecord | None
    seconds: float
    models: int
    widest: int
    models_skipped: int | None
    """The horde's `models` skip count on the last empty pop in the stretch: jobs it had that this offer excluded."""


def _longest_narrow_offer(records: list[LogRecord], session_end: datetime) -> _NarrowOffer | None:
    """The longest stretch the advertised offer stayed under half the session's widest, or None.

    The advertised line is edge-triggered on the offered set, so a stretch runs from a narrow line to the next
    wide one or the session end. Empty pops inside it carry the horde's skip counts.
    """
    offers = [(r, int(m["models"])) for r in records if (m := _ADVERTISED_MODELS_RE.search(r.message))]
    if not offers:
        return None
    widest = max(count for _, count in offers)
    stretches: list[_NarrowOffer] = []
    opened: LogRecord | None = None
    narrow_models = 0
    skipped: int | None = None

    def _close(closed: LogRecord | None) -> None:
        assert opened is not None and opened.timestamp is not None
        closed_at = closed.timestamp if closed is not None and closed.timestamp else session_end
        seconds = (closed_at - opened.timestamp).total_seconds()
        stretches.append(_NarrowOffer(opened, closed, seconds, narrow_models, widest, skipped))

    for record in records:
        if match := _ADVERTISED_MODELS_RE.search(record.message):
            count = int(match["models"])
            if count < widest * _NARROW_OFFER_FRACTION:
                if opened is None:
                    opened, narrow_models, skipped = record, count, None
            elif opened is not None:
                _close(record)
                opened = None
        elif opened is not None and (match := _NO_JOB_AVAILABLE_RE.search(record.message)):
            skip = _MODELS_SKIP_RE.search(match["reasons"])
            skipped = int(skip["count"]) if skip else 0
    if opened is not None:
        _close(None)
    return max(stretches, key=lambda s: s.seconds, default=None)


def detect_idle_fill_offer_stuck(context: SessionContext) -> list[Finding]:
    """The idle-fill breaker armed and stayed armed for minutes, narrowing the pop offer.

    An arm is closed by the next disarm line; one still open at the session end runs to it. The last ladder
    offer line inside the longest arm says how narrow the offer was. A log without arm lines (a worker that
    predates them) is read from the advertised offer instead: a narrowed offer that lasted as long while the
    horde reported jobs skipped for `models` is the same defect seen from the pop side.
    """
    start, end = context.session.start_ts, context.session.end_ts
    records = [r for r in _records_in_window(context.session.records, start, end) if r.timestamp is not None]
    if not records:
        return []
    session_end = end or records[-1].timestamp
    assert session_end is not None
    narrow = _longest_narrow_offer(records, session_end)
    narrow_evidence = (
        f"the worker asked for {narrow.models} of {narrow.widest} models for {_span_text(narrow.seconds)}"
        + (
            f", and the horde's last empty reply in that time skipped {_count_text(narrow.models_skipped, 'job')} "
            "because of the models asked for"
            if narrow.models_skipped
            else ""
        )
        if narrow is not None
        else None
    )
    arms: list[tuple[LogRecord, LogRecord | None, LogRecord | None]] = []
    armed: LogRecord | None = None
    last_offer: LogRecord | None = None
    for record in records:
        if _IDLE_FILL_ARMED_RE.search(record.message):
            if armed is None:
                armed, last_offer = record, None
        elif armed is not None and _IDLE_FILL_OFFER_RE.search(record.message):
            last_offer = record
        elif armed is not None and _IDLE_FILL_DISARMED_RE.search(record.message):
            arms.append((armed, record, last_offer))
            armed = None
    if armed is not None:
        arms.append((armed, None, last_offer))

    def _arm_seconds(arm: tuple[LogRecord, LogRecord | None, LogRecord | None]) -> float:
        closed_at = arm[1].timestamp if arm[1] is not None and arm[1].timestamp else session_end
        return (closed_at - arm[0].timestamp).total_seconds() if closed_at and arm[0].timestamp else 0.0

    stuck = [arm for arm in arms if _arm_seconds(arm) >= _IDLE_FILL_STUCK_SECONDS]
    if not stuck:
        if (
            arms
            or narrow is None
            or narrow.seconds < _IDLE_FILL_STUCK_SECONDS
            or not narrow.models_skipped
            or narrow_evidence is None
        ):
            return []
        return [
            Finding(
                kind=FindingKind.IDLE_FILL_OFFER_STUCK,
                severity=Severity.CRITICAL,
                headline=(
                    f"The worker asked for {narrow.models} of {narrow.widest} models for {_span_text(narrow.seconds)}."
                ),
                evidence=[
                    f"fewer models asked for from: {_evidence(narrow.opened)}",
                    f"full list asked for again from: {_evidence(narrow.closed)}"
                    if narrow.closed is not None
                    else "still asking for fewer models at the session end",
                    narrow_evidence,
                    "This log has no idle-fill breaker lines, so the finding is read from the model counts in the "
                    "worker's job requests.",
                ],
            ),
        ]
    longest = max(stuck, key=_arm_seconds)
    opened, closed, offer = longest
    offer_match = _IDLE_FILL_OFFER_RE.search(offer.message) if offer is not None else None
    evidence = [
        f"idle-fill breaker turned on: {_evidence(opened)}",
        f"turned off: {_evidence(closed)}" if closed is not None else "still on at the session end",
        f"last narrowed request while on: {_evidence(offer)}"
        if offer is not None
        else "no narrowed request was logged while it was on",
        f"The breaker turned on {_times_text(len(arms))} and stayed on for "
        f"{_IDLE_FILL_STUCK_SECONDS / 60:.0f} minutes or more {_times_text(len(stuck))}.",
        *([narrow_evidence] if narrow_evidence is not None else []),
    ]
    models_text = (
        f" with {_count_text(int(offer_match['models']), 'model')} offered" if offer_match is not None else ""
    )
    return [
        Finding(
            kind=FindingKind.IDLE_FILL_OFFER_STUCK,
            severity=Severity.CRITICAL,
            headline=(f"The worker asked for fewer models for {_span_text(_arm_seconds(longest))}{models_text}."),
            evidence=evidence,
        ),
    ]


_LANE_STOPPED_RE = pattern_for("service_lane_stopped")
_LANE_RESTARTED_RE = pattern_for("service_lane_restarted")
_LANE_HOLD_RE = pattern_for("reclaim_lane_hold")

# A lane cold start costs 6 to 27 seconds. A restart within this many seconds of its stop is typically a lane
# stopped for one post-processing job that did not fit beside it, which ran in a few seconds. Three such pairs on
# one lane is a pattern; one or two can be a single pressure spike.
_LANE_QUICK_RESTART_SECONDS = 5.0
_LANE_CHURN_QUICK_PAIRS = 3


def detect_service_lane_restart_churn(context: SessionContext) -> list[Finding]:
    """A service lane stopped by the reclaim ladder and restarted within seconds, repeatedly.

    Pairs are the reclaim ladder's stop line and the next restart of the same lane. A hold line for that lane
    between them means the lane was kept down as designed.
    """
    records = _records_in_window(context.session.records, context.session.start_ts, context.session.end_ts)
    open_stop: dict[str, LogRecord] = {}
    held: dict[str, bool] = {}
    pairs: dict[str, list[tuple[float, bool]]] = {}
    for record in records:
        if record.timestamp is None:
            continue
        if (match := _LANE_STOPPED_RE.search(record.message)) and match["owner"] == "Reclaim ladder":
            open_stop[match["lane"]] = record
            held[match["lane"]] = False
        elif match := _LANE_HOLD_RE.search(record.message):
            held[match["lane"]] = True
        elif (match := _LANE_RESTARTED_RE.search(record.message)) and match["lane"] in open_stop:
            stopped = open_stop.pop(match["lane"])
            seconds = (record.timestamp - stopped.timestamp).total_seconds() if stopped.timestamp else 0.0
            pairs.setdefault(match["lane"], []).append((seconds, held.get(match["lane"], False)))
    churning = {
        lane: lane_pairs
        for lane, lane_pairs in pairs.items()
        if sum(1 for seconds, _ in lane_pairs if seconds < _LANE_QUICK_RESTART_SECONDS) >= _LANE_CHURN_QUICK_PAIRS
    }
    if not churning:
        return []
    evidence = []
    for lane, lane_pairs in sorted(churning.items()):
        quick = sum(1 for seconds, _ in lane_pairs if seconds < _LANE_QUICK_RESTART_SECONDS)
        median = _median([seconds for seconds, _ in lane_pairs]) or 0.0
        kept = sum(1 for _, was_held in lane_pairs if was_held)
        evidence.append(
            f"{lane}: stopped and restarted {_times_text(len(lane_pairs))}, {quick} of them within "
            f"{_LANE_QUICK_RESTART_SECONDS:.0f} s, median {median:.1f} s from stop to restart, {kept} kept down "
            "until a job finished sampling",
        )
    worst = max(churning, key=lambda lane: len(churning[lane]))
    worst_quick = sum(1 for seconds, _ in churning[worst] if seconds < _LANE_QUICK_RESTART_SECONDS)
    return [
        Finding(
            kind=FindingKind.SERVICE_LANE_RESTART_CHURN,
            severity=Severity.WARNING,
            headline=(
                f"The {worst} process was stopped and restarted within {_LANE_QUICK_RESTART_SECONDS:.0f} seconds "
                f"{worst_quick} times."
            ),
            evidence=evidence,
        ),
    ]


_DOUBLE_DISPATCH_SIGNATURES = ("inference_result_lost", "inference_result_job_missing")
_DOUBLE_DISPATCH_EVIDENCE_LABELS: dict[str, str] = {
    "inference_result_lost": 'results lost while their job was still in progress ("its inference result was lost")',
    "inference_result_job_missing": 'results for a job the worker no longer tracked ("not found in jobs_in_progress")',
}
"""What each signature counts, in a reader's words, with the log phrase to search for."""
_OWNERSHIP_RELEASED_RE = pattern_for("ended_job_ownership_released")


def detect_lane_double_dispatch_signature(context: SessionContext) -> list[Finding]:
    """Jobs released as lost while a lane held them, or results for jobs no longer tracked.

    The ownership-release count is context only: a release on its own is the designed cleanup. Sampling
    without a recorded grant is not an ownership defect; :func:`detect_unpriced_sampling_windows` ranks it.
    """
    records = _records_in_window(context.session.records, context.session.start_ts, context.session.end_ts)
    patterns = {name: pattern_for(name) for name in _DOUBLE_DISPATCH_SIGNATURES}
    hits: dict[str, list[LogRecord]] = {name: [] for name in _DOUBLE_DISPATCH_SIGNATURES}
    released = 0
    for record in records:
        if _OWNERSHIP_RELEASED_RE.search(record.message):
            released += 1
            continue
        for name, pattern in patterns.items():
            if pattern.search(record.message):
                hits[name].append(record)
                break
    total = sum(len(found) for found in hits.values())
    if total == 0:
        return []
    first = min((found[0] for found in hits.values() if found), key=lambda record: record.timestamp or datetime.min)
    return [
        Finding(
            kind=FindingKind.LANE_DOUBLE_DISPATCH_SIGNATURE,
            severity=Severity.CRITICAL,
            headline=f"The worker lost track of which process owned a job {_times_text(total)}.",
            evidence=[
                *(f"{_DOUBLE_DISPATCH_EVIDENCE_LABELS[name]}: {len(found)}" for name, found in hits.items()),
                f'jobs released after ending with no result, which is normal cleanup ("Released process"): {released}',
                f"first: {_evidence(first)}",
            ],
        ),
    ]


_INFERENCE_DISPATCHED_RE = pattern_for("inference_dispatched")
_UNPRICED_SAMPLING_WINDOW_RE = pattern_for("unpriced_sampling_window")

# A wait this short from the start of inference means the child never waited for a grant.
_UNPRICED_SHORT_WAIT_SECONDS = 3.0
# A wait this long means the child sat through most of its lease-acquire timeout (60 s) with the card idle.
_UNPRICED_LONG_WAIT_SECONDS = 30.0
# One lease timeout can be a transient; this many in a session is a pricing or reclaim pattern worth a look.
_UNPRICED_LONG_WAIT_WARNING_COUNT = 3


def detect_unpriced_sampling_windows(context: SessionContext) -> list[Finding]:
    """Sampling without a recorded clearance grant, banded by the wait since that process's start of inference.

    Each window pairs with the most recent start of inference on the same process. A short wait is a child
    that never waited for a grant; a wait of 30 s or more is a child that ran out its lease-acquire timeout with
    the card idle, which is what costs throughput. A window with no prior start on its process is counted
    but not banded.
    """
    records = _records_in_window(context.session.records, context.session.start_ts, context.session.end_ts)
    last_start: dict[str, datetime] = {}
    short_windows: list[LogRecord] = []
    middle_windows: list[LogRecord] = []
    long_windows: list[LogRecord] = []
    waits: list[float] = []
    unpaired = 0
    for record in records:
        if record.timestamp is None:
            continue
        if match := _INFERENCE_DISPATCHED_RE.search(record.message):
            last_start[match["process"]] = record.timestamp
            continue
        if not (match := _UNPRICED_SAMPLING_WINDOW_RE.search(record.message)):
            continue
        started = last_start.get(match["process"])
        if started is None:
            unpaired += 1
            continue
        wait = (record.timestamp - started).total_seconds()
        waits.append(wait)
        if wait < _UNPRICED_SHORT_WAIT_SECONDS:
            short_windows.append(record)
        elif wait < _UNPRICED_LONG_WAIT_SECONDS:
            middle_windows.append(record)
        else:
            long_windows.append(record)
    total = len(waits) + unpaired
    if total == 0:
        return []
    severity = Severity.WARNING if len(long_windows) >= _UNPRICED_LONG_WAIT_WARNING_COUNT else Severity.INFO
    if total == 1:
        headline = "Image generation started without a recorded memory grant once, " + (
            f"after waiting {_UNPRICED_LONG_WAIT_SECONDS:.0f} seconds or more."
            if long_windows
            else f"after a wait under {_UNPRICED_LONG_WAIT_SECONDS:.0f} seconds."
        )
    elif long_windows:
        headline = (
            f"Image generation started without a recorded memory grant {total} times, {len(long_windows)} of them "
            f"after waiting {_UNPRICED_LONG_WAIT_SECONDS:.0f} seconds or more."
        )
    else:
        headline = (
            f"Image generation started without a recorded memory grant {total} times, each after a wait under "
            f"{_UNPRICED_LONG_WAIT_SECONDS:.0f} seconds."
        )
    if severity is Severity.WARNING:
        action = (
            'Read the "Clearance held for process" line before each long wait. '
            "It says why the worker held back the memory grant."
        )
    else:
        action = "No action needed."
    evidence = [
        f"waits: {len(short_windows)} under {_UNPRICED_SHORT_WAIT_SECONDS:.0f} s, {len(middle_windows)} from "
        f"{_UNPRICED_SHORT_WAIT_SECONDS:.0f} s to under {_UNPRICED_LONG_WAIT_SECONDS:.0f} s, {len(long_windows)} at "
        f"{_UNPRICED_LONG_WAIT_SECONDS:.0f} s or more",
    ]
    if (median := _median(waits)) is not None:
        evidence.append(f"median wait: {median:.1f} s")
    if unpaired:
        evidence.append(f"not timed, because the log has no job start on the same process beforehand: {unpaired}")
    costliest_bands = (
        (long_windows, f"{_UNPRICED_LONG_WAIT_SECONDS:.0f} s or more", "at"),
        (middle_windows, f"{_UNPRICED_SHORT_WAIT_SECONDS:.0f} s to under {_UNPRICED_LONG_WAIT_SECONDS:.0f} s", "from"),
        (short_windows, f"{_UNPRICED_SHORT_WAIT_SECONDS:.0f} s", "under"),
    )
    for band, label, preposition in costliest_bands:
        if band:
            evidence.append(f"first {preposition} {label}: {_evidence(band[0])}")
            break
    return [
        Finding(
            kind=FindingKind.UNPRICED_SAMPLING_WINDOWS,
            severity=severity,
            headline=headline,
            action_addendum=action,
            evidence=evidence,
        ),
    ]


_CLEARANCE_STAGE_GRANTED_RE = pattern_for("clearance_stage_granted")
_CLEARANCE_SAMPLE_GRANTED_RE = pattern_for("clearance_sample_granted")
_CLEARANCE_STAGE_REFUSED_RE = pattern_for("clearance_stage_refused")
_CLEARANCE_STAGE_HOLD_RE = pattern_for("clearance_stage_hold")

# Above this share of jobs refused an early load, "most" were refused, which on a large card points at the price.
_STAGE_REFUSAL_MOST_SHARE = 0.5


def detect_stage_outlived_sample_grant(context: SessionContext) -> list[Finding]:
    """Early weight loads whose sample grant did not follow within the child's lease-acquire timeout.

    A stage grant opens a window on its process. The next sample grant or stage grant on that process closes it,
    as do a denoise loop entered without a recorded grant and the end of the log. The window outlived its sample
    grant when it closed on an ungranted denoise loop, or when no sample grant came within the timeout. The
    controller moves a staged child it finds sampling to sampling without a line of its own, so a missing sample
    grant is the usual evidence.
    """
    from horde_worker_regen.process_management.scheduling.clearance_lease import (
        CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS,
    )

    records = _records_in_window(context.session.records, context.session.start_ts, context.session.end_ts)
    timeout = timedelta(seconds=CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS)
    open_stages: dict[str, LogRecord] = {}
    late: list[LogRecord] = []
    ungranted: list[LogRecord] = []
    last_seen: datetime | None = None
    for record in records:
        if record.timestamp is None:
            continue
        last_seen = record.timestamp
        if match := _CLEARANCE_STAGE_GRANTED_RE.search(record.message):
            previous = open_stages.get(match["process"])
            if (
                previous is not None
                and previous.timestamp is not None
                and record.timestamp - previous.timestamp > timeout
            ):
                late.append(previous)
            open_stages[match["process"]] = record
        elif match := _CLEARANCE_SAMPLE_GRANTED_RE.search(record.message):
            staged = open_stages.pop(match["process"], None)
            if staged is not None and staged.timestamp is not None and record.timestamp - staged.timestamp > timeout:
                late.append(staged)
        elif match := _UNPRICED_SAMPLING_WINDOW_RE.search(record.message):
            staged = open_stages.pop(match["process"], None)
            if staged is not None:
                ungranted.append(staged)
    end = context.session.end_ts or last_seen
    unresolved = [
        staged
        for staged in open_stages.values()
        if end is not None and staged.timestamp is not None and end - staged.timestamp > timeout
    ]
    outlived = [*late, *ungranted, *unresolved]
    if not outlived:
        return []
    first = min(outlived, key=lambda record: record.timestamp or datetime.min)
    return [
        Finding(
            kind=FindingKind.STAGE_OUTLIVED_SAMPLE_GRANT,
            severity=Severity.CRITICAL,
            headline=(
                "A job loaded its model early but got no grant to start generating within "
                f"{CLEARANCE_LEASE_ACQUIRE_TIMEOUT_SECONDS:.0f} seconds, {_times_text(len(outlived))}."
            ),
            evidence=[
                f"the start grant came after the limit: {len(late)}",
                'generation began with no grant on record ("entered its denoise loop without a recorded grant"): '
                f"{len(ungranted)}",
                f"still waiting for the start grant when the log ends: {len(unresolved)}",
                f"first early load: {_evidence(first)}",
            ],
        ),
    ]


def detect_stage_refusals(context: SessionContext) -> list[Finding]:
    """Jobs refused an early weight load, as a share of the jobs that asked for one, with the commonest clause.

    Counted per job, not per check: the controller asks again for the waiting job's stage every step, so the
    duty line's running tally counts one job's wait many times. A job's span on its process ends at its sample
    grant or at a denoise loop entered without one. The job was staged when a stage grant fell in the span, and
    refused when only refusals did. The clause is the disposition the scheduler's stage hold line names.
    """
    records = _records_in_window(context.session.records, context.session.start_ts, context.session.end_ts)
    staged: set[str] = set()
    first_refusal: dict[str, LogRecord] = {}
    staged_jobs = 0
    refused_jobs: list[LogRecord] = []
    clause_counts: dict[str, int] = {}
    for record in records:
        if match := _CLEARANCE_STAGE_GRANTED_RE.search(record.message):
            staged.add(match["process"])
        elif (match := _CLEARANCE_STAGE_REFUSED_RE.search(record.message)) or (
            match := _CLEARANCE_STAGE_HOLD_RE.search(record.message)
        ):
            first_refusal.setdefault(match["process"], record)
            if clause := match.groupdict().get("clause"):
                clause_counts[clause] = clause_counts.get(clause, 0) + 1
        elif (match := _CLEARANCE_SAMPLE_GRANTED_RE.search(record.message)) or (
            match := _UNPRICED_SAMPLING_WINDOW_RE.search(record.message)
        ):
            process = match["process"]
            if process in staged:
                staged_jobs += 1
            elif process in first_refusal:
                refused_jobs.append(first_refusal[process])
            staged.discard(process)
            first_refusal.pop(process, None)
    staged_jobs += len(staged)
    refused_jobs.extend(record for process, record in first_refusal.items() if process not in staged)
    if not refused_jobs:
        return []
    total = staged_jobs + len(refused_jobs)
    share = len(refused_jobs) / total
    if share > _STAGE_REFUSAL_MOST_SHARE:
        action = "On a card under 20 GB this is expected. On a larger card, run `horde-log bundle` and report it."
    else:
        action = "No action needed."
    evidence = [
        f"jobs that loaded early: {staged_jobs}",
        f"jobs refused an early load: {len(refused_jobs)}, {share:.0%} of those that asked",
    ]
    if clause_counts:
        clause, count = max(clause_counts.items(), key=lambda entry: (entry[1], entry[0]))
        evidence.append(
            f'most common reason on the stage hold lines: "{clause}", {count} of {sum(clause_counts.values())}'
        )
    else:
        evidence.append("no stage hold line names a reason")
    first = min(refused_jobs, key=lambda record: record.timestamp or datetime.min)
    evidence.append(f"first refusal: {_evidence(first)}")
    return [
        Finding(
            kind=FindingKind.STAGE_REFUSALS,
            severity=Severity.INFO,
            headline=(
                f"Early model loads were refused for {len(refused_jobs)} of {total} jobs that waited behind a "
                "running job."
            ),
            action_addendum=action,
            evidence=evidence,
        ),
    ]


# Residency-reconciliation holds are benign at low volume; sustained, they are a real swap-churn duty cost.
# A rate past this many holds per hour, or a cumulative parked share past this fraction of the session, is
# worth a warning rather than the informational default.
_RECONCILE_HOLD_RATE_WARNING_PER_HOUR = 30.0
_RECONCILE_HOLD_PARKED_FRACTION_WARNING = 0.05


_PP_DEFER_RE = pattern_for("post_processing_deferred")
_PP_FINISHED_RE = pattern_for("post_processing_finished")

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
        f'Job "{worst_job}" waited for post-processing room {len(worst)} times and '
        f"{_count_text(len(defer_records), 'job')} waited in total"
        + (", and no post-processing finished after the waits began." if lane_fully_starved else ".")
    )
    return [
        Finding(
            kind=FindingKind.POST_PROCESSING_DEFERRAL_STARVATION,
            severity=Severity.CRITICAL if starved and lane_fully_starved else Severity.WARNING,
            headline=verdict,
            evidence=[_evidence(r) for r in (worst[:2] + worst[-2:])],
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


@dataclass(frozen=True)
class _BrokenProcessVerdict:
    """Out-of-memory faults concentrated on one process while other processes kept finishing jobs."""

    process_id: int
    faults: int
    own_finishes: int
    """Jobs the faulting process itself finished between its first and last fault."""
    other_finishes: dict[int, int]
    """Jobs each other inference process finished in that same span, by process id."""


def _broken_process_verdict(records: list[LogRecord], faults: list[LogRecord]) -> _BrokenProcessVerdict | None:
    """Return the process the out-of-memory faults single out, when the rest of the card kept working.

    A card that is genuinely full fails jobs on whichever processes run on it. A process whose CUDA context is
    broken fails every job while its siblings succeed beside it, so concentration alone is not the signal: the
    siblings' finished jobs over the same span are what rule out a full card.
    """
    by_process: dict[int, list[LogRecord]] = {}
    for record in faults:
        if (match := _FAULTED_ON_PROCESS_RE.search(record.message)) is not None:
            by_process.setdefault(int(match.group("pid")), []).append(record)
    attributed = sum(len(process_faults) for process_faults in by_process.values())
    if attributed < _OOM_CONCENTRATION_MIN_FAULTS:
        return None
    process_id, process_faults = max(by_process.items(), key=lambda item: len(item[1]))
    if len(process_faults) / attributed < _OOM_CONCENTRATION_FRACTION:
        return None
    stamped = [record.timestamp for record in process_faults if record.timestamp is not None]
    if not stamped:
        return None
    first, last = min(stamped), max(stamped)
    finishes: dict[int, int] = {}
    for record in _records_in_window(records, first, last):
        if (finished := _INFERENCE_FINISHED_RE.search(record.message)) is not None:
            finisher = int(finished.group("process"))
            finishes[finisher] = finishes.get(finisher, 0) + 1
    own_finishes = finishes.pop(process_id, 0)
    if not finishes:
        return None
    return _BrokenProcessVerdict(
        process_id=process_id,
        faults=len(process_faults),
        own_finishes=own_finishes,
        other_finishes=finishes,
    )


@dataclass(frozen=True)
class _ResidentSamplingFault:
    """An out-of-memory fault whose job loaded its model beside other models hordelib did not evict."""

    job_id: str
    process_id: int
    resident_line: LogRecord
    resident_mb: int
    resident_names: str


def _resident_sampling_fault(context: SessionContext, fault: LogRecord) -> _ResidentSamplingFault | None:
    """Return the fault's most recent resident-models line from its own job, or None when its job logged none.

    The lane log is walked back from the fault to the most recent job start. That start has to name the faulting
    job, since a start for another job means the fault's own start is missing (a head-truncated lane log). A
    resident-models line from an earlier job on the same lane does not explain this fault.
    """
    job_match = _FAULTED_ON_PROCESS_JOB_RE.search(fault.message)
    if job_match is None or fault.timestamp is None:
        return None
    job_id = job_match["job"]
    process_id = int(job_match["pid"])
    lane_records = _records_in_window(context.bundle.child_records(process_id), None, fault.timestamp)
    latest: tuple[LogRecord, re.Match[str]] | None = None
    for record in reversed(lane_records):
        if (started := _CHILD_JOB_STARTED_RE.search(record.message)) is not None:
            if job_id not in started["jobs"] or latest is None:
                return None
            resident_line, resident = latest
            return _ResidentSamplingFault(
                job_id=job_id,
                process_id=process_id,
                resident_line=resident_line,
                resident_mb=int(resident["resident_mb"]),
                resident_names=resident["resident"],
            )
        if latest is None and (resident_match := _SAMPLING_BESIDE_RESIDENTS_RE.search(record.message)):
            latest = (record, resident_match)
    return None


def _resident_sampling_evidence(resident_faults: list[_ResidentSamplingFault], fault_count: int) -> list[str]:
    """Return the count of faults that sampled beside resident models and up to three of them by job."""
    lines = [
        f"{len(resident_faults)} of {_count_text(fault_count, 'fault')} loaded the model beside other models "
        'kept on the card ("Sampling beside resident models")',
    ]
    lines.extend(
        f"job {resident.job_id[:8]} on process {resident.process_id}, {resident.resident_mb} MB resident "
        f"({resident.resident_names}): {_evidence(resident.resident_line)}"
        for resident in resident_faults[:_OOM_RESIDENT_EVIDENCE_FAULTS]
    )
    return lines


def detect_oom(context: SessionContext) -> list[Finding]:
    """Out-of-memory faults (explicit CUDA OOM), naming the faulting model and the card's co-residency.

    The bare fault count does not tell a maintainer whether one model is simply too large for the card or
    many models were over-admitted onto it. The allocator's own message carries both: the model that
    faulted, and the sibling processes holding memory with almost nothing free. Naming them turns a plain
    count into a finding that identifies the faulting model and whether the card was over-committed (many
    processes co-resident with near-zero free VRAM), which points at the fix.

    The same wording also comes from a process whose CUDA context broke, which fails every job it runs while
    the card has room. When the faults single one process out and its siblings kept finishing jobs, the
    finding names that process instead of advising less work on the card. A CUDA runtime error
    (``torch.AcceleratorError``) is named as a failed CUDA call.

    hordelib forces no eviction before a load it does not force-load in full, and logs the models it left
    resident. A fault whose own job logged that line is counted and named in the evidence, so a fault caused
    by the models left in place is attributable from a bundle.
    """
    oom = _matching(context.session.records, _OOM_RE)
    if not oom:
        return []

    # The dispatcher logs each fault twice, once as the drained child message and once as the per-job fault line.
    # Counting the per-job lines alone counts each fault once. A log with no per-job line falls back to every match.
    faults = [record for record in oom if _FAULTED_ON_PROCESS_RE.search(record.message)] or oom
    runtime_errors = sum(1 for record in faults if _CUDA_RUNTIME_OOM_RE.search(record.message))
    broken = _broken_process_verdict(context.session.records, faults)
    resident_faults = [
        resident for record in faults if (resident := _resident_sampling_fault(context, record)) is not None
    ]
    resident_evidence = _resident_sampling_evidence(resident_faults, len(faults)) if resident_faults else []

    models = _faulting_models(faults)
    model_names = [f'"{model}"' for model in models[:_OOM_HEADLINE_MODELS]]
    if len(models) > _OOM_HEADLINE_MODELS:
        model_names.append(_count_text(len(models) - _OOM_HEADLINE_MODELS, "more model"))

    if broken is not None:
        others = sorted(broken.other_finishes)
        finished = sum(broken.other_finishes.values())
        noun = "process" if len(others) == 1 else "processes"
        verdict = (
            f"Process {broken.process_id} had {broken.faults} of {len(faults)} out-of-memory faults while "
            f"{noun} {_clause_join([str(other) for other in others])} finished {_count_text(finished, 'job')} in "
            "the same time."
        )
        if runtime_errors:
            addendum = (
                '"CUDA error: out of memory" comes from a failed CUDA call and can leave a process failing every '
                "job. Report it with `horde-log bundle`, and restart an older worker, which "
                "cannot replace the process by itself."
            )
        else:
            addendum = (
                "Here one process failed while others worked, so that process was broken. Report it with "
                "`horde-log bundle`, and restart an older worker, which cannot replace the process by itself."
            )
        evidence = [
            f"process {broken.process_id} finished {_count_text(broken.own_finishes, 'job')} between its first and "
            "last fault",
            *(
                f"process {other} finished {_count_text(count, 'job')} in the same time"
                for other, count in sorted(broken.other_finishes.items())
            ),
            *(_evidence(record) for record in faults[:4]),
            *resident_evidence,
        ]
        return [
            Finding(
                kind=FindingKind.OOM,
                severity=Severity.CRITICAL,
                headline=verdict,
                action_addendum=addendum,
                evidence=evidence,
            ),
        ]

    slots = _affected_slots(faults)
    free_vrams = [float(m.group(1)) for r in faults if (m := _OOM_FREE_VRAM_RE.search(r.message))]
    sibling_counts = [len(_OOM_SIBLING_RE.findall(r.message)) for r in faults]
    max_siblings = max(sibling_counts, default=0)

    verdict = _count_text(len(faults), "out-of-memory fault")
    if model_names:
        verdict += f" on {_clause_join(model_names)}"
    if slots:
        verdict += f" in {'process' if len(slots) == 1 else 'processes'} {_clause_join([str(s) for s in slots])}"
    if free_vrams:
        verdict += f", with as little as {min(free_vrams):.0f} MiB free"
        if max_siblings:
            # A sibling count here is (co-resident processes) - 1 (the message excludes the faulting
            # process's own line), so +1 to state the total sharing the card.
            verdict += f" and {max_siblings + 1} processes sharing the card"
    verdict += "."

    if resident_faults:
        # The copy guide caps "Do this" at three sentences, so this sentence replaces the general verdict. Models
        # left loaded beside the faulting one are the more specific explanation.
        likelier = (
            "Some faults came while other models stayed loaded on the card, so send the log with "
            "`horde-log bundle` to the worker maintainers."
        )
    elif runtime_errors:
        likelier = (
            'The evidence favours a full card, though a "CUDA error: out of memory" can also mean a broken process.'
        )
    else:
        likelier = "The evidence favours too much work on the card over a broken process."
    return [
        Finding(
            kind=FindingKind.OOM,
            severity=Severity.CRITICAL,
            headline=verdict,
            action_addendum=(
                f"{likelier} Lower `max_threads` or `queue_size`, or turn on the VRAM budget, and run fewer models "
                "when several share a full card."
            ),
            evidence=[*(_evidence(r) for r in faults[:4]), *resident_evidence],
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
    window_clause = f" over {window_minutes:.0f} minutes" if window_minutes >= 1 else ""

    # A recovery that replaced the affected slot after the leak (the "failed to load model" path), which is
    # the only thing that clears the poisoned process; naming it confirms the slot was eventually recycled.
    recovery = next(
        (r for r in context.recoveries if (not slots or r.process_id in slots) and "failed to load model" in r.reason),
        None,
    )

    slot_noun = "process" if len(slots) == 1 else "processes"
    slot_clause = f" in {slot_noun} {_clause_join([str(s) for s in slots])}" if slots else ""
    model_clause = f" while serving {_clause_join([f'"{m}"' for m in models])}" if models else ""
    resource_clause = (
        f"The refused opens include {_clause_join([f'"{path}"' for path in resources[:4]])}. " if resources else ""
    )
    recovery_clause = (
        f'The worker replaced the process, with the reason "{recovery.reason}".'
        if recovery is not None
        else "No replacement was recorded, so the process may have stayed broken until the session ended."
    )

    return [
        Finding(
            kind=FindingKind.FILE_DESCRIPTOR_EXHAUSTION,
            severity=Severity.CRITICAL,
            headline=(
                f"A process ran out of file handles {_times_text(len(faults))}{slot_clause}{model_clause}"
                f"{window_clause}."
            ),
            action_addendum=recovery_clause,
            evidence=([resource_clause.strip()] if resource_clause else [])
            + [_evidence(r) for r in faults[:4]]
            + ([_evidence(recovery.record)] if recovery else []),
        ),
    ]


def detect_pagefile_exhaustion(context: SessionContext) -> list[Finding]:
    """Worker processes refused memory by Windows because the commit limit was reached (error 1455).

    The failure lands in whichever process allocates next: a child still starting (its startup-crash
    backstop holds the traceback), a child whose log is open (its loguru catch), or the parent. The parent
    sees only a child that never reports ready, so without this finding the reader is told a process was
    stuck starting and never why. One crash can appear in both a child's loop log and its startup log, so
    matches are counted once per process per second.
    """
    start, end = context.session.start_ts, context.session.end_ts
    matches: list[tuple[int | None, LogRecord]] = [
        (None, record) for record in context.session.records if _PAGEFILE_EXHAUSTED_RE.search(record.full_text)
    ]
    for process_id in sorted(context.bundle.process_ids()):
        child_records = _records_in_window(context.bundle.child_records(process_id), start, end)
        startup_records = _records_in_window(context.bundle.startup_records(process_id), start, end)
        matches.extend(
            (process_id, record)
            for record in [*child_records, *startup_records]
            if _PAGEFILE_EXHAUSTED_RE.search(record.full_text)
        )
    if not matches:
        return []

    distinct: dict[tuple[int | None, datetime | None], LogRecord] = {}
    for process_id, record in matches:
        second = record.timestamp.replace(microsecond=0) if record.timestamp is not None else None
        distinct.setdefault((process_id, second), record)
    records = sorted(distinct.values(), key=_window_key)
    count = len(records)
    times = "once" if count == 1 else f"{count} times"
    return [
        Finding(
            kind=FindingKind.PAGEFILE_EXHAUSTION,
            severity=Severity.CRITICAL,
            headline=f'Worker processes failed {times} because Windows reported "the paging file is too small".',
            evidence=[_evidence(record) for record in records[:6]],
        ),
    ]


def detect_accelerator_driver_fault(context: SessionContext) -> list[Finding]:
    """The device raised on tensors moved to it, which is a host driver fault and not a worker or model one.

    Three lines say so. A current worker's start check reports the accelerator unusable and holds intake. An
    older worker has no such check, and the fault shows as jobs failing with torch's NVML assert. The
    management library's version-mismatch line names the usual cause, and is a warning by itself: it is
    logged at every start, before any job has had the chance to fail.
    """
    records = context.session.records
    reports = _matching(records, _ACCELERATOR_UNUSABLE_RE)
    # The start check's report quotes the error it met, so the assert's text is inside that line too.
    asserts = [
        record
        for record in _matching(records, _TORCH_DRIVER_ASSERT_RE)
        if not _ACCELERATOR_UNUSABLE_RE.search(record.message)
    ]
    mismatch = _matching(records, _NVML_DRIVER_MISMATCH_RE)
    if not reports and not asserts and not mismatch:
        return []

    # The parent logs a faulted job's reason twice, once as the received message and once as the fault.
    faulted_jobs = [record for record in asserts if _FAULTED_ON_PROCESS_RE.search(record.message)]
    counted = faulted_jobs or asserts
    if counted:
        times = "once" if len(counted) == 1 else f"{len(counted)} times"
        headline = f"The GPU driver refused work, and jobs failed on it {times}."
    elif reports:
        headline = "The GPU driver refused work at start, and the worker stopped taking jobs."
    else:
        headline = "The NVIDIA libraries and the loaded GPU driver do not match."
    evidence = [_evidence(record) for record in [*reports[:2], *counted[:3], *mismatch[:1]]]
    return [
        Finding(
            kind=FindingKind.ACCELERATOR_DRIVER_FAULT,
            severity=Severity.CRITICAL if reports or counted else Severity.WARNING,
            headline=headline,
            evidence=evidence,
        ),
    ]


_DEVICE_READING_MISSING_MIN_DEFERS = 60
"""Deferrals on a missing reading below which a session is a cold start, not a card with no reading.

A worker logs the line once per scheduling cycle per card, several times a second, and a cold start waits a
report interval or two. Sixty is past any cold start and well under one stalled minute."""


def detect_device_reading_missing(context: SessionContext) -> list[Finding]:
    """Model loads deferred for the whole session because the parent had no memory reading for the card.

    A worker that took the reading from NVML alone deferred every load on a card NVML does not answer for:
    a card of another vendor, or an NVIDIA card whose driver libraries no longer match its kernel module.
    """
    defers = _matching(context.session.records, _DEVICE_READING_MISSING_RE)
    if len(defers) < _DEVICE_READING_MISSING_MIN_DEFERS:
        return []

    cards = _distinct_ordered(
        [match.group("card") for record in defers if (match := _DEVICE_READING_MISSING_RE.search(record.message))],
    )
    card_clause = f"card {cards[0]}" if len(cards) == 1 else f"cards {_clause_join(cards)}"
    mismatch = _matching(context.session.records, _NVML_DRIVER_MISMATCH_RE)
    evidence = [_evidence(defers[0]), _evidence(defers[-1])]
    if mismatch:
        evidence.append(_evidence(mismatch[0]))
    return [
        Finding(
            kind=FindingKind.DEVICE_READING_MISSING,
            severity=Severity.CRITICAL,
            headline=f"Model loads waited {len(defers)} times for a memory reading of {card_clause} that never came.",
            evidence=evidence,
        ),
    ]


def detect_swallowed_oom(context: SessionContext) -> list[Finding]:
    """The 'no images were produced' classification gap (an OOM ComfyUI swallowed)."""
    no_images = _matching(context.session.records, _NO_IMAGES_RE)
    if not no_images:
        return []
    return [
        Finding(
            kind=FindingKind.SWALLOWED_OOM,
            severity=Severity.WARNING,
            headline=(
                f'{_count_text(len(no_images), "job")} failed with a plain "no images produced" message and no '
                "cause given."
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
            kind=FindingKind.ORPHAN_WEDGE,
            severity=Severity.WARNING,
            headline=f"{len(orphans)} jobs were dropped because the process running each one had disappeared.",
            evidence=[_evidence(r) for r in orphans[:4]],
        ),
    ]


@dataclass(frozen=True)
class _LaneReplacement:
    """One inference process the worker flagged for replacement after its own failed jobs."""

    record: LogRecord
    process_id: int
    cuda_context_fault: bool
    """Whether a CUDA runtime error replaced it at once. Otherwise a fault streak across several models did."""
    resource_faults: int = 0
    """How many of the streak's faults were resource or out-of-memory faults (zero for a context fault)."""


@dataclass(frozen=True)
class _CommitExhaustion:
    """The RAM governor's commit-bound pauses, the moments the host ran out of committable memory."""

    records: list[LogRecord]
    timestamps: list[datetime]

    def before(self, moment: datetime | None) -> LogRecord | None:
        """Return the latest commit exhaustion within the lookback before ``moment``, or None."""
        if moment is None:
            return None
        index = bisect.bisect_right(self.timestamps, moment)
        if index == 0:
            return None
        latest = self.records[index - 1]
        return latest if moment - self.timestamps[index - 1] <= _HOST_COMMIT_LOOKBACK else None


def _commit_exhaustion(records: list[LogRecord]) -> _CommitExhaustion:
    """Collect the commit-bound danger-floor pauses, in time order.

    Only the commit-bound form counts: a physical-RAM shortfall leaves committable memory, so it cannot fail a
    CUDA allocation or a channel write the way running out of commit can.
    """
    exhausted = sorted(
        (
            record
            for record in records
            if record.timestamp is not None
            and (match := _RAM_DANGER_PAUSE_RE.search(record.message)) is not None
            and match.group("physical") is not None
        ),
        key=_window_key,
    )
    return _CommitExhaustion(records=exhausted, timestamps=[_window_key(record) for record in exhausted])


def _lane_replacements(records: list[LogRecord]) -> list[_LaneReplacement]:
    """Every replacement the lane fault rules flagged, in log order."""
    replacements: list[_LaneReplacement] = []
    for record in records:
        if (context_fault := _LANE_CUDA_CONTEXT_FAULT_RE.search(record.message)) is not None:
            replacements.append(
                _LaneReplacement(
                    record=record,
                    process_id=int(context_fault.group("process")),
                    cuda_context_fault=True,
                ),
            )
        elif (streak := _LANE_FAULT_STREAK_RE.search(record.message)) is not None:
            replacements.append(
                _LaneReplacement(
                    record=record,
                    process_id=int(streak.group("process")),
                    cuda_context_fault=False,
                    resource_faults=int(streak.group("resource_faults")),
                ),
            )
    return replacements


def _seconds_before(earlier: LogRecord, later: LogRecord) -> float:
    """Seconds from ``earlier`` to ``later``. The callers that pair them timestamp both."""
    return (_window_key(later) - _window_key(earlier)).total_seconds()


def detect_inference_lane_replaced(context: SessionContext) -> list[Finding]:
    """Image processes the worker replaced after a CUDA runtime error or a fault streak.

    A CUDA runtime error is a verdict on the process, but its cause may sit outside it: Windows running out of
    commit can fail the allocation that raised it. The finding checks each context fault against the RAM
    governor's commit-bound pauses so the reader can tell a host condition the worker recovered from apart from a
    worker or driver defect.
    """
    replacements = _lane_replacements(context.session.records)
    if not replacements:
        return []

    exhaustion = _commit_exhaustion(context.session.records)
    context_faults = [replacement for replacement in replacements if replacement.cuda_context_fault]
    streaks = [replacement for replacement in replacements if not replacement.cuda_context_fault]
    process_ids = sorted({replacement.process_id for replacement in replacements})

    causes: list[str] = []
    if context_faults:
        times = "once" if len(context_faults) == 1 else f"{len(context_faults)} times"
        causes.append(f"{times} after a CUDA error")
    if streaks:
        times = "once" if len(streaks) == 1 else f"{len(streaks)} times"
        causes.append(f"{times} after failing jobs on several models")
    noun = "process" if len(process_ids) == 1 else "processes"
    headline = (
        f"The worker replaced image {noun} {_clause_join([str(pid) for pid in process_ids])} {_clause_join(causes)}."
    )

    evidence: list[str] = []
    host_explained = 0
    context_faults_per_process: dict[int, int] = {}
    for replacement in replacements:
        evidence.append(_evidence(replacement.record))
        if not replacement.cuda_context_fault:
            continue
        context_faults_per_process[replacement.process_id] = (
            context_faults_per_process.get(replacement.process_id, 0) + 1
        )
        exhausted = exhaustion.before(replacement.record.timestamp)
        if exhausted is None:
            evidence.append(
                f"Windows did not run out of memory to commit in the {_HOST_COMMIT_LOOKBACK.total_seconds():.0f}s "
                f"before the CUDA error on process {replacement.process_id}",
            )
            continue
        host_explained += 1
        evidence.append(
            f"Windows ran out of memory to commit {_seconds_before(exhausted, replacement.record):.0f}s before the "
            f"CUDA error on process {replacement.process_id}: {_evidence(exhausted)}",
        )

    sentences: list[str] = []
    repeated = any(count > 1 for count in context_faults_per_process.values())
    defect = False
    if context_faults:
        if host_explained == len(context_faults) and not repeated:
            sentences.append(
                "Each CUDA error came right after system memory ran out, and the worker recovered on its own.",
            )
        else:
            sentences.append(
                "A CUDA error without a system memory shortage, or one that repeats, is a worker or driver defect.",
            )
            defect = True
    if streaks:
        sentences.append("A process that fails jobs on several models is itself the common cause.")
        defect = True
    if defect:
        sentences.append("Report it with `horde-log bundle`.")

    return [
        Finding(
            kind=FindingKind.INFERENCE_LANE_REPLACED,
            severity=Severity.WARNING,
            headline=headline,
            action_addendum=" ".join(sentences),
            evidence=evidence[:12],
        ),
    ]


@dataclass(frozen=True)
class _SlotQuarantine:
    """One quarantined inference slot and the replacements of it that led there."""

    record: LogRecord
    slot: int
    reason: str
    recoveries: list[RecoveryDiagnostic]
    """Parent recovery diagnostics for the slot in the crash-loop window, the quarantining one included."""
    lane_replacements: list[_LaneReplacement]
    """Lane fault rule replacements of the slot in the same window, which name the cause the last one lost."""


def _slot_quarantines(context: SessionContext) -> list[_SlotQuarantine]:
    """Every inference slot quarantine in the session, with the slot's replacements over the crash-loop window.

    The quarantining replacement's recovery reason names only the breaker, so the lane fault lines in the window are
    read alongside the recovery diagnostics to find the cause.
    """
    from horde_worker_regen.process_management.lifecycle.process_lifecycle import CRASH_LOOP_WINDOW_SECONDS

    window = timedelta(seconds=CRASH_LOOP_WINDOW_SECONDS)
    lane_replacements = _lane_replacements(context.session.records)
    quarantines: list[_SlotQuarantine] = []
    for record in context.session.records:
        match = _SLOT_QUARANTINED_RE.search(record.message)
        if match is None or record.timestamp is None:
            continue
        slot = int(match.group("slot"))
        end = record.timestamp
        start = end - window
        quarantines.append(
            _SlotQuarantine(
                record=record,
                slot=slot,
                reason=match.group("reason"),
                recoveries=[
                    recovery
                    for recovery in context.recoveries
                    if recovery.process_id == slot
                    and recovery.timestamp is not None
                    and start <= recovery.timestamp <= end
                ],
                lane_replacements=[
                    replacement
                    for replacement in lane_replacements
                    if replacement.process_id == slot
                    and replacement.record.timestamp is not None
                    and start <= replacement.record.timestamp <= end
                ],
            ),
        )
    return quarantines


def detect_inference_slot_quarantined(context: SessionContext) -> list[Finding]:
    """Inference slots the worker stopped respawning, crash loop or crash on start, with what drove them there.

    Lane replacements count toward the crash-loop breaker, so a process replaced for CUDA errors or fault streaks
    can end out of service. The replacement reasons in the window say whether that, or a plain crash, drove it.
    """
    quarantines = _slot_quarantines(context)
    if not quarantines:
        return []

    if len(quarantines) == 1:
        only = quarantines[0]
        headline = f'Image process {only.slot} was taken out of service for "{only.reason}".'
    else:
        slots = _clause_join([str(slot) for slot in sorted({quarantine.slot for quarantine in quarantines})])
        headline = f"Image processes {slots} were taken out of service {len(quarantines)} times."

    lane_driven = any(
        quarantine.lane_replacements
        or any(
            marker in recovery.reason for recovery in quarantine.recoveries for marker in _LANE_FAULT_REASON_MARKERS
        )
        for quarantine in quarantines
    )
    evidence: list[str] = []
    for quarantine in quarantines:
        evidence.append(_evidence(quarantine.record))
        evidence.extend(
            f"replacement of process {quarantine.slot}: {recovery.reason}"
            for recovery in quarantine.recoveries
            if recovery.record is not quarantine.record
        )
        evidence.extend(_evidence(replacement.record) for replacement in quarantine.lane_replacements)

    return [
        Finding(
            kind=FindingKind.INFERENCE_SLOT_QUARANTINED,
            severity=Severity.CRITICAL,
            headline=headline,
            action_addendum=(
                "The reasons include CUDA errors or repeated failed jobs, so report it with `horde-log bundle`."
                if lane_driven
                else None
            ),
            evidence=evidence[:12],
        ),
    ]


def detect_lane_fault_rule_misfire(context: SessionContext) -> list[Finding]:
    """Lane replacements whose evidence says the process was not the problem.

    Two trade-offs in the lane fault rules can replace a healthy process. An allocator OOM counts toward the fault
    streak, so on an over-committed card the streak trips on card pressure. The same process faulting again soon
    after its replacement shows the replacement did not help. And a CUDA context fault counts toward the crash-loop
    breaker even when Windows commit exhaustion caused it, so a host condition can quarantine a slot. Surfacing both
    lets maintainers judge from operator reports whether the rules need changing.
    """
    records = context.session.records
    lane_replacements = _lane_replacements(records)
    faults_by_process: dict[int, list[LogRecord]] = {}
    for record in records:
        if record.timestamp is not None and (match := _FAULTED_ON_PROCESS_RE.search(record.message)) is not None:
            faults_by_process.setdefault(int(match.group("pid")), []).append(record)

    evidence: list[str] = []
    pressure_streaks: list[_LaneReplacement] = []
    for replacement in lane_replacements:
        moment = replacement.record.timestamp
        if replacement.cuda_context_fault or replacement.resource_faults < _MISFIRE_MIN_RESOURCE_FAULTS:
            continue
        if moment is None:
            continue
        # The tripping job's own fault line follows the streak line, so faults count only after the replacement.
        replaced_at = next(
            (
                recovery.timestamp
                for recovery in context.recoveries
                if recovery.process_id == replacement.process_id
                and recovery.timestamp is not None
                and recovery.timestamp >= moment
            ),
            moment,
        )
        refault = next(
            (
                record
                for record in faults_by_process.get(replacement.process_id, [])
                if replaced_at < _window_key(record) <= replaced_at + _REFAULT_WINDOW
            ),
            None,
        )
        later_streak = next(
            (
                other.record
                for other in lane_replacements
                if other.process_id == replacement.process_id
                and not other.cuda_context_fault
                and other.record.timestamp is not None
                and other.record.timestamp > moment
            ),
            None,
        )
        follow_up = refault or later_streak
        if follow_up is None:
            continue
        pressure_streaks.append(replacement)
        evidence.extend([_evidence(replacement.record), f"failed again afterwards: {_evidence(follow_up)}"])

    exhaustion = _commit_exhaustion(records)
    host_quarantines: list[_SlotQuarantine] = []
    for quarantine in _slot_quarantines(context):
        explained = [
            (replacement, exhausted)
            for replacement in quarantine.lane_replacements
            if replacement.cuda_context_fault
            and (exhausted := exhaustion.before(replacement.record.timestamp)) is not None
        ]
        if not explained:
            continue
        host_quarantines.append(quarantine)
        evidence.append(_evidence(quarantine.record))
        for replacement, exhausted in explained:
            evidence.extend(
                [
                    _evidence(replacement.record),
                    f"Windows ran out of memory to commit {_seconds_before(exhausted, replacement.record):.0f}s "
                    f"before: {_evidence(exhausted)}",
                ],
            )

    if not pressure_streaks and not host_quarantines:
        return []

    if pressure_streaks and host_quarantines:
        replaced_times = "once" if len(pressure_streaks) == 1 else f"{len(pressure_streaks)} times"
        removed_times = "once" if len(host_quarantines) == 1 else f"{len(host_quarantines)} times"
        headline = (
            f"Image processes were replaced {replaced_times} for out-of-memory faults and taken out of service "
            f"{removed_times} after system memory ran out."
        )
    elif pressure_streaks:
        process_ids = sorted({replacement.process_id for replacement in pressure_streaks})
        noun = "process" if len(process_ids) == 1 else "processes"
        verb = "was" if len(process_ids) == 1 else "were"
        headline = (
            f"Image {noun} {_clause_join([str(pid) for pid in process_ids])} {verb} replaced for mostly "
            "out-of-memory faults and failed again afterwards."
        )
    else:
        slots = sorted({quarantine.slot for quarantine in host_quarantines})
        noun = "process" if len(slots) == 1 else "processes"
        verb = "was" if len(slots) == 1 else "were"
        headline = (
            f"Image {noun} {_clause_join([str(slot) for slot in slots])} {verb} taken out of service after a CUDA "
            "error that followed system memory running out."
        )

    return [
        Finding(
            kind=FindingKind.LANE_FAULT_RULE_MISFIRE,
            severity=Severity.WARNING,
            headline=headline,
            evidence=evidence[:12],
        ),
    ]


def detect_supervisor_channel_lost(context: SessionContext) -> list[Finding]:
    """The worker's channel to its supervisor closed, so the supervisor may judge a healthy worker frozen.

    The channel's own write can fail under the same commit exhaustion that breaks a CUDA context, so each drop is
    checked against the RAM governor's commit-bound pauses.
    """
    drops = _matching(context.session.records, _SUPERVISOR_CHANNEL_DROPPED_RE)
    if not drops:
        return []
    exhaustion = _commit_exhaustion(context.session.records)
    first = drops[0]
    first_time = first.timestamp.strftime("%H:%M:%S") if first.timestamp is not None else "an unknown time"
    if len(drops) == 1:
        exhausted = exhaustion.before(first.timestamp)
        memory_clause = (
            f", {_seconds_before(exhausted, first):.0f} seconds after system memory ran out"
            if exhausted is not None
            else ""
        )
        headline = f"The worker lost its link to its supervisor at {first_time}{memory_clause}."
    else:
        headline = f"The worker lost its link to its supervisor {len(drops)} times, first at {first_time}."

    evidence: list[str] = []
    for drop in drops[:4]:
        evidence.append(_evidence(drop))
        exhausted = exhaustion.before(drop.timestamp)
        evidence.append(
            f"Windows ran out of memory to commit {_seconds_before(exhausted, drop):.0f}s before: "
            f"{_evidence(exhausted)}"
            if exhausted is not None
            else "Windows did not run out of memory to commit in the "
            f"{_HOST_COMMIT_LOOKBACK.total_seconds():.0f}s before"
        )
    return [
        Finding(
            kind=FindingKind.SUPERVISOR_CHANNEL_LOST,
            severity=Severity.WARNING,
            headline=headline,
            evidence=evidence,
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
    given_up = _count_text(giveups, "queued job")
    cancelled = _count_text(aborts, "job")
    if giveups and aborts:
        return f"Just before, the worker gave up on {given_up} and the horde cancelled {cancelled} as too slow."
    if aborts:
        return (
            f"Just before, the horde cancelled {cancelled} as too slow. The worker was generating slower than "
            "the horde's per-job deadline."
        )
    if giveups:
        return f"Just before, the worker gave up on {given_up}."
    return "The log does not say which jobs were dropped."


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
        dropped = _total_dropped_jobs(context.session.records) + _count_server_slow_aborts(context.session.records)
        # Point the operator at whichever upstream finding actually applies: a slow-generation spiral and a
        # scheduler wedge produce the same maintenance symptom but call for opposite fixes.
        see_also = FindingKind.SLOW_GENERATION_DROP_SPIRAL if slow_aborts else FindingKind.SCHEDULER_STARVATION_WEDGE
        return [
            Finding(
                kind=FindingKind.FORCED_MAINTENANCE,
                severity=Severity.CRITICAL,
                headline=(
                    f"The horde put the worker into maintenance after it dropped {_count_text(dropped, 'job')}, and "
                    f"refused {_count_text(len(maintenance), 'request')} for work."
                    if dropped
                    else (
                        "The horde put the worker into maintenance for dropping jobs, and refused "
                        f"{_count_text(len(maintenance), 'request')} for work."
                    )
                ),
                action_addendum=(
                    "Fix what is dropping jobs before clearing maintenance, or it comes back. For slow "
                    "generation lower `max_power`, `max_threads`, `queue_size` or `max_batch`. For a scheduler "
                    "stall turn off `unload_models_from_vram_often` and `high_performance_mode`."
                ),
                evidence=[drop_clause] + [_evidence(r) for r in (maintenance[:1] + giveups[:1] + slow_aborts[:1])],
                see_also=see_also,
            ),
        ]
    return [
        Finding(
            kind=FindingKind.FORCED_MAINTENANCE,
            severity=Severity.INFO,
            title_override="The worker was in maintenance mode",
            headline=(
                f"The horde refused {_count_text(len(maintenance), 'request')} for work because the worker was in "
                "maintenance for a reason other than dropped jobs."
            ),
            action_addendum=(
                "If you did not set maintenance yourself, clear it on the horde's worker page and check the API "
                "key. Otherwise nothing to do."
            ),
            evidence=[_evidence(maintenance[0])],
        ),
    ]


def detect_scheduler_starvation_wedge(context: SessionContext) -> list[Finding]:
    """An over-conservative VRAM budget deferring head-of-queue jobs on an idle device (the root cause).

    The budget refused to admit a head-of-queue model on a device with ample free VRAM, so the queue
    deadlocked with idle processes and the recovery supervisor soft-reset the pools and faulted the
    backlog. A lone starvation diagnostic that cleared without escalating is a near-miss (warning); one
    that still ended in a soft reset and faulted jobs is the self-inflicted wedge (critical). The starved
    duration is read straight off the diagnostic's own parked-seconds figure, not derived from timestamps.
    """
    starved = _matching(context.session.records, _HEAD_STARVATION_MODEL_RE)
    if not starved:
        return []

    durations = [int(m.group("seconds")) for r in starved if (m := _HEAD_STARVATION_MODEL_RE.search(r.message))]
    free_vrams = [int(m.group("free")) for r in starved if (m := _HEAD_STARVATION_AVAILABLE_RE.search(r.message))]
    max_starved = max(durations) if durations else 0
    free_hint = f" with as much as {max(free_vrams)} MB free" if free_vrams else ""

    soft_resets = _matching(context.session.records, _SOFT_RESET_RE)
    dropped = _total_dropped_jobs(context.session.records)
    escalated = bool(soft_resets) or dropped > 0

    if escalated:
        return [
            Finding(
                kind=FindingKind.SCHEDULER_STARVATION_WEDGE,
                severity=Severity.CRITICAL,
                headline=(
                    f"The VRAM budget held the next job back for up to {max_starved} seconds on an idle card, "
                    f"then rebuilt the processes and dropped {_count_text(dropped, 'job')}."
                ),
                evidence=[f"The worker rebuilt its processes {_times_text(len(soft_resets))}{free_hint}."]
                + [_evidence(r) for r in (starved[:2] + soft_resets[:1])],
                see_also=FindingKind.FORCED_MAINTENANCE,
            ),
        ]
    return [
        Finding(
            kind=FindingKind.SCHEDULER_STARVATION_WEDGE,
            severity=Severity.WARNING,
            title_override="The VRAM budget held the next job back, then let it through",
            headline=(
                f"The VRAM budget held the next job back for up to {max_starved} seconds on an idle card, "
                "then let it through."
            ),
            action_addendum="Watch for it under load.",
            evidence=([f"Held back{free_hint}."] if free_hint else []) + [_evidence(r) for r in starved[:3]],
        ),
    ]


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
    free_hint = f" with as much as {max(free_vrams)} MB free" if free_vrams else ""

    storm_start = min(timestamps) if timestamps else None
    corrective = _matching(context.session.records, _GIVE_UP_RE) + _matching(
        context.session.records,
        _CONSECUTIVE_PAUSE_RE,
    )
    resolved = any(
        storm_start is None or (record.timestamp is not None and record.timestamp >= storm_start)
        for record in corrective
    )

    outcome = "the worker eventually gave up or paused" if resolved else "nothing ever cleared it"
    verdict = (
        f'"{model}" was held back {len(entries)} times over {idle_window:.0f} seconds on an idle card, and {outcome}.'
    )
    stretch = f"Held back for up to {max_starved}s at a stretch{free_hint}."
    return [
        Finding(
            kind=FindingKind.UNSATISFIABLE_HEAD_STARVATION,
            severity=Severity.WARNING if resolved else Severity.CRITICAL,
            headline=verdict,
            evidence=[stretch] + [_evidence(record) for record, _ in (entries[:2] + entries[-1:])],
        ),
    ]


def _spiral_latency_samples(lifecycle: JobLifecycleModel) -> tuple[float | None, float | None]:
    """The median pop->submit latency and median generation time over the session's submitted jobs.

    Reads the lifecycle model rather than the submit line directly; the model reconstructs a missing pop
    from the submit line's own "popped N seconds ago", so the pair is identical to what parsing that line
    gave, and stays correct when the pop line is present too.
    """
    latencies: list[float] = []
    generations: list[float] = []
    for job in lifecycle.jobs.values():
        if job.submitted_at is None or job.popped_at is None:
            continue
        latencies.append((job.submitted_at - job.popped_at).total_seconds())
        if job.generation_seconds is not None:
            generations.append(job.generation_seconds)
    return _median(latencies), _median(generations)


def _aging_finding(
    context: SessionContext,
    lifecycle: JobLifecycleModel,
    *,
    severity: Severity,
    aborts: list[LogRecord],
    base_verdict: str,
    tail: str,
    median_latency: float,
    median_gen: float,
) -> list[Finding]:
    """Build the "aged outside generation" spiral finding, attributed to the stage that actually held it.

    Splits the non-generation wait into pop->dispatch and inference-finished->submit and names the larger,
    because the two have opposite remedies and the ratio alone cannot tell them apart. With neither
    segment measurable the finding says the wait could not be located rather than naming a stage.
    """
    pre_segment, _inference_segment, post_segment = lifecycle.wait_segments()
    pre_median, post_median = pre_segment.median, post_segment.median
    # The caller's verdict is a whole sentence. The headline is one sentence, so it continues that verdict
    # and the two medians it rests on go to the evidence.
    lead = base_verdict.rstrip(".")
    medians_clause = (
        f"Generation took a median {median_gen:.0f} seconds, and each job took a median {median_latency:.0f} "
        "seconds from offer to submit."
    )

    if pre_median is None and post_median is None:
        return [
            Finding(
                kind=FindingKind.SLOW_GENERATION_DROP_SPIRAL,
                severity=severity,
                title_override="Jobs were dropped, and the log cannot say where the time went",
                headline=(
                    f"{lead}, and this log cannot say where each job's median {median_latency:.0f} seconds went."
                ),
                action_addendum=(
                    "Capture the log again at DEBUG so the job start and finish lines are present, then run "
                    "`horde-log jobs` to see where the wait is. Lowering `max_power` will not help, since "
                    "generation was already fast."
                ),
                evidence=[medians_clause]
                + ([tail.strip()] if tail else [])
                + [_evidence(record) for record in aborts[:2]],
            ),
        ]

    pre_share = pre_median or 0.0
    post_share = post_median or 0.0
    split_clause = (
        f"Of the wait, a median {pre_share:.0f}s was spent before the job started and {post_share:.0f}s after "
        "generation finished."
    )

    if pre_share >= post_share:
        stall_evidence = _matching(context.session.records, _DISPATCH_STALL_RE)[:1]
        return [
            Finding(
                kind=FindingKind.SLOW_GENERATION_DROP_SPIRAL,
                severity=severity,
                title_override="Jobs were dropped while waiting to start",
                headline=(f"{lead}, and jobs spent most of a median {median_latency:.0f} seconds waiting to start."),
                action_addendum=(
                    "Lowering `max_power` will not help, and neither will a faster safety check. Look at what "
                    "held jobs from starting, and on a multi-card host whether work spread across the cards. "
                    "Run `horde-log jobs` for the per-job split."
                ),
                evidence=[medians_clause, split_clause]
                + ([tail.strip()] if tail else [])
                + [_evidence(record) for record in (aborts[:2] + stall_evidence)],
                see_also=FindingKind.MULTI_CARD_DISPATCH_SERIALIZATION,
            ),
        ]

    safety_times = [
        float(match.group(1))
        for record in _matching(context.session.records, _SAFETY_DURATION_RE)
        if (match := _SAFETY_DURATION_RE.search(record.message))
    ]
    safety_clause = f" Safety checks averaged {_mean(safety_times):.1f}s each." if safety_times else ""
    safety_evidence = _matching(context.session.records, _SAFETY_DURATION_RE)[:1]
    return [
        Finding(
            kind=FindingKind.SLOW_GENERATION_DROP_SPIRAL,
            severity=severity,
            title_override="Jobs were dropped while waiting after generation",
            headline=(
                f"{lead}, and jobs spent most of a median {median_latency:.0f} seconds waiting after generation."
            ),
            action_addendum=(
                "Lowering `max_power` will not help. Speed up the safety check by turning on "
                "`safety_on_gpu`, so it does not run on the CPU."
            ),
            evidence=[medians_clause, split_clause]
            + ([safety_clause.strip()] if safety_clause else [])
            + ([tail.strip()] if tail else [])
            + [_evidence(record) for record in (aborts[:2] + safety_evidence)],
        ),
    ]


def detect_slow_generation_drop_spiral(context: SessionContext) -> list[Finding]:
    """The horde aborting generations as too slow, the drop mechanism behind a slow-worker maintenance.

    This is the root cause the starvation-wedge detector does not cover: the worker is not wedged, it is
    simply submitting later than the horde's per-job deadline, so the server aborts each late submission
    ("took too long to process") and faults it. A sustained run of these aborts is what the horde counts
    as dropped jobs and answers with forced maintenance.

    Where the time went decides the remedy, and the three candidates call for opposite fixes, so the
    verdict is attributed from the lifecycle split rather than assumed. When jobs waited far longer than
    they generated, the wait is attributed to whichever of pop->dispatch (a scheduling problem) or
    inference-finished->submit (a pipeline-balance problem) is the larger, and both medians are named. A
    log that carries neither the dispatch nor the finish line cannot locate the wait at all, and the
    verdict says so instead of naming a stage on no evidence.
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

    # Decide whether jobs aged outside generation (fast generation, long pop->submit latency) or in
    # generation itself (slow GPU). The two share the "too slow" abort but call for opposite fixes, so the
    # detector measures the submitted jobs' own latency-vs-generation breakdown rather than guessing.
    lifecycle = job_lifecycle_for(context)
    median_latency, median_gen = _spiral_latency_samples(lifecycle)
    queue_aging = (
        median_latency is not None
        and median_gen is not None
        and median_gen > 0
        and median_latency >= median_gen * _QUEUE_AGING_LATENCY_RATIO
    )

    base_verdict = f"The horde dropped {_count_text(len(aborts), 'job')}{span_clause} for taking too long."
    tail = "" if spiral else "Few so far. If the drops continue, the horde puts the worker into maintenance."

    if queue_aging:
        assert median_latency is not None and median_gen is not None
        return _aging_finding(
            context,
            lifecycle,
            severity=severity,
            aborts=aborts,
            base_verdict=base_verdict,
            tail=tail,
            median_latency=median_latency,
            median_gen=median_gen,
        )

    slow_clause = ""
    slow_evidence: list[str] = []
    if ratios:
        slow_clause = f" Generation ran up to {max(ratios):.1f}x its expected time."
        if free_vrams:
            slow_evidence.append(
                f"As little as {min(free_vrams)} MB of VRAM was free, so the card was over-committed."
            )
    return [
        Finding(
            kind=FindingKind.SLOW_GENERATION_DROP_SPIRAL,
            severity=severity,
            title_override=None if spiral else "Jobs were dropped for slow generation",
            headline=(base_verdict + slow_clause).strip(),
            action_addendum=(
                "Lower `max_power`, `max_threads`, `queue_size` or `max_batch` so each job finishes in time. Put "
                "models on an SSD and free up VRAM and RAM. Clearing maintenance without slowing the intake "
                "triggers it again."
            ),
            evidence=slow_evidence + ([tail] if tail else []) + [_evidence(r) for r in (aborts[:2] + slowdowns[:2])],
        ),
    ]


def detect_consecutive_failure_pause(context: SessionContext) -> list[Finding]:
    """The worker self-pausing job pops after three consecutive faults (a downstream symptom)."""
    pauses = _matching(context.session.records, _CONSECUTIVE_PAUSE_RE)
    if not pauses:
        return []
    return [
        Finding(
            kind=FindingKind.CONSECUTIVE_FAILURE_PAUSE,
            severity=Severity.WARNING,
            headline=(
                f"The worker paused asking for work {_times_text(len(pauses))} after three failed jobs in a row."
            ),
            evidence=[_evidence(r) for r in pauses[:3]],
        ),
    ]


def _longest_busy_fault_run(records: list[LogRecord]) -> list[LogRecord]:
    """Return the fault reports of the longest run of busy-refused text jobs with no text job submitted between.

    A fault counts toward a run only for a job the backend was seen refusing as busy; a fault for another
    reason neither extends nor ends it. A submitted text job ends it. Returns the empty list when no run
    reaches :data:`_TEXT_BUSY_FAULT_RUN_MIN_JOBS` jobs over :data:`_TEXT_BUSY_FAULT_RUN_MIN_SPAN`.
    """
    refused_jobs: set[str] = set()
    longest: list[LogRecord] = []
    current: list[LogRecord] = []

    def qualifies(run: list[LogRecord]) -> bool:
        if len(run) < _TEXT_BUSY_FAULT_RUN_MIN_JOBS:
            return False
        first, last = run[0].timestamp, run[-1].timestamp
        return first is not None and last is not None and last - first >= _TEXT_BUSY_FAULT_RUN_MIN_SPAN

    for record in records:
        refusal = _TEXT_BUSY_REOFFER_RE.search(record.message) or _TEXT_BUSY_FAULT_LEGACY_RE.search(record.message)
        if refusal is not None:
            refused_jobs.add(refusal.group("job"))
            continue
        fault = _TEXT_JOB_FAULT_REPORTED_RE.search(record.message)
        if fault is not None:
            if fault.group("job") in refused_jobs:
                current.append(record)
            continue
        if _TEXT_JOB_SUBMITTED_RE.search(record.message):
            if qualifies(current) and len(current) > len(longest):
                longest = current
            current = []
    if qualifies(current) and len(current) > len(longest):
        longest = current
    return longest


def _text_backend_wedged_headline(record: LogRecord, *, times: str, outcome: str) -> str:
    """Return the headline for the wedge-hold lines, stating the backend's own counters from the first line.

    The count of finished generations standing still is what proves the hold was not a shared backend busy
    with other clients' work, so the headline carries it. A line from before the counters keeps the wording
    those workers' bundles have always had.
    """
    match = _TEXT_BACKEND_WEDGED_RE.search(record.message)
    if match is not None and match.group("generations_now") is not None:
        return (
            f"The text backend refused every job as busy {times}{outcome}, while its finished generations "
            f"stopped at {match.group('generations_now')}."
        )
    if match is not None and match.group("no_counters") is not None:
        return (
            f"The text backend refused every job as busy while generating nothing, {times}{outcome}, and it "
            "reported no generation counts."
        )
    return f"The text backend refused every job as busy while generating nothing, {times}{outcome}."


def detect_text_backend_wedged(context: SessionContext) -> list[Finding]:
    """A text backend refusing every job as busy while it generates nothing, from either worker generation.

    A current worker states the episode itself when it holds its text jobs, with the backend's own counters
    at the decision, and the headline carries the count of finished generations from the first such line.
    An older worker only faulted the refused jobs one by one, so its bundles are read for a long run of
    busy-refused faults with no text job submitted in between; the horde counts each of those faults toward
    maintenance.
    """
    records = context.session.records
    held = _matching(records, _TEXT_BACKEND_WEDGED_RE)
    if held:
        times = "once" if len(held) == 1 else f"{len(held)} times"
        restarts_exhausted = bool(_matching(held, _TEXT_BACKEND_RELAUNCH_BOUND_RE))
        outcome = ", and restarting it did not help" if restarts_exhausted else ""
        return [
            Finding(
                kind=FindingKind.TEXT_BACKEND_WEDGED,
                severity=Severity.CRITICAL,
                headline=_text_backend_wedged_headline(held[0], times=times, outcome=outcome),
                evidence=[_evidence(record) for record in held[:3]],
            ),
        ]

    run = _longest_busy_fault_run(records)
    if not run:
        return []
    first_ts, last_ts = run[0].timestamp, run[-1].timestamp
    minutes = round((last_ts - first_ts).total_seconds() / 60) if first_ts and last_ts else 0
    return [
        Finding(
            kind=FindingKind.TEXT_BACKEND_WEDGED,
            severity=Severity.CRITICAL,
            headline=(
                f"The text backend refused {len(run)} text jobs in a row as busy over {minutes} minutes, and "
                "each went back to the horde unfinished."
            ),
            evidence=[_evidence(record) for record in (run[:2] + run[-1:])],
        ),
    ]


@dataclass(frozen=True)
class _DeferredStartGaveUp:
    """Represents a deferred GPU start that gave up on its card, as its escalation or retirement line states it."""

    record: LogRecord
    position: int
    """The line's index in the session's records, so later lines can be paired against it."""
    device: int
    required_mb: int
    room_mb: int | None
    """The most the card can free beside its standing tenants for a structural shortfall; None for a stalled wait."""
    waited_seconds: int | None
    """How long a stalled wait lasted; None for a structural shortfall, which gives up without waiting."""
    text_backend_on_card: bool
    """Whether the text backend the worker launches was charged against this card when the start gave up."""

    @property
    def structural(self) -> bool:
        """Whether the card can never free what the start needs, rather than having stopped freeing it."""
        return self.room_mb is not None

    @property
    def room_text(self) -> str:
        """The card's achievable room as a headline states it; a room below zero reads as none."""
        return f"{max(self.room_mb or 0, 0)}MB"


def _deferred_starts_gave_up(
    records: list[LogRecord],
    *,
    stalled: re.Pattern[str],
    structural: re.Pattern[str],
) -> list[_DeferredStartGaveUp]:
    """Return every gives-up line of one process kind, noting whether the text backend held that card then.

    The text backend's charge lines are edge-triggered, so the set of cards it holds is replayed in order.
    """
    charged_devices: set[int] = set()
    gave_up: list[_DeferredStartGaveUp] = []
    for position, record in enumerate(records):
        charged = _MANAGED_TEXT_BACKEND_CHARGED_RE.search(record.message)
        if charged is not None:
            charged_devices.add(int(charged.group("device")))
            continue
        released = _MANAGED_TEXT_BACKEND_RELEASED_RE.search(record.message)
        if released is not None:
            charged_devices.discard(int(released.group("device")))
            continue
        stalled_match = stalled.search(record.message)
        structural_match = None if stalled_match is not None else structural.search(record.message)
        match = stalled_match or structural_match
        if match is None:
            continue
        device = int(match.group("device"))
        gave_up.append(
            _DeferredStartGaveUp(
                record=record,
                position=position,
                device=device,
                required_mb=int(match.group("required")),
                room_mb=int(structural_match.group("room")) if structural_match is not None else None,
                waited_seconds=int(stalled_match.group("waited")) if stalled_match is not None else None,
                text_backend_on_card=device in charged_devices,
            ),
        )
    return gave_up


def _span_text(seconds: float | None) -> str:
    """Return a duration in the unit a reader would say it in: minutes under an hour, hours above."""
    if seconds is None:
        return "an unknown time"
    if seconds < 3600:
        minutes = max(1, round(seconds / 60))
        return "1 minute" if minutes == 1 else f"{minutes} minutes"
    return f"{seconds / 3600:.1f} hours"


def _clock_text(moment: datetime | None) -> str:
    """Return a line's time of day, or the placeholder evidence lines use for a line without one."""
    return moment.strftime("%H:%M:%S") if moment is not None else "--:--:--"


def _seconds_between(start: datetime | None, end: datetime | None) -> float | None:
    """Return the seconds from ``start`` to ``end``, or None when either timestamp is missing."""
    if start is None or end is None:
        return None
    return max(0.0, (end - start).total_seconds())


def _structural_shortfall_action(gave_up: _DeferredStartGaveUp, *, remedy: str) -> str:
    """Return the advice for a card that cannot free what a start needs, naming the text backend when it is there."""
    if gave_up.text_backend_on_card:
        return (
            "The text backend the worker runs holds memory on that card. Set `text_gpu_device_index` to another "
            f"card, or {remedy}."
        )
    return f"Close the other programs using memory on that card, or {remedy}."


@dataclass
class _RetiredInferenceStart:
    """Represents one inference start taken out of its card's plan, and the line that put it back, if any."""

    gave_up: _DeferredStartGaveUp
    restored: LogRecord | None = None


def _inference_slot_retired_finding(
    device: int,
    retirements: list[_RetiredInferenceStart],
    *,
    session_end: datetime | None,
) -> Finding:
    """Return the finding for one card's retired inference starts.

    A retirement never restored is a card serving short for the rest of the session, so it leads the
    headline and makes the finding a warning. A structural one is the card's configuration: it cannot hold
    every process planned for it beside what else runs there, and the advice says so. Retirements that all
    came back are context.
    """
    evidence: list[str] = []
    for retirement in retirements[:3]:
        began_text = _clock_text(retirement.gave_up.record.timestamp)
        ended_text = "the session end" if retirement.restored is None else _clock_text(retirement.restored.timestamp)
        evidence.append(f"card {device}: one image process fewer from {began_text} to {ended_text}")
    evidence.append(_evidence(retirements[0].gave_up.record))
    evidence.extend(_evidence(retirement.restored) for retirement in retirements[:2] if retirement.restored)

    never_restored = [retirement for retirement in retirements if retirement.restored is None]
    if not never_restored:
        total_seconds = sum(
            _seconds_between(retirement.gave_up.record.timestamp, retirement.restored.timestamp) or 0.0
            for retirement in retirements
            if retirement.restored is not None
        )
        times = "once" if len(retirements) == 1 else f"{len(retirements)} times"
        return Finding(
            kind=FindingKind.INFERENCE_SLOT_RETIRED,
            severity=Severity.INFO,
            headline=(
                f"Card {device} ran one image process fewer {times}, for {_span_text(total_seconds)} in total, "
                "and the worker started it again each time."
            ),
            action_addendum=(
                "No action needed. If it happens often, look for another program using memory on that card."
            ),
            evidence=evidence,
        )

    first = never_restored[0].gave_up
    count = len(never_restored)
    processes = "1 image process" if count == 1 else f"{count} image processes"
    lasted = _span_text(_seconds_between(first.record.timestamp, session_end))
    if first.structural:
        headline = (
            f"Card {device} ran {processes} fewer for the session's last {lasted}, because it can free "
            f"{first.room_text} and a start needs {first.required_mb}MB."
        )
        action = (
            "This card cannot hold every image process planned for it. "
            f"{_structural_shortfall_action(first, remedy='lower `max_threads`')}"
        )
    else:
        headline = (
            f"Card {device} ran {processes} fewer for the session's last {lasted}, because free memory stayed "
            f"below the {first.required_mb}MB a start needs."
        )
        action = (
            "Look for another program holding memory on that card and close it. The worker starts the process "
            "again once the card has had room for a few minutes."
        )
    return Finding(
        kind=FindingKind.INFERENCE_SLOT_RETIRED,
        severity=Severity.WARNING,
        headline=headline,
        action_addendum=action,
        evidence=evidence,
    )


def detect_inference_slot_retired(context: SessionContext) -> list[Finding]:
    """A card that served with fewer image processes than planned because a start could not get room on it.

    A retired inference slot leaves no idle process and no error behind, so without this the session reads
    as spare capacity. Each retirement is paired with the next restore on the same card; one never restored
    lasted until the session ended. One finding per card.
    """
    records = context.session.records
    retirements_by_device: dict[int, list[_RetiredInferenceStart]] = {}
    for gave_up in _deferred_starts_gave_up(
        records,
        stalled=_INFERENCE_START_RETIRED_RE,
        structural=_INFERENCE_START_RETIRED_STRUCTURAL_RE,
    ):
        retirements_by_device.setdefault(gave_up.device, []).append(_RetiredInferenceStart(gave_up=gave_up))
    if not retirements_by_device:
        return []
    for position, record in enumerate(records):
        restored = _INFERENCE_SLOT_RESTORED_RE.search(record.message)
        if restored is None:
            continue
        open_retirement = next(
            (
                retirement
                for retirement in retirements_by_device.get(int(restored.group("device")), [])
                if retirement.restored is None and retirement.gave_up.position < position
            ),
            None,
        )
        if open_retirement is not None:
            open_retirement.restored = record
    session_end = context.session.end_ts
    return [
        _inference_slot_retired_finding(device, retirements, session_end=session_end)
        for device, retirements in sorted(retirements_by_device.items())
    ]


def _safety_returned_to_card_after(records: list[LogRecord], escalation: _DeferredStartGaveUp) -> LogRecord | None:
    """Return the first line restoring the safety process to a GPU after ``escalation``, or None."""
    for record in records[escalation.position + 1 :]:
        cycle = _SAFETY_PLACEMENT_CYCLE_RE.search(record.message)
        if cycle is not None and "restoring the safety process to the GPU" in cycle.group(0):
            return record
    return None


def detect_safety_start_escalated_to_cpu(context: SessionContext) -> list[Finding]:
    """A safety start that could not get room on its card, so the worker started safety on the CPU instead.

    The headline states the first escalation's numbers; the evidence says whether safety went back to a card
    before the session ended, which is when the slower CPU checks stopped.
    """
    records = context.session.records
    escalations = _deferred_starts_gave_up(
        records,
        stalled=_SAFETY_START_ESCALATED_RE,
        structural=_SAFETY_START_ESCALATED_STRUCTURAL_RE,
    )
    if not escalations:
        return []
    first = escalations[0]
    if first.structural:
        headline = (
            f"The safety check started on the CPU because card {first.device} can free {first.room_text} and the "
            f"check needs {first.required_mb}MB."
        )
        action = _structural_shortfall_action(first, remedy="lower `max_threads` to leave it room")
    else:
        headline = (
            f"The safety check started on the CPU after card {first.device} spent "
            f"{_span_text(first.waited_seconds)} failing to free the {first.required_mb}MB it needs."
        )
        action = (
            "Look for another program holding memory on that card and close it. Checks run slower on the CPU "
            "until the card has room again."
        )
    returned = _safety_returned_to_card_after(records, escalations[-1])
    outcome = (
        "stayed on the CPU until the session ended"
        if returned is None
        else f"returned to a card at {_clock_text(returned.timestamp)}"
    )
    count_note = (
        []
        if len(escalations) == 1
        else [f"The safety check was started on the CPU {len(escalations)} times this session."]
    )
    return [
        Finding(
            kind=FindingKind.SAFETY_START_ESCALATED_TO_CPU,
            severity=Severity.WARNING,
            headline=headline,
            action_addendum=action,
            evidence=[
                f"After the last move to the CPU, the safety check {outcome}.",
                *count_note,
                *(_evidence(escalation.record) for escalation in escalations[:3]),
            ],
        ),
    ]


def _asserted_safety_chain(*, cycle_count: int, owners: list[str], retired_count: int) -> str:
    """State the parent-ordered safety cycle as the cause, from the actuator lines that record it.

    The generic wording offers a menu (a crash, a residency cycle, a dropped message) because usually
    nothing in the log says which. When the parent's pause/restore lines are present they give the
    subsystem that ordered the move, and a verdict dropped on a retired launch is the exact mechanism that
    turns such a move into a stranded job, so the two together are an attribution rather than a candidate.
    Each part is a whole sentence, so the text reads cleanly whichever parts the log supplies.
    """
    sentences: list[str] = []
    if cycle_count:
        quoted_owners = _clause_join([f'"{owner}"' for owner in owners])
        by_whom = f", ordered by {quoted_owners}" if owners else ""
        sentences.append(
            f"The worker moved the safety process off the GPU or back onto it {_times_text(cycle_count)} this "
            f"session{by_whom}.",
        )
    if retired_count:
        sentences.append(
            f"{_count_text(retired_count, 'safety result')} arrived from a process such a move had already "
            f"replaced, and {'was' if retired_count == 1 else 'were'} dropped.",
        )
    sentences.append(
        "Each move replaces the safety process, so the checks it had in progress never return a result.",
    )
    return " ".join(sentences)


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
        detail_bits.append(f"{_count_text(len(lost_results), 'safety result')} never returned")
    if requeues:
        detail_bits.append(f"{_count_text(len(requeues), 'job')} checked again")
    if unrecoverable:
        detail_bits.append(f"{_count_text(len(unrecoverable), 'job')} failed with no image")
    if soft_pauses:
        detail_bits.append(
            f"{_count_text(len(soft_pauses), 'pause')} on asking for jobs after a check failed "
            '("Soft-pausing job pops")'
        )
    if backpressure:
        detail_bits.append(
            f"{_count_text(len(backpressure), 'hold')} on asking for jobs while checks were behind "
            '("Withholding job pops")'
        )
    detail = ", ".join(detail_bits)

    placement_cycles = _matching(records, _SAFETY_PLACEMENT_CYCLE_RE)
    retired_drops = _matching(records, _RETIRED_SAFETY_RESULT_RE)
    owners = _distinct_ordered(
        [m.group("owner").strip() for r in placement_cycles if (m := _SAFETY_PLACEMENT_CYCLE_RE.search(r.message))],
    )

    narrative: list[str] = [f"Counts: {detail}."]
    if escalated:
        verdict = (
            f"The safety check lost results and {_count_text(len(unrecoverable), 'job')} "
            f"{'was' if len(unrecoverable) == 1 else 'were'} dropped with no image, which the horde counts against "
            "the worker."
        )
        if placement_cycles or retired_drops:
            # The parent's own actuator is in the log, so the chain is readable end to end and there is no
            # reason to hand back a list of candidate causes.
            narrative.append(
                _asserted_safety_chain(
                    cycle_count=len(placement_cycles),
                    owners=owners,
                    retired_count=len(retired_drops),
                )
            )
            remediation = (
                "The evidence shows which part of the worker keeps moving the safety check off the GPU, "
                "and a result in progress is lost each time. Turn off `unload_models_from_vram_often` or "
                "whole-card reservations to reduce how often, and report it."
            )
        else:
            remediation = (
                "With `safety_on_gpu` on, frequent whole-card reservations or `unload_models_from_vram_often` "
                "can restart the safety process. Check the safety process log for crashes."
            )
    else:
        verdict = "The safety check backed up or lost a result, and the worker recovered without dropping jobs."
        remediation = (
            "If it recurs under load, turn on `safety_on_gpu` or reduce post-processing so the backlog does "
            "not grow. Nothing to do for a one-off."
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
            kind=FindingKind.SAFETY_STAGE_STALL,
            severity=severity,
            headline=verdict,
            action_addendum=remediation,
            evidence=narrative + [_evidence(r) for r in evidence[:4]],
            see_also=FindingKind.FORCED_MAINTENANCE if escalated else None,
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
            kind=FindingKind.WHOLE_CARD_CONVERGENCE_WEDGE,
            severity=Severity.CRITICAL,
            headline=(
                f"A model that needs the whole card was refused {_times_text(len(wedges))} because something "
                "never left the card to make room."
            ),
            evidence=[_evidence(r) for r in wedges[:3]],
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
            kind=FindingKind.WHOLE_CARD_NONHEAD_RESIDENCY_STARVATION,
            severity=Severity.CRITICAL if escalated else Severity.WARNING,
            headline=(
                f"The next job was held back {_times_text(len(starvations))} while the card was reserved for a later "
                "job"
                + (
                    f", then processes were rebuilt and {_count_text(dropped, 'job')} dropped."
                    if escalated
                    else ", and the stall cleared on its own."
                )
            ),
            evidence=[_evidence(r) for r in (starvations[:2] + soft_resets[:1])],
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
        return f"None of the {_count_text(count, 'reservation')} logged its image process counts."
    counts = _distinct_ordered([claim.counts_text for claim in claims])
    text = (
        f"Image process counts logged by {_count_text(len(claims), 'reservation')}, as running -> after of the "
        f"maximum, then the target: {_clause_join(counts)}."
    )
    frees = [claim.free_mb for claim in claims if claim.free_mb is not None]
    if frees:
        text += (
            f" Free VRAM on the card at reservation ranged from {min(frees)} to {max(frees)}MB."
            if min(frees) != max(frees)
            else f" Free VRAM on the card at reservation was {frees[0]}MB."
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
        """The cause figures, worded so the marginal's provenance is visible rather than assumed.

        The log's field name follows the one figure a reader may want to search for.
        """
        parts: list[str] = []
        if self.marginal_mb is not None and self.marginal_source is not None:
            provenance = (
                "a built-in estimate, since none was measured"
                if self.marginal_is_seeded
                else f"measured, source {self.marginal_source}"
            )
            parts.append(f"charged {self.marginal_mb:.0f}MB for each extra image process ({provenance})")
        elif self.marginal_source is not None:
            parts.append(f"had no figure for what each extra image process costs (source {self.marginal_source})")
        if self.unreclaimable_mb is not None:
            parts.append(f"counted {self.unreclaimable_mb:.0f}MB as memory that stopping processes cannot free")
        if self.needs_reduction is not None:
            judged = (
                "found that idle image processes had to be stopped for the model to fit"
                if self.needs_reduction
                else "did not call for idle image processes to be stopped"
            )
            parts.append(f"{judged} (needs_process_count_reduction={self.needs_reduction})")
        if not parts:
            return ""
        return f'The memory forecast for {self.model} ("Stream forecast" in the log) {_clause_join(parts)}.'


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

    outcome = (
        f", until the worker rebuilt its processes {_times_text(len(soft_resets))} and dropped "
        f"{_count_text(dropped, 'job')}"
        if escalated
        else ""
    )
    figures = [_claim_figures_text(claims, count=len(establishes))]
    if all_incoherent:
        figures.append(
            "Every reservation that logged its counts asked for at least as many image processes as were already "
            f"running, so it stopped none. One of them logged {incoherent[0].counts_text}."
        )
    if forecast is not None:
        figures.append(forecast.describe())
    if declined:
        figures.append(
            f"The worker declined to reserve the card for {_count_text(len(declined), 'model')} whose memory "
            'figures it did not trust ("Declined a whole-card residency" in the log).'
        )
    return [
        Finding(
            kind=FindingKind.WHOLE_CARD_RESIDENCY_CHURN,
            severity=Severity.CRITICAL if escalated or all_incoherent else Severity.WARNING,
            headline=f"The whole card was reserved and released {len(establishes)} times this session{outcome}.",
            action_addendum=_churn_remediation(
                all_incoherent=all_incoherent,
                unmeasured_marginal=unmeasured_marginal,
                forecast=forecast,
            ),
            evidence=figures + [_evidence(r) for r in establishes[:3]],
        ),
    ]


def _churn_remediation(
    *,
    all_incoherent: bool,
    unmeasured_marginal: bool,
    forecast: _StreamForecast | None,
) -> str:
    """The "Do this" line keyed to what the parsed figures show, rather than to one assumed cause.

    Blaming an unmeasured per-additional-context marginal is only true when the forecast says the marginal
    is seeded (nothing measurable); with a probe or idle-floor source it is measured, and asserting
    otherwise sends an operator after a number that is already correct.
    """
    if all_incoherent:
        unreclaimable = (
            f" The worker counts {forecast.unreclaimable_mb:.0f} MB as memory that stopping processes cannot free."
            if forecast is not None and forecast.unreclaimable_mb is not None
            else ""
        )
        return (
            "Report it. The reservation could not free anything, since the process count it asked for was "
            f"already met.{unreclaimable}"
        )
    if unmeasured_marginal:
        return (
            "Let the worker measure what each extra process costs. Run the start-up probe, or leave the "
            "worker idle with every process up so it can read a clean baseline. Until then it charges a "
            "guessed figure per process and gives the card to models that do not need it."
        )
    return (
        "Check that the models being given the whole card need it on this card. If they do not, "
        "report it with the log around a reservation."
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
        per_release[episode.release or "still in force at the session end"] = (
            per_release.get(episode.release or "still in force at the session end", 0) + 1
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
        f" The worker asked for one model only for {sum(durations):.0f}s in total, {max(durations):.0f}s at the "
        "longest."
        if durations
        else ""
    )
    return [
        Finding(
            kind=FindingKind.WHOLE_CARD_POP_CLAIM_EPISODES,
            severity=Severity.INFO,
            headline=(
                f"The worker asked the horde for one model only {_times_text(len(episodes))} this session while "
                "that model held the card."
            ),
            evidence=[f"By model: {model_breakdown}.", f"How each ended: {release_breakdown}.{held_text}"]
            + [_evidence(episode.engaged) for episode in episodes[:3]],
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
            kind=FindingKind.WHOLE_CARD_POP_CLAIM_MONOPOLY,
            severity=Severity.WARNING,
            headline=(
                f"{len(squeezing)} times the worker kept asking for one model until the hold cap stopped it, "
                "while jobs for other models waited."
            ),
            evidence=[
                f"The one model asked for: {', '.join(claimants)}.",
                f"Times a job for another model was logged as held back meanwhile: {starved_breakdown}.",
            ]
            + [_evidence(episode.engaged) for episode, _parked in squeezing[:3]],
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
                kind=FindingKind.HEAD_DISPATCH_STALL,
                severity=Severity.CRITICAL,
                title_override="The next job did not start although nothing was holding it",
                headline=(
                    f"The next job did not start {_times_text(len(bug_stalls))} although its model was loaded on "
                    "an idle process and nothing was holding it back."
                ),
                action_addendum=(
                    "Capture the log around the stall and report it. As a stopgap, turn off "
                    "`unload_models_from_vram_often` so a model is not unloaded just before its job starts."
                ),
                evidence=[_evidence(r) for r in bug_stalls[:3]],
                see_also=FindingKind.SCHEDULER_STARVATION_WEDGE,
            ),
        ]
    ram = _ram_pressure_in_window(context, stalls)
    return [
        Finding(
            kind=FindingKind.HEAD_DISPATCH_STALL,
            severity=Severity.WARNING,
            headline=(
                f"The next job was held back {_times_text(len(stalls))}, {'each time ' if len(stalls) > 1 else ''}by "
                "a rule the log reports."
            ),
            action_addendum=(
                "If throughput is low, change the setting behind the rule in the evidence, which is "
                "`max_threads`, `max_batch` or the VRAM budget."
            )
            if not ram
            else "System RAM pressure held jobs or model loads during this wait. Check the RAM finding.",
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
            kind=FindingKind.RESIDENCY_RECONCILIATION_HOLDS,
            severity=Severity.WARNING if escalated else Severity.INFO,
            headline=(
                f"The worker paused {_times_text(len(holds))} to unload an idle model before the next job"
                + (
                    f", with {_count_text(len(wedge_lines), 'stall or recovery line')} logged in that time."
                    if wedged
                    else f", about {holds_per_hour:.0f} an hour and {parked_seconds_total:.0f} seconds in total."
                )
            ),
            action_addendum=(
                "Something held the card that no job finishing gives back. Capture the log around the pause "
                "and report it."
                if wedged
                else (
                    "Serve fewer models at once, or lower `max_threads` or `queue_size`, so the card is not "
                    "packed past what fits at peak."
                    if escalated
                    else "No action needed at this volume."
                )
            ),
            evidence=[
                f"Pauses by model: {model_breakdown}."
                + (f" The pauses took {parked_fraction:.0%} of the session." if duration else ""),
            ]
            + (
                [
                    f"About {holds_per_hour:.0f} pauses an hour and {parked_seconds_total:.0f} seconds in total.",
                    *(_evidence(record) for record in wedge_lines[:2]),
                ]
                if wedged
                else []
            )
            + [_evidence(r) for r in holds[:3]],
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
    top_name, top_seconds = dominant[0]
    phrases = [
        f"{GOVERNOR_PLAIN_PHRASES.get(name, name)} ({name} in the log): {seconds / duration * 100:.0f}% of the "
        f"session, {seconds / 60:.1f} min"
        for name, seconds in dominant
    ]
    return [
        Finding(
            kind=FindingKind.POP_GOVERNOR_DOMINANCE,
            severity=Severity.INFO,
            headline=(
                f"{GOVERNOR_PLAIN_PHRASES.get(top_name, top_name)} held the worker back for "
                f"{top_seconds / duration * 100:.0f}% of the session, {top_seconds / 60:.1f} minutes."
            ),
            evidence=phrases + [_evidence(r) for r in _matching(session.records, GOVERNOR_ENTER_RE)[:3]],
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


_FAULT_CAUSE_NO_MODEL_NAME = "job offer with no model name"
_FAULT_CAUSE_STAGE = "a stage of the split pipeline failed (disaggregation)"
_FAULT_CAUSE_SAFETY = "the safety check could not be completed"
_FAULT_CAUSE_PROCESS = "fault in the image process"
_FAULT_CAUSE_GAVE_UP = "dropped when the worker gave up on its queue"
_FAULT_CAUSE_OTHER = "other"
"""The cause labels the failed-jobs evidence prints. Display text only, never compared."""


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
        return _FAULT_CAUSE_GAVE_UP
    return _FAULT_CAUSE_OTHER


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
            id_anchored[malformed.group("job")] = _FAULT_CAUSE_NO_MODEL_NAME
    for record in child + parent:
        stage = _STAGE_FAULT_RE.search(record.message)
        if stage is not None:
            id_anchored.setdefault(stage.group("job"), _FAULT_CAUSE_STAGE)
        safety = _SAFETY_UNRECOVERABLE_JOB_RE.search(record.message)
        if safety is not None:
            id_anchored[safety.group("job")] = _FAULT_CAUSE_SAFETY

    faults: dict[str, _FaultedJob] = {}
    for record in parent:
        slot_fault = _FAULTED_ON_PROCESS_JOB_RE.search(record.message)
        requeued = slot_fault is not None and _FAULT_REQUEUED_RE.search(record.message) is not None
        if slot_fault is not None:
            job_id = slot_fault.group("job")
            model = _FAULT_MODEL_RE.search(record.message)
            if not requeued:
                id_anchored.setdefault(job_id, _FAULT_CAUSE_PROCESS)
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
    breakdown = ", ".join(f"{cause} ({count})" for cause, count in sorted(counts.items(), key=lambda kv: -kv[1]))
    models = _distinct_ordered([fault.model for fault in faults if fault.model is not None])
    return [
        Finding(
            kind=FindingKind.FAULTED_JOB_CENSUS,
            severity=Severity.WARNING,
            headline=f"{_count_text(len(faults), 'job')} failed this session.",
            evidence=[f"By cause: {breakdown}."]
            + ([f"Models: {_clause_join(models)}."] if models else [])
            + [fault.describe() for fault in faults[:8]],
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
            kind=FindingKind.POP_LIVENESS_FULL_QUEUE,
            severity=Severity.CRITICAL,
            headline=(
                f"The queue was full and nothing moved {_times_text(len(frozen))}, for as long as {worst} seconds"
                + (f', with "{_clause_join(_distinct_ordered(heads))}" next in line.' if heads else ".")
            ),
            action_addendum=(
                "The card was reserved for one model while the next job needed another. Report it with the "
                "log around the freeze."
                if residency_held
                else "Something the card carries between jobs held it. The usual holders are an idle helper "
                "process, a process parked on an unused load, or weights a process said it freed and did not. "
                "Report it with the log around the freeze."
            ),
            evidence=(
                [f"The card was reserved for {_clause_join(held_models)} across the freeze."]
                if residency_held and held_models
                else []
            )
            + [_evidence(record) for record in frozen[:3]],
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
    quoted_categories = [f'"{category}"' for category in categories]
    return [
        Finding(
            kind=FindingKind.MODEL_REFERENCE_SAMPLE_FAULT,
            severity=Severity.WARNING,
            headline=(
                f"{_count_text(len(faults), 'job')} failed because the {_clause_join(quoted_categories)} model list "
                f"could not be read, affecting {'job' if len(jobs[:4]) == 1 else 'jobs'} "
                f"{_clause_join([job[:8] for job in jobs[:4]])}."
            ),
            evidence=(
                [
                    f"The log has {_count_text(len(stale), 'stale-cache line')} within a minute of the failures, so "
                    "the model list was being refreshed at the time."
                ]
                if stale
                else []
            )
            + [_evidence(record) for record in faults[:3]],
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
        blocked = (
            f", and block the blank name {_times_text(len(blank_quarantines))}"
            if blank_quarantines or quarantine_skips
            else ""
        )
        verdict = (
            f"{_count_text(len(empty_pops), 'blank job offer')} made the worker load a nameless model "
            f"{_times_text(len(blank_preloads))}, crash "
            f"{_count_text(len(blank_deaths), 'image process', 'image processes')}{blocked}."
        )
        remediation = (
            "Update the worker. Newer versions hand a blank offer straight back without loading anything or "
            "crashing a process. Your models are not at fault."
        )
        severity = Severity.CRITICAL
        evidence = (
            empty_pops[:1] + blank_preloads[:1] + blank_quarantines[:1] + quarantine_skips[:1],
            [r.record for r in blank_deaths[:1]],
        )
        evidence_lines = [_evidence(r) for r in evidence[0]] + [_evidence(r) for r in evidence[1]]
    else:
        verdict = (
            f"{_count_text(len(rejected), 'job offer')} arrived with no model name and the worker handed "
            f"{'it' if len(rejected) == 1 else 'each one'} back without loading anything or losing a process."
        )
        remediation = "Each refused offer is a job this worker could not serve, so watch the rate."
        severity = Severity.WARNING
        evidence_lines = [_evidence(r) for r in (rejected[:2] + refused_preloads[:1] + refused_incidents[:1])]

    return [
        Finding(
            kind=FindingKind.EMPTY_MODEL_POP_CASCADE,
            severity=severity,
            headline=verdict,
            action_addendum=remediation,
            evidence=evidence_lines,
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
            kind=FindingKind.PRELOAD_KILLS_CHILD_LOOP,
            severity=Severity.CRITICAL,
            headline=(
                f'Loading "{model}" crashed an image process {len(deaths)} times across '
                f"{'process' if len(slots) == 1 else 'processes'} {_clause_join([str(slot) for slot in slots])}"
                + (f', and "{_clause_join(others)}" did the same.' if others else ".")
            ),
            evidence=[_evidence(recovery.record) for recovery in deaths[:3]],
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
    span_text = f" over {span / 60:.0f} minutes" if span is not None and span >= 60 else ""
    operator_fixable = bool(_OPERATOR_FIXABLE_POP_ERRORS.search(message))
    others = len(by_message) - 1

    return [
        Finding(
            kind=FindingKind.POP_API_ERROR_DOMINANCE,
            severity=Severity.WARNING,
            headline=(
                f"The horde refused {len(occurrences)} requests for work{span_text}, always with the same message."
            ),
            action_addendum=(
                (
                    "This will not clear on its own, since it is about the account or the worker registration. "
                    "Do what the message says about the worker count or name, the API key, or the maintenance "
                    "flag. Then check that jobs resume."
                )
                if operator_fixable
                else (
                    "This reads as a transient server-side or rate-limit refusal and normally clears by itself. "
                    "The worker already backs off and retries. If it continues across restarts, check the "
                    "horde's status before changing anything locally."
                )
            ),
            evidence=[f'The message: "{message}".']
            + ([f"The horde's error code was {code}."] if code else [])
            + ([f"{_count_text(others, 'other refusal message')} also occurred."] if others > 0 else [])
            + [_evidence(record) for record in (occurrences[:1] + occurrences[-1:])],
        ),
    ]


_SUBMITTED_GENERATION_RE = pattern_for("submitted_generation")

# A model with fewer jobs than this has too few samples for its per-second rates to rank it against the rest.
_MODEL_ECONOMICS_MIN_JOBS = 20
_MODEL_ECONOMICS_TOP_MODELS = 8
# A well-sampled model earning under this fraction of the session's kudos per wall-second is listed even when
# it is outside the top earners, since a cheap model holding much of the wall clock is what the table is for.
_MODEL_ECONOMICS_CHEAP_FRACTION = 0.5


@dataclass
class _ModelEconomics:
    """One model's submitted jobs summed over a session: kudos, sampling seconds, and popped-to-submit seconds."""

    model: str
    jobs: int = 0
    kudos: float = 0.0
    sampling_seconds: float = 0.0
    wall_seconds: float = 0.0

    @property
    def kudos_per_sampling_second(self) -> float | None:
        return self.kudos / self.sampling_seconds if self.sampling_seconds > 0 else None

    @property
    def kudos_per_wall_second(self) -> float | None:
        return self.kudos / self.wall_seconds if self.wall_seconds > 0 else None


def _rate_text(rate: float | None) -> str:
    return f"{rate:.2f}" if rate is not None else "no time recorded"


def detect_model_economics(context: SessionContext) -> list[Finding]:
    """Kudos per sampling-second and per wall-second, per model, from each submitted job's line.

    Wall is popped-to-submit, so it counts every second a job held a place in the worker, including the waits
    around sampling. The kudos price is the horde's, so the rates rank models by what they pay; the share of
    wall a model takes is the worker's scheduling. Always informational: it is a table, not a problem.
    """
    records = _records_in_window(context.session.records, context.session.start_ts, context.session.end_ts)
    by_model: dict[str, _ModelEconomics] = {}
    for record in records:
        if not (match := _SUBMITTED_GENERATION_RE.search(record.message)):
            continue
        economics = by_model.setdefault(match["model"], _ModelEconomics(model=match["model"]))
        economics.jobs += 1
        economics.kudos += float(match["kudos"])
        economics.sampling_seconds += float(match["generate"])
        economics.wall_seconds += float(match["popped_ago"])
    if not by_model:
        return []

    models = list(by_model.values())
    total_kudos = sum(economics.kudos for economics in models)
    total_sampling = sum(economics.sampling_seconds for economics in models)
    total_wall = sum(economics.wall_seconds for economics in models)
    session_per_sampling = total_kudos / total_sampling if total_sampling > 0 else None
    session_per_wall = total_kudos / total_wall if total_wall > 0 else None

    well_sampled = [
        economics
        for economics in models
        if economics.jobs >= _MODEL_ECONOMICS_MIN_JOBS and economics.kudos_per_wall_second is not None
    ]
    headline = (
        f"Jobs earned {total_kudos:.0f} kudos, {_rate_text(session_per_sampling)} per sampling second and "
        f"{_rate_text(session_per_wall)} per wall second"
    )
    if well_sampled and total_wall > 0:
        lowest = min(well_sampled, key=lambda economics: economics.kudos_per_wall_second or 0.0)
        headline += (
            f', and the lowest payer, "{lowest.model}", took {100 * lowest.wall_seconds / total_wall:.1f}% of '
            "wall time."
        )
    else:
        headline += "."

    by_kudos = sorted(models, key=lambda economics: -economics.kudos)
    listed = by_kudos[:_MODEL_ECONOMICS_TOP_MODELS]
    if session_per_wall is not None:
        cheap_floor = _MODEL_ECONOMICS_CHEAP_FRACTION * session_per_wall
        well_sampled_models = {economics.model for economics in well_sampled}
        listed += [
            economics
            for economics in by_kudos[_MODEL_ECONOMICS_TOP_MODELS:]
            if economics.model in well_sampled_models and (economics.kudos_per_wall_second or 0.0) < cheap_floor
        ]

    evidence: list[str] = []
    for economics in listed:
        kudos_share = 100 * economics.kudos / total_kudos if total_kudos else 0.0
        wall_share = 100 * economics.wall_seconds / total_wall if total_wall > 0 else 0.0
        jobs_text = "1 job" if economics.jobs == 1 else f"{economics.jobs} jobs"
        evidence.append(
            f"{economics.model}: {jobs_text}, {economics.kudos:.0f} kudos ({kudos_share:.1f}%), "
            f"{_rate_text(economics.kudos_per_sampling_second)} per sampling-second, "
            f"{_rate_text(economics.kudos_per_wall_second)} per wall-second, {wall_share:.1f}% of wall time"
        )
    return [
        Finding(
            kind=FindingKind.MODEL_ECONOMICS,
            severity=Severity.INFO,
            headline=headline,
            evidence=evidence,
        ),
    ]


_SESSION_END_PHRASES: dict[SessionEndReason, str] = {
    SessionEndReason.CLEAN_EXIT: "ended with a normal shutdown",
    SessionEndReason.GAVE_UP_ABORTED: "stopped itself after its processes could not be restored",
    SessionEndReason.ABORTED: "was stopped by the abort file",
    SessionEndReason.SUPERVISOR_SHUTDOWN: "was shut down from the dashboard",
    SessionEndReason.CONTROL_LOOP_CRASH: "shut itself down after an error in its control loop",
    SessionEndReason.KILLED_OR_CRASHED: "was killed or crashed without shutting down",
    SessionEndReason.STILL_RUNNING: "is still running",
}
"""How each end reason reads in the summary's headline."""


def detect_session_summary(context: SessionContext) -> list[Finding]:
    """An always-present rollup: how the session ended and its recovery/fault headline numbers."""
    session = context.session
    duration = session.duration_seconds
    duration_text = f"{duration / 60:.1f} minutes" if duration is not None else "an unknown time"
    severity = (
        Severity.WARNING
        if session.end_reason in (SessionEndReason.KILLED_OR_CRASHED, SessionEndReason.CONTROL_LOOP_CRASH)
        else Severity.INFO
    )
    return [
        Finding(
            kind=FindingKind.SESSION_SUMMARY,
            severity=severity,
            headline=f"The session ran for {duration_text} and {_SESSION_END_PHRASES[session.end_reason]}.",
            evidence=[
                f"The process restart count reached {session.peak_process_recoveries}, and the log has "
                f"{_count_text(len(context.recoveries), 'restart report')}.",
                f"Version v{session.version or '?'}, {session.num_models or '?'} models, "
                f"{_count_text(session.max_threads, 'thread') if session.max_threads else '? threads'}.",
            ]
            + ([f"hordelib {session.hordelib}."] if session.hordelib else [])
            # A parent log rotates by size mid-run, so which archives were read decides the span every
            # figure above is measured over. Naming them keeps that attributable.
            + (
                [f"Rotation: {context.bundle.rotation_stitch.describe()}."]
                if context.bundle.rotation_stitch is not None
                else []
            ),
        ),
    ]


def _snapshot_evidence(lifecycle: JobLifecycleModel, *, count: int = 2) -> list[str]:
    """A few status prints rendered as one line each: busy/idle lane tally and queue depth."""
    snapshots = [snapshot for snapshot in lifecycle.status_snapshots if snapshot.inference_lanes]
    if not snapshots:
        return []
    chosen = [snapshots[len(snapshots) // 3], snapshots[2 * len(snapshots) // 3]][:count]
    placements = lifecycle.current_placements()
    lines: list[str] = []
    for snapshot in chosen:
        lanes = snapshot.inference_lanes
        sampling = [lane for lane in lanes if lane.sampling]
        cards = {
            placements[lane.process_id].device_index
            for lane in sampling
            if lane.process_id in placements and placements[lane.process_id].device_index is not None
        }
        stamp = snapshot.timestamp.strftime("%H:%M:%S") if snapshot.timestamp else "--:--:--"
        lines.append(
            f"{stamp}  status: {len(sampling)} of {_count_text(len(lanes), 'image process', 'image processes')} "
            f"sampling on {_count_text(len(cards), 'card')}, {sum(1 for lane in lanes if lane.idle)} idle, "
            f"{_count_text(len(snapshot.pending_jobs), 'job')} queued",
        )
    return lines


def detect_multi_card_dispatch_serialization(context: SessionContext) -> list[Finding]:
    """A multi-card worker running its queue through a handful of cards while the rest sit idle.

    On a host driving several cards the queue should spread across them: the intake budget is the sum of
    the per-card budgets precisely so every card can hold work. This detector measures the parallelism the
    fleet actually achieved (time-weighted count of distinct cards holding an in-flight job) against the
    card count, and fires only when that shortfall cannot be explained by there being nothing to run:
    the pending queue has to have sat near the worker's own intake budget, with idle lanes and pending
    jobs whose model was already resident on one of them.

    Every threshold is a fraction of the session's own card count and intake budget, so the same rule
    applies to a two-card host and a sixteen-card one. It cannot fire on a single-card worker, where
    "one card busy" is full utilisation rather than a shortfall.
    """
    lifecycle = job_lifecycle_for(context)
    card_count = lifecycle.card_count
    if card_count is None or card_count < 2:
        return []

    profile = lifecycle.sampling_concurrency()
    mean_busy = profile.mean_cards_busy
    measured = sum(
        1
        for job in lifecycle.jobs.values()
        if job.dispatched_at is not None and job.inference_finished_at is not None and job.device_index is not None
    )
    if mean_busy is None or measured < _MULTI_CARD_MIN_MEASURED_JOBS:
        return []
    if mean_busy > card_count * _MULTI_CARD_CONCURRENCY_WARNING_FRACTION:
        return []

    census = lifecycle.status_census()
    budget = lifecycle.intake_budget if lifecycle.intake_budget else card_count
    queue_stayed_full = census.median_pending >= budget * _MULTI_CARD_QUEUE_FULL_FRACTION
    seats_were_free = census.median_idle_lanes >= 1 and census.median_seatable_pending >= 1
    if not (queue_stayed_full and seats_were_free):
        return []

    pre_segment, inference_segment, post_segment = lifecycle.wait_segments()
    drain = lifecycle.ipc_drain_profile()
    parent_healthy = drain.drains >= _PARENT_LOOP_MIN_DRAINS and not drain.gaps_over_threshold
    if parent_healthy:
        loop_clause = (
            f" The main process kept reading its workers' messages throughout, with a longest gap of "
            f"{drain.max_gap_seconds:.0f}s against a {drain.threshold_seconds:.0f}s limit. It was not stalled, so "
            "the idle cards come from a scheduling decision."
        )
    elif drain.gaps_over_threshold:
        loop_clause = (
            " The main process also stopped reading its workers' messages at times. See the parent_loop_stall finding."
        )
    else:
        loop_clause = (
            ' This log has no "Received ... from process" lines, so it cannot show whether the main process stalled.'
        )

    heads = census.head_states
    head_clause = ""
    if heads.total:
        head_clause = (
            f" In {100 * heads.resident_but_blocked / heads.total:.0f}% of status prints the next job's model was "
            f"loaded only on processes whose card was busy. In {100 * heads.cold / heads.total:.0f}% it was loaded "
            "nowhere."
        )

    severity = (
        Severity.CRITICAL if mean_busy <= card_count * _MULTI_CARD_CONCURRENCY_CRITICAL_FRACTION else Severity.WARNING
    )
    evidence = [
        f"share of time by cards sampling at once: {profile.describe(top=card_count + 1)} "
        f"(time-weighted mean {mean_busy:.1f} of {card_count} cards over {profile.observed_seconds / 60:.0f} min)",
        f"where a job's time went: a median {pre_segment.median or 0:.0f}s from accepting it to starting it "
        f"(p90 {pre_segment.p90 or 0:.0f}s), {inference_segment.median or 0:.0f}s generating, and "
        f"{post_segment.median or 0:.0f}s from finishing to submitting, over {_count_text(pre_segment.count, 'job')}",
    ]
    skips = lifecycle.line_skip_census()
    if skips:
        by_reason = ", ".join(f"{reason} {count}" for reason, count in sorted(skips.items()))
        evidence.append(
            f"jobs started ahead of the next in line to fill an idle process ({sum(skips.values())} in total, "
            f'"skipped the line" in the log), by reason: {by_reason}',
        )
    else:
        evidence.append("No job was started ahead of the next in line to fill an idle process.")
    evidence.extend(_snapshot_evidence(lifecycle))

    return [
        Finding(
            kind=FindingKind.MULTI_CARD_DISPATCH_SERIALIZATION,
            severity=severity,
            headline=(
                f"Only {mean_busy:.1f} of the {card_count} cards were busy on average while jobs waited a "
                f"median {pre_segment.median or 0:.0f} seconds to start against {inference_segment.median or 0:.0f} "
                "seconds of generation."
            ),
            evidence=[
                f"Each status print showed a median {census.median_pending:.0f} jobs waiting and "
                f"{census.median_idle_lanes:.0f} idle processes, and the worker holds up to {budget} jobs at once "
                f"(its intake budget). A median {census.median_seatable_pending:.0f} of "
                "the waiting jobs already had their model loaded on an idle process whose card was running "
                f"nothing, and at most {census.max_seatable_pending}.{head_clause}{loop_clause}",
            ]
            + evidence,
        ),
    ]


def _ram_pressure_in_window(context: SessionContext, records: list[LogRecord]) -> list[LogRecord]:
    """RAM holds and deferred loads in the interval covered by these records."""
    if not records:
        return []
    start, end = _window_key(records[0]), _window_key(records[-1])
    return [
        record
        for record in context.session.records
        if start <= _window_key(record) <= end
        and (
            _RAM_DEFER_RE.search(record.message)
            or (match := _RAM_HOLD_RE.search(record.message)) is not None
            and match.group("edge") == "engaged"
        )
    ]


def _bundle_ram_requirements(context: SessionContext, charge_mb: float) -> str:
    """Describe requirements from bundle config and host inventory, without exposing config contents."""
    from horde_worker_regen.process_management.resources.resource_budget import ram_headroom

    root = context.bundle.root
    # A directory bundle may have its files under logs/, with metadata beside that directory.
    if not (root / "system_info.json").is_file() and (root.parent / "system_info.json").is_file():
        root = root.parent
    try:
        inventory = json.loads((root / "system_info.json").read_text(encoding="utf-8"))
        config_path = root / "config" / "bridgeData.redacted.yaml"
        if not config_path.is_file():
            config_path = root / "bridgeData.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not isinstance(inventory, dict):
            return (
                "The worker's RAM limits could not be worked out, because the bundle's system or config file is "
                "not in the expected form."
            )
        memory = inventory.get("ram", {})
        total = float(memory["total_bytes"]) / (1024 * 1024)
        available = float(memory["available_bytes"]) / (1024 * 1024)
        thresholds = ram_headroom(
            total,
            pause_percent=float(config.get("ram_pressure_pause_percent", 85)),
            min_free_mb=float(config.get("ram_pressure_min_free_mb", 1024)),
            reserve_mb=float(config.get("ram_reserve_mb", 4096)),
            staging_charge_mb=charge_mb,
            context_cost_mb=charge_mb,
        )
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as error:
        return (
            "The worker's RAM limits could not be worked out, because reading the bundle's system or config file "
            f"failed with {type(error).__name__}."
        )
    return (
        "Available system RAM the worker needs, from this bundle's config and hardware. It pauses intake below "
        f"{thresholds.hard_floor_mb:.0f} MB (danger floor). It holds new jobs below {thresholds.soft_hold_mb:.0f} MB "
        "plus what running jobs still need (soft hold). It needs "
        f"{thresholds.preload_requirement_mb:.0f} MB to load a model (preload) and at least "
        f"{thresholds.restore_requirement_mb:.0f} MB to bring a process back (restore), with a model load taken as "
        f"{charge_mb:.0f} MB, the median of the postponed loads in the log. The bundle recorded {available:.0f} MB "
        "available."
    )


def detect_host_ram_starvation(context: SessionContext) -> list[Finding]:
    """Correlate host holds, marginal RAM refusals and expensive process reclaim in one finding."""
    records = context.session.records
    holds = [
        (record, match)
        for record in records
        if (match := _RAM_HOLD_RE.search(record.message)) is not None and match.group("edge") == "engaged"
    ]
    defers = [(record, match) for record in records if (match := _RAM_DEFER_RE.search(record.message)) is not None]
    reclaims = [(record, match) for record in records if (match := _RAM_CEILING_RE.search(record.message)) is not None]
    creep = _matching(records, _RAM_CREEP_RE)
    if not holds and len(defers) < 3 and not reclaims and not creep:
        return []
    stamps = [record.timestamp for record in records if record.timestamp is not None]
    hours = (max(stamps) - min(stamps)).total_seconds() / 3600 if len(stamps) > 1 else 0.0
    engage_rate = len(holds) / hours if hours > 0 else 0.0
    drains = sum(match.group("action") == "draining" for _record, match in reclaims)
    recycles = sum(match.group("action") == "recycling" for _record, match in reclaims) + len(creep)
    readings = [float(match.group("available")) for _record, match in [*holds, *defers]]
    charge = statistics.median(float(match.group("charge")) for _record, match in defers) if defers else 0.0
    evidence = [
        _bundle_ram_requirements(context, charge),
        f"New jobs were held {_times_text(len(holds))}, {engage_rate:.1f} an hour. Model loads were postponed "
        f"{_times_text(len(defers))}. For holding too much RAM, {_count_text(drains, 'process', 'processes')} "
        f"{'was' if drains == 1 else 'were'} given no new work until restarted (draining) and "
        f"{_count_text(recycles, 'process', 'processes')} {'was' if recycles == 1 else 'were'} restarted "
        "(recycling or cycling).",
    ]
    if readings:
        evidence.append(
            f"available RAM when jobs were held or loads postponed: median {statistics.median(readings):.0f} MB"
        )
    if holds:
        evidence.append("The first hold line in the log: " + holds[0][0].message)
    # Sweep jobs rather than distinct cards: one card may run several inference processes.
    lifecycle = job_lifecycle_for(context)
    events: list[tuple[datetime, int, int]] = []
    for job in lifecycle.jobs.values():
        if job.popped_at is not None and job.dispatched_at is not None and job.inference_finished_at is not None:
            events.extend([(job.popped_at, 0, 1), (job.dispatched_at, 1, -1), (job.inference_finished_at, -1, 0)])
    running = queued = 0
    one = busy_queued = 0.0
    previous = None
    for stamp, run_delta, queue_delta in sorted(events):
        if previous is not None and queued > 0 and running > 0:
            elapsed = (stamp - previous).total_seconds()
            busy_queued += elapsed
            if running == 1:
                one += elapsed
        running += run_delta
        queued += queue_delta
        previous = stamp
    if busy_queued > 0:
        evidence.append(
            f"With jobs queued and at least one running, only one was running for {one / busy_queued:.1%} of "
            f"{busy_queued:.0f}s."
        )
    evidence.extend(_evidence(record) for record, _match in defers[:2])
    return [
        Finding(
            kind=FindingKind.HOST_RAM_STARVATION,
            severity=Severity.WARNING,
            headline=(
                f"System RAM held new jobs {_times_text(len(holds))} and postponed model loads "
                f"{_times_text(len(defers))}, with {_count_text(recycles, 'process restart')}."
            ),
            evidence=evidence,
        )
    ]


def detect_model_churn(context: SessionContext) -> list[Finding]:
    """Model weights moving in and out of the lanes faster than the work justifies.

    A lane that keeps its model serves the next job for that model with no load at all, so in a settled
    worker preloads are far rarer than dispatches. Preloads approaching a third of dispatches mean the
    resident set turns over faster than the queue does, and every turn costs a load off disk plus the
    eviction it forced. Preloads cleared before they ever ran are the pure-waste form: weights fetched
    for a job the scheduler then seated elsewhere.

    Both thresholds are ratios against the session's own dispatch count, so the rule is independent of
    how long the session ran or how many lanes it had.
    """
    lifecycle = job_lifecycle_for(context)
    dispatches = lifecycle.dispatch_count
    if dispatches < _MODEL_CHURN_MIN_DISPATCHES:
        return []

    movement = lifecycle.model_movement
    preload_ratio = movement.preloads / dispatches
    unload_ratio = movement.unloads / dispatches
    cleared_ratio = movement.cleared_preloads / dispatches
    if preload_ratio < _MODEL_CHURN_PRELOAD_RATIO and cleared_ratio < _MODEL_CHURN_CLEARED_RATIO:
        return []

    ram = _ram_pressure_in_window(context, context.session.records)
    severity = Severity.CRITICAL if movement.preloads >= dispatches else Severity.WARNING
    return [
        Finding(
            kind=FindingKind.MODEL_CHURN,
            severity=severity,
            action_addendum="System RAM pressure also held jobs or model loads during these swaps." if ram else None,
            headline=(
                f"Over {dispatches} jobs the worker loaded {_count_text(movement.preloads, 'model')} and unloaded "
                f"{movement.unloads}, and {movement.cleared_preloads} of those loads "
                f"{'was thrown away before it' if movement.cleared_preloads == 1 else 'were thrown away before they'}"
                " ran."
            ),
            evidence=[
                f"loads {movement.preloads} ({preload_ratio:.2f} per job), unloads {movement.unloads} "
                f"({unload_ratio:.2f} per job), loads cleared before use {movement.cleared_preloads} "
                f"({cleared_ratio:.2f} per job), records dropped for a model another model had replaced "
                f'{movement.displaced_expiries} ("Expiring displaced entry"), models moved from VRAM to system RAM '
                f"{movement.moved_to_system_ram}, against {dispatches} jobs",
            ],
        ),
    ]


_AUXILIARY_ROLES = (LaneRole.POST_PROCESS, LaneRole.UTILITIES, LaneRole.SAFETY)
"""The lanes whose card placement is a scheduling choice rather than a fixed per-card pool.

An inference lane belongs to its card's pool by construction, so its re-spawn is not a migration; the
auxiliary lanes are placed once and can land somewhere else on a later spawn."""

_SAFETY_GPU_DEVICE_INDEX = 0
"""The card the safety lane occupies when it runs on the GPU. Its spawn line carries no device index, so
this is the only way to reason about what shares a card with it."""

_LANE_ROLE_NAMES: dict[LaneRole, str] = {
    LaneRole.INFERENCE: "image",
    LaneRole.SAFETY: "safety",
    LaneRole.POST_PROCESS: "post-processing",
    LaneRole.UTILITIES: "image utilities",
}
"""What the evidence calls each kind of process, in the words the findings use."""

_AUX_LANE_SAFETY_SHARING_RE = pattern_for("auxiliary_lane_safety_sharing")
_UTILITIES_BRINGUP_FAILED_RE = pattern_for("utilities_bringup_failed")
_UTILITIES_BRINGUP_HEALTHY_RE = pattern_for("utilities_bringup_healthy")
_UTILITIES_LANE_STARTED_RE = pattern_for("utilities_lane_started")
"""The worker's own edge-triggered disclosure that safety came back to a card a pinned lane holds.

It corroborates the stacking the occupancy map already shows, and it carries what the map cannot: the
moment the sharing began, which is a pause/restore of the safety process rather than a placement choice."""


def detect_lane_placement(context: SessionContext) -> list[Finding]:
    """Auxiliary lanes that moved cards mid-session, or that stacked onto the safety lane's card.

    The post-processing, utilities and safety lanes each hold VRAM on whichever card they land on, and
    unlike the inference lanes they are not per-card pools: they are placed once. A re-spawn that lands on
    a different card silently changes which card carries the auxiliary load, and stacking them onto card 0
    (where GPU safety runs) concentrates that load on the one card that is already carrying it, on a host
    with other cards free. Both are placement facts a per-card duty reading cannot explain on its own.
    """
    lifecycle = job_lifecycle_for(context)
    card_count = lifecycle.card_count
    migrations = [migration for migration in lifecycle.lane_migrations() if migration.role in _AUXILIARY_ROLES]
    occupancy = lifecycle.card_occupancy()

    stacked = [
        placement
        for placement in occupancy.get(_SAFETY_GPU_DEVICE_INDEX, [])
        if placement.role in _AUXILIARY_ROLES and placement.role is not LaneRole.SAFETY
    ]
    co_located = bool(stacked) and card_count is not None and card_count > 1
    if not migrations and not co_located:
        return []

    evidence: list[str] = []
    # The worker's own notice, where the session carries one: it dates the sharing to a safety
    # pause/restore rather than to where the lane was first placed. Corroboration only, since the
    # occupancy map above is what decides whether there is anything to report.
    sharing_notices = _matching(context.session.records, _AUX_LANE_SAFETY_SHARING_RE)
    evidence.extend(_evidence(record) for record in sharing_notices[:3])
    for migration in migrations:
        stamp = migration.timestamp.strftime("%H:%M:%S") if migration.timestamp else "--:--:--"
        evidence.append(
            f"{stamp}  the {_LANE_ROLE_NAMES[migration.role]} process (process {migration.process_id}) moved "
            f"from card {migration.from_device_index} to card {migration.to_device_index}",
        )
    for device_index in sorted(occupancy):
        lanes = ", ".join(
            f"process {placement.process_id} ({_LANE_ROLE_NAMES[placement.role]})"
            for placement in occupancy[device_index]
        )
        evidence.append(f"card {device_index} now holds: {lanes}")

    verdict_parts: list[str] = []
    if migrations:
        moved = ", ".join(
            f"{_LANE_ROLE_NAMES[migration.role]} (process {migration.process_id})" for migration in migrations
        )
        verdict_parts.append(
            f"{_count_text(len(migrations), 'helper process restart')} landed on a different card than before."
        )
        evidence.insert(0, f"Moved: {moved}.")
    if co_located:
        names = ", ".join(
            f"{_LANE_ROLE_NAMES[placement.role]} (process {placement.process_id})" for placement in stacked
        )
        verdict_parts.append(
            f"Card {_SAFETY_GPU_DEVICE_INDEX} carries "
            f"{_count_text(len(stacked), 'helper process', 'helper processes')} beside the safety check on "
            f"a host driving {card_count} cards."
        )
        evidence.insert(0, f"Stacked on card {_SAFETY_GPU_DEVICE_INDEX}: {names}.")
    return [
        Finding(
            kind=FindingKind.LANE_PLACEMENT,
            severity=Severity.WARNING,
            headline=verdict_parts[0],
            evidence=verdict_parts[1:] + evidence,
        ),
    ]


def detect_utilities_lane_bringup_timeout(context: SessionContext) -> list[Finding]:
    """The image-utilities lane's health wait expiring before its capability service could answer.

    The service must import its whole image stack before uvicorn binds a port, so a start against a cold
    file cache can outlast the launcher's fixed startup budget while it is still making progress. The
    launcher then kills it, throwing that loading away, and the lane restarts from nothing. The kills warm
    the file cache, so a later attempt usually wins the same race; the cost is the lane's features being
    unavailable throughout, and each replacement counting against the session's process recoveries.

    Reported from the parent log alone: the timeout line names neither what the service was loading nor how
    far it got, which live in the lane's own console file, so the evidence points the reader at it.
    """
    records = context.session.records
    timeouts = _matching(records, _UTILITIES_BRINGUP_FAILED_RE)
    if not timeouts:
        return []

    healthy = _matching(records, _UTILITIES_BRINGUP_HEALTHY_RE)
    starts = _matching(records, _UTILITIES_LANE_STARTED_RE)
    last_timeout = timeouts[-1].timestamp
    # The recovery is the first healthy report *after* the last expiry, not merely any healthy report: a
    # program that came up, was restarted later and then timed out has not recovered from that timeout.
    recovery = next(
        (
            record
            for record in healthy
            if record.timestamp is not None and last_timeout is not None and record.timestamp > last_timeout
        ),
        None,
    )
    recovered = recovery is not None or (bool(healthy) and last_timeout is None)

    # How long the features were unavailable: the first expiry until the program answered, or until the
    # last expiry when it never did. A later restart's own healthy report is outside that span.
    first_timeout = timeouts[0].timestamp
    span_end = recovery.timestamp if recovery is not None else last_timeout
    lost_seconds = (span_end - first_timeout).total_seconds() if first_timeout and span_end else 0.0
    lost_clause = f", over {lost_seconds / 60:.0f} minutes" if lost_seconds >= 90 else ""

    attempts = "once" if len(timeouts) == 1 else f"{len(timeouts)} times"
    if recovered:
        headline = f"Image utilities ran out of time starting {attempts} before it came up{lost_clause}."
    else:
        headline = f"Image utilities ran out of time starting {attempts} and never came up{lost_clause}."

    narrative = [f"Starts: {len(starts)}. Timed out: {len(timeouts)}. Reported healthy: {len(healthy)}."]
    if not recovered:
        narrative.append(
            "Nothing reported healthy after the last time it ran out, so the features it serves were "
            "unavailable for the rest of the session.",
        )
    narrative.append('The program\'s output is in the log file "bridge_utilities_<slot>.log".')

    evidence = timeouts[:2] + ([recovery] if recovery is not None else [])
    return [
        Finding(
            kind=FindingKind.UTILITIES_LANE_BRINGUP_TIMEOUT,
            severity=Severity.WARNING,
            headline=headline,
            evidence=narrative + [_evidence(record) for record in evidence],
            see_also=FindingKind.SESSION_SUMMARY,
        ),
    ]


def detect_parent_loop_stall(context: SessionContext) -> list[Finding]:
    """The parent's IPC drain going silent for longer than its own slowest routine clock allows.

    Every message a child sends is logged as it is drained, so a silence in that stream means the single
    asyncio loop that drains IPC, dispatches jobs and prints status was not running. The threshold is
    derived from the session's own status-print cadence rather than a constant: the status print rides the
    same loop, so a silence longer than one and a half of its periods cannot be an idle worker. A healthy
    loop produces no finding; findings are for problems, and the multi-card dispatch verdict states
    parent-loop health directly so a reader is not left wondering.
    """
    lifecycle = job_lifecycle_for(context)
    drain = lifecycle.ipc_drain_profile()
    if drain.drains < _PARENT_LOOP_MIN_DRAINS or not drain.gaps_over_threshold:
        return []

    worst = drain.gaps_over_threshold[0][1]
    stalled_seconds = sum(seconds for _, seconds in drain.gaps_over_threshold)
    severity = (
        Severity.CRITICAL
        if worst >= drain.threshold_seconds * _PARENT_LOOP_STALL_CRITICAL_MULTIPLIER
        else Severity.WARNING
    )
    cadence = lifecycle.status_cadence_seconds()
    cadence_clause = (
        f"comes from this session's {cadence:.0f}s interval between status prints"
        if cadence is not None
        else "is the default, because this session has no status prints to work one out from"
    )
    return [
        Finding(
            kind=FindingKind.PARENT_LOOP_STALL,
            severity=severity,
            headline=(
                f"The main process stopped reading its workers' messages "
                f"{_times_text(len(drain.gaps_over_threshold))} "
                f"for over {drain.threshold_seconds:.0f} seconds, the worst {worst:.0f} seconds and "
                f"{stalled_seconds:.0f} in total."
            ),
            evidence=[f"The {drain.threshold_seconds:.0f}s limit {cadence_clause}."]
            + [
                f"{timestamp.strftime('%H:%M:%S')}  the main process read a worker message again after "
                f"{seconds:.0f}s without reading any"
                for timestamp, seconds in drain.gaps_over_threshold[:5]
            ],
        ),
    ]


_SAFETY_CAPACITY_MIN_CHECKED_JOBS = 10
"""How many checked jobs a session needs before its check duration and wait medians mean anything.

Below this a single cold first check or one slow upload dominates every figure the detector reads."""

_SAFETY_CAPACITY_DEMAND_FRACTION = 0.5
"""The share of the checker's capacity the fleet's finish rate has to reach before a queue is expected.

Safety is a single server with no batching, so arrivals that are bursty (several cards finishing within
the same second, which is exactly what a spread fleet does) build a queue well before the mean arrival
rate reaches the service rate. Half of capacity is where that starts to show as a standing wait."""

_SAFETY_CAPACITY_WAIT_MARGIN = 2.0
"""How many check durations the observed wait must exceed before it is a queue rather than the check.

A job that arrives at an idle checker waits nothing and is done in one check duration; the margin keeps a
session whose waits are simply one check long from being read as a backlog."""


def detect_safety_stage_capacity(context: SessionContext) -> list[Finding]:
    """A multi-card fleet finishing inference faster than the one safety process can check the results.

    Safety is one process for the whole worker while inference scales with the cards, so the two stages
    scale differently: N cards at one job per generation time hand images to a checker whose throughput is
    fixed at one check per check duration. Past the point where the finish rate approaches that
    throughput, every further card adds queue rather than output, and the cost lands on the finished->
    safety wait, which is invisible in the generation time an operator is most likely to look at.

    All three figures come from the session itself: the demand from its own finish stamps over the span
    they cover, the capacity from the median of its own safety check durations, and the margin the wait
    has to clear from that same check duration. Nothing here is sized to a particular card count or a
    particular checker, so a two-card CPU-safety host and a sixteen-card GPU-safety one are judged by the
    same rule. It is silent on a single-card worker, where one card cannot outrun the checker by
    definition, and on a capture whose safety lines are absent, where the wait cannot be located at all.
    """
    lifecycle = job_lifecycle_for(context)
    card_count = lifecycle.card_count
    if card_count is None or card_count < 2:
        return []

    checks: list[float] = []
    waits: list[float] = []
    finishes: list[datetime] = []
    for job in lifecycle.jobs.values():
        if job.safety_seconds is not None:
            checks.append(job.safety_seconds)
        wait = job.finished_to_safety_seconds
        if wait is not None:
            waits.append(wait)
        if job.inference_finished_at is not None:
            finishes.append(job.inference_finished_at)
    finishes.sort()

    if len(checks) < _SAFETY_CAPACITY_MIN_CHECKED_JOBS or len(waits) < _SAFETY_CAPACITY_MIN_CHECKED_JOBS:
        return []
    if len(finishes) < 2:
        return []

    busy_span = (finishes[-1] - finishes[0]).total_seconds()
    median_check = _median(checks)
    median_wait = _median(waits)
    if busy_span <= 0 or median_check is None or median_check <= 0 or median_wait is None:
        return []

    capacity = 1 / median_check
    demand = (len(finishes) - 1) / busy_span
    if demand < capacity * _SAFETY_CAPACITY_DEMAND_FRACTION:
        return []
    if median_wait <= median_check * _SAFETY_CAPACITY_WAIT_MARGIN:
        return []

    generation = lifecycle.wait_segments()[1].median
    severity = Severity.CRITICAL if generation is not None and median_wait > generation else Severity.WARNING
    generation_clause = (
        f"That wait is {'longer' if severity is Severity.CRITICAL else 'shorter'} than the median "
        f"{generation:.1f}s the same jobs spent generating."
        if generation is not None
        else ""
    )

    return [
        Finding(
            kind=FindingKind.SAFETY_STAGE_CAPACITY,
            severity=severity,
            headline=(
                f"{card_count} cards finished {demand:.2f} jobs a second but the single safety check handles "
                f"{capacity:.2f}, so jobs waited a median {median_wait:.1f} seconds for it."
            ),
            evidence=([generation_clause] if generation_clause else [])
            + [
                f"jobs finishing: {len(finishes)} over {busy_span / 60:.0f} min "
                f"({demand:.2f} a second) across {card_count} cards",
                f"safety check speed: {len(checks)} checks, median {median_check:.2f}s each "
                f"(p90 {percentile(checks, 0.9) or 0:.2f}s), so {capacity:.2f} checks a second",
                f"wait from finishing to the safety result: median {median_wait:.1f}s "
                f"(p90 {percentile(waits, 0.9) or 0:.1f}s) over {len(waits)} jobs",
            ],
        ),
    ]


_SAFETY_CPU_FALLBACK_MIN_RERUNS = 3
"""How many reruns a session needs before their cost is compared with the rest of its checks.

Below this, one rerun in a short session outweighs the few checks around it and would read as a pattern."""


@dataclass(frozen=True)
class _SafetyCpuFallback:
    """One safety check that ran out of device memory and was rerun on the CPU."""

    record: LogRecord
    job_id: str
    stage: SafetyEvaluationStage
    free_figure: str | None
    """The allocator's free figure as logged ("0 bytes"), or None when the error did not state one."""
    rerun_seconds: float


def _safety_cpu_fallbacks(records: list[LogRecord]) -> list[_SafetyCpuFallback]:
    """Return the safety checks among ``records`` that were rerun on the CPU, oldest first."""
    fallbacks: list[_SafetyCpuFallback] = []
    for record in records:
        rerun = _SAFETY_CPU_FALLBACK_RE.search(record.message)
        if rerun is None:
            continue
        fallbacks.append(
            _SafetyCpuFallback(
                record=record,
                job_id=rerun.group("job"),
                stage=SafetyEvaluationStage(rerun.group("stage")),
                free_figure=rerun.group("free"),
                rerun_seconds=float(rerun.group("seconds")),
            ),
        )
    fallbacks.sort(key=lambda fallback: _window_key(fallback.record))
    return fallbacks


def _safety_cpu_fallback_evidence(label: str, fallback: _SafetyCpuFallback) -> str:
    """Return one rerun as an evidence line with its log location.

    The line is rebuilt from its fields because the rerun time sits past the length evidence lines are
    clipped to.
    """
    record = fallback.record
    free = f"{fallback.free_figure} free" if fallback.free_figure else "free figure not stated"
    return (
        f"{label}: {_clock_text(record.timestamp)}  {record.source_path.name}:{record.raw_lineno}  job "
        f"{fallback.job_id[:8]} during {fallback.stage}, {free}, rerun in {fallback.rerun_seconds:.2f}s"
    )


def detect_safety_check_cpu_fallback(context: SessionContext) -> list[Finding]:
    """Safety checks on the GPU that found the card full and were rerun on the CPU.

    A rerun keeps the safety process up, so a few are the containment working and the finding is a note.
    It is a warning when at least :data:`_SAFETY_CPU_FALLBACK_MIN_RERUNS` reruns together took as long as
    every other safety check of the session, which is when they have doubled what the safety stage costs.
    Both sides of that comparison are the session's own timings, so a slow CPU reaches it at a lower rerun
    rate than a fast one. A capture without the parent's per-job safety lines has nothing to compare against
    and stays a note.
    """
    child_records = _child_records_in_session(context)
    fallbacks = _safety_cpu_fallbacks(child_records)
    if not fallbacks:
        return []
    failed_releases = _matching(child_records, _SAFETY_WEIGHTS_RELEASE_OOM_RE)
    failed_promotions = _matching(child_records, _SAFETY_WEIGHTS_PROMOTION_OOM_RE)

    rerun_job_ids = {fallback.job_id for fallback in fallbacks}
    checked_count = 0
    other_check_seconds: list[float] = []
    for record in context.session.records:
        checked = _SAFETY_CHECKED_RE.search(record.message)
        if checked is None:
            continue
        checked_count += 1
        if checked.group("job") not in rerun_job_ids:
            other_check_seconds.append(float(checked.group("seconds")))

    count = len(fallbacks)
    rerun_seconds = [fallback.rerun_seconds for fallback in fallbacks]
    total_rerun_seconds = sum(rerun_seconds)
    median_rerun_seconds = statistics.median(rerun_seconds)
    longest = max(fallbacks, key=lambda fallback: fallback.rerun_seconds)
    total_other_seconds = sum(other_check_seconds)
    session_checks_known = checked_count >= count
    reruns_outweigh_other_checks = session_checks_known and total_rerun_seconds >= total_other_seconds
    reportable = count >= _SAFETY_CPU_FALLBACK_MIN_RERUNS and reruns_outweigh_other_checks

    took = (
        f"taking {median_rerun_seconds:.1f} seconds"
        if count == 1
        else f"taking a median {median_rerun_seconds:.1f} seconds"
    )
    median_other_seconds = _median(other_check_seconds)
    if session_checks_known:
        subject = f"{count} of {_count_text(checked_count, 'safety check')} found the card full and reran on the CPU"
        usual = f" where others took {median_other_seconds:.2f}" if median_other_seconds is not None else ""
    else:
        subject = f"Safety checks found the card full and reran on the CPU {_times_text(count)}"
        usual = ""

    first, last = fallbacks[0], fallbacks[-1]
    if count == 1:
        span = f"at {_clock_text(first.record.timestamp)}"
    else:
        span = (
            f"over {_span_text(_seconds_between(first.record.timestamp, last.record.timestamp))} from "
            f"{_clock_text(first.record.timestamp)} to {_clock_text(last.record.timestamp)}"
        )
    if session_checks_known:
        reruns_line = f"reruns: {count} of {checked_count} safety checks ({count / checked_count:.1%}), {span}"
        against = f" against {total_other_seconds:.1f}s for the other {len(other_check_seconds)} checks"
    else:
        reruns_line = f"reruns: {count}, {span}. The main log has no safety check lines to compare them with."
        against = ""
    by_stage = [
        f"{stage_count} during {stage}"
        for stage in SafetyEvaluationStage
        if (stage_count := sum(1 for fallback in fallbacks if fallback.stage is stage))
    ]

    return [
        Finding(
            kind=FindingKind.SAFETY_CHECK_CPU_FALLBACK,
            severity=Severity.WARNING if reportable else Severity.INFO,
            headline=f"{subject}, {took}{usual}.",
            action_addendum=(
                "Run `horde-log bundle` and send it to the worker maintainers. No setting fixes this. The worker "
                "keeps checking every image meanwhile."
                if reportable
                else "No action needed."
            ),
            evidence=[
                reruns_line,
                f"by stage: {', '.join(by_stage)}",
                f"rerun time: median {median_rerun_seconds:.2f}s, longest {longest.rerun_seconds:.2f}s, "
                f"{total_rerun_seconds:.1f}s in total{against}",
                f"safety weight moves that ran out of card memory: {len(failed_releases)} releasing after a "
                f"demotion, {len(failed_promotions)} promoting",
                _safety_cpu_fallback_evidence("first", first),
                *([] if longest is first else [_safety_cpu_fallback_evidence("longest", longest)]),
                *(_evidence(record) for record in failed_releases[:1] + failed_promotions[:1]),
            ],
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
    detect_multi_card_dispatch_serialization,
    detect_parent_loop_stall,
    detect_model_churn,
    detect_host_ram_starvation,
    detect_lane_placement,
    detect_utilities_lane_bringup_timeout,
    detect_inference_slot_retired,
    detect_safety_start_escalated_to_cpu,
    detect_inference_slot_quarantined,
    detect_inference_lane_replaced,
    detect_lane_fault_rule_misfire,
    detect_supervisor_channel_lost,
    detect_safety_stage_stall,
    detect_safety_stage_capacity,
    detect_safety_check_cpu_fallback,
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
    detect_text_backend_wedged,
    detect_pop_governor_dominance,
    detect_pop_api_error_dominance,
    detect_stuck_inference_step,
    detect_post_processing_vram_stall,
    detect_post_processing_offer_withheld,
    detect_idle_fill_offer_stuck,
    detect_service_lane_restart_churn,
    detect_lane_double_dispatch_signature,
    detect_unpriced_sampling_windows,
    detect_stage_outlived_sample_grant,
    detect_stage_refusals,
    detect_post_processing_deferral_starvation,
    detect_oom,
    detect_file_descriptor_exhaustion,
    detect_pagefile_exhaustion,
    detect_accelerator_driver_fault,
    detect_device_reading_missing,
    detect_swallowed_oom,
    detect_orphan_wedge,
    detect_model_economics,
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
    findings.sort(key=lambda finding: SEVERITY_ORDER[finding.severity])
    return findings
