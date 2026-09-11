"""The registry of worker log lines the analysis package parses, and the contract that keeps it honest.

Every regex in this package that reads a line the worker emits is an invisible contract with an f-string
in ``process_management/``: nothing links the two, so a reworded emit silently retires the parser and the
triage tool goes quiet about a defect it used to name. This module makes that contract a single table.

Each :class:`LogSignature` names the pattern, the ``module:function`` that emits the line, and a literal
sample copied from a real worker log. Two tests drive off the table rather than off scattered constants:
a literal pin (every pattern still matches its recorded sample) and a contract test that runs the worker
through its dry-run flow and asserts the patterns match the log that run actually produced. A signature
whose line a dry run cannot produce says so in ``dry_run_reason`` and is covered by the literal pin alone.

Consumers import the compiled patterns from here (``SIGNATURES["popped_job"].pattern``) rather than
compiling their own, so a new parser cannot be added without a registered sample and an emitter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class LogSignature:
    """Represents one worker log line the analysis package parses, and where the worker emits it."""

    name: str
    pattern: re.Pattern[str]
    emitter: str
    """``module:function`` of the emitting site in ``horde_worker_regen``."""
    sample: str
    """A literal message (the part after loguru's ``location - `` prefix) copied from a real worker log."""
    dry_run_reason: str = ""
    """Why the dry-run contract test cannot exercise this line, or an empty string when it can.

    A non-empty value exempts the signature from the dry-run assertion; the literal pin still applies."""
    field_of: str | None = None
    """The signature this one parses a field out of, for patterns that read a fragment rather than a line."""

    @property
    def dry_run_reachable(self) -> bool:
        """Whether a dry-run harness run is expected to emit this line."""
        return not self.dry_run_reason


def _signature(
    name: str,
    pattern: str,
    *,
    emitter: str,
    sample: str,
    dry_run_reason: str = "",
    field_of: str | None = None,
    flags: re.RegexFlag = re.NOFLAG,
) -> LogSignature:
    """Compile and describe one registry entry."""
    return LogSignature(
        name=name,
        pattern=re.compile(pattern, flags),
        emitter=emitter,
        sample=sample,
        dry_run_reason=dry_run_reason,
        field_of=field_of,
    )


_NO_FAULT = "the dry-run contract scenario completes every job, so nothing faults"
_NOT_DETERMINISTIC = (
    "reachable only through a scheduling race a short deterministic run cannot be relied on to produce"
)
_SAFETY_PLACEMENT_NOT_EXERCISED = "the dry-run harness never pauses or restores the safety process's GPU placement"
_WHOLE_CARD_NOT_EXERCISED = (
    "reachable only with a whole-card-class model (e.g. Flux fp8) and a multi-process pool, which the "
    "dry-run harness's fake models never trigger"
)
_MALFORMED_POP_NOT_EXERCISED = "the dry-run harness never synthesizes a malformed pop response with a blank model name"
_HORDELIB_READOUT_NOT_EXERCISED = (
    "the dry-run harness's fake inference children never call hordelib's free-VRAM readout"
)

_SIGNATURE_LIST: list[LogSignature] = [
    # --- Per-job lifecycle ---
    _signature(
        "popped_job",
        r"Popped job (?P<job>[0-9a-fA-F-]{8,}) \((?P<emps>\d+) eMPS\) "
        r"\(model: (?P<model>.+?), batch: (?P<batch>\d+), loras: (?P<loras>\w+), post_processing: (?P<post>\w+)\)",
        emitter="process_management.jobs.job_popper:api_job_pop",
        sample=(
            "Popped job 019546e7-4a4a-4eea-afd6-d41078e14ac7 (35 eMPS) "
            "(model: NatViS, batch: 1, loras: False, post_processing: False)"
        ),
    ),
    _signature(
        "job_queue",
        r"^Job queue: (?P<body>.*)$",
        emitter="process_management.jobs.job_popper:_enqueue_popped_job",
        sample="Job queue: <3ef78e3d: NTR MIX IL-Noob XL>, <a5477739: Z-Image-Turbo>, <019546e7: NatViS>",
    ),
    _signature(
        "inference_dispatched",
        r"Starting inference for job (?P<job>[0-9a-fA-F]{8}) on process (?P<process>\d+)",
        emitter="process_management.scheduling.inference_scheduler:start_inference",
        sample="Starting inference for job f5fd3c31 on process 10",
    ),
    _signature(
        "inference_finished",
        r"Inference finished for job (?P<job>[0-9a-fA-F]{8}) \((?P<model>.+?)\) on process (?P<process>\d+)\. "
        r"It took (?P<seconds>[\d.]+) seconds",
        emitter="process_management.ipc.message_dispatcher:_handle_inference_result",
        sample=(
            "Inference finished for job 11cad94a (WAI-NSFW-illustrious-SDXL) on process 7. It took 6.58 "
            "seconds, finishing at 3.42 iterations per second and reported 1 faults."
        ),
    ),
    _signature(
        "safety_checked",
        r"Job (?P<job>[0-9a-fA-F-]{8,}) had (?P<censored>\d+) images censored and took "
        r"(?P<seconds>[\d.]+) seconds to check safety",
        emitter="process_management.ipc.message_dispatcher:_handle_safety_result",
        sample=(
            "Job 9c46c418-81a8-4727-b8c5-495da36f9efd had 0 images censored and took 0.11 seconds to check safety"
        ),
    ),
    _signature(
        "submitted_generation",
        r"Submitted generation (?P<job>[0-9a-fA-F]{8}) \(model: (?P<model>.+?)\) for (?P<kudos>[\d.-]+) kudos\. "
        r"Job popped (?P<popped_ago>[\d.]+) seconds ago and took (?P<generate>[\d.]+) to generate",
        emitter="process_management.jobs.job_submitter:submit_single_generation",
        sample=(
            "Submitted generation 9c46c418 (model: WAI-NSFW-illustrious-SDXL) for 22.62 kudos. Job popped "
            "12.98 seconds ago and took 6.55 to generate. (3.46 kudos/second for the whole batch. 0.4 or "
            "greater is ideal)"
        ),
        dry_run_reason="dry_run_skip_api completes the submit before the success line is reached",
    ),
    _signature(
        "job_faulted_on_process",
        r"Job (?P<job>[0-9a-fA-F-]{8,}) faulted on process (?P<process>\d+)",
        emitter="process_management.ipc.message_dispatcher:_handle_faulted_inference_result",
        sample=(
            "Job 597d4471-b223-4383-b874-86c6a1549594 faulted on process 3 (RuntimeError: Pipeline failed to "
            "run - declared output node(s) ['output_image'] produced no results."
        ),
        dry_run_reason=_NO_FAULT,
    ),
    _signature(
        "fault_reported",
        r"(?P<job>[0-9a-fA-F-]{8,}) faulted\. Reported fault to the horde\.",
        emitter="process_management.jobs.job_submitter:submit_single_generation",
        sample=(
            "350f913e-522a-4284-a5bb-0c7feefdfa3f faulted. Reported fault to the horde. Job popped 168.92 "
            "seconds ago and took 0.00 to generate."
        ),
        dry_run_reason=_NO_FAULT,
    ),
    _signature(
        "batch_ids",
        r"^All Batch IDs: \[(?P<ids>[^\]]*)\]",
        emitter="process_management.scheduling.inference_scheduler:_log_job_dispatch_details",
        sample=(
            "All Batch IDs: [7c8e526a-f055-420a-945e-e58784b93539, ac2168ef-06e5-495d-919a-7d33fde18d80, "
            "621f784a-480d-4e18-a4a9-43f2b8f815e2]"
        ),
    ),
    _signature(
        "batch_id_entry",
        r"[0-9a-fA-F]{8}-[0-9a-fA-F-]+",
        emitter="process_management.scheduling.inference_scheduler:_log_job_dispatch_details",
        sample="7c8e526a-f055-420a-945e-e58784b93539",
        field_of="batch_ids",
    ),
    _signature(
        "line_skip",
        r"Job (?P<job>[0-9a-fA-F-]{8,}) skipped the line \((?P<reason>[^)]+)\) and will run on process "
        r"(?P<process>\d+) ahead of job (?P<displaced>[0-9a-fA-F-]{8,})",
        emitter="process_management.scheduling.inference_scheduler:start_inference",
        sample=(
            "Job 6cf4693b-7af8-44c0-95b6-faf5a6df4658 skipped the line (diversity) and will run on process 13 "
            "ahead of job 272bdef7-0d3d-4d00-9a57-7de568abd5df: the displaced job's process is busy sampling "
            "its own model."
        ),
        dry_run_reason=_NOT_DETERMINISTIC,
    ),
    # --- Lane placement ---
    _signature(
        "inference_lane_started",
        r"Starting inference process on PID (?P<process>\d+) \(device (?P<device>\d+)\)",
        emitter="process_management.lifecycle.process_lifecycle:_start_inference_process",
        sample="Starting inference process on PID 3 (device 0)",
    ),
    _signature(
        "post_process_lane_started",
        r"Started post-process process \(id: (?P<process>\d+), device_index: (?P<device>\d+)\)",
        emitter="process_management.lifecycle.process_lifecycle:start_post_process_processes",
        sample="Started post-process process (id: 1, device_index: 1)",
    ),
    _signature(
        "utilities_lane_started",
        r"Started image utilities process \(id: (?P<process>\d+), device_index: (?P<device>\d+)\)",
        emitter="process_management.lifecycle.process_lifecycle:start_utilities_processes",
        sample="Started image utilities process (id: 2, device_index: 1)",
    ),
    _signature(
        "safety_lane_started",
        r"Started safety process \(id: (?P<process>\d+)\)",
        emitter="process_management.lifecycle.process_lifecycle:start_safety_processes",
        sample="Started safety process (id: 0)",
    ),
    _signature(
        "auxiliary_lane_safety_sharing",
        r"The safety process and the (?P<lanes>.+?) lane\(s\) now share device (?P<device>\d+)",
        emitter="process_management.lifecycle.process_lifecycle:_note_auxiliary_lane_safety_sharing",
        sample=(
            "The safety process and the post-processing lane(s) now share device 1; the lane(s) stay on "
            "the card they were placed on."
        ),
        dry_run_reason=(
            "edge-triggered when safety returns to a card a pinned auxiliary lane already holds, which "
            "needs a pause and restore of the safety process on a multi-card host; the dry-run harness "
            "does neither"
        ),
    ),
    # --- Worker-wide shape ---
    _signature(
        "driving_cards",
        r"Driving (?P<cards>\d+) cards, .*?Worker-wide intake budget: (?P<intake>\d+) job\(s\) held at once "
        r"\((?P<per_card>.*?)\)\.",
        emitter="process_management.process_manager:_build_card_runtimes",
        sample=(
            "Driving 8 cards, each with its own inference process pool; max_threads/queue_size apply per "
            "card. Worker-wide intake budget: 8 job(s) held at once (card 0: 1 process(es), intake 1, card "
            "1: 1 process(es), intake 1)."
        ),
    ),
    _signature(
        "per_card_intake",
        r"card (?P<device>\d+): (?P<processes>\d+) process\(es\), intake (?P<intake>\d+)",
        emitter="process_management.process_manager:_build_card_runtimes",
        sample="card 0: 1 process(es), intake 1",
        field_of="driving_cards",
    ),
    _signature(
        "per_card_concurrency",
        r"Resolved per-card concurrency: \{(?P<body>.*)\}",
        emitter="process_management.process_manager:_build_card_runtimes",
        sample=(
            "Resolved per-card concurrency: {0: CardConcurrency(target_process_count=1, "
            "max_concurrent_inference=1, inference_semaphore_size=1, vae_decode_semaphore_size=1, "
            "gpu_sampling_lease_slots=1, gpu_sampling_lease_tail_overlap=False)}"
        ),
    ),
    _signature(
        "per_card_concurrency_entry",
        r"(?P<device>\d+): CardConcurrency\(target_process_count=(?P<processes>\d+)",
        emitter="process_management.process_manager:_build_card_runtimes",
        sample="0: CardConcurrency(target_process_count=1",
        field_of="per_card_concurrency",
    ),
    # --- Model movement ---
    _signature(
        "model_preloading",
        r"Preloading model (?P<model>.+?) on process (?P<process>\d+)$",
        emitter="process_management.scheduling.inference_scheduler:_send_preload",
        sample="Preloading model Z-Image-Turbo on process 10",
    ),
    _signature(
        "model_unloaded",
        r"Process (?P<process>\d+) unloaded model (?P<model>.+)$",
        emitter="process_management.ipc.message_dispatcher:_handle_model_state_change",
        sample="Process 10 unloaded model WAI-NSFW-illustrious-SDXL",
    ),
    _signature(
        "preload_cleared",
        r"Clearing preloaded model (?P<model>.+?) from process (?P<process>\d+) as it is no longer needed",
        emitter="process_management.scheduling.inference_scheduler:preload_models",
        sample="Clearing preloaded model WAI-NSFW-illustrious-SDXL from process 3 as it is no longer needed",
        dry_run_reason=_NOT_DETERMINISTIC,
    ),
    _signature(
        "displaced_entry_expired",
        r"Expiring displaced entry for (?P<model>.+?) on process (?P<process>\d+): the slot now holds "
        r"(?P<holds>.+?)\.",
        emitter="process_management.scheduling.inference_scheduler:_expire_stale_model_map_entries",
        sample=(
            "Expiring displaced entry for WAI-NSFW-illustrious-SDXL on process 10: the slot now holds Z-Image-Turbo."
        ),
        dry_run_reason=_NOT_DETERMINISTIC,
    ),
    _signature(
        "model_loading",
        r"Process (?P<process>\d+) is loading model (?P<model>.+)$",
        emitter="process_management.ipc.message_dispatcher:_handle_model_state_change",
        sample="Process 10 is loading model Z-Image-Turbo",
    ),
    _signature(
        "model_moved_to_system_ram",
        r"Process (?P<process>\d+) moved model (?P<model>.+?) to system RAM\.",
        emitter="process_management.ipc.message_dispatcher:_handle_model_state_change",
        sample="Process 5 moved model Nova Anime XL to system RAM. ",
    ),
    # --- Status block. The lane lines are printed by the status reporter but built by the process map,
    # so a reword can come from either: the reporter owns the indent and the print, the map owns the
    # text after it. ---
    _signature(
        "status_header",
        r"^\^{10,}\s*$",
        emitter="reporting.status_reporter:print_status",
        sample="^" * 80,
    ),
    _signature(
        "status_inference_lane",
        r"^\s{2}Process (?P<process>\d+) \((?P<state>[^)]*)\) "
        r"\((?:<u>(?P<model>.*?)</u>[^)]*|(?P<no_model>No model loaded))\)",
        emitter="process_management.lifecycle.process_map:get_process_info_strings",
        sample=(
            "  Process 3 (WAITING_FOR_JOB) (<u>Rev Animated</u> stable_diffusion_1)) <fg #7b7d7d>"
            "[last message: 0.0 secs ago: START_INFERENCE heartbeat delta: 0.13]</>"
        ),
    ),
    _signature(
        "status_aux_lane",
        r"^\s{2}Process (?P<process>\d+): \((?P<role>SAFETY|POST_PROCESS|UTILITIES)\) (?P<state>[A-Z_]+)",
        emitter="process_management.lifecycle.process_map:get_process_info_strings",
        sample="  Process 1: (POST_PROCESS) WAITING_FOR_JOB  <fg #7b7d7d>vram 0 MB reserved (0 live)</>",
    ),
    _signature(
        "status_sampling_state",
        r"^(?P<percent>\d+)% of (?P<steps>\d+) steps using (?P<sampler>\S+)",
        emitter="process_management.lifecycle.process_map:get_process_info_strings",
        sample="63% of 30 steps using k_euler_a",
        dry_run_reason="the fake inference children report no per-step sampling progress",
        field_of="status_inference_lane",
    ),
    _signature(
        "status_jobs",
        r"^\s{2}Jobs: (?P<body>.*)$",
        emitter="reporting.status_reporter:_print_job_info",
        sample="  Jobs: <3ef78e3d: NTR MIX IL-Noob XL>, <a5477739: Z-Image-Turbo>",
    ),
    _signature(
        "queue_entry",
        r"<(?P<job>[^:>]{1,12}): (?P<model>.*?)>(?=, <|$)",
        emitter="reporting.status_reporter:_print_job_info",
        sample="<3ef78e3d: NTR MIX IL-Noob XL>, <a5477739: Z-Image-Turbo>",
        field_of="status_jobs",
    ),
    # --- Parent loop ---
    _signature(
        "ipc_drain",
        r"Received (?P<message_type>\w+) from process (?P<process>\d+)",
        emitter="process_management.ipc.message_dispatcher:_dispatch_buffered_message",
        sample="Received HordeInferenceResultMessage from process 7: 3.42 iterations per second",
    ),
    # --- Recovery supervisor (save-our-ship) ---
    _signature(
        "quarantine",
        r"quarantined \(crash on start",
        emitter="process_management.lifecycle.process_lifecycle:_replace_inference_process",
        sample=(
            "Inference slot 3 quarantined (crash on start: 4 consecutive failures before reaching "
            "readiness); not respawning it."
        ),
        dry_run_reason=_NO_FAULT,
    ),
    _signature(
        "soft_reset",
        r"Save-our-ship soft reset",
        emitter="process_management.lifecycle.worker_recovery_coordinator:perform_soft_reset",
        sample="Save-our-ship soft reset #1: rebuilding process pools after repeated crash/hang recoveries.",
        dry_run_reason=_NO_FAULT,
    ),
    _signature(
        "pools_recovered",
        r"pools recovered.*limp-by cleared",
        emitter="process_management.lifecycle.worker_recovery_coordinator:run_recovery_supervisor",
        sample="Save-our-ship: pools recovered (soft-reset episode cleared, limp-by cleared).",
        dry_run_reason=_NO_FAULT,
    ),
    _signature(
        "abandon_ship",
        r"abandoning ship|cannot restore a working process pool",
        emitter="process_management.lifecycle.worker_recovery_coordinator:run_recovery_supervisor",
        sample=(
            "Save-our-ship: the worker cannot restore a working process pool after repeated soft resets; "
            "abandoning ship (the last resort) rather than spinning indefinitely."
        ),
        dry_run_reason=_NO_FAULT,
    ),
    _signature(
        "give_up",
        r"gave up on (\d+) unservable job",
        emitter="process_management.lifecycle.worker_recovery_coordinator:give_up_on_wedged_jobs",
        sample=(
            "Save-our-ship: gave up on 4 unservable job(s) (scheduler wedged with idle processes (queue "
            "deadlock)) and reported them faulted so the horde reissues them."
        ),
        dry_run_reason=_NO_FAULT,
    ),
    # --- Per-job faults ---
    _signature(
        "faulted_on_process",
        r"faulted on process (?P<pid>\d+)",
        emitter="process_management.ipc.message_dispatcher:_handle_faulted_inference_result",
        sample=(
            "Job 597d4471-b223-4383-b874-86c6a1549594 faulted on process 3 (RuntimeError: Pipeline failed to "
            "run - declared output node(s) ['output_image'] produced no results."
        ),
        dry_run_reason=_NO_FAULT,
    ),
    _signature(
        "faulted_on_process_job",
        r"Job (?P<job>[0-9a-fA-F-]{8,}) faulted on process (?P<pid>\d+)",
        emitter="process_management.ipc.message_dispatcher:_handle_faulted_inference_result",
        sample=(
            "Job 597d4471-b223-4383-b874-86c6a1549594 faulted on process 3 (RuntimeError: Pipeline failed to "
            "run - declared output node(s) ['output_image'] produced no results."
        ),
        dry_run_reason=_NO_FAULT,
    ),
    _signature(
        "fault_requeued_suffix",
        r"requeued for (?:a degraded, isolated|another) attempt",
        emitter="process_management.ipc.message_dispatcher:_handle_faulted_inference_result",
        sample=(
            "Job 597d4471-b223-4383-b874-86c6a1549594 faulted on process 3 (RuntimeError: ...); requeued for "
            "another attempt."
        ),
        dry_run_reason=_NO_FAULT,
        field_of="faulted_on_process_job",
    ),
    _signature(
        "popped_job_model",
        r"Popped job (?P<job>\S+) .*?\(model: (?P<model>.+?), batch:",
        emitter="process_management.jobs.job_popper:api_job_pop",
        sample=(
            "Popped job 019546e7-4a4a-4eea-afd6-d41078e14ac7 (35 eMPS) "
            "(model: NatViS, batch: 1, loras: False, post_processing: False)"
        ),
    ),
    _signature(
        "safety_unrecoverable_job",
        r"Job (?P<job>\S+) could not be safety-checked",
        emitter="process_management.lifecycle.worker_recovery_coordinator:reconcile_orphaned_safety_jobs",
        sample=(
            "Job 9c46c418-81a8-4727-b8c5-495da36f9efd could not be safety-checked (requeue attempts "
            "exhausted); dropping its images and faulting the job."
        ),
        dry_run_reason=_NO_FAULT,
    ),
    _signature(
        "safety_unrecoverable",
        r"could not be safety-checked",
        emitter="process_management.lifecycle.worker_recovery_coordinator:reconcile_orphaned_safety_jobs",
        sample=(
            "Job 9c46c418-81a8-4727-b8c5-495da36f9efd could not be safety-checked (requeue attempts "
            "exhausted); dropping its images and faulting the job."
        ),
        dry_run_reason=_NO_FAULT,
    ),
    _signature(
        "stage_fault",
        r"(?P<stage>\w[\w -]*) stage faulted for job (?P<job>\S+?):",
        emitter="process_management.workers.inference_process:_run_sample_stage",
        sample="Sample stage faulted for job 9c46c418-81a8-4727-b8c5-495da36f9efd: RuntimeError: ...",
        dry_run_reason=_NO_FAULT,
    ),
    _signature(
        "model_reference_unreadable",
        r"Model reference for category (?P<category>\S+) not found or could not be parsed",
        emitter="process_management.workers.inference_process:_run_sample_stage",
        sample="Model reference for category image_generation not found or could not be parsed.",
        dry_run_reason=_NO_FAULT,
    ),
    _signature(
        "model_reference_stale",
        r"needs refresh|cache is stale",
        emitter="horde_model_reference.backends.replica_backend_base:needs_refresh",
        sample="Category blip cache is stale, needs refresh",
        dry_run_reason=(
            "the dry-run harness's model reference cache never goes stale within the scenario's short runtime"
        ),
    ),
    # --- Pop liveness and governors ---
    _signature(
        "pop_liveness_frozen",
        r"Pop liveness: the local job queue has been full and not draining",
        emitter="process_management.process_manager:_full_queue_frozen_line",
        sample=(
            "Pop liveness: the local job queue has been full and not draining for 180s (8 accepted job(s) "
            "waiting, head model 'Z-Image-Turbo')"
        ),
        dry_run_reason=_NO_FAULT,
    ),
    _signature(
        "pop_liveness_frozen_fields",
        r"Pop liveness: the local job queue has been full and not draining for (?P<seconds>\d+)s "
        r"\((?P<waiting>\d+) accepted job\(s\) waiting, head model '(?P<model>[^']*)'\)",
        emitter="process_management.process_manager:_full_queue_frozen_line",
        sample=(
            "Pop liveness: the local job queue has been full and not draining for 180s (8 accepted job(s) "
            "waiting, head model 'Z-Image-Turbo')"
        ),
        dry_run_reason=_NO_FAULT,
        field_of="pop_liveness_frozen",
    ),
    _signature(
        "residency_governor_model",
        r"Pop governor ENTER: whole_card_residency \((?P<model>.+?) holds the card",
        emitter="process_management.scheduling.pop_governor_registry:_enter_line",
        sample="Pop governor ENTER: whole_card_residency (Z-Image-Turbo holds the card (established))",
        dry_run_reason=_WHOLE_CARD_NOT_EXERCISED,
    ),
    # --- Swallowed/misclassified faults ---
    _signature(
        "no_images",
        r"no images were produced|no images produced",
        emitter="process_management.jobs.failure_classification:is_resource_failure",
        sample="Pipeline failed to run - no images were produced",
        dry_run_reason=_NO_FAULT,
    ),
    _signature(
        "orphaned_job_punt",
        r"orphaned? in-progress|punt(?:ing|ed) (?:an? )?orphan|orphaned-job watchdog",
        emitter="process_management.lifecycle.worker_recovery_coordinator:reconcile_orphaned_in_progress_jobs",
        sample="Job 9c46c418 in-progress for 320s; punting it so the queue can drain (orphaned-job watchdog).",
        dry_run_reason=_NO_FAULT,
    ),
    # --- Maintenance and pop rejection ---
    _signature(
        "maintenance_pop",
        r"Failed to pop job \(Maintenance Mode\)",
        emitter="process_management.jobs.job_popper:_handle_pop_error_response",
        sample=(
            "Failed to pop job (Maintenance Mode): message='Maintenance mode activated because worker is "
            "dropping too many jobs.Please investigate if your performance has been impacted and consider "
            "reducing your max_power or your max_threads' object_data=None rc='WorkerMaintenance'"
        ),
        dry_run_reason=_NO_FAULT,
    ),
    _signature(
        "dropping_jobs_reason",
        r"dropping too many jobs",
        emitter="process_management.jobs.job_popper:_handle_pop_error_response",
        sample=(
            "Failed to pop job (Maintenance Mode): message='Maintenance mode activated because worker is "
            "dropping too many jobs.'"
        ),
        dry_run_reason=_NO_FAULT,
        field_of="maintenance_pop",
    ),
    _signature(
        "consecutive_pause",
        r"Too many consecutive failed jobs, pausing job pops",
        emitter="process_management.jobs.job_popper:api_job_pop",
        sample=(
            "Too many consecutive failed jobs, pausing job pops. This may be due to a misconfiguration or other issue."
        ),
        dry_run_reason=_NO_FAULT,
    ),
    _signature(
        "server_slow_abort",
        r"took too long to process and has been aborted",
        emitter="process_management.jobs.job_submitter:submit_single_generation",
        sample=(
            "Processing Generation with ID d2a16369-e70c-4288-aefe-084d41e79f25 took too long to process and "
            "has been aborted! Please check your worker speed - generations must complete within 150 seconds "
            "(minimum ~1 it/s, higher for large requests)!"
        ),
        dry_run_reason=(
            "the dry-run contract scenario generates well inside the horde's per-job deadline, so nothing is "
            "aborted as too slow"
        ),
    ),
    _signature(
        "slowdown_grade",
        r"is ([\d.]+)x its expected sampling time",
        emitter="process_management.lifecycle.process_lifecycle:_grade_running_inference",
        sample=("Inference on process 5 is 4.0x its expected sampling time (22s vs ~6s); watching for a hang."),
        dry_run_reason=(
            "the fake inference children in the dry-run scenario complete at a fixed fast rate, so grading "
            "never reports a multiple of expected time"
        ),
    ),
    _signature(
        "safety_check_duration",
        r"took ([\d.]+) seconds to check safety",
        emitter="process_management.ipc.message_dispatcher:_handle_safety_result",
        sample=(
            "Job 9c46c418-81a8-4727-b8c5-495da36f9efd had 0 images censored and took 0.11 seconds to check safety"
        ),
        field_of="safety_checked",
    ),
    # --- Safety stage stall/recovery ---
    _signature(
        "safety_requeue",
        r"requeued it for a fresh safety check",
        emitter="process_management.lifecycle.worker_recovery_coordinator:reconcile_orphaned_safety_jobs",
        sample="Job 9c46c418 had no safety result after 60s; requeued it for a fresh safety check (attempt 1/3).",
        dry_run_reason=_NO_FAULT,
    ),
    _signature(
        "safety_soft_pause",
        r"Soft-pausing job pops.*safety could not check a result",
        emitter="process_management.lifecycle.worker_recovery_coordinator:engage_safety_soft_pause",
        sample="Soft-pausing job pops for 60s: safety could not check a result (requeue attempts exhausted).",
        dry_run_reason=_NO_FAULT,
    ),
    _signature(
        "safety_backpressure",
        r"Withholding job pops: post-inference safety backlog (\d+) >= cap (\d+)",
        emitter="process_management.jobs.job_popper:api_job_pop",
        sample="Withholding job pops: post-inference safety backlog 12 >= cap 10.",
        dry_run_reason=_NO_FAULT,
    ),
    _signature(
        "lost_safety_result",
        r"Expected to find a completed job .* none was found",
        emitter="process_management.ipc.message_dispatcher:_handle_safety_result",
        sample="Expected to find a completed job with ID 9c46c418-81a8-4727-b8c5-495da36f9efd but none was found.",
        dry_run_reason=_NO_FAULT,
    ),
    _signature(
        "safety_placement_cycle",
        r"(?P<owner>[\w -]+): (?:moving the safety process off-GPU|restoring the safety process to the GPU)",
        emitter="process_management.lifecycle.process_lifecycle:pause_safety_on_gpu",
        sample="Runtime safety placement: moving the safety process off-GPU to free its VRAM context.",
        dry_run_reason=_SAFETY_PLACEMENT_NOT_EXERCISED,
    ),
    _signature(
        "retired_safety_result",
        r"Ignoring result message from retired safety process",
        emitter="process_management.ipc.message_dispatcher:_classify_retired_launch_message",
        sample="Ignoring result message from retired safety process (launch retired by a placement cycle).",
        dry_run_reason=_SAFETY_PLACEMENT_NOT_EXERCISED,
    ),
    # --- Dispatch stall and whole-card residency ---
    _signature(
        "dispatch_stall",
        r"Inference dispatch stalled: head ",
        emitter="process_management.scheduling.inference_scheduler:_log_dispatch_stall_if_needed",
        sample=(
            "Inference dispatch stalled: head ddd85050 (Z-Image-Turbo) has been parked 10s: its model is "
            "resident and idle on process 10, but dispatch is held to reconcile residency (evicting idle "
            "VRAM): its materialisation would over-commit the card until an idle resident is evicted, and it "
            "dispatches once that eviction frees room."
        ),
        dry_run_reason=_NOT_DETERMINISTIC,
    ),
    _signature(
        "dispatch_stall_no_gate",
        r"dispatch was withheld with no matching gate",
        emitter="process_management.scheduling.inference_scheduler:_classify_dispatch_stall",
        sample=(
            "its model is resident and idle on process 1 but dispatch was withheld with no matching gate; "
            "this is a scheduler stall worth reporting"
        ),
        dry_run_reason=_NOT_DETERMINISTIC,
        field_of="dispatch_stall",
    ),
    _signature(
        "dispatch_stall_reconcile",
        r"held to reconcile residency",
        emitter="process_management.scheduling.inference_scheduler:_classify_dispatch_stall",
        sample=(
            "its model is resident and idle on process 10, but dispatch is held to reconcile residency "
            "(evicting idle VRAM): its materialisation would over-commit the card until an idle resident is "
            "evicted, and it dispatches once that eviction frees room"
        ),
        dry_run_reason=_NOT_DETERMINISTIC,
        field_of="dispatch_stall",
    ),
    _signature(
        "dispatch_stall_fields",
        r"Inference dispatch stalled: head \S+ \((?P<model>.+?)\) has been parked (?P<parked>\d+)s:",
        emitter="process_management.scheduling.inference_scheduler:_log_dispatch_stall_if_needed",
        sample="Inference dispatch stalled: head ddd85050 (Z-Image-Turbo) has been parked 10s:",
        dry_run_reason=_NOT_DETERMINISTIC,
        field_of="dispatch_stall",
    ),
    _signature(
        "whole_card_wedge",
        r"whole-card residency stuck: cannot reach sole residency",
        emitter="process_management.scheduling.inference_scheduler:_classify_dispatch_stall",
        sample=(
            "its model is resident and idle on process 10, but the whole-card residency stuck: cannot reach "
            "sole residency because process 4 holds queued model 'Z-Image-Turbo'; the convergence teardown "
            "should have stopped that idle sibling or ordered that lane off-GPU (only the head's holder is "
            "spared), so the gate's structural legs never pass and the head never dispatches"
        ),
        dry_run_reason=_WHOLE_CARD_NOT_EXERCISED,
        field_of="dispatch_stall",
    ),
    _signature(
        "whole_card_nonhead",
        r"whole-card residency is held for non-head model",
        emitter="process_management.scheduling.inference_scheduler:_classify_dispatch_stall",
        sample=(
            "its model is not resident because a whole-card residency is held for non-head model "
            "'Flux.1-Schnell fp8 (Compact)': the card is reserved for that model and its siblings were torn "
            "down, so this head cannot load until that residency restores"
        ),
        dry_run_reason=_WHOLE_CARD_NOT_EXERCISED,
        field_of="dispatch_stall",
    ),
    _signature(
        "whole_card_establish",
        r"Whole-card residency: reserving the device for",
        emitter="process_management.scheduling.inference_scheduler:_establish_whole_card_residency",
        sample=(
            "Whole-card residency: reserving the device for Z-Image-Turbo (inference processes 8 -> 1 of 8, "
            "target 1). Its weights + activations need nearly the whole card; co-resident siblings/safety "
            "would force the driver to stream activations to host RAM and run several times slower."
        ),
        dry_run_reason=_WHOLE_CARD_NOT_EXERCISED,
    ),
    _signature(
        "whole_card_establish_fields",
        r"Whole-card residency: reserving the device for (?P<model>.+?) \(inference processes "
        r"(?P<current>\d+) -> (?P<after>\d+) of (?P<total>\d+), target (?P<target>\d+)\)",
        emitter="process_management.scheduling.inference_scheduler:_establish_whole_card_residency",
        sample=(
            "Whole-card residency: reserving the device for Z-Image-Turbo (inference processes 8 -> 1 of 8, target 1)"
        ),
        dry_run_reason=_WHOLE_CARD_NOT_EXERCISED,
        field_of="whole_card_establish",
    ),
    _signature(
        "whole_card_declined",
        r"Declined a whole-card residency for",
        emitter="process_management.scheduling.inference_scheduler:_log_whole_card_declined",
        sample=(
            "Declined a whole-card residency for Z-Image-Turbo: its weights (~11800MB, 60% of the 19600MB "
            "card) do not dominate the device and the per-context overhead is unmeasured (using the "
            "conservative first-context fallback), so a teardown demand cannot be trusted. Serving it "
            "co-resident via model eviction instead of reserving the card."
        ),
        dry_run_reason=_WHOLE_CARD_NOT_EXERCISED,
    ),
    _signature(
        "pop_claim_engaged",
        r"Whole-card pop claim engaged for (?P<model>.+?): advertising that model alone",
        emitter="process_management.scheduling.governance.whole_card:disclose_edge",
        sample=(
            "Whole-card pop claim engaged for Z-Image-Turbo: advertising that model alone while it holds the card."
        ),
        dry_run_reason=_WHOLE_CARD_NOT_EXERCISED,
    ),
    _signature(
        "pop_claim_released",
        r"Whole-card pop claim released for (?P<model>.+?): (?P<release>[^;]+); advertising the full pool again",
        emitter="process_management.scheduling.governance.whole_card:disclose_edge",
        sample=(
            "Whole-card pop claim released for Z-Image-Turbo: the maximum hold elapsed; advertising the full "
            "pool again."
        ),
        dry_run_reason=_WHOLE_CARD_NOT_EXERCISED,
    ),
    # --- Stream forecast ---
    _signature(
        "stream_forecast",
        r"Stream forecast for (?P<model>.+?): ",
        emitter="process_management.scheduling.inference_scheduler:_log_stream_forecast",
        sample=(
            "Stream forecast for Z-Image-Turbo: weights ~11800 MB + 4135 MB reserve exceed 6653 MB free "
            "(after model evict: 15644 MB, alone: 15644 MB): whole-card baseline: sole residency on intent "
            "(evict sibling models) [free_now=6653.0, after_model_evict=15644.0, alone=15644.0, "
            "unreclaimable=0MB, live_procs=8, overhead/proc=204MB, marginal/ctx=198MB(src=probe,probe=198,"
            "idle_floor=?)] -> coresident=False, needs_exclusive=True, "
            "needs_process_count_reduction=False(max_resident=1), streams_unavoidably=False"
        ),
        dry_run_reason=_WHOLE_CARD_NOT_EXERCISED,
    ),
    _signature(
        "forecast_marginal",
        r"marginal/ctx=(?P<marginal>[\d?]+)MB\(src=(?P<source>\w+)",
        emitter="process_management.scheduling.inference_scheduler:_log_stream_forecast",
        sample="marginal/ctx=198MB(src=probe,probe=198,idle_floor=?)",
        dry_run_reason=_WHOLE_CARD_NOT_EXERCISED,
        field_of="stream_forecast",
    ),
    _signature(
        "forecast_unreclaimable",
        r"unreclaimable=(?P<unreclaimable>[\d.]+)MB",
        emitter="process_management.scheduling.inference_scheduler:_log_stream_forecast",
        sample="unreclaimable=0MB",
        dry_run_reason=_WHOLE_CARD_NOT_EXERCISED,
        field_of="stream_forecast",
    ),
    _signature(
        "forecast_reduction",
        r"needs_process_count_reduction=(?P<reduction>True|False)",
        emitter="process_management.scheduling.inference_scheduler:_log_stream_forecast",
        sample="needs_process_count_reduction=False(max_resident=1)",
        dry_run_reason=_WHOLE_CARD_NOT_EXERCISED,
        field_of="stream_forecast",
    ),
    # --- Blank/malformed model-name pop cascade ---
    _signature(
        "empty_model_pop",
        r"Popped job (?P<job>\S+) .*\(model: , ",
        emitter="process_management.jobs.job_popper:api_job_pop",
        sample="Popped job 019546e7-4a4a-4eea-afd6-d41078e14ac7 (35 eMPS) (model: , batch: 1, loras: False)",
        dry_run_reason=_MALFORMED_POP_NOT_EXERCISED,
    ),
    _signature(
        "blank_preload",
        r"Preloading model {2}on process (?P<pid>\d+)",
        emitter="process_management.scheduling.inference_scheduler:_send_preload",
        sample="Preloading model  on process 10",
        dry_run_reason=_MALFORMED_POP_NOT_EXERCISED,
    ),
    _signature(
        "blank_model_quarantine",
        r"Model {2}caused (?P<count>\d+) (?P<kind>\w+) incident\(s\)",
        emitter="process_management.lifecycle.process_lifecycle:record_model_incident",
        sample="Model  caused 3 load_failure incident(s) within 300s; quarantining it.",
        dry_run_reason=_MALFORMED_POP_NOT_EXERCISED,
    ),
    _signature(
        "blank_quarantine_skip",
        r"Skipping preload of quarantined model ;",
        emitter="process_management.scheduling.admission.executor:execute_commands",
        sample="Skipping preload of quarantined model ; faulting its job for reissue.",
        dry_run_reason=_MALFORMED_POP_NOT_EXERCISED,
    ),
    _signature(
        "malformed_pop_rejected",
        r"Popped job (?P<job>\S+) carries no model name \(got .*?\); returning it to the horde",
        emitter="process_management.jobs.job_popper:_reject_malformed_pop",
        sample=(
            "Popped job 019546e7-4a4a-4eea-afd6-d41078e14ac7 carries no model name (got ''); returning it to "
            "the horde for reissue."
        ),
        dry_run_reason=_MALFORMED_POP_NOT_EXERCISED,
    ),
    _signature(
        "blank_preload_refused",
        r"Refusing to preload a blank model name",
        emitter="process_management.workers.inference_process:preload_model",
        sample="Refusing to preload a blank model name (got '') for job 019546e7.",
        dry_run_reason=_MALFORMED_POP_NOT_EXERCISED,
    ),
    _signature(
        "blank_incident_refused",
        r"incident reported against a blank model name",
        emitter="process_management.lifecycle.process_lifecycle:record_model_incident",
        sample="Ignoring a load_failure incident reported against a blank model name (got '').",
        dry_run_reason=_MALFORMED_POP_NOT_EXERCISED,
    ),
    _signature(
        "load_failure_model",
        r"inference process replaced \(failed to load model (?P<model>.*)\)$",
        emitter="process_management.lifecycle.process_lifecycle:_replace_inference_process",
        sample="inference process replaced (failed to load model WAI-NSFW-illustrious-SDXL)",
        dry_run_reason=_NO_FAULT,
    ),
    # --- Pop API errors ---
    _signature(
        "pop_api_error",
        r"Failed to pop job \(API Error\): message='(?P<message>[^']*)'",
        emitter="process_management.jobs.job_popper:_handle_pop_error_response",
        sample="Failed to pop job (API Error): message='Invalid API key.' object_data=None rc='InvalidAPIKey'",
        dry_run_reason=_NO_FAULT,
    ),
    _signature(
        "pop_api_error_code",
        r"rc='(?P<code>[^']*)'",
        emitter="process_management.jobs.job_popper:_handle_pop_error_response",
        sample="rc='InvalidAPIKey'",
        dry_run_reason=_NO_FAULT,
        field_of="pop_api_error",
    ),
    # --- Stuck/stalled slots ---
    _signature(
        "stuck_step",
        r"stuck on a non-advancing sampling step|stuck-step watchdog",
        emitter="process_management.lifecycle.process_lifecycle:replace_hung_processes",
        sample=(
            "Inference slot 3 is stuck on a non-advancing sampling step (reported step 24/25 without "
            "advancing 3060 times); the ComfyUI generation will not return a result, replacing it "
            "(stuck-step watchdog)."
        ),
        dry_run_reason="the dry-run contract scenario's fake inference never loops on a non-advancing step",
    ),
    _signature(
        "post_processing_stall",
        r"seems to be stuck post processing",
        emitter="process_management.lifecycle.process_lifecycle:replace_hung_processes",
        sample="Process 1 seems to be stuck post processing (no progress for 120s); replacing it.",
        dry_run_reason="the dry-run contract scenario's fake post-processing lane never goes silent",
    ),
    _signature(
        "dedicated_post_process_activity",
        r"(?:last_process_state=HordeProcessState\.(?:INFERENCE_POST_PROCESSING|POST_PROCESSING)\b|"
        r"Post-processing (?:job|for job|finished for job) [0-9a-fA-F]{8})",
        emitter=(
            "process_management.ipc.message_dispatcher:_dispatch_buffered_message wrapping "
            "process_management.lifecycle.process_info:HordeProcessInfo.__repr__"
        ),
        sample="Post-processing finished for job 284ef66f in 3.86 seconds on process 1.",
        flags=re.IGNORECASE,
    ),
    # --- VRAM readouts and breakers (hordelib-emitted, read verbatim from the child log) ---
    _signature(
        "low_vram_readout",
        r"Free VRAM: (?P<free_mb>\d+) MB",
        emitter="hordelib.comfy.model_management:free_memory",
        sample="Free VRAM: 316 MB (+5272 MB reclaimable torch cache; comfy sees 5588)",
        dry_run_reason=_HORDELIB_READOUT_NOT_EXERCISED,
    ),
    _signature(
        "low_vram_reserve_warning",
        r"Free VRAM \d+ MB is below the \d+ MB inference reserve",
        emitter="hordelib.comfy.model_management:free_memory",
        sample=(
            "Free VRAM 11 MB is below the 1219 MB inference reserve: sampling activations will stream from host RAM."
        ),
        dry_run_reason=_HORDELIB_READOUT_NOT_EXERCISED,
    ),
    _signature(
        "post_processing_breaker_tripped",
        r"Post-processing fault breaker tripped",
        emitter="process_management.process_manager:_apply_post_processing_fault_breaker",
        sample="Post-processing fault breaker tripped: 5 post-processing over-commit fault(s) in the last 10 minutes.",
        dry_run_reason=_NO_FAULT,
    ),
    _signature(
        "wddm_paging",
        r"WDDM demand-paging detected on worker processes",
        emitter="process_management.scheduling.inference_scheduler:note_wddm_paging",
        sample="WDDM demand-paging detected on worker processes (elevated shared GPU memory on 2 process(es)).",
        dry_run_reason="WDDM demand-paging is a Windows driver signal the dry-run harness never produces",
    ),
    _signature(
        "post_processing_deferred",
        r"Deferring post-processing for job ([0-9a-f][0-9a-f-]{7,35})",
        emitter="process_management.workers.post_process_orchestrator:_ensure_lane_liveness_for_pending_work",
        sample=(
            "Deferring post-processing for job 3b678c36-1282-44b4-b062-f416ed7a1ba7: its chain (2629MB) "
            "cannot share the card with the sampling in progress; waiting for the card."
        ),
        dry_run_reason=_NOT_DETERMINISTIC,
    ),
    _signature(
        "post_processing_finished",
        r"Post-processing finished for job",
        emitter="process_management.ipc.message_dispatcher:_handle_post_process_result",
        sample="Post-processing finished for job 284ef66f in 3.86 seconds on process 1.",
    ),
    # --- Head-of-queue starvation (VRAM arbiter) ---
    _signature(
        "head_starvation_model",
        r"Head-of-queue (?P<model>.+?) deferred (?P<seconds>\d+)s >= \d+s with no verified progress",
        emitter="process_management.resources.vram_arbiter:_note_starvation_diagnostic",
        sample=(
            "Head-of-queue AlbedoBase XL (SDXL) deferred 130s >= 60s with no verified progress; it stays "
            "queued for the structural-wedge recovery supervisor to reroute. Measured: candidate 14573 MB vs "
            "available (device-free 19000 - reservations 0 - noise 819) = 18181 MB: does NOT fit."
        ),
        dry_run_reason=(
            "reachable only through a sustained scheduling starvation a short deterministic dry run does not sustain"
        ),
    ),
    _signature(
        "head_starvation_available",
        r"device-free (?P<free>\d+)",
        emitter="process_management.resources.vram_arbiter:_note_starvation_diagnostic",
        sample="device-free 19000 - reservations 0 - noise 819",
        dry_run_reason=(
            "reachable only through a sustained scheduling starvation a short deterministic dry run does not sustain"
        ),
        field_of="head_starvation_model",
    ),
]

SIGNATURES: dict[str, LogSignature] = {signature.name: signature for signature in _SIGNATURE_LIST}
"""Every worker log line the analysis package parses, keyed by signature name."""


def pattern_for(name: str) -> re.Pattern[str]:
    """Return the compiled pattern registered under ``name``.

    Args:
        name: A key of :data:`SIGNATURES`.

    Returns:
        The compiled pattern.

    Raises:
        KeyError: If no signature is registered under that name.
    """
    return SIGNATURES[name].pattern
