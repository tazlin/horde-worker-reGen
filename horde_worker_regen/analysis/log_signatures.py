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
) -> LogSignature:
    """Compile and describe one registry entry."""
    return LogSignature(
        name=name,
        pattern=re.compile(pattern),
        emitter=emitter,
        sample=sample,
        dry_run_reason=dry_run_reason,
        field_of=field_of,
    )


_NO_FAULT = "the dry-run contract scenario completes every job, so nothing faults"
_NOT_DETERMINISTIC = (
    "reachable only through a scheduling race a short deterministic run cannot be relied on to produce"
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
