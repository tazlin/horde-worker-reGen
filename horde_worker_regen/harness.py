"""End-to-end harness for running the worker against canned job scenarios.

The harness runs the *real* orchestration layer (``HordeWorkerProcessManager`` and
its full asyncio main loop, with real OS child processes and real IPC primitives)
while letting the caller choose which heavy subsystems are real:

- **API**: ``skip_api=True`` replaces job pops/submits with a canned scenario and
  makes zero network calls. ``skip_api=False`` talks to the live AI Horde API.
- **Worker processes** (``process_mode``):
    - ``"fake"``: child processes run the protocol-faithful fakes from
      ``fake_worker_processes``; no hordelib/torch anywhere, no GPU needed.
    - ``"dry_run"``: the real ``HordeInferenceProcess``/``HordeSafetyProcess`` run,
      but skip model loading and inference (requires the ML deps installed).
    - ``"real"``: full production behavior (GPU, model downloads); benchmark mode.

This is the foundation used by the e2e tests and intended for the future
ramping-difficulty benchmark CLI.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import multiprocessing
import os
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE, MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_model_reference.model_reference_records import ImageGenerationModelRecord
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from horde_sdk.ai_horde_api.fields import GenerationID
from loguru import logger

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.consts import VRAM_HEAVY_MODELS
from horde_worker_regen.process_management.ipc.messages import AlchemyFormSpec
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.process_manager import (
    HordeWorkerProcessManager,
    SystemResources,
)
from horde_worker_regen.process_management.resources.device_info import TorchDeviceInfo, TorchDeviceMap
from horde_worker_regen.process_management.resources.run_metrics import RunMetricsSnapshot
from horde_worker_regen.process_management.simulation._canned_scenarios import (
    ArrivalSchedule,
    CannedAlchemySource,
    CannedJobSource,
    GeneratingAlchemySource,
    GeneratingJobSource,
    SoakAlchemyForm,
    SoakImageTemplate,
    TimedJobSource,
    make_canned_job,
    make_simple_scenario,
)
from horde_worker_regen.process_management.simulation.fake_worker_processes import (
    start_fake_download_process,
    start_fake_inference_process,
    start_fake_post_process_process,
    start_fake_safety_process,
    start_fake_vae_lane_process,
)
from horde_worker_regen.process_management.simulation.fault_injection import FaultProfile
from horde_worker_regen.process_management.simulation.sim_vram import SimVramLedger
from horde_worker_regen.process_management.worker_entry_points import ProcessEntryPoints
from horde_worker_regen.utils.gpu_monitor import GpuUtilizationSampler

if TYPE_CHECKING:
    from horde_worker_regen.benchmark.scenarios import Scenario

HarnessProcessMode = Literal["fake", "dry_run", "real"]

_REAL_BENCHMARK_STARTUP_TIMEOUT_SECONDS = 600
"""Startup budget (preload/process timeout) handed to a real-mode worker under benchmarking.

Benchmarking deliberately cold-starts the worker, and on heavier tiers loads large models, so a
cold ``hordelib.initialise`` plus first model load can take well over a minute. The production
stuck-start/all-unresponsive timers are tuned for fast API failover and would kill that legitimate
startup, triggering a kill/respawn storm that only deepens the contention. The level subprocess
timeout (and the lifecycle's dead-child detection) are the real backstops in a benchmark, so real
mode runs with this generous budget instead. Fake/dry-run start instantly and keep the defaults, so
the recovery machinery stays exercised by the tests."""


@dataclass
class HarnessConfig:
    """Describes one harness run."""

    scenario: list[ImageGenerateJobPopResponse] | None = None
    """The jobs to run. If None, a simple scenario of `num_jobs` Deliberate jobs is used."""

    num_jobs: int = 3
    """Number of jobs in the default scenario (ignored when `scenario` is provided)."""

    alchemy_forms: list[AlchemyFormSpec] | None = None
    """Alchemy forms to run alongside the image scenario (enables `alchemist` in the bridge data)."""

    arrival: ArrivalSchedule | None = None
    """When set, image jobs are released to the popper on this schedule instead of all at once."""

    process_mode: HarnessProcessMode = "fake"
    """Which child processes to launch: protocol-faithful fakes, real processes in
    dry-run mode, or fully real processes."""

    skip_api: bool = True
    """If True, job pops/submits are faked from the scenario and no network calls are made."""

    job_delay_seconds: float = 0.0
    """How long each fake/dry-run inference job pretends to take."""

    timeout_seconds: float = 120.0
    """Abort the run if the scenario has not completed within this time."""

    bridge_data_overrides: dict[str, object] = field(default_factory=dict)
    """Extra fields applied to the constructed bridge data (e.g. max_threads, queue_size)."""

    horde_model_reference_manager: ModelReferenceManager | None = None
    """Required for non-skip_api runs that need live model reference data; optional otherwise."""

    system_resources: SystemResources | None = None
    """Optional fake/dry-run hardware topology override for e2e simulations.

    Real-mode runs still detect the actual host. Fake/dry-run runs default to a single synthetic 8 GB
    card, but canary tests can inject varied RAM, card count, VRAM capacity, backend kind, and process
    overheads so the real manager plans the same topology an operator's machine would present.
    """

    fail_every_n: int = 0
    """If > 0, every nth fake inference job reports a faulted result (fake process mode only)."""

    inference_fault_profile: FaultProfile | None = None
    """If set (fake process mode only), scripts the inference fakes' misbehaviour (hang, crash,
    drop heartbeats, slow, OOM, corrupt message) so the chaos tests can probe the recovery paths."""

    safety_fault_profile: FaultProfile | None = None
    """If set (fake process mode only), scripts the safety fakes' misbehaviour on the eval path."""

    post_process_fault_profile: FaultProfile | None = None
    """If set (fake process mode only), scripts the post-processing fake's misbehaviour on the job path."""

    sim_vram_ledger: SimVramLedger | None = None
    """If set (fake process mode only), a shared simulated-device-VRAM ledger the inference fakes report
    from and allocate against. Paired with an ``inference_fault_profile`` carrying ``post_processing_peak_mb``
    it drives deterministic post-processing VRAM pressure (stall + recovery vs. completion) without a GPU.
    The caller owns the backing ``multiprocessing.Manager`` and must keep it alive for the run."""

    sim_inference_weights_mb: float = 0.0
    """Per-inference-process resident model-weight footprint (MB) to register on ``sim_vram_ledger`` when a
    model loads. Ignored without a ledger."""

    sim_inference_context_mb: float = 0.0
    """Per-inference-process fixed CUDA-context overhead (MB) to register on ``sim_vram_ledger`` at startup.
    Ignored without a ledger."""

    fake_initially_available_models: list[str] | None = None
    """Optional fake-mode model set present before the fake download process starts.

    None preserves the historical harness behavior: every scenario model is already present. Supplying a
    subset lets e2e simulations exercise cold-start/background-download availability without real downloads.
    """

    fake_download_delay_seconds: float = 0.0
    """Per-model delay for fake-mode image-model downloads."""

    fake_download_fail_models: list[str] = field(default_factory=list)
    """Image models the fake download process should report as failed in fake mode."""

    download_fault_profile: FaultProfile | None = None
    """If set (fake process mode only), scripts fake download process startup/slow behaviour."""

    audit: bool = True
    """If True, attach a JobLifecycleAuditor and report invariant violations in the result."""

    soak_seconds: float | None = None
    """When set, run a time-bounded sustained-load soak instead of a fixed scenario.

    Jobs (and alchemy forms) are *generated* continuously from `soak_image_templates`
    (and `soak_alchemy_templates`), minting fresh IDs each pop, keeping the worker
    saturated for this many seconds, after which generation stops and in-flight work is
    drained. Used by the post-ramp validation phase."""

    soak_image_templates: list[SoakImageTemplate] = field(default_factory=list)
    """Weighted job templates the soak generates image jobs from (required when soaking)."""

    soak_alchemy_templates: Sequence[SoakAlchemyForm] = field(default_factory=list)
    """Weighted ``(form, weight)`` or ``(form, weight, control_type)`` entries the soak generates alchemy
    forms from (optional). The control type carries an ``annotation`` form's detector identity."""

    soak_drain_timeout_seconds: float = 60.0
    """After the soak period, how long to wait for in-flight work to drain before shutting down."""

    on_progress: Callable[[RunMetricsSnapshot, float], None] | None = None
    """Optional best-effort progress hook, invoked roughly every ``progress_interval_seconds`` with the
    live run-metrics snapshot and elapsed seconds. The benchmark uses it to stream intra-level metrics;
    it is None for ordinary harness runs, leaving their behaviour unchanged."""

    progress_interval_seconds: float = 2.0
    """How often :attr:`on_progress` is sampled and invoked during a run."""

    @classmethod
    def from_scenario(
        cls,
        scenario: Scenario,
        *,
        process_mode: HarnessProcessMode,
        timeout_seconds: float,
        bridge_data_overrides: dict[str, object] | None = None,
        audit: bool = True,
        on_progress: Callable[[RunMetricsSnapshot, float], None] | None = None,
        progress_interval_seconds: float = 2.0,
    ) -> HarnessConfig:
        """Build a canned (``skip_api``) harness config from a :class:`Scenario`.

        The single seam every scenario-driven caller (benchmark CLI, e2e probes, the gpu catalog)
        uses, so one workload runs identically regardless of which driver launched it. The
        scenario's shape decides the mode: a soak streams generated jobs/forms for ``soak_seconds``,
        while a fixed scenario expands into a concrete job list released on its arrival schedule.
        Workload lives here; perturbation (faults, simulated VRAM, arrival overrides) stays on the
        low-level constructor for the chaos/stress/sim-VRAM tests that need it.
        """
        overrides = dict(bridge_data_overrides) if bridge_data_overrides is not None else {}
        if scenario.soak_seconds is not None:
            image_templates, alchemy_templates = scenario.to_soak_templates()
            return cls(
                soak_seconds=scenario.soak_seconds,
                soak_image_templates=image_templates,
                soak_alchemy_templates=alchemy_templates,
                process_mode=process_mode,
                skip_api=True,
                timeout_seconds=timeout_seconds,
                bridge_data_overrides=overrides,
                audit=audit,
                on_progress=on_progress,
                progress_interval_seconds=progress_interval_seconds,
            )
        arrival = scenario.arrival_schedule()
        return cls(
            # An empty list is a real (alchemy-only) scenario; None would trigger the default image scenario.
            scenario=scenario.expand_image_jobs(),
            alchemy_forms=scenario.expand_alchemy_forms() or None,
            arrival=arrival if arrival.kind != "all_at_once" else None,
            process_mode=process_mode,
            skip_api=True,
            timeout_seconds=timeout_seconds,
            bridge_data_overrides=overrides,
            audit=audit,
            on_progress=on_progress,
            progress_interval_seconds=progress_interval_seconds,
        )


@dataclass
class HarnessResult:
    """The outcome of one harness run."""

    num_jobs_expected: int
    num_jobs_completed: int
    num_jobs_faulted: int
    elapsed_seconds: float
    timed_out: bool
    started_at_epoch: float = 0.0
    """Wall-clock epoch the run's measured window began (set by the driver).

    For a full harness run this is the moment the worker began booting, so the gap to the first job's
    inference start measures process spawn plus engine/model cold-load; for a warm session it is the
    measured pass's start (boot already amortized). 0.0 means a path that did not record it (e.g. the
    on-disk subprocess reconstitution), in which case the startup/teardown split is reported as unknown."""
    audit_failures: list[str] = field(default_factory=list)
    """Invariant violations detected by the JobLifecycleAuditor (empty when auditing is off
    or the run timed out, since an aborted run purges the tracker)."""
    num_jobs_submitted_faulted: int = 0
    """How many jobs reached submission in a faulted state (per the auditor)."""
    exit_reason: str = ""
    """Human-readable reason why the run ended (e.g. 'completed', 'timed_out',
    'aborted_stale_abort_file', 'exception')."""
    diagnostics: list[str] = field(default_factory=list)
    """Non-fatal warnings and diagnostic messages collected during the run (e.g. zero
    processes started, no pops). Empty list means no diagnostics."""
    metrics: RunMetricsSnapshot | None = None
    """The worker-wide run metrics snapshot taken at the end of the run."""
    num_alchemy_forms_expected: int = 0
    num_alchemy_forms_completed: int = 0
    num_alchemy_forms_faulted: int = 0
    model_availability_known: bool = False
    """Whether the background download process reported image-model availability during the run."""
    available_model_names: list[str] = field(default_factory=list)
    """Final image-model present set reported by the download process, sorted."""
    failed_download_model_names: list[str] = field(default_factory=list)
    """Final image-model failures reported by the download process, sorted."""
    safety_gpu_pause_count: int = 0
    """Number of whole-card residency safety-off-GPU pauses initiated during the run."""
    safety_gpu_restore_count: int = 0
    """Number of whole-card residency safety-on-GPU restores initiated during the run."""
    num_jobs_completed_with_loras: int = 0
    """Finalized jobs that carried LoRA references and completed successfully (auditor-derived; 0 when
    auditing is off). The soak's LoRA traffic should complete at the same rate as the plain control group."""
    num_jobs_completed_without_loras: int = 0
    """Finalized plain (no-LoRA) jobs that completed successfully: the soak's liveness control group."""
    num_jobs_faulted_with_loras: int = 0
    """Finalized jobs that carried LoRA references and ended faulted. Any nonzero value on a LoRA-storm soak
    points at the pop-time prefetch / auxiliary-preparation path rather than inference itself."""
    num_jobs_faulted_without_loras: int = 0
    """Finalized plain (no-LoRA) jobs that ended faulted."""
    consecutive_failed_jobs_pause_count: int = 0
    """How many times the consecutive-failures backstop armed a pop pause during the run. The acceptance
    criterion for a scheduling-clean soak is that this stays zero: no scheduling-caused failure backoff
    fired. Give-up faults tagged ``SCHEDULING_RECOVERY`` are already excluded from the count that arms it."""

    boot_failed_no_progress: bool = False
    """True when a fixed-scenario run ended with no job accounted for (none completed, none faulted) via an
    early graceful shutdown rather than a timeout: the worker gave up (or never brought a child up) before it
    could do any work. Distinguishes this boot failure from a legitimately scored run so a driver cannot read
    a zero-work early exit as a real (if poor) result; the run's ``exit_reason`` carries the underlying cause."""

    @property
    def all_jobs_accounted_for(self) -> bool:
        """Whether every expected job either completed or faulted."""
        return (self.num_jobs_completed + self.num_jobs_faulted) >= self.num_jobs_expected

    @property
    def succeeded(self) -> bool:
        """Whether the run finished in time with everything completed and nothing faulted."""
        return (
            not self.timed_out
            and self.num_jobs_faulted == 0
            and (self.num_jobs_completed >= self.num_jobs_expected)
            and self.num_alchemy_forms_faulted == 0
            and (self.num_alchemy_forms_completed >= self.num_alchemy_forms_expected)
        )

    def failure_summary(self) -> str:
        """Return a compact summary suitable for assertion error messages."""
        parts: list[str] = []
        if self.exit_reason and self.exit_reason != "completed":
            parts.append(f"exit_reason={self.exit_reason}")
        if self.timed_out:
            parts.append("timed_out=True")
        if self.num_jobs_faulted > 0:
            parts.append(f"jobs_faulted={self.num_jobs_faulted}")
        if self.num_jobs_completed < self.num_jobs_expected:
            parts.append(f"jobs_completed={self.num_jobs_completed}/{self.num_jobs_expected}")
        if self.audit_failures:
            parts.append(f"audit_failures={len(self.audit_failures)}")
        if self.num_jobs_submitted_faulted > 0:
            parts.append(f"jobs_submitted_faulted={self.num_jobs_submitted_faulted}")
        if self.num_alchemy_forms_faulted > 0:
            parts.append(f"alchemy_faulted={self.num_alchemy_forms_faulted}")
        if self.num_alchemy_forms_completed < self.num_alchemy_forms_expected:
            parts.append(f"alchemy_completed={self.num_alchemy_forms_completed}/{self.num_alchemy_forms_expected}")
        if self.diagnostics:
            parts.append(f"diagnostics={self.diagnostics}")
        return "; ".join(parts) if parts else "no issues detected"


class JobLifecycleAuditor:
    """Records job lifecycle events on a JobTracker and verifies invariants post-run.

    The auditor wraps the tracker's ``record_popped_job`` and ``finalize_submitted``
    methods to count per-job events, then :meth:`verify` checks:

    1. Every popped job was finalized exactly once (none lost, none double-submitted).
    2. Nothing was finalized that was never popped.
    3. The tracker drained completely (no job left in any stage).

    Not suitable for cycling job sources, which legitimately re-pop the same
    generation IDs.
    """

    def __init__(self) -> None:
        """Initialize the auditor with empty event counts."""
        self.pop_counts: Counter[GenerationID] = Counter()
        self.finalize_counts: Counter[GenerationID] = Counter()
        self.num_jobs_submitted_faulted = 0
        # Terminal jobs split by whether they carried LoRA references, classified at finalize by the
        # submit state. These four counters partition every finalized job, so a soak can tell whether the
        # LoRA-carrying traffic (the pop-time-prefetch path) completed as cleanly as the plain control group.
        self.num_jobs_completed_with_loras = 0
        self.num_jobs_completed_without_loras = 0
        self.num_jobs_faulted_with_loras = 0
        self.num_jobs_faulted_without_loras = 0
        self._manager: HordeWorkerProcessManager | None = None

    def attach(self, manager: HordeWorkerProcessManager) -> None:
        """Wrap the manager's job tracker so lifecycle events are recorded."""
        tracker = manager._job_tracker
        original_record = tracker.record_popped_job
        original_finalize = tracker.finalize_submitted

        @functools.wraps(original_record)
        async def record_popped_job(job_pop_response, time_popped=None):  # type: ignore[no-untyped-def]  # noqa: ANN202, ANN001
            if job_pop_response.id_ is not None:
                self.pop_counts[job_pop_response.id_] += 1
            return await original_record(job_pop_response, time_popped)

        @functools.wraps(original_finalize)
        async def finalize_submitted(completed_job_info):  # type: ignore[no-untyped-def]  # noqa: ANN202, ANN001
            job_id = completed_job_info.sdk_api_job_info.id_
            if job_id is not None:
                self.finalize_counts[job_id] += 1
                self._record_terminal_lora_split(completed_job_info)
                if completed_job_info.state == GENERATION_STATE.faulted:
                    self.num_jobs_submitted_faulted += 1
            return await original_finalize(completed_job_info)

        tracker.record_popped_job = record_popped_job  # type: ignore[method-assign]
        tracker.finalize_submitted = finalize_submitted  # type: ignore[method-assign]
        self._manager = manager

    def _record_terminal_lora_split(self, completed_job_info: object) -> None:
        """Tally one finalized job into the LoRA-carrying / plain, completed / faulted partition.

        Classification reads the job's own submit payload (``payload.loras``) and its terminal state, so
        the split is self-contained per finalize event and needs no correlation back to the pop. A job
        finalized in the faulted state counts as faulted; anything else counts as completed.
        """
        from horde_worker_regen.process_management.jobs.job_models import HordeJobInfo

        if not isinstance(completed_job_info, HordeJobInfo):
            return
        has_loras = bool(completed_job_info.sdk_api_job_info.payload.loras)
        faulted = completed_job_info.state == GENERATION_STATE.faulted
        if faulted and has_loras:
            self.num_jobs_faulted_with_loras += 1
        elif faulted:
            self.num_jobs_faulted_without_loras += 1
        elif has_loras:
            self.num_jobs_completed_with_loras += 1
        else:
            self.num_jobs_completed_without_loras += 1

    def verify(self) -> list[str]:
        """Return a list of invariant violations observed over the run (empty = clean)."""
        failures: list[str] = []

        for job_id, count in self.finalize_counts.items():
            if job_id not in self.pop_counts:
                failures.append(f"Job {job_id} was finalized but never popped")
            if count > 1:
                failures.append(f"Job {job_id} was finalized {count} times (double submit)")

        for job_id in self.pop_counts:
            if self.finalize_counts.get(job_id, 0) == 0:
                failures.append(f"Job {job_id} was popped but never finalized (lost job)")

        tracker = self._manager._job_tracker if self._manager is not None else None
        if tracker is not None:
            if tracker.num_jobs_total != 0:
                failures.append(f"Tracker did not drain: {tracker.num_jobs_total} job(s) left in stages")
            if len(tracker.jobs_lookup) != 0:
                failures.append(f"Tracker lookup did not drain: {len(tracker.jobs_lookup)} entrie(s) left")

        return failures


def build_harness_bridge_data(config: HarnessConfig, scenario: list[ImageGenerateJobPopResponse]) -> reGenBridgeData:
    """Construct bridge data appropriate for the given harness configuration."""
    models_in_scenario = sorted({job.model for job in scenario if job.model is not None})

    # Field aliases (dreamer_name, models_to_load) are required here: the bridge data
    # model populates by alias, matching the on-disk config file format.
    bridge_data_fields: dict[str, object] = {
        "api_key": "0000000000",
        "dreamer_name": "e2e-harness-worker",
        "models_to_load": models_in_scenario,
        "max_threads": 1,
        "queue_size": 1,
        # Real runs mirror the production posture (safety rides the GPU); fake/dry-run children manage no
        # device, so safety must stay off it there. CPU-side safety in a real run distorts the measurement:
        # a stream of fast jobs saturates the CPU with checks production would run on-device.
        "safety_on_gpu": config.process_mode == "real",
        "cycle_process_on_model_change": False,
        "remove_maintenance_on_init": False,
        "exit_on_unhandled_faults": False,
        "suppress_speed_warnings": True,
        "dry_run_skip_api": config.skip_api,
        "dry_run_skip_inference": config.process_mode != "real",
        "dry_run_skip_safety": config.process_mode != "real",
        "dry_run_skip_post_processing": config.process_mode != "real",
        "dry_run_inference_delay": config.job_delay_seconds,
    }
    if config.alchemy_forms or config.soak_alchemy_templates:
        bridge_data_fields["alchemist"] = True
    # A workload that carries LoRA/TI references needs the worker to advertise LoRA support, or the
    # simulated pop matching (which honours the request exactly as the live API does) filters every
    # auxiliary-bearing job out of the run and the harness silently measures only the control group.
    carries_aux_references = any(bool(job.payload.loras) or bool(job.payload.tis) for job in scenario) or any(
        bool(template.loras) or bool(template.tis) for template in config.soak_image_templates
    )
    if carries_aux_references:
        bridge_data_fields["allow_lora"] = True
    carries_post_processing = any(bool(job.payload.post_processing) for job in scenario) or any(
        bool(template.post_processing) for template in config.soak_image_templates
    )
    if carries_post_processing:
        bridge_data_fields["allow_post_processing"] = True
    # max_power gates the largest resolution the pop request advertises (max_pixels = power * 8 * 64 * 64),
    # so it must cover the workload's largest job or the simulated pop matching silently filters every
    # heavier template and the run degrades to its smallest jobs.
    max_pixels_needed = max(
        [
            *(int(job.payload.width or 0) * int(job.payload.height or 0) for job in scenario),
            *(template.width * template.height for template in config.soak_image_templates),
            0,
        ],
    )
    if max_pixels_needed > 0:
        bridge_data_fields["max_power"] = max(8, -(-max_pixels_needed // (8 * 64 * 64)))
    if config.process_mode == "real":
        startup_budget = max(_REAL_BENCHMARK_STARTUP_TIMEOUT_SECONDS, int(config.timeout_seconds))
        bridge_data_fields["preload_timeout"] = startup_budget
        bridge_data_fields["process_timeout"] = startup_budget
    bridge_data_fields.update(config.bridge_data_overrides)
    bridge_data = reGenBridgeData(**bridge_data_fields)  # type: ignore[arg-type]
    # Prevent the manager from watching/reloading a bridge data file from disk.
    bridge_data._loaded_from_env_vars = True
    if config.process_mode == "real":
        # Real child processes read their configuration transports from the environment exactly as a
        # production worker's do (CivitAI download token, LoRA cache size and disk floor, cache home), and
        # every export is skipped when the variable is already set, so this mirrors run_worker's startup.
        bridge_data.load_env_vars()
    return bridge_data


def _fallback_baseline_for_harness_model(model_name: str) -> KNOWN_IMAGE_GENERATION_BASELINE:
    """Return a representative baseline for a synthetic harness model record."""
    if model_name == "Flux.1-Schnell fp8 (Compact)":
        return KNOWN_IMAGE_GENERATION_BASELINE.flux_schnell
    if model_name == "Flux.1-Schnell fp16 (Compact)":
        return KNOWN_IMAGE_GENERATION_BASELINE.flux_schnell
    if model_name == "Stable Cascade 1.0":
        return KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade
    if model_name in VRAM_HEAVY_MODELS:
        return KNOWN_IMAGE_GENERATION_BASELINE.flux_schnell
    return KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1


def build_harness_model_reference(
    scenario: list[ImageGenerateJobPopResponse],
    reference_manager: ModelReferenceManager | None = None,
) -> dict[str, ImageGenerationModelRecord]:
    """Build a model reference covering every model in the scenario.

    When a real reference manager is supplied, each scenario model resolves to its actual record
    (and therefore its real baseline, e.g. flux_1 for Flux rather than a blanket stable_diffusion_1).
    This matters for real-process benchmarks: the worker derives every VRAM/RAM burden estimate from
    the baseline, so a stubbed stable_diffusion_1 would make a heavy model (Flux) look like a small
    SD1.5 checkpoint and silently mask the very residency dynamics a real run is meant to exercise.
    Models genuinely absent from the real reference (synthetic test-only names) fall back to a minimal
    stable_diffusion_1 record so fake-process scenarios keep working without a populated reference.
    """
    real_reference: dict[str, ImageGenerationModelRecord] = {}
    if reference_manager is not None:
        resolved = reference_manager.get_model_reference(MODEL_REFERENCE_CATEGORY.image_generation)
        if isinstance(resolved, dict):
            real_reference = resolved

    reference: dict[str, ImageGenerationModelRecord] = {}
    for job in scenario:
        if job.model is None or job.model in reference:
            continue
        real_record = real_reference.get(job.model)
        if real_record is not None:
            reference[job.model] = real_record
            continue
        reference[job.model] = ImageGenerationModelRecord(
            name=job.model,
            baseline=_fallback_baseline_for_harness_model(job.model),
            nsfw=False,
            description="e2e harness model record",
        )
    return reference


def _build_harness_system_resources() -> SystemResources:
    """Fake hardware info so the harness never probes torch/psutil in the main process."""
    return SystemResources(
        total_ram_bytes=32 * 1024 * 1024 * 1024,
        device_map=TorchDeviceMap(
            root={
                0: TorchDeviceInfo(
                    device_name="HarnessGPU",
                    device_index=0,
                    total_memory=8 * 1024 * 1024 * 1024,
                ),
            },
        ),
    )


def _representative_soak_scenario(templates: list[SoakImageTemplate]) -> list[ImageGenerateJobPopResponse]:
    """One job per distinct model in the soak templates, for model-reference/bridge derivation.

    The soak's actual jobs are minted on demand by `GeneratingJobSource`; this list only
    exists so the harness can build a model reference and `models_to_load` covering them.
    """
    seen: dict[str, ImageGenerateJobPopResponse] = {}
    for template in templates:
        if template.model not in seen:
            seen[template.model] = make_canned_job(template.model, width=template.width, height=template.height)
    return list(seen.values())


def build_harness_process_manager(config: HarnessConfig) -> tuple[HordeWorkerProcessManager, int]:
    """Construct a process manager wired according to the harness configuration.

    Returns:
        The manager and the number of jobs the scenario expects to complete.
    """
    if config.soak_seconds is not None:
        scenario = _representative_soak_scenario(config.soak_image_templates)
    elif config.scenario is not None:
        scenario = config.scenario
    else:
        scenario = make_simple_scenario(config.num_jobs)

    bridge_data = build_harness_bridge_data(config, scenario)

    entry_points: ProcessEntryPoints | None = None
    if config.process_mode == "fake":
        # functools.partial of a module-level function stays picklable under spawn, so we can bind the
        # fault scripting (and the legacy fail_every_n) without losing the spawn-compatible target.
        inference_kwargs: dict[str, object] = {}
        if config.fail_every_n > 0:
            inference_kwargs["fail_every_n"] = config.fail_every_n
        if config.inference_fault_profile is not None:
            inference_kwargs["fault_profile"] = config.inference_fault_profile
        if config.sim_vram_ledger is not None:
            inference_kwargs["sim_vram_ledger"] = config.sim_vram_ledger
            inference_kwargs["sim_weights_mb"] = config.sim_inference_weights_mb
            inference_kwargs["sim_context_mb"] = config.sim_inference_context_mb
        inference_entry_point = (
            functools.partial(start_fake_inference_process, **inference_kwargs)
            if inference_kwargs
            else start_fake_inference_process
        )

        safety_entry_point = (
            functools.partial(start_fake_safety_process, fault_profile=config.safety_fault_profile)
            if config.safety_fault_profile is not None
            else start_fake_safety_process
        )

        post_process_kwargs: dict = {}
        if config.post_process_fault_profile is not None:
            post_process_kwargs["fault_profile"] = config.post_process_fault_profile
        if config.sim_vram_ledger is not None:
            post_process_kwargs["sim_vram_ledger"] = config.sim_vram_ledger
        post_process_entry_point = (
            functools.partial(start_fake_post_process_process, **post_process_kwargs)
            if post_process_kwargs
            else start_fake_post_process_process
        )

        available_models = (
            config.fake_initially_available_models
            if config.fake_initially_available_models is not None
            else [job.model for job in scenario if job.model is not None]
        )
        download_entry_point = functools.partial(
            start_fake_download_process,
            scripted_present=available_models,
            download_delay_seconds=config.fake_download_delay_seconds,
            fail_models=config.fake_download_fail_models,
            fault_profile=config.download_fault_profile,
        )

        entry_points = ProcessEntryPoints(
            inference_entry_point=inference_entry_point,
            safety_entry_point=safety_entry_point,
            post_process_entry_point=post_process_entry_point,
            vae_lane_entry_point=start_fake_vae_lane_process,
            download_entry_point=download_entry_point,
        )

    canned_job_source: CannedJobSource | None = None
    if config.skip_api:
        if config.soak_seconds is not None:
            canned_job_source = GeneratingJobSource(config.soak_image_templates)
        elif config.arrival is not None:
            canned_job_source = TimedJobSource(scenario, config.arrival)
        else:
            canned_job_source = CannedJobSource(scenario)

    canned_alchemy_source: CannedAlchemySource | None = None
    if config.skip_api and config.soak_seconds is not None and config.soak_alchemy_templates:
        canned_alchemy_source = GeneratingAlchemySource(config.soak_alchemy_templates)
    elif config.skip_api and config.alchemy_forms:
        canned_alchemy_source = CannedAlchemySource(config.alchemy_forms)

    system_resources = None
    if config.process_mode != "real":
        system_resources = config.system_resources or _build_harness_system_resources()

    manager = HordeWorkerProcessManager(
        ctx=multiprocessing.get_context("spawn"),
        bridge_data=bridge_data,
        horde_model_reference_manager=config.horde_model_reference_manager,
        system_resources=system_resources,
        skip_api_init=True,
        stable_diffusion_reference=build_harness_model_reference(scenario, config.horde_model_reference_manager),
        process_entry_points=entry_points,
        canned_job_source=canned_job_source,
        canned_alchemy_source=canned_alchemy_source,
        enable_background_downloads=config.process_mode == "real"
        or (config.process_mode == "fake" and config.fake_initially_available_models is not None),
    )

    return manager, len(scenario)


async def _watch_for_scenario_completion(
    manager: HordeWorkerProcessManager,
    *,
    num_jobs_expected: int,
    num_forms_expected: int = 0,
    timeout_seconds: float,
) -> bool:
    """Trigger shutdown once all jobs and alchemy forms are accounted for, or abort on timeout.

    Returns:
        True if the run timed out, False otherwise.
    """
    time_started = time.time()

    while True:
        await asyncio.sleep(0.1)

        jobs_accounted_for = manager._job_tracker.total_num_completed_jobs + manager._job_tracker.num_jobs_faulted
        coordinator = manager._alchemy_coordinator
        forms_accounted_for = coordinator.num_canned_forms_completed + coordinator.num_canned_forms_faulted
        if jobs_accounted_for >= num_jobs_expected and forms_accounted_for >= num_forms_expected:
            logger.info(
                f"Harness scenario complete ({jobs_accounted_for}/{num_jobs_expected} jobs, "
                f"{forms_accounted_for}/{num_forms_expected} alchemy forms accounted for)",
            )
            manager._shutdown()
            return False

        if time.time() - time_started > timeout_seconds:
            logger.error(
                f"Harness timed out after {timeout_seconds}s with "
                f"{jobs_accounted_for}/{num_jobs_expected} jobs and "
                f"{forms_accounted_for}/{num_forms_expected} alchemy forms accounted for",
            )
            manager._abort()
            return True


def _stop_soak_sources(manager: HordeWorkerProcessManager) -> None:
    """Tell the soak's generating sources to stop minting new work."""
    job_source = manager._job_popper._canned_job_source
    if isinstance(job_source, GeneratingJobSource):
        job_source.stop()
    alchemy_source = manager._alchemy_coordinator._canned_alchemy_source
    if isinstance(alchemy_source, GeneratingAlchemySource):
        alchemy_source.stop()


async def _watch_for_soak_period(
    manager: HordeWorkerProcessManager,
    *,
    soak_seconds: float,
    drain_timeout_seconds: float,
    timeout_seconds: float,
    gpu_sampler: GpuUtilizationSampler | None = None,
) -> bool:
    """Saturate the worker for `soak_seconds`, then stop generating and drain in-flight work.

    GPU utilization is sampled only after the first job completes (model is loaded, pipeline
    primed) and until the load phase ends, so the reported duty cycle reflects steady-state
    sustained load rather than cold start.

    Returns True only if the hard `timeout_seconds` was hit during the load phase (a genuine
    failure: the worker stopped making progress); completing the period and draining cleanly
    (or hitting the bounded drain timeout) returns False.
    """
    time_started = time.time()
    gpu_sampling_started = False

    # Phase 1: sustained load: the generating sources keep the worker saturated.
    while time.time() - time_started < soak_seconds:
        await asyncio.sleep(0.2)
        if not gpu_sampling_started and gpu_sampler is not None and manager._job_tracker.total_num_completed_jobs >= 1:
            gpu_sampler.start()
            gpu_sampling_started = True
        if manager._state.shutting_down:
            return False
        if time.time() - time_started > timeout_seconds:
            logger.error(f"Soak hit the hard timeout of {timeout_seconds}s during the load phase")
            manager._abort()
            return True

    if gpu_sampler is not None:
        gpu_sampler.stop()

    # Phase 2: drain: stop minting work and let everything already accepted finish.
    _stop_soak_sources(manager)
    logger.info(f"Soak period of {soak_seconds:.0f}s elapsed; draining in-flight work")
    drain_started = time.time()
    while time.time() - drain_started < drain_timeout_seconds:
        await asyncio.sleep(0.2)
        forms_in_flight = manager._state.alchemy_forms_in_flight
        if manager._job_tracker.num_jobs_total == 0 and forms_in_flight == 0:
            logger.info("Soak drain complete; all in-flight work finished")
            manager._shutdown()
            return False

    logger.warning(f"Soak drain did not finish within {drain_timeout_seconds:.0f}s; shutting down anyway")
    manager._shutdown()
    return False


def _cleanup_stale_abort_file() -> None:
    """Remove a stale `.abort` file left behind by a previous crashed or aborted run.

    The process manager checks for this sentinel file on every control-loop tick
    and aborts immediately if it exists.  Leaving one behind guarantees the next
    harness run will exit instantly with zero jobs processed.
    """
    cwd = os.getcwd()
    abort_path = os.path.join(cwd, ".abort")
    if os.path.exists(abort_path):
        logger.warning(f"Removing stale .abort file from {abort_path}")
        os.remove(abort_path)


async def _emit_progress_periodically(
    manager: HordeWorkerProcessManager,
    *,
    on_progress: Callable[[RunMetricsSnapshot, float], None],
    interval_seconds: float,
    time_started: float,
) -> None:
    """Sample the manager's live run metrics on a timer and hand them to the progress hook.

    Best-effort and side-effect-only: any error building or delivering a sample is logged at debug and
    the loop continues, so a flaky consumer can never disturb the run it is observing.
    """
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            snapshot = manager.get_run_metrics_snapshot()
            on_progress(snapshot, time.time() - time_started)
        except Exception as progress_error:  # noqa: BLE001 - progress sampling must never break the run
            logger.debug(f"Progress sampling failed: {progress_error}")


async def run_harness_async(config: HarnessConfig) -> HarnessResult:
    """Run a full worker lifecycle against the configured scenario and report the outcome."""
    from horde_worker_regen.telemetry import configure_telemetry

    configure_telemetry()

    # Remove any stale .abort sentinel before starting, so a previous crashed/
    # aborted run doesn't cause an immediate spurious abort.
    _cleanup_stale_abort_file()

    diagnostics: list[str] = []

    manager, num_jobs_expected = build_harness_process_manager(config)

    auditor: JobLifecycleAuditor | None = None
    if config.audit:
        auditor = JobLifecycleAuditor()
        auditor.attach(manager)

    time_started = time.time()

    num_forms_expected = len(config.alchemy_forms) if (config.alchemy_forms and config.skip_api) else 0

    # Sample real GPU core utilization across the soak's *steady-state* window (the watcher
    # starts it once warmed up and stops it before the drain), so cold-start model load does
    # not drag down the reported duty cycle.
    gpu_sampler = (
        GpuUtilizationSampler() if (config.process_mode == "real" and config.soak_seconds is not None) else None
    )

    if config.soak_seconds is not None:
        watcher_task = asyncio.create_task(
            _watch_for_soak_period(
                manager,
                soak_seconds=config.soak_seconds,
                drain_timeout_seconds=config.soak_drain_timeout_seconds,
                timeout_seconds=config.timeout_seconds,
                gpu_sampler=gpu_sampler,
            ),
        )
    else:
        watcher_task = asyncio.create_task(
            _watch_for_scenario_completion(
                manager,
                num_jobs_expected=num_jobs_expected,
                num_forms_expected=num_forms_expected,
                timeout_seconds=config.timeout_seconds,
            ),
        )

    progress_task: asyncio.Task[None] | None = None
    if config.on_progress is not None:
        progress_task = asyncio.create_task(
            _emit_progress_periodically(
                manager,
                on_progress=config.on_progress,
                interval_seconds=config.progress_interval_seconds,
                time_started=time_started,
            ),
        )

    exception_raised: BaseException | None = None
    try:
        await manager._main_loop()
    except Exception as exc:
        exception_raised = exc
        logger.exception(f"Harness main loop raised an exception: {exc}")
    finally:
        if gpu_sampler is not None:
            gpu_sampler.stop()
        if not watcher_task.done():
            watcher_task.cancel()
        if progress_task is not None and not progress_task.done():
            progress_task.cancel()
        # The main loop has returned, so this run's own teardown owns the process from here. Neutralize any
        # force-kill backstop it armed and join its thread before returning: an embedder that runs several
        # lifecycles in one interpreter must never inherit a thread that can later os._exit the process.
        manager._cancel_timed_shutdown()

    timed_out = False
    with contextlib.suppress(asyncio.CancelledError):
        timed_out = await watcher_task

    # Determine the exit reason for diagnostic purposes.
    # Drain any messages still in the IPC queue (e.g. metrics emitted by a child just
    # before shutdown), so the final run-metrics snapshot is complete.
    with contextlib.suppress(Exception):
        await manager.receive_and_handle_process_messages()

    exit_reason = _determine_exit_reason(
        manager=manager,
        num_jobs_expected=num_jobs_expected,
        timed_out=timed_out,
        exception_raised=exception_raised,
    )

    # Collect diagnostics about the run.
    diagnostics.extend(
        _collect_run_diagnostics(
            manager=manager,
            num_jobs_expected=num_jobs_expected,
            elapsed=time.time() - time_started,
        ),
    )

    audit_failures: list[str] = []
    num_jobs_submitted_faulted = 0
    lora_split = (0, 0, 0, 0)
    if auditor is not None:
        num_jobs_submitted_faulted = auditor.num_jobs_submitted_faulted
        lora_split = (
            auditor.num_jobs_completed_with_loras,
            auditor.num_jobs_completed_without_loras,
            auditor.num_jobs_faulted_with_loras,
            auditor.num_jobs_faulted_without_loras,
        )
        # An aborted (timed-out) run purges the tracker, so its invariants are meaningless.
        if not timed_out:
            audit_failures = auditor.verify()
            for failure in audit_failures:
                logger.error(f"Harness audit failure: {failure}")

    metrics_snapshot = manager.get_run_metrics_snapshot()
    if gpu_sampler is not None:
        metrics_snapshot = metrics_snapshot.model_copy(
            update={
                "gpu_utilization_mean_percent": gpu_sampler.mean_percent(),
                "gpu_utilization_busy_fraction": gpu_sampler.busy_fraction(),
                "gpu_utilization_samples": gpu_sampler.sample_count,
            },
        )
        # Opt-in diagnostics: persist the timestamped util series for offline correlation with
        # the sampling spans (e.g. util@1-active vs util@2-active to settle the duty ceiling).
        timeline_path = os.environ.get("BENCHMARK_GPU_TIMESERIES_PATH")
        if timeline_path:
            gpu_sampler.dump_timeline(timeline_path)

    num_jobs_completed = manager._job_tracker.total_num_completed_jobs
    num_forms_completed = manager._alchemy_coordinator.num_canned_forms_completed

    # A soak generates an open-ended amount of work, so "expected" is whatever it completed;
    # its pass/fail rests on faults, timeout, and (in the benchmark layer) throughput retention.
    if config.soak_seconds is not None:
        num_jobs_expected = num_jobs_completed
        num_forms_expected = num_forms_completed

    num_jobs_faulted = manager._job_tracker.num_jobs_faulted
    boot_failed_no_progress = _is_boot_failure_no_progress(
        is_soak=config.soak_seconds is not None,
        num_jobs_expected=num_jobs_expected,
        num_jobs_completed=num_jobs_completed,
        num_jobs_faulted=num_jobs_faulted,
        timed_out=timed_out,
        exit_reason=exit_reason,
    )
    if boot_failed_no_progress:
        logger.error(
            f"Harness run did no work: 0 of {num_jobs_expected} expected jobs completed or faulted, "
            f"ended by early shutdown (exit_reason={exit_reason!r}). Treating as a boot failure, not a "
            "scored run.",
        )

    availability = manager._model_availability
    return HarnessResult(
        num_jobs_expected=num_jobs_expected,
        num_jobs_completed=num_jobs_completed,
        num_jobs_faulted=num_jobs_faulted,
        elapsed_seconds=time.time() - time_started,
        started_at_epoch=time_started,
        timed_out=timed_out,
        audit_failures=audit_failures,
        num_jobs_submitted_faulted=num_jobs_submitted_faulted,
        exit_reason=exit_reason,
        diagnostics=diagnostics,
        metrics=metrics_snapshot,
        num_alchemy_forms_expected=num_forms_expected,
        num_alchemy_forms_completed=num_forms_completed,
        num_alchemy_forms_faulted=manager._alchemy_coordinator.num_canned_forms_faulted,
        model_availability_known=availability.is_known,
        available_model_names=sorted(availability.present or []),
        failed_download_model_names=sorted(availability.failed),
        safety_gpu_pause_count=manager._process_lifecycle.safety_gpu_pause_count,
        safety_gpu_restore_count=manager._process_lifecycle.safety_gpu_restore_count,
        num_jobs_completed_with_loras=lora_split[0],
        num_jobs_completed_without_loras=lora_split[1],
        num_jobs_faulted_with_loras=lora_split[2],
        num_jobs_faulted_without_loras=lora_split[3],
        consecutive_failed_jobs_pause_count=manager._state.consecutive_failed_jobs_pause_count,
        boot_failed_no_progress=boot_failed_no_progress,
    )


def _determine_exit_reason(
    *,
    manager: HordeWorkerProcessManager,
    num_jobs_expected: int,
    timed_out: bool,
    exception_raised: BaseException | None,
) -> str:
    """Produce a human-readable explanation of why the harness run ended."""
    if exception_raised is not None:
        return f"exception: {type(exception_raised).__name__}: {exception_raised}"
    if timed_out:
        return "timed_out"
    jobs_accounted = manager._job_tracker.total_num_completed_jobs + manager._job_tracker.num_jobs_faulted
    if jobs_accounted >= num_jobs_expected:
        return "completed"
    if manager._state.shut_down:
        return "shut_down_before_completion"
    return "unknown"


_EARLY_SHUTDOWN_EXIT_REASONS = frozenset({"shut_down_before_completion", "unknown"})
"""Exit reasons that mean the run ended by an early graceful shutdown rather than a timeout, an
exception, or completion: the worker decided it was done (gave up, or never had a live child) with the
scenario unfinished."""


def _is_boot_failure_no_progress(
    *,
    is_soak: bool,
    num_jobs_expected: int,
    num_jobs_completed: int,
    num_jobs_faulted: int,
    timed_out: bool,
    exit_reason: str,
) -> bool:
    """Whether a fixed-scenario run ended having done no work at all via an early shutdown.

    True only when image jobs were expected, none were accounted for (neither completed nor faulted), the
    run was not a soak (whose expected count is whatever it produced) and did not time out, and the exit
    reason signals an early graceful shutdown. That combination is a boot failure: the worker gave up or
    never brought a child to readiness before it could touch the scenario, which must not be mistaken for a
    real (if empty) result.
    """
    return (
        not is_soak
        and not timed_out
        and num_jobs_expected > 0
        and num_jobs_completed == 0
        and num_jobs_faulted == 0
        and exit_reason in _EARLY_SHUTDOWN_EXIT_REASONS
    )


def _collect_run_diagnostics(
    *,
    manager: HordeWorkerProcessManager,
    num_jobs_expected: int,
    elapsed: float,
) -> list[str]:
    """Collect non-fatal diagnostic messages about the run for debugging failures."""
    diags: list[str] = []

    num_inference = manager._process_map.num_inference_processes()
    num_safety = manager._process_map.num_safety_processes()
    if num_inference == 0:
        diags.append("No inference processes were started")
    if num_safety == 0:
        diags.append("No safety processes were started")

    completed = manager._job_tracker.total_num_completed_jobs
    faulted = manager._job_tracker.num_jobs_faulted
    if completed == 0 and faulted == 0 and elapsed > 2.0:
        diags.append(
            f"No jobs completed or faulted after {elapsed:.1f}s "
            f"(expected {num_jobs_expected}); check for stale .abort file or blocked job pop"
        )

    popped = len(manager._job_tracker.jobs_lookup)
    if popped == 0 and completed == 0 and faulted == 0 and elapsed > 2.0:
        diags.append("No jobs were ever popped; check canned_job_source or process availability")

    return diags


def run_harness(config: HarnessConfig) -> HarnessResult:
    """Synchronous wrapper around `run_harness_async`."""
    return asyncio.run(run_harness_async(config))


def _summarize_worker_processes(manager: HordeWorkerProcessManager) -> str:
    """A one-line ``id=state`` (with ``/DEAD`` when the OS process has exited) summary of every child.

    The warm benchmark's silent failure mode is a child that never reaches readiness or dies during
    startup; the level then just times out with zero context. Folding the per-process state into the
    readiness/timeout logs turns "0 jobs, timed out" into an actionable trace of which process is wedged
    or already gone (the matching startup-crash trace, if any, is in ``logs/bridge_*_startup.log``).
    """
    infos = list(manager._process_map.values())
    if not infos:
        return "no child processes in the process map"
    parts: list[str] = []
    for info in infos:
        alive = ""
        with contextlib.suppress(Exception):
            if not info.mp_process.is_alive():
                alive = "/DEAD"
        parts.append(f"{info.process_type.name.lower()}#{info.process_id}={info.last_process_state.name}{alive}")
    return ", ".join(parts)


def _all_inference_processes_dead(manager: HordeWorkerProcessManager) -> bool:
    """Whether the worker has inference processes and every one of them has exited at the OS level.

    Used to fast-fail a draining level against a wedged worker. ``False`` when no inference process has
    been launched yet (nothing to be dead) or when at least one is alive (including a freshly respawned
    replacement during recovery), so a normal recovery window does not trip it.
    """
    inference = [info for info in manager._process_map.values() if info.process_type == HordeProcessType.INFERENCE]
    if not inference:
        return False
    return all(not info.mp_process.is_alive() for info in inference)


def _warm_model_reference(
    model_names: list[str],
    reference_manager: ModelReferenceManager | None = None,
) -> dict[str, ImageGenerationModelRecord]:
    """Build a model reference covering every model the warm session may run.

    As with build_harness_model_reference, a supplied reference manager resolves real records (and
    real baselines) so a real warm benchmark exercises production VRAM/RAM dynamics; absent models
    fall back to a minimal stable_diffusion_1 record.
    """
    real_reference: dict[str, ImageGenerationModelRecord] = {}
    if reference_manager is not None:
        resolved = reference_manager.get_model_reference(MODEL_REFERENCE_CATEGORY.image_generation)
        if isinstance(resolved, dict):
            real_reference = resolved

    reference: dict[str, ImageGenerationModelRecord] = {}
    for name in model_names:
        real_record = real_reference.get(name)
        if real_record is not None:
            reference[name] = real_record
            continue
        reference[name] = ImageGenerationModelRecord(
            name=name,
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            nsfw=False,
            description="warm benchmark session model record",
        )
    return reference


def _build_warm_bridge_data(
    *,
    model_names: list[str],
    process_mode: HarnessProcessMode,
    max_threads_ceiling: int,
) -> reGenBridgeData:
    """Construct bridge data for a warm session covering every level's models."""
    fields: dict[str, object] = {
        "api_key": "0000000000",
        "dreamer_name": "warm-benchmark-worker",
        "models_to_load": sorted(set(model_names)),
        "max_threads": 1,
        "queue_size": 1,
        "alchemist": True,
        "safety_on_gpu": False,
        "cycle_process_on_model_change": False,
        "remove_maintenance_on_init": False,
        "exit_on_unhandled_faults": False,
        "suppress_speed_warnings": True,
        # Keep the inference semaphore the sole concurrency gate so the live effective cap governs
        # how many inferences run at once per level (no lease pre-staging to muddy the measurement).
        "gpu_sampling_lease_enabled": False,
        "dry_run_skip_api": True,
        "dry_run_skip_inference": process_mode != "real",
        "dry_run_skip_safety": process_mode != "real",
        "dry_run_skip_post_processing": process_mode != "real",
    }
    if process_mode == "real":
        # See _REAL_BENCHMARK_STARTUP_TIMEOUT_SECONDS: the warm worker cold-starts once, and must not
        # be torn down by the production startup timers before it finishes coming up.
        fields["preload_timeout"] = _REAL_BENCHMARK_STARTUP_TIMEOUT_SECONDS
        fields["process_timeout"] = _REAL_BENCHMARK_STARTUP_TIMEOUT_SECONDS
    bridge_data = reGenBridgeData(**fields)  # type: ignore[arg-type]
    bridge_data._loaded_from_env_vars = True
    return bridge_data


_WARMUP_DRAIN_TIMEOUT_SECONDS = 300.0
"""Bound on a level's pre-warm pass: enough for a cold heavy-model load plus one recovery-and-reload
cycle, after which the warm pass is abandoned and the measured pass runs anyway."""

_WARM_PROGRESS_INTERVAL_SECONDS = 1.0
"""How often the warm session samples the live worker metrics for the progress hook. Snappier than the
subprocess path's 2s so the reused warm worker's live card visibly advances while a level runs."""

_WARM_READINESS_HEARTBEAT_SECONDS = 15.0
"""How often the warm worker's (otherwise silent) wait for inference readiness logs a progress heartbeat
with the per-process states, so a slow cold start or a wedged child is visible in the log rather than a
minutes-long dark window."""

_WARM_DEAD_WORKER_GRACE_SECONDS = 15.0
"""How long every inference process may be dead (with no replacement coming up) before a draining level
abandons early instead of burning its full timeout against a worker that can never make progress. The
grace absorbs the normal recovery window, where a hung process is briefly dead before its replacement is
spawned; a genuine wedge (nothing respawns) persists past it. Levels default to a 900s timeout, so without
this a dead worker silently wastes ~15 minutes per level."""


class WarmHarnessSession:
    """A long-lived worker reused across benchmark levels, to eliminate per-level warm-up.

    Builds one :class:`HordeWorkerProcessManager` with real OS child processes, keeps it running, and
    runs each level by swapping the canned scenario, setting the concurrent-inference cap, and
    resetting per-level metrics, then waiting for the level's jobs to drain. The heavy child-process
    startup (hordelib/comfyui init) happens once for the whole ramp instead of once per level.

    The inference processes are launched once to the ceiling and never torn down between levels (that
    is the warmth); only the *effective* concurrency cap changes per level, so a level's ``max_threads``
    is honoured without respawning. Use as an async context manager.
    """

    def __init__(
        self,
        *,
        process_mode: HarnessProcessMode,
        model_names: list[str],
        max_threads_ceiling: int,
        horde_model_reference_manager: ModelReferenceManager | None = None,
    ) -> None:
        """Initialise the session description (the worker is built on ``__aenter__``)."""
        self._process_mode = process_mode
        self._model_names = sorted(set(model_names))
        self._max_threads_ceiling = max(1, max_threads_ceiling)
        self._horde_model_reference_manager = horde_model_reference_manager
        self._manager: HordeWorkerProcessManager | None = None
        self._loop_task: asyncio.Task[None] | None = None

    @property
    def manager(self) -> HordeWorkerProcessManager:
        """The underlying process manager (only valid inside the session)."""
        if self._manager is None:
            raise RuntimeError("WarmHarnessSession is not started")
        return self._manager

    def _build_manager(self) -> HordeWorkerProcessManager:
        entry_points: ProcessEntryPoints | None = None
        if self._process_mode == "fake":
            entry_points = ProcessEntryPoints(
                inference_entry_point=start_fake_inference_process,
                safety_entry_point=start_fake_safety_process,
                post_process_entry_point=start_fake_post_process_process,
                vae_lane_entry_point=start_fake_vae_lane_process,
            )
        system_resources = _build_harness_system_resources() if self._process_mode != "real" else None
        # Inject a minimal reference covering every level's models in all modes (mirrors
        # build_harness_process_manager); the real model load on disk uses hordelib's own reference.
        reference = _warm_model_reference(self._model_names, self._horde_model_reference_manager)
        return HordeWorkerProcessManager(
            ctx=multiprocessing.get_context("spawn"),
            bridge_data=_build_warm_bridge_data(
                model_names=self._model_names,
                process_mode=self._process_mode,
                max_threads_ceiling=self._max_threads_ceiling,
            ),
            horde_model_reference_manager=self._horde_model_reference_manager,
            system_resources=system_resources,
            skip_api_init=True,
            stable_diffusion_reference=reference,
            process_entry_points=entry_points,
            # Start idle with empty sources; each level installs its own via install_benchmark_scenario.
            canned_job_source=CannedJobSource([]),
            canned_alchemy_source=CannedAlchemySource([]),
            max_threads_ceiling=self._max_threads_ceiling,
            enable_background_downloads=self._process_mode == "real",
        )

    async def __aenter__(self) -> WarmHarnessSession:
        """Build the worker and start its main loop; wait briefly for processes to come up."""
        from horde_worker_regen.telemetry import configure_telemetry

        configure_telemetry()
        _cleanup_stale_abort_file()

        self._manager = self._build_manager()
        self._loop_task = asyncio.create_task(self._manager._main_loop())
        await self._wait_for_inference_ready()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Shut the worker down."""
        await self.aclose()

    async def _wait_for_inference_ready(self, timeout_seconds: float = 600.0) -> None:
        """Wait until an inference process is actually ready to take a job (warm gate).

        Waiting for a process to merely *exist* (the historical behaviour) let the first level start
        while the worker was still doing its minute-plus cold start, so the level "ran" against a
        not-yet-ready worker. Gate on real readiness instead: a process the scheduler would hand a job
        to (WAITING_FOR_JOB / PRELOADED).

        The wait is no longer silent. It logs when it begins, emits a heartbeat with the per-process
        states every :data:`_WARM_READINESS_HEARTBEAT_SECONDS` (so a slow cold start or a wedge is
        visible rather than a minutes-long dark window), logs the latency on success, and **fast-fails**
        the moment every launched inference process has exited: continuing to wait the full timeout for a
        child that has already died only hides the crash (whose trace is in ``logs/bridge_*_startup.log``).
        On a genuine timeout it logs and proceeds, leaving the per-level timeout as the backstop.
        """
        manager = self._manager
        if manager is None:
            return

        started = time.time()
        deadline = started + timeout_seconds
        expected = manager._process_map.num_inference_processes()
        logger.info(
            f"Warm worker: waiting up to {timeout_seconds:.0f}s for an inference process to become ready "
            f"({expected} inference process(es) launched). Process states: {_summarize_worker_processes(manager)}",
        )

        next_heartbeat = started + _WARM_READINESS_HEARTBEAT_SECONDS
        while time.time() < deadline:
            if manager._process_map.get_first_available_inference_process() is not None:
                logger.info(
                    f"Warm worker: inference process ready after {time.time() - started:.1f}s "
                    f"({_summarize_worker_processes(manager)})",
                )
                return

            now = time.time()
            inference_infos = [
                info for info in manager._process_map.values() if info.process_type == HordeProcessType.INFERENCE
            ]
            if inference_infos and all(not info.mp_process.is_alive() for info in inference_infos):
                logger.error(
                    f"Warm worker: every inference process exited during startup after {now - started:.1f}s "
                    f"without becoming ready ({_summarize_worker_processes(manager)}); abandoning the readiness "
                    "wait. Inspect logs/bridge_*_startup.log and logs/bridge_*.faulthandler for the cause.",
                )
                return

            if now >= next_heartbeat:
                logger.info(
                    f"Warm worker: still waiting for inference readiness ({now - started:.0f}s elapsed); "
                    f"process states: {_summarize_worker_processes(manager)}",
                )
                next_heartbeat = now + _WARM_READINESS_HEARTBEAT_SECONDS
            await asyncio.sleep(0.1)

        logger.warning(
            f"Warm worker: no inference process became ready within {timeout_seconds:.0f}s; "
            "starting levels anyway (the per-level timeout will catch a wedged worker). "
            f"Process states: {_summarize_worker_processes(manager)}",
        )

    async def aclose(self) -> None:
        """Gracefully shut the worker down and await its main loop."""
        if self._manager is not None and not self._manager._state.shutting_down:
            self._manager._shutdown()
        if self._loop_task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._loop_task, timeout=60.0)
        # The loop has returned, so neutralize any armed force-kill backstop before handing the interpreter
        # back to the caller: the warm session is a multi-lifecycle embedder and must leave no thread that
        # can later os._exit the process.
        if self._manager is not None:
            self._manager._cancel_timed_shutdown()

    async def _drain_installed_scenario(
        self,
        *,
        num_jobs_expected: int,
        num_forms_expected: int,
        base_completed: int,
        base_faulted: int,
        timeout_seconds: float,
    ) -> bool:
        """Wait until the currently installed scenario's jobs and alchemy forms are accounted for.

        Job completion is a delta over the job tracker's cumulative counters (the tracker is not reset
        between levels); the alchemy-form counters are reset by ``install_benchmark_scenario`` so they
        are read absolutely. Returns ``True`` if it drained within ``timeout_seconds``, ``False`` if it
        timed out.
        """
        manager = self.manager
        time_started = time.time()
        dead_since: float | None = None
        while True:
            await asyncio.sleep(0.1)
            completed = manager._job_tracker.total_num_completed_jobs - base_completed
            faulted = manager._job_tracker.num_jobs_faulted - base_faulted
            forms_done = (
                manager._alchemy_coordinator.num_canned_forms_completed
                + manager._alchemy_coordinator.num_canned_forms_faulted
            )
            if (completed + faulted) >= num_jobs_expected and forms_done >= num_forms_expected:
                return True
            if time.time() - time_started > timeout_seconds:
                return False

            # Fast-fail a wedged worker: if every inference process has been dead for longer than the
            # recovery grace, no further progress is possible, so abandon the level now (with a clear
            # reason) instead of waiting out the full, often minutes-long, timeout.
            now = time.time()
            if _all_inference_processes_dead(manager):
                if dead_since is None:
                    dead_since = now
                elif now - dead_since > _WARM_DEAD_WORKER_GRACE_SECONDS:
                    logger.error(
                        f"Warm level abandoned after {now - time_started:.0f}s: every inference process has been "
                        f"dead for over {_WARM_DEAD_WORKER_GRACE_SECONDS:.0f}s with no replacement "
                        f"({_summarize_worker_processes(manager)}); the worker cannot make progress. See "
                        "logs/bridge_*_startup.log and logs/bridge_*.faulthandler for the cause.",
                    )
                    return False
            else:
                dead_since = None

    async def run_level(
        self,
        *,
        jobs: list[ImageGenerateJobPopResponse] | None,
        alchemy_forms: list[AlchemyFormSpec] | None = None,
        threads: int = 1,
        timeout_seconds: float = 120.0,
        warmup: bool = False,
        on_progress: Callable[[RunMetricsSnapshot, float], None] | None = None,
    ) -> HarnessResult:
        """Run one fixed-scenario level on the warm worker and report its outcome.

        Completion is tracked by the delta in the job tracker's cumulative counters (the tracker is
        not reset between levels), so this returns once the level's own jobs and alchemy forms are
        accounted for, or when ``timeout_seconds`` elapses.

        When ``warmup`` is set, the level's full scenario is run once first to load any
        feature-specific weights (controlnet/QR checkpoints, upscaler/face-fixer/BLIP models) the warm
        worker has not touched yet. The whole scenario is warmed (not just its first job) because a
        sweep level loads a *distinct* model per variant (canny/depth/openpose controlnets; each
        upscaler/face-fixer), so warming only the first would leave the rest to cold-load while
        measured. A first cold load happens inside ``INFERENCE_STARTING`` (or blocks the safety loop)
        and can exceed ``inference_step_timeout``, tripping a benign process recovery that completes the
        work on the replacement. Absorbing that here makes the measured pass reflect steady state,
        matching a production worker that has preloaded its models. The measured pass re-installs the
        scenario, which resets the per-level metrics and the recovery counter, so the warmup's recovery
        never counts against the level.

        ``on_progress`` is invoked roughly every :data:`_WARM_PROGRESS_INTERVAL_SECONDS` with the live run
        metrics and the seconds elapsed since this call began, so a caller can stream the level's progress.
        """
        manager = self.manager
        scenario_jobs = jobs or []
        scenario_forms = alchemy_forms or []
        num_jobs_expected = len(scenario_jobs)
        num_forms_expected = len(alchemy_forms or [])

        manager._apply_set_concurrency(target_threads=threads, target_processes=None)

        # Stream live metrics for the whole call (warmup included) so the TUI/console live card advances
        # while the level runs. Without this the warm path is silent between LevelStarted and LevelFinished
        # (the subprocess path streams via the on-disk live file; the warm worker has no such file), which
        # is what made the live card look frozen. Started before the warmup drain so even a long cold
        # feature-model load reads as motion; the elapsed clock therefore spans warmup + measured.
        progress_task: asyncio.Task[None] | None = None
        call_started = time.time()
        if on_progress is not None:
            progress_task = asyncio.create_task(
                _emit_progress_periodically(
                    manager,
                    on_progress=on_progress,
                    interval_seconds=_WARM_PROGRESS_INTERVAL_SECONDS,
                    time_started=call_started,
                ),
            )

        try:
            if warmup and (scenario_jobs or scenario_forms):
                manager.install_benchmark_scenario(jobs=scenario_jobs, alchemy_forms=scenario_forms)
                await self._drain_installed_scenario(
                    num_jobs_expected=num_jobs_expected,
                    num_forms_expected=num_forms_expected,
                    base_completed=manager._job_tracker.total_num_completed_jobs,
                    base_faulted=manager._job_tracker.num_jobs_faulted,
                    timeout_seconds=min(timeout_seconds, _WARMUP_DRAIN_TIMEOUT_SECONDS),
                )

            base_completed = manager._job_tracker.total_num_completed_jobs
            base_faulted = manager._job_tracker.num_jobs_faulted

            manager.install_benchmark_scenario(jobs=scenario_jobs, alchemy_forms=alchemy_forms)

            time_started = time.time()
            drained = await self._drain_installed_scenario(
                num_jobs_expected=num_jobs_expected,
                num_forms_expected=num_forms_expected,
                base_completed=base_completed,
                base_faulted=base_faulted,
                timeout_seconds=timeout_seconds,
            )
            timed_out = not drained

            with contextlib.suppress(Exception):
                await manager.receive_and_handle_process_messages()
        finally:
            if progress_task is not None and not progress_task.done():
                progress_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await progress_task

        completed = manager._job_tracker.total_num_completed_jobs - base_completed
        faulted = manager._job_tracker.num_jobs_faulted - base_faulted

        # A warm level that times out previously returned ``timed_out=True`` with no explanation, which is
        # the warm-path equivalent of the subprocess path's "no useful logs". Collect the same diagnostics
        # the full harness does (processes started? jobs popped? counts?) plus a per-process state snapshot,
        # attach them to the result, and log them, so a 0/N timeout is self-explaining in the benchmark log.
        diagnostics: list[str] = []
        if timed_out:
            elapsed = time.time() - time_started
            diagnostics = _collect_run_diagnostics(
                manager=manager,
                num_jobs_expected=num_jobs_expected,
                elapsed=elapsed,
            )
            diagnostics.append(f"worker process states at timeout: {_summarize_worker_processes(manager)}")
            logger.warning(
                f"Warm level timed out after {elapsed:.0f}s with {completed}/{num_jobs_expected} jobs completed "
                f"({faulted} faulted, {num_forms_expected} alchemy forms expected); "
                f"diagnostics: {'; '.join(diagnostics)}",
            )

        return HarnessResult(
            num_jobs_expected=num_jobs_expected,
            num_jobs_completed=completed,
            num_jobs_faulted=faulted,
            elapsed_seconds=time.time() - time_started,
            started_at_epoch=time_started,
            timed_out=timed_out,
            exit_reason="timed_out" if timed_out else "completed",
            diagnostics=diagnostics,
            metrics=manager.get_run_metrics_snapshot(),
            num_alchemy_forms_expected=num_forms_expected,
            num_alchemy_forms_completed=manager._alchemy_coordinator.num_canned_forms_completed,
            num_alchemy_forms_faulted=manager._alchemy_coordinator.num_canned_forms_faulted,
        )
