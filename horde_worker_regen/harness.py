"""End-to-end harness for running the worker against canned job scenarios.

The harness runs the *real* orchestration layer (``HordeWorkerProcessManager`` and
its full asyncio main loop, with real OS child processes and real IPC primitives)
while letting the caller choose which heavy subsystems are real:

- **API**: ``skip_api=True`` replaces job pops/submits with a canned scenario and
  makes zero network calls. ``skip_api=False`` talks to the live AI Horde API.
- **Worker processes** (``process_mode``):
    - ``"fake"``: child processes run the protocol-faithful fakes from
      ``fake_worker_processes`` — no hordelib/torch anywhere, no GPU needed.
    - ``"dry_run"``: the real ``HordeInferenceProcess``/``HordeSafetyProcess`` run,
      but skip model loading and inference (requires the ML deps installed).
    - ``"real"``: full production behavior (GPU, model downloads) — benchmark mode.

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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_model_reference.model_reference_records import ImageGenerationModelRecord
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse
from horde_sdk.ai_horde_api.fields import GenerationID
from loguru import logger

from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.process_management._canned_scenarios import (
    ArrivalSchedule,
    CannedAlchemySource,
    CannedJobSource,
    GeneratingAlchemySource,
    GeneratingJobSource,
    SoakImageTemplate,
    TimedJobSource,
    make_canned_job,
    make_simple_scenario,
)
from horde_worker_regen.process_management.device_info import TorchDeviceInfo, TorchDeviceMap
from horde_worker_regen.process_management.fake_worker_processes import (
    start_fake_inference_process,
    start_fake_safety_process,
)
from horde_worker_regen.process_management.fault_injection import FaultProfile
from horde_worker_regen.process_management.messages import AlchemyFormSpec
from horde_worker_regen.process_management.process_manager import (
    HordeWorkerProcessManager,
    SystemResources,
)
from horde_worker_regen.process_management.run_metrics import RunMetricsSnapshot
from horde_worker_regen.process_management.worker_entry_points import ProcessEntryPoints
from horde_worker_regen.utils.gpu_monitor import GpuUtilizationSampler

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

    fail_every_n: int = 0
    """If > 0, every nth fake inference job reports a faulted result (fake process mode only)."""

    inference_fault_profile: FaultProfile | None = None
    """If set (fake process mode only), scripts the inference fakes' misbehaviour (hang, crash,
    drop heartbeats, slow, OOM, corrupt message) so the chaos tests can probe the recovery paths."""

    safety_fault_profile: FaultProfile | None = None
    """If set (fake process mode only), scripts the safety fakes' misbehaviour on the eval path."""

    audit: bool = True
    """If True, attach a JobLifecycleAuditor and report invariant violations in the result."""

    soak_seconds: float | None = None
    """When set, run a time-bounded sustained-load soak instead of a fixed scenario.

    Jobs (and alchemy forms) are *generated* continuously from `soak_image_templates`
    (and `soak_alchemy_templates`) — minting fresh IDs each pop — keeping the worker
    saturated for this many seconds, after which generation stops and in-flight work is
    drained. Used by the post-ramp validation phase."""

    soak_image_templates: list[SoakImageTemplate] = field(default_factory=list)
    """Weighted job templates the soak generates image jobs from (required when soaking)."""

    soak_alchemy_templates: list[tuple[str, float]] = field(default_factory=list)
    """Weighted ``(form_name, weight)`` pairs the soak generates alchemy forms from (optional)."""

    soak_drain_timeout_seconds: float = 60.0
    """After the soak period, how long to wait for in-flight work to drain before shutting down."""

    on_progress: Callable[[RunMetricsSnapshot, float], None] | None = None
    """Optional best-effort progress hook, invoked roughly every ``progress_interval_seconds`` with the
    live run-metrics snapshot and elapsed seconds. The benchmark uses it to stream intra-level metrics;
    it is None for ordinary harness runs, leaving their behaviour unchanged."""

    progress_interval_seconds: float = 2.0
    """How often :attr:`on_progress` is sampled and invoked during a run."""


@dataclass
class HarnessResult:
    """The outcome of one harness run."""

    num_jobs_expected: int
    num_jobs_completed: int
    num_jobs_faulted: int
    elapsed_seconds: float
    timed_out: bool
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
                if completed_job_info.state == GENERATION_STATE.faulted:
                    self.num_jobs_submitted_faulted += 1
            return await original_finalize(completed_job_info)

        tracker.record_popped_job = record_popped_job  # type: ignore[method-assign]
        tracker.finalize_submitted = finalize_submitted  # type: ignore[method-assign]
        self._manager = manager

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
        "safety_on_gpu": False,
        "cycle_process_on_model_change": False,
        "remove_maintenance_on_init": False,
        "exit_on_unhandled_faults": False,
        "suppress_speed_warnings": True,
        "dry_run_skip_api": config.skip_api,
        "dry_run_skip_inference": config.process_mode != "real",
        "dry_run_skip_safety": config.process_mode != "real",
        "dry_run_inference_delay": config.job_delay_seconds,
    }
    if config.alchemy_forms or config.soak_alchemy_templates:
        bridge_data_fields["alchemist"] = True
    if config.process_mode == "real":
        startup_budget = max(_REAL_BENCHMARK_STARTUP_TIMEOUT_SECONDS, int(config.timeout_seconds))
        bridge_data_fields["preload_timeout"] = startup_budget
        bridge_data_fields["process_timeout"] = startup_budget
    bridge_data_fields.update(config.bridge_data_overrides)
    bridge_data = reGenBridgeData(**bridge_data_fields)  # type: ignore[arg-type]
    # Prevent the manager from watching/reloading a bridge data file from disk.
    bridge_data._loaded_from_env_vars = True
    return bridge_data


def build_harness_model_reference(
    scenario: list[ImageGenerateJobPopResponse],
) -> dict[str, ImageGenerationModelRecord]:
    """Build a minimal but real model reference covering every model in the scenario."""
    reference: dict[str, ImageGenerationModelRecord] = {}
    for job in scenario:
        if job.model is None or job.model in reference:
            continue
        reference[job.model] = ImageGenerationModelRecord(
            name=job.model,
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
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

        entry_points = ProcessEntryPoints(
            inference_entry_point=inference_entry_point,
            safety_entry_point=safety_entry_point,
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

    system_resources = _build_harness_system_resources() if config.process_mode != "real" else None

    manager = HordeWorkerProcessManager(
        ctx=multiprocessing.get_context("spawn"),
        bridge_data=bridge_data,
        horde_model_reference_manager=config.horde_model_reference_manager,
        system_resources=system_resources,
        skip_api_init=True,
        stable_diffusion_reference=build_harness_model_reference(scenario),
        process_entry_points=entry_points,
        canned_job_source=canned_job_source,
        canned_alchemy_source=canned_alchemy_source,
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
    failure — the worker stopped making progress); completing the period and draining cleanly
    (or hitting the bounded drain timeout) returns False.
    """
    time_started = time.time()
    gpu_sampling_started = False

    # Phase 1 — sustained load: the generating sources keep the worker saturated.
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

    # Phase 2 — drain: stop minting work and let everything already accepted finish.
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
    if auditor is not None:
        num_jobs_submitted_faulted = auditor.num_jobs_submitted_faulted
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

    return HarnessResult(
        num_jobs_expected=num_jobs_expected,
        num_jobs_completed=num_jobs_completed,
        num_jobs_faulted=manager._job_tracker.num_jobs_faulted,
        elapsed_seconds=time.time() - time_started,
        timed_out=timed_out,
        audit_failures=audit_failures,
        num_jobs_submitted_faulted=num_jobs_submitted_faulted,
        exit_reason=exit_reason,
        diagnostics=diagnostics,
        metrics=metrics_snapshot,
        num_alchemy_forms_expected=num_forms_expected,
        num_alchemy_forms_completed=num_forms_completed,
        num_alchemy_forms_faulted=manager._alchemy_coordinator.num_canned_forms_faulted,
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
    if popped == 0 and elapsed > 2.0:
        diags.append("No jobs were ever popped; check canned_job_source or process availability")

    return diags


def run_harness(config: HarnessConfig) -> HarnessResult:
    """Synchronous wrapper around `run_harness_async`."""
    return asyncio.run(run_harness_async(config))


def _warm_model_reference(model_names: list[str]) -> dict[str, ImageGenerationModelRecord]:
    """Build a minimal model reference covering every model the warm session may run."""
    return {
        name: ImageGenerationModelRecord(
            name=name,
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            nsfw=False,
            description="warm benchmark session model record",
        )
        for name in model_names
    }


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
    }
    if process_mode == "real":
        # See _REAL_BENCHMARK_STARTUP_TIMEOUT_SECONDS: the warm worker cold-starts once, and must not
        # be torn down by the production startup timers before it finishes coming up.
        fields["preload_timeout"] = _REAL_BENCHMARK_STARTUP_TIMEOUT_SECONDS
        fields["process_timeout"] = _REAL_BENCHMARK_STARTUP_TIMEOUT_SECONDS
    bridge_data = reGenBridgeData(**fields)  # type: ignore[arg-type]
    bridge_data._loaded_from_env_vars = True
    return bridge_data


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
            )
        system_resources = _build_harness_system_resources() if self._process_mode != "real" else None
        # Inject a minimal reference covering every level's models in all modes (mirrors
        # build_harness_process_manager); the real model load on disk uses hordelib's own reference.
        reference = _warm_model_reference(self._model_names)
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
        to (WAITING_FOR_JOB / PRELOADED). Best-effort: on timeout we log and proceed, leaving the
        per-level timeout as the backstop.
        """
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if (
                self._manager is not None
                and self._manager._process_map.get_first_available_inference_process() is not None
            ):
                return
            await asyncio.sleep(0.1)
        logger.warning(
            f"Warm worker: no inference process became ready within {timeout_seconds:.0f}s; "
            "starting levels anyway (the per-level timeout will catch a wedged worker)",
        )

    async def aclose(self) -> None:
        """Gracefully shut the worker down and await its main loop."""
        if self._manager is not None and not self._manager._state.shutting_down:
            self._manager._shutdown()
        if self._loop_task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._loop_task, timeout=60.0)

    async def run_level(
        self,
        *,
        jobs: list[ImageGenerateJobPopResponse] | None,
        alchemy_forms: list[AlchemyFormSpec] | None = None,
        threads: int = 1,
        timeout_seconds: float = 120.0,
    ) -> HarnessResult:
        """Run one fixed-scenario level on the warm worker and report its outcome.

        Completion is tracked by the delta in the job tracker's cumulative counters (the tracker is
        not reset between levels), so this returns once the level's own jobs and alchemy forms are
        accounted for, or when ``timeout_seconds`` elapses.
        """
        manager = self.manager
        scenario_jobs = jobs or []
        num_jobs_expected = len(scenario_jobs)
        num_forms_expected = len(alchemy_forms or [])

        base_completed = manager._job_tracker.total_num_completed_jobs
        base_faulted = manager._job_tracker.num_jobs_faulted

        manager._apply_set_concurrency(target_threads=threads, target_processes=None)
        manager.install_benchmark_scenario(jobs=scenario_jobs, alchemy_forms=alchemy_forms)

        time_started = time.time()
        timed_out = False
        while True:
            await asyncio.sleep(0.1)
            completed = manager._job_tracker.total_num_completed_jobs - base_completed
            faulted = manager._job_tracker.num_jobs_faulted - base_faulted
            forms_done = (
                manager._alchemy_coordinator.num_canned_forms_completed
                + manager._alchemy_coordinator.num_canned_forms_faulted
            )
            if (completed + faulted) >= num_jobs_expected and forms_done >= num_forms_expected:
                break
            if time.time() - time_started > timeout_seconds:
                timed_out = True
                break

        with contextlib.suppress(Exception):
            await manager.receive_and_handle_process_messages()

        completed = manager._job_tracker.total_num_completed_jobs - base_completed
        faulted = manager._job_tracker.num_jobs_faulted - base_faulted
        return HarnessResult(
            num_jobs_expected=num_jobs_expected,
            num_jobs_completed=completed,
            num_jobs_faulted=faulted,
            elapsed_seconds=time.time() - time_started,
            timed_out=timed_out,
            exit_reason="timed_out" if timed_out else "completed",
            metrics=manager.get_run_metrics_snapshot(),
            num_alchemy_forms_expected=num_forms_expected,
            num_alchemy_forms_completed=manager._alchemy_coordinator.num_canned_forms_completed,
            num_alchemy_forms_faulted=manager._alchemy_coordinator.num_canned_forms_faulted,
        )
