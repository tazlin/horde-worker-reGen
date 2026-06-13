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
    TimedJobSource,
    make_simple_scenario,
)
from horde_worker_regen.process_management.device_info import TorchDeviceInfo, TorchDeviceMap
from horde_worker_regen.process_management.fake_worker_processes import (
    start_fake_inference_process,
    start_fake_safety_process,
)
from horde_worker_regen.process_management.messages import AlchemyFormSpec
from horde_worker_regen.process_management.process_manager import (
    HordeWorkerProcessManager,
    SystemResources,
)
from horde_worker_regen.process_management.run_metrics import RunMetricsSnapshot
from horde_worker_regen.process_management.worker_entry_points import ProcessEntryPoints

HarnessProcessMode = Literal["fake", "dry_run", "real"]


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

    audit: bool = True
    """If True, attach a JobLifecycleAuditor and report invariant violations in the result."""


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
    if config.alchemy_forms:
        bridge_data_fields["alchemist"] = True
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


def build_harness_process_manager(config: HarnessConfig) -> tuple[HordeWorkerProcessManager, int]:
    """Construct a process manager wired according to the harness configuration.

    Returns:
        The manager and the number of jobs the scenario expects to complete.
    """
    scenario = config.scenario if config.scenario is not None else make_simple_scenario(config.num_jobs)

    bridge_data = build_harness_bridge_data(config, scenario)

    entry_points: ProcessEntryPoints | None = None
    if config.process_mode == "fake":
        inference_entry_point = start_fake_inference_process
        if config.fail_every_n > 0:
            # functools.partial of a module-level function stays picklable under spawn
            inference_entry_point = functools.partial(
                start_fake_inference_process,
                fail_every_n=config.fail_every_n,
            )
        entry_points = ProcessEntryPoints(
            inference_entry_point=inference_entry_point,
            safety_entry_point=start_fake_safety_process,
        )

    canned_job_source: CannedJobSource | None = None
    if config.skip_api:
        if config.arrival is not None:
            canned_job_source = TimedJobSource(scenario, config.arrival)
        else:
            canned_job_source = CannedJobSource(scenario)

    canned_alchemy_source: CannedAlchemySource | None = None
    if config.skip_api and config.alchemy_forms:
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

    watcher_task = asyncio.create_task(
        _watch_for_scenario_completion(
            manager,
            num_jobs_expected=num_jobs_expected,
            num_forms_expected=num_forms_expected,
            timeout_seconds=config.timeout_seconds,
        ),
    )

    exception_raised: BaseException | None = None
    try:
        await manager._main_loop()
    except Exception as exc:
        exception_raised = exc
        logger.exception(f"Harness main loop raised an exception: {exc}")
    finally:
        if not watcher_task.done():
            watcher_task.cancel()

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

    return HarnessResult(
        num_jobs_expected=num_jobs_expected,
        num_jobs_completed=manager._job_tracker.total_num_completed_jobs,
        num_jobs_faulted=manager._job_tracker.num_jobs_faulted,
        elapsed_seconds=time.time() - time_started,
        timed_out=timed_out,
        audit_failures=audit_failures,
        num_jobs_submitted_faulted=num_jobs_submitted_faulted,
        exit_reason=exit_reason,
        diagnostics=diagnostics,
        metrics=manager.get_run_metrics_snapshot(),
        num_alchemy_forms_expected=num_forms_expected,
        num_alchemy_forms_completed=manager._alchemy_coordinator.num_canned_forms_completed,
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
