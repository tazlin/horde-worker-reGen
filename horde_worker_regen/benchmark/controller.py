"""The ramp controller: runs ladder levels in isolated subprocesses and builds the report.

Each level runs via ``python -m horde_worker_regen.benchmark.level_runner`` so a level
that OOMs or hangs kills only its own process tree. The controller applies pre-flight
checks (disk/VRAM burden estimates), evaluates outcomes against the criteria, classifies
robustness findings, and applies the skip rules (a failed tier baseline skips the tier's
dependent levels; a failure on an axis stops higher rungs of that axis).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from hordelib.feature_impact import BurdenEstimate

from horde_worker_regen.benchmark.criteria import TierBaseline, evaluate_level
from horde_worker_regen.benchmark.enums import BenchAxis, BenchStage, BenchTier, FindingKind, LevelOutcome
from horde_worker_regen.benchmark.ladder import (
    _TIER_BASELINES,
    _TIER_RESOLUTIONS,
    BENCH_TIER_MODEL_POOLS,
    BENCH_TIER_MODELS,
    BETA_TIERS,
    HUGE_TIERS,
    RampLevel,
)
from horde_worker_regen.benchmark.memory_preflight import plan_soak_topology
from horde_worker_regen.benchmark.progress_channel import (
    LevelFinished,
    LevelLiveSnapshot,
    LevelPlanRow,
    LevelProgress,
    LevelStarted,
    NullProgressSink,
    ProgressSink,
    RampFinished,
    RampPlanned,
    RampStarted,
    RampStarting,
)
from horde_worker_regen.benchmark.report import (
    BenchmarkReport,
    Finding,
    HarnessSummary,
    LevelReport,
    LevelRunResult,
    MachineInfo,
    SuggestedBridgeData,
    compute_level_stats,
    render_markdown,
    synthesize_bridge_data,
    synthesize_capabilities,
)
from horde_worker_regen.benchmark.requirements import (
    LevelRequirements,
    civitai_token_available,
    compute_level_requirements,
    model_present_on_disk,
    requirement_skip_reason,
)
from horde_worker_regen.benchmark.soak import build_validation_level
from horde_worker_regen.process_management.owned_process_registry import kill_process_tree
from horde_worker_regen.process_management.worker_entry_points import WORKER_LOG_VERBOSITY_ENV

_SUBPROCESS_GRACE_SECONDS = 120.0
_LOG_TAIL_LINES = 100
_LEVEL_POLL_INTERVAL_SECONDS = 1.0
"""How often the controller waits on the level subprocess (and republishes its live metrics)."""
_SUBPROCESS_KILL_WAIT_SECONDS = 10.0
"""How long to wait for a killed (hung) level subprocess to actually exit."""

_SPAWN_TIMING_ENV = "AIWORKER_SPAWN_TIMING"

_MODEL_FILE_SUFFIXES = (".safetensors", ".ckpt", ".gguf")
"""Checkpoint file extensions the real-mode model-presence pre-flight looks for."""
_MODEL_SCAN_BUDGET_SECONDS = 5.0
"""Soft wall-clock budget for the model-presence scan; on expiry it fails open (the level proceeds)."""

_WARM_BOOT_HEARTBEAT_SECONDS = 5.0
"""How often the warm-worker boot republishes a startup-phase heartbeat so the (tens-of-seconds to
minutes) cold start reads as motion in the live view and TUI rather than a silent hang."""


def _weights_root_has_checkpoint(root: Path, *, budget_seconds: float = _MODEL_SCAN_BUDGET_SECONDS) -> bool | None:
    """Whether the weights root holds at least one checkpoint, scanning for at most ``budget_seconds``.

    Walks the tree once, testing every suffix per file, and returns on the first match. The previous
    implementation ran an unbounded ``rglob`` per suffix, so an empty-but-deep weights root (or a slow
    network volume) was walked three times in full: on a cold first level this was the bulk of the silent
    pre-flight stall. Returns None when the budget elapses before a verdict so the caller fails open
    (proceeds) rather than skipping the level on an inconclusive scan.
    """
    deadline = time.monotonic() + budget_seconds
    for _dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(_MODEL_FILE_SUFFIXES):
                return True
        if time.monotonic() >= deadline:
            return None
    return False


def _plan_row(req: LevelRequirements, verdict: str | None) -> LevelPlanRow:
    """Project a level's requirements and pre-flight verdict into a compact plan row."""
    return LevelPlanRow(
        level_id=req.level_id,
        stage=req.stage,
        tier=req.tier,
        estimated_vram_mb=req.estimated_vram_mb,
        min_disk_free_gb=req.min_disk_free_gb,
        requires_network=req.requires_network,
        requires_civitai_key=req.requires_civitai_key,
        features=req.features,
        will_run=verdict is None,
        verdict=verdict or "",
    )


def _log_system_snapshot(label: str) -> None:
    """Diagnostic (opt-in via ``AIWORKER_SPAWN_TIMING``): log python-process count, RSS, and GPU memory.

    Logged before and after each level so the deltas reveal child processes or VRAM that a prior level
    failed to release. Best-effort and never raises; a no-op when the env var is unset so it costs nothing
    in normal runs.

    Currently hardcoded to nvidia-smi but could be extended to AMD GPUs with rocm-smi or similar if needed.
    """
    if not os.environ.get(_SPAWN_TIMING_ENV):
        return
    py_count = -1
    py_rss_mb = -1.0
    with contextlib.suppress(Exception):
        import psutil

        procs = [p for p in psutil.process_iter(["name", "memory_info"]) if "python" in (p.info["name"] or "").lower()]
        py_count = len(procs)
        py_rss_mb = sum((p.info["memory_info"].rss for p in procs if p.info["memory_info"]), 0) / (1024 * 1024)
    vram = "n/a"
    with contextlib.suppress(Exception):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            vram = result.stdout.strip().replace("\n", " ; ")
    logger.info(
        f"[sys-snapshot] {label}: python_procs={py_count} python_rss_mb={py_rss_mb:.0f} vram_used/total_mb=[{vram}]",
    )


def _expected_jobs(level: RampLevel) -> int | None:
    """Return the number of image jobs a level expects, or None for an open-ended soak."""
    if level.scenario.soak_seconds is not None:
        return None
    expected = sum(job.count for job in level.scenario.image_jobs)
    return expected or None


def _override_int(level: RampLevel, key: str, default: int) -> int:
    """Read a positive integer bridge-data override off a level, falling back to *default*."""
    value = level.bridge_data_overrides.get(key, default)
    return int(value) if isinstance(value, (int, float)) and value >= 1 else default


_OOM_PATTERN = re.compile(r"CUDA out of memory|torch\.OutOfMemoryError|cudaErrorMemoryAllocation", re.IGNORECASE)
_WARM_LEVEL_TIMEOUT_MARGIN_SECONDS = 90.0
"""Extra wall-clock budget over a level's own timeout before the warm driver gives up on it."""


def _build_level_run_result(level_id: str, harness_result: object) -> object:
    """Build the on-disk ``LevelRunResult`` from a warm-session ``HarnessResult`` (mirrors level_runner)."""
    harness_dict = dataclasses.asdict(harness_result)  # type: ignore[call-overload]
    metrics = harness_dict.pop("metrics", None)
    return LevelRunResult(
        level_id=level_id,
        harness=HarnessSummary(**{k: v for k, v in harness_dict.items() if k in HarnessSummary.model_fields}),
        metrics=getattr(harness_result, "metrics", None) or metrics,
    )


class _WarmSessionDriver:
    """Runs a :class:`WarmHarnessSession` on a private event loop so the sync controller can drive it.

    The session's manager main loop runs as a task on a background thread's event loop; the controller
    submits per-level coroutines to it and blocks on the result. All manager state is therefore mutated
    only on the loop thread, avoiding cross-thread races.
    """

    def __init__(self, *, process_mode: str, model_names: list[str], max_threads_ceiling: int) -> None:
        """Create the driver for the given process mode, model union, and concurrency ceiling."""
        from horde_worker_regen.harness import WarmHarnessSession

        self._session = WarmHarnessSession(
            process_mode=process_mode,  # type: ignore[arg-type]
            model_names=model_names,
            max_threads_ceiling=max_threads_ceiling,
        )
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="warm-benchmark-loop", daemon=True)

    def start(
        self,
        timeout_seconds: float = 180.0,
        *,
        on_heartbeat: Callable[[float], None] | None = None,
        heartbeat_interval: float = _WARM_BOOT_HEARTBEAT_SECONDS,
    ) -> None:
        """Start the loop thread and bring the worker up (processes warm).

        Bringing the worker up (hordelib init plus the first model load(s)) is tens of seconds to a few
        minutes cold, all of it on the loop thread. When ``on_heartbeat`` is given the controller thread
        polls the boot future on ``heartbeat_interval`` and calls it with the elapsed seconds between
        polls, so the otherwise-dark window can be surfaced as progress; the boot itself is unaffected.
        """
        self._thread.start()
        future = asyncio.run_coroutine_threadsafe(self._session.__aenter__(), self._loop)
        if on_heartbeat is None:
            future.result(timeout=timeout_seconds)
            return
        start = time.monotonic()
        deadline = start + timeout_seconds
        while True:
            try:
                future.result(timeout=heartbeat_interval)
                return
            except TimeoutError:
                if time.monotonic() >= deadline:
                    future.cancel()
                    raise
                on_heartbeat(time.monotonic() - start)

    def run_level(
        self,
        *,
        jobs: list,  # type: ignore[type-arg]
        alchemy_forms: list | None,  # type: ignore[type-arg]
        threads: int,
        timeout_seconds: float,
        warmup: bool = False,
    ) -> object:
        """Run one level on the warm worker and return its ``HarnessResult``.

        When ``warmup`` is set the session runs a bounded pre-warm pass before measuring, so the
        blocking future is allowed that much extra wall-clock on top of the level's own budget.
        """
        from horde_worker_regen.harness import _WARMUP_DRAIN_TIMEOUT_SECONDS

        warmup_budget = _WARMUP_DRAIN_TIMEOUT_SECONDS if warmup else 0.0
        future = asyncio.run_coroutine_threadsafe(
            self._session.run_level(
                jobs=jobs,
                alchemy_forms=alchemy_forms,
                threads=threads,
                timeout_seconds=timeout_seconds,
                warmup=warmup,
            ),
            self._loop,
        )
        return future.result(timeout=timeout_seconds + warmup_budget + _WARM_LEVEL_TIMEOUT_MARGIN_SECONDS)

    def close(self) -> None:
        """Shut the worker down and stop the loop thread."""
        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(self._session.aclose(), self._loop).result(timeout=90.0)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10.0)


def detect_machine_info() -> MachineInfo:
    """Best-effort hardware detection (no-op on machines without torch/CUDA)."""
    info = MachineInfo()
    try:
        import psutil

        info.total_ram_bytes = psutil.virtual_memory().total
    except Exception:  # noqa: BLE001 - purely informational
        pass
    try:
        from hordelib.api import enumerate_accelerators

        accelerators = enumerate_accelerators()
        if accelerators:
            primary = accelerators[0]
            info.gpu_name = primary.name
            info.total_vram_mb = primary.total_vram_mb
    except Exception:  # noqa: BLE001 - purely informational
        pass
    return info


class BenchmarkController:
    """Drives the ramp: one isolated subprocess per level, then report synthesis."""

    def __init__(
        self,
        ladder: list[RampLevel],
        out_dir: Path,
        *,
        process_mode: str = "real",
        resume: bool = False,
        only_level: str | None = None,
        skip_downloads: bool = False,
        validate: bool = False,
        soak_seconds: float = 300.0,
        progress_sink: ProgressSink | None = None,
        warm: bool = False,
        verbose: bool = False,
        abort_on_catastrophe: bool = True,
        force: bool = False,
        machine: MachineInfo | None = None,
    ) -> None:
        """Initialize the controller.

        Args:
            ladder: The ordered levels to run.
            out_dir: Where level artifacts and the final report are written.
            process_mode: Passed through to the harness ("fake"/"dry_run"/"real").
            resume: Skip levels that already have a result JSON in `out_dir`.
            only_level: Run just this level id (the remediation-loop primitive).
            skip_downloads: Skip levels that require network access.
            validate: After the ramp, soak the synthesized config under sustained load.
            soak_seconds: How long each per-tier validation soak runs.
            progress_sink: Where to emit structured progress events; defaults to discarding them.
            warm: Run fixed-scenario levels against a single warm worker reused across the ramp
                (one child-process startup instead of one per level). Soak/validation levels still
                use their own isolated subprocess.
            verbose: Raise the spawned worker children's log verbosity to TRACE so their per-process
                logs capture the full cold-start detail (for diagnosing slow/wedged startups).
            abort_on_catastrophe: Stop the whole ramp the first time a level hangs, crashes, or a tier
                baseline does no work while its processes crash-loop. The worker stack is shared across
                every level, so a fundamental failure (a broken dependency, a wedged startup) repeats on
                every subsequent level; aborting turns ~30 wasted minutes into one finding. Disable to
                collect every level's outcome regardless (e.g. when characterising a flaky failure).
            force: Attempt levels that would otherwise be skipped for not fitting the machine
                (insufficient VRAM/disk) or lacking a CivitAI token. A genuinely-absent checkpoint is
                still skipped, since there is nothing to run.
            machine: Pre-detected hardware info; when None it is detected on ``run()``. The CLI passes
                the same info it used to size the ladder so detection happens once.
        """
        self._ladder = ladder
        self._out_dir = out_dir
        self._process_mode = process_mode
        self._resume = resume
        self._only_level = only_level
        self._skip_downloads = skip_downloads
        self._validate = validate
        self._soak_seconds = soak_seconds
        self._sink = progress_sink if progress_sink is not None else NullProgressSink()
        self._warm = warm
        self._verbose = verbose
        self._abort_on_catastrophe = abort_on_catastrophe
        self._force = force
        self._warm_driver: _WarmSessionDriver | None = None
        self._machine = machine

        if verbose:
            # Spawned worker children inherit this; resolve_worker_log_verbosity() maps 5 to TRACE.
            os.environ[WORKER_LOG_VERBOSITY_ENV] = "5"

        self._tier_baselines: dict[BenchTier, TierBaseline] = {}
        self._failed_tier_baselines: set[BenchTier] = set()
        self._failed_axes: set[tuple[BenchTier, BenchAxis]] = set()

    def run(self) -> BenchmarkReport:
        """Run the ramp and return (and persist) the full report."""
        self._out_dir.mkdir(parents=True, exist_ok=True)
        machine = self._machine if self._machine is not None else detect_machine_info()
        self._prewarm_inference_imports()
        self._warn_about_huge_tiers(machine)
        self._emit_ramp_started(machine)
        self._emit_ramp_plan(machine)
        reports: list[LevelReport] = []

        self._maybe_start_warm_driver(machine)
        aborted_reason: str | None = None
        try:
            for level_index, level in enumerate(self._ladder):
                if self._only_level is not None and level.id != self._only_level:
                    continue

                self._emit_level_started(level, level_index=level_index, num_levels=len(self._ladder))

                # Once the ramp has aborted, the remaining levels are recorded as skipped (with the
                # abort reason) so the report and progress stream stay complete, but none are run.
                if aborted_reason is not None:
                    report = LevelReport(level=level, outcome=LevelOutcome.SKIPPED, reasons=[aborted_reason])
                    reports.append(report)
                    self._emit_level_finished(report)
                    continue

                report = self._resolve_level_report(level, machine)
                reports.append(report)
                self._emit_level_finished(report)
                self._record_baseline_bookkeeping(level, report)

                if self._abort_on_catastrophe and self._is_catastrophic_outcome(level, report):
                    aborted_reason = (
                        f"ramp aborted after catastrophic failure in {level.id} ({report.outcome}): "
                        "the worker stack is shared across all levels, so a fundamental failure repeats; "
                        "skipping all remaining levels"
                    )
                    logger.error(aborted_reason)
        finally:
            self._stop_warm_driver()

        # Synthesize a preliminary recommendation to soak, then re-synthesize once the validation
        # levels are in so the soak's outcome can downgrade the recommendation (drop an unstable
        # tier's model, disable concurrent alchemy after an unstable soak).
        suggested = synthesize_bridge_data(reports, total_vram_mb=machine.total_vram_mb)

        if self._validate and self._only_level is None and self._tier_baselines:
            reports.extend(self._run_validation(suggested, machine))
            suggested = synthesize_bridge_data(reports, total_vram_mb=machine.total_vram_mb)

        benchmark_report = BenchmarkReport(
            run_id=self._out_dir.name,
            machine=machine,
            levels=reports,
            capabilities=synthesize_capabilities(reports, total_vram_mb=machine.total_vram_mb),
            suggested_bridge_data=suggested,
            tier_baselines_its={str(tier): baseline.its_p50 for tier, baseline in self._tier_baselines.items()},
        )

        (self._out_dir / "report.json").write_text(benchmark_report.model_dump_json(indent=2), encoding="utf-8")
        (self._out_dir / "report.md").write_text(render_markdown(benchmark_report), encoding="utf-8")
        self._emit_ramp_finished(benchmark_report)
        return benchmark_report

    def _resolve_level_report(self, level: RampLevel, machine: MachineInfo) -> LevelReport:
        """Produce a level's report by reusing a prior result, pre-flight skipping, or running it."""
        if self._resume:
            prior_result = self._load_result(self._out_dir / f"level_{level.id}.json")
            if prior_result is not None:
                logger.info(f"Resuming level {level.id} from its existing result")
                return self._evaluate_result(level, prior_result, machine, log_tail=[])

        self._emit_preflight_progress(level)
        skip_reason = self._pre_flight_skip_reason(level, machine)
        if skip_reason is not None:
            logger.warning(f"Skipping level {level.id}: {skip_reason}")
            return LevelReport(level=level, outcome=LevelOutcome.SKIPPED, reasons=[skip_reason])

        logger.info(f"Running level {level.id}: {level.description}")
        return self._run_level(level, machine)

    def _record_baseline_bookkeeping(self, level: RampLevel, report: LevelReport) -> None:
        """Update the tier-baseline and failed-axis tracking that drives later skip decisions."""
        if report.outcome == LevelOutcome.PASSED:
            if level.establishes_tier_baseline and report.stats is not None and report.stats.its_p50 is not None:
                self._tier_baselines[level.tier] = TierBaseline(tier=str(level.tier), its_p50=report.stats.its_p50)
            return
        if level.establishes_tier_baseline:
            self._failed_tier_baselines.add(level.tier)
        self._failed_axes.add((level.tier, level.axis))
        logger.warning(f"Level {level.id} did not pass: {'; '.join(report.reasons) or report.outcome}")

    def _is_catastrophic_outcome(self, level: RampLevel, report: LevelReport) -> bool:
        """Whether a level's outcome means the shared worker stack is broken (so the ramp should abort).

        Two shapes qualify, both distinct from a level that merely ran too slow (a normal ``failed`` that
        only skips its own tier/axis):

        - ``crashed`` / ``crashed_hang``: the level subprocess hung or died without a usable result, i.e.
          the worker process itself fell over. Every later level reuses that same stack.
        - a tier *baseline* that completed zero jobs while its processes crashed or had to be recovered:
          the simplest possible workload could not be served, so nothing heavier in the ramp will be.
        """
        if report.outcome in (LevelOutcome.CRASHED, LevelOutcome.CRASHED_HANG):
            return True
        if report.outcome == LevelOutcome.FAILED and level.establishes_tier_baseline:
            completed = report.harness.num_jobs_completed if report.harness is not None else 0
            fell_over = any(
                finding.kind in (FindingKind.CRASH, FindingKind.OOM, FindingKind.HANG, FindingKind.PROCESS_RECOVERY)
                for finding in report.findings
            )
            if completed == 0 and fell_over:
                return True
        return False

    # region progress events

    def _emit_ramp_started(self, machine: MachineInfo) -> None:
        """Announce the ramp: run identity, level count, tiers, and machine summary."""
        self._sink.emit(
            RampStarted(
                run_id=self._out_dir.name,
                num_levels=len(self._ladder),
                tiers=sorted({level.tier for level in self._ladder}),
                process_mode=self._process_mode,
                gpu_name=machine.gpu_name,
                total_vram_mb=machine.total_vram_mb,
            ),
        )

    def _emit_ramp_plan(self, machine: MachineInfo) -> None:
        """Publish the resource plan (one row per level) before the first level runs."""
        self._sink.emit(RampPlanned(run_id=self._out_dir.name, rows=self.build_plan_rows(machine)))

    def build_plan_rows(self, machine: MachineInfo) -> list[LevelPlanRow]:
        """Build the per-level resource plan and predicted verdict for this machine.

        Reuses the same pre-flight decision the ramp makes, so the preview cannot drift from the run.
        At plan time the failed-baseline/axis sets are empty, so the verdicts reflect the machine-fit
        and configuration gates an operator can act on (VRAM, disk, downloads, CivitAI key, --only-level).
        """
        rows: list[LevelPlanRow] = []
        for level in self._ladder:
            req = compute_level_requirements(level, present_resolver=model_present_on_disk)
            if self._only_level is not None and level.id != self._only_level:
                verdict: str | None = "not selected (--only-level)"
            else:
                verdict = self._pre_flight_skip_reason(level, machine)
            rows.append(_plan_row(req, verdict))
        return rows

    def _emit_level_started(self, level: RampLevel, *, level_index: int, num_levels: int) -> None:
        """Announce that a level is beginning (or about to be skipped)."""
        self._sink.emit(
            LevelStarted(
                level_id=level.id,
                description=level.description,
                stage=level.stage,
                tier=level.tier,
                axis=level.axis,
                level_index=level_index,
                num_levels=num_levels,
                jobs_expected=_expected_jobs(level),
                timeout_seconds=level.timeout_seconds,
            ),
        )

    def _emit_preflight_progress(self, level: RampLevel) -> None:
        """Emit a pre-flight phase event so the dark window before a level runs reads as motion.

        The pre-flight checks (disk free, VRAM burden, on-disk model presence) run synchronously on the
        controller thread after the level is announced but before its subprocess or warm run begins. On a
        cold first real level that window can be several seconds with no other output, which reads as a
        hang; this gives the live view a phase to show. Only emitted in real mode, where the checks
        actually do work (fake/dry-run pre-flight is a no-op).
        """
        if self._process_mode != "real":
            return
        self._sink.emit(
            LevelProgress(
                level_id=level.id,
                jobs_expected=_expected_jobs(level),
                phase="pre-flight checks (disk, VRAM, model presence)",
            ),
        )

    def _emit_level_finished(self, report: LevelReport) -> None:
        """Announce a level's outcome and headline statistics."""
        stats = report.stats
        self._sink.emit(
            LevelFinished(
                level_id=report.level.id,
                outcome=report.outcome,
                reasons=report.reasons,
                advisories=report.advisories,
                its_p50=stats.its_p50 if stats is not None else None,
                gpu_busy_percent=stats.gpu_utilization_mean_percent if stats is not None else None,
                vram_used_high_water_mb=stats.vram_used_high_water_mb if stats is not None else None,
                num_findings=len(report.findings),
            ),
        )

    def _emit_ramp_finished(self, report: BenchmarkReport) -> None:
        """Announce the ramp totals and the synthesized recommendation."""
        levels_passed = sum(1 for level in report.levels if level.outcome == LevelOutcome.PASSED)
        self._sink.emit(
            RampFinished(
                run_id=report.run_id,
                levels_passed=levels_passed,
                levels_total=len(report.levels),
                num_findings=len(report.findings),
                report_path=str(self._out_dir / "report.json"),
                suggested_bridge_data_yaml=report.suggested_bridge_data.as_yaml_block(),
            ),
        )

    # endregion

    def _run_validation(self, suggested: SuggestedBridgeData, machine: MachineInfo) -> list[LevelReport]:
        """Soak the synthesized config under sustained load, one stage-V level per passing tier."""
        validation_reports: list[LevelReport] = []
        for tier in self._tier_baselines:
            model_pool, pool_skip_reason = self._plan_soak_models(tier, suggested, machine)
            if pool_skip_reason is not None:
                logger.warning(f"Skipping validation V-{tier}-soak: {pool_skip_reason}")
                placeholder = build_validation_level(suggested, tier, soak_seconds=self._soak_seconds)
                self._emit_level_started(placeholder, level_index=0, num_levels=0)
                skipped_report = LevelReport(
                    level=placeholder,
                    outcome=LevelOutcome.SKIPPED,
                    reasons=[pool_skip_reason],
                )
                self._emit_level_finished(skipped_report)
                validation_reports.append(skipped_report)
                continue

            level = build_validation_level(suggested, tier, soak_seconds=self._soak_seconds, model_pool=model_pool)
            self._emit_level_started(level, level_index=0, num_levels=0)
            report = self._resolve_level_report(level, machine)
            self._emit_level_finished(report)
            if report.outcome != LevelOutcome.PASSED:
                logger.warning(
                    f"Validation {level.id} did not pass: {'; '.join(report.reasons) or report.outcome}",
                )
            validation_reports.append(report)

        return validation_reports

    def _plan_soak_models(
        self,
        tier: BenchTier,
        suggested: SuggestedBridgeData,
        machine: MachineInfo,
    ) -> tuple[list[str], str | None]:
        """Pick the soak's distinct-model pool, trimmed to what VRAM/RAM can hold resident.

        The soak wants one resident model per inference process (``max_threads + queue_size``)
        so every process is exercised and the 2-per-model pop cap never throttles it. But N
        co-resident models may not fit, so this runs the memory preflight: it trims the pool to
        the largest count that fits (logging the trim), or returns a skip reason when not even
        one model fits. Returns ``([single_model], None)`` when only one process or one pool
        entry is available, preserving the original single-model soak.
        """
        full_pool = BENCH_TIER_MODEL_POOLS.get(tier, [])
        num_inference_processes = max(1, suggested.max_threads + suggested.queue_size)
        desired = min(num_inference_processes, len(full_pool))

        if desired <= 1:
            single = full_pool[:1] or []
            return single, None

        # Only the real path downloads/holds models; fake/dry-run cannot OOM, so don't gate it.
        if self._process_mode != "real" or not machine.total_vram_mb:
            return full_pool[:desired], None

        try:
            from hordelib.api import estimate_job_burden

            resolution = _TIER_RESOLUTIONS[tier]
            burden = estimate_job_burden(
                baseline=_TIER_BASELINES[tier],
                width=resolution,
                height=resolution,
                batch=1,
            )
            plan = plan_soak_topology(
                desired_models=desired,
                per_model_vram_mb=burden.vram_mb,
                total_vram_mb=machine.total_vram_mb,
                per_model_ram_mb=float(burden.ram_mb),
                total_ram_mb=(machine.total_ram_bytes / (1024 * 1024)) if machine.total_ram_bytes else None,
            )
        except Exception as e:  # noqa: BLE001 - preflight must never crash the ramp
            logger.debug(f"Soak memory preflight unavailable for {tier}: {e}")
            return full_pool[:desired], None

        if not plan.is_viable:
            return [], plan.reason
        if not plan.fits:
            logger.warning(f"Soak pool for {tier}: {plan.reason}")
        return full_pool[: plan.fitting_models], None

    # region per-level

    @staticmethod
    def _model_cache_path() -> Path:
        """Return the volume where model downloads land (``AIWORKER_CACHE_HOME``, else cwd)."""
        cache_home = os.getenv("AIWORKER_CACHE_HOME")
        if cache_home:
            candidate = Path(cache_home)
            if candidate.exists():
                return candidate
        return Path.cwd()

    @staticmethod
    def _no_local_models_reason() -> str | None:
        """Real-mode guard: fast-skip when the resolved weights root holds no checkpoints at all.

        The non-warm benchmark path never downloads checkpoints on the fly, so an empty or
        misconfigured weights root (the classic ``AIWORKER_CACHE_HOME``-unset case) makes every
        inference child exit with "No models available" and the level would otherwise wedge until
        its timeout. Skipped when extra model directories are configured, since the checkpoints may
        legitimately live outside the primary root.
        """
        try:
            from horde_model_reference import resolve_weights_root

            from horde_worker_regen.model_download_plan import ENV_EXTRA_MODEL_DIRECTORIES

            if os.environ.get(ENV_EXTRA_MODEL_DIRECTORIES):
                return None
            root = resolve_weights_root(os.getenv("AIWORKER_CACHE_HOME"))
        except Exception as e:  # noqa: BLE001 - pre-flight must never block the ramp
            logger.debug(f"Model-presence pre-flight unavailable: {e}")
            return None

        hint = "set `cache_home` in bridgeData.yaml or AIWORKER_CACHE_HOME"
        if not root.exists():
            return f"no model weights root at {root}; {hint}"
        has_checkpoint = _weights_root_has_checkpoint(root)
        if has_checkpoint is None:
            logger.debug(f"Model-presence scan of {root} exceeded its time budget; proceeding without the guard")
            return None
        if has_checkpoint:
            return None
        return f"no model files found under {root}; {hint} (real-mode benchmarking does not download checkpoints)"

    def _warn_about_huge_tiers(self, machine: MachineInfo) -> None:
        """Log a prominent warning for any opted-in huge tier (flux/qwen) before the ramp starts.

        These models are 17-20 GB on disk and need 13-16 GB VRAM, so an operator who typed
        ``--tiers flux`` should see the cost up front. The pre-flight then skips the tier outright
        when the GPU cannot hold it or the checkpoint is not present.
        """
        huge_tiers = sorted({level.tier for level in self._ladder} & HUGE_TIERS)
        for tier in huge_tiers:
            burden = self._tier_burden(tier)
            disk_gb = burden.disk_bytes_needed / 1024**3 if burden is not None else None
            vram_gb = burden.vram_mb / 1024 if burden is not None else None
            detail = ""
            if disk_gb is not None and vram_gb is not None:
                detail = f" (~{disk_gb:.0f} GB download, ~{vram_gb:.0f} GB VRAM)"
            beta_note = " It is a beta model sourced from the pending reference." if tier in BETA_TIERS else ""
            logger.warning(
                f"Benchmarking the very large {tier} tier ({BENCH_TIER_MODELS[tier]}){detail}.{beta_note} "
                "It will be skipped automatically if this machine cannot hold it or the checkpoint is absent.",
            )

    @staticmethod
    def _tier_burden(tier: BenchTier) -> BurdenEstimate | None:
        """Return the hordelib burden estimate for a tier's baseline, or None when unavailable."""
        try:
            from hordelib.api import estimate_job_burden

            resolution = _TIER_RESOLUTIONS[tier]
            return estimate_job_burden(baseline=_TIER_BASELINES[tier], width=resolution, height=resolution, batch=1)
        except Exception as e:  # noqa: BLE001 - informational only, never blocks the ramp
            logger.debug(f"Burden estimate unavailable for {tier}: {e}")
            return None

    def _pre_flight_skip_reason(self, level: RampLevel, machine: MachineInfo) -> str | None:
        """Return why the level should be skipped without running, or None to proceed.

        The dynamic ramp gates live here (``--skip-downloads``, the failed-baseline/axis cascades, and
        the empty-weights-root guard); the per-level resource verdict (disk, model presence, VRAM, a
        CivitAI key) is delegated to :func:`requirement_skip_reason` so the ``plan`` preview and this
        decision share one source of truth.
        """
        if self._skip_downloads and level.requires_network:
            return "requires network access (--skip-downloads)"
        if level.establishes_tier_baseline and level.tier in self._failed_tier_baselines:
            pass  # a baseline level never skips itself
        elif level.tier in self._failed_tier_baselines:
            return f"tier {level.tier} baseline failed"
        if (level.tier, level.axis) in self._failed_axes and not level.establishes_tier_baseline:
            return f"axis {level.axis} already failed at a lower rung"

        # The empty-weights-root guard is a global (not per-level) check: the non-warm path never
        # downloads checkpoints, so a misconfigured root would wedge every level until timeout.
        if self._process_mode == "real":
            no_models_reason = self._no_local_models_reason()
            if no_models_reason is not None:
                return no_models_reason

        req = compute_level_requirements(level, present_resolver=model_present_on_disk)
        return requirement_skip_reason(
            req,
            machine=machine,
            process_mode=self._process_mode,
            cache_path=self._model_cache_path(),
            civitai_available=civitai_token_available(),
            force=self._force,
        )

    def _warm_session_params(self) -> tuple[list[str], int]:
        """Return the union of fixed-scenario level models and the max thread count to provision."""
        models: set[str] = set()
        ceiling = 1
        for level in self._ladder:
            if level.scenario.soak_seconds is not None:
                continue
            for job in level.scenario.expand_image_jobs():
                if job.model is not None:
                    models.add(job.model)
            ceiling = max(ceiling, _override_int(level, "max_threads", 1))
        return sorted(models), ceiling

    def _prewarm_inference_imports(self) -> None:
        """Import the hordelib burden API once, up front, so no individual level pays the cold import.

        ``estimate_job_burden`` (used by the per-level VRAM pre-flight) lives in ``hordelib.api``, whose
        first import pulls in torch and is tens of seconds cold. Machine detection usually pays this
        already, but doing it here under timing guarantees the per-level hot path is warm and surfaces the
        cost in the log rather than burying it inside the first level's otherwise-silent pre-flight.
        """
        if self._process_mode != "real":
            return
        start = time.monotonic()
        try:
            import hordelib.api  # noqa: F401 - imported for its side effect of warming the module cache
        except Exception as e:  # noqa: BLE001 - prewarm is best-effort; per-level guards already fail open
            logger.debug(f"Could not pre-warm hordelib inference imports: {e}")
            return
        logger.info(f"Pre-warmed hordelib inference imports in {time.monotonic() - start:.1f}s")

    def _maybe_start_warm_driver(self, machine: MachineInfo) -> None:
        """Bring up the shared warm worker, if warm mode is enabled and any level needs it."""
        if not self._warm or self._resume:
            return
        fixed_levels = [level for level in self._ladder if level.scenario.soak_seconds is None]
        if self._only_level is not None:
            fixed_levels = [level for level in fixed_levels if level.id == self._only_level]
        if not fixed_levels:
            return

        # Don't pay the (tens of seconds to minutes) worker-boot cost when every fixed level would be
        # skipped anyway: in real mode an insufficient-disk or no-checkpoints-on-disk machine fails the
        # pre-flight for all of them, so booting a worker only to tear it back down is pure dead time.
        # At boot the per-tier baseline/axis failure sets are still empty, so this sees the true
        # cold-machine verdict. Fake/dry-run never trip these gates, so they still boot.
        if not any(self._pre_flight_skip_reason(level, machine) is None for level in fixed_levels):
            logger.warning(
                "Skipping warm-worker startup: every fixed-scenario level fails pre-flight "
                "(see the per-level skip reasons that follow); recording them without booting a worker",
            )
            return

        model_names, ceiling = self._warm_session_params()
        logger.info(f"Starting warm benchmark worker (models: {model_names}, thread ceiling: {ceiling})")
        driver = _WarmSessionDriver(
            process_mode=self._process_mode,
            model_names=model_names,
            max_threads_ceiling=ceiling,
        )
        try:
            driver.start(on_heartbeat=lambda elapsed: self._emit_warm_boot_heartbeat(model_names, elapsed))
        except Exception as e:  # noqa: BLE001 - fall back to per-level subprocesses if warmup fails
            logger.error(f"Warm worker failed to start ({type(e).__name__}: {e}); using per-level subprocesses")
            with contextlib.suppress(Exception):
                driver.close()
            return
        self._warm_driver = driver

    def _emit_warm_boot_heartbeat(self, model_names: list[str], elapsed: float) -> None:
        """Republish a startup-phase event while the warm worker boots so the dark window reads as motion.

        Re-emitting :class:`RampStarting` (rather than a level event) matches how both consumers already
        render the pre-first-level window: the console prints a starting line and the TUI updates its
        ``startup_phase``, which a later ``LevelStarted`` clears.
        """
        self._sink.emit(
            RampStarting(
                run_id=self._out_dir.name,
                process_mode=self._process_mode,
                phase=(
                    f"starting warm worker: loading {len(model_names)} model(s), "
                    f"hordelib init + first model load ({elapsed:.0f}s elapsed)"
                ),
            ),
        )

    def _stop_warm_driver(self) -> None:
        """Tear down the shared warm worker, if one is running."""
        if self._warm_driver is None:
            return
        logger.info("Shutting down warm benchmark worker")
        self._warm_driver.close()
        self._warm_driver = None

    def _run_level_warm(self, level: RampLevel, machine: MachineInfo) -> LevelReport:
        """Run one fixed-scenario level on the shared warm worker and evaluate its outcome."""
        assert self._warm_driver is not None
        threads = _override_int(level, "max_threads", 1)
        logger.info(f"Running level {level.id} on warm worker (threads={threads})")

        # Feature and alchemy levels are the first to touch their model on the warm worker, so they
        # pre-warm to absorb the one-time cold-load process recovery; baseline/concurrency levels reuse
        # the already-warm base checkpoint and need no warmup.
        warmup = level.stage in (BenchStage.FEATURES, BenchStage.ALCHEMY)
        try:
            harness_result = self._warm_driver.run_level(
                jobs=level.scenario.expand_image_jobs(),
                alchemy_forms=level.scenario.expand_alchemy_forms() or None,
                threads=threads,
                timeout_seconds=level.timeout_seconds,
                warmup=warmup,
            )
        except Exception as e:  # noqa: BLE001 - a wedged warm level is a finding, not a ramp abort
            logger.error(f"Warm level {level.id} failed: {type(e).__name__}: {e}")
            return LevelReport(
                level=level,
                outcome=LevelOutcome.CRASHED,
                reasons=[f"warm worker raised running level: {type(e).__name__}: {e}"],
                findings=self._classify_findings(level, None, [], crashed=True),
            )

        result = _build_level_run_result(level.id, harness_result)
        with contextlib.suppress(OSError):
            (self._out_dir / f"level_{level.id}.json").write_text(
                result.model_dump_json(indent=2),  # type: ignore[attr-defined]
                encoding="utf-8",
            )
        return self._evaluate_result(level, result, machine, log_tail=[])  # type: ignore[arg-type]

    def _run_level(self, level: RampLevel, machine: MachineInfo) -> LevelReport:
        """Run one level and evaluate its outcome.

        Fixed-scenario levels run on the warm worker when one is active; soak levels (and every
        level without a warm driver) run in their own isolated subprocess.
        """
        if self._warm_driver is not None and level.scenario.soak_seconds is None:
            return self._run_level_warm(level, machine)

        level_json_path = self._out_dir / f"level_{level.id}.def.json"
        level_json_path.write_text(level.model_dump_json(indent=2), encoding="utf-8")
        result_path = self._out_dir / f"level_{level.id}.json"
        log_path = self._out_dir / f"level_{level.id}.log"
        heartbeat_path = self._out_dir / f"level_{level.id}.heartbeat"

        command = [
            sys.executable,
            "-m",
            "horde_worker_regen.benchmark.level_runner",
            "--level-json",
            str(level_json_path),
            "--out",
            str(self._out_dir),
            "--process-mode",
            self._process_mode,
        ]

        exit_code, hung = self._run_level_subprocess(level, command)

        log_tail = self._read_log_tail(log_path)
        result = self._load_result(result_path)

        if hung:
            findings = [
                Finding(
                    kind="hang",
                    level_id=level.id,
                    evidence=(
                        f"level subprocess exceeded {level.timeout_seconds + _SUBPROCESS_GRACE_SECONDS:.0f}s and "
                        f"was killed (last heartbeat: {self._heartbeat_age_description(heartbeat_path)})"
                    ),
                ),
            ]
            return LevelReport(
                level=level,
                outcome=LevelOutcome.CRASHED_HANG,
                reasons=["subprocess hung and was killed"],
                findings=findings,
                log_tail=log_tail,
            )

        if result is None:
            return LevelReport(
                level=level,
                outcome=LevelOutcome.CRASHED,
                reasons=[f"level subprocess died (exit code {exit_code}) without writing a result"],
                findings=self._classify_findings(level, None, log_tail, crashed=True),
                log_tail=log_tail,
            )

        return self._evaluate_result(level, result, machine, log_tail=log_tail)

    def _run_level_subprocess(self, level: RampLevel, command: list[str]) -> tuple[int, bool]:
        """Run the level subprocess to completion, streaming live progress and enforcing the timeout.

        Replaces a single blocking ``subprocess.run`` with a poll loop so a level's live metrics (written
        by the runner to ``level_<id>.live.json``) can be republished as progress events while it runs.
        Returns ``(exit_code, hung)``; on a timeout the subprocess is killed and ``hung`` is True.
        """
        live_path = self._out_dir / f"level_{level.id}.live.json"
        subprocess_log_path = self._out_dir / f"level_{level.id}.subprocess.log"
        deadline = time.time() + level.timeout_seconds + _SUBPROCESS_GRACE_SECONDS
        last_live_signature: str | None = None

        # Capture the level subprocess's stdout/stderr (and, by inheritance under spawn, its worker
        # children's) to a per-level file. This was previously discarded to DEVNULL, which is the main
        # reason a crashed or wedged child left "no useful logs": its traceback, OOM message, and
        # ComfyUI console output had nowhere to land. The structured `level_<id>.log` only carries the
        # manager's own loguru output, never the children's.
        _log_system_snapshot(f"pre-level {level.id}")
        subprocess_log = subprocess_log_path.open("w", encoding="utf-8", errors="replace")
        process = subprocess.Popen(command, stdout=subprocess_log, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    exit_code = process.wait(timeout=_LEVEL_POLL_INTERVAL_SECONDS)
                    return exit_code, False
                except subprocess.TimeoutExpired:
                    last_live_signature = self._emit_live_progress(level, live_path, last_live_signature)
                    if time.time() >= deadline:
                        # Kill the whole tree, not just the level runner: under spawn the runner's worker
                        # children (inference/safety/download) are grandchildren of this controller and
                        # would otherwise be orphaned (GPU still resident) when the runner is killed.
                        kill_process_tree(process.pid, grace_seconds=_SUBPROCESS_KILL_WAIT_SECONDS)
                        with contextlib.suppress(Exception):
                            process.wait(timeout=_SUBPROCESS_KILL_WAIT_SECONDS)
                        return -1, True
        finally:
            if process.poll() is None:
                with contextlib.suppress(Exception):
                    kill_process_tree(process.pid, grace_seconds=_SUBPROCESS_KILL_WAIT_SECONDS)
            subprocess_log.close()
            _log_system_snapshot(f"post-level {level.id}")

    def _emit_live_progress(self, level: RampLevel, live_path: Path, last_signature: str | None) -> str | None:
        """Emit a :class:`LevelProgress` event when the level's live snapshot has changed; return its signature."""
        live = self._read_live_snapshot(live_path)
        if live is None:
            return last_signature
        signature = live.model_dump_json()
        if signature == last_signature:
            return last_signature
        self._sink.emit(
            LevelProgress(
                level_id=level.id,
                jobs_completed=live.jobs_completed,
                jobs_faulted=live.jobs_faulted,
                jobs_expected=_expected_jobs(level),
                iterations_per_second=live.iterations_per_second,
                vram_used_mb=live.vram_used_mb,
                gpu_busy_percent=live.gpu_busy_percent,
                elapsed_seconds=live.elapsed_seconds,
                phase=live.phase,
                process_summary=live.process_summary,
                num_process_recoveries=live.num_process_recoveries,
            ),
        )
        return signature

    @staticmethod
    def _read_live_snapshot(live_path: Path) -> LevelLiveSnapshot | None:
        """Read the level's latest live snapshot, tolerating an absent or mid-write file."""
        if not live_path.exists():
            return None
        try:
            return LevelLiveSnapshot.model_validate_json(live_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _evaluate_result(
        self,
        level: RampLevel,
        result: LevelRunResult,
        machine: MachineInfo,
        *,
        log_tail: list[str],
    ) -> LevelReport:
        """Apply criteria and finding classification to a (fresh or resumed) raw result."""
        stats = compute_level_stats(result, total_vram_mb=machine.total_vram_mb)
        verdict = evaluate_level(stats, level.criteria, self._tier_baselines.get(level.tier))
        findings = self._classify_findings(level, result, log_tail, crashed=False)

        reasons = list(verdict.reasons)
        if result.runner_error is not None:
            reasons.append(f"runner error: {result.runner_error}")

        return LevelReport(
            level=level,
            outcome=LevelOutcome.PASSED if verdict.passed and result.runner_error is None else LevelOutcome.FAILED,
            reasons=reasons,
            advisories=verdict.advisories,
            stats=stats,
            harness=result.harness,
            findings=findings,
            log_tail=log_tail if (reasons or findings) else [],
        )

    # endregion

    # region findings

    def _classify_findings(
        self,
        level: RampLevel,
        result: LevelRunResult | None,
        log_tail: list[str],
        *,
        crashed: bool,
    ) -> list[Finding]:
        """Derive robustness findings from the run result and log output."""
        findings: list[Finding] = []
        log_text = "\n".join(log_tail)

        oom_match = _OOM_PATTERN.search(log_text)
        if oom_match:
            findings.append(
                Finding(
                    kind=FindingKind.OOM, level_id=level.id, evidence=f"OOM signature in log: {oom_match.group(0)}"
                ),
            )
        if crashed:
            findings.append(
                Finding(
                    kind=FindingKind.CRASH,
                    level_id=level.id,
                    evidence="level subprocess died without a result file",
                ),
            )

        if result is not None:
            for failure in result.harness.audit_failures:
                findings.append(
                    Finding(
                        kind=FindingKind.DOUBLE_SUBMIT if "double submit" in failure else FindingKind.LOST_JOB,
                        level_id=level.id,
                        evidence=failure,
                    ),
                )
            if result.metrics is not None:
                for crash in result.metrics.process_crash_events:
                    findings.append(
                        Finding(
                            kind=FindingKind.PROCESS_RECOVERY,
                            level_id=level.id,
                            evidence=(
                                f"process {crash.process_id} replaced (last state {crash.last_state}): {crash.reason}"
                            ),
                        ),
                    )
                for download in result.metrics.downloads:
                    if not download.success:
                        findings.append(
                            Finding(
                                kind=FindingKind.DOWNLOAD_STALL,
                                level_id=level.id,
                                evidence=f"download of {download.name} failed after {download.retries} retries",
                            ),
                        )

        return findings

    # endregion

    @staticmethod
    def _read_log_tail(log_path: Path) -> list[str]:
        if not log_path.exists():
            return []
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        return lines[-_LOG_TAIL_LINES:]

    @staticmethod
    def _heartbeat_age_description(heartbeat_path: Path) -> str:
        try:
            last_beat = float(heartbeat_path.read_text())
            return f"{time.time() - last_beat:.0f}s ago"
        except (OSError, ValueError):
            return "never"

    @staticmethod
    def _load_result(result_path: Path) -> LevelRunResult | None:
        if not result_path.exists():
            return None
        try:
            return LevelRunResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001 - a corrupt result is treated as a crash
            logger.error(f"Failed to parse level result {result_path}: {e}")
            return None


def load_existing_report(out_dir: Path) -> BenchmarkReport | None:
    """Load a previously written report.json from a benchmark output directory."""
    report_path = out_dir / "report.json"
    if not report_path.exists():
        return None
    return BenchmarkReport.model_validate_json(report_path.read_text(encoding="utf-8"))


__all__ = [
    "BenchmarkController",
    "HarnessSummary",
    "detect_machine_info",
    "load_existing_report",
]
