"""Launch and supervise a benchmark ramp as a subprocess for the TUI.

The TUI cannot run a benchmark in-process: a real ramp needs exclusive use of the GPU (so the worker must
be stopped first) and spawns its own level subprocesses. So the supervisor launches ``horde-benchmark ramp``
as a child process pointed at a known output directory, and follows its progress by tailing that run's
``progress.jsonl``, the same durable stream the CLI writes (see
[`progress_channel`][horde_worker_regen.benchmark.progress_channel]). This keeps the TUI decoupled from the
benchmark's heavy import chain: only light modules are imported at module load; the report/controller adapters
are imported lazily when a run finishes.

The benchmark's own console output (the live view the CLI prints) is redirected to ``console.log`` in the run
directory so it cannot corrupt the Textual terminal; the TUI renders from the structured event stream instead.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from horde_worker_regen.benchmark.progress_channel import (
    PROGRESS_FILENAME,
    BenchmarkProgressEvent,
    LevelFinished,
    LevelPlanRow,
    LevelProgress,
    LevelStarted,
    ProgressTailer,
    RampFinished,
    RampPlanned,
    RampStarted,
    RampStarting,
    SuggestionDecisionRow,
)
from horde_worker_regen.process_management.owned_process_registry import kill_process_tree
from horde_worker_regen.tui.config_form import DEFAULT_CONFIG_PATH, load_config, save_config

if TYPE_CHECKING:
    from horde_worker_regen.benchmark.report import BenchmarkReport, SuggestedBridgeData

_CANCEL_GRACE_SECONDS = 5.0

_SUGGESTED_CONFIG_KEYS = (
    "max_threads",
    "queue_size",
    "max_batch",
    "allow_lora",
    "allow_controlnet",
    "allow_post_processing",
    "models_to_load",
    "alchemist",
    "alchemy_allow_concurrent",
)
"""The bridgeData keys the benchmark recommendation writes when applied to the config file."""


class BenchmarkSupervisorStatus(enum.StrEnum):
    """The supervisor's view of the benchmark subprocess lifecycle."""

    IDLE = "idle"
    PREPARING = "preparing"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclasses.dataclass
class BenchmarkOptions:
    """The choices that parameterize a ramp launched from the TUI."""

    tiers: list[str] = dataclasses.field(default_factory=lambda: ["sd15", "sdxl"])
    process_mode: str = "real"
    validate: bool = True
    soak_minutes: float = 5.0
    include_downloads: bool = False
    include_concurrency: bool = True
    include_features: bool = True
    include_alchemy: bool = True
    excluded_axes: list[str] = dataclasses.field(default_factory=list)
    """Individual ramp axes to drop (BenchAxis values), independent of the coarse stage toggles."""
    warm: bool = True
    force: bool = False
    verbose: bool = False

    def _stage_selection_args(self) -> list[str]:
        """The stage-inclusion + tier flags shared by the ``ramp`` and ``plan`` argv."""
        args = ["--process-mode", self.process_mode, "--tiers", ",".join(self.tiers)]
        if self.include_downloads:
            args.append("--include-downloads")
        if not self.include_concurrency:
            args.append("--no-concurrency")
        if not self.include_features:
            args.append("--no-features")
        if not self.include_alchemy:
            args.append("--no-alchemy")
        for axis in self.excluded_axes:
            args.extend(["--exclude-axis", axis])
        if self.force:
            args.append("--force")
        return args

    def build_command(self, out_dir: Path) -> list[str]:
        """Return the ``horde-benchmark ramp`` argv that runs this configuration into ``out_dir``."""
        command = [
            sys.executable,
            # Unbuffered: the child's stdout/stderr is redirected to console.log (a regular file, not a
            # TTY), so CPython would otherwise block-buffer it and loguru's stderr sink would not reach
            # disk until the process exits. A wedged startup would then leave console.log empty, hiding
            # exactly the tracebacks/early logs needed to diagnose the hang. -u flushes them live.
            "-u",
            "-m",
            "horde_worker_regen.benchmark.cli",
            "ramp",
            *self._stage_selection_args(),
            "--out",
            str(out_dir),
            "--soak-minutes",
            str(self.soak_minutes),
        ]
        if not self.validate:
            command.append("--no-validate")
        if not self.warm:
            command.append("--no-warm")
        if self.verbose:
            command.append("--verbose")
        return command

    def build_plan_command(self) -> list[str]:
        """Return the ``horde-benchmark plan --json`` argv that previews this configuration (no worker)."""
        return [
            sys.executable,
            "-m",
            "horde_worker_regen.benchmark.cli",
            "plan",
            *self._stage_selection_args(),
            "--json",
        ]

    def build_download_command(self, *, dry_run: bool = False) -> list[str]:
        """Return the ``horde-benchmark download`` argv that fetches this configuration's models.

        The download path is always real-mode and never forced (it only fetches the checkpoints the selected
        tiers and stages reference), so it does not share ``_stage_selection_args`` (which carries
        ``--process-mode``/``--force``). ``dry_run`` previews the plan without downloading.
        """
        command = [
            sys.executable,
            # Unbuffered so the parent (the Download models modal) sees each progress line live rather than
            # in one block-buffered burst when the child exits.
            "-u",
            "-m",
            "horde_worker_regen.benchmark.cli",
            "download",
            "--tiers",
            ",".join(self.tiers),
            "--json-progress",
        ]
        if self.include_downloads:
            command.append("--include-downloads")
        if not self.include_concurrency:
            command.append("--no-concurrency")
        if not self.include_features:
            command.append("--no-features")
        if not self.include_alchemy:
            command.append("--no-alchemy")
        for axis in self.excluded_axes:
            command.extend(["--exclude-axis", axis])
        if dry_run:
            command.append("--dry-run")
        return command


@dataclasses.dataclass
class LevelState:
    """The live, accumulated state of one ramp level, built up from its progress events."""

    level_id: str
    description: str = ""
    stage: str = ""
    tier: str = ""
    axis: str = ""
    jobs_expected: int | None = None
    jobs_completed: int = 0
    jobs_faulted: int = 0
    iterations_per_second: float | None = None
    vram_used_mb: int | None = None
    gpu_busy_percent: float | None = None
    elapsed_seconds: float = 0.0
    phase: str = ""
    num_process_recoveries: int = 0
    outcome: str | None = None
    reasons: list[str] = dataclasses.field(default_factory=list)
    advisories: list[str] = dataclasses.field(default_factory=list)


class BenchmarkRunState:
    """The accumulated view of a ramp, reduced from its ordered progress events.

    Apply each event in arrival order with :meth:`apply`; the screen renders from the resulting state. Kept
    separate from the supervisor and the widget so the reduction is unit-testable on a synthetic event stream.
    """

    def __init__(self) -> None:
        """Start an empty run state (before any events have arrived)."""
        self.run_id: str = ""
        self.num_levels: int = 0
        self.tiers: list[str] = []
        self.process_mode: str = ""
        self.startup_phase: str = ""
        """A pre-level phase to display while the run has not produced any levels yet (worker stop, the
        subprocess's import/hardware-probe window). Cleared once the first level starts."""
        self.gpu_name: str | None = None
        self.plan_rows: list[LevelPlanRow] = []
        """The per-level resource plan and predicted verdicts, from the RampPlanned event."""
        self.level_order: list[str] = []
        self.levels: dict[str, LevelState] = {}
        self.current_level_id: str | None = None
        self.levels_passed: int = 0
        self.levels_total: int = 0
        self.num_findings: int = 0
        self.suggested_bridge_data_yaml: str = ""
        self.suggestion_decisions: list[SuggestionDecisionRow] = []
        """Per-setting recommendation provenance, from the RampFinished event."""
        self.consistency_warnings: list[str] = []
        """Recommendation self-consistency warnings, from the RampFinished event."""
        self.report_path: str | None = None
        self.finished: bool = False

    def apply(self, event: BenchmarkProgressEvent) -> None:
        """Fold one progress event into the accumulated state."""
        if isinstance(event, RampStarting):
            self.run_id = event.run_id
            self.process_mode = event.process_mode
            self.startup_phase = event.phase
        elif isinstance(event, RampStarted):
            self.run_id = event.run_id
            self.num_levels = event.num_levels
            self.tiers = list(event.tiers)
            self.process_mode = event.process_mode
            self.gpu_name = event.gpu_name
        elif isinstance(event, RampPlanned):
            self.plan_rows = list(event.rows)
        elif isinstance(event, LevelStarted):
            self.startup_phase = ""  # a level is running now; the pre-level phase no longer applies
            level = self._level(event.level_id)
            level.description = event.description
            level.stage = event.stage
            level.tier = event.tier
            level.axis = event.axis
            level.jobs_expected = event.jobs_expected
            self.current_level_id = event.level_id
        elif isinstance(event, LevelProgress):
            level = self._level(event.level_id)
            level.jobs_completed = event.jobs_completed
            level.jobs_faulted = event.jobs_faulted
            level.jobs_expected = event.jobs_expected if event.jobs_expected is not None else level.jobs_expected
            level.iterations_per_second = event.iterations_per_second
            level.vram_used_mb = event.vram_used_mb
            level.gpu_busy_percent = event.gpu_busy_percent
            level.elapsed_seconds = event.elapsed_seconds
            level.phase = event.phase
            level.num_process_recoveries = event.num_process_recoveries
        elif isinstance(event, LevelFinished):
            level = self._level(event.level_id)
            level.outcome = event.outcome
            level.reasons = list(event.reasons)
            level.advisories = list(event.advisories)
            if event.its_p50 is not None:
                level.iterations_per_second = event.its_p50
            if self.current_level_id == event.level_id:
                self.current_level_id = None
        elif isinstance(event, RampFinished):
            self.levels_passed = event.levels_passed
            self.levels_total = event.levels_total
            self.num_findings = event.num_findings
            self.suggested_bridge_data_yaml = event.suggested_bridge_data_yaml
            self.suggestion_decisions = list(event.suggestion_decisions)
            self.consistency_warnings = list(event.consistency_warnings)
            self.report_path = event.report_path
            self.finished = True

    def ordered_levels(self) -> list[LevelState]:
        """Return the levels in the order they first appeared."""
        return [self.levels[level_id] for level_id in self.level_order]

    def _level(self, level_id: str) -> LevelState:
        """Return the state for a level, creating and ordering it on first sight."""
        level = self.levels.get(level_id)
        if level is None:
            level = LevelState(level_id=level_id)
            self.levels[level_id] = level
            self.level_order.append(level_id)
        return level


class BenchmarkSupervisor:
    """Owns the benchmark subprocess and follows its progress file; records the result when it finishes.

    Drive it by calling :meth:`tick` on the TUI's timer: ``tick`` tails new progress events into
    :attr:`run_state` and, when the subprocess exits, finalizes the status and records the run in app state.
    All transport/IO errors are swallowed so the TUI never crashes with the benchmark.
    """

    def __init__(
        self, *, config_path: Path = DEFAULT_CONFIG_PATH, results_root: Path = Path("benchmark_results")
    ) -> None:
        """Initialize the supervisor (does not launch; call :meth:`start`)."""
        self._config_path = config_path
        self._results_root = results_root
        self._process: subprocess.Popen[bytes] | None = None
        self._tailer: ProgressTailer | None = None
        self._console_handle: object | None = None
        self._status = BenchmarkSupervisorStatus.IDLE
        self._out_dir: Path | None = None
        self.run_state = BenchmarkRunState()
        self.report: BenchmarkReport | None = None

    @property
    def status(self) -> BenchmarkSupervisorStatus:
        """The current benchmark lifecycle status."""
        return self._status

    @property
    def out_dir(self) -> Path | None:
        """The output directory of the current/last run, or None before the first launch."""
        return self._out_dir

    @property
    def is_active(self) -> bool:
        """Whether a benchmark is being prepared or is currently running.

        Includes the ``PREPARING`` window (the worker is being stopped to free the GPU, before the
        subprocess launches) so the app rejects a second run request during that blocking hand-off.
        """
        return self._status in (BenchmarkSupervisorStatus.PREPARING, BenchmarkSupervisorStatus.RUNNING)

    def mark_preparing(self) -> None:
        """Enter the pre-launch ``PREPARING`` state so the tab shows motion while the worker is stopped.

        The app stops the worker (a blocking call that can take up to ~100s) before launching the
        benchmark subprocess; without a visible state for that window the stop reads as a hang. Call this
        on the UI thread before kicking off the stop-then-launch flow.
        """
        self._status = BenchmarkSupervisorStatus.PREPARING
        self.run_state = BenchmarkRunState()
        self.run_state.startup_phase = "Stopping the worker to free the GPU for the benchmark…"

    def start(self, options: BenchmarkOptions) -> None:
        """Launch a benchmark ramp subprocess into a fresh timestamped output directory."""
        out_dir = self._results_root / time.strftime("%Y%m%d-%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self.run_state = BenchmarkRunState()
        self.run_state.startup_phase = "Launching benchmark subprocess…"
        self.report = None

        # Redirect the benchmark's console output to a file: it would otherwise corrupt the Textual terminal.
        self._console_handle = (out_dir / "console.log").open("w", encoding="utf-8")
        command = options.build_command(out_dir)
        self._process = subprocess.Popen(command, stdout=self._console_handle, stderr=subprocess.STDOUT)
        self._tailer = ProgressTailer(out_dir / PROGRESS_FILENAME)
        self._status = BenchmarkSupervisorStatus.RUNNING
        logger.info(f"Launched benchmark (pid={self._process.pid}, mode={options.process_mode}) into {out_dir}.")

    def tick(self) -> list[BenchmarkProgressEvent]:
        """Fold any new progress events into the run state and finalize when the subprocess exits."""
        if self._status is not BenchmarkSupervisorStatus.RUNNING or self._tailer is None or self._process is None:
            return []

        events = self._tailer.poll()
        for event in events:
            self.run_state.apply(event)

        exit_code = self._process.poll()
        if exit_code is None:
            return events

        trailing = self._tailer.poll()
        for event in trailing:
            self.run_state.apply(event)
        events.extend(trailing)
        self._finalize(exit_code)
        return events

    def cancel(self) -> None:
        """Terminate the benchmark subprocess tree (controller, level runner, and worker children).

        Killing only the controller process orphans the level runner and its GPU-resident worker
        children under spawn (no parent-child lifetime link on Windows), which is exactly why cancelled
        benchmarks were observed to keep running. ``kill_process_tree`` reaps the whole descendant tree.
        """
        process = self._process
        if process is not None and process.poll() is None:
            kill_process_tree(process.pid, grace_seconds=_CANCEL_GRACE_SECONDS)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(_CANCEL_GRACE_SECONDS)
        self._close_console()
        if self._status is BenchmarkSupervisorStatus.RUNNING:
            self._status = BenchmarkSupervisorStatus.CANCELLED

    def stop(self) -> None:
        """Alias for :meth:`cancel`, for symmetry with the worker supervisor's app-teardown call."""
        self.cancel()

    def _finalize(self, exit_code: int) -> None:
        """Settle the status from the exit code and record the run when it succeeded."""
        self._close_console()
        if exit_code == 0:
            self._status = BenchmarkSupervisorStatus.FINISHED
            self._load_and_record_report()
        else:
            self._status = BenchmarkSupervisorStatus.FAILED
            logger.warning(f"Benchmark subprocess exited with code {exit_code}.")

    def _load_and_record_report(self) -> None:
        """Load the finished report and record it as the canonical benchmark, best-effort."""
        if self._out_dir is None:
            return
        try:
            from horde_worker_regen.app_state import AppStateStore, build_benchmark_record
            from horde_worker_regen.benchmark.controller import load_existing_report

            report = load_existing_report(self._out_dir)
            if report is not None:
                self.report = report
                AppStateStore().record_benchmark(build_benchmark_record(report, results_dir=self._out_dir))
        except Exception as record_error:  # noqa: BLE001 - recording must not break the TUI
            logger.debug(f"Could not load/record benchmark report: {record_error}")

    def _close_console(self) -> None:
        """Close the redirected console-output file handle if it is open."""
        handle = self._console_handle
        if handle is not None:
            with contextlib.suppress(OSError):
                handle.close()  # type: ignore[attr-defined]
            self._console_handle = None


def suggested_config_overrides(suggested: SuggestedBridgeData) -> dict[str, object]:
    """Return the bridgeData key/value overrides a benchmark recommendation sets.

    ``max_batch`` is included (unlike the validation soak's ``to_bridge_overrides``) because it is a real
    worker config field the user runs with, not just a per-job payload value.
    """
    return {
        "max_threads": suggested.max_threads,
        "queue_size": suggested.queue_size,
        "max_batch": suggested.max_batch,
        "allow_lora": suggested.allow_lora,
        "allow_controlnet": suggested.allow_controlnet,
        "allow_post_processing": suggested.allow_post_processing,
        "models_to_load": list(suggested.models_to_load),
        "alchemist": suggested.alchemist,
        "alchemy_allow_concurrent": suggested.alchemy_allow_concurrent,
    }


def apply_suggested_to_config(suggested: SuggestedBridgeData, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Write the benchmark's recommended values into ``bridgeData.yaml``, preserving comments."""
    apply_known_good_to_config(suggested_config_overrides(suggested), config_path)


def apply_known_good_to_config(config_snapshot: Mapping[str, object], config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Write a configuration snapshot back into ``bridgeData.yaml``, preserving comments and other keys.

    Reuses the config editor's ruamel read/write so untouched keys and comments survive. Shared by applying a
    fresh benchmark recommendation and restoring a previously known-good configuration.
    """
    data = load_config(config_path)
    for key in _SUGGESTED_CONFIG_KEYS:
        if key in config_snapshot:
            data[key] = config_snapshot[key]
    save_config(data, config_path)


def record_suggested_as_known_good(suggested: SuggestedBridgeData, *, worker_version: str) -> None:
    """Record an applied benchmark recommendation as the last benchmark-validated known-good config.

    Best-effort: a failure only loses bookkeeping, so it is logged at debug. Uses a lazy app-state import to
    keep this module's import cost off the TUI's hot path.
    """
    try:
        from horde_worker_regen.app_state import (
            AppStateStore,
            KnownGoodSettings,
            KnownGoodSource,
            compute_config_digest,
        )

        snapshot = suggested_config_overrides(suggested)
        AppStateStore().record_known_good(
            KnownGoodSettings(
                config_digest=compute_config_digest(snapshot),
                config_snapshot=snapshot,
                validated_at=time.time(),
                worker_version=worker_version,
                source=KnownGoodSource.BENCHMARK,
            ),
        )
    except Exception as record_error:  # noqa: BLE001 - known-good bookkeeping must not break the TUI
        logger.debug(f"Could not record known-good settings: {record_error}")


__all__ = [
    "BenchmarkOptions",
    "BenchmarkRunState",
    "BenchmarkSupervisor",
    "BenchmarkSupervisorStatus",
    "LevelState",
    "apply_known_good_to_config",
    "apply_suggested_to_config",
    "record_suggested_as_known_good",
    "suggested_config_overrides",
]
