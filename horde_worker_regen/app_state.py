"""Durable, app-managed state that persists across worker runs.

This is the structured, on-disk counterpart to the in-memory
[`WorkerState`][horde_worker_regen.process_management.config.worker_state.WorkerState]: it records what the
application needs to remember *between* invocations: the last benchmark and where its results live,
the last worker run, the last-known-good settings, and which worker version last ran (so that a
version bump can mark a stale benchmark for re-running).

The store lives in a grouped directory in the working directory (``.horde_worker_regen/state.json``),
alongside ``bridgeData.yaml``, ``logs/`` and ``benchmark_results/``. Reads never raise: a missing or
unparseable file yields a fresh state, so a corrupt file can never block worker startup. Writes are
atomic (temp file in the same directory, then ``os.replace``).

The module is deliberately dependency-light; it does not import the benchmark or hordelib chains, so
it can be imported early in worker startup and by the TUI. The report-to-record conversion
([`build_capability_benchmark_record`][horde_worker_regen.app_state.build_capability_benchmark_record])
uses a local import.
"""

from __future__ import annotations

import contextlib
import enum
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from horde_worker_regen.benchmark.capabilities.result import CapabilityReport

APP_STATE_SCHEMA_VERSION = 2
"""The schema version stamped onto every state file this build writes.

Reads never reject an older file: every field carries a default, so an unknown or absent key degrades
to that default rather than discarding accumulated state. The number exists so a load can tell which
migrations a file predates, which is how the dashboard knows an installation existed before the
progressive experience levels and therefore needs to be told the default changed.
"""

APP_STATE_DIR_NAME = ".horde_worker_regen"
"""The grouped state directory, created in the working directory next to ``bridgeData.yaml``."""

APP_STATE_FILENAME = "state.json"

DEFAULT_THEME_NAME = "horde-dark"
"""The theme a state file falls back to when it names one this build cannot restore."""

KNOWN_THEME_NAMES = frozenset({DEFAULT_THEME_NAME, "horde-light", "horde-ansi"})
"""Theme names this build can restore, declared here rather than in ``tui.design``.

State is read by the worker as well as the dashboard, so this module must stay free of Textual, which
rules out importing the names from the module that builds the themes. Declaring them here and having
``tui.design`` assert its themes against this set keeps one source of truth without the import.

A name absent here (a theme from a newer build, or a hand-edited file) falls back to
:data:`DEFAULT_THEME_NAME` instead of failing the load, so an unknown theme can never block startup.
"""


class KnownGoodSource(enum.StrEnum):
    """How a known-good configuration came to be trusted."""

    BENCHMARK = "benchmark"
    """The configuration was synthesized and validated by a benchmark soak."""
    CLEAN_RUN = "clean_run"
    """The configuration ran a worker session for long enough without failing out."""


class OverviewViewMode(enum.StrEnum):
    """How densely the Overview tab renders; cycled by the F6 view-mode toggle."""

    NORMAL = "normal"
    """The lean redesign: enriched hero, health, trends, pipeline, and a lean processes table."""
    DETAILS = "details"
    """Normal plus the demoted panels (worker config, alchemy, queue, recent jobs) and extra columns."""
    THIN = "thin"
    """A single compact status bar only; the rest of the dashboard is hidden."""


class ExperienceLevel(enum.StrEnum):
    """How much worker implementation detail the dashboard exposes.

    The level changes the depth and vocabulary of what each destination renders. It never changes which
    destinations exist: every tab is reachable at every level, so raising the level reveals detail in
    place rather than unlocking somewhere new.
    """

    SIMPLE = "simple"
    """Plain-language status and guided contribution tasks."""
    ADVANCED = "advanced"
    """The complete everyday operator surface."""
    DEVELOPER = "developer"
    """Sensitive controls, uncommon tuning, and forensic detail."""


class DisplayDensity(enum.StrEnum):
    """Spacing density for the technical operator experiences."""

    COMFORTABLE = "comfortable"
    COMPACT = "compact"


class OverviewTrendWindow(enum.StrEnum):
    """The time span used by the Overview trend sparklines."""

    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    SIXTY_MINUTES = "60m"
    TWO_HOURS = "120m"
    ALL = "all"


class OnboardingChoice(enum.StrEnum):
    """The user's response to the first-run benchmark prompt."""

    ACCEPTED = "accepted"
    """The user chose to run the benchmark."""
    DECLINED = "declined"
    """The user chose not to be asked again (sticky)."""
    DEFERRED = "deferred"
    """The user skipped for now; the prompt may appear again later."""


class BenchmarkAvailability(enum.StrEnum):
    """Whether a usable benchmark exists for the running worker version."""

    NONE = "none"
    """No benchmark has ever been recorded."""
    STALE = "stale"
    """A benchmark exists but was produced by a different worker version."""
    CURRENT = "current"
    """A benchmark exists and matches the running worker version."""


class WorkerRunRecord(BaseModel):
    """Represents the outcome of one completed worker session."""

    started_at: float
    ended_at: float | None = None
    duration_seconds: float | None = None
    worker_version: str
    jobs_submitted: int = 0
    jobs_faulted: int = 0
    kudos_this_session: float = 0.0
    clean_exit: bool = False


class BenchmarkRecord(BaseModel):
    """Represents the most recent benchmark run and where its artifacts live."""

    run_id: str
    results_dir: str
    created_at: float
    worker_version: str
    levels_passed: int = 0
    levels_total: int = 0
    gpu_name: str | None = None
    had_findings: bool = False
    suggested_bridge_data: dict[str, object] = Field(default_factory=dict)
    """The synthesized recommendation (``SuggestedBridgeData.model_dump()``) as a plain dict, so this
    module stays free of the benchmark import chain."""


class KnownGoodSettings(BaseModel):
    """Represents a bridgeData configuration that ran cleanly or passed benchmark validation."""

    config_digest: str
    config_snapshot: dict[str, object] = Field(default_factory=dict)
    validated_at: float
    worker_version: str
    source: KnownGoodSource


class OnboardingState(BaseModel):
    """Represents whether the user has been prompted to benchmark, and their choice."""

    benchmark_prompt_choice: OnboardingChoice | None = None
    prompt_last_shown_at: float | None = None


class WorkerAppState(BaseModel):
    """Represents the full durable application state persisted between worker runs."""

    schema_version: int = APP_STATE_SCHEMA_VERSION
    auto_start_worker: bool = False
    """Whether the TUI starts the worker automatically on launch. Off by default so the dashboard
    never begins real GPU work unprompted; the user opts in via the first-run prompt (or F4)."""
    setup_complete: bool = False
    """Whether the guided first-run wizard has been satisfied at least once. Defaults False so a brand-new
    install runs the wizard; an existing, already-configured install is marked complete on first launch
    without ever showing it (see the wizard's incomplete-setup detection)."""
    detailed_info: bool = False
    """Deprecated: superseded by ``overview_view_mode``. Retained only so an older persisted state can
    be migrated (a stored ``detailed_info: true`` maps to the ``details`` view mode on load)."""
    overview_view_mode: OverviewViewMode = OverviewViewMode.NORMAL
    """How densely the Overview tab renders. The F6 toggle cycles normal -> details -> thin; the
    redesigned lean overview is the default, with the older verbose dashboard behind ``details``."""
    overview_trend_window: OverviewTrendWindow = OverviewTrendWindow.FIFTEEN_MINUTES
    """Time span used by the Overview trend sparklines; samples remain session-local."""
    overview_hidden_elements: list[str] = Field(default_factory=list)
    """Stable keys of Overview elements the operator has hidden (see the overview layout registry).
    Persisted so a curated dashboard survives restarts; an unknown key is ignored rather than rejected,
    so the list stays forward-compatible when elements are renamed or removed."""
    worker_version_last_ran: str | None = None
    experience_level: ExperienceLevel = ExperienceLevel.SIMPLE
    """How much detail the dashboard exposes. Simple is the default so an unattended first launch is
    approachable; an installation that predates this field is told the default changed rather than being
    demoted silently (see ``needs_experience_introduction``)."""
    needs_experience_introduction: bool = False
    """Whether the operator is owed the one-time notice that the dashboard now starts in Simple.

    False on a fresh state because a new contributor has no prior experience to be surprised by. Set true
    only when loading a state file stamped before the experience levels existed, and cleared once the
    operator has answered the notice."""
    display_density: DisplayDensity = DisplayDensity.COMFORTABLE
    """Spacing for the Advanced and Developer surfaces; Simple stays responsive on its own."""
    theme_name: str = DEFAULT_THEME_NAME
    """Registered Textual theme name. An unrecognised value falls back to the default on load."""
    developer_warning_acknowledged: bool = False
    """Whether the one-time Developer-level warning has been accepted."""
    onboarding: OnboardingState = Field(default_factory=OnboardingState)
    last_worker_run: WorkerRunRecord | None = None
    last_benchmark: BenchmarkRecord | None = None
    last_known_good_settings: KnownGoodSettings | None = None


def default_app_state_dir() -> Path:
    """Return the grouped state directory in the current working directory."""
    return Path.cwd() / APP_STATE_DIR_NAME


def default_app_state_path() -> Path:
    """Return the default state file path (``.horde_worker_regen/state.json`` in the working dir)."""
    return default_app_state_dir() / APP_STATE_FILENAME


def compute_config_digest(config_snapshot: Mapping[str, object]) -> str:
    """Return a stable, order-independent sha256 digest of a configuration snapshot."""
    serialized = json.dumps(dict(config_snapshot), sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def is_benchmark_stale(state: WorkerAppState, *, current_version: str) -> bool:
    """Return whether the recorded benchmark is missing or was produced by a different version.

    This is the mechanism by which a version bump invalidates prior benchmark assertions: the
    recommendation and pass/fail gates were measured against a worker that may no longer behave the
    same, so they should be re-established.
    """
    benchmark = state.last_benchmark
    if benchmark is None:
        return True
    return benchmark.worker_version != current_version


def benchmark_status_summary(state: WorkerAppState, *, current_version: str) -> BenchmarkAvailability:
    """Return whether a current, stale, or no benchmark exists for the running version."""
    if state.last_benchmark is None:
        return BenchmarkAvailability.NONE
    if state.last_benchmark.worker_version != current_version:
        return BenchmarkAvailability.STALE
    return BenchmarkAvailability.CURRENT


def should_prompt_onboarding(state: WorkerAppState, *, current_version: str) -> bool:
    """Return whether to prompt the user to benchmark on first run.

    True when no current benchmark exists *and* the user has not stickily declined. A ``DEFERRED`` skip
    does not suppress a later prompt; only ``DECLINED`` does.
    """
    if state.onboarding.benchmark_prompt_choice is OnboardingChoice.DECLINED:
        return False
    return benchmark_status_summary(state, current_version=current_version) is not BenchmarkAvailability.CURRENT


def build_capability_benchmark_record(
    report: CapabilityReport,
    *,
    results_dir: str | os.PathLike[str],
) -> BenchmarkRecord:
    """Convert a finished capability report into the durable record stored in app state.

    Counts proven probes as passed and produces the lean, durable record stored in app state. Uses the
    report's own version/run stamps so the CLI and TUI entry points record identical metadata; the
    suggested bridge data is flattened to a plain dict to keep the record free of the benchmark chain.
    """
    probes_proven = sum(1 for probe in report.probes if probe.verdict == "proven")
    return BenchmarkRecord(
        run_id=report.run_id,
        results_dir=str(results_dir),
        created_at=report.created_at,
        worker_version=report.worker_version,
        levels_passed=probes_proven,
        levels_total=len(report.probes),
        gpu_name=report.machine.gpu_name,
        had_findings=bool(report.findings),
        suggested_bridge_data=report.suggested_bridge_data.model_dump(),
    )


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text to ``path`` atomically: a temp file in the same directory, then ``os.replace``."""
    handle, temp_path_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    temp_path = Path(temp_path_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as temp_file:
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            temp_path.unlink()
        raise


class AppStateStore:
    """Loads and atomically persists [`WorkerAppState`][horde_worker_regen.app_state.WorkerAppState].

    Thread Safety:
        Not internally synchronized. In practice the worker writes only from its single shutdown
        path and the TUI from its single UI thread, so there is no concurrent writer.

    Subclass Integration:
        Reads are deliberately tolerant: a missing or unparseable file yields a fresh state rather
        than raising, so a corrupt state file can never block worker startup. Only the targeted
        mutators write, and they always load-modify-save so they never clobber unrelated fields.
    """

    def __init__(self, path: Path | None = None) -> None:
        """Initialize the store, defaulting to ``.horde_worker_regen/state.json`` in the working dir."""
        self._path = path if path is not None else default_app_state_path()

    @property
    def path(self) -> Path:
        """The JSON file this store reads from and writes to."""
        return self._path

    def load(self) -> WorkerAppState:
        """Return the persisted state, or a fresh state when the file is missing or unparseable."""
        if not self._path.exists():
            return WorkerAppState()
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            state = WorkerAppState.model_validate(raw)
        except (OSError, ValueError) as read_error:
            logger.debug(f"Could not read app state at {self._path} ({read_error}); using a fresh state.")
            return WorkerAppState()
        # Migrate a pre-view-mode state: a stored detailed_info flag maps onto the new view mode.
        if state.overview_view_mode is OverviewViewMode.NORMAL and state.detailed_info:
            state.overview_view_mode = OverviewViewMode.DETAILS
        # A file stamped before the experience levels belongs to an installation that has been using the
        # full dashboard all along. Reading it as Simple is correct (that is the new default) but doing so
        # without saying anything would look like the dashboard lost its detail, so the notice is owed.
        persisted_version = raw.get("schema_version", 1) if isinstance(raw, dict) else 1
        if not isinstance(persisted_version, int) or persisted_version < 2:
            state.needs_experience_introduction = True
        if state.theme_name not in KNOWN_THEME_NAMES:
            state.theme_name = DEFAULT_THEME_NAME
        state.schema_version = APP_STATE_SCHEMA_VERSION
        return state

    def save(self, state: WorkerAppState) -> None:
        """Persist ``state`` atomically, creating the state directory on first write."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(self._path, state.model_dump_json(indent=2))

    # region mutators

    def record_worker_started(self, *, worker_version: str) -> WorkerAppState:
        """Record that a worker session started on ``worker_version`` and return the updated state."""
        state = self.load()
        state.worker_version_last_ran = worker_version
        self.save(state)
        return state

    def record_worker_finished(self, record: WorkerRunRecord) -> None:
        """Record the outcome of the just-finished worker session."""
        state = self.load()
        state.last_worker_run = record
        self.save(state)

    def record_benchmark(self, record: BenchmarkRecord) -> None:
        """Record the most recent benchmark as the canonical run-to-run pointer."""
        state = self.load()
        state.last_benchmark = record
        self.save(state)

    def record_known_good(self, settings: KnownGoodSettings) -> None:
        """Record a configuration as the last one known to run cleanly or pass validation."""
        state = self.load()
        state.last_known_good_settings = settings
        self.save(state)

    def record_onboarding_choice(self, choice: OnboardingChoice) -> None:
        """Record the user's response to the first-run benchmark prompt (timestamped)."""
        state = self.load()
        state.onboarding.benchmark_prompt_choice = choice
        state.onboarding.prompt_last_shown_at = time.time()
        self.save(state)

    def set_auto_start_worker(self, enabled: bool) -> None:
        """Persist whether the TUI should start the worker automatically on launch."""
        state = self.load()
        state.auto_start_worker = enabled
        self.save(state)

    def set_setup_complete(self, complete: bool) -> None:
        """Persist whether the guided first-run wizard has been satisfied."""
        state = self.load()
        state.setup_complete = complete
        self.save(state)

    def set_view_mode(self, mode: OverviewViewMode) -> None:
        """Persist the Overview tab's density (the F6 view-mode toggle)."""
        state = self.load()
        state.overview_view_mode = mode
        self.save(state)

    def set_trend_window(self, window: OverviewTrendWindow) -> None:
        """Persist the Overview tab's trend window."""
        state = self.load()
        state.overview_trend_window = window
        self.save(state)

    def set_overview_hidden_elements(self, keys: Iterable[str]) -> None:
        """Persist the set of Overview element keys the operator has chosen to hide.

        Stored sorted and de-duplicated so the on-disk file is stable regardless of the modal's
        iteration order, which keeps diffs and round-trip tests deterministic.
        """
        state = self.load()
        state.overview_hidden_elements = sorted(set(keys))
        self.save(state)

    def set_experience_level(self, level: ExperienceLevel) -> None:
        """Persist the selected experience level."""
        state = self.load()
        state.experience_level = level
        self.save(state)

    def resolve_experience_introduction(self, level: ExperienceLevel) -> None:
        """Persist the operator's answer to the one-time Simple-default notice.

        Clears the debt in the same write that stores the chosen level, so an interrupted launch either
        shows the notice again or records an answer, never both.
        """
        state = self.load()
        state.experience_level = level
        state.needs_experience_introduction = False
        self.save(state)

    def set_display_density(self, density: DisplayDensity) -> None:
        """Persist the selected Advanced/Developer display density."""
        state = self.load()
        state.display_density = density
        self.save(state)

    def set_theme_name(self, theme_name: str) -> None:
        """Persist a theme name, ignoring one this build cannot restore."""
        if theme_name not in KNOWN_THEME_NAMES:
            return
        state = self.load()
        state.theme_name = theme_name
        self.save(state)

    def acknowledge_developer_warning(self) -> None:
        """Remember that the operator accepted the one-time Developer warning."""
        state = self.load()
        state.developer_warning_acknowledged = True
        self.save(state)

    # endregion


__all__ = [
    "APP_STATE_SCHEMA_VERSION",
    "AppStateStore",
    "BenchmarkAvailability",
    "DEFAULT_THEME_NAME",
    "KNOWN_THEME_NAMES",
    "BenchmarkRecord",
    "DisplayDensity",
    "ExperienceLevel",
    "KnownGoodSettings",
    "KnownGoodSource",
    "OnboardingChoice",
    "OnboardingState",
    "OverviewTrendWindow",
    "OverviewViewMode",
    "WorkerAppState",
    "WorkerRunRecord",
    "benchmark_status_summary",
    "build_capability_benchmark_record",
    "compute_config_digest",
    "default_app_state_dir",
    "default_app_state_path",
    "is_benchmark_stale",
    "should_prompt_onboarding",
]
