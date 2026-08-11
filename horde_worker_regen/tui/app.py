"""The worker TUI application: launches/supervises the worker and renders its live state.

Entry point ``horde_worker_regen.tui.app:main`` (console script ``horde-worker``). Runs in a terminal
or, via ``textual serve "horde-worker"``, in a browser. The headless ``run_worker`` path is unchanged;
this is an optional supervising frontend.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import multiprocessing
import os
import sys
import time
from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

from loguru import logger
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static, TabbedContent, TabPane

from horde_worker_regen import __version__
from horde_worker_regen.app_state import (
    DEFAULT_THEME_NAME,
    KNOWN_THEME_NAMES,
    AppStateStore,
    DisplayDensity,
    ExperienceLevel,
    OnboardingChoice,
    OverviewTrendWindow,
    OverviewViewMode,
    benchmark_status_summary,
    should_prompt_onboarding,
)
from horde_worker_regen.process_management.ipc.supervisor_channel import DownloadPhase, WorkerStateSnapshot
from horde_worker_regen.run_worker import WorkerLaunchOptions
from horde_worker_regen.runtime_version import runtime_version
from horde_worker_regen.tui import socket_protocol as sp
from horde_worker_regen.tui.attach import AttachedWorkerSupervisor, SupervisorLike
from horde_worker_regen.tui.benchmark_launcher import (
    BenchmarkOptions,
    BenchmarkSupervisor,
    BenchmarkSupervisorStatus,
    apply_known_good_to_config,
    apply_suggested_to_config,
    record_suggested_as_known_good,
)
from horde_worker_regen.tui.beta_models import apply_beta_model_env
from horde_worker_regen.tui.cache_home import apply_cache_home_env
from horde_worker_regen.tui.config_form import DEFAULT_CONFIG_PATH
from horde_worker_regen.tui.design import register_horde_themes
from horde_worker_regen.tui.formatters import configure_fidelity, format_percent
from horde_worker_regen.tui.health import HealthReport, HealthStatus, WorkerPhase, build_offline_checks, derive
from horde_worker_regen.tui.logging_setup import setup_supervisor_file_logging
from horde_worker_regen.tui.responsive import PHONE_BAND_MAX_WIDTH, ResponsiveModalScreen
from horde_worker_regen.tui.update_check import UpdateInfo, check_for_update
from horde_worker_regen.tui.widgets.benchmark import BenchmarkView, BenchmarkWaitingState
from horde_worker_regen.tui.widgets.config_editor import (
    MODELS_SUBTAB_ID,
    ConfigEditorView,
    ConfigLeaveChoice,
    ConfigLeaveModal,
)
from horde_worker_regen.tui.widgets.control import ControlView
from horde_worker_regen.tui.widgets.diagnostics import DiagnosticsView
from horde_worker_regen.tui.widgets.download_picker import (
    DownloadPickerModal,
    DownloadPickerRow,
    DownloadSelection,
)
from horde_worker_regen.tui.widgets.downloads import DownloadsView
from horde_worker_regen.tui.widgets.experience import (
    DashboardPreferencesView,
    DeveloperWarningModal,
    ExperienceIntroductionModal,
    HelpModal,
)
from horde_worker_regen.tui.widgets.gpus import GpusView
from horde_worker_regen.tui.widgets.insights import InsightsView
from horde_worker_regen.tui.widgets.live_view import LiveView
from horde_worker_regen.tui.widgets.logs import LogsView
from horde_worker_regen.tui.widgets.model_manager import ModelManagerView
from horde_worker_regen.tui.widgets.onboarding import (
    BenchmarkOnboardingModal,
    WorkerStartChoice,
    WorkerStartModal,
)
from horde_worker_regen.tui.widgets.overview import OverviewView
from horde_worker_regen.tui.widgets.overview_layout import OverviewLayoutModal, valid_hidden_keys
from horde_worker_regen.tui.widgets.simple import (
    PRIMERS,
    SimpleActivityView,
    SimpleDestination,
    SimpleHomeView,
    SimpleModelStatusView,
    TabPrimer,
)
from horde_worker_regen.tui.widgets.stats import StatsView
from horde_worker_regen.tui.wizard import GettingStartedScreen, is_setup_incomplete
from horde_worker_regen.tui.worker_launcher import (
    SupervisorStallStats,
    SupervisorStatus,
    WorkerProcessMode,
    WorkerSupervisor,
)
from horde_worker_regen.utils import get_system_appropriate_updater

if TYPE_CHECKING:
    # Imported for annotations only; the modal module is imported lazily at use (its subprocess plumbing
    # stays off the TUI hot path), so the live-state type must not pull it in at module load.
    from horde_worker_regen.tui.widgets.benchmark_download import DownloadLiveState


_BENCHMARK_DRAIN_TIMEOUT_SECONDS = 150.0
"""How long to let a live worker drain its in-flight jobs before falling back to a hard stop. Sized above the
worker's own drain backstop (a job plus its grace) so a normally-finishing job is never cut short."""
_BENCHMARK_SCALE_TIMEOUT_SECONDS = 45.0
"""How long to wait for the scaled-down inference processes (and their GPU contexts) to actually exit."""
_BENCHMARK_DRAIN_POLL_SECONDS = 0.5
"""How often the drain wait re-checks the worker's latest snapshot."""

_DESTINATION_ID_PREFIX = "tab-"
"""Identifier prefix marking a top-level destination, as opposed to a nested sub-tab."""

REMOTE_EXPOSED_CLASS = "-remote-exposed"
"""Screen class marking a session anyone on the network can reach, which withholds the credentials."""

PHONE_CLASS = "-phone"
"""Screen class for the width band below the terminal floor, which is what a phone browser gets."""


_COMPACT_TAB_LABELS: dict[str, str] = {
    "tab-overview": "Ovr",
    "tab-stats": "St",
    "tab-control": "Ctl",
    "tab-gpus": "GPU",
    "tab-live": "Live",
    "tab-downloads": "DL",
    "tab-logs": "Log",
    "tab-config": "Cfg",
    "tab-insights": "In",
    "tab-diagnostics": "Dx",
    "tab-benchmark": "Bmk",
}
"""Tab labels used in the phone width band, where the full ones do not fit.

The strip has no shedding logic of its own: eleven full labels need around 97 columns, so on a phone the
tabs past the first few scroll out of reach, and reaching them by tap is the only navigation a phone
reliably has. These are truncations of the real names rather than different words, so the tab a reader
learned from the documentation is still the tab they tap. Their total (including the two cells of
padding Textual gives each tab) must stay within the roughly 53 columns a 320px viewport gets at the
served page's 10px readability floor.
"""

_CONFIG_TAB_ID = "tab-config"
_DOWNLOADS_TAB_ID = "tab-downloads"
_LIVE_TAB_ID = "tab-live"


def _no_inference_contexts(snapshot: WorkerStateSnapshot) -> bool:
    """Whether no inference process is alive (so its GPU VRAM is released and the benchmark can take the card)."""
    return not any(process.process_type == "INFERENCE" and process.is_alive for process in snapshot.processes)


class WebQuitWarningModal(ResponsiveModalScreen[bool]):
    """Warn the user that closing this browser tab leaves the worker running in the background."""

    DEFAULT_CSS = """
    WebQuitWarningModal {
        align: center middle;
    }
    WebQuitWarningModal #web-quit-dialog {
        width: 72;
        max-width: 95%;
        height: auto;
        padding: 1 2;
        border: thick $warning;
        background: $surface;
    }
    WebQuitWarningModal #web-quit-dialog Button {
        width: 100%;
        margin-top: 1;
    }
    """

    BINDINGS = [("escape", "stay", "Stay")]

    def compose(self) -> ComposeResult:
        """Lay out the warning text and the two choices."""
        with Vertical(id="web-quit-dialog"):
            yield Static(self._message(), id="web-quit-message")
            yield Button("Close this dashboard (worker keeps running)", id="web-quit-close", variant="warning")
            yield Button("Stay", id="web-quit-stay", variant="primary")

    @staticmethod
    def _message() -> Text:
        return Text.assemble(
            ("Worker stays running after you close this tab\n\n", "bold"),
            (
                "This dashboard is a browser view of a worker process running on your computer. "
                "Closing this tab or pressing Ctrl+Q only closes the view - the worker keeps "
                "contributing to the horde.\n\n"
                "To stop the worker completely, right-click the AI Horde icon in the taskbar "
                "notification area (the small icons in the bottom-right corner of your screen) "
                "and choose 'Stop worker'.",
                "grey70",
            ),
        )

    def action_stay(self) -> None:
        """Dismiss as False (do not quit) when Escape is pressed."""
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Close the dashboard on confirm; stay on cancel."""
        if event.button.id == "web-quit-close":
            self.dismiss(True)
        elif event.button.id == "web-quit-stay":
            self.dismiss(False)


class BenchmarkOverWorkerModal(ResponsiveModalScreen[bool]):
    """Confirm handing the GPU to a benchmark while the worker is serving jobs.

    The benchmark needs the GPU to itself. Rather than tear the worker down, the app drains its queue (letting
    in-flight jobs finish) and frees the GPU while keeping the worker alive and ready to resume with Go live.
    That still interrupts serving, so it must never happen on a single click without the operator agreeing.
    """

    DEFAULT_CSS = """
    BenchmarkOverWorkerModal {
        align: center middle;
    }
    BenchmarkOverWorkerModal #bench-over-worker-dialog {
        width: 72;
        max-width: 95%;
        height: auto;
        padding: 1 2;
        border: thick $warning;
        background: $surface;
    }
    BenchmarkOverWorkerModal #bench-over-worker-dialog Button {
        width: 100%;
        margin-top: 1;
    }
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, *, serving: bool) -> None:
        """Store whether the worker is currently serving jobs, so the warning describes the real disruption.

        Args:
            serving: True when the worker has live inference (jobs would be interrupted); False when it is
                alive but not serving (e.g. held while downloading), where the benchmark only takes the idle GPU.
        """
        super().__init__()
        self._serving = serving

    def compose(self) -> ComposeResult:
        """Lay out the warning text and the run / cancel choices."""
        with Vertical(id="bench-over-worker-dialog"):
            yield Static(self._message(), id="bench-over-worker-message")
            confirm_label = "Drain worker & run benchmark" if self._serving else "Use the GPU & run benchmark"
            yield Button(confirm_label, id="bench-over-worker-confirm", variant="warning")
            yield Button("Cancel (keep worker as-is)", id="bench-over-worker-cancel", variant="primary")

    def _message(self) -> Text:
        if self._serving:
            body = (
                "The worker is running and serving jobs. Starting the benchmark drains its queue, lets "
                "in-flight jobs finish, then frees the GPU for the benchmark while keeping the worker alive "
                "(its downloads keep running). It will not serve jobs again until you press Go live. "
                "Cancel to keep serving."
            )
        else:
            body = (
                "The worker is running but not serving jobs (it is held, e.g. while downloading). Starting the "
                "benchmark takes the idle GPU while the worker stays alive and keeps downloading; it resumes "
                "serving when you press Go live. Cancel to leave the worker as it is."
            )
        return Text.assemble(("Start benchmark?\n\n", "bold"), (body, "grey70"))

    def action_cancel(self) -> None:
        """Dismiss as False (leave the worker as it is) when Escape is pressed."""
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Confirm frees the GPU for the benchmark; cancel leaves the worker as it is."""
        if event.button.id == "bench-over-worker-confirm":
            self.dismiss(True)
        elif event.button.id == "bench-over-worker-cancel":
            self.dismiss(False)


class BenchmarkActionConfirmModal(ResponsiveModalScreen[bool]):
    """Confirm an action that would interfere with an in-progress benchmark download, with a plain explanation.

    A reusable yes/no for the benchmark↔download coordination guards (running before the models finish, or
    going live while benchmark-only downloads are still in flight): the body spells out the consequence so the
    operator chooses with the trade-off in front of them, rather than a contradictory action happening silently.
    """

    DEFAULT_CSS = """
    BenchmarkActionConfirmModal {
        align: center middle;
    }
    BenchmarkActionConfirmModal #bench-confirm-dialog {
        width: 72;
        max-width: 95%;
        height: auto;
        padding: 1 2;
        border: thick $warning;
        background: $surface;
    }
    BenchmarkActionConfirmModal #bench-confirm-dialog Button {
        width: 100%;
        margin-top: 1;
    }
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, *, title: str, body: str, confirm_label: str) -> None:
        """Store the dialog's title, explanatory body, and the label for its affirmative button."""
        super().__init__()
        self._title = title
        self._body = body
        self._confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        """Lay out the explanation and the confirm / cancel choices."""
        with Vertical(id="bench-confirm-dialog"):
            yield Static(
                Text.assemble((f"{self._title}\n\n", "bold"), (self._body, "grey70")),
                id="bench-confirm-message",
            )
            yield Button(self._confirm_label, id="bench-confirm-confirm", variant="warning")
            yield Button("Cancel", id="bench-confirm-cancel", variant="primary")

    def action_cancel(self) -> None:
        """Dismiss as False (do not proceed) when Escape is pressed."""
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Proceed on confirm; cancel leaves things unchanged."""
        if event.button.id == "bench-confirm-confirm":
            self.dismiss(True)
        elif event.button.id == "bench-confirm-cancel":
            self.dismiss(False)


class HordeWorkerTUI(App[None]):
    """A Textual dashboard that owns and visualises the reGen worker."""

    TITLE = f"AI Horde Worker - v{__version__}"
    CSS_PATH = "horde.tcss"
    """The Horde design system projection. Layout rules that depend on runtime intent stay in ``CSS``
    below; this file carries the shared surfaces (hero, card, muted) and the level/density policy."""

    HORIZONTAL_BREAKPOINTS = [
        (0, PHONE_CLASS),
        (PHONE_BAND_MAX_WIDTH, "-narrow"),
        (100, "-normal"),
        (150, "-wide"),
    ]
    """Width bands Textual stamps onto the Screen as classes, mirroring the table column tiers.

    These drive only *layout* rules in the CSS below (e.g. reclaiming side padding on a cramped terminal).
    Panel show/hide stays in Python because it depends on the F6 view intent, which CSS cannot see; and an
    inline ``display`` set per tick from Python would in any case win over a CSS ``display`` rule. The
    within-table column shedding that actually fixes the wide tables is done in ``responsive.py``.

    ``-narrow`` starts at the 80-column floor, so a terminal at that floor keeps exactly the rules it had
    before the phone band existed. ``-phone`` covers what a phone browser gets (roughly 40 to 70 columns),
    which is below anything a terminal is expected to run at.

    Textual applies at most **one** class per dimension, so ``-phone`` replaces ``-narrow`` rather than
    adding to it: every rule that should hold for both has to name both.

    There is deliberately no vertical band. Height is not what a phone is short of: served at the font
    size a narrow viewport picks, a phone gives dozens of rows in either orientation, and a band low
    enough to catch one would also catch the 80x24 terminal floor and change a layout that is fine.
    """

    CSS = """
    #status-bar {
        height: 1;
        padding: 0 1;
        background: $panel;
    }
    TabbedContent {
        height: 1fr;
    }
    OverviewView, StatsView, GpusView, LiveView, InsightsView, ConfigEditorView, LogsView, BenchmarkView,
    DownloadsView, ControlView, SimpleHomeView, SimpleActivityView, SimpleModelStatusView {
        height: 1fr;
        padding: 1 1;
    }
    /* On a cramped terminal, drop the horizontal padding so the tables get those columns back. */
    Screen.-narrow OverviewView, Screen.-phone OverviewView,
    Screen.-narrow StatsView, Screen.-phone StatsView,
    Screen.-narrow GpusView, Screen.-phone GpusView,
    Screen.-narrow LiveView, Screen.-phone LiveView,
    Screen.-narrow InsightsView, Screen.-phone InsightsView,
    Screen.-narrow ConfigEditorView, Screen.-phone ConfigEditorView,
    Screen.-narrow LogsView, Screen.-phone LogsView,
    Screen.-narrow BenchmarkView, Screen.-phone BenchmarkView,
    Screen.-narrow DownloadsView, Screen.-phone DownloadsView,
    Screen.-narrow ControlView, Screen.-phone ControlView,
    Screen.-narrow SimpleHomeView, Screen.-phone SimpleHomeView,
    Screen.-narrow SimpleActivityView, Screen.-phone SimpleActivityView,
    Screen.-narrow SimpleModelStatusView, Screen.-phone SimpleModelStatusView {
        padding: 1 0;
    }
    #overview-worker, #overview-processes {
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("f3", "start_stop_worker", "Start/Stop"),
        ("?", "show_help", "Help"),
        # No ctrl+p entry: Textual's Footer already pins its own command-palette key, so declaring one
        # here renders it twice and spends a slot in a list that is already truncated on real terminals.
        ("f6", "cycle_view_mode", "View mode"),
        ("c", "customize_overview", "Customize"),
        ("h", "toggle_hidden_reveal", "Reveal hidden"),
        ("f7", "toggle_download_pause", "Pause downloads"),
        ("t", "cycle_trend_window", "Trend window"),
        ("r", "reset_trends", "Reset trends"),
        ("j", "toggle_work_ledger_recent_jobs", "Ledger recent"),
        ("f11", "restart_worker", "Restart worker"),
        ("m", "toggle_server_maintenance", "Maintenance (horde)"),
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(
        self,
        supervisor: SupervisorLike,
        *,
        config_path: Path = DEFAULT_CONFIG_PATH,
        app_state_store: AppStateStore | None = None,
        load_config_from_env_vars: bool = False,
        remote_exposed: bool = False,
    ) -> None:
        """Store the (unstarted) supervisor, config path, and durable state store."""
        super().__init__()
        self._supervisor = supervisor
        self._benchmark_supervisor = BenchmarkSupervisor(config_path=config_path)
        self._config_path = config_path
        self._app_state_store = app_state_store if app_state_store is not None else AppStateStore()
        self._load_config_from_env_vars = load_config_from_env_vars
        # Set when this session is served on an address other than loopback, so anyone on the network can
        # reach it. Stamped onto the Screen so the stylesheet can withhold the credential fields.
        self._remote_exposed = remote_exposed
        # The tab strip's full labels, captured before the first swap to the compact ones so the swap can
        # be undone when the window widens again.
        self._full_tab_labels: dict[str, str] = {}
        self._tab_labels_are_compact = False
        self._frame = 0
        self._last_benchmark_status = BenchmarkSupervisorStatus.IDLE
        self._pending_benchmark_options = BenchmarkOptions()
        # True when the benchmark freed the GPU by gracefully draining a live worker (which stays alive, held)
        # rather than hard-stopping it. Drives the post-run guidance: "Go live" to resume vs "restart".
        self._benchmark_drained_worker = False
        # True while a benchmark download is in progress: the "waiting for benchmark models" mode that shows
        # the banner, gates Run, and warns before actions that would interrupt the fetch. Tracked as a flag
        # (not just a non-empty set) so a features-only request, whose image-model set is empty because the
        # controlnet/post-proc checkpoints fetch via the aux pass, still engages the mode.
        self._benchmark_waiting_active = False
        # The image model names a benchmark download requested, for the banner's N/M progress and the
        # config-subset check; feature models are not named here (the aux pass fetches them).
        self._benchmark_waiting_models: set[str] = set()
        # Set once the download subsystem is observed busy after a benchmark request, so the wait does not
        # complete on an initial idle snapshot taken before the fetch has actually started.
        self._benchmark_download_seen_active = False
        # Set when "Download only" starts a stopped worker: the hold command is sent once its pipe is up
        # (see _tick), since send_command would otherwise race the child's connection.
        self._pending_downloads_only_hold = False
        # A picker selection chosen before a freshly-started worker's pipe is up; sent once it is (see
        # _tick), after the hold command, so the models are fetched without the GPU committing.
        self._pending_download_models: DownloadSelection | None = None
        persisted_state = self._app_state_store.load()
        self._view_mode = persisted_state.overview_view_mode
        self._trend_window = persisted_state.overview_trend_window
        self._experience_level = persisted_state.experience_level
        self._display_density = persisted_state.display_density
        self._developer_warning_acknowledged = persisted_state.developer_warning_acknowledged
        # Owed to an installation that predates the levels; consumed once on mount so the Simple default
        # is announced rather than looking like the dashboard lost its detail.
        self._needs_experience_introduction = persisted_state.needs_experience_introduction
        # Cached because the render loop asks every frame and the answer costs a YAML parse; invalidated
        # by the config file's own change stamp rather than by a save hook, so an external edit counts.
        self._setup_required = False
        self._setup_config_stamp: tuple[int, int] | None = None
        register_horde_themes(self)
        self._theme_name = (
            persisted_state.theme_name if persisted_state.theme_name in KNOWN_THEME_NAMES else DEFAULT_THEME_NAME
        )
        self.theme = self._theme_name
        # Operator-hidden Overview elements (registry keys), restored from durable state. Unknown/renamed
        # keys are dropped on load so a stale preference never blocks rendering.
        self._overview_hidden: set[str] = valid_hidden_keys(persisted_state.overview_hidden_elements)
        # Session-only: the 'h' quick-reveal toggle temporarily un-suppresses hidden elements. Not persisted,
        # so a restart returns to the operator's curated layout.
        self._reveal_hidden_elements = False
        self._show_recent_work_ledger_jobs = True
        self._last_trend_config_fingerprint: tuple[object, ...] | None = None
        self._last_main_tab = "tab-overview"
        self._allow_tab_switch_to: str | None = None
        self._config_leave_warning_suppressed = False
        # Optimistic intent for the "m" server-maintenance toggle: set to the desired state immediately
        # after a command is sent, so a rapid second press toggles correctly before the 15 s poll catches up.
        # Cleared once a snapshot confirms the advisory poll has reflected the new state.
        self._intended_server_maintenance: bool | None = None
        self._server_maintenance_intent_pop_count: int | None = None
        # Tracks the previous-tick value of last_pop_maintenance_mode to detect False → True transitions
        # and fire a toast exactly once when the horde forces maintenance via the pop response.
        self._prev_pop_maintenance_mode: bool = False
        self._prev_maintenance_mode: bool = False
        self._maintenance_started_at: float | None = None
        self._update_info: UpdateInfo | None = None
        self._start_time: float = 0.0
        # True once a graceful quit (Ctrl+Q/Ctrl+C) has been requested; on a second quit attempt while
        # this flag is set, the app escalates to force-kill the worker immediately instead of waiting
        # for the graceful drain deadline.
        self._graceful_quit_in_progress = False

    def compose(self) -> ComposeResult:
        """Lay out the header, status bar, tabbed views, and footer."""
        yield Header(show_clock=True)
        yield Static(id="status-bar")
        with TabbedContent(initial="tab-overview", id="main-tabs"):
            with TabPane("Overview", id="tab-overview"):
                yield SimpleHomeView()
                yield OverviewView()
            with TabPane("Stats", id="tab-stats"):
                yield TabPrimer(
                    "Totals and averages for the work this computer has done. Nothing here needs "
                    "changing; it is a record of what happened.",
                    PRIMERS.get("Stats", ()),
                )
                yield StatsView()
            with TabPane("Control", id="tab-control"):
                yield TabPrimer(
                    "The background programs that do the work. The worker starts and restarts these on "
                    "its own; this is where you would look if one stopped.",
                    PRIMERS.get("Control", ()),
                )
                yield ControlView()
            with TabPane("GPUs", id="tab-gpus"):
                yield TabPrimer(
                    "What your graphics card is doing, and how much of its memory the worker is using.",
                )
                yield GpusView()
            with TabPane("Live", id="tab-live"):
                yield SimpleActivityView()
                yield LiveView()
            with TabPane("Downloads", id="tab-downloads"):
                yield SimpleModelStatusView()
                yield DownloadsView()
            with TabPane("Logs", id="tab-logs"):
                yield TabPrimer(
                    "The worker's running commentary. Useful to copy from when asking for help; you do "
                    "not need to read it to contribute.",
                    PRIMERS.get("Logs", ()),
                )
                yield LogsView()
            with TabPane("Config", id="tab-config"):
                yield ConfigEditorView(
                    self._config_path,
                    experience_level=self._experience_level,
                    display_density=self._display_density,
                    theme_name=self._theme_name,
                )
            with TabPane("Insights", id="tab-insights"):
                yield TabPrimer(
                    "Where the time goes, and what is holding throughput back. Worth a look if you want "
                    "to earn more kudos per hour.",
                    PRIMERS.get("Insights", ()),
                )
                yield InsightsView()
            with TabPane("Diagnostics", id="tab-diagnostics"):
                yield TabPrimer(
                    "Checks on this computer's setup, and the details to include when reporting a problem.",
                )
                yield DiagnosticsView()
            with TabPane("Benchmark", id="tab-benchmark"):
                yield TabPrimer(
                    "Measures what this computer can handle and suggests settings to match. Running one "
                    "is the easiest way to get good settings without tuning by hand.",
                )
                yield BenchmarkView(worker_mode=self._supervisor.mode.value)
        # The level rides the Footer's row rather than claiming one of its own: at the 80x24 floor a
        # dedicated indicator strip costs a line of content for information that changes rarely.
        with Horizontal(id="footer-bar"):
            yield Footer()
            yield Static(id="level-indicator")

    @staticmethod
    def _detect_low_fidelity(encoding: str) -> bool:
        """Return True when the terminal is unlikely to render Unicode block elements correctly.

        The env var HORDE_WORKER_TUI_LOW_FIDELITY=1 forces ASCII mode regardless of encoding.
        Absent that, any non-UTF-8 console encoding (e.g. PuTTY with CP1252) triggers the fallback.
        """
        env_override = os.environ.get("HORDE_WORKER_TUI_LOW_FIDELITY", "").strip().lower()
        if env_override in ("1", "true", "yes"):
            return True
        return encoding.lower().replace("-", "") not in ("utf8", "utf8sig")

    def _build_title(self) -> str:
        snapshot = self._supervisor.latest_snapshot
        if snapshot is not None and snapshot.session_start_time:
            elapsed = int(max(0.0, time.time() - snapshot.session_start_time))
        else:
            elapsed = int(time.monotonic() - self._start_time)
        h, remainder = divmod(elapsed, 3600)
        m, s = divmod(remainder, 60)
        clock = f"{h:02d}:{m:02d}:{s:02d}"
        suffix = " (Update Available)" if self._update_info is not None else ""
        return f"AI Horde Worker - v{__version__} [{clock}]{suffix}"

    def _refresh_title(self) -> None:
        self.title = self._build_title()

    def on_mount(self) -> None:
        """Begin the refresh loop, then run first-run setup or the usual start/onboarding prompts."""
        self._start_time = time.monotonic()
        configure_fidelity(self._detect_low_fidelity(self.console.encoding))
        # Pilot tests drive state changes explicitly. Letting these timers run as well creates an ever-growing
        # render/message backlog that makes teardown take seconds per test without exercising extra behaviour.
        testing = bool(os.environ.get("AI_HORDE_TESTING"))
        if testing:
            self.call_after_refresh(self._tick)
        else:
            self.set_interval(0.1, self._tick)
            self.set_interval(1.0, self._refresh_title)
        # Resolve the models volume from config before any disk figures are computed, so free space and
        # on-disk checks match the worker's configured cache_home instead of defaulting to ./models.
        with contextlib.suppress(Exception):
            apply_cache_home_env(self._config_path)
        # Mirror the worker's default beta opt-in into this process before the catalog warms, so the model
        # picker surfaces pending-queue (beta) models like qwen instead of the canonical-only set.
        with contextlib.suppress(Exception):
            apply_beta_model_env(self._config_path)
        self._maybe_check_for_updates()
        from horde_worker_regen.update_check import UPDATE_CHECK_INTERVAL_SECONDS

        if not testing:
            self.set_interval(UPDATE_CHECK_INTERVAL_SECONDS, self._periodic_update_check)
        self._warm_model_catalog()
        # Fixed for the life of the session (the bind address cannot change under a running server), so it
        # is stamped once here rather than re-evaluated with the level and density classes.
        if self._remote_exposed:
            self.screen.add_class(REMOTE_EXPOSED_CLASS)
        self._apply_experience_level(self._experience_level)
        # An installation that predates the levels answers the notice before anything else, so the choice
        # is made against an untouched dashboard rather than behind a start prompt.
        if self._needs_experience_introduction:
            self.push_screen(ExperienceIntroductionModal(), self._on_experience_introduction)
        else:
            self._resume_launch_flow()

    def _on_experience_introduction(self, level: ExperienceLevel | None) -> None:
        """Apply and persist the answer to the one-time level notice, then resume launching."""
        chosen = level if level is not None else ExperienceLevel.ADVANCED
        self._needs_experience_introduction = False
        with contextlib.suppress(OSError):
            self._app_state_store.resolve_experience_introduction(chosen)
        self._apply_experience_level(chosen)
        self._resume_launch_flow()

    def _resume_launch_flow(self) -> None:
        """Run first-run setup or the usual start/onboarding prompts."""
        if self._attached_lifecycle_pending(self._supervisor):
            # The attach reader starts in the background. Its STOPPED defaults are placeholders until the
            # host sends the first status; acting on them flashes a start/download/stay-stopped modal over
            # an already-running worker. Re-check without issuing any lifecycle intent.
            self.set_timer(0.1, self._resume_launch_flow)
            return
        if self._should_show_getting_started():
            self._open_getting_started()
        elif self._supervisor.is_alive():
            self._maybe_prompt_onboarding()
        elif self._should_auto_start():
            self._supervisor.start()
            self._maybe_prompt_onboarding()
        else:
            self._prompt_worker_start()

    @staticmethod
    def _attached_lifecycle_pending(supervisor: SupervisorLike) -> bool:
        """Whether an attached supervisor still holds its pre-status placeholder lifecycle."""
        return isinstance(supervisor, AttachedWorkerSupervisor) and not supervisor.lifecycle_resolved

    def _warm_model_catalog(self) -> None:
        """Pre-load the image-model catalog in the background so views open instantly (best-effort).

        Funnels through the shared cache, so by the time the operator opens the picker or the Models
        config panel the catalog is usually already in memory instead of triggering a slow first fetch.
        A failure here is silent: the views still load on demand and surface their own errors.
        """
        if os.environ.get("AI_HORDE_TESTING"):
            return
        self.run_worker(self._warm_model_catalog_blocking, thread=True, exclusive=True, group="catalog-warm")

    @staticmethod
    def _warm_model_catalog_blocking() -> None:
        """Blocking catalog warm for the worker thread; swallows failures (the warm is best-effort)."""
        from horde_worker_regen.tui.catalog_cache import CATALOG_CACHE

        with contextlib.suppress(Exception):
            CATALOG_CACHE.ensure_loaded()

    def _maybe_check_for_updates(self) -> None:
        """Kick off a background release check, unless disabled, in fake mode, or under tests."""
        if self._supervisor.mode is not WorkerProcessMode.REAL:
            return
        if os.environ.get("AI_HORDE_TESTING") or os.environ.get("HORDE_WORKER_NO_UPDATE_CHECK"):
            return
        self.run_worker(self._update_check(), group="update-check", exclusive=True)

    def _periodic_update_check(self) -> None:
        """Re-check for a newer release every 30 minutes (called by the interval timer).

        Delegates to :meth:`_maybe_check_for_updates` so the same guards (fake mode, disabled, tests)
        and the same result-surfacing logic apply.
        """
        self._maybe_check_for_updates()

    async def _update_check(self) -> None:
        """Notify (non-blocking) when a newer release is available and how to get it."""
        info = await asyncio.to_thread(check_for_update)
        if info is None:
            return
        self._update_info = info
        self._refresh_title()
        with contextlib.suppress(Exception):
            self.query_one(OverviewView).set_update_available(info)

        self.notify(
            f"Update available: v{runtime_version()} -> v{info.latest_version}. Update with "
            f"'{get_system_appropriate_updater()}', or by re-running the installer.",
            title="Update available",
            timeout=10,
        )

    def _should_show_getting_started(self) -> bool:
        """Whether to open Getting started: a real worker whose bridgeData is not yet configured.

        Skipped for the fake/demo worker and for env-var config (both are power-user paths). When the
        config is already complete, the durable setup-complete flag is set so existing installs are
        never sent to setup.

        Answering this reads and parses ``bridgeData.yaml`` (comment-preserving, so not cheap) and may
        read the durable state, so the answer is cached and refreshed only when the config file itself
        changes. See :meth:`_refresh_setup_required`.
        """
        self._setup_required = self._compute_setup_required()
        self._setup_config_stamp = self._config_stamp()
        return self._setup_required

    def _config_stamp(self) -> tuple[int, int] | None:
        """Return a cheap change stamp for the config file, or None when it is absent."""
        try:
            stat_result = self._config_path.stat()
        except OSError:
            return None
        return (stat_result.st_mtime_ns, stat_result.st_size)

    def _refresh_setup_required(self) -> bool:
        """Return whether setup is outstanding, re-reading the config only when the file has changed.

        The render loop asks this every frame, so the steady-state cost is one ``stat`` rather than a
        comment-preserving YAML parse. Watching the file (rather than only the dashboard's own save path)
        keeps this correct when the config is edited in another editor while the dashboard is open.
        """
        stamp = self._config_stamp()
        if stamp != self._setup_config_stamp:
            return self._should_show_getting_started()
        return self._setup_required

    def _compute_setup_required(self) -> bool:
        """Read the config from disk and decide whether first-run setup is still outstanding."""
        if self._supervisor.mode is not WorkerProcessMode.REAL or self._load_config_from_env_vars:
            return False
        try:
            if not is_setup_incomplete(self._config_path):
                if not self._app_state_store.load().setup_complete:
                    with contextlib.suppress(Exception):
                        self._app_state_store.set_setup_complete(True)
                return False
        except Exception as setup_error:  # noqa: BLE001 - detection must never block the TUI
            self.log(f"Could not determine setup state: {setup_error}")
            return False
        return True

    def _open_getting_started(self) -> None:
        """Show the Getting started page; fall back to the usual start prompt if it cannot be shown."""
        try:
            self.push_screen(GettingStartedScreen(config_path=self._config_path), self._on_getting_started_closed)
        except Exception as setup_error:  # noqa: BLE001 - setup must never block the TUI
            self.log(f"Could not show Getting started: {setup_error}")
            self._prompt_worker_start()

    def _on_getting_started_closed(self, saved: bool | None) -> None:
        """Pick up a saved setup: record it, reload the config views, and say what happens next."""
        if not saved:
            self.notify("Nothing was saved. Reopen Getting started, or use the Config tab, when you're ready.")
            return
        with contextlib.suppress(Exception):
            self._app_state_store.set_setup_complete(True)
        with contextlib.suppress(NoMatches):
            self.query_one(ConfigEditorView).reload_from_disk()
        self.notify(
            "Setup saved. Press F3 to start contributing. Your chosen models download on the first "
            "start; that can take 30-60 minutes, and the worker serves each one as it finishes.",
            title="Setup saved",
            timeout=12,
        )

    def _should_auto_start(self) -> bool:
        """Whether the persisted preference says to start the worker automatically on launch."""
        try:
            return self._app_state_store.load().auto_start_worker
        except Exception as state_error:  # noqa: BLE001 - reading app state must never block the TUI
            self.log(f"Could not read auto-start preference: {state_error}")
            return False

    def _prompt_worker_start(self) -> None:
        """Show the first-run "start the worker?" prompt; the worker stays stopped until the user acts."""
        try:
            self.push_screen(WorkerStartModal(), self._on_worker_start_choice)
        except Exception as prompt_error:  # noqa: BLE001 - the prompt must never block the TUI
            self.log(f"Could not show worker-start prompt: {prompt_error}")

    def _on_worker_start_choice(self, choice: WorkerStartChoice | None) -> None:
        """Apply the first-run choice: start now, persist-and-start, or stay stopped."""
        if choice is None or choice is WorkerStartChoice.STAY_STOPPED:
            self.notify("Worker is stopped. Press F3 to start it.")
            return
        if choice is WorkerStartChoice.DOWNLOAD_ONLY:
            # Start the worker and hold it in download-only mode (the hold is sent once its pipe is up).
            with contextlib.suppress(NoMatches):
                self.query_one("#main-tabs", TabbedContent).active = "tab-downloads"
            self._supervisor.start()
            self._pending_downloads_only_hold = True
            self.notify("Starting in download-only mode: fetching models, the GPU stays idle until you Go live.")
            return
        if choice is WorkerStartChoice.START_AND_REMEMBER:
            with contextlib.suppress(Exception):
                self._app_state_store.set_auto_start_worker(True)
            self.notify("Auto-start enabled. Starting worker…")
        else:
            self.notify("Starting worker…")
        self._supervisor.start()

    def _maybe_prompt_onboarding(self) -> None:
        """Show the first-run benchmark prompt when no current benchmark exists and not declined."""
        try:
            state = self._app_state_store.load()
            if not should_prompt_onboarding(state, current_version=__version__):
                return
            availability = benchmark_status_summary(state, current_version=__version__)
            self.push_screen(BenchmarkOnboardingModal(availability), self._on_onboarding_choice)
        except Exception as onboarding_error:  # noqa: BLE001 - onboarding must never block the TUI
            self.log(f"Could not show onboarding prompt: {onboarding_error}")

    def _on_onboarding_choice(self, choice: OnboardingChoice | None) -> None:
        """Persist the onboarding choice and, when accepted, start the benchmark."""
        if choice is None:
            return
        with contextlib.suppress(Exception):
            self._app_state_store.record_onboarding_choice(choice)
        if choice is OnboardingChoice.ACCEPTED:
            self._pending_benchmark_options = BenchmarkOptions(process_mode=self._supervisor.mode.value)
            self.notify("Stopping worker to free the GPU for the benchmark…")
            self.run_worker(self._start_benchmark_flow, thread=True, exclusive=True, group="lifecycle")

    def _tick(self) -> None:
        """Drain worker state, restart on crash, derive health, and refresh the data views."""
        self._supervisor.tick()
        self._benchmark_supervisor.tick()
        if self._graceful_quit_in_progress and not self._supervisor.is_alive():
            # Cooperative shutdown is complete. Exit from the same UI thread that owns tick/lifecycle state;
            # this avoids racing a background blocking stop against the supervisor tick.
            self._benchmark_supervisor.stop()
            self._supervisor.close()
            self.exit()
            return
        # Flush a deferred download-only hold (and any picker selection) once the freshly-started worker's
        # pipe is up. The hold goes first so inference never starts; the selection follows once it is sent.
        if self._supervisor.is_alive():
            if self._pending_downloads_only_hold and self._supervisor.request_downloads_only_hold():
                self._pending_downloads_only_hold = False
            if self._pending_download_models is not None and not self._pending_downloads_only_hold:
                selection = self._pending_download_models
                if self._supervisor.request_download_models(
                    selection.model_names,
                    include_aux=selection.include_aux,
                ):
                    self._pending_download_models = None
        self._frame += 1
        snapshot = self._supervisor.latest_snapshot
        # Clear the "m" intent once the advisory poll confirms the horde reflects the requested state,
        # or once a real job pop proves the worker is no longer in horde maintenance.
        if self._intended_server_maintenance is not None and snapshot is not None:
            confirmed_by_poll = snapshot.worker_details_maintenance == self._intended_server_maintenance
            cleared_by_successful_pop = (
                self._intended_server_maintenance
                and self._server_maintenance_intent_pop_count is not None
                and snapshot.num_jobs_popped > self._server_maintenance_intent_pop_count
            )
            if confirmed_by_poll or cleared_by_successful_pop:
                self._clear_server_maintenance_intent()
        # Toast exactly once when the pop loop first sees a maintenance-mode error from the horde.
        pop_maint = snapshot.last_pop_maintenance_mode if snapshot is not None else False
        if pop_maint and not self._prev_pop_maintenance_mode:
            self.notify("Server maintenance active: the horde has stopped sending jobs.", severity="warning")
        self._prev_pop_maintenance_mode = pop_maint
        maintenance_mode = snapshot.maintenance_mode if snapshot is not None else False
        if maintenance_mode and not self._prev_maintenance_mode:
            self._maintenance_started_at = time.monotonic()
        elif not maintenance_mode:
            self._maintenance_started_at = None
        self._prev_maintenance_mode = maintenance_mode
        now = time.time()
        snapshot_age = (now - snapshot.timestamp) if snapshot is not None else None
        # Judge responsiveness on liveness (the loop's last tick), not on full-snapshot freshness:
        # a coalesced or briefly-failing snapshot build must not read as "unresponsive". Fall back to
        # snapshot age for an older worker that never sends liveness frames.
        liveness_wall_time = self._supervisor.last_liveness_wall_time
        liveness_age = (now - liveness_wall_time) if liveness_wall_time is not None else snapshot_age
        offline_checks = build_offline_checks(self._config_path) if snapshot is None else None
        report = derive(
            snapshot,
            self._supervisor.status,
            liveness_age,
            offline_checks=offline_checks,
            optimistic_server_maintenance=self._intended_server_maintenance is True,
            fatal_error=self._supervisor.last_fatal_error,
        )
        try:
            self._update_status_bar(report, snapshot)
            overview = self.query_one(OverviewView)
            self._maybe_mark_trend_config_change(overview, snapshot)
            overview.update_view(
                report,
                snapshot,
                frame=self._frame,
                mode=self._view_mode,
                trend_window=self._trend_window,
                show_recent_work_ledger_jobs=self._show_recent_work_ledger_jobs,
                hidden_keys=frozenset(self._overview_hidden),
                reveal_hidden=self._reveal_hidden_elements,
            )
            # The Simple presentations are fed every tick regardless of level, so their trend history and
            # recent-request feed accumulate continuously; switching to Simple then shows real history
            # rather than starting from an empty chart.
            simple_home = self.query_one(SimpleHomeView)
            simple_home.set_setup_required(self._refresh_setup_required())
            simple_home.update_view(report, snapshot, is_alive=self._supervisor.is_alive())
            for primer in self.query(TabPrimer):
                primer.update_view(snapshot, report)
            self.query_one(SimpleActivityView).update_view(snapshot)
            self.query_one(SimpleModelStatusView).update_view(snapshot)
            self.query_one(GpusView).update_view(
                snapshot,
                mode=self._view_mode,
                simple=self._experience_level is ExperienceLevel.SIMPLE,
            )
            self.query_one(DownloadsView).update_view(snapshot, mode=self._view_mode)
            self.query_one(ControlView).update_view(
                snapshot,
                supervisor_status=self._supervisor.status,
                is_alive=self._supervisor.is_alive(),
                restart_attempts=self._supervisor.restart_attempts,
                auto_start=self._auto_start_enabled(),
            )
            self._update_downloads_tab_label(snapshot)
            self.query_one(LogsView).set_view_mode(self._view_mode)
            config_editor = self.query_one(ConfigEditorView)
            config_editor.set_view_mode(self._view_mode)
            config_editor.update_worker_models(
                snapshot.active_models if snapshot is not None else [],
            )
            config_editor.update_cards(snapshot.per_card if snapshot is not None else [])
            if snapshot is not None:
                self.query_one(LiveView).update_snapshot(
                    snapshot,
                    snapshot_age,
                    detailed=self._view_mode is OverviewViewMode.DETAILS,
                )
                self.query_one(InsightsView).update_snapshot(snapshot)
                self.query_one(StatsView).update_snapshot(snapshot)
            self.query_one(StatsView).update_maintenance_start(self._maintenance_started_at)
            self.query_one(DiagnosticsView).update_maintenance_start(self._maintenance_started_at)
            self.query_one(BenchmarkView).update_view(
                self._benchmark_supervisor.run_state,
                self._benchmark_supervisor.status,
                frame=self._frame,
                mode=self._view_mode,
            )
            self._update_benchmark_waiting()
        except NoMatches:
            # The refresh interval can fire during mount or teardown; skip until the DOM is ready.
            pass
        self._handle_benchmark_status_transition()

    def on_stats_view_export_toggled(self, message: StatsView.ExportToggled) -> None:
        """Forward the Stats tab JSONL export toggle to the worker."""
        if not self._supervisor.request_set_stats_export(message.enabled):
            self.notify("Stats export toggle could not be sent; worker is not connected.", severity="warning")
            return
        state = "enabled" if message.enabled else "disabled"
        self.notify(f"Stats JSONL export {state}.")

    def _update_downloads_tab_label(self, snapshot: WorkerStateSnapshot | None) -> None:
        """Badge the Downloads tab with live progress so an active fetch is visible from any tab.

        Idle (no download in flight) the tab reads plainly "Downloads"; while fetching it shows the
        ready/total model count and a pause marker, so the operator does not have to open the tab to see
        that work is happening.
        """
        from horde_worker_regen.tui.widgets.downloads import summarize_download_activity

        activity = summarize_download_activity(snapshot)
        try:
            tab = self.query_one("#main-tabs", TabbedContent).get_tab("tab-downloads")
        except (NoMatches, ValueError):
            return
        if activity is None:
            tab.label = "Downloads"
        else:
            marker = "⏸" if activity.paused else "⬇"
            count = f" {activity.ready}/{activity.total}" if activity.total is not None else ""
            tab.label = f"Downloads {marker}{count}"

    def _handle_benchmark_status_transition(self) -> None:
        """Notify and refresh persisted status when the benchmark finishes, fails, or is cancelled."""
        status = self._benchmark_supervisor.status
        if status == self._last_benchmark_status:
            return
        self._last_benchmark_status = status
        if status is BenchmarkSupervisorStatus.FINISHED:
            if self._benchmark_drained_worker and self._supervisor.is_alive():
                self.notify("Benchmark finished. Apply the suggested config, or press Go live to resume serving.")
            else:
                self.notify("Benchmark finished. Apply the suggested config, or press F9 to restart the worker.")
        elif status is BenchmarkSupervisorStatus.FAILED:
            self.notify(f"Benchmark failed; see the run's console.log.{self._resume_hint()}", severity="error")
        elif status is BenchmarkSupervisorStatus.CANCELLED:
            self.notify(f"Benchmark cancelled.{self._resume_hint()}")
        if status in (BenchmarkSupervisorStatus.FINISHED, BenchmarkSupervisorStatus.FAILED):
            with contextlib.suppress(NoMatches):
                self.query_one(BenchmarkView).refresh_app_state_summary()

    def action_show_benchmark(self) -> None:
        """Switch to the Benchmark tab."""
        with contextlib.suppress(NoMatches):
            self.query_one("#main-tabs", TabbedContent).active = "tab-benchmark"

    def action_show_diagnostics(self) -> None:
        """Switch to the Diagnostics tab (the activation handler kicks off its first analysis)."""
        with contextlib.suppress(NoMatches):
            self.query_one("#main-tabs", TabbedContent).active = "tab-diagnostics"

    def action_toggle_download_pause(self) -> None:
        """Pause or resume background downloads based on the latest reported state."""
        snapshot = self._supervisor.latest_snapshot
        currently_paused = snapshot is not None and snapshot.downloads is not None and snapshot.downloads.paused
        self._set_downloads_paused(currently_paused=currently_paused)

    def _set_downloads_paused(self, *, currently_paused: bool) -> None:
        """Send the resume/pause download command and notify, given the current paused state."""
        if currently_paused:
            sent = self._supervisor.request_resume_downloads()
            self.notify("Resuming downloads." if sent else "Worker not running; resume not sent.")
        else:
            sent = self._supervisor.request_pause_downloads()
            self.notify("Pausing downloads." if sent else "Worker not running; pause not sent.")

    def on_downloads_view_pause_toggle_requested(self, message: DownloadsView.PauseToggleRequested) -> None:
        """Forward a Downloads-panel pause/resume click to the worker."""
        self._set_downloads_paused(currently_paused=message.currently_paused)

    def on_control_view_toggle_pause_requested(self, _message: ControlView.TogglePauseRequested) -> None:
        """Forward the Control tab's local pause/resume request."""
        self.action_toggle_pause()

    def on_control_view_toggle_auto_start_requested(self, _message: ControlView.ToggleAutoStartRequested) -> None:
        """Forward the Control tab's auto-start toggle request."""
        self.action_toggle_autostart()
        self._tick()

    def on_control_view_start_stop_requested(self, _message: ControlView.StartStopRequested) -> None:
        """Forward the Control tab's start/stop request."""
        self.action_start_stop_worker()

    def on_control_view_restart_requested(self, _message: ControlView.RestartRequested) -> None:
        """Forward the Control tab's restart request."""
        self.action_restart_worker()

    def on_control_view_toggle_server_maintenance_requested(
        self,
        _message: ControlView.ToggleServerMaintenanceRequested,
    ) -> None:
        """Forward the Control tab's horde-maintenance request."""
        self.action_toggle_server_maintenance()

    def on_downloads_view_rate_limit_requested(self, message: DownloadsView.RateLimitRequested) -> None:
        """Forward a Downloads-panel bandwidth-cap change to the worker."""
        sent = self._supervisor.request_download_rate_limit(message.kbps)
        if not sent:
            self.notify("Worker not running; rate limit not sent.", severity="warning")
        elif message.kbps <= 0:
            self.notify("Download rate limit cleared (unlimited).")
        else:
            self.notify(f"Download rate limited to {message.kbps} KB/s.")

    def on_downloads_view_downloads_only_hold_requested(
        self,
        _message: DownloadsView.DownloadsOnlyHoldRequested,
    ) -> None:
        """Pre-fetch models without committing the GPU: start the worker (if needed), then hold it.

        Starting the worker brings up the download process (which fetches the configured models); the
        hold keeps inference and job-popping deferred until the operator presses Go live.
        """
        with contextlib.suppress(NoMatches):
            self.query_one("#main-tabs", TabbedContent).active = "tab-downloads"
        if not self._supervisor.is_alive():
            # Start the worker, then send the hold once its control pipe is up (see _tick); sending now
            # would race the child's connection. A cold install has no models present, so inference will
            # not start in the meantime.
            self._supervisor.start()
            self._pending_downloads_only_hold = True
            self.notify("Starting the worker in download-only mode: fetching models, the GPU stays idle.")
            return
        if self._supervisor.request_downloads_only_hold():
            self.notify("Download-only mode: fetching models; the worker will not serve jobs until you Go live.")
        else:
            self.notify("Could not enter download-only mode (worker not reachable).", severity="warning")

    def on_downloads_view_go_live_requested(self, _message: DownloadsView.GoLiveRequested) -> None:
        """Leave download-only mode so the worker serves jobs, warning first if it would strand a benchmark fetch."""
        if self._benchmark_waiting_incomplete() and self._benchmark_waiting_outside_config():
            # Going live resumes serving, which can stop benchmark-only downloads that the config would not
            # re-fetch. (When every waited-for model IS in the config, serving downloads them anyway: no warning.)
            self.push_screen(
                BenchmarkActionConfirmModal(
                    title="Go live while benchmark models download?",
                    body=(
                        "Going live resumes serving and may stop the benchmark-only model downloads still in "
                        "progress; they are not in this worker's config, so serving will not re-fetch them. "
                        "Continue, or cancel and let them finish first?"
                    ),
                    confirm_label="Go live anyway",
                ),
                self._on_go_live_while_waiting_choice,
            )
            return
        self._do_go_live()

    def _on_go_live_while_waiting_choice(self, confirmed: bool | None) -> None:
        """Go live only if the operator accepted interrupting the in-progress benchmark download."""
        if confirmed:
            self._clear_benchmark_waiting()  # serving may stop the benchmark-only fetch; leave the waiting mode
            self._do_go_live()

    def _do_go_live(self) -> None:
        """Send the go-live request and report whether the worker will start serving."""
        sent = self._supervisor.request_go_live()
        self.notify(
            "Going live: the worker will start serving jobs as models finish downloading."
            if sent
            else "Worker not running; Go live not sent.",
            severity="information" if sent else "warning",
        )

    def on_downloads_view_download_picker_requested(
        self,
        _message: DownloadsView.DownloadPickerRequested,
    ) -> None:
        """Open the picker (defaulted to the config's missing models), then download the chosen set."""
        rows = self._download_picker_rows()
        self.push_screen(DownloadPickerModal(rows), self._on_download_selection)

    def _download_picker_rows(self) -> list[DownloadPickerRow]:
        """Build the picker rows from the Config tab's resolved model set (empty when not resolved yet)."""
        try:
            manager = self.query_one(ModelManagerView)
        except NoMatches:
            return []
        return [
            DownloadPickerRow(
                name=model.name,
                baseline=model.baseline,
                size_bytes=model.size_bytes,
                on_disk=model.on_disk,
            )
            for model in manager.configured_included_models()
        ]

    def _on_download_selection(self, selection: DownloadSelection | None) -> None:
        """Turn a confirmed picker selection into a download request (entering the hold first when needed)."""
        if selection is None:
            return
        with contextlib.suppress(NoMatches):
            self.query_one("#main-tabs", TabbedContent).active = "tab-downloads"
        if not self._supervisor.is_alive():
            # Start the worker, then send the hold + the selection once its pipe is up (see _tick); a cold
            # install has nothing present, so inference will not start in the meantime.
            self._supervisor.start()
            self._pending_downloads_only_hold = True
            self._pending_download_models = selection
            self.notify("Starting the worker to download the selected models (the GPU stays idle).")
            return
        self._supervisor.request_downloads_only_hold()
        sent = self._supervisor.request_download_models(selection.model_names, include_aux=selection.include_aux)
        if sent:
            count = len(selection.model_names)
            aux = " plus auxiliary models" if selection.include_aux else ""
            self.notify(f"Downloading {count} selected model(s){aux}; the worker stays in download-only hold.")
        else:
            self.notify("Could not request the download (worker not reachable).", severity="warning")

    def _clear_server_maintenance_intent(self) -> None:
        """Drop the optimistic server-maintenance command tracking once live state supersedes it."""
        self._intended_server_maintenance = None
        self._server_maintenance_intent_pop_count = None

    def _paused_source(self, snapshot: WorkerStateSnapshot | None) -> str:
        """Return a short source tag for the PAUSED badge (e.g. 'server', 'local', 'auto', 'pop')."""
        if snapshot is None:
            return ""
        if snapshot.worker_details_maintenance or snapshot.worker_details_paused:
            return "server"
        if snapshot.last_pop_maintenance_mode:
            return "pop"
        if snapshot.self_throttle_paused:
            return "auto"
        if snapshot.supervisor_paused:
            return "local"
        return ""

    @staticmethod
    def _trend_config_fingerprint(snapshot: WorkerStateSnapshot | None) -> tuple[object, ...] | None:
        """Return the capacity/workload config fields that affect trend interpretation."""
        if snapshot is None:
            return None
        config = snapshot.config
        return (
            config.num_models,
            config.max_power,
            config.max_threads,
            config.queue_size,
            config.max_batch,
            config.safety_on_gpu,
            config.allow_img2img,
            config.allow_lora,
            config.effective_allow_lora,
            config.allow_controlnet,
            config.allow_sdxl_controlnet,
            config.allow_post_processing,
            config.high_performance_mode,
            config.moderate_performance_mode,
            config.extra_slow_worker,
            config.alchemist,
            config.alchemy_concurrent,
            config.alchemy_max_concurrency,
            config.alchemy_vram_headroom_mb,
            tuple(config.alchemy_forms),
        )

    def _maybe_mark_trend_config_change(self, overview: OverviewView, snapshot: WorkerStateSnapshot | None) -> None:
        """Show a trend stabilization disclaimer when relevant worker config changes."""
        fingerprint = self._trend_config_fingerprint(snapshot)
        if fingerprint is None:
            return
        if self._last_trend_config_fingerprint is None:
            self._last_trend_config_fingerprint = fingerprint
            return
        if fingerprint != self._last_trend_config_fingerprint:
            self._last_trend_config_fingerprint = fingerprint
            overview.note_config_changed()

    # Status-bar segments are laid out most-important-first and shed from the end on a narrow terminal.
    _STATUS_BAR_SEPARATOR = "   "

    def _update_status_bar(self, report: HealthReport, snapshot: WorkerStateSnapshot | None) -> None:
        """Render the always-visible top status bar: phase, a health summary, and at-a-glance vitals.

        The bar is the single cross-tab summary. Health, RAM, GPU and pipeline are inlined here (rather than
        only on the Overview) so they are legible from any tab, and the segments shed from the end when the
        terminal is too narrow to hold them all, so nothing is ever truncated mid-word at 80 columns.
        """
        phase_text = report.phase.value.upper()
        if report.phase is WorkerPhase.MAINTENANCE:
            source = self._paused_source(snapshot) or "server"
            phase_text = f"MAINT·{source}"
        elif report.phase is WorkerPhase.PAUSED:
            source = self._paused_source(snapshot)
            if source:
                phase_text = f"PAUSED·{source}"

        # Priority order: the phase badge and any instability signal lead, then the health summary and live
        # vitals, then the slower-moving identity/session counters that shed first.
        parts = [f"[black on {self._badge_colour(report.severity)}] {phase_text} [/]"]
        if self._supervisor.restart_attempts:
            parts.append(f"[yellow]restarts {self._supervisor.restart_attempts}[/]")
        stall_markup = self._supervisor_stall_markup(self._supervisor.stall_stats)
        if stall_markup is not None:
            parts.append(stall_markup)
        parts.append(self._health_summary_markup(report))
        if snapshot is not None:
            parts.append(self._pipeline_markup(snapshot))
            if snapshot.gpu_utilization_mean_percent is not None:
                parts.append(f"[grey62]gpu[/] {format_percent(snapshot.gpu_utilization_mean_percent)}")
            ram = self._ram_percent_markup(snapshot)
            if ram is not None:
                parts.append(ram)
            kudos = "-" if snapshot.kudos_per_hour is None else f"{snapshot.kudos_per_hour:,.0f}"
            parts.append(f"[grey62]kudos/hr[/] {kudos}")
            parts.append(f"[grey62]done[/] {snapshot.num_jobs_submitted}")
            parts.append(f"[yellow]faulted[/] {snapshot.num_jobs_faulted}" if snapshot.num_jobs_faulted else "")
            parts.append(f"[grey62]worker[/] {snapshot.config.dreamer_name}")
        parts.append(f"[grey62]mode[/] {self._supervisor.mode.value}")

        parts = [part for part in parts if part]
        self.query_one("#status-bar", Static).update(Text.from_markup(self._fit_status_parts(parts)))

    def _fit_status_parts(self, parts: list[str]) -> str:
        """Join ``parts`` (priority-ordered markup) with separators, dropping the tail that will not fit.

        The first segment (the phase badge) is always kept even if it alone exceeds the width, so the bar is
        never blank; every later segment is admitted only while the running visible width still fits.
        """
        width = self.size.width
        if not width or width <= 0:
            return self._STATUS_BAR_SEPARATOR.join(parts)
        budget = width - 2  # the status bar carries one column of horizontal padding on each side
        kept: list[str] = []
        used = 0
        for part in parts:
            visible = Text.from_markup(part).cell_len
            extra = visible + (len(self._STATUS_BAR_SEPARATOR) if kept else 0)
            if kept and used + extra > budget:
                break
            kept.append(part)
            used += extra
        return self._STATUS_BAR_SEPARATOR.join(kept or parts[:1])

    @staticmethod
    def _health_summary_markup(report: HealthReport) -> str:
        """One compact health segment: the worst check when something needs attention, else an OK tally.

        Relies on :class:`HealthStatus` being an ``IntEnum`` ordered so a worse outcome compares greater.
        """
        checks = report.checks
        if not checks:
            return "[grey62]health[/] [grey50]-[/]"
        worst = max(checks, key=lambda check: check.status)
        if worst.status <= HealthStatus.INFO:
            ok_count = sum(1 for check in checks if check.status is HealthStatus.OK)
            return f"[grey62]health[/] [green]{ok_count}/{len(checks)} ok[/]"
        colour = worst.status.colour
        return f"[grey62]health[/] [{colour}]{worst.status.glyph} {worst.name}[/]"

    @staticmethod
    def _pipeline_markup(snapshot: WorkerStateSnapshot) -> str:
        """The compact ``q▸inf▸saf▸sub`` pipeline segment for the status bar."""
        safety = snapshot.jobs_pending_safety_check + snapshot.jobs_being_safety_checked
        return (
            f"[grey62]q[/][cyan]{snapshot.jobs_pending_inference}[/]"
            f"[grey62]▸inf[/][green]{snapshot.jobs_in_progress}[/]"
            f"[grey62]▸saf[/][grey70]{safety}[/]"
            f"[grey62]▸sub[/][cyan]{snapshot.jobs_pending_submit}[/]"
        )

    @staticmethod
    def _ram_percent_markup(snapshot: WorkerStateSnapshot) -> str | None:
        """The system-RAM percentage segment, or None when no memory sample has arrived yet."""
        wire = snapshot.system_memory
        if wire is None or wire.total_bytes <= 0:
            return None
        used_fraction = wire.to_summary().used_fraction
        if used_fraction is None:
            return None
        return f"[grey62]ram[/] {format_percent(used_fraction * 100)}"

    @staticmethod
    def _supervisor_stall_markup(stall: SupervisorStallStats) -> str | None:
        """The supervisor's own liveness segment, or None while it has had nothing to forgive.

        Every other segment describes the worker; this one describes the process watching it. A supervisor
        starved by its host re-graces the worker's wedge baseline to avoid blaming it for time it could not
        observe, and while that is happening the worker's wedge backstop is effectively held off, so the
        condition has to be visible rather than inferred from an alarm that never arrives. Shown only once
        a gap has been forgiven, since the healthy case is the absence of the segment.
        """
        if not stall.resets_in_window and not stall.budget_spent:
            return None
        colour = "red" if stall.budget_spent else "yellow"
        gap = f" gap {stall.largest_tick_gap_seconds:.0f}s" if stall.largest_tick_gap_seconds else ""
        return f"[{colour}]stalls {stall.resets_in_window}/{stall.max_forgiven_resets}{gap}[/]"

    @staticmethod
    def _badge_colour(severity: HealthStatus) -> str:
        """Background colour for the status-bar phase badge."""
        return {
            HealthStatus.OK: "green",
            HealthStatus.INFO: "grey50",
            HealthStatus.WARN: "yellow",
            HealthStatus.ERROR: "red",
        }[severity]

    def _auto_start_enabled(self) -> bool:
        """Return the persisted launch auto-start flag, defaulting safely on read failure."""
        try:
            return self._app_state_store.load().auto_start_worker
        except Exception:
            return False

    _TREND_WINDOW_CYCLE = (
        OverviewTrendWindow.FIVE_MINUTES,
        OverviewTrendWindow.FIFTEEN_MINUTES,
        OverviewTrendWindow.THIRTY_MINUTES,
        OverviewTrendWindow.SIXTY_MINUTES,
        OverviewTrendWindow.TWO_HOURS,
        OverviewTrendWindow.ALL,
    )
    """The order the Overview trend-window shortcut cycles through."""
    _VIEW_MODE_CYCLE = (OverviewViewMode.NORMAL, OverviewViewMode.DETAILS, OverviewViewMode.THIN)
    """The order F6 steps through: the lean redesign, the verbose detail view, then the thin bar."""

    _VIEW_MODE_NOTICE = {
        OverviewViewMode.NORMAL: "View: normal (the everyday density, across all tabs).",
        OverviewViewMode.DETAILS: "View: details (every diagnostic: extra columns, log tally, all config sub-tabs).",
        OverviewViewMode.THIN: "View: thin (essentials only: status bar, slim downloads, bare log, Essentials).",
    }

    def destinations(self) -> list[tuple[str, str]]:
        """Return ``(label, tab id)`` for every top-level destination, in tab order.

        Read from the mounted tab bar rather than a hand-kept list, so a destination added to ``compose``
        is offered in the command palette without a second edit; a copy kept alongside would silently
        omit it. The labels are the tab captions themselves: the level changes what a destination renders,
        not what it is called, so a name learned in Simple still finds the same place in Developer.
        """
        try:
            tabs = self.query_one("#main-tabs", TabbedContent)
        except NoMatches:
            return []
        found: list[tuple[str, str]] = []
        for pane in tabs.query(TabPane):
            # Config nests its own TabbedContent, whose panes carry the "cfgtab-" prefix; only the
            # top-level destinations belong in the palette.
            if not pane.id or not pane.id.startswith(_DESTINATION_ID_PREFIX):
                continue
            with contextlib.suppress(ValueError):
                found.append((tabs.get_tab(pane.id).label_text, pane.id))
        return found

    def on_resize(self, event: events.Resize) -> None:
        """Match the tab strip's labels to the new width band.

        Reads the width off the event rather than ``self.size``, which still holds the old value while
        this handler runs and would leave the strip a resize behind. On a phone, a height reduction is
        commonly the software keyboard opening; Textual preserves focus but does not automatically move
        the focused field above the new bottom edge, so expose it again after the resized layout settles.
        """
        self._apply_tab_labels(compact=event.size.width < PHONE_BAND_MAX_WIDTH)
        if event.size.width < PHONE_BAND_MAX_WIDTH:
            self.call_after_refresh(self._scroll_focused_widget_visible)

    def _scroll_focused_widget_visible(self) -> None:
        """Bring the focused control into the phone viewport after browser chrome changes its height."""
        focused = self.focused
        if focused is not None:
            focused.scroll_visible(animate=False)

    def _apply_tab_labels(self, *, compact: bool) -> None:
        """Show the compact or the full tab labels, doing nothing when they are already the wanted set.

        Args:
            compact: Whether to show the truncated labels that fit a phone-width strip.
        """
        if compact == self._tab_labels_are_compact and self._full_tab_labels:
            return
        try:
            tabs = self.query_one("#main-tabs", TabbedContent)
        except NoMatches:
            return
        for pane_id, compact_label in _COMPACT_TAB_LABELS.items():
            with contextlib.suppress(ValueError):
                tab = tabs.get_tab(pane_id)
                self._full_tab_labels.setdefault(pane_id, tab.label_text)
                tab.label = compact_label if compact else self._full_tab_labels[pane_id]
        self._tab_labels_are_compact = compact

    def _apply_experience_level(self, level: ExperienceLevel) -> None:
        """Apply ``level`` to the screen classes, the footer indicator, and the views that vary by level.

        Deliberately does not touch tab visibility. Every destination exists at every level, so raising
        the level reveals detail in place rather than making navigation appear or disappear underneath
        someone who has just learned where things are.
        """
        self._experience_level = level
        # Each class mutation on the Screen re-applies the stylesheet across every node beneath it, and
        # this DOM runs to hundreds of widgets. Compute the target set and write it once, skipping the
        # write entirely when it already matches, rather than dropping and re-adding four classes.
        wanted = {f"level-{level.value}", f"density-{self._display_density.value}"}
        current = {name for name in self.screen.classes if name.startswith(("level-", "density-"))}
        if current != wanted:
            self.screen.set_classes(sorted((set(self.screen.classes) - current) | wanted))
        with contextlib.suppress(NoMatches):
            self.query_one("#level-indicator", Static).update(level.value.title())
        # Overview, Live and Downloads each host a Simple presentation beside the operator one and swap
        # between them. The destination itself never moves, so this changes depth in place.
        simple = level is ExperienceLevel.SIMPLE
        for simple_view, operator_view in (
            (SimpleHomeView, OverviewView),
            (SimpleActivityView, LiveView),
            (SimpleModelStatusView, DownloadsView),
        ):
            for view_type, wanted in ((simple_view, simple), (operator_view, not simple)):
                with contextlib.suppress(NoMatches):
                    view = self.query_one(view_type)
                    # Assigning display re-applies the stylesheet beneath the widget even when the value
                    # is unchanged, and this runs on every level application.
                    if view.display != wanted:
                        view.display = wanted
        with contextlib.suppress(NoMatches):
            self.query_one(ConfigEditorView).set_experience_level(level)

    _OPERATOR_VIEW_ACTIONS = frozenset({"customize_overview", "toggle_hidden_reveal", "cycle_view_mode"})
    """Actions whose subject is the operator Overview, which Simple replaces with its own view.

    Customising the Overview layout, revealing elements hidden from it, and cycling its density all
    address a widget that is off screen at that level, so a footer hint for one of them offers a control
    with nothing to act on. Navigation is unaffected: every destination stays reachable everywhere.
    """

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Withhold the operator-view actions in Simple so no shortcut is offered without a subject."""
        withheld = action in self._OPERATOR_VIEW_ACTIONS and self._experience_level is ExperienceLevel.SIMPLE
        return not withheld

    def _request_experience_level(self, level: ExperienceLevel) -> None:
        """Change level, guarding the first entry into Developer behind its one-time warning."""
        if level is self._experience_level:
            return
        if level is ExperienceLevel.DEVELOPER and not self._developer_warning_acknowledged:
            self.push_screen(DeveloperWarningModal(), self._on_developer_warning)
            return
        self._commit_experience_level(level)

    def _on_developer_warning(self, accepted: bool | None) -> None:
        """Enter Developer only if the warning was accepted, remembering the acknowledgement."""
        if not accepted:
            return
        self._developer_warning_acknowledged = True
        with contextlib.suppress(OSError):
            self._app_state_store.acknowledge_developer_warning()
        self._commit_experience_level(ExperienceLevel.DEVELOPER)

    def _commit_experience_level(self, level: ExperienceLevel) -> None:
        """Persist and apply ``level``, then refresh so the change is visible immediately."""
        with contextlib.suppress(OSError):
            self._app_state_store.set_experience_level(level)
        self._apply_experience_level(level)
        self.notify(f"Experience: {level.value.title()}.")
        self._tick()

    def set_display_density(self, density: DisplayDensity) -> None:
        """Persist and apply the Advanced/Developer spacing density.

        A no-op change returns early. Textual emits ``Select.Changed`` when the control takes its
        starting value during compose, so without this guard simply opening the Config tab restyles the
        whole DOM and rewrites the state file for a density nobody chose.
        """
        if density is self._display_density:
            return
        self._display_density = density
        with contextlib.suppress(OSError):
            self._app_state_store.set_display_density(density)
        self._apply_experience_level(self._experience_level)

    def set_theme_name(self, theme_name: str) -> None:
        """Persist and apply a registered Horde theme, ignoring an unknown or unchanged one."""
        if theme_name not in KNOWN_THEME_NAMES or theme_name == self._theme_name:
            return
        self._theme_name = theme_name
        self.theme = theme_name
        with contextlib.suppress(OSError):
            self._app_state_store.set_theme_name(theme_name)

    def keyboard_actions(self, screen: Screen[Any] | None = None) -> list[tuple[str, str, str]]:
        """Return ``(keys, description, action)`` for every shortcut ``screen`` currently offers.

        Read from Textual's own active-binding table, which is the same source the Footer renders from,
        so the palette and the help modal list exactly what the Footer would show if it had the room:
        a disabled or level-withheld key drops out of all three together, and a binding added to
        ``BINDINGS`` appears in all three without a second edit.

        Keys that share an action are grouped onto one entry, so the two quit keys read as one command
        with two shortcuts rather than as two commands.

        Args:
            screen: The screen whose bindings to describe; defaults to the active one. The command
                palette passes the screen it was opened over, since by then it is itself on top and its
                own bindings are not what the operator is looking for.
        """
        source = screen if screen is not None else self.screen
        by_action: dict[str, tuple[list[str], str]] = {}
        for key, active in source.active_bindings.items():
            binding = active.binding
            if not binding.show or not active.enabled:
                continue
            display = self._key_display(binding, fallback=key)
            keys, _description = by_action.setdefault(binding.action, ([], binding.description))
            if display not in keys:
                keys.append(display)
        return [(", ".join(keys), description, action) for action, (keys, description) in by_action.items()]

    def _key_display(self, binding: Binding, *, fallback: str) -> str:
        """Return the printable form of a binding's key, falling back to the raw key name.

        Textual derives this from the active keymap and terminal capabilities, so it can fail on an
        unusual key definition. The raw key name still identifies the shortcut, and this list has to stay
        complete for the palette and the help modal to remain usable when the footer truncates.
        """
        try:
            return self.get_key_display(binding)
        except (KeyError, ValueError) as display_error:
            self.log(f"Could not render the key for {binding.action}: {display_error}")
            return fallback

    def _invoke_action(self, action: str) -> None:
        """Run a named action off the palette, scheduled so the palette can close first."""
        self.call_later(self.run_action, action)

    @override
    def get_system_commands(self, screen: Screen[Any]) -> Iterable[SystemCommand]:
        """Add every destination, keyboard action, and experience level to Textual's command palette."""
        yield from super().get_system_commands(screen)
        for label, destination in self.destinations():
            yield SystemCommand(
                f"Go to {label}",
                "Open this dashboard destination",
                partial(self._navigate_to, destination),
            )
        # Every keyboard action is offered here with its shortcut attached. The Footer can only fit a
        # handful of hints on a real terminal (and drops the tail entirely when narrow), so the palette
        # is the surface that stays complete, and it teaches the shortcut rather than merely running it.
        for keys, description, action in self.keyboard_actions(screen):
            yield SystemCommand(
                f"{description}  ({keys})",
                "Keyboard action",
                partial(self._invoke_action, action),
            )
        for level in ExperienceLevel:
            yield SystemCommand(
                f"Use {level.value.title()} experience",
                "Change how much worker detail is shown",
                partial(self._request_experience_level, level),
            )

    def _navigate_to(self, destination: str) -> None:
        """Activate a destination by tab id.

        No level promotion is needed: every tab is reachable at every level, so a palette jump never has
        to change the operator's chosen depth to satisfy itself.

        Focus is dropped before the switch. Hiding the outgoing pane would otherwise blur its focused
        widget, and Textual's focus reset then lands on a visible sibling inside that same pane, whose
        ``TabPane.Focused`` immediately reactivates the tab being left.
        """
        with contextlib.suppress(NoMatches, ValueError):
            self.screen.set_focus(None)
            self.query_one("#main-tabs", TabbedContent).active = destination

    def action_show_help(self) -> None:
        """Explain the dashboard at the level currently in effect, listing every shortcut."""
        self.push_screen(HelpModal(self._experience_level, self.keyboard_actions()))

    def on_simple_home_view_start_stop_requested(self, message: SimpleHomeView.StartStopRequested) -> None:
        """Start or stop the worker from the Simple home action."""
        self.action_start_stop_worker()

    def on_simple_home_view_setup_requested(self, message: SimpleHomeView.SetupRequested) -> None:
        """Open Getting started, which is the one place setup is explained and performed."""
        self._open_getting_started()

    _SIMPLE_DESTINATION_TABS: dict[SimpleDestination, str] = {
        SimpleDestination.ACTIVITY: _LIVE_TAB_ID,
        SimpleDestination.MODELS: _DOWNLOADS_TAB_ID,
    }
    """Where each Simple intent lands. The dashboard owns this mapping so the Simple views need not
    know any tab identifier."""

    def on_simple_home_view_navigate_requested(self, message: SimpleHomeView.NavigateRequested) -> None:
        """Follow a Simple home link to another destination."""
        self._navigate_to(self._SIMPLE_DESTINATION_TABS[message.destination])

    def on_simple_model_status_view_manage_requested(
        self,
        message: SimpleModelStatusView.ManageRequested,
    ) -> None:
        """Send the contributor to the one place models are chosen."""
        self._navigate_to(_CONFIG_TAB_ID)
        with contextlib.suppress(NoMatches, ValueError):
            self.query_one(ConfigEditorView).open_subtab(MODELS_SUBTAB_ID)

    def on_dashboard_preferences_view_experience_level_changed(
        self,
        message: DashboardPreferencesView.ExperienceLevelChanged,
    ) -> None:
        """Apply a level chosen from the Config tab."""
        self._request_experience_level(message.level)

    def on_dashboard_preferences_view_density_changed(
        self,
        message: DashboardPreferencesView.DensityChanged,
    ) -> None:
        """Apply a density chosen from the Config tab."""
        self.set_display_density(message.density)

    def on_dashboard_preferences_view_theme_changed(
        self,
        message: DashboardPreferencesView.ThemeChanged,
    ) -> None:
        """Apply a theme chosen from the Config tab."""
        self.set_theme_name(message.theme_name)

    def action_cycle_view_mode(self) -> None:
        """Cycle (and persist) the shared density mode: normal -> details -> thin, then refresh now.

        The mode is app-wide: every tab that honours the density contract (Overview, Live, Downloads,
        Logs, Config, Benchmark) reads the same setting, so one F6 press re-densifies the whole dashboard.
        """
        index = self._VIEW_MODE_CYCLE.index(self._view_mode) if self._view_mode in self._VIEW_MODE_CYCLE else 0
        self._view_mode = self._VIEW_MODE_CYCLE[(index + 1) % len(self._VIEW_MODE_CYCLE)]
        with contextlib.suppress(Exception):
            self._app_state_store.set_view_mode(self._view_mode)
        self.notify(self._VIEW_MODE_NOTICE[self._view_mode])
        self._tick()

    def action_cycle_trend_window(self) -> None:
        """Cycle the Overview trend window and persist the selected span."""
        index = (
            self._TREND_WINDOW_CYCLE.index(self._trend_window) if self._trend_window in self._TREND_WINDOW_CYCLE else 0
        )
        self._trend_window = self._TREND_WINDOW_CYCLE[(index + 1) % len(self._TREND_WINDOW_CYCLE)]
        with contextlib.suppress(Exception):
            self._app_state_store.set_trend_window(self._trend_window)
        label = "All" if self._trend_window is OverviewTrendWindow.ALL else self._trend_window.value
        with contextlib.suppress(NoMatches):
            self.query_one(OverviewView).set_trend_window(self._trend_window)
        self.notify(f"Trend window: {label}")
        self._tick()

    def action_reset_trends(self) -> None:
        """Reset the Overview trend view: clear the (display-only) sample buffers and start a fresh epoch."""
        with contextlib.suppress(NoMatches):
            self.query_one(OverviewView).soft_reset_trends()
        self.notify("Overview trends reset.")
        self._tick()

    def action_toggle_work_ledger_recent_jobs(self) -> None:
        """Show or hide recently finished rows in the Overview work ledger."""
        self._show_recent_work_ledger_jobs = not self._show_recent_work_ledger_jobs
        state = "shown" if self._show_recent_work_ledger_jobs else "summarized"
        self.notify(f"Work ledger recent jobs: {state}.")
        self._tick()

    def action_customize_overview(self) -> None:
        """Open the customize-layout modal; persist the chosen hidden set when it closes."""
        self.push_screen(OverviewLayoutModal(frozenset(self._overview_hidden)), self._on_overview_layout_chosen)

    def _on_overview_layout_chosen(self, hidden: frozenset[str] | None) -> None:
        """Store the operator's hidden-element choice and refresh the Overview immediately."""
        if hidden is None:
            return
        self._overview_hidden = set(hidden)
        # A fresh customize pass is an explicit re-curation, so drop any temporary quick-reveal state: what
        # the operator just chose is what they should see.
        self._reveal_hidden_elements = False
        with contextlib.suppress(Exception):
            self._app_state_store.set_overview_hidden_elements(self._overview_hidden)
        self.notify(f"Overview layout saved ({len(self._overview_hidden)} hidden).")
        self._tick()

    def action_toggle_hidden_reveal(self) -> None:
        """Temporarily reveal (or re-hide) every element the operator has marked hidden.

        This is a session-only convenience: it never changes the persisted hidden set, so it is a quick way
        to glance at a demoted panel without re-editing the layout.
        """
        if not self._overview_hidden:
            self.notify("No overview elements are hidden.")
            return
        self._reveal_hidden_elements = not self._reveal_hidden_elements
        state = "revealed" if self._reveal_hidden_elements else "re-hidden"
        self.notify(f"Hidden overview elements {state}.")
        self._tick()

    def action_toggle_pause(self) -> None:
        """Pause or resume the worker (a *local* pop-pause) depending on its current state.

        This is the local pause: in-flight jobs finish, no new ones are popped. It does not by itself
        change the worker's server-side maintenance on the horde; the worker clears that on resume only
        when its ``remove_maintenance_on_init`` config is set. Use the Maintenance (horde) key for an
        explicit server-side toggle.
        """
        snapshot = self._supervisor.latest_snapshot
        # Read supervisor_paused directly: it is the flag the local pause control changes. Using the aggregate
        # maintenance_mode here would latch permanently when the horde forces maintenance, because
        # that flag (last_pop_maintenance_mode) is not cleared by RESUME - only a successful pop clears it.
        if snapshot is not None and snapshot.supervisor_paused:
            self._supervisor.request_resume()
            self.notify("Resume requested.")
        else:
            self._supervisor.request_pause()
            self.notify("Pause requested (in-flight jobs will finish).")

    def action_toggle_server_maintenance(self) -> None:
        """Toggle the worker's server-side (horde) maintenance flag via the horde API.

        Distinct from local pause: this asks the horde itself to stop (or resume) sending the worker
        jobs, matching the maintenance the job-pop response reports. The current state is taken from the
        polled worker-details flag.
        """
        snapshot = self._supervisor.latest_snapshot
        # Prefer the pending intent over the (up-to-15-s stale) advisory poll so that a rapid second
        # press reverses the first instead of duplicating it.
        if self._intended_server_maintenance is not None:
            currently_in_maintenance = self._intended_server_maintenance
        else:
            currently_in_maintenance = snapshot is not None and snapshot.worker_details_maintenance
        enable = not currently_in_maintenance
        sent = self._supervisor.request_set_server_maintenance(enable)
        if sent:
            self._intended_server_maintenance = enable
            self._server_maintenance_intent_pop_count = (
                snapshot.num_jobs_popped if enable and snapshot is not None else None
            )
        if not sent:
            self.notify("Worker not running; maintenance change not sent.")
        elif enable:
            self.notify("Requested horde maintenance ON (worker stops receiving jobs).")
        else:
            self.notify("Requested horde maintenance OFF (worker receives jobs again).")

    def action_start_stop_worker(self) -> None:
        """Start the worker if stopped, or gracefully stop it (without quitting) if running."""
        if self._supervisor.status is SupervisorStatus.STOPPED or not self._supervisor.is_alive():
            self.notify("Starting worker…")
            self._supervisor.start()
            return
        already_stopping = self._supervisor.status is SupervisorStatus.STOPPING
        self._supervisor.request_graceful_stop()
        if already_stopping:
            # The request is re-sent, but its force-kill deadline is not extended; say so, because the
            # obvious reading of an unchanged screen is that the press did nothing.
            self.notify("Stop already under way; the worker is still draining in-flight jobs.")
        else:
            self.notify("Stopping worker (in-flight jobs will finish)…")

    def action_toggle_autostart(self) -> None:
        """Flip and persist whether the worker auto-starts on launch."""
        try:
            new_value = not self._app_state_store.load().auto_start_worker
            self._app_state_store.set_auto_start_worker(new_value)
        except Exception as toggle_error:  # noqa: BLE001 - must not crash the TUI
            self.notify(f"Could not update auto-start: {toggle_error}", severity="error")
            return
        self.notify(f"Auto-start on launch is now {'ON' if new_value else 'OFF'}.")

    def action_reload_config(self) -> None:
        """Ask the worker to reload bridgeData.yaml from disk."""
        if self._supervisor.request_reload_config():
            self.notify("Config reload sent to worker.")
        else:
            self.notify("Worker not running; reload not sent.", severity="warning")

    def action_restart_worker(self) -> None:
        """Begin a cooperative worker restart driven by the normal supervisor ticks."""
        self.notify("Restarting worker…")
        self._supervisor.request_restart()

    def on_tabbed_content_tab_activated(self, message: TabbedContent.TabActivated) -> None:
        """Guard against leaving the Config tab with unsaved edits.

        Textual switches the tab before this fires, so when the user navigates off a dirty Config tab we
        revert to it and prompt, then honour their choice. Sub-tab activations (the config/benchmark inner
        TabbedContents) are ignored here; only the top-level ``main-tabs`` is gated.
        """
        if message.tabbed_content.id != "main-tabs" or message.pane is None or message.pane.id is None:
            return
        new_tab = message.pane.id
        if self._allow_tab_switch_to == new_tab:
            self._allow_tab_switch_to = None
            self._last_main_tab = new_tab
            return
        leaving_config = self._last_main_tab == "tab-config" and new_tab != "tab-config"
        if leaving_config and not self._config_leave_warning_suppressed and self._config_is_dirty():
            target = new_tab
            self._allow_tab_switch_to = "tab-config"
            message.tabbed_content.active = "tab-config"
            self._last_main_tab = "tab-config"
            self.push_screen(ConfigLeaveModal(), lambda outcome: self._on_config_leave_choice(outcome, target))
            return
        self._last_main_tab = new_tab

    def _config_is_dirty(self) -> bool:
        """Whether the Config tab has unsaved edits (best-effort; a lookup failure reads as clean)."""
        try:
            return self.query_one(ConfigEditorView).is_dirty()
        except Exception:  # noqa: BLE001 - the guard must never block navigation
            return False

    def _on_config_leave_choice(self, outcome: ConfigLeaveChoice | None, target: str) -> None:
        """Apply the unsaved-edits choice: stay, discard-and-leave, leave, or leave-and-suppress."""
        if outcome is None or outcome is ConfigLeaveChoice.STAY:
            return
        if outcome is ConfigLeaveChoice.DISCARD:
            with contextlib.suppress(NoMatches):
                self.query_one(ConfigEditorView).reload_from_disk()
        elif outcome is ConfigLeaveChoice.NEVER:
            self._config_leave_warning_suppressed = True
        with contextlib.suppress(NoMatches):
            self._allow_tab_switch_to = target
            self.query_one("#main-tabs", TabbedContent).active = target

    def on_config_editor_view_apply_requested(self, message: ConfigEditorView.ApplyRequested) -> None:
        """Restart the worker for a saved change to a restart-locked field.

        Plain saves are not routed here: the worker watches bridgeData.yaml and hot-reloads on its own,
        so only restart-locked fields (⟳) need the app to act.
        """
        if message.restart:
            self.action_restart_worker()

    def on_benchmark_view_run_requested(self, message: BenchmarkView.RunRequested) -> None:
        """Launch the benchmark, first gating on an in-progress download and the GPU takeover of a live worker."""
        if self._benchmark_supervisor.is_active:
            self.notify("A benchmark is already running.", severity="warning")
            return
        if self._benchmark_waiting_incomplete():
            # The benchmark's own models are still downloading; running now fetches them mid-run (slow, skewed).
            self.push_screen(
                BenchmarkActionConfirmModal(
                    title="Benchmark models still downloading",
                    body=(
                        "The benchmark's models are still downloading in the background. Run now anyway? They "
                        "will be fetched mid-run, which slows and skews the measurement. Or cancel and wait for "
                        "the waiting banner to clear, then run."
                    ),
                    confirm_label="Run anyway",
                ),
                partial(self._on_run_while_waiting_choice, message.options),
            )
            return
        self._proceed_with_run_request(message.options)

    def _on_run_while_waiting_choice(self, options: BenchmarkOptions, confirmed: bool | None) -> None:
        """Proceed with the run only if the operator chose to run despite the in-progress download."""
        if confirmed:
            self._clear_benchmark_waiting()  # the operator abandoned the wait to run now
            self._proceed_with_run_request(options)

    def _proceed_with_run_request(self, options: BenchmarkOptions) -> None:
        """Run the benchmark, confirming the GPU takeover first when a worker is alive."""
        if self._supervisor.is_alive():
            # Freeing a live worker's GPU interrupts it; require an explicit yes, and describe the real
            # disruption (a serving worker loses its queue; a held one only yields the idle GPU).
            snapshot = self._supervisor.latest_snapshot
            serving = snapshot is None or not _no_inference_contexts(snapshot)
            self.push_screen(
                BenchmarkOverWorkerModal(serving=serving),
                partial(self._on_benchmark_over_worker_choice, options),
            )
            return
        self._launch_benchmark(options)

    def _on_benchmark_over_worker_choice(self, options: BenchmarkOptions, confirmed: bool | None) -> None:
        """Proceed with the benchmark only when the operator agreed to stop the running worker."""
        if confirmed:
            self._launch_benchmark(options)

    def _launch_benchmark(self, options: BenchmarkOptions) -> None:
        """Stop the worker (freeing the GPU) and launch the benchmark, off the UI thread."""
        # The run is past the download stage: leave the waiting mode so its banner and gate do not linger.
        self._clear_benchmark_waiting()
        self._pending_benchmark_options = options
        # Show the PREPARING state immediately: the stop below blocks for up to ~100s, and without a
        # visible phase on the Benchmark tab that wait is indistinguishable from a hang.
        self._benchmark_supervisor.mark_preparing()
        with contextlib.suppress(NoMatches):
            self.query_one(BenchmarkView).update_view(
                self._benchmark_supervisor.run_state,
                self._benchmark_supervisor.status,
            )
            self.query_one("#main-tabs", TabbedContent).active = "tab-benchmark"
        self.notify("Stopping worker to free the GPU for the benchmark…")
        self.run_worker(self._start_benchmark_flow, thread=True, exclusive=True, group="lifecycle")

    def on_benchmark_view_download_requested(self, message: BenchmarkView.DownloadRequested) -> None:
        """Open the benchmark model-download modal, delegating to a live worker's downloads when one runs.

        Imported lazily to keep the modal's subprocess plumbing off the TUI's hot path. The delegate folds
        the benchmark's download phase into a running worker's single download surface (no second, contending
        downloader); when no worker is live the modal self-downloads out-of-process.
        """
        from horde_worker_regen.tui.widgets.benchmark_download import BenchmarkDownloadModal

        self.push_screen(
            BenchmarkDownloadModal(
                message.options,
                delegate=self._benchmark_download_delegate(),
                live_state=self._benchmark_live_state(),
            ),
            self._after_benchmark_download,
        )

    def _benchmark_download_delegate(self) -> Callable[[list[str]], bool]:
        """Return a delegate that routes the benchmark's missing models through the download orchestration.

        Always available, so the benchmark never runs a second, contending downloader. A live worker
        background-fetches the models into the shared cache while it keeps serving (a download takes no GPU);
        a stopped worker is started into a download-only hold (GPU idle) and the request is sent once its
        control pipe is up. Auxiliary models are included since a benchmark level may exercise
        controlnet/post-processing.
        """

        def _delegate(model_names: list[str]) -> bool:
            if self._supervisor.is_alive():
                if not self._supervisor.request_download_models(model_names, include_aux=True):
                    return False
                self._enter_benchmark_waiting(model_names)
                return True
            # A stopped worker: start it GPU-idle, then send the hold and the request once the pipe is up
            # (see _tick); a cold install has nothing present, so inference will not start in the meantime.
            self._supervisor.start()
            self._pending_downloads_only_hold = True
            self._pending_download_models = DownloadSelection(model_names=list(model_names), include_aux=True)
            self._enter_benchmark_waiting(model_names)
            return True

        return _delegate

    def _enter_benchmark_waiting(self, model_names: list[str]) -> None:
        """Enter the "waiting for benchmark models" mode for a freshly requested download set.

        Records the requested image models (so the run gate and the start/go-live warnings can reckon them
        against the live download state) and arms the "seen active" guard, so the wait does not complete on an
        idle snapshot captured before the worker has begun fetching. The mode engages even when *model_names*
        is empty (a features-only request), since the feature files still download via the aux pass.
        """
        self._benchmark_waiting_active = True
        self._benchmark_waiting_models = set(model_names)
        self._benchmark_download_seen_active = False

    def _clear_benchmark_waiting(self) -> None:
        """Leave the waiting mode and clear its banner (the models are ready, or the wait was abandoned)."""
        if not self._benchmark_waiting_active:
            return
        self._benchmark_waiting_active = False
        self._benchmark_waiting_models = set()
        self._benchmark_download_seen_active = False
        with contextlib.suppress(NoMatches):
            self.query_one(BenchmarkView).set_benchmark_waiting(None)

    def _benchmark_waiting_incomplete(self) -> bool:
        """Whether a benchmark download is still in progress (the waiting mode is active)."""
        return self._benchmark_waiting_active

    def _benchmark_waiting_outside_config(self) -> bool:
        """Whether any waited-for model is NOT in the worker config's would-download set.

        When every benchmark-requested model is already in the configured set, starting/serving the worker
        would download them anyway, so an action that resumes serving need not warn. Only models outside that
        set are genuinely "benchmark-only" downloads an action could strand, which is what the warning guards.
        Fails safe to True (warn) when the configured set cannot be read.
        """
        if not self._benchmark_waiting_models:
            return False
        try:
            manager = self.query_one(ModelManagerView)
        except NoMatches:
            return True
        configured = {model.name for model in manager.configured_included_models()}
        return not self._benchmark_waiting_models <= configured

    def _update_benchmark_waiting(self) -> None:
        """Reflect background benchmark-download progress into the banner, completing the wait when done.

        Completion is judged by the download subsystem rather than per-model presence: the requested set may
        include feature models (controlnet checkpoints, annotators) the worker's present-set never names, so a
        name-by-name wait would stall. The wait ends when the requested image models are all present, or when
        the subsystem, having been seen busy, returns to idle (everything it was asked to fetch is done).
        """
        if not self._benchmark_waiting_active:
            return
        try:
            view = self.query_one(BenchmarkView)
        except NoMatches:
            return
        snapshot = self._supervisor.latest_snapshot
        downloads = snapshot.downloads if snapshot is not None else None
        total = len(self._benchmark_waiting_models)
        if downloads is None:
            view.set_benchmark_waiting(BenchmarkWaitingState(total=total, ready=0))
            return
        if downloads.phase in (DownloadPhase.SCANNING, DownloadPhase.DOWNLOADING, DownloadPhase.PAUSED):
            self._benchmark_download_seen_active = True
        present = set(downloads.present_model_names)
        ready = len(self._benchmark_waiting_models & present)
        # An empty requested set (a features-only request) is never "all present": those files do not appear
        # in the present-set, so completion must come from the subsystem settling, not a vacuous subset check.
        all_present = bool(self._benchmark_waiting_models) and self._benchmark_waiting_models <= present
        subsystem_settled = self._benchmark_download_seen_active and downloads.phase is DownloadPhase.IDLE
        if all_present or subsystem_settled:
            self._clear_benchmark_waiting()
            view.refresh_plan_preview()
            self.notify("Benchmark models ready. You can run the benchmark now.")
            return
        if self._benchmark_download_seen_active and downloads.phase is DownloadPhase.ERROR:
            self._clear_benchmark_waiting()
            self.notify("Some benchmark model downloads failed; see the Downloads tab.", severity="warning")
            return
        view.set_benchmark_waiting(BenchmarkWaitingState(total=total, ready=ready))

    def _benchmark_live_state(self) -> Callable[[], DownloadLiveState | None] | None:
        """A reader of the live worker's present/in-flight model set for the benchmark plan, or None.

        Bound to a *live* worker only: with none running, the plan must fall back to its own disk scan, so a
        stale last-snapshot from a previous run does not masquerade as current truth. Returns a closure read
        lazily on each render, folding the worker's queued, in-flight and current downloads into one
        in-flight set so a model being fetched is never shown as ready nor offered for a redundant fetch.
        """
        from horde_worker_regen.tui.widgets.benchmark_download import DownloadLiveState

        def _read() -> DownloadLiveState | None:
            if not self._supervisor.is_alive():
                return None
            snapshot = self._supervisor.latest_snapshot
            downloads = snapshot.downloads if snapshot is not None else None
            if downloads is None:
                return None
            in_flight = {item.model_name for item in downloads.pending}
            in_flight.update(active.model_name for active in downloads.active)
            if downloads.current is not None:
                in_flight.add(downloads.current.model_name)
            return DownloadLiveState(
                present=frozenset(downloads.present_model_names),
                in_flight=frozenset(in_flight),
            )

        return _read

    def _after_benchmark_download(self, download_requested: bool | None) -> None:
        """Refresh the benchmark plan preview once a download has been requested through the orchestration.

        The fetch itself runs in the background (tracked on the Downloads tab); refreshing here updates the
        plan's live overlay so the requested models read as downloading rather than still-missing.
        """
        if download_requested:
            with contextlib.suppress(NoMatches):
                self.query_one(BenchmarkView).refresh_plan_preview()

    def _start_benchmark_flow(self) -> None:
        """Free the GPU for the benchmark, then start it (runs in a thread).

        A running worker is freed *gracefully* first: drain its queue, let in-flight jobs finish, and scale
        its inference processes to nothing while keeping the worker alive (its downloads keep running). That
        leaves it cheaply resumable with Go live afterwards, instead of a full stop/restart. The hard stop is
        the backstop only when the graceful drain cannot free the GPU within the time budget.
        """
        self._benchmark_drained_worker = False
        if self._supervisor.is_alive():
            self._benchmark_drained_worker = self._drain_worker_for_benchmark()
        if not self._benchmark_drained_worker:
            # No worker was running, or the graceful drain timed out: hard-stop so a wedged job never blocks
            # the run. (stop() on an already-stopped worker is a harmless no-op.)
            self._supervisor.stop()
        self._benchmark_supervisor.start(self._pending_benchmark_options)
        self.call_from_thread(self._after_benchmark_started)

    def _drain_worker_for_benchmark(self) -> bool:
        """Gracefully free the GPU from a live worker without stopping it; True only if the GPU came free.

        Three bounded steps: stop popping and let in-flight inference finish; hold the worker (so the
        scheduler will not re-grow inference) and scale its inference processes to zero; then wait for those
        GPU contexts to actually go away. Any step exceeding its budget returns False so the caller falls back
        to a hard stop: the GPU must be free before the benchmark can use it.
        """
        self._supervisor.request_drain()
        if not self._wait_for_worker(
            lambda snapshot: snapshot.jobs_in_progress == 0 and snapshot.jobs_pending_inference == 0,
            timeout=_BENCHMARK_DRAIN_TIMEOUT_SECONDS,
        ):
            return False
        # The hold keeps inference/popping deferred (so nothing re-grows) while the worker, and its download
        # process, stay alive; scaling to 0 sheds the now-idle inference processes that hold GPU VRAM.
        self._supervisor.request_downloads_only_hold()
        self._supervisor.request_set_concurrency(target_processes=0)
        return self._wait_for_worker(_no_inference_contexts, timeout=_BENCHMARK_SCALE_TIMEOUT_SECONDS)

    def _wait_for_worker(
        self,
        predicate: Callable[[WorkerStateSnapshot], bool],
        *,
        timeout: float,
    ) -> bool:
        """Poll the worker's latest snapshot until *predicate* holds or *timeout* elapses (worker thread).

        Ticks on the UI thread keep draining fresh snapshots while this blocks, so a simple poll observes the
        worker's progress. Returns False on timeout (the caller treats that as "could not free the GPU").
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = self._supervisor.latest_snapshot
            if snapshot is not None and predicate(snapshot):
                return True
            if not self._supervisor.is_alive():
                # The worker exited from under us (crash/operator stop): nothing to drain, let the caller stop.
                return False
            time.sleep(_BENCHMARK_DRAIN_POLL_SECONDS)
        return False

    def _resume_hint(self) -> str:
        """A trailing ' Press Go live...' hint when the worker was left held by the graceful drain, else ''.

        Keeps the post-run messaging honest: a drained worker is alive but not serving, so the operator is
        told how to resume rather than being left to wonder why it sits idle.
        """
        if self._benchmark_drained_worker and self._supervisor.is_alive():
            return " The worker is held; press Go live to resume serving."
        return ""

    def _after_benchmark_started(self) -> None:
        """Focus the Benchmark tab once the run is launched (UI thread)."""
        with contextlib.suppress(NoMatches):
            self.query_one("#main-tabs", TabbedContent).active = "tab-benchmark"
        if self._benchmark_drained_worker:
            self.notify("Benchmark started; the worker is held (GPU freed) and resumes with Go live when it finishes.")
        else:
            self.notify("Benchmark started; the worker is stopped until it completes.")

    def on_benchmark_view_cancel_requested(self, message: BenchmarkView.CancelRequested) -> None:
        """Cancel the running benchmark, off the UI thread."""
        self.run_worker(self._cancel_benchmark, thread=True, exclusive=True, group="lifecycle")

    def _cancel_benchmark(self) -> None:
        """Terminate the benchmark subprocess (runs in a thread)."""
        self._benchmark_supervisor.cancel()
        self.call_from_thread(self.notify, "Benchmark cancelled.")

    def on_benchmark_view_apply_config_requested(self, message: BenchmarkView.ApplyConfigRequested) -> None:
        """Write the benchmark's suggested bridgeData to disk and restart the worker to use it."""
        report = self._benchmark_supervisor.report
        if report is None:
            self.notify("No benchmark result to apply.", severity="warning")
            return
        try:
            apply_suggested_to_config(report.suggested_bridge_data, self._config_path)
        except OSError as write_error:
            self.notify(f"Failed to write {self._config_path}: {write_error}", severity="error")
            return
        record_suggested_as_known_good(report.suggested_bridge_data, worker_version=__version__)
        with contextlib.suppress(NoMatches):
            self.query_one(BenchmarkView).refresh_app_state_summary()
        self.notify("Applied suggested config to bridgeData.yaml. Restarting worker…")
        self.action_restart_worker()

    def on_benchmark_view_restore_known_good_requested(self, message: BenchmarkView.RestoreKnownGoodRequested) -> None:
        """Write the last benchmark/clean-run known-good config back to disk and restart the worker."""
        try:
            known_good = self._app_state_store.load().last_known_good_settings
        except Exception as load_error:  # noqa: BLE001 - reading app state must not crash the TUI
            self.notify(f"Could not read known-good settings: {load_error}", severity="error")
            return
        if known_good is None:
            self.notify("No known-good settings on record.", severity="warning")
            return
        try:
            apply_known_good_to_config(known_good.config_snapshot, self._config_path)
        except OSError as write_error:
            self.notify(f"Failed to write {self._config_path}: {write_error}", severity="error")
            return
        self.notify(f"Restored last known-good config ({known_good.source.value}). Restarting worker…")
        self.action_restart_worker()

    async def action_quit(self) -> None:
        """Stop the worker (off the UI thread) and exit.

        When running as a browser session on Windows (attached to a worker host), the worker
        survives this close. A warning modal explains this and offers the user a way back.

        The first Ctrl+Q/Ctrl+C starts a graceful shutdown (drain in-flight jobs, submit results,
        then exit). Pressing it again while that graceful stop is underway escalates to an
        immediate force-kill of the worker process tree, so the TUI is never stuck waiting on a
        worker whose control loop is frozen (UNRESPONSIVE).
        """
        if sys.platform == "win32" and isinstance(self._supervisor, AttachedWorkerSupervisor):
            self.push_screen(WebQuitWarningModal(), self._on_web_quit_choice)
            return
        if self._graceful_quit_in_progress:
            # The operator pressed quit again while a graceful stop is already in progress.
            # Escalate: force-kill the worker immediately and exit.
            self.notify("Force-stopping worker (repeated quit)…")
            self._supervisor.force_kill()
            self.exit()
            return
        self._graceful_quit_in_progress = True
        self._do_quit()

    def _on_web_quit_choice(self, confirmed: bool | None) -> None:
        """Proceed with quitting only when the user confirmed the web-session close warning.

        ``None`` (the modal dismissed without a choice, e.g. Escape) is treated as "do not quit".
        """
        if confirmed:
            self._graceful_quit_in_progress = True
            self._do_quit()

    def _do_quit(self) -> None:
        """Begin stop-and-exit without blocking or concurrently mutating the supervisor."""
        self.notify("Stopping worker…")
        self._benchmark_supervisor.stop()
        if isinstance(self._supervisor, AttachedWorkerSupervisor):
            # The host owns the worker; closing this browser/session only detaches it.
            self._supervisor.close()
            self.exit()
            return
        self._supervisor.request_graceful_stop()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the TUI command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="horde-worker",
        description="Textual frontend that launches and supervises the AI Horde reGen worker.",
    )
    parser.add_argument(
        "--process-mode",
        choices=[mode.value for mode in WorkerProcessMode],
        default=WorkerProcessMode.REAL.value,
        help="'real' runs the GPU worker; 'fake' runs a synthetic worker for UI demos/tests.",
    )
    parser.add_argument(
        "-e",
        "--load-config-from-env-vars",
        action="store_true",
        help="Load worker config from AIWORKER_* environment variables instead of bridgeData.yaml.",
    )
    parser.add_argument("--amd", "--amd-gpu", action="store_true", help="Enable AMD GPU optimisations.")
    parser.add_argument("-n", "--worker-name", type=str, default=None, help="Override the worker name.")
    parser.add_argument("--directml", type=int, default=None, help="Enable directml on the given device index.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the bridgeData.yaml the config editor reads/writes.",
    )
    parser.add_argument("--no-auto-restart", action="store_true", help="Do not relaunch the worker if it crashes.")
    parser.add_argument(
        "--attach",
        type=str,
        nargs="?",
        default=None,
        const=f"{sp.DEFAULT_HOST_ADDRESS}:{sp.DEFAULT_HOST_PORT}",
        help="Attach to a running worker host instead of owning the worker (used by the web launcher, and "
        f"to reattach a terminal dashboard). With no value, attaches to {sp.DEFAULT_HOST_ADDRESS}:"
        f"{sp.DEFAULT_HOST_PORT}; pass host:port to target another. The worker survives this session "
        "closing.",
    )
    parser.add_argument(
        "--remote-exposed",
        action="store_true",
        help="Treat this session as reachable from other machines (set by the web launcher when it binds "
        "an address other than loopback). Withholds the credential fields from the config editor.",
    )
    return parser.parse_args(argv)


def _build_supervisor(args: argparse.Namespace) -> SupervisorLike:
    """Build either an owning supervisor or, when ``--attach`` is set, an attach client."""
    mode = WorkerProcessMode(args.process_mode)
    if args.attach:
        return AttachedWorkerSupervisor(sp.resolve_attach_address(args.attach), mode=mode)
    options = WorkerLaunchOptions(
        load_config_from_env_vars=args.load_config_from_env_vars,
        amd=args.amd,
        worker_name=args.worker_name,
        directml=args.directml,
    )
    return WorkerSupervisor(options, mode=mode, auto_restart=not args.no_auto_restart)


def main(argv: list[str] | None = None) -> None:
    """Console-script entry point (``horde-worker``): build the supervisor and run the TUI."""
    multiprocessing.freeze_support()
    args = _parse_args(argv)

    # Give the supervisor process its own on-disk log before the worker is launched, so worker
    # launch/restart/crash diagnostics survive even when no worker runs (the worker writes its own
    # bridge.log, but only once it starts). quiet_console: this is a full-screen Textual app, so the
    # default stderr sink would corrupt the display.
    setup_supervisor_file_logging("tui", quiet_console=True)

    # Record an unhandled crash of the TUI process itself to bridge_tui.log. Textual lets such an
    # exception propagate out here and its traceback would otherwise reach only stderr, which a
    # double-click launch or the alternate-screen buffer discards, leaving no on-disk trace.
    try:
        _run_app(args)
    except Exception:
        logger.exception("The worker TUI exited with an unhandled exception.")
        raise


def _run_app(args: argparse.Namespace) -> None:
    """Build the supervisor and run the Textual app (the body of :func:`main`, wrapped for logging)."""
    supervisor = _build_supervisor(args)

    from multiprocessing import resource_tracker

    # While the Textual app is running, it replaces sys.stdout / sys.stderr with its own capture/redirect objects so
    # library writes don't corrupt the rendered screen. Those replacement stream objects return -1 (or otherwise
    # don't map to a real OS fd) from .fileno() rather than raising, so the except Exception guard doesn't catch
    # it. The -1 sails through into fork_exec, which rejects it and the app crashes on any attempt to spawn a process
    # (e.g. the worker or benchmark subprocesses). By calling ensure_running() here, the resource tracker starts with
    # the original sys.stdout/sys.stderr and their real file descriptors. This eager start is not sufficient on its
    # own: if the tracker later dies, ensure_running() relaunches it under the redirected streams. The actual
    # guarantee is WorkerSupervisor._spawn restoring the real streams around every spawn; see
    # worker_launcher._real_std_streams_for_spawn.

    # Only works on Linux, so let's make sure this is a linux system
    if sys.platform.startswith("linux"):
        resource_tracker.ensure_running()

    app = HordeWorkerTUI(
        supervisor,
        config_path=args.config,
        load_config_from_env_vars=args.load_config_from_env_vars,
        remote_exposed=args.remote_exposed,
    )
    try:
        app.run()
    finally:
        # Safety net on unexpected exit. close() stops a locally-owned worker, but only detaches when
        # attached to a host, so an attached session never kills the shared worker.
        supervisor.close()


if __name__ == "__main__":
    main()
