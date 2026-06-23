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
from pathlib import Path

from loguru import logger
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Static, TabbedContent, TabPane

from horde_worker_regen import __version__
from horde_worker_regen.app_state import (
    AppStateStore,
    OnboardingChoice,
    OverviewViewMode,
    benchmark_status_summary,
    should_prompt_onboarding,
)
from horde_worker_regen.process_management.supervisor_channel import WorkerStateSnapshot
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
from horde_worker_regen.tui.health import HealthReport, HealthStatus, build_offline_checks, derive
from horde_worker_regen.tui.logging_setup import setup_supervisor_file_logging
from horde_worker_regen.tui.update_check import check_for_update
from horde_worker_regen.tui.widgets.benchmark import BenchmarkView
from horde_worker_regen.tui.widgets.config_editor import ConfigEditorView, ConfigLeaveChoice, ConfigLeaveModal
from horde_worker_regen.tui.widgets.download_picker import (
    DownloadPickerModal,
    DownloadPickerRow,
    DownloadSelection,
)
from horde_worker_regen.tui.widgets.downloads import DownloadsView
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
from horde_worker_regen.tui.wizard import SetupWizardModal, WizardOutcome, is_setup_incomplete
from horde_worker_regen.tui.worker_launcher import SupervisorStatus, WorkerProcessMode, WorkerSupervisor


class WebQuitWarningModal(ModalScreen[bool]):
    """Warn the user that closing this browser tab leaves the worker running in the background."""

    DEFAULT_CSS = """
    WebQuitWarningModal {
        align: center middle;
    }
    WebQuitWarningModal #web-quit-dialog {
        width: 72;
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


class HordeWorkerTUI(App[None]):
    """A Textual dashboard that owns and visualises the reGen worker."""

    TITLE = "AI Horde Worker"

    HORIZONTAL_BREAKPOINTS = [(0, "-narrow"), (100, "-normal"), (150, "-wide")]
    """Width bands Textual stamps onto the Screen as classes, mirroring the table column tiers.

    These drive only *layout* rules in the CSS below (e.g. reclaiming side padding on a cramped terminal).
    Panel show/hide stays in Python because it depends on the F6 view intent, which CSS cannot see; and an
    inline ``display`` set per tick from Python would in any case win over a CSS ``display`` rule. The
    within-table column shedding that actually fixes the wide tables is done in ``responsive.py``.
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
    OverviewView, LiveView, InsightsView, ConfigEditorView, LogsView, BenchmarkView, DownloadsView {
        height: 1fr;
        padding: 1 1;
    }
    /* On a cramped terminal, drop the horizontal padding so the tables get those columns back. */
    Screen.-narrow OverviewView,
    Screen.-narrow LiveView,
    Screen.-narrow InsightsView,
    Screen.-narrow ConfigEditorView,
    Screen.-narrow LogsView,
    Screen.-narrow BenchmarkView,
    Screen.-narrow DownloadsView {
        padding: 1 0;
    }
    #overview-worker, #overview-processes {
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("f2", "toggle_pause", "Pause/Resume"),
        ("f3", "start_stop_worker", "Start/Stop"),
        ("f4", "toggle_autostart", "Auto-start"),
        ("f5", "reload_config", "Reload config"),
        ("f6", "cycle_view_mode", "View mode"),
        ("f7", "toggle_download_pause", "Pause downloads"),
        ("f8", "show_benchmark", "Benchmark"),
        ("f9", "restart_worker", "Restart worker"),
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
    ) -> None:
        """Store the (unstarted) supervisor, config path, and durable state store."""
        super().__init__()
        self._supervisor = supervisor
        self._benchmark_supervisor = BenchmarkSupervisor(config_path=config_path)
        self._config_path = config_path
        self._app_state_store = app_state_store if app_state_store is not None else AppStateStore()
        self._load_config_from_env_vars = load_config_from_env_vars
        self._frame = 0
        self._last_benchmark_status = BenchmarkSupervisorStatus.IDLE
        self._pending_benchmark_options = BenchmarkOptions()
        # Set when "Download only" starts a stopped worker: the hold command is sent once its pipe is up
        # (see _tick), since send_command would otherwise race the child's connection.
        self._pending_downloads_only_hold = False
        # A picker selection chosen before a freshly-started worker's pipe is up; sent once it is (see
        # _tick), after the hold command, so the models are fetched without the GPU committing.
        self._pending_download_models: DownloadSelection | None = None
        self._view_mode = self._app_state_store.load().overview_view_mode
        self._last_main_tab = "tab-overview"
        self._allow_tab_switch_to: str | None = None
        self._config_leave_warning_suppressed = False

    def compose(self) -> ComposeResult:
        """Lay out the header, status bar, tabbed views, and footer."""
        yield Header(show_clock=True)
        yield Static(id="status-bar")
        with TabbedContent(initial="tab-overview", id="main-tabs"):
            with TabPane("Overview", id="tab-overview"):
                yield OverviewView()
            with TabPane("Live", id="tab-live"):
                yield LiveView()
            with TabPane("Downloads", id="tab-downloads"):
                yield DownloadsView()
            with TabPane("Logs", id="tab-logs"):
                yield LogsView()
            with TabPane("Config", id="tab-config"):
                yield ConfigEditorView(self._config_path)
            with TabPane("Insights", id="tab-insights"):
                yield InsightsView()
            with TabPane("Benchmark", id="tab-benchmark"):
                yield BenchmarkView(worker_mode=self._supervisor.mode.value)
        yield Footer()

    def on_mount(self) -> None:
        """Begin the refresh loop, then run first-run setup or the usual start/onboarding prompts."""
        self.set_interval(0.1, self._tick)
        # Resolve the models volume from config before any disk figures are computed, so free space and
        # on-disk checks match the worker's configured cache_home instead of defaulting to ./models.
        with contextlib.suppress(Exception):
            apply_cache_home_env(self._config_path)
        # Mirror the worker's default beta opt-in into this process before the catalog warms, so the model
        # picker surfaces pending-queue (beta) models like qwen instead of the canonical-only set.
        with contextlib.suppress(Exception):
            apply_beta_model_env(self._config_path)
        self._maybe_check_for_updates()
        self._warm_model_catalog()
        if self._should_run_setup_wizard():
            self._run_setup_wizard()
        elif self._should_auto_start():
            self._supervisor.start()
            self._maybe_prompt_onboarding()
        else:
            self._prompt_worker_start()

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

    async def _update_check(self) -> None:
        """Notify (non-blocking) when a newer release is available and how to get it."""
        info = await asyncio.to_thread(check_for_update)
        if info is None:
            return
        self.notify(
            f"Update available: v{runtime_version()} -> v{info.latest_version}. Update with "
            "'update.cmd'/'update.sh', or by re-running the installer.",
            title="Update available",
            timeout=10,
        )

    def _should_run_setup_wizard(self) -> bool:
        """Whether to show the guided wizard: a real worker whose bridgeData is not yet configured.

        Skipped for the fake/demo worker and for env-var config (both are power-user paths). When the
        config is already complete, the durable setup-complete flag is set so existing installs never
        see the wizard.
        """
        if self._supervisor.mode is not WorkerProcessMode.REAL or self._load_config_from_env_vars:
            return False
        try:
            if not is_setup_incomplete(self._config_path):
                if not self._app_state_store.load().setup_complete:
                    with contextlib.suppress(Exception):
                        self._app_state_store.set_setup_complete(True)
                return False
        except Exception as wizard_error:  # noqa: BLE001 - detection must never block the TUI
            self.log(f"Could not determine setup state: {wizard_error}")
            return False
        return True

    def _run_setup_wizard(self) -> None:
        """Show the first-run wizard; fall back to the usual start prompt if it cannot be shown."""
        try:
            self.push_screen(SetupWizardModal(config_path=self._config_path), self._on_wizard_outcome)
        except Exception as wizard_error:  # noqa: BLE001 - the wizard must never block the TUI
            self.log(f"Could not show the setup wizard: {wizard_error}")
            self._prompt_worker_start()

    def _on_wizard_outcome(self, outcome: WizardOutcome | None) -> None:
        """Act on the wizard's result: start, benchmark, or stay stopped (None means cancelled)."""
        if outcome is None:
            self.notify("Setup cancelled. Edit it on the Config tab, then press F3 to start.")
            return
        with contextlib.suppress(Exception):
            self._app_state_store.set_setup_complete(True)
        with contextlib.suppress(NoMatches):
            self.query_one(ConfigEditorView).reload_from_disk()
        if outcome is WizardOutcome.BENCHMARK:
            self._pending_benchmark_options = BenchmarkOptions(process_mode=self._supervisor.mode.value)
            self.notify("Saved. Starting a benchmark to tune your settings…")
            self.run_worker(self._start_benchmark_flow, thread=True, exclusive=True, group="lifecycle")
        elif outcome is WizardOutcome.START:
            self.notify(
                "Saved. Starting the worker. Your selected models download now if they are not already "
                "on disk; on first run this can take 30-60 minutes. The worker serves models as they "
                "finish, so keep this window open.",
                title="Downloading models",
                severity="warning",
                timeout=12,
            )
            with contextlib.suppress(NoMatches):
                self.query_one("#main-tabs", TabbedContent).active = "tab-downloads"
            self._supervisor.start()
        else:
            self.notify("Setup saved. Press F3 to start the worker when you're ready.")

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
        now = time.time()
        snapshot_age = (now - snapshot.timestamp) if snapshot is not None else None
        # Judge responsiveness on liveness (the loop's last tick), not on full-snapshot freshness:
        # a coalesced or briefly-failing snapshot build must not read as "unresponsive". Fall back to
        # snapshot age for an older worker that never sends liveness frames.
        liveness_wall_time = self._supervisor.last_liveness_wall_time
        liveness_age = (now - liveness_wall_time) if liveness_wall_time is not None else snapshot_age
        offline_checks = build_offline_checks(self._config_path) if snapshot is None else None
        report = derive(snapshot, self._supervisor.status, liveness_age, offline_checks=offline_checks)
        try:
            self._update_status_bar(report, snapshot)
            self.query_one(OverviewView).update_view(
                report,
                snapshot,
                frame=self._frame,
                mode=self._view_mode,
            )
            self.query_one(DownloadsView).update_view(snapshot, mode=self._view_mode)
            self._update_downloads_tab_label(snapshot)
            self.query_one(LogsView).set_view_mode(self._view_mode)
            config_editor = self.query_one(ConfigEditorView)
            config_editor.set_view_mode(self._view_mode)
            config_editor.update_worker_models(
                snapshot.active_models if snapshot is not None else [],
            )
            if snapshot is not None:
                self.query_one(LiveView).update_snapshot(
                    snapshot,
                    snapshot_age,
                    detailed=self._view_mode is OverviewViewMode.DETAILS,
                )
                self.query_one(InsightsView).update_snapshot(snapshot)
            self.query_one(BenchmarkView).update_view(
                self._benchmark_supervisor.run_state,
                self._benchmark_supervisor.status,
                frame=self._frame,
                mode=self._view_mode,
            )
        except NoMatches:
            # The refresh interval can fire during mount or teardown; skip until the DOM is ready.
            pass
        self._handle_benchmark_status_transition()

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
            self.notify("Benchmark finished. Apply the suggested config, or press F9 to restart the worker.")
        elif status is BenchmarkSupervisorStatus.FAILED:
            self.notify("Benchmark failed; see the run's console.log.", severity="error")
        elif status is BenchmarkSupervisorStatus.CANCELLED:
            self.notify("Benchmark cancelled.")
        if status in (BenchmarkSupervisorStatus.FINISHED, BenchmarkSupervisorStatus.FAILED):
            with contextlib.suppress(NoMatches):
                self.query_one(BenchmarkView).refresh_app_state_summary()

    def action_show_benchmark(self) -> None:
        """Switch to the Benchmark tab."""
        with contextlib.suppress(NoMatches):
            self.query_one("#main-tabs", TabbedContent).active = "tab-benchmark"

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
        """Leave download-only mode so the worker starts serving jobs."""
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

    def _update_status_bar(self, report: HealthReport, snapshot: WorkerStateSnapshot | None) -> None:
        """Render the top status bar, led by the worker's current lifecycle phase."""
        badge = f"[black on {self._badge_colour(report.severity)}] {report.phase.value.upper()} [/]"
        parts = [badge, f"[grey62]mode[/] {self._supervisor.mode.value}"]
        if self._supervisor.restart_attempts:
            parts.append(f"[yellow]restarts {self._supervisor.restart_attempts}[/]")
        if snapshot is not None:
            kudos = "-" if snapshot.kudos_per_hour is None else f"{snapshot.kudos_per_hour:,.0f}"
            parts.append(f"[grey62]worker[/] {snapshot.config.dreamer_name}")
            parts.append(f"[grey62]submitted[/] {snapshot.num_jobs_submitted}")
            parts.append(f"[grey62]faulted[/] {snapshot.num_jobs_faulted}")
            parts.append(f"[grey62]kudos/hr[/] {kudos}")
        self.query_one("#status-bar", Static).update(Text.from_markup("   ".join(parts)))

    @staticmethod
    def _badge_colour(severity: HealthStatus) -> str:
        """Background colour for the status-bar phase badge."""
        return {
            HealthStatus.OK: "green",
            HealthStatus.INFO: "grey50",
            HealthStatus.WARN: "yellow",
            HealthStatus.ERROR: "red",
        }[severity]

    _VIEW_MODE_CYCLE = (OverviewViewMode.NORMAL, OverviewViewMode.DETAILS, OverviewViewMode.THIN)
    """The order F6 steps through: the lean redesign, the verbose detail view, then the thin bar."""

    _VIEW_MODE_NOTICE = {
        OverviewViewMode.NORMAL: "View: normal (the everyday density, across all tabs).",
        OverviewViewMode.DETAILS: "View: details (every diagnostic: extra columns, log tally, all config sub-tabs).",
        OverviewViewMode.THIN: "View: thin (essentials only: status bar, slim downloads, bare log, Essentials).",
    }

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

    def action_toggle_pause(self) -> None:
        """Pause or resume the worker (a *local* pop-pause) depending on its current state.

        This is the local pause: in-flight jobs finish, no new ones are popped. It does not by itself
        change the worker's server-side maintenance on the horde; the worker clears that on resume only
        when its ``remove_maintenance_on_init`` config is set. Use the Maintenance (horde) key for an
        explicit server-side toggle.
        """
        snapshot = self._supervisor.latest_snapshot
        if snapshot is not None and snapshot.maintenance_mode:
            self._supervisor.request_resume()
            self.notify("Resume requested.")
        else:
            self._supervisor.request_pause()
            self.notify("Pause requested (in-flight jobs will finish).")

    def action_toggle_server_maintenance(self) -> None:
        """Toggle the worker's server-side (horde) maintenance flag via the horde API.

        Distinct from F2 (local pause): this asks the horde itself to stop (or resume) sending the worker
        jobs, matching the maintenance the job-pop response reports. The current state is taken from the
        polled worker-details flag.
        """
        snapshot = self._supervisor.latest_snapshot
        currently_in_maintenance = snapshot is not None and snapshot.worker_details_maintenance
        enable = not currently_in_maintenance
        sent = self._supervisor.request_set_server_maintenance(enable)
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
        else:
            self.notify("Stopping worker (in-flight jobs will finish)…")
            self.run_worker(self._stop_worker_only, thread=True, exclusive=True, group="lifecycle")

    def _stop_worker_only(self) -> None:
        """Gracefully stop the worker without exiting the app (runs in a thread)."""
        self._supervisor.stop()

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
        """Restart the worker process (off the UI thread)."""
        self.notify("Restarting worker…")
        self.run_worker(self._restart_worker, thread=True, exclusive=True, group="lifecycle")

    def _restart_worker(self) -> None:
        """Restart the worker (runs in a thread).

        Delegated to the supervisor as a single intent so that, when attached to a host, the stop and the
        subsequent start are not raced by the non-blocking shutdown (the host completes the stop before
        starting again).
        """
        self._supervisor.restart()

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
        """Stop the worker (freeing the GPU) and launch the benchmark, off the UI thread."""
        if self._benchmark_supervisor.is_active:
            self.notify("A benchmark is already running.", severity="warning")
            return
        self._pending_benchmark_options = message.options
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

    def _start_benchmark_flow(self) -> None:
        """Stop the worker, then start the benchmark subprocess (runs in a thread)."""
        self._supervisor.stop()
        self._benchmark_supervisor.start(self._pending_benchmark_options)
        self.call_from_thread(self._after_benchmark_started)

    def _after_benchmark_started(self) -> None:
        """Focus the Benchmark tab once the run is launched (UI thread)."""
        with contextlib.suppress(NoMatches):
            self.query_one("#main-tabs", TabbedContent).active = "tab-benchmark"
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
        """
        if sys.platform == "win32" and isinstance(self._supervisor, AttachedWorkerSupervisor):
            self.push_screen(WebQuitWarningModal(), self._on_web_quit_choice)
            return
        self._do_quit()

    def _on_web_quit_choice(self, confirmed: bool) -> None:
        """Proceed with quitting only when the user confirmed the web-session close warning."""
        if confirmed:
            self._do_quit()

    def _do_quit(self) -> None:
        """Kick off the stop-and-exit worker (common path for both the direct and confirmed quit)."""
        self.notify("Stopping worker…")
        self.run_worker(self._stop_and_exit, thread=True, exclusive=True, group="lifecycle")

    def _stop_and_exit(self) -> None:
        """Close the worker connection and any benchmark, then exit (runs in a thread).

        Uses ``close()`` rather than ``stop()``: when owning the worker locally this stops it, but when
        attached to a host (browser mode) it only detaches, so closing the dashboard leaves the worker
        running.
        """
        self._benchmark_supervisor.stop()
        self._supervisor.close()
        self.call_from_thread(self.exit)


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
    )
    try:
        app.run()
    finally:
        # Safety net on unexpected exit. close() stops a locally-owned worker, but only detaches when
        # attached to a host, so an attached session never kills the shared worker.
        supervisor.close()


if __name__ == "__main__":
    main()
