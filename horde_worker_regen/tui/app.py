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
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

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
from horde_worker_regen.tui.health import HealthReport, HealthStatus, WorkerPhase, build_offline_checks, derive
from horde_worker_regen.tui.logging_setup import setup_supervisor_file_logging
from horde_worker_regen.tui.update_check import check_for_update
from horde_worker_regen.tui.widgets.benchmark import BenchmarkView, BenchmarkWaitingState
from horde_worker_regen.tui.widgets.config_editor import ConfigEditorView, ConfigLeaveChoice, ConfigLeaveModal
from horde_worker_regen.tui.widgets.diagnostics import DiagnosticsView
from horde_worker_regen.tui.widgets.download_picker import (
    DownloadPickerModal,
    DownloadPickerRow,
    DownloadSelection,
)
from horde_worker_regen.tui.widgets.downloads import DownloadsView
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
from horde_worker_regen.tui.wizard import SetupWizardModal, WizardOutcome, is_setup_incomplete
from horde_worker_regen.tui.worker_launcher import SupervisorStatus, WorkerProcessMode, WorkerSupervisor

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


def _no_inference_contexts(snapshot: WorkerStateSnapshot) -> bool:
    """Whether no inference process is alive (so its GPU VRAM is released and the benchmark can take the card)."""
    return not any(process.process_type == "INFERENCE" and process.is_alive for process in snapshot.processes)


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


class BenchmarkOverWorkerModal(ModalScreen[bool]):
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


class BenchmarkActionConfirmModal(ModalScreen[bool]):
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
    OverviewView, GpusView, LiveView, InsightsView, ConfigEditorView, LogsView, BenchmarkView, DownloadsView {
        height: 1fr;
        padding: 1 1;
    }
    /* On a cramped terminal, drop the horizontal padding so the tables get those columns back. */
    Screen.-narrow OverviewView,
    Screen.-narrow GpusView,
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
        ("f11", "restart_worker", "Restart worker"),
        ("f10", "show_diagnostics", "Diagnostics"),
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
        # True when the benchmark freed the GPU by gracefully draining a live worker (which stays alive, held)
        # rather than hard-stopping it. Drives the post-run guidance: "Go live" to resume vs "restart".
        self._benchmark_drained_worker = False
        # True while a benchmark download is in progress: the "waiting for benchmark models" mode that shows
        # the banner, gates Run, and warns before actions that would interrupt the fetch. Tracked as a flag
        # (not just a non-empty set) so a features-only request -- whose image-model set is empty because the
        # controlnet/post-proc checkpoints fetch via the aux pass -- still engages the mode.
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
        self._view_mode = self._app_state_store.load().overview_view_mode
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

    def compose(self) -> ComposeResult:
        """Lay out the header, status bar, tabbed views, and footer."""
        yield Header(show_clock=True)
        yield Static(id="status-bar")
        with TabbedContent(initial="tab-overview", id="main-tabs"):
            with TabPane("Overview", id="tab-overview"):
                yield OverviewView()
            with TabPane("GPUs", id="tab-gpus"):
                yield GpusView()
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
            with TabPane("Diagnostics", id="tab-diagnostics"):
                yield DiagnosticsView()
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
        )
        try:
            self._update_status_bar(report, snapshot)
            self.query_one(OverviewView).update_view(
                report,
                snapshot,
                frame=self._frame,
                mode=self._view_mode,
            )
            self.query_one(GpusView).update_view(snapshot, mode=self._view_mode)
            self.query_one(DownloadsView).update_view(snapshot, mode=self._view_mode)
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
                        "progress -- they are not in this worker's config, so serving will not re-fetch them. "
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

    def _update_status_bar(self, report: HealthReport, snapshot: WorkerStateSnapshot | None) -> None:
        """Render the top status bar, led by the worker's current lifecycle phase."""
        phase_text = report.phase.value.upper()
        if report.phase is WorkerPhase.MAINTENANCE:
            source = self._paused_source(snapshot) or "server"
            phase_text = f"MAINT·{source}"
        elif report.phase is WorkerPhase.PAUSED:
            source = self._paused_source(snapshot)
            if source:
                phase_text = f"PAUSED·{source}"
        badge = f"[black on {self._badge_colour(report.severity)}] {phase_text} [/]"
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
        # Read supervisor_paused directly: it is the flag F2 controls. Using the aggregate
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

        Distinct from F2 (local pause): this asks the horde itself to stop (or resume) sending the worker
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
        the subsystem -- having been seen busy -- returns to idle (everything it was asked to fetch is done).
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
        # An empty requested set (a features-only request) is never "all present" -- those files do not appear
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
        to a hard stop -- the GPU must be free before the benchmark can use it.
        """
        self._supervisor.request_drain()
        if not self._wait_for_worker(
            lambda snapshot: snapshot.jobs_in_progress == 0 and snapshot.jobs_pending_inference == 0,
            timeout=_BENCHMARK_DRAIN_TIMEOUT_SECONDS,
        ):
            return False
        # The hold keeps inference/popping deferred (so nothing re-grows) while the worker -- and its download
        # process -- stay alive; scaling to 0 sheds the now-idle inference processes that hold GPU VRAM.
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
        """
        if sys.platform == "win32" and isinstance(self._supervisor, AttachedWorkerSupervisor):
            self.push_screen(WebQuitWarningModal(), self._on_web_quit_choice)
            return
        self._do_quit()

    def _on_web_quit_choice(self, confirmed: bool | None) -> None:
        """Proceed with quitting only when the user confirmed the web-session close warning.

        ``None`` (the modal dismissed without a choice, e.g. Escape) is treated as "do not quit".
        """
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
