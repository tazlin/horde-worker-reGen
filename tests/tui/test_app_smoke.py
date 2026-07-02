"""Headless end-to-end smoke test for the TUI app against the fake worker."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from textual.widgets import Static, TabbedContent

from horde_worker_regen.app_state import AppStateStore, OnboardingChoice, OverviewTrendWindow, OverviewViewMode
from horde_worker_regen.process_management.ipc.supervisor_channel import WorkerConfigSummary, WorkerStateSnapshot
from horde_worker_regen.run_worker import WorkerLaunchOptions
from horde_worker_regen.tui.app import HordeWorkerTUI
from horde_worker_regen.tui.health import WorkerPhase, derive
from horde_worker_regen.tui.widgets.overview import OverviewView
from horde_worker_regen.tui.wizard import WizardOutcome
from horde_worker_regen.tui.worker_launcher import WorkerProcessMode, WorkerSupervisor
from tests.tui._fake_supervisor import FakeSupervisor

# Headless run_test throttles timers, so the test drives the app's tick directly.
_LIVE_PHASES = {
    WorkerPhase.WARMING_UP,
    WorkerPhase.SERVING,
    WorkerPhase.READY,
    WorkerPhase.IDLE,
    WorkerPhase.MAINTENANCE,
    WorkerPhase.PAUSED,
    WorkerPhase.DISCONNECTED,
}


@pytest.mark.e2e
async def test_app_boots_renders_and_cycles_tabs(tmp_path: Path) -> None:
    """The app boots the fake worker, renders the status hero, and cycles all tabs without error."""
    store = AppStateStore(tmp_path / ".horde_worker_regen" / "state.json")
    store.set_auto_start_worker(True)  # opt in so the worker auto-starts instead of prompting
    supervisor = WorkerSupervisor(WorkerLaunchOptions(worker_name="SmokeApp"), mode=WorkerProcessMode.FAKE)
    app = HordeWorkerTUI(supervisor, config_path=Path("bridgeData.yaml"), app_state_store=store)
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(30):
                app._tick()
                await pilot.pause()
                await asyncio.sleep(0.12)

            # The 30 ticks above exercised update_view (hero + health checklist + cards) without raising.
            assert supervisor.latest_snapshot is not None, "no snapshot reached the app"
            assert supervisor.latest_snapshot.processes, "no processes in snapshot"

            for tab in (
                "tab-control",
                "tab-live",
                "tab-downloads",
                "tab-logs",
                "tab-config",
                "tab-insights",
                "tab-overview",
            ):
                app.query_one("#main-tabs", TabbedContent).active = tab
                app._tick()
                await pilot.pause()
                await asyncio.sleep(0.1)

            snapshot = supervisor.latest_snapshot
            report = derive(snapshot, supervisor.status, time.time() - snapshot.timestamp)
            assert report.phase in _LIVE_PHASES, f"unexpected phase {report.phase}"

            # The F6 view-mode toggle cycles normal -> details -> thin, renders each without error,
            # and persists the active mode.
            assert app._view_mode is OverviewViewMode.NORMAL
            app.action_cycle_view_mode()
            await pilot.pause()
            assert app._view_mode is OverviewViewMode.DETAILS
            # At 120 cols the overview lays out two-up (columns are width-driven now, in every density mode).
            assert app.query_one(OverviewView).has_class("-cols-2")
            assert store.load().overview_view_mode is OverviewViewMode.DETAILS
            app.action_cycle_view_mode()
            await pilot.pause()
            assert app._view_mode is OverviewViewMode.THIN
            app.action_cycle_view_mode()
            await pilot.pause()
            assert app._view_mode is OverviewViewMode.NORMAL
    finally:
        supervisor.stop(timeout=10.0)
    assert not supervisor.is_alive()


async def test_trend_window_cycle_does_not_soft_reset_existing_history(tmp_path: Path) -> None:
    """Changing the trend span is a view change; only the explicit reset action moves the trend epoch."""
    store = AppStateStore(tmp_path / ".horde_worker_regen" / "state.json")
    store.set_auto_start_worker(True)
    store.record_onboarding_choice(OnboardingChoice.DECLINED)
    fake = FakeSupervisor(alive=True)
    fake.latest_snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Trend", worker_version="0.0.0")
    )
    app = HordeWorkerTUI(fake, config_path=Path("bridgeData.yaml"), app_state_store=store)

    async with app.run_test(size=(120, 40)) as pilot:
        app._tick()
        await pilot.pause()
        overview = app.query_one(OverviewView)
        overview._trend_epoch = 123.0

        app.action_cycle_trend_window()
        await pilot.pause()

        assert app._trend_window is OverviewTrendWindow.THIRTY_MINUTES
        assert overview.trend_window() is OverviewTrendWindow.THIRTY_MINUTES
        assert overview._trend_epoch == 123.0
        assert store.load().overview_trend_window is OverviewTrendWindow.THIRTY_MINUTES


async def test_control_tab_forwards_relegated_controls(tmp_path: Path) -> None:
    """The Control tab owns local pause, auto-start, and horde-maintenance controls."""
    store = AppStateStore(tmp_path / ".horde_worker_regen" / "state.json")
    store.set_auto_start_worker(True)
    store.record_onboarding_choice(OnboardingChoice.DECLINED)
    fake = FakeSupervisor(alive=True)
    fake.latest_snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Control", worker_version="0.0.0")
    )
    fake.last_liveness_wall_time = time.time()
    app = HordeWorkerTUI(fake, config_path=Path("bridgeData.yaml"), app_state_store=store)

    async with app.run_test(size=(120, 40)) as pilot:
        app.query_one("#main-tabs", TabbedContent).active = "tab-control"
        app._tick()
        await pilot.pause()

        await pilot.click("#control-pause")
        await pilot.pause()
        await pilot.click("#control-autostart")
        await pilot.pause()
        await pilot.click("#control-maintenance")
        await pilot.pause()

    assert fake.pause_calls == 1
    assert store.load().auto_start_worker is False
    assert fake.server_maintenance == [True]


@pytest.mark.e2e
async def test_view_mode_density_propagates_to_every_tab(tmp_path: Path) -> None:
    """F6 density is app-wide: thin trims Downloads/Logs/Config/Benchmark; details restores them."""
    from textual.widgets import Static

    from horde_worker_regen.tui.config_form import CONFIG_SUBTABS
    from horde_worker_regen.tui.widgets.benchmark import BenchmarkView
    from horde_worker_regen.tui.widgets.config_editor import ConfigEditorView, _subtab_id
    from horde_worker_regen.tui.widgets.downloads import DownloadsView
    from horde_worker_regen.tui.widgets.logs import LogsView

    store = AppStateStore(tmp_path / ".horde_worker_regen" / "state.json")
    store.set_auto_start_worker(True)
    supervisor = WorkerSupervisor(WorkerLaunchOptions(worker_name="DensityApp"), mode=WorkerProcessMode.FAKE)
    app = HordeWorkerTUI(supervisor, config_path=Path("bridgeData.yaml"), app_state_store=store)
    essentials_id = _subtab_id(CONFIG_SUBTABS[0][0])
    advanced_id = _subtab_id(CONFIG_SUBTABS[-1][0])
    try:
        async with app.run_test(size=(160, 48)) as pilot:
            for _ in range(8):
                app._tick()
                await pilot.pause()
                await asyncio.sleep(0.1)

            downloads = app.query_one(DownloadsView)
            logs = app.query_one(LogsView)
            config = app.query_one(ConfigEditorView)
            benchmark = app.query_one(BenchmarkView)
            from textual.widgets import TabbedContent as _TC

            cfg_tabs = config.query_one("#config-subtabs", _TC)

            # Normal: everything visible.
            assert downloads.query_one("#downloads-queue", Static).display is True
            assert logs.query_one("#log-controls").display is True
            assert benchmark.query_one("#benchmark-setup").display is True

            # Thin: the trimmed essentials.
            app._view_mode = OverviewViewMode.THIN
            app._tick()
            await pilot.pause()
            assert downloads.query_one("#downloads-queue", Static).display is False
            assert logs.query_one("#log-controls").display is False
            assert benchmark.query_one("#benchmark-setup").display is False
            assert cfg_tabs.active == essentials_id
            assert cfg_tabs.get_tab(advanced_id).display is False

            # Details: the fullest view (the log tally appears; all config sub-tabs return).
            app._view_mode = OverviewViewMode.DETAILS
            app._tick()
            await pilot.pause()
            assert logs.query_one("#log-tally").display is True
            assert benchmark.query_one("#benchmark-setup").display is True
            assert cfg_tabs.get_tab(advanced_id).display is True
    finally:
        supervisor.stop(timeout=10.0)
    assert not supervisor.is_alive()


async def test_tick_judges_responsiveness_on_liveness_not_snapshot_age(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh liveness frame keeps the worker responsive even when the last full snapshot is old.

    Regression guard for the liveness/snapshot decoupling: ``_tick`` must feed ``derive`` the liveness
    age (loop progress), not the snapshot age (data freshness), so a coalesced or briefly-failing
    snapshot build cannot read as UNRESPONSIVE.
    """
    store = AppStateStore(tmp_path / ".horde_worker_regen" / "state.json")
    supervisor = WorkerSupervisor(WorkerLaunchOptions(worker_name="LivenessApp"), mode=WorkerProcessMode.FAKE)
    app = HordeWorkerTUI(supervisor, config_path=Path("bridgeData.yaml"), app_state_store=store)

    now = time.time()
    stale_snapshot = WorkerStateSnapshot(config=WorkerConfigSummary(dreamer_name="L", worker_version="12.0.0"))
    stale_snapshot.timestamp = now - 100.0  # far past any staleness budget
    supervisor.latest_snapshot = stale_snapshot

    captured: list[float | None] = []

    def _recording_derive(snapshot: object, status: object, age: float | None, **kwargs: object) -> object:
        captured.append(age)
        return derive(snapshot, status, age, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("horde_worker_regen.tui.app.derive", _recording_derive)

    async with app.run_test(size=(120, 40)):
        # Fresh liveness despite the 100s-old snapshot: the age fed to derive must reflect liveness.
        supervisor.last_liveness_wall_time = time.time()
        app._tick()
        assert captured, "derive was not called"
        assert captured[-1] is not None and captured[-1] < 5.0

        # With no liveness frame ever seen, it falls back to the (old) snapshot age and reads as stale.
        supervisor.last_liveness_wall_time = None
        app._tick()
        assert captured[-1] is not None and captured[-1] > 90.0


async def test_downloads_tab_label_badges_active_download(tmp_path: Path) -> None:
    """The Downloads tab label gains a live ready/total badge while a download is in flight, then clears."""
    from horde_worker_regen.process_management.ipc.supervisor_channel import (
        CurrentDownloadStatus,
        DownloadPhase,
        DownloadPlanSummary,
        DownloadStatusSnapshot,
    )

    store = AppStateStore(tmp_path / ".horde_worker_regen" / "state.json")
    supervisor = WorkerSupervisor(WorkerLaunchOptions(worker_name="TabBadge"), mode=WorkerProcessMode.FAKE)
    app = HordeWorkerTUI(supervisor, config_path=Path("bridgeData.yaml"), app_state_store=store)

    downloading = WorkerStateSnapshot(config=WorkerConfigSummary(dreamer_name="T", worker_version="0.0.0"))
    downloading.download_plan = DownloadPlanSummary(num_present=1, num_to_download=2)
    downloading.downloads = DownloadStatusSnapshot(
        phase=DownloadPhase.DOWNLOADING,
        current=CurrentDownloadStatus(
            model_name="BigModel",
            feature="image model",
            target_dir="/models",
            downloaded_bytes=512,
            total_bytes=1024,
            speed_bps=4096.0,
        ),
    )

    async with app.run_test(size=(120, 40)) as pilot:
        tabs = app.query_one("#main-tabs", TabbedContent)

        supervisor.latest_snapshot = downloading
        app._tick()
        await pilot.pause()
        # plan: 1 present + 2 to download => 1 of 3 ready (single-sourced from on-disk presence).
        label = tabs.get_tab("tab-downloads").label_text
        assert "Downloads" in label and "1/3" in label and "⬇" in label

        idle = WorkerStateSnapshot(config=WorkerConfigSummary(dreamer_name="T", worker_version="0.0.0"))
        idle.downloads = DownloadStatusSnapshot(phase=DownloadPhase.IDLE)
        supervisor.latest_snapshot = idle
        app._tick()
        await pilot.pause()
        assert tabs.get_tab("tab-downloads").label_text == "Downloads"


async def test_wizard_start_focuses_downloads_tab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Choosing 'Start' in the wizard surfaces the Downloads tab so first-run downloads are visible (P1.2)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "bridgeData.yaml").write_text("api_key: real-key\ndreamer_name: MyWorker\n", encoding="utf-8")
    store = AppStateStore(tmp_path / ".horde_worker_regen" / "state.json")
    supervisor = WorkerSupervisor(WorkerLaunchOptions(worker_name="FocusTest"), mode=WorkerProcessMode.FAKE)
    started = False

    def _record_start() -> None:
        nonlocal started
        started = True

    monkeypatch.setattr(supervisor, "start", _record_start)
    app = HordeWorkerTUI(supervisor, config_path=Path("bridgeData.yaml"), app_state_store=store)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._on_wizard_outcome(WizardOutcome.START)
        await pilot.pause()
        assert app.query_one("#main-tabs", TabbedContent).active == "tab-downloads"
        assert started is True


async def test_tick_clears_optimistic_maintenance_after_successful_pop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful job pop is authoritative evidence that optimistic maintenance is no longer active."""
    store = AppStateStore(tmp_path / ".horde_worker_regen" / "state.json")
    supervisor = FakeSupervisor(alive=True)
    supervisor.latest_snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="T", worker_version="0.0.0"),
        num_jobs_popped=7,
    )
    app = HordeWorkerTUI(supervisor, config_path=Path("bridgeData.yaml"), app_state_store=store)
    app._intended_server_maintenance = True
    app._server_maintenance_intent_pop_count = 6

    captured: list[bool] = []

    def _recording_derive(snapshot: object, status: object, age: float | None, **kwargs: object) -> object:
        captured.append(bool(kwargs.get("optimistic_server_maintenance")))
        return derive(snapshot, status, age, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("horde_worker_regen.tui.app.derive", _recording_derive)

    async with app.run_test(size=(120, 40)):
        app._tick()

    assert app._intended_server_maintenance is None
    assert app._server_maintenance_intent_pop_count is None
    assert captured[-1] is False


def _overview_app(tmp_path: Path) -> tuple[HordeWorkerTUI, AppStateStore, FakeSupervisor]:
    """Build a TUI over a fake, already-running worker with a minimal snapshot for overview tests."""
    store = AppStateStore(tmp_path / ".horde_worker_regen" / "state.json")
    store.set_auto_start_worker(True)
    store.record_onboarding_choice(OnboardingChoice.DECLINED)
    supervisor = FakeSupervisor(alive=True)
    supervisor.latest_snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Layout", worker_version="0.0.0"),
    )
    app = HordeWorkerTUI(supervisor, config_path=Path("bridgeData.yaml"), app_state_store=store)
    return app, store, supervisor


async def test_overview_hide_persists_masks_and_reveal_toggles(tmp_path: Path) -> None:
    """Hiding an element persists and masks its node; the reveal toggle un-suppresses it, then re-hides."""
    app, store, _ = _overview_app(tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        app._tick()
        await pilot.pause()
        trends = app.query_one(OverviewView).query_one("#overview-trends", Static)
        assert trends.display is True

        app._on_overview_layout_chosen(frozenset({"trends"}))
        await pilot.pause()
        assert store.load().overview_hidden_elements == ["trends"]
        assert trends.display is False

        app.action_toggle_hidden_reveal()  # 'h': reveal
        await pilot.pause()
        assert trends.display is True

        app.action_toggle_hidden_reveal()  # 'h' again: re-hide
        await pilot.pause()
        assert trends.display is False


async def test_customize_modal_saves_selection(tmp_path: Path) -> None:
    """The 'c' modal opens, a toggled element is persisted on escape (save), and the node is masked."""
    from textual.widgets import Checkbox

    from horde_worker_regen.tui.widgets.overview_layout import OverviewLayoutModal

    app, store, _ = _overview_app(tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        app._tick()
        await pilot.pause()

        app.action_customize_overview()
        await pilot.pause()
        assert isinstance(app.screen, OverviewLayoutModal)
        app.screen.query_one("#layout-cb-health", Checkbox).value = True
        await pilot.pause()
        await pilot.press("escape")  # save & close
        await pilot.pause()

        assert store.load().overview_hidden_elements == ["health"]
        assert app.query_one(OverviewView).query_one("#overview-health", Static).display is False


@pytest.mark.parametrize(
    ("width", "expect_cols2", "expect_cols3"),
    [(80, False, False), (120, True, False), (200, True, True)],
)
async def test_overview_column_classes_track_width(
    tmp_path: Path, width: int, expect_cols2: bool, expect_cols3: bool
) -> None:
    """The overview stamps -cols-2/-cols-3 from its width, so narrow stays single-column and wide spreads."""
    app, _, _ = _overview_app(tmp_path)
    async with app.run_test(size=(width, 40)) as pilot:
        app._tick()
        await pilot.pause()
        overview = app.query_one(OverviewView)
        assert overview.has_class("-cols-2") is expect_cols2
        assert overview.has_class("-cols-3") is expect_cols3


async def test_trends_panel_gets_a_wide_band_when_multicolumn(tmp_path: Path) -> None:
    """Regression: Trends spans the full grid width (not a squeezed fractional column) on a wide terminal."""
    app, _, _ = _overview_app(tmp_path)
    async with app.run_test(size=(200, 45)) as pilot:
        app._tick()
        await pilot.pause()
        overview = app.query_one(OverviewView)
        assert overview.has_class("-cols-3")
        trends = overview.query_one("#overview-trends", Static)
        health = overview.query_one("#overview-health", Static)
        # Trends must be far wider than the fractional core column Health sits in: it carries a ~48-char
        # sparkline and would scrunch if squeezed into a 1/3 slot.
        assert trends.size.width > health.size.width
