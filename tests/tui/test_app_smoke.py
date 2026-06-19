"""Headless end-to-end smoke test for the TUI app against the fake worker."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from textual.widgets import TabbedContent

from horde_worker_regen.app_state import AppStateStore, OverviewViewMode
from horde_worker_regen.process_management.supervisor_channel import WorkerConfigSummary, WorkerStateSnapshot
from horde_worker_regen.run_worker import WorkerLaunchOptions
from horde_worker_regen.tui.app import HordeWorkerTUI
from horde_worker_regen.tui.health import WorkerPhase, derive
from horde_worker_regen.tui.wizard import WizardOutcome
from horde_worker_regen.tui.worker_launcher import WorkerProcessMode, WorkerSupervisor

# Headless run_test throttles timers, so the test drives the app's tick directly.
_LIVE_PHASES = {
    WorkerPhase.WARMING_UP,
    WorkerPhase.SERVING,
    WorkerPhase.READY,
    WorkerPhase.IDLE,
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

            for tab in ("tab-live", "tab-downloads", "tab-logs", "tab-config", "tab-insights", "tab-overview"):
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
