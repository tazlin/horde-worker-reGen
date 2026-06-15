"""Headless end-to-end smoke test for the TUI app against the fake worker."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from textual.widgets import TabbedContent

from horde_worker_regen.app_state import AppStateStore
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
    finally:
        supervisor.stop(timeout=10.0)
    assert not supervisor.is_alive()


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
