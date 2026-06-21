"""Headless tests for the unsaved-config-edits guard when leaving the Config tab."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from textual.widgets import Input, TabbedContent

from horde_worker_regen.app_state import AppStateStore
from horde_worker_regen.run_worker import WorkerLaunchOptions
from horde_worker_regen.tui.app import HordeWorkerTUI
from horde_worker_regen.tui.widgets.config_editor import ConfigEditorView, ConfigLeaveChoice, ConfigLeaveModal
from horde_worker_regen.tui.worker_launcher import WorkerProcessMode, WorkerSupervisor


def _make_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[HordeWorkerTUI, WorkerSupervisor]:
    """Build a fake-worker app with the first-run prompts suppressed so the only modal is the guard's."""
    store = AppStateStore(tmp_path / ".horde_worker_regen" / "state.json")
    store.set_auto_start_worker(True)
    supervisor = WorkerSupervisor(WorkerLaunchOptions(worker_name="LeaveGuard"), mode=WorkerProcessMode.FAKE)
    app = HordeWorkerTUI(supervisor, config_path=Path("bridgeData.yaml"), app_state_store=store)
    monkeypatch.setattr(app, "_maybe_prompt_onboarding", lambda: None)
    return app, supervisor


async def _boot(app: HordeWorkerTUI, pilot: object) -> ConfigEditorView:
    """Pump a few ticks so the panes mount and the config baseline is captured."""
    for _ in range(5):
        app._tick()
        await pilot.pause()  # type: ignore[attr-defined]
        await asyncio.sleep(0.05)
    return app.query_one(ConfigEditorView)


def _make_dirty(editor: ConfigEditorView) -> None:
    """Edit a field so the editor reports unsaved changes."""
    field = editor.query_one("#cfg-dreamer_name", Input)
    field.value = field.value + "_edited"


@pytest.mark.e2e
async def test_leaving_dirty_config_reverts_and_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Navigating off a dirty Config tab snaps back to it and shows the warning modal."""
    app, supervisor = _make_app(tmp_path, monkeypatch)
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            editor = await _boot(app, pilot)
            tabs = app.query_one("#main-tabs", TabbedContent)

            tabs.active = "tab-config"
            await pilot.pause()
            _make_dirty(editor)
            assert editor.is_dirty() is True

            tabs.active = "tab-overview"
            await pilot.pause()

            assert isinstance(app.screen, ConfigLeaveModal)
            assert tabs.active == "tab-config"
    finally:
        supervisor.stop(timeout=10.0)


@pytest.mark.e2e
async def test_leave_choice_keeps_edits_and_navigates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """'Leave' navigates to the target while the unsaved edits stay live in the form."""
    app, supervisor = _make_app(tmp_path, monkeypatch)
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            editor = await _boot(app, pilot)
            tabs = app.query_one("#main-tabs", TabbedContent)
            tabs.active = "tab-config"
            await pilot.pause()
            _make_dirty(editor)

            tabs.active = "tab-logs"
            await pilot.pause()
            assert isinstance(app.screen, ConfigLeaveModal)
            app.screen.dismiss(ConfigLeaveChoice.LEAVE)
            await pilot.pause()

            assert tabs.active == "tab-logs"
            assert editor.is_dirty() is True  # edits were kept, not written
    finally:
        supervisor.stop(timeout=10.0)


@pytest.mark.e2e
async def test_discard_choice_reverts_form_and_navigates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """'Discard' reloads the form from disk (clean) and then navigates to the target."""
    app, supervisor = _make_app(tmp_path, monkeypatch)
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            editor = await _boot(app, pilot)
            tabs = app.query_one("#main-tabs", TabbedContent)
            tabs.active = "tab-config"
            await pilot.pause()
            _make_dirty(editor)

            tabs.active = "tab-overview"
            await pilot.pause()
            assert isinstance(app.screen, ConfigLeaveModal)
            app.screen.dismiss(ConfigLeaveChoice.DISCARD)
            await pilot.pause()

            assert tabs.active == "tab-overview"
            assert editor.is_dirty() is False
    finally:
        supervisor.stop(timeout=10.0)


@pytest.mark.e2e
async def test_stay_choice_keeps_user_on_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """'Stay' cancels the navigation and leaves the user on the Config tab with edits intact."""
    app, supervisor = _make_app(tmp_path, monkeypatch)
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            editor = await _boot(app, pilot)
            tabs = app.query_one("#main-tabs", TabbedContent)
            tabs.active = "tab-config"
            await pilot.pause()
            _make_dirty(editor)

            tabs.active = "tab-overview"
            await pilot.pause()
            assert isinstance(app.screen, ConfigLeaveModal)
            app.screen.dismiss(ConfigLeaveChoice.STAY)
            await pilot.pause()

            assert tabs.active == "tab-config"
            assert editor.is_dirty() is True
    finally:
        supervisor.stop(timeout=10.0)


@pytest.mark.e2e
async def test_never_choice_suppresses_further_warnings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """'Never' navigates away and, for the rest of the session, leaving is no longer gated."""
    app, supervisor = _make_app(tmp_path, monkeypatch)
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            editor = await _boot(app, pilot)
            tabs = app.query_one("#main-tabs", TabbedContent)
            tabs.active = "tab-config"
            await pilot.pause()
            _make_dirty(editor)

            tabs.active = "tab-overview"
            await pilot.pause()
            assert isinstance(app.screen, ConfigLeaveModal)
            app.screen.dismiss(ConfigLeaveChoice.NEVER)
            await pilot.pause()
            assert tabs.active == "tab-overview"
            assert app._config_leave_warning_suppressed is True

            # A second dirty departure is no longer gated.
            tabs.active = "tab-config"
            await pilot.pause()
            _make_dirty(editor)
            tabs.active = "tab-logs"
            await pilot.pause()
            assert not isinstance(app.screen, ConfigLeaveModal)
            assert tabs.active == "tab-logs"
    finally:
        supervisor.stop(timeout=10.0)


@pytest.mark.e2e
async def test_clean_config_navigates_without_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A pristine Config tab never triggers the guard."""
    app, supervisor = _make_app(tmp_path, monkeypatch)
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            await _boot(app, pilot)
            tabs = app.query_one("#main-tabs", TabbedContent)
            tabs.active = "tab-config"
            await pilot.pause()
            tabs.active = "tab-overview"
            await pilot.pause()

            assert not isinstance(app.screen, ConfigLeaveModal)
            assert tabs.active == "tab-overview"
    finally:
        supervisor.stop(timeout=10.0)
