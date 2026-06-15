"""Pilot tests for the benchmark screen: button gating by status, option collection, and run requests."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from horde_worker_regen.app_state import (
    AppStateStore,
    BenchmarkAvailability,
    KnownGoodSettings,
    KnownGoodSource,
    OnboardingChoice,
)
from horde_worker_regen.benchmark.progress_channel import LevelStarted, RampFinished, RampStarted
from horde_worker_regen.tui.benchmark_launcher import BenchmarkOptions, BenchmarkRunState, BenchmarkSupervisorStatus
from horde_worker_regen.tui.widgets.benchmark import BenchmarkView
from horde_worker_regen.tui.widgets.onboarding import BenchmarkOnboardingModal


class _BenchmarkHarness(App[None]):
    """A minimal app that mounts only the benchmark view and captures its run request."""

    def __init__(self) -> None:
        super().__init__()
        self.run_requested: BenchmarkOptions | None = None

    def compose(self) -> ComposeResult:
        yield BenchmarkView(worker_mode="fake")

    def on_benchmark_view_run_requested(self, message: BenchmarkView.RunRequested) -> None:
        self.run_requested = message.options


async def test_buttons_gate_on_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The action buttons enable only for actions valid in the current status."""
    monkeypatch.chdir(tmp_path)  # isolate the app-state read to an empty directory
    app = _BenchmarkHarness()
    async with app.run_test() as pilot:
        view = app.query_one(BenchmarkView)

        view.update_view(BenchmarkRunState(), BenchmarkSupervisorStatus.IDLE)
        await pilot.pause()
        assert app.query_one("#benchmark-run", Button).disabled is False
        assert app.query_one("#benchmark-cancel", Button).disabled is True
        assert app.query_one("#benchmark-apply", Button).disabled is True

        running = BenchmarkRunState()
        running.apply(RampStarted(run_id="r", num_levels=1))
        running.apply(LevelStarted(level_id="A", num_levels=1))
        view.update_view(running, BenchmarkSupervisorStatus.RUNNING)
        await pilot.pause()
        assert app.query_one("#benchmark-run", Button).disabled is True
        assert app.query_one("#benchmark-cancel", Button).disabled is False

        finished = BenchmarkRunState()
        finished.apply(RampFinished(run_id="r", levels_passed=1, levels_total=1, suggested_bridge_data_yaml="x: 1"))
        view.update_view(finished, BenchmarkSupervisorStatus.FINISHED)
        await pilot.pause()
        assert app.query_one("#benchmark-apply", Button).disabled is False


async def test_run_button_posts_request_with_worker_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pressing Run posts a request whose options carry the worker's process mode and chosen tiers."""
    monkeypatch.chdir(tmp_path)
    app = _BenchmarkHarness()
    async with app.run_test() as pilot:
        await pilot.click("#benchmark-run")
        await pilot.pause()

    assert app.run_requested is not None
    assert app.run_requested.process_mode == "fake"
    assert app.run_requested.tiers == ["sd15", "sdxl"]


async def test_restore_button_enabled_with_known_good(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The restore button enables when a known-good configuration is on record."""
    monkeypatch.chdir(tmp_path)
    AppStateStore().record_known_good(
        KnownGoodSettings(
            config_digest="digest",
            config_snapshot={"max_threads": 1},
            validated_at=1.0,
            worker_version="12.0.0",
            source=KnownGoodSource.CLEAN_RUN,
        ),
    )
    app = _BenchmarkHarness()
    async with app.run_test() as pilot:
        view = app.query_one(BenchmarkView)
        view.update_view(BenchmarkRunState(), BenchmarkSupervisorStatus.IDLE)
        await pilot.pause()
        assert app.query_one("#benchmark-restore", Button).disabled is False


class _OnboardingHarness(App[None]):
    """A minimal app that pushes the onboarding modal and records its dismissal value."""

    def __init__(self) -> None:
        super().__init__()
        self.choice: OnboardingChoice | None = None

    def compose(self) -> ComposeResult:
        yield Button("host", id="host")

    def on_mount(self) -> None:
        self.push_screen(BenchmarkOnboardingModal(BenchmarkAvailability.NONE), self._record)

    def _record(self, choice: OnboardingChoice | None) -> None:
        self.choice = choice


async def test_onboarding_modal_dismisses_with_choice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pressing a modal button dismisses it with the matching onboarding choice."""
    monkeypatch.chdir(tmp_path)
    app = _OnboardingHarness()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#onboarding-decline")
        await pilot.pause()

    assert app.choice is OnboardingChoice.DECLINED
