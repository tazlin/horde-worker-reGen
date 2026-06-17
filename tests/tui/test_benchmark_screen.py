"""Pilot tests for the benchmark screen: button gating by status, option collection, and run requests."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console, RenderableType
from textual.app import App, ComposeResult
from textual.widgets import Button, Collapsible, Switch

from horde_worker_regen.app_state import (
    AppStateStore,
    BenchmarkAvailability,
    KnownGoodSettings,
    KnownGoodSource,
    OnboardingChoice,
)
from horde_worker_regen.benchmark.enums import BenchTier, LevelOutcome
from horde_worker_regen.benchmark.ladder import LadderOptions, build_default_ladder
from horde_worker_regen.benchmark.progress_channel import (
    LevelStarted,
    RampFinished,
    RampStarted,
    SuggestionDecisionRow,
)
from horde_worker_regen.benchmark.report import BenchmarkReport, LevelReport, MachineInfo, SuggestedBridgeData
from horde_worker_regen.tui.benchmark_launcher import BenchmarkOptions, BenchmarkRunState, BenchmarkSupervisorStatus
from horde_worker_regen.tui.widgets.benchmark import BenchmarkView
from horde_worker_regen.tui.widgets.benchmark_history import BenchmarkHistoryModal
from horde_worker_regen.tui.widgets.onboarding import BenchmarkOnboardingModal


def _render_to_text(renderable: RenderableType) -> str:
    """Render a rich renderable to plain text so a test can assert on what the user would see.

    Uses an isolated capture buffer rather than printing, so it never routes through the app's
    redirected (cp1252 on Windows) stdout, which would choke on the panels' box-drawing characters.
    """
    console = Console(width=140)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


def _write_fake_report(run_dir: Path, *, run_id: str, created_at: float, outcome: LevelOutcome) -> None:
    """Write a minimal valid report.json into a run directory for history tests."""
    run_dir.mkdir(parents=True, exist_ok=True)
    level = next(
        lvl for lvl in build_default_ladder(LadderOptions(tiers=[BenchTier.SD15])) if lvl.establishes_tier_baseline
    )
    report = BenchmarkReport(
        run_id=run_id,
        created_at=created_at,
        machine=MachineInfo(gpu_name="Test GPU", total_vram_mb=16000),
        levels=[LevelReport(level=level, outcome=outcome)],
        suggested_bridge_data=SuggestedBridgeData(),
        tier_baselines_its={"sd15": 5.0},
    )
    (run_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")


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


async def test_advanced_options_start_collapsed_and_collect_through_collapsible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The advanced options begin collapsed, yet their switches are still readable by _collect_options."""
    monkeypatch.chdir(tmp_path)
    app = _BenchmarkHarness()
    async with app.run_test() as pilot:
        assert app.query_one("#benchmark-advanced", Collapsible).collapsed is True
        # Preview plan leads the action bar as the recommended first step.
        assert app.query_one("#benchmark-preview", Button) is not None

        view = app.query_one(BenchmarkView)
        options = view._collect_options()
        assert options.warm is True
        assert options.force is False

        # Flipping a switch inside the collapsed section still flows into the collected options.
        app.query_one("#benchmark-force", Switch).value = True
        app.query_one("#benchmark-warm", Switch).value = False
        await pilot.pause()
        updated = view._collect_options()
        assert updated.force is True
        assert updated.warm is False


async def test_per_axis_switch_excludes_only_that_axis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deselecting a single capability switch excludes just that axis; siblings stay selected."""
    monkeypatch.chdir(tmp_path)
    app = _BenchmarkHarness()
    async with app.run_test() as pilot:
        view = app.query_one(BenchmarkView)
        assert view._collect_options().excluded_axes == []  # everything on by default

        app.query_one("#benchmark-axis-controlnet", Switch).value = False
        await pilot.pause()
        excluded = view._collect_options().excluded_axes
        assert excluded == ["controlnet"]


async def test_tier_switches_drive_selected_tiers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The tier toggles replace the old free-text box: sd15/sdxl default on, flux/qwen opt-in."""
    monkeypatch.chdir(tmp_path)
    app = _BenchmarkHarness()
    async with app.run_test() as pilot:
        view = app.query_one(BenchmarkView)
        assert view._collect_options().tiers == ["sd15", "sdxl"]

        app.query_one("#benchmark-tier-sdxl", Switch).value = False
        app.query_one("#benchmark-tier-flux", Switch).value = True
        await pilot.pause()
        assert view._collect_options().tiers == ["sd15", "flux"]


async def test_finished_run_renders_provenance_in_suggestion_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finished run with provenance shows the per-setting basis (and any consistency warning)."""
    monkeypatch.chdir(tmp_path)
    app = _BenchmarkHarness()
    async with app.run_test() as pilot:
        view = app.query_one(BenchmarkView)
        finished = BenchmarkRunState()
        finished.apply(
            RampFinished(
                run_id="r",
                levels_passed=1,
                levels_total=1,
                suggested_bridge_data_yaml="allow_lora: false",
                suggestion_decisions=[
                    SuggestionDecisionRow(
                        setting="allow_lora",
                        value_text="off",
                        basis="untested_skipped",
                        basis_label="off: never tested (skipped)",
                        detail="insufficient disk",
                    ),
                ],
                consistency_warnings=["allow_lora untested"],
            ),
        )
        view.update_view(finished, BenchmarkSupervisorStatus.FINISHED)
        await pilot.pause()

        rendered = _render_to_text(view._render_suggestion(finished))
        assert "allow_lora" in rendered
        assert "never tested" in rendered
        assert "allow_lora untested" in rendered  # the consistency warning surfaces too


async def test_history_button_opens_modal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The History button opens the past-runs modal, even with no runs on disk."""
    monkeypatch.chdir(tmp_path)
    app = _BenchmarkHarness()
    async with app.run_test() as pilot:
        await pilot.click("#benchmark-history")
        await pilot.pause()
        assert isinstance(app.screen, BenchmarkHistoryModal)


class _HistoryHarness(App[None]):
    """A minimal app that pushes the history modal pointed at a chosen results root."""

    def __init__(self, results_root: Path) -> None:
        super().__init__()
        self._results_root = results_root

    def compose(self) -> ComposeResult:
        yield Button("host", id="host")

    def on_mount(self) -> None:
        self.push_screen(BenchmarkHistoryModal(results_root=self._results_root))


async def test_history_modal_views_report_and_compares_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The modal renders a run's markdown report and a run-to-run diff into its detail pane."""
    monkeypatch.chdir(tmp_path)
    results_root = tmp_path / "benchmark_results"
    _write_fake_report(results_root / "20260101-000000", run_id="old", created_at=100.0, outcome=LevelOutcome.PASSED)
    _write_fake_report(results_root / "20260201-000000", run_id="new", created_at=200.0, outcome=LevelOutcome.FAILED)

    app = _HistoryHarness(results_root)
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, BenchmarkHistoryModal)

        # The newest run is selected by default (cursor row 0); View renders its markdown report.
        await pilot.click("#history-view")
        await pilot.pause()
        viewed = modal._last_detail
        assert viewed is not None
        assert "Worker Benchmark Report" in _render_to_text(viewed)

        # Compare-with-previous diffs the newest run against the older one (a regression here).
        await pilot.click("#history-compare")
        await pilot.pause()
        compared = modal._last_detail
        assert compared is not None
        compared_text = _render_to_text(compared)
        assert "failed" in compared_text  # the regressed level shows up in the diff
        assert "new" in compared_text and "old" in compared_text  # the run ids head the diff
