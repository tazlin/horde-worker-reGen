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
from horde_worker_regen.benchmark.capabilities.capability import Capability, CapabilityKind, CapabilityVerdict
from horde_worker_regen.benchmark.capabilities.result import (
    CapabilityProbeResult,
    CapabilityReport,
    MachineInfo,
    SuggestedBridgeData,
)
from horde_worker_regen.benchmark.enums import BenchTier
from horde_worker_regen.benchmark.progress_channel import (
    LevelPlanRow,
    LevelStarted,
    RampFinished,
    RampPlanned,
    RampStarted,
    SuggestionDecisionRow,
)
from horde_worker_regen.tui.benchmark_launcher import BenchmarkOptions, BenchmarkRunState, BenchmarkSupervisorStatus
from horde_worker_regen.tui.widgets.benchmark import BenchmarkView, _Phase
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


def _write_fake_report(run_dir: Path, *, run_id: str, created_at: float, verdict: CapabilityVerdict) -> None:
    """Write a minimal valid report.json into a run directory for history tests."""
    run_dir.mkdir(parents=True, exist_ok=True)
    baseline = Capability(tier=BenchTier.SD15, kind=CapabilityKind.BASELINE)
    report = CapabilityReport(
        run_id=run_id,
        created_at=created_at,
        machine=MachineInfo(gpu_name="Test GPU", total_vram_mb=16000),
        probes=[CapabilityProbeResult(capability=baseline, verdict=verdict)],
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
        assert options.force is False

        # Flipping a switch inside the collapsed section still flows into the collected options.
        app.query_one("#benchmark-force", Switch).value = True
        await pilot.pause()
        updated = view._collect_options()
        assert updated.force is True


async def test_per_axis_switch_excludes_only_that_axis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deselecting a single capability switch excludes just that axis; siblings stay selected."""
    monkeypatch.chdir(tmp_path)
    app = _BenchmarkHarness()
    async with app.run_test() as pilot:
        view = app.query_one(BenchmarkView)
        assert view._collect_options().excluded_capabilities == []  # everything on by default

        app.query_one("#benchmark-axis-controlnet", Switch).value = False
        await pilot.pause()
        excluded = view._collect_options().excluded_capabilities
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


def _plan_rows() -> list[LevelPlanRow]:
    """A two-level plan: one that fits this machine and one skipped for insufficient VRAM."""
    return [
        LevelPlanRow(level_id="sd15-base", stage="baseline", tier="sd15", estimated_vram_mb=4096, will_run=True),
        LevelPlanRow(
            level_id="flux-base",
            stage="baseline",
            tier="flux",
            estimated_vram_mb=16384,
            will_run=False,
            verdict="insufficient VRAM: estimated 16384 MB needed, 12000 MB available",
        ),
    ]


async def test_setup_collapses_and_plan_folds_once_a_run_is_live(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Starting a run hides the setup chrome and folds the per-level plan so live progress leads."""
    monkeypatch.chdir(tmp_path)
    app = _BenchmarkHarness()
    async with app.run_test() as pilot:
        view = app.query_one(BenchmarkView)

        # Idle setup: the launch pad is visible and the plan block is hidden (no plan yet).
        view.update_view(BenchmarkRunState(), BenchmarkSupervisorStatus.IDLE)
        await pilot.pause()
        assert app.query_one("#benchmark-setup").display is True
        assert app.query_one("#benchmark-plan-summary").display is False

        running = BenchmarkRunState()
        running.apply(RampStarted(run_id="r", num_levels=2))
        running.apply(RampPlanned(run_id="r", rows=_plan_rows()))
        running.apply(LevelStarted(level_id="sd15-base", num_levels=2))
        view.update_view(running, BenchmarkSupervisorStatus.RUNNING)
        await pilot.pause()

        # Setup is demoted, the plan summary stays visible, and its per-level table folds away.
        assert app.query_one("#benchmark-setup").display is False
        assert app.query_one("#benchmark-plan-summary").display is True
        assert app.query_one("#benchmark-plan-detail", Collapsible).collapsed is True


async def test_plan_table_states_readiness_in_plain_language(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The plan reads as readiness, not a command: ``Ready``/``Skip`` with a reason, never a bare ``RUN``."""
    monkeypatch.chdir(tmp_path)
    app = _BenchmarkHarness()
    async with app.run_test():
        view = app.query_one(BenchmarkView)
        rows = _plan_rows()

        summary = _render_to_text(view._plan_summary(rows))
        assert "levels will run on this machine" in summary
        assert "not enough VRAM" in summary  # the skip reason is condensed to plain words

        table = _render_to_text(view._plan_table(rows))
        assert "Ready" in table
        assert "Skip" in table
        assert "RUN" not in table  # the old command-like verdict cell is gone


async def test_plan_table_shows_download_first_for_runnable_but_incomplete_levels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A level that fits the machine but lacks models/checkpoints/annotators reads as an actionable third state.

    Not a green ``Ready`` (it would otherwise download mid-run and skew the timing) and not a grey ``Skip``
    (it runs once fetched): the operator sees ``Download first`` with what to fetch, and the summary nags to
    pre-download. Guards the reported bug where a controlnet level read ``Ready`` despite missing checkpoints.
    """
    monkeypatch.chdir(tmp_path)
    app = _BenchmarkHarness()
    async with app.run_test():
        view = app.query_one(BenchmarkView)
        rows = [
            LevelPlanRow(
                level_id="sd15-controlnet",
                stage="controlnet",
                tier="sd15",
                estimated_vram_mb=4096,
                will_run=True,
                needs_download=True,
                download_summary="controlnet checkpoints, controlnet annotators",
            ),
        ]

        table = _render_to_text(view._plan_table(rows))
        assert "Download first" in table
        assert "controlnet checkpoints" in table  # the status names what must be fetched
        assert "Ready" not in table  # it must never read as ready while required artifacts are missing

        summary = _render_to_text(view._plan_summary(rows))
        assert "Download models" in summary  # the unified pre-download nag points at the action


async def test_stepper_marks_the_current_phase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The phase spine names every step and marks earlier ones done once the screen reaches a later phase."""
    monkeypatch.chdir(tmp_path)
    app = _BenchmarkHarness()
    async with app.run_test():
        view = app.query_one(BenchmarkView)

        setup = _render_to_text(view._render_stepper(_Phase.SETUP))
        for step in ("Preview", "Download", "Run", "Apply"):
            assert step in setup

        # By the Run phase, the earlier Preview/Download steps read as completed (the check glyph).
        running = _render_to_text(view._render_stepper(_Phase.RUNNING))
        assert "✓ Preview" in running and "✓ Download" in running


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
    _write_fake_report(
        results_root / "20260101-000000", run_id="old", created_at=100.0, verdict=CapabilityVerdict.PROVEN
    )
    _write_fake_report(
        results_root / "20260201-000000", run_id="new", created_at=200.0, verdict=CapabilityVerdict.DISPROVEN
    )

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
        assert "disproven" in compared_text  # the regressed probe verdict shows up in the diff
        assert "new" in compared_text and "old" in compared_text  # the run ids head the diff
