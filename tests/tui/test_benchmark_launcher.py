"""Tests for the TUI benchmark launcher: run-state reduction, command building, config apply, lifecycle."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from horde_worker_regen.benchmark.progress_channel import (
    PROGRESS_FILENAME,
    JsonlProgressSink,
    LevelFinished,
    LevelPlanRow,
    LevelProgress,
    LevelStarted,
    RampFinished,
    RampPlanned,
    RampStarted,
    RampStarting,
    SuggestionDecisionRow,
)
from horde_worker_regen.tui.benchmark_launcher import (
    BenchmarkOptions,
    BenchmarkRunState,
    BenchmarkSupervisor,
    BenchmarkSupervisorStatus,
    apply_known_good_to_config,
    apply_suggested_to_config,
    record_suggested_as_known_good,
)


def _ramp_events() -> list[object]:
    return [
        RampStarted(run_id="run1", num_levels=2, tiers=["sd15"], process_mode="fake", gpu_name="Test GPU"),
        LevelStarted(level_id="A", stage="A", tier="sd15", axis="baseline", jobs_expected=2, num_levels=2),
        LevelProgress(level_id="A", jobs_completed=1, jobs_expected=2, iterations_per_second=8.0, vram_used_mb=3000),
        LevelFinished(level_id="A", outcome="passed", its_p50=8.5),
        LevelStarted(
            level_id="B", stage="B", tier="sd15", axis="threads", jobs_expected=2, num_levels=2, level_index=1
        ),
        LevelFinished(level_id="B", outcome="failed", reasons=["too slow"]),
        RampFinished(
            run_id="run1",
            levels_passed=1,
            levels_total=2,
            num_findings=1,
            report_path="run1/report.json",
            suggested_bridge_data_yaml="max_threads: 1",
        ),
    ]


def test_run_state_reduces_event_stream() -> None:
    """Folding the full event stream yields the expected per-level outcomes and totals."""
    state = BenchmarkRunState()
    for event in _ramp_events():
        state.apply(event)  # type: ignore[arg-type]

    assert state.run_id == "run1"
    assert state.process_mode == "fake"
    assert [level.level_id for level in state.ordered_levels()] == ["A", "B"]
    assert state.levels["A"].outcome == "passed"
    assert state.levels["A"].iterations_per_second == 8.5  # the finished it/s p50 supersedes the live value
    assert state.levels["B"].outcome == "failed"
    assert state.current_level_id is None
    assert state.finished is True
    assert (state.levels_passed, state.levels_total) == (1, 2)
    assert state.num_findings == 1
    assert state.suggested_bridge_data_yaml == "max_threads: 1"


def test_run_state_tracks_current_level() -> None:
    """The current level is set on start and cleared when that level finishes."""
    state = BenchmarkRunState()
    state.apply(RampStarted(run_id="r", num_levels=1))
    state.apply(LevelStarted(level_id="A", num_levels=1))
    assert state.current_level_id == "A"
    state.apply(LevelProgress(level_id="A", jobs_completed=1))
    assert state.levels["A"].jobs_completed == 1
    state.apply(LevelFinished(level_id="A", outcome="passed"))
    assert state.current_level_id is None


def test_run_state_shows_startup_phase_until_first_level() -> None:
    """A RampStarting heartbeat sets a pre-level phase that the first LevelStarted clears."""
    state = BenchmarkRunState()
    state.apply(RampStarting(run_id="run1", process_mode="real", phase="detecting hardware"))
    assert state.run_id == "run1"
    assert state.process_mode == "real"
    assert state.startup_phase == "detecting hardware"
    assert not state.level_order  # nothing to render as a level yet

    state.apply(RampStarted(run_id="run1", num_levels=1))
    assert state.startup_phase == "detecting hardware"  # still pre-level; warm worker may be coming up

    state.apply(LevelStarted(level_id="A", num_levels=1))
    assert state.startup_phase == ""  # a level is running now


def test_mark_preparing_sets_visible_phase() -> None:
    """mark_preparing enters a busy, non-running state with a worker-stop phase to render."""
    supervisor = BenchmarkSupervisor()
    assert supervisor.status is BenchmarkSupervisorStatus.IDLE
    assert supervisor.is_active is False

    supervisor.mark_preparing()
    assert supervisor.status is BenchmarkSupervisorStatus.PREPARING
    assert supervisor.is_active is True  # blocks a second run request during the blocking worker stop
    assert "GPU" in supervisor.run_state.startup_phase


def test_build_command_includes_selected_flags() -> None:
    """The CLI command reflects the chosen tiers, mode, soak, and toggles."""
    options = BenchmarkOptions(
        tiers=["sd15", "sdxl"],
        process_mode="real",
        validate=False,
        soak_minutes=3.0,
        include_downloads=True,
    )
    command = options.build_command(Path("out"))
    assert command[1] == "-u"  # unbuffered so console.log populates live during a wedged startup
    assert command[2:5] == ["-m", "horde_worker_regen.benchmark.cli", "run"]
    assert "real" in command
    assert "sd15,sdxl" in command
    assert "--no-validate" in command
    assert "--include-downloads" in command


def test_build_command_maps_stage_toggles_and_force() -> None:
    """The stage toggles map to the run flags; force is a plan-preview flag (run has no --force)."""
    options = BenchmarkOptions(
        tiers=["sd15"],
        include_concurrency=False,
        include_features=False,
        include_alchemy=False,
        force=True,
    )
    command = options.build_command(Path("out"))
    assert command[command.index("-m") + 2] == "run"
    assert {"--no-concurrency", "--no-features", "--no-alchemy"} <= set(command)
    # Force is a machine-fit override for the no-boot preview, not the run.
    assert "--force" in options.build_plan_command()


def test_build_command_emits_one_exclude_capability_per_deselected_kind() -> None:
    """Each excluded capability becomes its own ``--exclude-capability KIND`` pair in run and plan argv."""
    options = BenchmarkOptions(tiers=["sd15"], excluded_capabilities=["controlnet", "alchemy_graph"])
    command = options.build_command(Path("out"))
    assert command.count("--exclude-capability") == 2
    assert "controlnet" in command
    assert "alchemy_graph" in command
    # The same selection flows into the plan preview so the preview matches what the run would do.
    assert options.build_plan_command().count("--exclude-capability") == 2


def test_build_plan_command_previews_without_running() -> None:
    """The plan command targets the `plan` subcommand with --json and carries the same selection flags."""
    options = BenchmarkOptions(tiers=["sd15"], include_features=False, force=True)
    command = options.build_plan_command()
    assert command[1:4] == ["-m", "horde_worker_regen.benchmark.cli", "plan"]
    assert "--json" in command
    assert "--no-features" in command
    assert "--force" in command
    assert "--out" not in command  # the plan starts no run, so it has no output directory


def test_run_state_captures_suggestion_provenance() -> None:
    """A RampFinished event populates the run state's per-setting provenance and consistency warnings."""
    state = BenchmarkRunState()
    state.apply(
        RampFinished(
            run_id="run1",
            levels_passed=1,
            levels_total=1,
            suggested_bridge_data_yaml="allow_lora: false",
            suggestion_decisions=[
                SuggestionDecisionRow(
                    setting="allow_lora",
                    value_text="off",
                    basis="untested_skipped",
                    basis_label="off: never tested (skipped)",
                ),
            ],
            consistency_warnings=["allow_lora untested"],
        ),
    )
    assert len(state.suggestion_decisions) == 1
    assert state.suggestion_decisions[0].setting == "allow_lora"
    assert state.suggestion_decisions[0].basis == "untested_skipped"
    assert state.consistency_warnings == ["allow_lora untested"]


def test_run_state_captures_ramp_plan() -> None:
    """A RampPlanned event populates the run state's per-level resource plan."""
    state = BenchmarkRunState()
    state.apply(
        RampPlanned(
            run_id="run1",
            rows=[LevelPlanRow(level_id="A-sd15-baseline", stage="A", tier="sd15", will_run=True)],
        ),
    )
    assert len(state.plan_rows) == 1
    assert state.plan_rows[0].level_id == "A-sd15-baseline"


def test_apply_suggested_to_config_preserves_untouched_keys(tmp_path: Path) -> None:
    """Applying the recommendation writes the tuned keys and leaves other keys intact."""
    from horde_worker_regen.benchmark.report import SuggestedBridgeData
    from horde_worker_regen.tui.config_form import load_config

    config_path = tmp_path / "bridgeData.yaml"
    config_path.write_text("dreamer_name: Tester\nmax_threads: 9\n", encoding="utf-8")

    suggested = SuggestedBridgeData(
        max_threads=2,
        queue_size=2,
        max_batch=4,
        allow_lora=True,
        models_to_load=["Deliberate"],
        alchemist=True,
    )
    apply_suggested_to_config(suggested, config_path)

    data = load_config(config_path)
    assert data["max_threads"] == 2
    assert data["queue_size"] == 2
    assert data["max_batch"] == 4
    assert data["allow_lora"] is True
    assert list(data["models_to_load"]) == ["Deliberate"]
    assert data["alchemist"] is True
    assert data["dreamer_name"] == "Tester"


def test_apply_known_good_writes_only_recognized_keys(tmp_path: Path) -> None:
    """Restoring a snapshot writes recognized config keys and ignores foreign keys and other content."""
    from horde_worker_regen.tui.config_form import load_config

    config_path = tmp_path / "bridgeData.yaml"
    config_path.write_text("dreamer_name: Tester\n", encoding="utf-8")

    apply_known_good_to_config({"max_threads": 3, "queue_size": 2, "unknown_key": 5}, config_path)

    data = load_config(config_path)
    assert data["max_threads"] == 3
    assert data["queue_size"] == 2
    assert "unknown_key" not in data
    assert data["dreamer_name"] == "Tester"


def test_record_suggested_as_known_good(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Applying a recommendation records it as benchmark-sourced known-good settings."""
    from horde_worker_regen.app_state import AppStateStore, KnownGoodSource
    from horde_worker_regen.benchmark.report import SuggestedBridgeData

    monkeypatch.chdir(tmp_path)
    record_suggested_as_known_good(SuggestedBridgeData(max_threads=2, queue_size=2), worker_version="12.0.0")

    known_good = AppStateStore().load().last_known_good_settings
    assert known_good is not None
    assert known_good.source is KnownGoodSource.BENCHMARK
    assert known_good.config_snapshot["max_threads"] == 2


class _FakePopen:
    """A stand-in for a benchmark subprocess: runs for two polls, then exits cleanly."""

    def __init__(self) -> None:
        self.pid = 4321
        self._poll_results = [None, None, 0]
        self._poll_index = 0

    def poll(self) -> int | None:
        result = self._poll_results[min(self._poll_index, len(self._poll_results) - 1)]
        self._poll_index += 1
        return result

    def terminate(self) -> None:
        pass

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        pass


def test_supervisor_tails_progress_and_finalizes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The supervisor folds tailed events into its run state and finishes when the subprocess exits."""

    def _fake_popen(command: object, stdout: object = None, stderr: object = None) -> _FakePopen:
        return _FakePopen()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)

    supervisor = BenchmarkSupervisor(
        config_path=tmp_path / "bridgeData.yaml",
        results_root=tmp_path / "results",
    )
    supervisor.start(BenchmarkOptions(process_mode="fake"))
    assert supervisor.status is BenchmarkSupervisorStatus.RUNNING
    out_dir = supervisor.out_dir
    assert out_dir is not None

    sink = JsonlProgressSink(out_dir / PROGRESS_FILENAME)
    for event in _ramp_events():
        sink.emit(event)  # type: ignore[arg-type]

    for _ in range(5):
        supervisor.tick()
        if supervisor.status is not BenchmarkSupervisorStatus.RUNNING:
            break

    assert supervisor.status is BenchmarkSupervisorStatus.FINISHED
    assert supervisor.run_state.finished is True
    assert supervisor.run_state.run_id == "run1"
