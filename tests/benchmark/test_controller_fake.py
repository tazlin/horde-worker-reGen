"""Controller tests: fake-mode ramps, crash capture, skip rules, and resume."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from horde_worker_regen.benchmark import controller as controller_mod
from horde_worker_regen.benchmark.controller import (
    BenchmarkController,
    _WarmSessionDriver,
    _weights_root_has_checkpoint,
)
from horde_worker_regen.benchmark.enums import BenchAxis, BenchStage
from horde_worker_regen.benchmark.ladder import LadderOptions, RampLevel, build_default_ladder
from horde_worker_regen.benchmark.report import MachineInfo
from horde_worker_regen.benchmark.scenarios import CannedImageJobSpec, ScenarioSpec


def _mini_ladder(jobs: int = 2) -> list[RampLevel]:
    """A single-level sd15 ladder for fast fake-mode runs."""
    return build_default_ladder(
        LadderOptions(
            tiers=["sd15"],
            jobs_per_level=jobs,
            include_concurrency=False,
            include_features=False,
            include_alchemy=False,
        ),
    )


def test_build_plan_rows_one_per_level_all_run_in_fake(tmp_path: Path) -> None:
    """In fake mode no resource gate applies, so every level's plan row is a RUN."""
    controller = BenchmarkController(_mini_ladder(), tmp_path, process_mode="fake", warm=False, validate=False)
    rows = controller.build_plan_rows(MachineInfo())
    assert len(rows) == len(_mini_ladder())
    assert all(row.will_run for row in rows)
    assert all(not row.verdict for row in rows)


def test_build_plan_rows_marks_only_level_others_not_selected(tmp_path: Path) -> None:
    """--only-level keeps one level RUN and marks the rest as not selected (without a resource verdict)."""
    ladder = build_default_ladder(LadderOptions(tiers=["sd15"], include_features=False, include_alchemy=False))
    assert len(ladder) > 1  # the concurrency stage gives us more than the baseline to deselect
    controller = BenchmarkController(ladder, tmp_path, process_mode="fake", only_level=ladder[0].id)
    rows = controller.build_plan_rows(MachineInfo())
    running = [row for row in rows if row.will_run]
    assert [row.level_id for row in running] == [ladder[0].id]
    assert any("not selected" in row.verdict for row in rows)


def test_weights_root_scan_finds_a_checkpoint(tmp_path: Path) -> None:
    """The bounded scan returns True on the first checkpoint, regardless of suffix or nesting depth."""
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    (nested / "model.safetensors").write_bytes(b"x")
    assert _weights_root_has_checkpoint(tmp_path) is True


def test_weights_root_scan_reports_empty_root(tmp_path: Path) -> None:
    """An empty (or checkpoint-free) root scans to completion and returns False, so the level is skipped."""
    (tmp_path / "notes.txt").write_text("no weights here", encoding="utf-8")
    assert _weights_root_has_checkpoint(tmp_path) is False


def test_weights_root_scan_fails_open_when_budget_exhausted(tmp_path: Path) -> None:
    """A zero-length budget yields None (inconclusive) so the caller fails open rather than skipping."""
    (tmp_path / "sub").mkdir()
    assert _weights_root_has_checkpoint(tmp_path, budget_seconds=0.0) is None


def test_warm_driver_skipped_when_every_level_fails_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no fixed level can pass pre-flight, the warm worker is never booted (no wasted startup).

    This is the disk-starved / no-checkpoints-on-disk case: previously the controller paid the full
    (minutes-long) worker boot only to skip every level anyway.
    """
    controller = BenchmarkController(_mini_ladder(), tmp_path, process_mode="fake", warm=True, validate=False)
    monkeypatch.setattr(BenchmarkController, "_pre_flight_skip_reason", lambda self, level, machine: "no disk")

    def _must_not_boot(*args: object, **kwargs: object) -> object:
        raise AssertionError("the warm worker must not boot when every fixed level fails pre-flight")

    monkeypatch.setattr(controller_mod, "_WarmSessionDriver", _must_not_boot)

    controller._maybe_start_warm_driver(MachineInfo())
    assert controller._warm_driver is None


def test_warm_driver_start_emits_heartbeats_during_slow_boot() -> None:
    """A boot slower than the heartbeat interval republishes startup-phase heartbeats, then completes.

    The boot runs on the driver's loop thread; the controller thread polls and surfaces the otherwise
    silent window. This guards the visibility fix for healthy machines, where the real boot (hordelib
    init plus the first model load) is the tens-of-seconds-to-minutes dark window.
    """
    import asyncio
    import threading

    class _SlowSession:
        async def __aenter__(self) -> _SlowSession:
            await asyncio.sleep(0.25)
            return self

    driver = object.__new__(_WarmSessionDriver)
    driver._session = _SlowSession()  # type: ignore[assignment]  # white-box: only __aenter__ is exercised
    driver._loop = asyncio.new_event_loop()
    driver._thread = threading.Thread(target=driver._loop.run_forever, name="test-warm-loop", daemon=True)

    beats: list[float] = []
    try:
        driver.start(on_heartbeat=beats.append, heartbeat_interval=0.05)
    finally:
        driver._loop.call_soon_threadsafe(driver._loop.stop)
        driver._thread.join(timeout=5.0)

    assert beats, "a boot slower than the heartbeat interval must emit at least one heartbeat"
    assert all(elapsed >= 0.0 for elapsed in beats)


@pytest.mark.e2e
def test_fake_mode_ramp_end_to_end(tmp_path: Path) -> None:
    """A fake-mode ramp passes its level, records a tier baseline, and writes the reports."""
    ladder = _mini_ladder()
    controller = BenchmarkController(ladder, tmp_path, process_mode="fake")
    report = controller.run()

    assert len(report.levels) == 1
    level_report = report.levels[0]
    assert level_report.outcome == "passed", level_report.reasons
    assert level_report.stats is not None
    # The fake processes emit synthetic sampling stats, so a baseline must be recorded.
    assert "sd15" in report.tier_baselines_its

    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / f"level_{ladder[0].id}.json").exists()
    assert (tmp_path / f"level_{ladder[0].id}.log").exists()
    assert "Deliberate" in report.suggested_bridge_data.models_to_load


@pytest.mark.e2e
def test_warm_mode_runs_fixed_levels_without_subprocesses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A warm-mode ramp runs its fixed-scenario levels on one reused worker, spawning no subprocess."""
    ladder = build_default_ladder(
        LadderOptions(
            tiers=["sd15"],
            jobs_per_level=2,
            include_concurrency=True,
            include_features=False,
            include_alchemy=False,
        ),
    )
    fixed_levels = [level for level in ladder if level.scenario.soak_seconds is None]
    assert len(fixed_levels) >= 2, "need multiple fixed levels to exercise warm reuse"

    # Assert the warm path never takes the controller's per-level subprocess seam, rather than patching
    # subprocess.Popen wholesale: the warm worker legitimately shells out to git during startup and status
    # reporting (logfire's service_version, runtime_version()'s dev suffix), so breaking Popen sabotages the
    # worker itself instead of detecting a per-level level_runner spawn.
    def _fail_if_called(self: BenchmarkController, level: object, command: object) -> tuple[int, bool]:
        raise AssertionError("warm mode must not spawn per-level subprocesses")

    monkeypatch.setattr(BenchmarkController, "_run_level_subprocess", _fail_if_called)

    controller = BenchmarkController(ladder, tmp_path, process_mode="fake", warm=True, validate=False)
    report = controller.run()

    passed = [level for level in report.levels if level.outcome == "passed"]
    assert len(passed) >= 2, [(level.level.id, level.outcome, level.reasons) for level in report.levels]
    assert (tmp_path / "report.json").exists()


def test_warm_mode_streams_live_progress(tmp_path: Path) -> None:
    """A warm-mode level republishes the reused worker's live metrics as LevelProgress while it runs.

    This is the warm-path equivalent of the subprocess path's live-file streaming: without it the live
    card sits frozen between LevelStarted and LevelFinished for the whole level.
    """
    from horde_worker_regen.benchmark.progress_channel import (
        BenchmarkProgressEvent,
        LevelProgress,
        ProgressSink,
    )

    class _RecordingSink(ProgressSink):
        def __init__(self) -> None:
            self.events: list[BenchmarkProgressEvent] = []

        def emit(self, event: BenchmarkProgressEvent) -> None:
            self.events.append(event)

    # More jobs per level => the level runs long enough for at least one live sample to land.
    ladder = build_default_ladder(
        LadderOptions(
            tiers=["sd15"],
            jobs_per_level=6,
            include_concurrency=False,
            include_features=False,
            include_alchemy=False,
        ),
    )
    sink = _RecordingSink()
    BenchmarkController(ladder, tmp_path, process_mode="fake", warm=True, validate=False, progress_sink=sink).run()

    progress = [event for event in sink.events if isinstance(event, LevelProgress)]
    assert progress, "warm mode emitted no LevelProgress events; the live card would never update"
    # The streamed metrics advance: at least one sample reports jobs actually completing.
    assert any(event.jobs_completed > 0 for event in progress)


@pytest.mark.e2e
def test_validation_soak_runs_after_ramp(tmp_path: Path) -> None:
    """After the ramp, a stage-V soak runs the synthesized config under sustained generated load."""
    ladder = _mini_ladder()
    controller = BenchmarkController(ladder, tmp_path, process_mode="fake", validate=True, soak_seconds=18.0)
    report = controller.run()

    validation = [level for level in report.levels if level.level.stage == "V"]
    assert len(validation) == 1
    soak = validation[0]
    assert soak.level.axis == "validation"
    # The generating source fed the worker for the soak period, so real jobs flowed through.
    assert soak.stats is not None
    assert soak.stats.num_jobs_completed >= 1, soak.reasons
    assert soak.stats.num_jobs_faulted == 0

    report_md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Validation (sustained load)" in report_md
    assert (tmp_path / f"level_{soak.level.id}.json").exists()


@pytest.mark.e2e
def test_no_validate_skips_soak(tmp_path: Path) -> None:
    """`validate=False` produces no stage-V level."""
    ladder = _mini_ladder()
    report = BenchmarkController(ladder, tmp_path, process_mode="fake", validate=False).run()
    assert not any(level.level.stage == "V" for level in report.levels)


@pytest.mark.e2e
def test_failed_axis_skips_higher_rungs(tmp_path: Path) -> None:
    """A failing level stops higher rungs on its axis; the ramp itself continues."""
    scenario = ScenarioSpec(
        name="never-arrives",
        image_jobs=[CannedImageJobSpec(count=2)],
        # Steady arrival at 0.1 jobs/min: the first job is available immediately, but the
        # second only "arrives" after 10 minutes, far beyond the 8s level timeout, so the
        # level can never complete all of its expected jobs and is guaranteed to time out.
        arrival_kind="steady",
        arrival_rate_per_minute=0.1,
    )
    base = _mini_ladder()[0]
    failing = base.model_copy(
        update={
            "id": "B-sd15-slow1",
            "stage": BenchStage.CONCURRENCY,
            "axis": BenchAxis.BATCH,
            "rung": 1,
            "scenario": scenario,
            "timeout_seconds": 8.0,
            "establishes_tier_baseline": False,
        },
    )
    dependent = failing.model_copy(update={"id": "B-sd15-slow2", "rung": 2})
    ladder = [*_mini_ladder(), failing, dependent]

    controller = BenchmarkController(ladder, tmp_path, process_mode="fake")
    report = controller.run()

    outcomes = {level.level.id: level.outcome for level in report.levels}
    assert outcomes[ladder[0].id] == "passed"
    assert outcomes["B-sd15-slow1"] == "failed"
    assert outcomes["B-sd15-slow2"] == "skipped"
    skip_report = next(level for level in report.levels if level.level.id == "B-sd15-slow2")
    assert any("axis" in reason for reason in skip_report.reasons)


def test_crashed_subprocess_recorded_and_ramp_survives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A level subprocess that dies without a result is a 'crashed' finding, not a ramp abort."""

    def _fake_subprocess(self: BenchmarkController, level: object, command: object) -> tuple[int, bool]:
        return 139, False

    monkeypatch.setattr(BenchmarkController, "_run_level_subprocess", _fake_subprocess)

    ladder = _mini_ladder()
    controller = BenchmarkController(ladder, tmp_path, process_mode="fake")
    report = controller.run()

    assert report.levels[0].outcome == "crashed"
    assert any(finding.kind == "crash" for finding in report.levels[0].findings)
    assert (tmp_path / "report.md").exists()


def test_catastrophic_level_aborts_whole_ramp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A hung/crashed level aborts the entire ramp: later levels (even in other tiers) are skipped, not run.

    The worker stack is shared across every level, so a fundamental failure (the observed broken-dependency
    crash-on-start, or a wedged startup) would repeat on every later level and burn the full ramp's
    timeouts doing nothing. The first catastrophic outcome must short-circuit the rest.
    """
    calls = {"n": 0}

    def _hung_subprocess(self: BenchmarkController, level: object, command: object) -> tuple[int, bool]:
        calls["n"] += 1
        return -1, True  # every level "hangs"

    monkeypatch.setattr(BenchmarkController, "_run_level_subprocess", _hung_subprocess)

    ladder = build_default_ladder(
        LadderOptions(
            tiers=["sd15", "sdxl"],
            jobs_per_level=2,
            include_concurrency=False,
            include_features=False,
            include_alchemy=False,
        ),
    )
    assert len(ladder) >= 2, "need a second (different-tier) level to prove the abort skips it"

    controller = BenchmarkController(ladder, tmp_path, process_mode="fake", validate=False)
    report = controller.run()

    assert report.levels[0].outcome == "crashed_hang"
    # Only the first level ran; the rest were skipped by the abort, not executed.
    assert calls["n"] == 1, "only the first (catastrophic) level should have run"
    later = report.levels[1:]
    assert later and all(level.outcome == "skipped" for level in later)
    assert all(any("aborted" in reason for reason in level.reasons) for level in later)


def test_abort_on_catastrophe_can_be_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the abort disabled, the ramp runs every level despite a catastrophic first level."""
    calls = {"n": 0}

    def _hung_subprocess(self: BenchmarkController, level: object, command: object) -> tuple[int, bool]:
        calls["n"] += 1
        return -1, True

    monkeypatch.setattr(BenchmarkController, "_run_level_subprocess", _hung_subprocess)

    ladder = build_default_ladder(
        LadderOptions(
            tiers=["sd15", "sdxl"],
            jobs_per_level=2,
            include_concurrency=False,
            include_features=False,
            include_alchemy=False,
        ),
    )

    controller = BenchmarkController(
        ladder,
        tmp_path,
        process_mode="fake",
        validate=False,
        abort_on_catastrophe=False,
    )
    report = controller.run()

    # Both tier baselines run (different tiers, so the tier-skip rule never engages); none skipped-by-abort.
    assert calls["n"] == len(ladder)
    assert all(level.outcome == "crashed_hang" for level in report.levels)


def test_hung_subprocess_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A level subprocess that exceeds its timeout is killed and recorded as a hang."""

    def _fake_subprocess(self: BenchmarkController, level: object, command: object) -> tuple[int, bool]:
        return -1, True

    monkeypatch.setattr(BenchmarkController, "_run_level_subprocess", _fake_subprocess)

    ladder = _mini_ladder()
    controller = BenchmarkController(ladder, tmp_path, process_mode="fake")
    report = controller.run()

    assert report.levels[0].outcome == "crashed_hang"
    assert any(finding.kind == "hang" for finding in report.levels[0].findings)


@pytest.mark.e2e
def test_resume_reuses_existing_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--resume evaluates stored results instead of re-running subprocesses."""
    ladder = _mini_ladder()
    first_report = BenchmarkController(ladder, tmp_path, process_mode="fake").run()
    assert first_report.levels[0].outcome == "passed"

    def _fail_if_called(*args: object, **kwargs: object) -> subprocess.Popen:
        raise AssertionError("resume must not re-run level subprocesses")

    monkeypatch.setattr(subprocess, "Popen", _fail_if_called)

    resumed_report = BenchmarkController(ladder, tmp_path, process_mode="fake", resume=True).run()
    assert resumed_report.levels[0].outcome == "passed"
    assert "sd15" in resumed_report.tier_baselines_its


def test_skip_downloads_skips_networked_levels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--skip-downloads pre-flight-skips network levels without running anything."""

    def _fail_if_called(*args: object, **kwargs: object) -> subprocess.Popen:
        raise AssertionError("skipped levels must not spawn subprocesses")

    monkeypatch.setattr(subprocess, "Popen", _fail_if_called)

    ladder = build_default_ladder(
        LadderOptions(
            tiers=["sd15"],
            include_concurrency=False,
            include_features=False,
            include_alchemy=False,
            include_downloads=True,
        ),
    )
    download_levels = [level for level in ladder if level.requires_network]
    assert download_levels

    controller = BenchmarkController(download_levels, tmp_path, process_mode="fake", skip_downloads=True)
    report = controller.run()
    assert all(level.outcome == "skipped" for level in report.levels)
