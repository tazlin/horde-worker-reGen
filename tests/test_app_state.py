"""Unit tests for the durable app-state store: round-trips, tolerant reads, atomic writes, staleness."""

from __future__ import annotations

from pathlib import Path

import pytest

from horde_worker_regen.app_state import (
    AppStateStore,
    BenchmarkAvailability,
    BenchmarkRecord,
    KnownGoodSettings,
    KnownGoodSource,
    OnboardingChoice,
    OnboardingState,
    WorkerAppState,
    WorkerRunRecord,
    benchmark_status_summary,
    build_benchmark_record,
    compute_config_digest,
    default_app_state_path,
    is_benchmark_stale,
    should_prompt_onboarding,
)


def _store(tmp_path: Path) -> AppStateStore:
    return AppStateStore(tmp_path / ".horde_worker_regen" / "state.json")


def _benchmark_record(*, worker_version: str = "12.0.0", run_id: str = "20260613-120000") -> BenchmarkRecord:
    return BenchmarkRecord(
        run_id=run_id,
        results_dir=f"benchmark_results/{run_id}",
        created_at=1.0,
        worker_version=worker_version,
        levels_passed=3,
        levels_total=4,
        gpu_name="Test GPU",
        suggested_bridge_data={"max_threads": 2, "queue_size": 2},
    )


def test_default_path_is_grouped_dir_in_cwd() -> None:
    """The default state file lives in the grouped .horde_worker_regen dir under the working directory."""
    path = default_app_state_path()
    assert path.name == "state.json"
    assert path.parent.name == ".horde_worker_regen"
    assert path.parent.parent == Path.cwd()


def test_load_missing_returns_fresh_state(tmp_path: Path) -> None:
    """Loading when no file exists yields a default state, not an error."""
    state = _store(tmp_path).load()
    assert isinstance(state, WorkerAppState)
    assert state.last_benchmark is None
    assert state.worker_version_last_ran is None


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    """A saved state round-trips through the JSON file unchanged."""
    store = _store(tmp_path)
    state = WorkerAppState(worker_version_last_ran="12.0.0", last_benchmark=_benchmark_record())
    store.save(state)

    restored = store.load()
    assert restored.worker_version_last_ran == "12.0.0"
    assert restored.last_benchmark is not None
    assert restored.last_benchmark.run_id == "20260613-120000"
    assert restored.last_benchmark.suggested_bridge_data["max_threads"] == 2


def test_corrupt_file_recovers_to_fresh_state(tmp_path: Path) -> None:
    """A garbage state file is tolerated: load returns a fresh state instead of raising."""
    store = _store(tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{not valid json", encoding="utf-8")

    state = store.load()
    assert state.last_benchmark is None


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path: Path) -> None:
    """A successful save leaves exactly the state file and no stray temp files behind."""
    store = _store(tmp_path)
    store.save(WorkerAppState())

    state_dir = store.path.parent
    assert store.path.exists()
    assert list(state_dir.glob("*.tmp")) == []


def test_record_mutators_preserve_unrelated_fields(tmp_path: Path) -> None:
    """Each mutator load-modify-saves, so writing one section never clobbers another."""
    store = _store(tmp_path)

    store.record_worker_started(worker_version="12.0.0")
    store.record_benchmark(_benchmark_record())
    store.record_worker_finished(
        WorkerRunRecord(
            started_at=0.0,
            ended_at=10.0,
            duration_seconds=10.0,
            worker_version="12.0.0",
            jobs_submitted=5,
        ),
    )
    store.record_onboarding_choice(OnboardingChoice.DECLINED)
    store.record_known_good(
        KnownGoodSettings(
            config_digest="abc",
            config_snapshot={"max_threads": 1},
            validated_at=1.0,
            worker_version="12.0.0",
            source=KnownGoodSource.CLEAN_RUN,
        ),
    )

    state = store.load()
    assert state.worker_version_last_ran == "12.0.0"
    assert state.last_benchmark is not None and state.last_benchmark.run_id == "20260613-120000"
    assert state.last_worker_run is not None and state.last_worker_run.jobs_submitted == 5
    assert state.onboarding.benchmark_prompt_choice is OnboardingChoice.DECLINED
    assert state.onboarding.prompt_last_shown_at is not None
    assert state.last_known_good_settings is not None
    assert state.last_known_good_settings.source is KnownGoodSource.CLEAN_RUN


def test_auto_start_worker_defaults_off(tmp_path: Path) -> None:
    """A fresh state has auto-start disabled, so the TUI never starts the worker unprompted."""
    assert _store(tmp_path).load().auto_start_worker is False


def test_set_auto_start_worker_round_trips_and_preserves_fields(tmp_path: Path) -> None:
    """Toggling auto-start persists and does not clobber other recorded state."""
    store = _store(tmp_path)
    store.record_worker_started(worker_version="12.0.0")
    store.record_benchmark(_benchmark_record())

    store.set_auto_start_worker(True)

    state = store.load()
    assert state.auto_start_worker is True
    assert state.worker_version_last_ran == "12.0.0"
    assert state.last_benchmark is not None and state.last_benchmark.run_id == "20260613-120000"

    store.set_auto_start_worker(False)
    assert store.load().auto_start_worker is False


def test_setup_complete_defaults_off(tmp_path: Path) -> None:
    """A fresh state has setup-complete off, so a new install runs the wizard."""
    assert _store(tmp_path).load().setup_complete is False


def test_set_setup_complete_round_trips_and_preserves_fields(tmp_path: Path) -> None:
    """Marking setup complete persists and does not clobber other recorded state."""
    store = _store(tmp_path)
    store.record_worker_started(worker_version="12.0.0")
    store.set_auto_start_worker(True)

    store.set_setup_complete(True)

    state = store.load()
    assert state.setup_complete is True
    assert state.auto_start_worker is True
    assert state.worker_version_last_ran == "12.0.0"


def test_old_state_file_without_auto_start_loads_with_default(tmp_path: Path) -> None:
    """A state file written before the auto-start field still loads, defaulting it off."""
    store = _store(tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text('{"schema_version": 1, "worker_version_last_ran": "11.0.0"}', encoding="utf-8")

    state = store.load()
    assert state.auto_start_worker is False
    assert state.worker_version_last_ran == "11.0.0"


def test_is_benchmark_stale_tracks_version() -> None:
    """A benchmark is stale when absent or produced by a different worker version."""
    empty = WorkerAppState()
    assert is_benchmark_stale(empty, current_version="12.0.0") is True

    matched = WorkerAppState(last_benchmark=_benchmark_record(worker_version="12.0.0"))
    assert is_benchmark_stale(matched, current_version="12.0.0") is False
    assert is_benchmark_stale(matched, current_version="12.1.0") is True


def test_benchmark_status_summary_classifies() -> None:
    """The status summary distinguishes none / stale / current for the running version."""
    assert benchmark_status_summary(WorkerAppState(), current_version="12.0.0") is BenchmarkAvailability.NONE

    state = WorkerAppState(last_benchmark=_benchmark_record(worker_version="12.0.0"))
    assert benchmark_status_summary(state, current_version="12.0.0") is BenchmarkAvailability.CURRENT
    assert benchmark_status_summary(state, current_version="12.1.0") is BenchmarkAvailability.STALE


def test_compute_config_digest_is_order_independent() -> None:
    """The config digest depends on content, not key order."""
    digest_one = compute_config_digest({"max_threads": 2, "queue_size": 1})
    digest_two = compute_config_digest({"queue_size": 1, "max_threads": 2})
    assert digest_one == digest_two
    assert compute_config_digest({"max_threads": 3}) != digest_one


def test_should_prompt_onboarding() -> None:
    """The first-run prompt shows only when no current benchmark exists and the user hasn't declined."""
    assert should_prompt_onboarding(WorkerAppState(), current_version="12.0.0") is True

    current = WorkerAppState(last_benchmark=_benchmark_record(worker_version="12.0.0"))
    assert should_prompt_onboarding(current, current_version="12.0.0") is False
    assert should_prompt_onboarding(current, current_version="12.1.0") is True

    declined = WorkerAppState(onboarding=OnboardingState(benchmark_prompt_choice=OnboardingChoice.DECLINED))
    assert should_prompt_onboarding(declined, current_version="12.0.0") is False


def test_log_benchmark_hint_emits_when_no_benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The headless hint fires when no current benchmark exists for the running version."""
    from loguru import logger

    import horde_worker_regen.run_worker as run_worker

    monkeypatch.chdir(tmp_path)
    captured: list[str] = []
    sink_id = logger.add(lambda message: captured.append(str(message)), level="INFO")
    try:
        run_worker._log_benchmark_hint()
    finally:
        logger.remove(sink_id)
    assert any("benchmark" in line.lower() for line in captured)


def test_log_benchmark_hint_silent_when_current(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The headless hint stays silent when a benchmark for the running version is on record."""
    from loguru import logger

    import horde_worker_regen.run_worker as run_worker
    from horde_worker_regen import __version__

    monkeypatch.chdir(tmp_path)
    AppStateStore().record_benchmark(_benchmark_record(worker_version=__version__))

    captured: list[str] = []
    sink_id = logger.add(lambda message: captured.append(str(message)), level="INFO")
    try:
        run_worker._log_benchmark_hint()
    finally:
        logger.remove(sink_id)
    assert not any("benchmark" in line.lower() for line in captured)


def test_build_benchmark_record_from_report() -> None:
    """The report-to-record adapter carries the report's stamps and flattens the suggestion."""
    from horde_worker_regen.benchmark.report import BenchmarkReport, MachineInfo, SuggestedBridgeData

    report = BenchmarkReport(
        run_id="20260613-130000",
        worker_version="9.9.9",
        machine=MachineInfo(gpu_name="Adapter GPU"),
        suggested_bridge_data=SuggestedBridgeData(max_threads=2, queue_size=2),
    )

    record = build_benchmark_record(report, results_dir="benchmark_results/20260613-130000")
    assert record.run_id == "20260613-130000"
    assert record.worker_version == "9.9.9"
    assert record.gpu_name == "Adapter GPU"
    assert record.levels_total == 0
    assert record.suggested_bridge_data["max_threads"] == 2
