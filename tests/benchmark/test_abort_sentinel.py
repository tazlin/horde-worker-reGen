"""A benchmark run and a live worker in one checkout each watch, write and clear only their own abort sentinel."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest
from loguru import logger

from horde_worker_regen import harness as harness_module
from horde_worker_regen.harness import (
    HarnessConfig,
    HarnessResult,
    WarmHarnessSession,
    _claim_benchmark_abort_sentinel,
)
from horde_worker_regen.process_management.config.worker_state import WorkerState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.lifecycle.shutdown_manager import ShutdownManager
from horde_worker_regen.run_root import (
    ABORT_SENTINEL_NAME,
    BENCHMARK_ABORT_SENTINEL_NAME,
    AbortSentinelKind,
    abort_sentinel_path,
    claim_abort_sentinel,
    claimed_abort_sentinel_kind,
    run_root,
)


@pytest.fixture(autouse=True)
def _restore_claim() -> Iterator[None]:
    """Each test starts as a worker and leaves the process-wide claim as it found it."""
    previous = claim_abort_sentinel(AbortSentinelKind.WORKER)
    yield
    claim_abort_sentinel(previous)


@pytest.fixture
def _harness_run_markers(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Enter the harness run context without reconfiguring telemetry, and detach its log sink afterwards.

    Telemetry configuration replaces every loguru handler in the test process. The harness sink outlives
    the run by design, so a test that arms it removes it here rather than leaving an open file in this
    test's run root for later tests to log into.
    """
    monkeypatch.setattr("horde_worker_regen.telemetry.configure_telemetry", lambda: None)
    yield
    if harness_module._HARNESS_LOG_SINK_ID is not None:
        logger.remove(harness_module._HARNESS_LOG_SINK_ID)
        harness_module._HARNESS_LOG_SINK_ID = None


def _worker_sentinel() -> Path:
    return run_root() / ABORT_SENTINEL_NAME


def _benchmark_sentinel() -> Path:
    return run_root() / BENCHMARK_ABORT_SENTINEL_NAME


def test_the_two_kinds_have_distinct_names() -> None:
    """Distinct names are what keep the two runs apart under one run root."""
    assert ABORT_SENTINEL_NAME != BENCHMARK_ABORT_SENTINEL_NAME
    assert abort_sentinel_path(AbortSentinelKind.WORKER).name == ABORT_SENTINEL_NAME
    assert abort_sentinel_path(AbortSentinelKind.BENCHMARK).name == BENCHMARK_ABORT_SENTINEL_NAME


def test_the_worker_ignores_a_benchmark_sentinel() -> None:
    """The worker's control loop polls ``abort_sentinel_path()``; a benchmark's file does not trip it."""
    _benchmark_sentinel().write_text("")

    assert claimed_abort_sentinel_kind() is AbortSentinelKind.WORKER
    assert not abort_sentinel_path().exists()

    _worker_sentinel().write_text("")
    assert abort_sentinel_path().exists()


def test_the_harness_ignores_a_worker_sentinel_and_stops_on_its_own() -> None:
    """Once the harness claims its sentinel, the worker's ``.abort`` is invisible and its own file is seen."""
    _worker_sentinel().write_text("")

    _claim_benchmark_abort_sentinel()

    assert claimed_abort_sentinel_kind() is AbortSentinelKind.BENCHMARK
    assert not abort_sentinel_path().exists()

    _benchmark_sentinel().write_text("")
    assert abort_sentinel_path().exists()


def test_startup_cleanup_removes_only_the_benchmark_sentinel() -> None:
    """A stale benchmark sentinel is cleared at startup; a live worker's ``.abort`` is left alone."""
    _worker_sentinel().write_text("")
    _benchmark_sentinel().write_text("")

    _claim_benchmark_abort_sentinel()

    assert _worker_sentinel().exists()
    assert not _benchmark_sentinel().exists()


def test_an_abort_under_the_benchmark_claim_writes_the_benchmark_sentinel() -> None:
    """The embedded worker's abort path writes the benchmark's file, which a live worker never polls."""
    _claim_benchmark_abort_sentinel()
    shutdown_manager = ShutdownManager(
        state=WorkerState(),
        job_tracker=JobTracker(),
        process_map=ProcessMap({}),
        process_lifecycle=Mock(),
        main_loop_finished=threading.Event(),
    )
    shutdown_manager.start_timed_shutdown = Mock()

    shutdown_manager.abort()

    assert _benchmark_sentinel().exists()
    assert not _worker_sentinel().exists()


@pytest.mark.usefixtures("_harness_run_markers")
def test_a_harness_run_holds_the_benchmark_claim_and_restores_the_previous_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The claim covers the run and is released afterwards, even when the run raises."""
    seen_during_run: list[AbortSentinelKind] = []

    async def _fake_run(config: HarnessConfig) -> HarnessResult:
        seen_during_run.append(claimed_abort_sentinel_kind())
        raise RuntimeError("run failed")

    monkeypatch.setattr(harness_module, "_run_harness_in_run_context", _fake_run)

    with pytest.raises(RuntimeError, match="run failed"):
        asyncio.run(harness_module.run_harness_async(HarnessConfig(process_mode="fake", skip_api=True)))

    assert seen_during_run == [AbortSentinelKind.BENCHMARK]
    assert claimed_abort_sentinel_kind() is AbortSentinelKind.WORKER


@pytest.mark.usefixtures("_harness_run_markers")
def test_a_raising_harness_run_arms_both_markers_and_releases_only_the_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The log sink and the claim are armed together; the claim is restored and the sink stays for the next run.

    The sink is replaced when the next run re-arms it, so the process holds at most one.
    """
    sink_during_run: list[int | None] = []

    async def _fake_run(config: HarnessConfig) -> HarnessResult:
        sink_during_run.append(harness_module._HARNESS_LOG_SINK_ID)
        raise RuntimeError("run failed")

    monkeypatch.setattr(harness_module, "_run_harness_in_run_context", _fake_run)
    config = HarnessConfig(process_mode="fake", skip_api=True)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="run failed"):
            asyncio.run(harness_module.run_harness_async(config))
        assert claimed_abort_sentinel_kind() is AbortSentinelKind.WORKER
        assert sink_during_run[-1] == harness_module._HARNESS_LOG_SINK_ID

    first_sink, second_sink = sink_during_run
    assert first_sink is not None
    assert second_sink is not None
    assert first_sink != second_sink
    with pytest.raises(ValueError):
        logger.remove(first_sink)


@pytest.mark.usefixtures("_harness_run_markers")
async def test_a_warm_session_that_fails_to_start_restores_the_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure after the run context is entered releases the claim; ``__aexit__`` never runs for it."""

    def _failing_build(self: WarmHarnessSession) -> Mock:
        assert claimed_abort_sentinel_kind() is AbortSentinelKind.BENCHMARK
        raise RuntimeError("build failed")

    monkeypatch.setattr(WarmHarnessSession, "_build_manager", _failing_build)

    session = WarmHarnessSession(process_mode="fake", model_names=["Deliberate"], max_threads_ceiling=1)
    with pytest.raises(RuntimeError, match="build failed"):
        async with session:
            pytest.fail("the session body must not run when start fails")

    assert claimed_abort_sentinel_kind() is AbortSentinelKind.WORKER
    assert session._run_context is None


@pytest.mark.usefixtures("_harness_run_markers")
async def test_a_warm_session_holds_both_markers_until_it_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A warm session enters the same run context on start and restores the claim on close."""
    manager = Mock()

    async def _main_loop() -> None:
        return None

    async def _ready(timeout_seconds: float = 600.0) -> None:
        return None

    manager._main_loop = _main_loop
    monkeypatch.setattr(WarmHarnessSession, "_build_manager", lambda self: manager)
    monkeypatch.setattr(WarmHarnessSession, "_wait_for_inference_ready", lambda self: _ready())
    monkeypatch.setattr(harness_module, "_kill_owned_children_after_run", lambda manager: None)
    _benchmark_sentinel().write_text("")

    session = WarmHarnessSession(process_mode="fake", model_names=["Deliberate"], max_threads_ceiling=1)
    async with session:
        assert claimed_abort_sentinel_kind() is AbortSentinelKind.BENCHMARK
        assert harness_module._HARNESS_LOG_SINK_ID is not None
        assert not _benchmark_sentinel().exists()

    assert claimed_abort_sentinel_kind() is AbortSentinelKind.WORKER
