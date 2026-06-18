"""Observability/robustness guards for the warm benchmark worker's silent-startup failure mode.

These exercise the diagnostics the warm path emits when a reused worker never becomes ready (or a level
makes no progress) *without spawning real child processes*: spawning is exactly what is slow/wedged on
the machines this matters for, so the tests drive the harness against a stub manager and assert the
logging/diagnostic behaviour directly. The one spawn-target test runs its target in-process.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from loguru import logger

from horde_worker_regen.harness import (
    HarnessResult,
    WarmHarnessSession,
    _summarize_worker_processes,
)
from horde_worker_regen.process_management._canned_scenarios import make_canned_job
from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.messages import HordeProcessState


class _StubMpProcess:
    """Minimal ``BaseProcess`` stand-in exposing only the liveness probe the diagnostics read."""

    def __init__(self, *, alive: bool) -> None:
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


class _StubProcessInfo:
    def __init__(self, *, process_id: int, process_type: HordeProcessType, alive: bool) -> None:
        self.process_id = process_id
        self.process_type = process_type
        self.last_process_state = HordeProcessState.PROCESS_STARTING
        self.mp_process = _StubMpProcess(alive=alive)


class _StubProcessMap:
    def __init__(self, infos: list[_StubProcessInfo]) -> None:
        self._infos = infos

    def values(self) -> list[_StubProcessInfo]:
        return self._infos

    def num_inference_processes(self) -> int:
        return sum(1 for i in self._infos if i.process_type == HordeProcessType.INFERENCE)

    def num_safety_processes(self) -> int:
        return sum(1 for i in self._infos if i.process_type == HordeProcessType.SAFETY)

    def get_first_available_inference_process(self) -> None:
        # Never ready: the whole point of these tests is the not-ready/dead worker.
        return None


class _StubJobTracker:
    total_num_completed_jobs = 0
    num_jobs_faulted = 0
    jobs_lookup: dict[str, object] = {}


class _StubAlchemy:
    num_canned_forms_completed = 0
    num_canned_forms_faulted = 0


class _StubManager:
    """A process manager stand-in whose worker never makes progress, so a level always times out."""

    def __init__(self, infos: list[_StubProcessInfo]) -> None:
        self._process_map = _StubProcessMap(infos)
        self._job_tracker = _StubJobTracker()
        self._alchemy_coordinator = _StubAlchemy()
        self.set_concurrency_calls: list[tuple[int | None, int | None]] = []

    def _apply_set_concurrency(self, target_threads: int | None, target_processes: int | None) -> None:
        self.set_concurrency_calls.append((target_threads, target_processes))

    def install_benchmark_scenario(self, *, jobs: object, alchemy_forms: object = None) -> None:
        return None

    async def receive_and_handle_process_messages(self) -> None:
        return None

    def get_run_metrics_snapshot(self) -> None:
        return None


def _make_session(infos: list[_StubProcessInfo]) -> WarmHarnessSession:
    session = WarmHarnessSession(process_mode="fake", model_names=["Deliberate"], max_threads_ceiling=1)
    session._manager = _StubManager(infos)  # type: ignore[assignment]  # white-box: stub stands in for the manager
    return session


def test_summarize_worker_processes_flags_dead_children() -> None:
    """The summary tags an exited OS process with ``/DEAD`` so a startup death is obvious in the log."""
    manager = _StubManager(
        [
            _StubProcessInfo(process_id=0, process_type=HordeProcessType.INFERENCE, alive=False),
            _StubProcessInfo(process_id=1, process_type=HordeProcessType.SAFETY, alive=True),
        ],
    )
    summary = _summarize_worker_processes(manager)  # type: ignore[arg-type]
    assert "inference#0=PROCESS_STARTING/DEAD" in summary
    assert "safety#1=PROCESS_STARTING" in summary
    assert "safety#1=PROCESS_STARTING/DEAD" not in summary


async def test_wait_for_inference_ready_fast_fails_when_all_children_dead() -> None:
    """A readiness wait abandons early (not after the full timeout) once every inference child has exited."""
    session = _make_session([_StubProcessInfo(process_id=0, process_type=HordeProcessType.INFERENCE, alive=False)])

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m.record["message"]), level="DEBUG")
    try:
        started = time.monotonic()
        await session._wait_for_inference_ready(timeout_seconds=30.0)
        elapsed = time.monotonic() - started
    finally:
        logger.remove(sink_id)

    assert elapsed < 5.0, f"fast-fail should not wait the full timeout (took {elapsed:.1f}s)"
    assert any("exited during startup" in message for message in messages)


async def test_warm_run_level_timeout_populates_diagnostics() -> None:
    """A timed-out warm level returns non-empty diagnostics, including the per-process state snapshot."""
    session = _make_session([_StubProcessInfo(process_id=0, process_type=HordeProcessType.INFERENCE, alive=False)])

    result = await session.run_level(
        jobs=[make_canned_job("Deliberate"), make_canned_job("Deliberate")],
        threads=1,
        timeout_seconds=0.3,
    )

    assert isinstance(result, HarnessResult)
    assert result.timed_out is True
    assert result.num_jobs_completed == 0
    assert result.diagnostics, "a timed-out warm level must explain itself with diagnostics"
    # The safety process is absent in this stub, so that deterministic diagnostic must surface...
    assert any("No safety processes were started" in entry for entry in result.diagnostics)
    # ...alongside the per-process state snapshot showing the dead inference child.
    assert any("process states at timeout" in entry and "DEAD" in entry for entry in result.diagnostics)


async def test_warm_level_abandons_dead_worker_before_full_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A level draining against an all-dead worker abandons after the grace, not after the full timeout."""
    import horde_worker_regen.harness as harness_mod

    # Shrink the recovery grace so the test asserts the fast-fail without a real 15s wait.
    monkeypatch.setattr(harness_mod, "_WARM_DEAD_WORKER_GRACE_SECONDS", 0.2)
    session = _make_session([_StubProcessInfo(process_id=0, process_type=HordeProcessType.INFERENCE, alive=False)])

    started = time.monotonic()
    result = await session.run_level(
        jobs=[make_canned_job("Deliberate")],
        threads=1,
        timeout_seconds=60.0,  # would hang for a minute without the dead-worker fast-fail
    )
    elapsed = time.monotonic() - started

    assert result.timed_out is True
    assert elapsed < 10.0, f"dead-worker fast-fail should abandon quickly (took {elapsed:.1f}s)"
    assert result.diagnostics


def test_fake_inference_entry_point_records_startup_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fake inference child that dies during construction leaves a discoverable startup-crash log.

    Previously the fake entry points had no crash capture (unlike the real ones), so a startup death
    was completely silent and the warm worker just wedged until the per-level timeout.
    """
    from horde_worker_regen.process_management import fake_worker_processes as fwp

    monkeypatch.chdir(tmp_path)  # crash-capture writes under ./logs; isolate it here
    # Don't disturb the global loguru sink state from inside the test process.
    monkeypatch.setattr(fwp.logger, "remove", lambda *a, **k: None)

    class _BoomProcess:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated fake-child startup crash")

    monkeypatch.setattr(fwp, "FakeInferenceProcess", _BoomProcess)

    with pytest.raises(RuntimeError, match="simulated fake-child startup crash"):
        fwp.start_fake_inference_process(
            7,
            None,  # type: ignore[arg-type]  # construction raises before any arg is used
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            0,
        )

    crash_log = tmp_path / "logs" / "bridge_fake_inference_7_startup.log"
    assert crash_log.exists(), "the fake child's startup crash must be written to a discoverable log"
    contents = crash_log.read_text(encoding="utf-8")
    assert "simulated fake-child startup crash" in contents
    assert "RuntimeError" in contents
