"""Chaos / fault-injection probes for the process lifecycle (worker -> subprocess layer).

These drive ``ProcessLifecycleManager`` directly with hand-constructed process state to probe
specific failure shapes: a child wedged before its first step, an idle slot stuck while work is
pending, a crash that orphans a held semaphore, and a slot that crash-loops. They assert the
*intended* resilient behaviour, which a subprocess-resiliency overhaul now provides. See
``tests/e2e/test_chaos_e2e.py`` for the full spawned-process counterparts.
"""

from __future__ import annotations

import multiprocessing
import time
from unittest.mock import Mock

import pytest

import horde_worker_regen.process_management.process_lifecycle as process_lifecycle_module
from horde_worker_regen.process_management.action_ledger import LedgerEventType
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.messages import HordeHeartbeatType, HordeProcessState
from horde_worker_regen.process_management.process_lifecycle import (
    CRASH_LOOP_MAX_START_FAILURES,
    CRASH_LOOP_WINDOW_SECONDS,
    ProcessLifecycleManager,
)
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.worker_state import WorkerState

from .conftest import make_mock_process_info, make_test_runtime_config


def _make_plm(*, process_map: ProcessMap | None = None) -> ProcessLifecycleManager:
    """Build a ProcessLifecycleManager with mostly-mocked dependencies (mirrors test_process_lifecycle)."""
    bridge_data = Mock()
    bridge_data.image_models_to_load = ["stable_diffusion"]
    bridge_data.max_threads = 1
    bridge_data.safety_on_gpu = False
    bridge_data.high_memory_mode = False
    bridge_data.very_high_memory_mode = False
    bridge_data.process_timeout = 300
    bridge_data.inference_step_timeout = 15
    bridge_data.preload_timeout = 80
    bridge_data.download_timeout = 120
    bridge_data.post_process_timeout = 60
    bridge_data.max_batch = 1
    bridge_data.exit_on_unhandled_faults = False

    plm = ProcessLifecycleManager(
        ctx=multiprocessing.get_context("spawn"),
        process_map=process_map or ProcessMap({}),
        horde_model_map=Mock(),
        job_tracker=JobTracker(),
        process_message_queue=Mock(),
        inference_semaphore=Mock(),
        disk_lock=Mock(),
        aux_model_lock=Mock(),
        vae_decode_semaphore=Mock(),
        gpu_sampling_lease=Mock(),
        download_bandwidth_semaphore=Mock(),
        runtime_config=make_test_runtime_config(bridge_data=bridge_data),
        max_inference_processes=2,
        max_safety_processes=1,
        amd_gpu=False,
        directml=None,
        abort_callback=Mock(),
        state=WorkerState(),
    )
    # Detection of stuck/idle processes is gated on there being work to do; simulate a busy worker.
    plm._state.last_pop_no_jobs_available = False
    return plm


def _age(process: object, seconds: float = 1000.0) -> None:
    """Push a process's last-seen timestamps into the past so any elapsed-time check trips."""
    past = time.time() - seconds
    process.last_received_timestamp = past  # type: ignore[attr-defined]
    process.last_heartbeat_timestamp = past  # type: ignore[attr-defined]


def test_hung_before_first_step_is_detected() -> None:
    """An inference process wedged in INFERENCE_STARTING before its first step must be replaced.

    A healthy idle peer keeps checking in, so the coarse 'all processes timed out' fallback cannot
    fire; only the heartbeat-aware ``is_stuck_on_inference`` check (now measuring elapsed time live)
    catches the wedge.
    """
    hung = make_mock_process_info(0, model_name="m", state=HordeProcessState.INFERENCE_STARTING)
    hung.last_heartbeat_type = HordeHeartbeatType.OTHER  # never reached an INFERENCE_STEP
    hung.last_heartbeat_percent_complete = None
    _age(hung)
    healthy = make_mock_process_info(1, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)

    plm = _make_plm(process_map=ProcessMap({0: hung, 1: healthy}))
    plm._replace_inference_process = Mock()  # type: ignore[method-assign]

    plm.replace_hung_processes()

    plm._replace_inference_process.assert_called_once_with(hung)


def test_idle_slot_stuck_with_work_pending_is_detected() -> None:
    """A slot stuck WAITING_FOR_JOB while work is pending must eventually be recovered, not ignored."""
    stuck = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
    _age(stuck)
    plm = _make_plm(process_map=ProcessMap({0: stuck}))
    plm._replace_inference_process = Mock()  # type: ignore[method-assign]

    plm.replace_hung_processes()

    plm._replace_inference_process.assert_called_once_with(stuck)


def test_crash_in_postprocessing_releases_inference_semaphore() -> None:
    """A process that crashes while holding the inference semaphore must have it released on replacement.

    INFERENCE_POST_PROCESSING is a state in which the slot can still hold concurrency; if the release
    is keyed only on INFERENCE_STARTING the semaphore leaks and caps throughput forever.
    """
    dead = make_mock_process_info(0, model_name="m", state=HordeProcessState.INFERENCE_POST_PROCESSING)
    plm = _make_plm(process_map=ProcessMap({0: dead}))
    # Avoid touching real OS processes: stub the end/start so only the release logic under test runs.
    plm._end_inference_process = Mock()  # type: ignore[method-assign]
    plm._start_inference_process = Mock()  # type: ignore[method-assign]

    plm._replace_inference_process(dead)

    plm._inference_semaphore.release.assert_called()  # pyrefly: ignore


def test_crash_looping_slot_is_eventually_quarantined() -> None:
    """A slot that dies on every launch must stop being respawned after a few attempts (circuit breaker)."""
    spawn_count = 0

    def _respawn_dead(pid: int) -> None:
        nonlocal spawn_count
        spawn_count += 1
        replacement = make_mock_process_info(pid, model_name=None, state=HordeProcessState.PROCESS_STARTING)
        replacement.mp_process.is_alive.return_value = False  # the replacement is dead too
        replacement.mp_process.exitcode = 1  # pyrefly: ignore
        plm._process_map[pid] = replacement

    dead = make_mock_process_info(0, model_name=None, state=HordeProcessState.PROCESS_STARTING)
    dead.mp_process.is_alive.return_value = False
    dead.mp_process.exitcode = 1  # pyrefly: ignore
    plm = _make_plm(process_map=ProcessMap({0: dead}))
    plm._end_inference_process = Mock()  # type: ignore[method-assign]
    plm._start_inference_process = _respawn_dead  # type: ignore[method-assign]

    # Drive many reap cycles, clearing the recovery debounce so each cycle is free to act.
    for _ in range(8):
        plm._recently_recovered = False
        plm.replace_hung_processes()

    assert spawn_count <= 3, f"slot was respawned {spawn_count} times with no circuit breaker"


def test_replacement_is_recorded_in_action_ledger() -> None:
    """Replacing a crashed slot self-audits the actions taken: release, then replace, in order."""
    dead = make_mock_process_info(0, model_name="m", state=HordeProcessState.INFERENCE_STARTING)
    plm = _make_plm(process_map=ProcessMap({0: dead}))
    plm._end_inference_process = Mock()  # type: ignore[method-assign]
    plm._start_inference_process = Mock()  # type: ignore[method-assign]

    plm._replace_inference_process(dead)

    events = [event.event_type for event in plm.action_ledger.recent(process_id=0, limit=10)]
    assert LedgerEventType.SEMAPHORE_RELEASED in events
    assert LedgerEventType.PROCESS_REPLACED in events
    assert events.index(LedgerEventType.SEMAPHORE_RELEASED) < events.index(LedgerEventType.PROCESS_REPLACED)


def test_quarantine_is_recorded_in_action_ledger() -> None:
    """A crash-looped slot records a PROCESS_QUARANTINED audit event when taken out of the pool."""
    spawn_count = 0

    def _respawn_dead(pid: int) -> None:
        nonlocal spawn_count
        spawn_count += 1
        replacement = make_mock_process_info(pid, model_name=None, state=HordeProcessState.PROCESS_STARTING)
        replacement.mp_process.is_alive.return_value = False
        replacement.mp_process.exitcode = 1  # pyrefly: ignore
        plm._process_map[pid] = replacement

    dead = make_mock_process_info(0, model_name=None, state=HordeProcessState.PROCESS_STARTING)
    dead.mp_process.is_alive.return_value = False
    dead.mp_process.exitcode = 1  # pyrefly: ignore
    plm = _make_plm(process_map=ProcessMap({0: dead}))
    plm._end_inference_process = Mock()  # type: ignore[method-assign]
    plm._start_inference_process = _respawn_dead  # type: ignore[method-assign]

    for _ in range(8):
        plm._recently_recovered = False
        plm.replace_hung_processes()

    events = [event.event_type for event in plm.action_ledger.recent(process_id=0, limit=50)]
    assert LedgerEventType.PROCESS_QUARANTINED in events


def test_slow_crash_on_start_is_quarantined_despite_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deterministic crash-on-start that dies slower than the window can count must still be quarantined.

    This is the empirical shape of the observed benchmark wedge: an inference child crashed during
    ``hordelib.initialise()`` (a broken dependency) every launch, but each crash took ~30-250s, so the
    sliding-window breaker -- which only fires on more than ``CRASH_LOOP_MAX_REPLACEMENTS`` replacements
    *within* ``CRASH_LOOP_WINDOW_SECONDS`` -- could never accumulate enough before the early ones aged
    out. The slot respawned for the full level timeout (~15 min, 0 jobs). The consecutive crash-on-start
    streak quarantines it regardless of how slow each crash is.
    """
    dead = make_mock_process_info(0, model_name=None, state=HordeProcessState.PROCESS_STARTING)
    dead.mp_process.is_alive.return_value = False
    dead.mp_process.exitcode = 1  # pyrefly: ignore
    plm = _make_plm(process_map=ProcessMap({0: dead}))
    plm._end_inference_process = Mock()  # type: ignore[method-assign]
    plm._start_inference_process = Mock()  # type: ignore[method-assign]

    # Space the crashes so that the sliding window can hold at most two at a time: the rate breaker
    # provably cannot fire, isolating the consecutive-start-failure path under test.
    clock = [1000.0]
    spacing = CRASH_LOOP_WINDOW_SECONDS / 2 + 1.0
    monkeypatch.setattr(process_lifecycle_module.time, "time", lambda: clock[0])

    for _ in range(CRASH_LOOP_MAX_START_FAILURES):
        plm._replace_inference_process(dead)
        clock[0] += spacing

    assert 0 in plm.quarantined_inference_slots
    # The window never held more than two replacements, so the rate breaker alone would not have fired.
    assert len(plm._slot_recovery_history.get(0, [])) <= 2


def test_consecutive_start_failures_reset_once_a_slot_reaches_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    """A slot that crashes on start, then successfully initialises, must not carry its old failure streak.

    The crash-on-start breaker counts *consecutive* failures before readiness. Reaching any later state
    proves the slot can initialise, so a single later crash must not tip an old, unrelated streak over
    the quarantine threshold (which would wrongly retire a healthy-but-occasionally-faulting slot).
    """
    clock = [1000.0]
    monkeypatch.setattr(process_lifecycle_module.time, "time", lambda: clock[0])

    dead = make_mock_process_info(0, model_name=None, state=HordeProcessState.PROCESS_STARTING)
    dead.mp_process.is_alive.return_value = False
    dead.mp_process.exitcode = 1  # pyrefly: ignore
    plm = _make_plm(process_map=ProcessMap({0: dead}))
    plm._end_inference_process = Mock()  # type: ignore[method-assign]
    plm._start_inference_process = Mock()  # type: ignore[method-assign]

    # Two crash-on-start failures: one short of the quarantine threshold.
    for _ in range(CRASH_LOOP_MAX_START_FAILURES - 1):
        plm._replace_inference_process(dead)
        clock[0] += 1.0
    assert plm._slot_consecutive_start_failures.get(0) == CRASH_LOOP_MAX_START_FAILURES - 1

    # The slot now comes up healthy (reaches WAITING_FOR_JOB); the watchdog tick clears its streak.
    ready = make_mock_process_info(0, model_name="m", state=HordeProcessState.WAITING_FOR_JOB)
    plm._process_map[0] = ready
    plm.replace_hung_processes()
    assert 0 not in plm._slot_consecutive_start_failures

    # A later, isolated crash-on-start starts a fresh streak rather than tipping into quarantine.
    plm._replace_inference_process(dead)
    assert 0 not in plm.quarantined_inference_slots
    assert plm._slot_consecutive_start_failures.get(0) == 1
