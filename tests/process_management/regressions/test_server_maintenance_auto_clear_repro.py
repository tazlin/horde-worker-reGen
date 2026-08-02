"""Reproduction and fix for a worker that sat in horde-forced maintenance until a human noticed.

Failure mode:
    The horde force-set maintenance on a worker for "dropping too many jobs". Nothing on either side ever
    lifted it: the horde has no expiry for a forced pause, and the worker's own latch is only cleared by a
    successful pop, which cannot happen while every pop is rejected. The worker also went dark, because the
    rejection is logged once, the periodic status print is suppressed while the latch holds, and the kudos
    loop returns early. Eight hours of silent five-second retries followed, indistinguishable in the log from
    a dead pop loop.

What is pinned here:
    - A maintenance episode counts every rejected pop and remembers when it engaged.
    - The horde's own maintenance reason is what marks the pause as server-forced; any other reason is
      treated as somebody's deliberate choice and left alone.
    - A server-forced pause is cleared by the worker itself on a widening backoff, but only while the pool is
      healthy, no pop pause stands, and nothing has faulted recently.
    - A pause the horde re-imposes shortly after a clear widens the interval, to a cap; a long healthy
      stretch resets it.
    - The episode stays audible: a heartbeat restates it on an interval, with the rejection count and what
      the worker intends to do about it.
    - A successful pop remains the single end-to-end signal that ends the episode.
"""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest
from horde_sdk import RequestErrorResponse
from loguru import logger

from horde_worker_regen.process_management import process_manager as process_manager_module
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    SupervisorCommand,
    SupervisorControlMessage,
)
from horde_worker_regen.process_management.jobs.job_tracker import JobFaultOrigin
from horde_worker_regen.process_management.process_manager import (
    SERVER_MAINTENANCE_CLEAR_BACKOFF_SECONDS,
    SERVER_MAINTENANCE_CLEAR_MAX_INTERVAL_SECONDS,
    SERVER_MAINTENANCE_ESCALATION_DECAY_SECONDS,
    SERVER_MAINTENANCE_FAULT_QUIET_SECONDS,
    SERVER_MAINTENANCE_HEARTBEAT_INTERVAL_SECONDS,
    SERVER_MAINTENANCE_RETRIP_WINDOW_SECONDS,
    HordeWorkerProcessManager,
)
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_process_info,
    make_testable_process_manager,
    track_popped_job_async,
)

_FIRST_INTERVAL, _SECOND_INTERVAL, _THIRD_INTERVAL = SERVER_MAINTENANCE_CLEAR_BACKOFF_SECONDS

_SERVER_FORCED_MESSAGE = "Maintenance mode activated because worker is dropping too many jobs"
"""The reason the horde writes when it pauses a worker itself."""

_OWNER_SET_MESSAGE = "This worker has been put into maintenance mode by its owner"
"""The reason a pause an operator asked for carries."""


def _make_manager(**bridge_overrides: object) -> HordeWorkerProcessManager:
    """A process manager with one inference slot free, so it reads as fit to serve."""
    manager = make_testable_process_manager(**bridge_overrides)
    manager._process_map[0] = make_mock_process_info(0, state=HordeProcessState.WAITING_FOR_JOB)
    assert manager.is_free_inference_process_available() is True
    return manager


def _force_maintenance(manager: HordeWorkerProcessManager, *, message: str = _SERVER_FORCED_MESSAGE) -> None:
    """Reject one pop the way the horde does while it holds this worker in maintenance."""
    manager._job_popper._handle_pop_error_response(RequestErrorResponse(message=message))


def _advance(manager: HordeWorkerProcessManager, seconds: float) -> None:
    """Move every clock the maintenance episode reads ``seconds`` into the past.

    Equivalent to letting ``seconds`` of wall time pass, without sleeping or patching the clock: each
    recorded instant (and each scheduled deadline) is shifted back by the same amount, including the episode
    marker, which the driver compares against the latch to detect a *new* episode.
    """
    state = manager._state
    if state.server_maintenance_latched_at:
        state.server_maintenance_latched_at -= seconds
    if manager._server_maintenance_episode_marker:
        manager._server_maintenance_episode_marker -= seconds
    if manager._server_maintenance_next_clear_at:
        manager._server_maintenance_next_clear_at -= seconds
    if manager._server_maintenance_next_heartbeat_at:
        manager._server_maintenance_next_heartbeat_at -= seconds
    if manager._server_maintenance_last_clear_at:
        manager._server_maintenance_last_clear_at -= seconds
    manager._terminal_fault_history = [(when - seconds, model) for when, model in manager._terminal_fault_history]


class _InlineThread:
    """A ``threading.Thread`` stand-in that runs its target inline on ``start()``.

    The clear is dispatched off the control loop on a daemon thread; running it inline keeps the assertions
    deterministic, with no real background thread to race them.
    """

    def __init__(self, *, target: object, args: tuple[object, ...] = (), **_kwargs: object) -> None:
        self._target = target
        self._args = args

    def start(self) -> None:
        """Run the captured target synchronously."""
        self._target(*self._args)  # type: ignore[operator]


class _ClearRecorder:
    """Record every maintenance flag the worker asks the horde for, without touching the network."""

    def __init__(self, manager: HordeWorkerProcessManager, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[bool] = []
        monkeypatch.setattr(manager, "_set_server_maintenance_safe", self.calls.append)
        monkeypatch.setattr(process_manager_module.threading, "Thread", _InlineThread)


class _LogCapture:
    """Capture loguru messages at or above a level for the duration of a with-block."""

    def __init__(self, level: str = "WARNING") -> None:
        self._level = level
        self.messages: list[str] = []
        self._sink_id: int | None = None

    def __enter__(self) -> _LogCapture:
        self._sink_id = logger.add(lambda message: self.messages.append(message.record["message"]), level=self._level)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._sink_id is not None:
            logger.remove(self._sink_id)

    def containing(self, needle: str) -> list[str]:
        """Return the captured messages containing ``needle``."""
        return [message for message in self.messages if needle in message]


class TestMaintenanceEpisodeIsRecorded:
    """The worker has to know it is in maintenance, since when, and at what cost."""

    def test_the_latch_records_when_it_engaged_and_that_the_horde_forced_it(self) -> None:
        """The horde's own reason marks the episode as server-forced and stamps its start."""
        manager = _make_manager()

        _force_maintenance(manager)

        assert manager._state.last_pop_maintenance_mode is True
        assert manager._state.server_maintenance_forced_by_server is True
        assert manager._state.server_maintenance_latched_at == pytest.approx(time.time(), abs=5.0)
        assert manager._state.server_maintenance_pop_rejections == 1

    def test_an_owner_set_pause_is_not_marked_as_forced(self) -> None:
        """A maintenance reason the horde did not write is treated as somebody's deliberate choice."""
        manager = _make_manager()

        _force_maintenance(manager, message=_OWNER_SET_MESSAGE)

        assert manager._state.last_pop_maintenance_mode is True
        assert manager._state.server_maintenance_forced_by_server is False

    def test_every_rejected_pop_is_counted(self) -> None:
        """The rejection log fires once per episode, so the count is the only measure of the cost."""
        manager = _make_manager()

        for _ in range(4):
            _force_maintenance(manager)

        assert manager._state.server_maintenance_pop_rejections == 4


class TestAutoClearSchedule:
    """A forced pause is lifted by the worker itself, on a widening but never-ending schedule."""

    def test_no_clear_is_attempted_before_the_first_interval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The first attempt waits for the local remedy to have taken effect."""
        manager = _make_manager()
        recorder = _ClearRecorder(manager, monkeypatch)
        _force_maintenance(manager)

        manager._drive_server_maintenance_recovery()
        _advance(manager, _FIRST_INTERVAL - 60)
        manager._drive_server_maintenance_recovery()

        assert recorder.calls == []

    def test_one_clear_is_attempted_once_the_first_interval_elapses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exactly one clear goes to the horde, announced at WARNING, and it is not repeated per tick."""
        manager = _make_manager()
        recorder = _ClearRecorder(manager, monkeypatch)
        _force_maintenance(manager)
        manager._drive_server_maintenance_recovery()
        _advance(manager, _FIRST_INTERVAL + 1)

        with _LogCapture() as logs:
            manager._drive_server_maintenance_recovery()
            manager._drive_server_maintenance_recovery()
            manager._drive_server_maintenance_recovery()

        assert recorder.calls == [False]
        assert logs.containing("attempt 1"), logs.messages

    def test_the_local_latch_is_left_for_a_successful_pop_to_clear(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An accepted API call is not proof the horde is sending work; only a popped job is."""
        manager = _make_manager()
        _ClearRecorder(manager, monkeypatch)
        _force_maintenance(manager)
        manager._drive_server_maintenance_recovery()
        _advance(manager, _FIRST_INTERVAL + 1)

        manager._drive_server_maintenance_recovery()

        assert manager._state.last_pop_maintenance_mode is True

    def test_later_attempts_wait_the_longer_intervals(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The second attempt waits the second interval, and the third the third."""
        manager = _make_manager()
        recorder = _ClearRecorder(manager, monkeypatch)
        _force_maintenance(manager)
        manager._drive_server_maintenance_recovery()

        _advance(manager, _FIRST_INTERVAL + 1)
        manager._drive_server_maintenance_recovery()
        _advance(manager, _SECOND_INTERVAL - 60)
        manager._drive_server_maintenance_recovery()
        assert recorder.calls == [False], "the second attempt must wait the longer interval"

        _advance(manager, 61)
        manager._drive_server_maintenance_recovery()
        assert recorder.calls == [False, False]

        _advance(manager, _THIRD_INTERVAL - 60)
        manager._drive_server_maintenance_recovery()
        assert recorder.calls == [False, False]
        _advance(manager, 61)
        manager._drive_server_maintenance_recovery()
        assert recorder.calls == [False, False, False]

    def test_the_worker_keeps_retrying_indefinitely(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The schedule settles at the longest interval rather than giving up on rejoining the horde."""
        manager = _make_manager()
        recorder = _ClearRecorder(manager, monkeypatch)
        _force_maintenance(manager)
        manager._drive_server_maintenance_recovery()

        for interval in (_FIRST_INTERVAL, _SECOND_INTERVAL, _THIRD_INTERVAL, _THIRD_INTERVAL, _THIRD_INTERVAL):
            _advance(manager, interval + 1)
            manager._drive_server_maintenance_recovery()

        assert recorder.calls == [False] * 5


class TestRetripEscalation:
    """A clear the horde reverses buys a longer wait before the next one."""

    def _clear_and_rejoin(self, manager: HordeWorkerProcessManager) -> None:
        """Run one episode to a clear attempt, then end it the way a successful pop does."""
        _force_maintenance(manager)
        manager._drive_server_maintenance_recovery()
        _advance(manager, _FIRST_INTERVAL + 1)
        manager._drive_server_maintenance_recovery()
        manager._state.last_pop_maintenance_mode = False
        manager._state.server_maintenance_latched_at = 0.0
        manager._drive_server_maintenance_recovery()

    def test_a_prompt_retrip_doubles_the_next_interval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Being pushed straight back out says the worker was not as fit as its checks believed."""
        manager = _make_manager()
        recorder = _ClearRecorder(manager, monkeypatch)
        self._clear_and_rejoin(manager)
        assert recorder.calls == [False]

        _force_maintenance(manager)
        manager._drive_server_maintenance_recovery()
        _advance(manager, _FIRST_INTERVAL + 1)
        manager._drive_server_maintenance_recovery()
        assert recorder.calls == [False], "the re-trip must wait twice the base interval"

        _advance(manager, _FIRST_INTERVAL + 1)
        manager._drive_server_maintenance_recovery()
        assert recorder.calls == [False, False]

    def test_the_escalated_interval_is_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """However many times the horde reverses a clear, the wait stops widening at the ceiling."""
        manager = _make_manager()
        _ClearRecorder(manager, monkeypatch)
        manager._server_maintenance_clear_escalation = 20

        assert manager._server_maintenance_clear_interval(0) == SERVER_MAINTENANCE_CLEAR_MAX_INTERVAL_SECONDS
        assert manager._server_maintenance_clear_interval(5) == SERVER_MAINTENANCE_CLEAR_MAX_INTERVAL_SECONDS

    def test_a_healthy_stretch_resets_the_escalation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A worker that served for a long stretch between pauses is starting an unrelated episode."""
        manager = _make_manager()
        recorder = _ClearRecorder(manager, monkeypatch)
        self._clear_and_rejoin(manager)
        manager._server_maintenance_clear_escalation = 3

        _advance(manager, SERVER_MAINTENANCE_ESCALATION_DECAY_SECONDS + 60)
        _force_maintenance(manager)
        manager._drive_server_maintenance_recovery()
        assert manager._server_maintenance_clear_escalation == 0

        _advance(manager, _FIRST_INTERVAL + 1)
        manager._drive_server_maintenance_recovery()
        assert recorder.calls == [False, False]

    def test_a_retrip_inside_the_window_is_what_escalates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The escalation keys off the gap since the clear, not off the episode count."""
        manager = _make_manager()
        _ClearRecorder(manager, monkeypatch)
        self._clear_and_rejoin(manager)

        _advance(manager, SERVER_MAINTENANCE_RETRIP_WINDOW_SECONDS - 60)
        _force_maintenance(manager)
        manager._drive_server_maintenance_recovery()

        assert manager._server_maintenance_clear_escalation == 1


class TestAutoClearEligibility:
    """The worker may undo the horde's pause; it may never undo anybody's deliberate one."""

    def _elapse_to_first_attempt(self, manager: HordeWorkerProcessManager) -> None:
        """Latch a server-forced pause and let its first attempt come due."""
        _force_maintenance(manager)
        manager._drive_server_maintenance_recovery()
        _advance(manager, _FIRST_INTERVAL + 1)

    def test_a_deliberate_local_set_is_never_auto_cleared(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Maintenance a dashboard key, supervisor, or attach guard asked for is left standing."""
        manager = _make_manager()
        recorder = _ClearRecorder(manager, monkeypatch)
        manager._apply_supervisor_command(
            SupervisorControlMessage(
                command=SupervisorCommand.SET_SERVER_MAINTENANCE,
                server_maintenance_enabled=True,
            ),
        )
        assert manager._state.server_maintenance_locally_intended is True
        recorder.calls.clear()

        self._elapse_to_first_attempt(manager)
        manager._drive_server_maintenance_recovery()

        assert recorder.calls == []

    def test_unsetting_it_deliberately_re_enables_auto_clear(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The same surface releasing the intent puts the worker back in charge of its own recovery."""
        manager = _make_manager()
        recorder = _ClearRecorder(manager, monkeypatch)
        manager._state.server_maintenance_locally_intended = True
        self._elapse_to_first_attempt(manager)
        manager._drive_server_maintenance_recovery()
        assert recorder.calls == []

        manager._apply_supervisor_command(
            SupervisorControlMessage(
                command=SupervisorCommand.SET_SERVER_MAINTENANCE,
                server_maintenance_enabled=False,
            ),
        )
        recorder.calls.clear()
        manager._drive_server_maintenance_recovery()

        assert recorder.calls == [False]

    def test_a_pause_the_horde_did_not_force_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An owner-set pause reads as intent even though the worker never saw who set it."""
        manager = _make_manager()
        recorder = _ClearRecorder(manager, monkeypatch)
        _force_maintenance(manager, message=_OWNER_SET_MESSAGE)
        manager._drive_server_maintenance_recovery()
        _advance(manager, _FIRST_INTERVAL + 1)

        manager._drive_server_maintenance_recovery()

        assert recorder.calls == []

    def test_the_config_key_disables_it_entirely(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An operator who wants a forced pause to stand until they intervene can have that."""
        manager = _make_manager(auto_clear_server_maintenance=False)
        recorder = _ClearRecorder(manager, monkeypatch)

        self._elapse_to_first_attempt(manager)
        manager._drive_server_maintenance_recovery()

        assert recorder.calls == []


class TestFitnessGate:
    """Rejoining before the worker can serve would only re-earn the pause."""

    def _elapse_to_first_attempt(self, manager: HordeWorkerProcessManager) -> None:
        """Latch a server-forced pause and let its first attempt come due."""
        _force_maintenance(manager)
        manager._drive_server_maintenance_recovery()
        _advance(manager, _FIRST_INTERVAL + 1)

    def test_an_empty_pool_defers_without_consuming_the_attempt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no slot free the attempt is held, and it fires the moment one is."""
        manager = _make_manager()
        recorder = _ClearRecorder(manager, monkeypatch)
        self._elapse_to_first_attempt(manager)
        manager._process_map.clear()

        manager._drive_server_maintenance_recovery()
        assert recorder.calls == []
        assert manager._server_maintenance_clear_attempts == 0

        manager._process_map[0] = make_mock_process_info(0, state=HordeProcessState.WAITING_FOR_JOB)
        manager._drive_server_maintenance_recovery()

        assert recorder.calls == [False], "the held attempt fires as soon as the pool recovers"

    def test_a_standing_pop_pause_defers_the_attempt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A worker its own breaker is holding back is not fit to be offered more work."""
        manager = _make_manager()
        recorder = _ClearRecorder(manager, monkeypatch)
        self._elapse_to_first_attempt(manager)
        manager._state.self_throttle_paused = True
        manager._state.self_throttle_paused_until = time.time() + 300

        manager._drive_server_maintenance_recovery()
        assert recorder.calls == []

        manager._state.self_throttle_paused = False
        manager._drive_server_maintenance_recovery()

        assert recorder.calls == [False]

    async def test_a_recent_terminal_fault_defers_the_attempt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The quiet since the last drop is the evidence that the remedy removed the cause."""
        manager = _make_manager()
        recorder = _ClearRecorder(manager, monkeypatch)
        self._elapse_to_first_attempt(manager)
        job = await track_popped_job_async(manager._job_tracker, make_job_pop_response(model="stable_diffusion"))
        manager._job_tracker.handle_job_fault_now(job, retryable=False, fault_origin=JobFaultOrigin.GENERATION)

        manager._drive_server_maintenance_recovery()
        assert recorder.calls == []

        _advance(manager, SERVER_MAINTENANCE_FAULT_QUIET_SECONDS + 1)
        manager._drive_server_maintenance_recovery()

        assert recorder.calls == [False]


class TestMaintenanceHeartbeat:
    """An episode that lasts hours must be unmistakable in the log for its whole length."""

    def test_it_restates_the_episode_on_the_interval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One line per interval, carrying the age of the episode and the rejected-pop count."""
        manager = _make_manager()
        _ClearRecorder(manager, monkeypatch)
        for _ in range(7):
            _force_maintenance(manager)

        with _LogCapture() as logs:
            manager._drive_server_maintenance_recovery()
            assert logs.containing("still has this worker in maintenance") == []

            _advance(manager, SERVER_MAINTENANCE_HEARTBEAT_INTERVAL_SECONDS + 1)
            manager._drive_server_maintenance_recovery()
            manager._drive_server_maintenance_recovery()

        beats = logs.containing("still has this worker in maintenance")
        assert len(beats) == 1, logs.messages
        assert "7 job pops have been rejected" in beats[0]

    def test_it_names_the_reason_auto_clear_will_not_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An operator reading the log must be told the worker is waiting on them, not on itself."""
        manager = _make_manager(auto_clear_server_maintenance=False)
        _ClearRecorder(manager, monkeypatch)
        _force_maintenance(manager)
        manager._drive_server_maintenance_recovery()
        _advance(manager, SERVER_MAINTENANCE_HEARTBEAT_INTERVAL_SECONDS + 1)

        with _LogCapture() as logs:
            manager._drive_server_maintenance_recovery()

        beats = logs.containing("still has this worker in maintenance")
        assert len(beats) == 1
        assert "auto_clear_server_maintenance is off" in beats[0]

    def test_it_names_the_health_condition_holding_the_attempt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A held attempt is reported as held, so a quiet worker is never mistaken for a stuck one."""
        manager = _make_manager()
        _ClearRecorder(manager, monkeypatch)
        _force_maintenance(manager)
        manager._drive_server_maintenance_recovery()
        manager._process_map.clear()
        _advance(manager, SERVER_MAINTENANCE_HEARTBEAT_INTERVAL_SECONDS + 1)

        with _LogCapture() as logs:
            manager._drive_server_maintenance_recovery()

        beats = logs.containing("still has this worker in maintenance")
        assert len(beats) == 1
        assert "no inference process is free" in beats[0]

    def test_it_stops_once_the_worker_is_out_of_maintenance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The heartbeat belongs to the episode; nothing beats for a worker that is taking work."""
        manager = _make_manager()
        _ClearRecorder(manager, monkeypatch)
        _force_maintenance(manager)
        manager._drive_server_maintenance_recovery()
        _advance(manager, SERVER_MAINTENANCE_HEARTBEAT_INTERVAL_SECONDS + 1)
        manager._state.last_pop_maintenance_mode = False

        with _LogCapture() as logs:
            manager._drive_server_maintenance_recovery()

        assert logs.containing("still has this worker in maintenance") == []


class TestUnrelatedBehaviourIsUnchanged:
    """The pre-existing maintenance surfaces keep working exactly as they did."""

    def test_remove_maintenance_on_init_is_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The boot-time clear is a separate, opt-in path and is not gated on the new key."""
        manager = _make_manager(remove_maintenance_on_init=True, auto_clear_server_maintenance=False)
        recorded: list[bool] = []
        monkeypatch.setattr(manager, "set_maintenance", lambda enabled: recorded.append(enabled))

        manager.remove_maintenance()

        assert recorded == [False]

    def test_a_worker_that_is_not_in_maintenance_does_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The driver is inert on a healthy worker: no API call, no log, no bookkeeping."""
        manager = _make_manager()
        recorder = _ClearRecorder(manager, monkeypatch)

        with _LogCapture() as logs:
            manager._drive_server_maintenance_recovery()

        assert recorder.calls == []
        assert logs.messages == []
        assert manager._server_maintenance_episode_marker == 0.0


def test_the_off_loop_dispatch_is_a_real_thread_by_default() -> None:
    """The clear runs off the control loop, so a slow or unreachable horde cannot stall the tick."""
    manager = _make_manager()
    started: list[str] = []

    class _RecordingThread:
        def __init__(self, *, target: object, args: tuple[object, ...] = (), name: str = "", **_kw: object) -> None:
            started.append(name)

        def start(self) -> None:
            return None

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(process_manager_module.threading, "Thread", _RecordingThread)
        monkeypatch.setattr(manager, "_set_server_maintenance_safe", Mock())
        _force_maintenance(manager)
        manager._drive_server_maintenance_recovery()
        _advance(manager, _FIRST_INTERVAL + 1)
        manager._drive_server_maintenance_recovery()

    assert started == ["auto-clear-maintenance"]
