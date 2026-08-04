"""How a cooperative graceful stop presents on the dashboard, for both the owning and attach supervisors.

Every lifecycle transition records presentation state of its own: ``start`` sets STARTING, an unexpected
exit sets RESTARTING or CRASHED, ``request_restart`` pins RESTARTING, a finished stop sets STOPPED, and the
drain window of a cooperative stop pins STOPPING. That last one is what keeps a deliberate teardown from
falling through to UNRESPONSIVE: the outgoing worker's frame and liveness stamp stop advancing during
teardown, so a stop with no state of its own would look indistinguishable from a worker gone silent.

The tests compose the real supervisors with the real :func:`horde_worker_regen.tui.health.derive` the way
the app tick does, so a phase asserted here is the phase an operator sees. Alongside them are the checks
that the exemption stays narrow: a worker that went quiet without a stop being asked for still reads
UNRESPONSIVE, and a stop that overruns its window is still force-killed and lands on STOPPED.
"""

from __future__ import annotations

import time

import pytest

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.run_worker import WorkerLaunchOptions
from horde_worker_regen.tui import socket_protocol as sp
from horde_worker_regen.tui import worker_launcher
from horde_worker_regen.tui.attach import AttachedWorkerSupervisor
from horde_worker_regen.tui.health import WorkerPhase, derive
from horde_worker_regen.tui.worker_launcher import (
    SupervisorStatus,
    WorkerProcessMode,
    WorkerSupervisor,
)

_STALE_LIVENESS_SECONDS = 60.0
"""Well past ``health.STALE_SNAPSHOT_SECONDS``: the silence a worker's teardown legitimately produces."""


# region harness


class _FakeConn:
    """A no-traffic stand-in for a pipe connection."""

    def __init__(self) -> None:
        self.closed = False
        self.sent: list[object] = []

    def poll(self, timeout: float | None = None) -> bool:
        return False

    def recv(self) -> object:
        raise EOFError

    def send(self, obj: object) -> None:
        self.sent.append(obj)

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    """A controllable stand-in for a spawned worker process."""

    def __init__(self) -> None:
        self._alive = False
        self.pid = 4321
        self.exitcode: int | None = None

    def start(self) -> None:
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def exit_cleanly(self) -> None:
        """Model the worker finishing its own teardown and exiting with a success code."""
        self._alive = False
        self.exitcode = 0

    def join(self, timeout: float | None = None) -> None:
        self._alive = False

    def terminate(self) -> None:
        self._alive = False


class _FakeCtx:
    """A multiprocessing-context stand-in that records spawns."""

    def __init__(self) -> None:
        self.process_count = 0
        self.last_process: _FakeProcess | None = None

    def Pipe(self, duplex: bool = True) -> tuple[_FakeConn, _FakeConn]:  # noqa: N802 - mirrors ctx API
        return _FakeConn(), _FakeConn()

    def Process(self, **kwargs: object) -> _FakeProcess:  # noqa: N802 - mirrors ctx API
        self.process_count += 1
        self.last_process = _FakeProcess()
        return self.last_process


class _Clock:
    """A ``time``-module stand-in exposing only the ``time()`` the supervisor uses."""

    def __init__(self, now: float) -> None:
        self.now = now

    def time(self) -> float:
        """The current fake wall clock."""
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the fake wall clock forward."""
        self.now += seconds


def _running_worker_snapshot() -> WorkerStateSnapshot:
    """The last frame a healthy worker sent before it was asked to stop (no shutdown flag yet)."""
    return WorkerStateSnapshot(config=WorkerConfigSummary(dreamer_name="Test", worker_version="12.0.0"))


def _dashboard_phase(supervisor: object, *, now: float) -> WorkerPhase:
    """Derive the phase the dashboard would show, composing state exactly as the app tick does.

    Mirrors the composition in ``HordeWorkerTUI._tick``: responsiveness is judged from the worker loop's
    last liveness stamp, falling back to snapshot age when the worker never sent a liveness frame.
    """
    snapshot = supervisor.latest_snapshot  # type: ignore[attr-defined]
    snapshot_age = (now - snapshot.timestamp) if snapshot is not None else None
    liveness_wall_time = supervisor.last_liveness_wall_time  # type: ignore[attr-defined]
    liveness_age = (now - liveness_wall_time) if liveness_wall_time is not None else snapshot_age
    report = derive(
        snapshot,
        supervisor.status,  # type: ignore[attr-defined]
        liveness_age,
        fatal_error=supervisor.last_fatal_error,  # type: ignore[attr-defined]
    )
    return report.phase


def _owning_supervisor_mid_stop(ctx: _FakeCtx) -> tuple[WorkerSupervisor, float]:
    """A started supervisor whose worker was asked to stop and has since gone quiet.

    Returns the supervisor and the wall-clock time to derive at. The retained frame is a pre-shutdown one:
    the worker either never received the shutdown request or its control loop stopped before it could
    publish a frame carrying ``shutting_down``, which is precisely the teardown window in question.
    """
    supervisor = WorkerSupervisor(WorkerLaunchOptions(), mode=WorkerProcessMode.FAKE, ctx=ctx)  # type: ignore[arg-type]
    supervisor.start()
    now = time.time()
    supervisor.latest_snapshot = _running_worker_snapshot()
    supervisor.last_liveness_wall_time = now - _STALE_LIVENESS_SECONDS
    supervisor.request_graceful_stop()
    return supervisor, now


def _attached_supervisor(monkeypatch: pytest.MonkeyPatch) -> AttachedWorkerSupervisor:
    """An attach client with its reader thread disabled, presenting a connected and running host."""
    monkeypatch.setattr(AttachedWorkerSupervisor, "_read_loop", lambda self: None)
    client = AttachedWorkerSupervisor(("127.0.0.1", 0))
    client._status = SupervisorStatus.RUNNING
    client._worker_running = True
    return client


# endregion


# region owning supervisor


def test_graceful_stop_presents_as_shutting_down_not_unresponsive() -> None:
    """A stop the operator asked for must never present as the worker having gone wrong.

    ``request_graceful_stop`` pins STOPPING and drops the outgoing worker's frames, so the phase comes from
    the supervisor's own intent. Without that, the only thing standing between the dashboard and
    UNRESPONSIVE would be the worker having managed to publish a frame with ``shutting_down`` set before it
    went quiet to tear down.
    """
    ctx = _FakeCtx()
    supervisor, now = _owning_supervisor_mid_stop(ctx)

    assert supervisor.status is SupervisorStatus.STOPPING
    assert _dashboard_phase(supervisor, now=now) is WorkerPhase.SHUTTING_DOWN


def test_stopping_never_alarms_however_long_the_teardown_runs() -> None:
    """The stopping presentation must hold for the whole drain, not just until the frames age out."""
    ctx = _FakeCtx()
    supervisor, now = _owning_supervisor_mid_stop(ctx)

    # A late frame from the draining worker is still useful detail, but it must not revive the alarm as it
    # ages, nor undo the pinned intent.
    supervisor.latest_snapshot = _running_worker_snapshot()
    supervisor.last_liveness_wall_time = now - _STALE_LIVENESS_SECONDS

    for elapsed in (0.0, 60.0, worker_launcher.GRACEFUL_STOP_TIMEOUT_SECONDS):
        assert _dashboard_phase(supervisor, now=now + elapsed) is WorkerPhase.SHUTTING_DOWN


def test_cooperative_restart_presents_as_restarting() -> None:
    """The restart flow does record its intent, and is the behaviour the stop flow has to match."""
    ctx = _FakeCtx()
    supervisor = WorkerSupervisor(WorkerLaunchOptions(), mode=WorkerProcessMode.FAKE, ctx=ctx)  # type: ignore[arg-type]
    supervisor.start()
    now = time.time()
    supervisor.latest_snapshot = _running_worker_snapshot()
    supervisor.last_liveness_wall_time = now - _STALE_LIVENESS_SECONDS

    supervisor.request_restart()

    assert _dashboard_phase(supervisor, now=now) is WorkerPhase.RESTARTING


def test_silent_worker_with_no_stop_requested_still_presents_as_unresponsive() -> None:
    """The alarm must keep firing for a worker that went quiet on its own; only a stop is exempt."""
    ctx = _FakeCtx()
    supervisor = WorkerSupervisor(WorkerLaunchOptions(), mode=WorkerProcessMode.FAKE, ctx=ctx)  # type: ignore[arg-type]
    supervisor.start()
    now = time.time()
    supervisor.latest_snapshot = _running_worker_snapshot()
    supervisor.last_liveness_wall_time = now - _STALE_LIVENESS_SECONDS

    assert _dashboard_phase(supervisor, now=now) is WorkerPhase.UNRESPONSIVE


def test_completed_graceful_stop_presents_as_stopped() -> None:
    """Once the worker exits, the finalizing tick clears the frame and the phase resolves to STOPPED."""
    ctx = _FakeCtx()
    supervisor, _ = _owning_supervisor_mid_stop(ctx)
    process = ctx.last_process
    assert process is not None

    process.exit_cleanly()
    supervisor.tick()

    assert supervisor.latest_snapshot is None
    assert _dashboard_phase(supervisor, now=time.time()) is WorkerPhase.STOPPED


def test_second_stop_request_does_not_defer_the_force_kill_backstop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pressing stop again must not push out the deadline that force-kills a worker that will not exit.

    The app routes a repeat press back into ``request_graceful_stop`` because the worker is still alive.
    If each press moved ``_graceful_stop_deadline`` a full timeout into the future, the force-kill that
    ends a stop the worker is ignoring would never arrive, and the flow would hang for as long as the
    operator kept trying. The request itself is re-sent, in case the first never landed.
    """
    clock = _Clock(1000.0)
    monkeypatch.setattr(worker_launcher, "time", clock)
    ctx = _FakeCtx()
    supervisor = WorkerSupervisor(WorkerLaunchOptions(), mode=WorkerProcessMode.FAKE, ctx=ctx)  # type: ignore[arg-type]
    supervisor.start()
    connection = supervisor._connection
    assert isinstance(connection, _FakeConn)

    supervisor.request_graceful_stop(timeout=150.0)
    first_deadline = supervisor._graceful_stop_deadline
    sends_after_first = len(connection.sent)

    clock.advance(60.0)
    # The app's start/stop key sends a live worker back to stop, which is what the operator presses when
    # a long teardown looks like nothing is happening.
    assert supervisor.status is SupervisorStatus.STOPPING
    assert supervisor.is_alive()
    supervisor.request_graceful_stop(timeout=150.0)

    assert supervisor._graceful_stop_deadline == first_deadline
    assert len(connection.sent) > sends_after_first, "the repeat press should still re-send the request"


def test_stop_that_overruns_its_window_is_still_force_killed_and_lands_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pinning STOPPING must not disarm the backstop: an overrunning stop still ends in a tree kill.

    This is what bounds the calm stopping presentation. A worker that ignores the shutdown request is
    terminated once its window lapses, and the phase resolves to STOPPED rather than sitting on a
    reassuring screen forever.
    """
    clock = _Clock(1000.0)
    monkeypatch.setattr(worker_launcher, "time", clock)
    killed_pids: list[int] = []
    monkeypatch.setattr(
        worker_launcher,
        "kill_process_tree",
        lambda pid, **_kwargs: killed_pids.append(pid) or [pid],
    )
    ctx = _FakeCtx()
    supervisor = WorkerSupervisor(WorkerLaunchOptions(), mode=WorkerProcessMode.FAKE, ctx=ctx)  # type: ignore[arg-type]
    supervisor.start()
    process = ctx.last_process
    assert process is not None

    supervisor.request_graceful_stop(timeout=150.0)
    clock.advance(60.0)
    supervisor.tick()
    assert killed_pids == [], "the worker is still inside its window"
    assert _dashboard_phase(supervisor, now=clock.now) is WorkerPhase.SHUTTING_DOWN

    clock.advance(100.0)
    supervisor.tick()
    assert killed_pids == [process.pid]

    process.exit_cleanly()
    supervisor.tick()
    assert _dashboard_phase(supervisor, now=clock.now) is WorkerPhase.STOPPED


def test_restart_requested_while_stopping_upgrades_the_stop_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    """A restart pressed during a stop revises the intent rather than starting a second teardown.

    The worker is already draining and being asked to exit, so the restart adds only the replacement:
    RESTARTING takes over the presentation, the original force-kill deadline stands, and the successor
    spawns once (and only once) the draining worker is confirmed dead.
    """
    clock = _Clock(1000.0)
    monkeypatch.setattr(worker_launcher, "time", clock)
    ctx = _FakeCtx()
    supervisor = WorkerSupervisor(WorkerLaunchOptions(), mode=WorkerProcessMode.FAKE, ctx=ctx)  # type: ignore[arg-type]
    supervisor.start()
    process = ctx.last_process
    assert process is not None

    supervisor.request_graceful_stop(timeout=150.0)
    first_deadline = supervisor._graceful_stop_deadline

    clock.advance(10.0)
    supervisor.request_restart(timeout=150.0)

    assert supervisor.status is SupervisorStatus.RESTARTING
    assert supervisor._graceful_stop_deadline == first_deadline
    assert ctx.process_count == 1, "no replacement may spawn while the old worker still drains"

    process.exit_cleanly()
    supervisor.tick()

    assert ctx.process_count == 2
    assert supervisor.status is SupervisorStatus.RESTARTING


def test_start_requested_while_stopping_upgrades_the_stop_instead_of_double_spawning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A start pressed during a drain must not put a second worker beside the one still exiting.

    The draining worker still owns the GPU, the worker name, and its own subprocess tree, and the handle
    to it would be overwritten by an immediate spawn, leaving it orphaned. The start therefore becomes the
    replacement the stop already knows how to make: one worker, spawned once the old one is confirmed dead.
    """
    clock = _Clock(1000.0)
    monkeypatch.setattr(worker_launcher, "time", clock)
    ctx = _FakeCtx()
    supervisor = WorkerSupervisor(WorkerLaunchOptions(), mode=WorkerProcessMode.FAKE, ctx=ctx)  # type: ignore[arg-type]
    supervisor.start()
    process = ctx.last_process
    assert process is not None

    supervisor.request_graceful_stop(timeout=150.0)
    first_deadline = supervisor._graceful_stop_deadline

    clock.advance(10.0)
    supervisor.start()

    assert ctx.process_count == 1, "no worker may spawn while the old one still drains"
    assert supervisor.status is SupervisorStatus.RESTARTING
    assert supervisor._graceful_stop_deadline == first_deadline, "the force-kill backstop keeps its window"
    assert supervisor.is_alive(), "the draining worker is still the one being tracked"

    process.exit_cleanly()
    supervisor.tick()

    assert ctx.process_count == 2
    assert supervisor.status is SupervisorStatus.RESTARTING


def test_repeated_starts_during_a_drain_still_produce_exactly_one_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator pressing start repeatedly through a long drain gets one successor, not a queue of them."""
    clock = _Clock(1000.0)
    monkeypatch.setattr(worker_launcher, "time", clock)
    ctx = _FakeCtx()
    supervisor = WorkerSupervisor(WorkerLaunchOptions(), mode=WorkerProcessMode.FAKE, ctx=ctx)  # type: ignore[arg-type]
    supervisor.start()
    process = ctx.last_process
    assert process is not None

    supervisor.request_graceful_stop(timeout=150.0)
    for _ in range(5):
        clock.advance(1.0)
        supervisor.start()
        supervisor.tick()

    assert ctx.process_count == 1

    process.exit_cleanly()
    supervisor.tick()
    supervisor.tick()

    assert ctx.process_count == 2


def test_start_on_an_idle_supervisor_still_spawns_immediately() -> None:
    """The plain start path is untouched: nothing is draining, so the worker launches at once."""
    ctx = _FakeCtx()
    supervisor = WorkerSupervisor(WorkerLaunchOptions(), mode=WorkerProcessMode.FAKE, ctx=ctx)  # type: ignore[arg-type]

    supervisor.start()

    assert ctx.process_count == 1
    assert supervisor.status is SupervisorStatus.STARTING
    assert supervisor.is_alive()


def test_start_after_a_completed_stop_spawns_a_fresh_worker() -> None:
    """Once the stop has finished there is nothing to upgrade, so a start is an ordinary launch."""
    ctx = _FakeCtx()
    supervisor, _ = _owning_supervisor_mid_stop(ctx)
    process = ctx.last_process
    assert process is not None
    process.exit_cleanly()
    supervisor.tick()
    assert supervisor.status is SupervisorStatus.STOPPED

    supervisor.start()

    assert ctx.process_count == 2
    assert supervisor.status is SupervisorStatus.STARTING


# endregion


# region attach client


def test_attached_graceful_stop_presents_as_shutting_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """An attached session must present its own stop intent without waiting for the host to agree.

    The host keeps the stopping worker's last frame and its frozen liveness stamp and re-broadcasts both
    every interval while its own status still reads running, so a client with no local record would show
    the stop it asked for as a worker that stopped responding.
    """
    client = _attached_supervisor(monkeypatch)
    now = time.time()
    client.latest_snapshot = _running_worker_snapshot()
    client.last_liveness_wall_time = now - _STALE_LIVENESS_SECONDS

    client.request_graceful_stop()

    assert client.status is SupervisorStatus.STOPPING
    assert _dashboard_phase(client, now=now) is WorkerPhase.SHUTTING_DOWN


def test_attached_stop_intent_outlives_the_hosts_stale_running_broadcasts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local intent must survive the host's pre-stop status frames, then yield to its verdict."""
    client = _attached_supervisor(monkeypatch)
    now = time.time()
    client.request_graceful_stop()

    # The host has not applied the request yet and re-broadcasts the running worker's state.
    client._apply_status({"status": SupervisorStatus.RUNNING.value, "worker_running": True})
    client._apply({"type": sp.MSG_SNAPSHOT, "snapshot": _running_worker_snapshot().model_dump(mode="json")})
    client.last_liveness_wall_time = now - _STALE_LIVENESS_SECONDS

    assert client.status is SupervisorStatus.STOPPING
    assert _dashboard_phase(client, now=now) is WorkerPhase.SHUTTING_DOWN

    client._apply_status({"status": SupervisorStatus.STOPPED.value, "worker_running": False})

    assert client.status is SupervisorStatus.STOPPED
    assert _dashboard_phase(client, now=now) is WorkerPhase.STOPPED


def test_attached_restart_presents_as_restarting(monkeypatch: pytest.MonkeyPatch) -> None:
    """The attach client's restart drops the stale frame and pins RESTARTING; stop must match it."""
    client = _attached_supervisor(monkeypatch)
    now = time.time()
    client.latest_snapshot = _running_worker_snapshot()
    client.last_liveness_wall_time = now - _STALE_LIVENESS_SECONDS

    client.request_restart()

    assert client.latest_snapshot is None
    assert _dashboard_phase(client, now=now) is WorkerPhase.RESTARTING


def test_attached_host_reporting_stopped_presents_as_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once the host reports the stop finished, the client drops the frame and resolves to STOPPED."""
    client = _attached_supervisor(monkeypatch)
    now = time.time()
    client.latest_snapshot = _running_worker_snapshot()
    client.last_liveness_wall_time = now - _STALE_LIVENESS_SECONDS

    client._apply_status({"status": SupervisorStatus.STOPPED.value, "worker_running": False})

    assert client.latest_snapshot is None
    assert _dashboard_phase(client, now=now) is WorkerPhase.STOPPED


def test_snapshot_arriving_behind_a_terminal_status_does_not_resurrect_the_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frame acceptance follows the lifecycle, not the order the host happens to send its two frames in.

    One host broadcast writes the status frame and then the retained snapshot, so the snapshot of a worker
    that has just died lands immediately behind the status that reported its death. Re-populating from it
    would hand the session a frame belonging to a dead incarnation, which the next running lifecycle would
    then age into the unresponsive alarm.
    """
    client = _attached_supervisor(monkeypatch)
    now = time.time()
    stale = _running_worker_snapshot()

    client._apply({"type": sp.MSG_STATUS, "status": SupervisorStatus.STOPPED.value, "worker_running": False})
    client._apply({"type": sp.MSG_SNAPSHOT, "snapshot": stale.model_dump(mode="json")})

    assert client.latest_snapshot is None
    assert _dashboard_phase(client, now=now) is WorkerPhase.STOPPED

    # The worker is started again; the dead incarnation's frame must not be what this session presents.
    client._apply({"type": sp.MSG_STATUS, "status": SupervisorStatus.STARTING.value, "worker_running": True})

    assert client.latest_snapshot is None
    assert _dashboard_phase(client, now=now + _STALE_LIVENESS_SECONDS) is WorkerPhase.INITIALIZING


def test_snapshot_arriving_behind_a_crashed_status_does_not_resurrect_the_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash is terminal for the reporting incarnation too, so its trailing frame is equally spent."""
    client = _attached_supervisor(monkeypatch)

    client._apply({"type": sp.MSG_STATUS, "status": SupervisorStatus.CRASHED.value, "worker_running": False})
    client._apply({"type": sp.MSG_SNAPSHOT, "snapshot": _running_worker_snapshot().model_dump(mode="json")})

    assert client.latest_snapshot is None
    assert _dashboard_phase(client, now=time.time()) is WorkerPhase.CRASHED


def test_host_snapshot_is_refused_while_this_sessions_stop_intent_stands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The frames dropped on a local stop must stay dropped while the host is still catching up."""
    client = _attached_supervisor(monkeypatch)
    now = time.time()

    client.request_graceful_stop()
    client._apply({"type": sp.MSG_SNAPSHOT, "snapshot": _running_worker_snapshot().model_dump(mode="json")})

    assert client.latest_snapshot is None
    assert _dashboard_phase(client, now=now) is WorkerPhase.SHUTTING_DOWN


def test_running_host_snapshots_are_still_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate stays narrow: a live worker's frames are what the dashboard renders from."""
    client = _attached_supervisor(monkeypatch)

    client._apply({"type": sp.MSG_STATUS, "status": SupervisorStatus.RUNNING.value, "worker_running": True})
    client._apply({"type": sp.MSG_SNAPSHOT, "snapshot": _running_worker_snapshot().model_dump(mode="json")})

    assert client.latest_snapshot is not None
    assert client.latest_snapshot.config.dreamer_name == "Test"


# endregion
