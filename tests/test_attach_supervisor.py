"""Unit tests for the headless attach supervisor's observation/control loop.

The loop is driven as a pure state machine over a fake controller and an explicit clock: no worker, no
GPU, no real pipe. Each test asserts one contract: bounded state lines, edge-triggered threshold rules,
once-only inbox command application (malformed skipped), and the single-shot, re-armable auto-guard.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path

from horde_worker_regen.attach_supervisor import (
    AUTO_GUARD_CONFIRM_SECONDS,
    OVERRUN_INTERVAL_FACTOR,
    STATE_LINE_MAX_BYTES,
    AttachSupervisor,
    ThreadedWatchRunner,
    _resolve_command,
)
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    CurrentDownloadStatus,
    DownloadPhase,
    DownloadStatusSnapshot,
    ProcessSnapshot,
    SupervisorCommand,
    SupervisorControlMessage,
    WorkerConfigSummary,
    WorkerStateSnapshot,
)


class FakeController:
    """A stand-in for WorkerSupervisor exposing only what the loop reads and drives."""

    def __init__(self) -> None:
        """Start with no snapshot, no liveness, and empty command/stop records."""
        self.latest_snapshot: WorkerStateSnapshot | None = None
        self.last_liveness_wall_time: float | None = None
        self.sent: list[SupervisorControlMessage] = []
        self.graceful_stops = 0
        self.ticks = 0
        self.alive = True

    def tick(self) -> None:
        """Record a tick (the fake has no worker to advance)."""
        self.ticks += 1

    def send_command(self, command: SupervisorControlMessage) -> bool:
        """Record a sent command and report success."""
        self.sent.append(command)
        return True

    def request_graceful_stop(self, *, timeout: float = 150.0) -> None:
        """Record a graceful-stop request; the fake worker is considered exited immediately."""
        self.graceful_stops += 1
        self.alive = False

    def is_alive(self) -> bool:
        """Whether the fake worker is still running."""
        return self.alive


def _config() -> WorkerConfigSummary:
    return WorkerConfigSummary(dreamer_name="w", worker_version="9.9.9")


def _proc(pid: int = 0, *, busy: bool = False, ptype: str = "INFERENCE") -> ProcessSnapshot:
    return ProcessSnapshot(
        process_id=pid,
        process_type=ptype,
        last_process_state="WAITING_FOR_JOB",
        is_alive=True,
        is_busy=busy,
    )


def _snapshot(**overrides: object) -> WorkerStateSnapshot:
    fields: dict[str, object] = {"config": _config()}
    fields.update(overrides)
    return WorkerStateSnapshot(**fields)  # type: ignore[arg-type]


def _server_maintenance_commands(controller: FakeController, *, enabled: bool) -> list[SupervisorControlMessage]:
    return [
        command
        for command in controller.sent
        if command.command is SupervisorCommand.SET_SERVER_MAINTENANCE
        and command.server_maintenance_enabled is enabled
    ]


def _alerts(session_dir: Path, rule: str | None = None) -> list[dict[str, object]]:
    path = session_dir / "alerts.jsonl"
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [record for record in records if rule is None or record.get("rule") == rule]


def _attach(controller: FakeController, session_dir: Path) -> AttachSupervisor:
    return AttachSupervisor(controller, session_dir=session_dir, log_dir=None)


class TestStateLine:
    """The per-interval state line: written, bounded, and honest before the first snapshot."""

    def test_written_and_bounded(self, tmp_path: Path) -> None:
        """A state line is written each poll, stays under the byte cap, and sheds rows to do so."""
        controller = FakeController()
        controller.latest_snapshot = _snapshot(
            processes=[_proc(index) for index in range(40)],
            jobs_pending_inference=3,
        )
        controller.last_liveness_wall_time = 1000.0
        attach = _attach(controller, tmp_path)

        attach.poll_once(now=1000.0)

        lines = (tmp_path / "state.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert len(lines[0].encode("utf-8")) <= STATE_LINE_MAX_BYTES
        record = json.loads(lines[0])
        assert record["worker_up"] is True
        assert record["queue"] == 3
        assert record["procs_truncated"] is True  # 40 processes cannot fit under the cap

    def test_worker_up_false_before_first_snapshot(self, tmp_path: Path) -> None:
        """Before any snapshot arrives the state line still records that the worker is not yet reporting."""
        attach = _attach(FakeController(), tmp_path)
        attach.poll_once(now=1.0)
        record = json.loads((tmp_path / "state.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert record["worker_up"] is False


class TestThresholdRules:
    """Each snapshot threshold rule is edge-triggered: one alert per episode, re-arming when it clears."""

    def test_frozen_parent_fires_once_then_rearms(self, tmp_path: Path) -> None:
        """The frozen-parent rule fires once while stale, and again after a fresh liveness stamp re-arms it."""
        controller = FakeController()
        controller.latest_snapshot = _snapshot()
        controller.last_liveness_wall_time = 100.0
        attach = _attach(controller, tmp_path)

        attach.poll_once(now=131.0)  # 31s stale (> 30): fires
        attach.poll_once(now=140.0)  # still stale: no re-fire
        alerts = _alerts(tmp_path, "frozen_parent")
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "critical"
        assert alerts[0]["liveness_age_seconds"] >= 30

        controller.last_liveness_wall_time = 200.0
        attach.poll_once(now=205.0)  # fresh: re-arms
        attach.poll_once(now=236.0)  # stale again: fires again
        assert len(_alerts(tmp_path, "frozen_parent")) == 2

    def test_consecutive_failure_pause_fires_once(self, tmp_path: Path) -> None:
        """The consecutive-failure-pause rule fires once per armed episode."""
        controller = FakeController()
        attach = _attach(controller, tmp_path)

        controller.latest_snapshot = _snapshot(too_many_consecutive_failed_jobs=True, consecutive_failed_jobs=3)
        attach.poll_once(now=10.0)
        attach.poll_once(now=20.0)
        alerts = _alerts(tmp_path, "consecutive_failure_pause")
        assert len(alerts) == 1
        assert alerts[0]["consecutive_failed_jobs"] == 3

    def test_fault_burst_fires_and_rearms(self, tmp_path: Path) -> None:
        """A burst of faults in the trailing window fires once; a later burst after it clears fires again."""
        controller = FakeController()
        attach = _attach(controller, tmp_path)

        controller.latest_snapshot = _snapshot(num_jobs_faulted=0)
        attach.poll_once(now=0.0)
        controller.latest_snapshot = _snapshot(num_jobs_faulted=5)
        attach.poll_once(now=10.0)  # 5 faults since start: fires
        controller.latest_snapshot = _snapshot(num_jobs_faulted=6)
        attach.poll_once(now=20.0)  # still elevated: no re-fire
        assert len(_alerts(tmp_path, "fault_burst")) == 1

        # Let the window slide so the delta drops below threshold (clears), then a fresh burst re-fires.
        controller.latest_snapshot = _snapshot(num_jobs_faulted=6)
        attach.poll_once(now=700.0)
        controller.latest_snapshot = _snapshot(num_jobs_faulted=12)
        attach.poll_once(now=705.0)
        assert len(_alerts(tmp_path, "fault_burst")) == 2

    def test_gpu_idle_with_pending_needs_persistence(self, tmp_path: Path) -> None:
        """The GPU-idle-with-pending rule fires only after the condition holds past 120s, then once."""
        controller = FakeController()
        controller.latest_snapshot = _snapshot(jobs_pending_inference=2, processes=[_proc(0, busy=False)])
        attach = _attach(controller, tmp_path)

        attach.poll_once(now=0.0)
        attach.poll_once(now=60.0)
        assert _alerts(tmp_path, "gpu_idle_with_pending") == []
        attach.poll_once(now=130.0)  # sustained > 120s: fires
        attach.poll_once(now=140.0)  # no re-fire
        alerts = _alerts(tmp_path, "gpu_idle_with_pending")
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "critical"
        assert alerts[0]["pending_jobs"] == 2
        assert alerts[0]["idle_seconds"] >= 120

    def test_gpu_idle_not_fired_when_a_process_is_busy(self, tmp_path: Path) -> None:
        """A busy inference process means the GPU is being fed, so the rule stays silent."""
        controller = FakeController()
        controller.latest_snapshot = _snapshot(jobs_pending_inference=2, processes=[_proc(0, busy=True)])
        attach = _attach(controller, tmp_path)
        attach.poll_once(now=0.0)
        attach.poll_once(now=200.0)
        assert _alerts(tmp_path, "gpu_idle_with_pending") == []

    def test_download_no_progress_fires_and_resets_on_progress(self, tmp_path: Path) -> None:
        """A stalled download fires after 120s of no byte progress; renewed progress re-arms the rule."""
        controller = FakeController()
        stalled = DownloadStatusSnapshot(
            phase=DownloadPhase.DOWNLOADING,
            current=CurrentDownloadStatus(
                model_name="huge-model",
                feature="image model",
                target_dir="d",
                downloaded_bytes=100,
                total_bytes=1000,
            ),
        )
        controller.latest_snapshot = _snapshot(downloads=stalled)
        attach = _attach(controller, tmp_path)

        attach.poll_once(now=0.0)
        attach.poll_once(now=60.0)
        assert _alerts(tmp_path, "download_no_progress") == []
        attach.poll_once(now=130.0)  # no byte movement for > 120s: fires
        assert len(_alerts(tmp_path, "download_no_progress")) == 1

        progressed = DownloadStatusSnapshot(
            phase=DownloadPhase.DOWNLOADING,
            current=CurrentDownloadStatus(
                model_name="huge-model",
                feature="image model",
                target_dir="d",
                downloaded_bytes=500,
                total_bytes=1000,
            ),
        )
        controller.latest_snapshot = _snapshot(downloads=progressed)
        attach.poll_once(now=140.0)  # progress: re-arms
        controller.latest_snapshot = _snapshot(downloads=progressed)  # then stalls again at the new byte count
        attach.poll_once(now=280.0)
        assert len(_alerts(tmp_path, "download_no_progress")) == 2


class TestCommandInbox:
    """The command inbox: full verb coverage, once-only application, malformed lines skipped."""

    def test_verb_resolution_covers_every_command(self) -> None:
        """Every SupervisorCommand verb resolves, plus the GRACEFUL_SHUTDOWN alias; unknowns return None."""
        for command in SupervisorCommand:
            assert _resolve_command(command.name) is command
            assert _resolve_command(command.name.lower()) is command
        assert _resolve_command("graceful_shutdown") is SupervisorCommand.SHUTDOWN
        assert _resolve_command("definitely-not-a-command") is None

    def test_commands_applied_once_and_malformed_skipped(self, tmp_path: Path) -> None:
        """Well-formed lines map to control messages and apply once; malformed/unknown lines are skipped."""
        controller = FakeController()
        attach = _attach(controller, tmp_path)
        (tmp_path / "commands.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"command": "SET_CONCURRENCY", "target_threads": 2, "target_processes": 3}),
                    "{ this is not json",
                    json.dumps({"command": "BOGUS_VERB"}),
                    json.dumps({"command": "SET_SERVER_MAINTENANCE", "server_maintenance_enabled": True}),
                ],
            )
            + "\n",
            encoding="utf-8",
        )

        attach.poll_once(now=0.0)

        assert len(controller.sent) == 2
        first = controller.sent[0]
        assert first.command is SupervisorCommand.SET_CONCURRENCY
        assert first.target_threads == 2
        assert first.target_processes == 3
        second = controller.sent[1]
        assert second.command is SupervisorCommand.SET_SERVER_MAINTENANCE
        assert second.server_maintenance_enabled is True

        attach.poll_once(now=5.0)  # no new lines: nothing re-applied
        assert len(controller.sent) == 2

        rejected = _alerts(tmp_path, "command_rejected")
        assert len(rejected) == 2  # the torn JSON and the unknown verb

    def test_shutdown_routes_through_graceful_stop(self, tmp_path: Path) -> None:
        """SHUTDOWN/GRACEFUL_SHUTDOWN uses the orphan-proof graceful path, not a raw control command."""
        controller = FakeController()
        attach = _attach(controller, tmp_path)
        (tmp_path / "commands.jsonl").write_text(
            json.dumps({"command": "GRACEFUL_SHUTDOWN"}) + "\n",
            encoding="utf-8",
        )
        attach.poll_once(now=0.0)
        assert controller.graceful_stops == 1
        assert controller.sent == []


class TestAutoGuard:
    """The single pre-authorized auto-guard: one-shot maintenance on persistence, re-armable by the operator."""

    def test_fires_after_persistence_once_and_rearms(self, tmp_path: Path) -> None:
        """The guard sets maintenance on once after the condition persists, and re-arms when it is lifted."""
        controller = FakeController()
        controller.latest_snapshot = _snapshot(jobs_pending_inference=1, processes=[_proc(0, busy=False)])
        attach = _attach(controller, tmp_path)

        attach.poll_once(now=0.0)
        attach.poll_once(now=AUTO_GUARD_CONFIRM_SECONDS - 1.0)  # not yet persisted long enough
        assert _server_maintenance_commands(controller, enabled=True) == []

        attach.poll_once(now=AUTO_GUARD_CONFIRM_SECONDS + 1.0)  # persisted: fires once
        attach.poll_once(now=AUTO_GUARD_CONFIRM_SECONDS + 20.0)  # no re-fire
        assert len(_server_maintenance_commands(controller, enabled=True)) == 1

        guard_alerts = _alerts(tmp_path, "auto_guard_server_maintenance")
        assert len(guard_alerts) == 1
        assert guard_alerts[0]["severity"] == "critical"

        # Operator lifts maintenance via the inbox: the guard re-arms and must re-confirm before acting again.
        base = AUTO_GUARD_CONFIRM_SECONDS + 30.0
        (tmp_path / "commands.jsonl").write_text(
            json.dumps({"command": "SET_SERVER_MAINTENANCE", "server_maintenance_enabled": False}) + "\n",
            encoding="utf-8",
        )
        attach.poll_once(now=base)  # applies the lift, re-arms; condition still true so the timer restarts
        assert len(_server_maintenance_commands(controller, enabled=True)) == 1  # not re-fired yet
        attach.poll_once(now=base + AUTO_GUARD_CONFIRM_SECONDS + 1.0)  # persisted again: fires again
        assert len(_server_maintenance_commands(controller, enabled=True)) == 2

    def test_does_not_fire_without_a_guard_condition(self, tmp_path: Path) -> None:
        """A healthy worker (fed GPU, fresh liveness) never triggers the guard."""
        controller = FakeController()
        controller.latest_snapshot = _snapshot(jobs_pending_inference=1, processes=[_proc(0, busy=True)])
        controller.last_liveness_wall_time = 0.0
        attach = _attach(controller, tmp_path)
        for now in (0.0, 100.0, 300.0, 600.0):
            controller.last_liveness_wall_time = now  # keep liveness fresh
            attach.poll_once(now=now)
        assert _server_maintenance_commands(controller, enabled=True) == []


class TestWatchOffThread:
    """The live-log watch runs off the poll thread, so its cost cannot delay observation or control."""

    def test_slow_watch_pass_does_not_block_inbox_or_state(self, tmp_path: Path) -> None:
        """While a watch pass is blocked on the runner thread, a poll still applies inbox commands and writes state."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        session_dir = tmp_path / "session"
        runner = ThreadedWatchRunner()
        controller = FakeController()
        attach = AttachSupervisor(
            controller,
            session_dir=session_dir,
            log_dir=log_dir,
            interval=5.0,
            watch_runner=runner,
        )

        started = threading.Event()
        release = threading.Event()

        def _blocking_pass(now: float) -> list[dict[str, object]]:
            started.set()
            release.wait(10.0)
            return [{"t": now, "rule": "log_watch", "severity": "info", "summary": "late"}]

        attach._run_watch_pass = _blocking_pass  # type: ignore[assignment,method-assign]

        try:
            attach.poll_once(now=1.0)  # dispatches the blocking pass to the background thread
            assert started.wait(3.0), "the watch pass never started on the runner thread"
            assert runner.in_flight()

            # A command lands while the watch pass is still blocked; the next poll must service it promptly.
            (session_dir / "commands.jsonl").write_text(
                json.dumps({"command": "PAUSE"}) + "\n",
                encoding="utf-8",
            )
            wall_start = time.monotonic()
            attach.poll_once(now=2.0)
            elapsed = time.monotonic() - wall_start

            assert elapsed < 2.0, "poll_once blocked on the in-flight watch pass"
            assert runner.in_flight(), "the second poll should not have waited for the blocked pass"
            assert any(command.command is SupervisorCommand.PAUSE for command in controller.sent)
            state_lines = (session_dir / "state.jsonl").read_text(encoding="utf-8").splitlines()
            assert len(state_lines) == 2  # one state line per poll, both written despite the blocked pass
        finally:
            release.set()
            runner.shutdown()


class TestOverrunSelfAlert:
    """A poll cycle running past twice the interval self-alerts, edge-triggered."""

    def test_overrun_alert_edge_triggered(self, tmp_path: Path) -> None:
        """The overrun alert fires on entry, stays silent while it persists, and re-fires after it clears."""
        over = OVERRUN_INTERVAL_FACTOR * 5.0 + 1.0
        # Two monotonic reads per poll (cycle start, cycle end): craft the elapsed of each poll.
        durations: Iterator[float] = iter(
            [
                0.0,
                0.0,  # poll 1: fast
                0.0,
                over,  # poll 2: overrun -> fires
                0.0,
                over,  # poll 3: still overrun -> no re-fire
                0.0,
                0.0,  # poll 4: fast -> clears
                0.0,
                over,  # poll 5: overrun again -> re-fires
            ],
        )
        controller = FakeController()
        attach = AttachSupervisor(
            controller,
            session_dir=tmp_path,
            log_dir=None,
            interval=5.0,
            monotonic=lambda: next(durations),
        )

        for now in (1.0, 2.0, 3.0, 4.0, 5.0):
            attach.poll_once(now=now)

        overruns = _alerts(tmp_path, "supervisor_overrun")
        assert len(overruns) == 2
        assert overruns[0]["severity"] == "warning"
        assert overruns[0]["elapsed_seconds"] >= OVERRUN_INTERVAL_FACTOR * 5.0


class TestInboxShutdownExit:
    """An inbox-requested graceful shutdown, once the worker exits, returns from the run loop (no relaunch)."""

    def test_run_forever_returns_after_inbox_shutdown_completes(self, tmp_path: Path) -> None:
        """A GRACEFUL_SHUTDOWN inbox line drives a graceful stop and then exits the loop with a final state line."""
        controller = FakeController()
        (tmp_path / "commands.jsonl").write_text(
            json.dumps({"command": "GRACEFUL_SHUTDOWN"}) + "\n",
            encoding="utf-8",
        )
        attach = AttachSupervisor(controller, session_dir=tmp_path, log_dir=None, interval=0.0)

        finished = threading.Event()

        def _run() -> None:
            attach.run_forever()
            finished.set()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        assert finished.wait(5.0), "run_forever did not return after the inbox shutdown completed"
        assert controller.graceful_stops == 1
        # The worker exited, so the launcher would not relaunch; the loop's return is the completion signal.
        state_lines = (tmp_path / "state.jsonl").read_text(encoding="utf-8").splitlines()
        assert json.loads(state_lines[-1])["worker_up"] is False
