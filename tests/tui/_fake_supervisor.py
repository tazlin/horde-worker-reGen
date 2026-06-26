"""An in-process supervisor double for driving the TUI in run_test()/Pilot integration tests.

The real [`WorkerSupervisor`][horde_worker_regen.tui.worker_launcher.WorkerSupervisor] spawns a worker
subprocess and streams snapshots over a pipe, which makes a full-app test slow and timing-dependent.
``FakeSupervisor`` satisfies the same [`SupervisorLike`][horde_worker_regen.tui.attach.SupervisorLike]
contract the app depends on, but keeps everything in-process and deterministic.

It serves the two directions of the TUI <-> worker contract separately, on purpose:

* It records each control request the app issues, so a test can assert that a real click reached the
  worker boundary (the click -> command direction).
* It exposes a settable ``latest_snapshot``, so a test can feed back the state the worker would report
  and assert the UI reflects it (the worker-state -> UI direction).

The fake never auto-reflects a recorded command into its snapshot. Mixing the two directions would make a
test restate the fake's own behaviour instead of the app's, which is exactly the tautology these tests
exist to avoid: a workflow test pushes the snapshot a real worker would produce and asserts the rendered
UI, while command delivery is asserted against the recorded requests.
"""

from __future__ import annotations

from dataclasses import dataclass

from horde_worker_regen.process_management.ipc.supervisor_channel import WorkerStateSnapshot
from horde_worker_regen.tui.attach import SupervisorLike
from horde_worker_regen.tui.worker_launcher import SupervisorStatus, WorkerProcessMode


@dataclass(frozen=True)
class RecordedDownloadRequest:
    """One on-demand download the app asked for: the chosen models and whether aux was included."""

    model_names: list[str]
    include_aux: bool


class FakeSupervisor:
    """A deterministic, in-process ``SupervisorLike`` that records control requests for assertions.

    ``alive`` seeds whether the worker starts out running; :meth:`start` and :meth:`stop` flip it so the
    app's "is the worker up?" branches (deferred download-only flush, "worker not running" notices) take
    the realistic path. Tests read the recorded counters/lists and the ordered :attr:`requests` log.
    """

    def __init__(self, *, mode: WorkerProcessMode = WorkerProcessMode.FAKE, alive: bool = False) -> None:
        """Start recording, seeded with the worker's running state and reported mode."""
        self.latest_snapshot: WorkerStateSnapshot | None = None
        self.last_liveness_wall_time: float | None = None
        self._mode = mode
        self._alive = alive
        self._status = SupervisorStatus.RUNNING if alive else SupervisorStatus.STOPPED
        self._restart_attempts = 0

        self.start_calls = 0
        self.stop_calls = 0
        self.restart_calls = 0
        self.close_calls = 0
        self.tick_calls = 0

        self.requests: list[str] = []
        """Every control request, in the order issued, for sequencing assertions (e.g. hold before fetch)."""
        self.pause_calls = 0
        self.resume_calls = 0
        self.reload_config_calls = 0
        self.pause_downloads_calls = 0
        self.resume_downloads_calls = 0
        self.downloads_only_hold_calls = 0
        self.go_live_calls = 0
        self.rate_limits_kbps: list[int] = []
        self.download_requests: list[RecordedDownloadRequest] = []
        self.server_maintenance: list[bool] = []
        self.stats_export: list[bool] = []
        self.set_concurrency_calls: list[tuple[int | None, int | None]] = []
        """Every ``request_set_concurrency`` call as ``(target_processes, target_threads)``."""

    # region lifecycle and status

    @property
    def status(self) -> SupervisorStatus:
        """The worker's current lifecycle status (flipped by start/stop)."""
        return self._status

    @property
    def mode(self) -> WorkerProcessMode:
        """The reported worker mode (fake by default, so the app skips the wizard and update check)."""
        return self._mode

    @property
    def restart_attempts(self) -> int:
        """How many restarts have been attempted (always zero for the fake)."""
        return self._restart_attempts

    def is_alive(self) -> bool:
        """Whether the (fake) worker is currently running."""
        return self._alive

    def tick(self) -> None:
        """Count the supervisor tick the app drives each refresh; the fake has no state to drain."""
        self.tick_calls += 1

    def start(self) -> None:
        """Record a start and mark the worker running, so deferred commands can flush."""
        self.start_calls += 1
        self._alive = True
        self._status = SupervisorStatus.RUNNING

    def stop(self, *, timeout: float = 0.0) -> None:
        """Record a stop and mark the worker not running."""
        self.stop_calls += 1
        self._alive = False
        self._status = SupervisorStatus.STOPPED

    def restart(self) -> None:
        """Record a restart and leave the worker running."""
        self.restart_calls += 1
        self._alive = True
        self._status = SupervisorStatus.RUNNING

    def close(self) -> None:
        """Record the frontend releasing the supervisor and mark the worker not running."""
        self.close_calls += 1
        self._alive = False
        self._status = SupervisorStatus.STOPPED

    # endregion

    # region control requests

    def request_pause(self) -> bool:
        """Record a pop-pause request; True (delivered) only when the worker is running."""
        self.requests.append("pause")
        self.pause_calls += 1
        return self._alive

    def request_resume(self) -> bool:
        """Record a pop-resume request; True only when the worker is running."""
        self.requests.append("resume")
        self.resume_calls += 1
        return self._alive

    def request_drain(self) -> bool:
        """Record a drain request (stop popping, finish in-flight); True only when the worker is running."""
        self.requests.append("drain")
        return self._alive

    def request_set_concurrency(
        self,
        *,
        target_processes: int | None = None,
        target_threads: int | None = None,
    ) -> bool:
        """Record an inference-scaling request; True only when the worker is running."""
        self.requests.append(f"set_concurrency:processes={target_processes}:threads={target_threads}")
        self.set_concurrency_calls.append((target_processes, target_threads))
        return self._alive

    def request_reload_config(self) -> bool:
        """Record a bridgeData reload request; True only when the worker is running."""
        self.requests.append("reload_config")
        self.reload_config_calls += 1
        return self._alive

    def request_pause_downloads(self) -> bool:
        """Record a hold-downloads request; True only when the worker is running."""
        self.requests.append("pause_downloads")
        self.pause_downloads_calls += 1
        return self._alive

    def request_resume_downloads(self) -> bool:
        """Record a resume-downloads request; True only when the worker is running."""
        self.requests.append("resume_downloads")
        self.resume_downloads_calls += 1
        return self._alive

    def request_download_rate_limit(self, rate_limit_kbps: int) -> bool:
        """Record a bandwidth-cap request (in KB/s); True only when the worker is running."""
        self.requests.append("rate_limit")
        self.rate_limits_kbps.append(rate_limit_kbps)
        return self._alive

    def request_downloads_only_hold(self) -> bool:
        """Record a download-only hold request; True only when the worker is running."""
        self.requests.append("downloads_only_hold")
        self.downloads_only_hold_calls += 1
        return self._alive

    def request_go_live(self) -> bool:
        """Record a go-live request; True only when the worker is running."""
        self.requests.append("go_live")
        self.go_live_calls += 1
        return self._alive

    def request_download_models(self, model_names: list[str], *, include_aux: bool) -> bool:
        """Record an on-demand fetch of a chosen model set; True only when the worker is running."""
        self.requests.append("download_models")
        self.download_requests.append(RecordedDownloadRequest(model_names=list(model_names), include_aux=include_aux))
        return self._alive

    def request_set_server_maintenance(self, enabled: bool) -> bool:
        """Record a server-side maintenance toggle; True only when the worker is running."""
        self.requests.append("set_server_maintenance")
        self.server_maintenance.append(enabled)
        return self._alive

    def request_set_stats_export(self, enabled: bool) -> bool:
        """Record a stats-export toggle; True only when the worker is running."""
        self.requests.append("set_stats_export")
        self.stats_export.append(enabled)
        return self._alive

    # endregion


def _assert_satisfies_protocol() -> SupervisorLike:
    """Static guard that ``FakeSupervisor`` still satisfies the contract the app drives.

    A signature drift in ``SupervisorLike`` that the fake no longer matches fails the type checker here,
    rather than surfacing as a confusing runtime ``AttributeError`` deep inside a Pilot test.
    """
    return FakeSupervisor()
