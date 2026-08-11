"""Minimal browser-native overview backed by the worker-host supervisor protocol.

The native page is deliberately a small companion to the Textual dashboard, not a second implementation
of it. One :class:`AttachedWorkerSupervisor` consumes the existing length-prefixed JSON protocol and this
module projects its latest typed snapshot into a stable, intentionally narrow HTTP response. The browser
polls that response and posts a short allow-listed action name; it never reaches the worker-host socket
directly.
"""

from __future__ import annotations

import enum
import functools
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

from aiohttp import web
from pydantic import BaseModel, Field

from horde_worker_regen.process_management.ipc.supervisor_channel import WorkerStateSnapshot, WorkLedgerStage
from horde_worker_regen.tui.attach import AttachedWorkerSupervisor
from horde_worker_regen.tui.worker_launcher import SupervisorStatus

NATIVE_DASHBOARD_PATH = "/native"
NATIVE_STATE_PATH = "/native/api/state"
NATIVE_ACTION_PATH = "/native/api/action"


class NativeDashboardAction(enum.StrEnum):
    """The intentionally small set of worker controls exposed by the native overview."""

    START = "start"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    MAINTENANCE_ON = "maintenance_on"
    MAINTENANCE_OFF = "maintenance_off"


class NativeJobState(BaseModel):
    """Selected progress facts for one active work-ledger entry."""

    job_id: str
    stage: str
    model: str | None = None
    process_id: int | None = None
    device_index: int | None = None
    progress_percent: float | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    iterations_per_second: float | None = None
    age_seconds: float | None = None


class NativeProcessState(BaseModel):
    """Selected liveness, work, and resource facts for one worker process."""

    process_id: int
    process_type: str
    device_index: int
    state: str
    alive: bool
    busy: bool
    model: str | None = None
    current_job_id: str | None = None
    progress_percent: float | None = None
    iterations_per_second: float | None = None
    vram_used_mb: int = 0
    vram_total_mb: int = 0
    heartbeat_age_seconds: float | None = None


class NativeDashboardState(BaseModel):
    """Stable browser-facing projection of the supervisor and its latest worker snapshot."""

    server_time: float = Field(default_factory=time.time)
    connected: bool = False
    lifecycle_resolved: bool = False
    lifecycle: str = SupervisorStatus.STOPPED.value
    worker_running: bool = False
    shutting_down: bool = False

    snapshot_timestamp: float | None = None
    session_start_time: float | None = None
    worker_name: str | None = None
    worker_version: str | None = None
    horde_username: str | None = None

    jobs_popped: int = 0
    jobs_submitted: int = 0
    jobs_faulted: int = 0
    process_recoveries: int = 0
    job_slowdowns: int = 0
    no_jobs_seconds: float = 0.0
    kudos_per_hour: float | None = None
    kudos_this_session: float | None = None
    gpu_duty_percent: float | None = None

    jobs_queued: int = 0
    jobs_active: int = 0
    jobs_post_processing: int = 0
    jobs_safety: int = 0
    jobs_submitting: int = 0

    alchemy_submitted: int = 0
    alchemy_faulted: int = 0
    alchemy_pending: int = 0
    alchemy_active: int = 0
    alchemy_submitting: int = 0

    effective_maintenance: bool = False
    local_paused: bool = False
    horde_maintenance: bool = False
    self_throttle_paused: bool = False
    processes_alive: int = 0
    processes_total: int = 0
    processes_busy: int = 0
    active_jobs: list[NativeJobState] = Field(default_factory=list)
    process_states: list[NativeProcessState] = Field(default_factory=list)
    active_models: list[str] = Field(default_factory=list)
    api_messages: list[str] = Field(default_factory=list)


class NativeActionResult(BaseModel):
    """Result returned after the adapter accepts or rejects one native dashboard action."""

    accepted: bool
    message: str


class NativeSupervisor(Protocol):
    """Supervisor surface used by the adapter, kept small for focused tests."""

    latest_snapshot: WorkerStateSnapshot | None

    @property
    def status(self) -> SupervisorStatus:
        """Current worker lifecycle."""

    @property
    def connected(self) -> bool:
        """Whether the supervisor has a live worker-host socket."""

    @property
    def lifecycle_resolved(self) -> bool:
        """Whether the current connection received an authoritative status frame."""

    def is_alive(self) -> bool:
        """Whether the host reports a live worker."""

    def start(self) -> None:
        """Request worker start."""

    def request_graceful_stop(self, *, timeout: float = 0.0) -> None:
        """Request a non-blocking graceful stop."""

    def request_pause(self) -> bool:
        """Request local pop pause."""

    def request_resume(self) -> bool:
        """Request local pop resume."""

    def request_set_server_maintenance(self, enabled: bool) -> bool:
        """Request the explicit Horde-side maintenance state."""


def _worker_name(snapshot: WorkerStateSnapshot) -> str:
    """Choose the identity matching the workload this worker actually serves."""
    if set(snapshot.enabled_workloads) == {"alchemy"} and snapshot.config.alchemist_name:
        return snapshot.config.alchemist_name
    return snapshot.config.dreamer_name


def _progress_percent(
    current: int | None,
    total: int | None,
    reported_percent: int | None = None,
) -> float | None:
    """Return a bounded progress percentage without inventing progress for an indeterminate task."""
    if reported_percent is not None and 0 <= reported_percent <= 100:
        return float(reported_percent)
    if current is None or not total:
        return None
    return max(0.0, min(100.0, current / total * 100.0))


def _active_job_states(snapshot: WorkerStateSnapshot) -> list[NativeJobState]:
    """Project active work-ledger rows, supplementing progress from their owning process when useful."""
    active_stages = set(WorkLedgerStage) - {WorkLedgerStage.COMPLETED, WorkLedgerStage.FAULTED}
    process_by_job = {process.current_job_id: process for process in snapshot.processes if process.current_job_id}
    jobs: list[NativeJobState] = []
    for entry in snapshot.work_ledger:
        if entry.stage not in active_stages:
            continue
        process = process_by_job.get(entry.job_id)
        current = (
            entry.progress_current
            if entry.progress_current is not None
            else getattr(process, "last_current_step", None)
        )
        total = (
            entry.progress_total if entry.progress_total is not None else getattr(process, "last_total_steps", None)
        )
        reported_percent = getattr(process, "last_heartbeat_percent_complete", None)
        jobs.append(
            NativeJobState(
                job_id=entry.job_id,
                stage=entry.stage.value,
                model=entry.model,
                process_id=entry.process_id,
                device_index=entry.device_index,
                progress_percent=_progress_percent(current, total, reported_percent),
                progress_current=current,
                progress_total=total,
                iterations_per_second=entry.iterations_per_second,
                age_seconds=entry.age_seconds,
            ),
        )
    return jobs


def _process_states(snapshot: WorkerStateSnapshot) -> list[NativeProcessState]:
    """Project the process table into compact semantic rows suitable for both native view densities."""
    states: list[NativeProcessState] = []
    for process in sorted(snapshot.processes, key=lambda item: (item.device_index, item.process_id)):
        heartbeat_age = None
        if process.last_heartbeat_timestamp:
            heartbeat_age = max(0.0, snapshot.timestamp - process.last_heartbeat_timestamp)
        states.append(
            NativeProcessState(
                process_id=process.process_id,
                process_type=process.process_type,
                device_index=process.device_index,
                state=process.last_process_state,
                alive=process.is_alive,
                busy=process.is_busy,
                model=process.loaded_horde_model_name,
                current_job_id=process.current_job_id,
                progress_percent=_progress_percent(
                    process.last_current_step,
                    process.last_total_steps,
                    process.last_heartbeat_percent_complete,
                ),
                iterations_per_second=process.last_iterations_per_second,
                vram_used_mb=process.vram_usage_mb,
                vram_total_mb=process.total_vram_mb,
                heartbeat_age_seconds=heartbeat_age,
            ),
        )
    return states


def build_native_dashboard_state(supervisor: NativeSupervisor) -> NativeDashboardState:
    """Project the latest supervisor state into the native page's narrow public contract."""
    snapshot = supervisor.latest_snapshot
    common = {
        "connected": supervisor.connected,
        "lifecycle_resolved": supervisor.lifecycle_resolved,
        "lifecycle": supervisor.status.value,
        "worker_running": supervisor.is_alive(),
    }
    if snapshot is None:
        return NativeDashboardState(**common)
    process_states = _process_states(snapshot)
    return NativeDashboardState(
        **common,
        shutting_down=snapshot.shutting_down,
        snapshot_timestamp=snapshot.timestamp,
        session_start_time=snapshot.session_start_time or None,
        worker_name=_worker_name(snapshot),
        worker_version=snapshot.config.worker_version,
        horde_username=snapshot.config.horde_username,
        jobs_popped=snapshot.num_jobs_popped,
        jobs_submitted=snapshot.num_jobs_submitted,
        jobs_faulted=snapshot.num_jobs_faulted,
        process_recoveries=snapshot.num_process_recoveries,
        job_slowdowns=snapshot.num_job_slowdowns,
        no_jobs_seconds=snapshot.time_spent_no_jobs_available,
        kudos_per_hour=snapshot.kudos_per_hour,
        kudos_this_session=snapshot.kudos_this_session,
        gpu_duty_percent=snapshot.gpu_utilization_mean_percent,
        jobs_queued=snapshot.jobs_pending_inference,
        jobs_active=snapshot.jobs_in_progress,
        jobs_post_processing=snapshot.jobs_pending_post_processing + snapshot.jobs_being_post_processed,
        jobs_safety=snapshot.jobs_pending_safety_check + snapshot.jobs_being_safety_checked,
        jobs_submitting=snapshot.jobs_pending_submit,
        alchemy_submitted=snapshot.alchemy_total_submitted,
        alchemy_faulted=snapshot.alchemy_total_faulted,
        alchemy_pending=snapshot.alchemy_forms_pending,
        alchemy_active=snapshot.alchemy_forms_in_flight,
        alchemy_submitting=snapshot.alchemy_forms_awaiting_submit,
        effective_maintenance=snapshot.maintenance_mode,
        local_paused=snapshot.supervisor_paused,
        horde_maintenance=snapshot.worker_details_maintenance,
        self_throttle_paused=snapshot.self_throttle_paused,
        processes_alive=sum(process.alive for process in process_states),
        processes_total=len(process_states),
        processes_busy=sum(process.busy for process in process_states),
        active_jobs=_active_job_states(snapshot),
        process_states=process_states,
        active_models=list(snapshot.active_models),
        api_messages=list(snapshot.api_messages[:3]),
    )


class NativeDashboardBridge:
    """Translate native HTTP state/actions to one attached supervisor client."""

    def __init__(self, supervisor: NativeSupervisor) -> None:
        """Store the attached supervisor; the caller owns and closes it."""
        self.supervisor = supervisor

    def state(self) -> NativeDashboardState:
        """Return the current browser-facing state projection."""
        return build_native_dashboard_state(self.supervisor)

    def perform(self, action: NativeDashboardAction) -> NativeActionResult:
        """Perform one explicit action without deriving a toggle from aggregate maintenance state."""
        if action is NativeDashboardAction.START:
            self.supervisor.start()
            return NativeActionResult(accepted=True, message="Worker start requested.")
        if action is NativeDashboardAction.STOP:
            self.supervisor.request_graceful_stop()
            return NativeActionResult(accepted=True, message="Graceful stop requested; in-flight work will drain.")
        if action is NativeDashboardAction.PAUSE:
            accepted = self.supervisor.request_pause()
            return NativeActionResult(
                accepted=accepted,
                message="Local pause requested." if accepted else "Worker host is not connected.",
            )
        if action is NativeDashboardAction.RESUME:
            accepted = self.supervisor.request_resume()
            return NativeActionResult(
                accepted=accepted,
                message="Local resume requested." if accepted else "Worker host is not connected.",
            )
        enable = action is NativeDashboardAction.MAINTENANCE_ON
        accepted = self.supervisor.request_set_server_maintenance(enable)
        return NativeActionResult(
            accepted=accepted,
            message=(
                f"Horde maintenance {'ON' if enable else 'OFF'} requested."
                if accepted
                else "Worker host is not connected."
            ),
        )


@functools.lru_cache(maxsize=1)
def load_native_dashboard_html() -> str:
    """Load and cache the package-owned dependency-free native page."""
    return (Path(__file__).with_name("templates") / "native_dashboard.html").read_text(encoding="utf-8")


class NativeDashboardWeb:
    """Register and own the native page's aiohttp routes and supervisor attachment."""

    def __init__(self, host_address: tuple[str, int]) -> None:
        """Store the loopback worker-host address; attachment begins with aiohttp startup."""
        self._host_address = host_address
        self._supervisor: AttachedWorkerSupervisor | None = None
        self._bridge: NativeDashboardBridge | None = None

    def register(self, app: web.Application) -> None:
        """Add the native page/state/action routes and their paired attach-client cleanup context."""
        app.router.add_get(NATIVE_DASHBOARD_PATH, self.handle_page)
        app.router.add_get(NATIVE_STATE_PATH, self.handle_state)
        app.router.add_post(NATIVE_ACTION_PATH, self.handle_action)
        app.cleanup_ctx.append(self._supervisor_context)

    async def _supervisor_context(self, _app: web.Application) -> AsyncIterator[None]:
        """Own exactly one attach client for the aiohttp application's lifetime."""
        supervisor = AttachedWorkerSupervisor(self._host_address)
        self._supervisor = supervisor
        self._bridge = NativeDashboardBridge(supervisor)
        try:
            yield
        finally:
            self._bridge = None
            self._supervisor = None
            supervisor.close()

    async def handle_page(self, _request: web.Request) -> web.Response:
        """Serve the static native overview page."""
        return web.Response(
            text=load_native_dashboard_html(),
            content_type="text/html",
            charset="utf-8",
            headers={"Cache-Control": "no-store"},
        )

    async def handle_state(self, _request: web.Request) -> web.Response:
        """Return the newest selected supervisor facts, never the full snapshot/config payload."""
        if self._bridge is None:
            raise web.HTTPServiceUnavailable(text="Native dashboard is still attaching.")
        return web.json_response(
            self._bridge.state().model_dump(mode="json"),
            headers={"Cache-Control": "no-store"},
        )

    async def handle_action(self, request: web.Request) -> web.Response:
        """Validate and dispatch one same-origin JSON control action."""
        if self._bridge is None:
            raise web.HTTPServiceUnavailable(text="Native dashboard is still attaching.")
        if request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType(text="Actions require application/json.")
        try:
            payload = await request.json()
            raw_action = payload.get("action") if isinstance(payload, dict) else None
            action = NativeDashboardAction(raw_action)
        except (ValueError, TypeError):
            raise web.HTTPBadRequest(text="Unknown native dashboard action.") from None
        result = self._bridge.perform(action)
        return web.json_response(result.model_dump(mode="json"), status=202 if result.accepted else 503)
