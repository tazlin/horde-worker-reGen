"""Derive a worker's lifecycle phase and a health checklist from a snapshot.

This is what makes the dashboard feel like a status monitor: it turns the worker's instrumentation
(process states, connectivity flags, fault counters, snapshot freshness) into a single headline phase
plus a set of pass/warn/fail checks. It is pure and dependency-free so it is easy to test and cannot
stall the UI.
"""

from __future__ import annotations

import dataclasses
import enum

from horde_worker_regen.process_management.supervisor_channel import WorkerStateSnapshot
from horde_worker_regen.tui.formatters import human_bytes, human_duration
from horde_worker_regen.tui.worker_launcher import SupervisorStatus

STALE_SNAPSHOT_SECONDS = 8.0
"""No snapshot for this long (while the process is alive) means the worker is likely stuck."""

IDLE_SECONDS = 600.0
"""No work for this long is treated as an idle/low-demand state rather than active serving."""

_DISK_FLOOR_BYTES = 20 * 1024**3

_INFERENCE_STATES = frozenset(
    {"INFERENCE_STARTING", "INFERENCE_POST_PROCESSING", "ALCHEMY_STARTING", "JOB_RECEIVED"},
)
_READY_STATES = frozenset({"WAITING_FOR_JOB", "INFERENCE_COMPLETE", "ALCHEMY_COMPLETE", "PRELOADED_MODEL"})
_LOADING_STATES = frozenset(
    {"PROCESS_STARTING", "DOWNLOADING_MODEL", "DOWNLOADING_AUX_MODEL", "PRELOADING_MODEL"},
)


class HealthStatus(enum.IntEnum):
    """A single check's outcome, ordered so worse outcomes compare greater."""

    OK = 0
    INFO = 1
    WARN = 2
    ERROR = 3

    @property
    def glyph(self) -> str:
        """A status glyph for the checklist."""
        return {HealthStatus.OK: "✓", HealthStatus.INFO: "•", HealthStatus.WARN: "⚠", HealthStatus.ERROR: "✗"}[self]

    @property
    def colour(self) -> str:
        """A Rich colour for this status."""
        return {
            HealthStatus.OK: "green",
            HealthStatus.INFO: "grey62",
            HealthStatus.WARN: "yellow",
            HealthStatus.ERROR: "bold red",
        }[self]


class WorkerPhase(enum.StrEnum):
    """The worker's headline lifecycle phase."""

    STOPPED = "stopped"
    CRASHED = "crashed"
    RESTARTING = "restarting"
    INITIALIZING = "initializing"
    WARMING_UP = "warming up"
    SERVING = "serving"
    READY = "ready"
    IDLE = "idle"
    PAUSED = "paused"
    SHUTTING_DOWN = "shutting down"
    DEGRADED = "degraded"
    DISCONNECTED = "disconnected"
    UNRESPONSIVE = "unresponsive"


@dataclasses.dataclass(frozen=True)
class HealthCheck:
    """One named health check and its outcome."""

    name: str
    status: HealthStatus
    detail: str


@dataclasses.dataclass(frozen=True)
class HealthReport:
    """The derived headline phase plus the supporting checklist."""

    phase: WorkerPhase
    severity: HealthStatus
    headline: str
    detail: str
    checks: list[HealthCheck]
    animated: bool
    """Whether the hero indicator should pulse/spin (work in progress or attention needed)."""


def derive(
    snapshot: WorkerStateSnapshot | None,
    supervisor_status: SupervisorStatus,
    snapshot_age: float | None,
) -> HealthReport:
    """Compute the worker's phase and health checklist (handles the no-snapshot startup case too)."""
    if supervisor_status is SupervisorStatus.STOPPED:
        return HealthReport(
            WorkerPhase.STOPPED, HealthStatus.INFO, "Worker stopped", "The worker is not running.", [], False
        )
    if supervisor_status is SupervisorStatus.CRASHED:
        return HealthReport(
            WorkerPhase.CRASHED,
            HealthStatus.ERROR,
            "Worker crashed",
            "The worker process exited unexpectedly. Check the logs; it was not (or could not be) restarted.",
            [],
            False,
        )
    if supervisor_status is SupervisorStatus.RESTARTING:
        return HealthReport(
            WorkerPhase.RESTARTING,
            HealthStatus.WARN,
            "Restarting worker…",
            "Relaunching after an unexpected exit.",
            [],
            True,
        )
    if snapshot is None:
        return HealthReport(
            WorkerPhase.INITIALIZING,
            HealthStatus.INFO,
            "Starting worker…",
            "Loading the model reference and spawning inference processes. The first run can take several minutes.",
            [],
            True,
        )

    checks = _build_checks(snapshot, snapshot_age)

    if snapshot_age is not None and snapshot_age > STALE_SNAPSHOT_SECONDS:
        return HealthReport(
            WorkerPhase.UNRESPONSIVE,
            HealthStatus.ERROR,
            "Worker not responding",
            f"No update received for {human_duration(snapshot_age)}. The worker may be stuck or overloaded.",
            checks,
            True,
        )
    if snapshot.shutting_down:
        return HealthReport(
            WorkerPhase.SHUTTING_DOWN,
            HealthStatus.INFO,
            "Shutting down…",
            "Finishing in-flight jobs before exit.",
            checks,
            True,
        )
    if snapshot.user_info_failed or snapshot.in_error_backoff:
        reason = snapshot.user_info_failed_reason or "Repeated job-pop failures; the server or network is unreachable."
        return HealthReport(
            WorkerPhase.DISCONNECTED,
            HealthStatus.ERROR,
            "Connection problem",
            f"Cannot reach the AI Horde API. {reason}",
            checks,
            False,
        )
    if snapshot.too_many_consecutive_failed_jobs:
        return HealthReport(
            WorkerPhase.DEGRADED,
            HealthStatus.ERROR,
            "Repeated job failures",
            f"{snapshot.consecutive_failed_jobs} jobs failed in a row; the worker paused to recover. Check the logs.",
            checks,
            False,
        )
    if snapshot.maintenance_mode:
        return HealthReport(
            WorkerPhase.PAUSED,
            HealthStatus.WARN,
            "Paused",
            "Not popping new jobs. In-flight jobs will finish.",
            checks,
            False,
        )

    serving = any(process.last_process_state in _INFERENCE_STATES for process in snapshot.processes)
    if serving:
        kudos = "" if snapshot.kudos_per_hour is None else f" · {snapshot.kudos_per_hour:,.0f} kudos/hr"
        return HealthReport(
            WorkerPhase.SERVING,
            HealthStatus.OK,
            f"Serving: {snapshot.num_jobs_submitted} jobs done{kudos}",
            "Generating for the horde.",
            checks,
            True,
        )

    if _is_warming_up(snapshot):
        what, detail = _warmup_detail(snapshot)
        return HealthReport(WorkerPhase.WARMING_UP, HealthStatus.INFO, f"Warming up: {what}", detail, checks, True)

    idle_for = snapshot.seconds_since_last_pop
    if snapshot.time_spent_no_jobs_available > IDLE_SECONDS or (idle_for is not None and idle_for > IDLE_SECONDS):
        return HealthReport(
            WorkerPhase.IDLE,
            HealthStatus.INFO,
            "Idle; waiting for work",
            "No jobs have been available recently (low demand). Offering more models or raising max_power can help.",
            checks,
            False,
        )

    return HealthReport(
        WorkerPhase.READY,
        HealthStatus.OK,
        "Ready; waiting for the next job",
        "Models are loaded and the worker is ready to serve.",
        checks,
        False,
    )


def _is_warming_up(snapshot: WorkerStateSnapshot) -> bool:
    """True before any process has become ready/serving (startup, first model load)."""
    if not snapshot.processes:
        return True
    if any(process.last_process_state in (_READY_STATES | _INFERENCE_STATES) for process in snapshot.processes):
        return False
    return any(process.last_process_state in _LOADING_STATES for process in snapshot.processes)


def _warmup_detail(snapshot: WorkerStateSnapshot) -> tuple[str, str]:
    """Describe what the worker is doing while warming up (downloading/loading which model)."""
    for process in snapshot.processes:
        model = process.loaded_horde_model_name or "models"
        if process.last_process_state in ("DOWNLOADING_MODEL", "DOWNLOADING_AUX_MODEL"):
            return f"downloading {model}", "Fetching model weights. First-time downloads can be large."
        if process.last_process_state == "PRELOADING_MODEL":
            return f"loading {model} into VRAM", "Moving the model onto the GPU."
    return "starting processes", "Spawning and initialising the inference processes."


def _build_checks(snapshot: WorkerStateSnapshot, snapshot_age: float | None) -> list[HealthCheck]:
    """Build the supporting health checklist from a snapshot."""
    checks: list[HealthCheck] = []

    if snapshot.user_info_failed:
        reason = snapshot.user_info_failed_reason or "request failed"
        checks.append(HealthCheck("API connectivity", HealthStatus.ERROR, f"AI Horde API unreachable ({reason})"))
    elif snapshot.in_error_backoff:
        checks.append(HealthCheck("API connectivity", HealthStatus.WARN, "Backing off after pop failures"))
    else:
        checks.append(HealthCheck("API connectivity", HealthStatus.OK, "AI Horde API reachable"))

    if snapshot.worker_registered:
        checks.append(
            HealthCheck("Registration", HealthStatus.OK, f"Known to the horde as {snapshot.config.dreamer_name}")
        )
    else:
        checks.append(HealthCheck("Registration", HealthStatus.INFO, "Not yet acknowledged by the horde"))

    alive = sum(1 for process in snapshot.processes if process.is_alive)
    total = len(snapshot.processes)
    if total and alive == total:
        checks.append(HealthCheck("Processes", HealthStatus.OK, f"{alive}/{total} worker processes alive"))
    elif total:
        checks.append(HealthCheck("Processes", HealthStatus.WARN, f"{alive}/{total} worker processes alive"))
    else:
        checks.append(HealthCheck("Processes", HealthStatus.INFO, "No processes reported yet"))

    if snapshot.active_models:
        checks.append(HealthCheck("Models", HealthStatus.OK, f"{len(snapshot.active_models)} model(s) loaded"))
    else:
        checks.append(HealthCheck("Models", HealthStatus.INFO, "No model loaded yet"))

    checks.append(_gpu_check(snapshot))
    checks.append(_disk_check(snapshot))

    if snapshot.too_many_consecutive_failed_jobs:
        checks.append(
            HealthCheck("Job health", HealthStatus.ERROR, f"{snapshot.consecutive_failed_jobs} consecutive failures")
        )
    elif snapshot.consecutive_failed_jobs > 0:
        checks.append(
            HealthCheck("Job health", HealthStatus.WARN, f"{snapshot.consecutive_failed_jobs} consecutive failure(s)")
        )
    elif snapshot.num_process_recoveries > 0:
        checks.append(
            HealthCheck("Job health", HealthStatus.WARN, f"{snapshot.num_process_recoveries} process recover(ies)")
        )
    else:
        checks.append(HealthCheck("Job health", HealthStatus.OK, "No recent failures or recoveries"))

    if snapshot_age is not None:
        responsive = snapshot_age <= STALE_SNAPSHOT_SECONDS
        status = HealthStatus.OK if responsive else HealthStatus.ERROR
        checks.append(HealthCheck("Responsiveness", status, f"Last update {human_duration(snapshot_age)} ago"))

    return checks


def _gpu_check(snapshot: WorkerStateSnapshot) -> HealthCheck:
    """A GPU-activity check from the sampled duty cycle (when available)."""
    duty = snapshot.gpu_utilization_mean_percent
    if duty is None:
        return HealthCheck("GPU", HealthStatus.INFO, "Utilisation not sampled")
    busy = any(process.last_process_state in _INFERENCE_STATES for process in snapshot.processes)
    if busy and duty < 5:
        return HealthCheck("GPU", HealthStatus.WARN, f"Idle ({duty:.0f}%) while a job is running")
    return HealthCheck("GPU", HealthStatus.OK, f"{duty:.0f}% duty cycle")


def _disk_check(snapshot: WorkerStateSnapshot) -> HealthCheck:
    """A free-disk check against the worker's warning floor."""
    if not snapshot.disk_free_bytes:
        return HealthCheck("Disk", HealthStatus.INFO, "Free space not sampled")
    worst_path, worst_free = min(snapshot.disk_free_bytes.items(), key=lambda item: item[1])
    if worst_free < _DISK_FLOOR_BYTES:
        return HealthCheck("Disk", HealthStatus.WARN, f"Low: {human_bytes(worst_free)} free on {worst_path}")
    return HealthCheck("Disk", HealthStatus.OK, f"{human_bytes(worst_free)} free")
