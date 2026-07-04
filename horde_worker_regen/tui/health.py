"""Derive a worker's lifecycle phase and a health checklist from a snapshot.

This is what makes the dashboard feel like a status monitor: it turns the worker's instrumentation
(process states, connectivity flags, fault counters, snapshot freshness) into a single headline phase
plus a set of pass/warn/fail checks. It is pure and dependency-free so it is easy to test and cannot
stall the UI.
"""

from __future__ import annotations

import dataclasses
import enum
import os
import shutil
from pathlib import Path

from horde_worker_regen.process_management.ipc.supervisor_channel import WorkerFatalConfigError, WorkerStateSnapshot
from horde_worker_regen.tui.formatters import human_bytes, human_duration
from horde_worker_regen.tui.worker_launcher import SupervisorStatus

STALE_SNAPSHOT_SECONDS = 20.0
"""No snapshot for this long (while the process is alive) means the worker is likely stuck.

Deliberately well above the worker's snapshot floor (~2s) so a momentarily busy control loop never
reads as "unresponsive". The previous 8s was aggressive enough that a single slow control-loop tick
(or a child blocking the parent on a control-pipe send) flipped the dashboard to UNRESPONSIVE and,
in attached/host setups, churned restarts. The underlying parent-stall is fixed at the source (the
child now drains its control pipe on a dedicated thread), so this is a backstop, not the front line."""

STALE_SNAPSHOT_DOWNLOAD_SECONDS = 90.0
"""More generous staleness budget when the last snapshot showed a model download/load in flight.

A worker fetching weights (base or aux/LoRA) or loading a model is legitimately busy and the operator
expects it to take a while, so the "unresponsive" alarm should hold off longer in that case rather
than cry wolf on a healthy WAN transfer. Applied via :func:`_stale_threshold`."""

IDLE_SECONDS = 600.0
"""No work for this long is treated as an idle/low-demand state rather than active serving."""

_DISK_FLOOR_BYTES = 20 * 1024**3

_INFERENCE_STATES = frozenset(
    {"INFERENCE_STARTING", "POST_PROCESSING", "ALCHEMY_STARTING", "JOB_RECEIVED"},
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
    MAINTENANCE = "maintenance"
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
    *,
    offline_checks: list[HealthCheck] | None = None,
    optimistic_server_maintenance: bool = False,
    fatal_error: WorkerFatalConfigError | None = None,
) -> HealthReport:
    """Compute the worker's phase and health checklist (handles the no-snapshot startup case too).

    ``offline_checks`` are pre-flight checks the caller computes from the filesystem (see
    :func:`build_offline_checks`); they are shown while there is no live snapshot so the panel is
    useful before the worker starts rather than empty. ``fatal_error`` is the worker's reported
    non-retryable config problem (e.g. a taken worker name); when set on a crashed worker it replaces
    the generic crash message with the specific reason and remedy. This function itself stays pure.
    """
    pre_flight = offline_checks or []
    if supervisor_status is SupervisorStatus.STOPPED:
        return HealthReport(
            WorkerPhase.STOPPED, HealthStatus.INFO, "Worker stopped", "The worker is not running.", pre_flight, False
        )
    if supervisor_status is SupervisorStatus.CRASHED:
        if fatal_error is not None:
            return HealthReport(
                WorkerPhase.CRASHED,
                HealthStatus.ERROR,
                fatal_error.title,
                f"{fatal_error.detail} The worker will not restart until this is fixed.",
                pre_flight,
                False,
            )
        return HealthReport(
            WorkerPhase.CRASHED,
            HealthStatus.ERROR,
            "Worker crashed",
            "The worker process exited unexpectedly. Check the logs; it was not (or could not be) restarted.",
            pre_flight,
            False,
        )
    if supervisor_status is SupervisorStatus.RESTARTING:
        return HealthReport(
            WorkerPhase.RESTARTING,
            HealthStatus.WARN,
            "Restarting worker…",
            "Relaunching the worker. This is expected after a manual restart or an automatic recovery.",
            pre_flight,
            True,
        )
    if snapshot is None:
        return HealthReport(
            WorkerPhase.INITIALIZING,
            HealthStatus.INFO,
            "Starting worker…",
            "Loading the model reference and spawning inference processes. The first run can take several minutes.",
            pre_flight,
            True,
        )

    checks = _build_checks(snapshot, snapshot_age, optimistic_server_maintenance=optimistic_server_maintenance)

    stale = snapshot_age is not None and snapshot_age > _stale_threshold(snapshot)
    # A worker that announced it is shutting down and then goes quiet is finishing its teardown, not
    # wedged: ending its inference/safety children and unwinding the control loop legitimately stops it
    # stamping liveness for a stretch. Reading that silence as UNRESPONSIVE is exactly the false alarm
    # operators saw on a clean stop, so shutdown is checked *before* staleness. A genuine hang here is
    # still bounded: the supervisor force-kills a worker that overruns its graceful-stop deadline and the
    # phase then flips to STOPPED.
    if snapshot.shutting_down:
        detail = (
            "Finishing teardown; it has gone quiet while stopping its processes."
            if stale
            else "Finishing in-flight jobs before exit."
        )
        return HealthReport(
            WorkerPhase.SHUTTING_DOWN,
            HealthStatus.INFO,
            "Shutting down…",
            detail,
            checks,
            True,
        )
    if snapshot_age is not None and snapshot_age > _stale_threshold(snapshot):
        return HealthReport(
            WorkerPhase.UNRESPONSIVE,
            HealthStatus.ERROR,
            "Worker not responding",
            f"No update received for {human_duration(snapshot_age)}. The worker may be stuck or overloaded.",
            checks,
            True,
        )
    if snapshot.gpu_torch_incompatible:
        reason = snapshot.gpu_torch_incompatible_reason or (
            "The installed PyTorch has no CUDA kernels for this GPU, so no job can run."
        )
        return HealthReport(
            WorkerPhase.DEGRADED,
            HealthStatus.ERROR,
            "PyTorch cannot run this GPU",
            f"{reason} The worker has stopped popping jobs until this is fixed.",
            checks,
            True,
        )
    if _server_maintenance_active(snapshot, optimistic_server_maintenance=optimistic_server_maintenance):
        return HealthReport(
            WorkerPhase.MAINTENANCE,
            HealthStatus.WARN,
            "Maintenance mode",
            _maintenance_detail(snapshot, optimistic_server_maintenance=optimistic_server_maintenance),
            checks,
            False,
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
    if snapshot.maintenance_mode or snapshot.worker_details_paused:
        return HealthReport(
            WorkerPhase.PAUSED,
            HealthStatus.WARN,
            "Paused",
            _maintenance_detail(snapshot, optimistic_server_maintenance=optimistic_server_maintenance),
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

    if snapshot.last_pop_no_jobs_available:
        detail = "No jobs were available on the last check (low demand or your config matched none)."
        why = summarize_skips(snapshot.last_pop_skipped_reasons)
        if why:
            detail += f" Recently skipped: {why}."
        return HealthReport(
            WorkerPhase.READY,
            HealthStatus.OK,
            "Ready; no jobs available right now",
            detail,
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


def _stale_threshold(snapshot: WorkerStateSnapshot) -> float:
    """The staleness budget for this snapshot: extended while a model download/load is in flight.

    Reads the *last* snapshot we have (possibly the stale one itself): if it shows any process mid
    download or load, a longer silence is expected and tolerated before declaring the worker stuck.
    """
    if any(process.last_process_state in _LOADING_STATES for process in snapshot.processes):
        return STALE_SNAPSHOT_DOWNLOAD_SECONDS
    return STALE_SNAPSHOT_SECONDS


def summarize_skips(skipped_reasons: dict[str, int], *, limit: int = 4) -> str:
    """Render the last pop's skip reasons as a compact, count-ordered phrase ("3 models · 1 nsfw")."""
    ranked = sorted(((reason, count) for reason, count in skipped_reasons.items() if count), key=lambda r: -r[1])
    return " · ".join(f"{count} {reason}" for reason, count in ranked[:limit])


def _server_maintenance_active(
    snapshot: WorkerStateSnapshot,
    *,
    optimistic_server_maintenance: bool = False,
) -> bool:
    """True when the horde, the last pop response, or the TUI's pending command says maintenance is active."""
    return optimistic_server_maintenance or snapshot.worker_details_maintenance or snapshot.last_pop_maintenance_mode


def _maintenance_detail(
    snapshot: WorkerStateSnapshot,
    *,
    optimistic_server_maintenance: bool = False,
) -> str:
    """Explain a paused/maintenance worker, naming the source (horde-forced, self-throttle, or local)."""
    if (
        optimistic_server_maintenance
        and not snapshot.worker_details_maintenance
        and not snapshot.last_pop_maintenance_mode
    ):
        return (
            "The TUI has requested horde maintenance ON and is showing it immediately while the horde "
            "registers the change."
        )
    note_on_maintenance = (
        "NOTE: This happening expectedly means your worker dropped or timed out performing several jobs."
        "Please check your diagnostics and logs to see what the issue is and fix it before clearing maintenance"
        " mode, otherwise the horde will just set it again."
    )

    if snapshot.worker_details_maintenance or snapshot.worker_details_paused:
        what = "maintenance" if snapshot.worker_details_maintenance else "paused"
        return (
            f"The horde has this worker set to {what} (server-side); it will not be given new jobs until "
            "cleared. In-flight jobs finish. Press the Maintenance (horde) key to toggle it."
            f"{note_on_maintenance}"
        )
    if snapshot.last_pop_maintenance_mode:
        return (
            "The job-pop response returned a maintenance-mode error; the horde has stopped sending this "
            "worker jobs. Press the Maintenance (horde) key to clear it."
            f"{note_on_maintenance}"
        )
    if snapshot.self_throttle_paused:
        return (
            "The worker paused itself: too many resource/OOM faults recently, so it backed off to avoid the "
            "horde forcing maintenance. It will resume automatically after a cooldown. In-flight jobs finish."
        )
    return "Not popping new jobs (locally paused). In-flight jobs will finish."


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


def _build_checks(
    snapshot: WorkerStateSnapshot,
    snapshot_age: float | None,
    *,
    optimistic_server_maintenance: bool = False,
) -> list[HealthCheck]:
    """Build the supporting health checklist from a snapshot."""
    checks: list[HealthCheck] = []

    checks.append(_api_check(snapshot, optimistic_server_maintenance=optimistic_server_maintenance))

    if snapshot.gpu_torch_incompatible:
        checks.append(
            HealthCheck(
                "PyTorch/GPU",
                HealthStatus.ERROR,
                snapshot.gpu_torch_incompatible_reason or "Installed PyTorch has no kernels for this GPU.",
            )
        )
    checks.extend(_per_card_checks(snapshot))
    checks.append(_disk_check(snapshot))
    if snapshot.lora_pops_blocked_by_disk:
        checks.append(
            HealthCheck(
                "LoRA",
                HealthStatus.ERROR,
                "Disabled: LoRA cache disk is full and cannot be cleared. Free disk space to restore.",
            )
        )
    if snapshot.post_processing_disabled:
        checks.append(
            HealthCheck(
                "Post-processing",
                HealthStatus.WARN,
                snapshot.post_processing_disabled_reason
                or "Disabled for this session after structural post-processing failures.",
            )
        )

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
        checks.append(HealthCheck("Job health", HealthStatus.OK, "OK"))

    skips = summarize_skips(snapshot.last_pop_skipped_reasons)
    if skips:
        checks.append(HealthCheck("Work", HealthStatus.INFO, f"Last pop skipped: {skips}"))

    if snapshot_age is not None:
        responsive = snapshot_age <= _stale_threshold(snapshot)
        status = HealthStatus.OK if responsive else HealthStatus.ERROR
        checks.append(HealthCheck("Responsiveness", status, f"Last update {human_duration(snapshot_age)} ago"))

    return checks


def _api_check(
    snapshot: WorkerStateSnapshot,
    *,
    optimistic_server_maintenance: bool = False,
) -> HealthCheck:
    """A single check folding API reachability and horde registration into one row.

    Connectivity dominates: an unreachable or backing-off API is reported first, since registration is
    meaningless while the worker cannot talk to the horde. Once reachable, the detail names whether the
    horde has acknowledged this worker so both facts read from one line.
    """
    if _server_maintenance_active(snapshot, optimistic_server_maintenance=optimistic_server_maintenance):
        return HealthCheck("API", HealthStatus.INFO, "Worker is in horde maintenance mode")
    if snapshot.user_info_failed:
        reason = snapshot.user_info_failed_reason or "request failed"
        return HealthCheck("API", HealthStatus.ERROR, f"AI Horde API unreachable ({reason})")
    if snapshot.in_error_backoff:
        return HealthCheck("API", HealthStatus.WARN, "Backing off after pop failures")
    if snapshot.worker_registered:
        return HealthCheck("API", HealthStatus.OK, f"Reachable; registered as {snapshot.config.dreamer_name}")
    return HealthCheck("API", HealthStatus.INFO, "Reachable; not yet acknowledged by the horde")


def _per_card_checks(snapshot: WorkerStateSnapshot) -> list[HealthCheck]:
    """Per-card VRAM-pressure / unservable-model checks, only on a multi-GPU host (quiet on single-GPU).

    A single-GPU host's VRAM and faults are already covered by the GPU and job-health checks, so this stays
    silent there to avoid a redundant row. On a multi-GPU host a healthy fleet gets one reassuring summary,
    while a pressured or quarantining card gets its own named WARN so the operator can see *which* card needs
    attention rather than a blurred worker-wide figure.
    """
    cards = snapshot.per_card
    if len(cards) <= 1:
        return []
    problems: list[HealthCheck] = []
    for card in cards:
        name = f"GPU {card.device_index}"
        if card.is_vram_pressured:
            free = "?" if card.free_vram_mb is None else f"{card.free_vram_mb / 1024:.1f}G"
            problems.append(HealthCheck(name, HealthStatus.WARN, f"VRAM pressure: only {free} free"))
        elif card.unservable_models:
            problems.append(
                HealthCheck(name, HealthStatus.WARN, f"{len(card.unservable_models)} model(s) locally unservable"),
            )
    if problems:
        return problems
    return [HealthCheck("GPUs", HealthStatus.OK, f"{len(cards)} cards healthy")]


def is_gpu_duty_low(snapshot: WorkerStateSnapshot) -> bool:
    """True when the GPU is near-idle while a job is in flight (the low-duty attention condition).

    The duty cycle itself is surfaced in the Trends region rather than as a health row, so this predicate
    lets that region flag the concerning case (a job running against an idle GPU) without duplicating a
    check. Returns False when the duty cycle has not been sampled or no job is running.
    """
    duty = snapshot.gpu_utilization_mean_percent
    if duty is None:
        return False
    busy = any(process.last_process_state in _INFERENCE_STATES for process in snapshot.processes)
    return busy and duty < 5


def _disk_check(snapshot: WorkerStateSnapshot) -> HealthCheck:
    """A free-disk check against the worker's warning floor."""
    if not snapshot.disk_free_bytes:
        return HealthCheck("Disk", HealthStatus.INFO, "Free space not sampled")
    worst_path, worst_free = min(snapshot.disk_free_bytes.items(), key=lambda item: item[1])
    if worst_free < _DISK_FLOOR_BYTES:
        return HealthCheck("Disk", HealthStatus.WARN, f"Low: {human_bytes(worst_free)} free on {worst_path}")
    return HealthCheck("Disk", HealthStatus.OK, f"{human_bytes(worst_free)} free")


def build_offline_checks(config_path: Path) -> list[HealthCheck]:
    """Build pre-flight checks shown while the worker is stopped (config presence, free disk).

    Unlike :func:`derive`, this touches the filesystem: it reads the config file and queries free disk
    space. Both are cheap and wrapped so a failure degrades to an informational check rather than
    raising into the UI tick.
    """
    return [_config_file_check(config_path), _offline_disk_check()]


def _config_file_check(config_path: Path) -> HealthCheck:
    """Check that the bridgeData config exists and parses, for the stopped-worker pre-flight."""
    if not config_path.exists():
        return HealthCheck("Config", HealthStatus.WARN, f"{config_path} not found; run setup to create it")
    from ruamel.yaml import YAMLError

    from horde_worker_regen.tui.config_form import load_config

    try:
        load_config(config_path)
    except (OSError, YAMLError) as config_error:
        return HealthCheck("Config", HealthStatus.ERROR, f"{config_path} is unreadable: {config_error}")
    return HealthCheck("Config", HealthStatus.OK, f"{config_path} present and valid")


def _offline_disk_check() -> HealthCheck:
    """Check free space on the model-cache disk (or the working directory) before the worker starts."""
    cache_home = os.environ.get("AIWORKER_CACHE_HOME")
    target = Path(cache_home) if cache_home else Path.cwd()
    probe = target if target.exists() else Path.cwd()
    try:
        free_bytes = shutil.disk_usage(probe).free
    except OSError as disk_error:
        return HealthCheck("Disk", HealthStatus.INFO, f"Free space unavailable: {disk_error}")
    if free_bytes < _DISK_FLOOR_BYTES:
        return HealthCheck("Disk", HealthStatus.WARN, f"Low: {human_bytes(free_bytes)} free on {probe}")
    return HealthCheck("Disk", HealthStatus.OK, f"{human_bytes(free_bytes)} free on {probe}")
