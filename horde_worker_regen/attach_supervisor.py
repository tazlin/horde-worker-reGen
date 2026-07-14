"""A headless attach supervisor: run the worker, observe it through files, control it through a file inbox.

This is the file-driven counterpart to the TUI and the socket-serving worker host. It owns exactly one
[`WorkerSupervisor`][horde_worker_regen.tui.worker_launcher.WorkerSupervisor] (reusing its spawn,
auto-restart, wedge backstop, and orphan-proof shutdown), and instead of a UI or a socket it exposes the
worker to an autonomous operator through three JSONL files under a session directory:

* ``state.jsonl`` -- one compact line per interval summarizing the latest worker snapshot (liveness, per
  process states, queue depth, job counters, kudos/hr, download progress, maintenance, duty).
* ``alerts.jsonl`` -- append-only, edge-triggered alerts worth waking an operator: the live-log
  [`watch_pass`][horde_worker_regen.analysis.watch.watch_pass] findings plus snapshot threshold rules.
* ``commands.jsonl`` -- an inbox the supervisor polls each interval; each line maps onto a
  [`SupervisorControlMessage`][horde_worker_regen.process_management.ipc.supervisor_channel.SupervisorControlMessage]
  and is applied exactly once.

A single pre-authorized auto-guard sets server-side maintenance on when a GPU-idle-with-pending or a
frozen-parent condition persists past a confirmation window; it never restarts or shuts the worker down.

The process is torch-free by contract (it runs the orchestrator role, not inference): everything imported
here stays out of the torch/textual import chain, mirroring the TUI supervisor and the worker host.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing
import os
import signal
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Protocol

from loguru import logger
from pydantic import ValidationError

from horde_worker_regen.analysis.bundle import LogBundle
from horde_worker_regen.analysis.watch import WatchState, watch_pass
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    DownloadPhase,
    SupervisorCommand,
    SupervisorControlMessage,
    WorkerStateSnapshot,
)
from horde_worker_regen.run_worker import WorkerLaunchOptions

DEFAULT_INTERVAL_SECONDS = 5.0
"""How often the loop ticks the worker, samples state, and re-evaluates the rules and the command inbox."""

LIVENESS_STALE_SECONDS = 30.0
"""A worker liveness stamp older than this (the loop stopped advancing) reads as a frozen parent."""

GPU_IDLE_PENDING_SECONDS = 120.0
"""Pending work while every inference process is idle for this long is a GPU-idle-with-pending stall."""

DOWNLOAD_STALL_SECONDS = 120.0
"""An in-flight download whose byte count has not advanced for this long reads as a stuck download."""

FAULT_BURST_WINDOW_SECONDS = 600.0
"""The trailing window (10 minutes) the fault-burst rule counts job faults over."""

FAULT_BURST_COUNT = 5
"""Job faults within :data:`FAULT_BURST_WINDOW_SECONDS` at or above this fire the fault-burst rule."""

AUTO_GUARD_CONFIRM_SECONDS = 180.0
"""A guard-eligible condition must persist continuously this long before the auto-guard acts (one shot)."""

STATE_LINE_MAX_BYTES = 2048
"""Hard cap on a single ``state.jsonl`` line; per-process rows are dropped to keep a line under it."""

_SEVERITY_WARNING = "warning"
_SEVERITY_CRITICAL = "critical"
_SEVERITY_INFO = "info"


class WorkerController(Protocol):
    """The slice of :class:`WorkerSupervisor` the attach loop drives (so tests can supply a fake)."""

    latest_snapshot: WorkerStateSnapshot | None
    last_liveness_wall_time: float | None

    def tick(self) -> None:
        """Drain pending snapshots and advance the worker lifecycle."""

    def send_command(self, command: SupervisorControlMessage) -> bool:
        """Send a control command to the worker; False if the transport is unusable."""

    def request_graceful_stop(self, *, timeout: float = ...) -> None:
        """Begin a non-blocking, orphan-proof graceful shutdown of the worker."""


def _resolve_command(name: str) -> SupervisorCommand | None:
    """Map an inbox ``command`` string onto a :class:`SupervisorCommand` (case-insensitive), or None.

    ``GRACEFUL_SHUTDOWN`` is accepted as an alias for the real ``SHUTDOWN`` verb; every other verb is the
    enum member name, so the inbox supports the full command surface without a hand-maintained table.
    """
    key = name.strip().upper()
    if key == "GRACEFUL_SHUTDOWN":
        key = SupervisorCommand.SHUTDOWN.name
    return SupervisorCommand.__members__.get(key)


# The optional per-command fields carried on SupervisorControlMessage; only those present in an inbox line
# are forwarded, so a command line stays minimal and pydantic validates the values.
_CONTROL_MESSAGE_FIELDS = (
    "process_id",
    "target_threads",
    "target_processes",
    "download_rate_limit_kbps",
    "server_maintenance_enabled",
    "download_model_names",
    "download_include_aux",
    "stats_export_enabled",
)


def _build_control_message(command: SupervisorCommand, data: dict[str, object]) -> SupervisorControlMessage:
    """Build a control message for ``command`` from an inbox line's fields (pydantic validates the values)."""
    fields = {key: data[key] for key in _CONTROL_MESSAGE_FIELDS if key in data}
    return SupervisorControlMessage(command=command, **fields)  # type: ignore[arg-type]


def _inference_processes_all_idle(snapshot: WorkerStateSnapshot) -> bool:
    """Whether the worker has at least one inference process and every one of them is currently idle."""
    inference = [process for process in snapshot.processes if process.process_type == "INFERENCE"]
    if not inference:
        return False
    return all(not process.is_busy for process in inference)


class AttachSupervisor:
    """Drive one worker through observation files and a command inbox on a fixed interval.

    The loop is deliberately a pure state machine over an injected :class:`WorkerController`: each
    :meth:`poll_once` ticks the worker, applies any new inbox commands, writes one state line, and appends
    the edge-triggered alerts (live-log findings plus snapshot threshold rules). All rule state lives here,
    so a test can drive it with a fake controller and an explicit clock, no worker or GPU required.
    """

    def __init__(
        self,
        controller: WorkerController,
        *,
        session_dir: Path,
        log_dir: Path | None = None,
        interval: float = DEFAULT_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Bind the loop to a controller and a session directory (creating it); does not start ticking."""
        self._controller = controller
        self._session_dir = session_dir
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = session_dir / "state.jsonl"
        self._alerts_path = session_dir / "alerts.jsonl"
        self._commands_path = session_dir / "commands.jsonl"
        self._log_dir = log_dir
        self._interval = interval
        self._clock = clock

        self._stop = threading.Event()

        self._watch_state = WatchState()
        self._fired: dict[str, bool] = {}
        self._since: dict[str, float | None] = {}
        self._faults: list[tuple[float, int]] = []
        self._download_last_bytes: int | None = None
        self._download_stall_since: float | None = None
        self._commands_consumed = 0
        self._guard_since: float | None = None
        self._guard_fired = False

    # region loop

    def run_forever(self) -> None:
        """Poll on the interval until :meth:`stop`. A poll error is logged and the loop continues."""
        logger.info(f"Attach supervisor writing session files to {self._session_dir} (interval {self._interval}s).")
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception:  # noqa: BLE001 - one bad poll must not tear the observation loop down
                logger.exception("Attach supervisor poll failed; continuing.")
            self._stop.wait(self._interval)

    def stop(self) -> None:
        """Signal the loop to finish its current poll and return from :meth:`run_forever`."""
        self._stop.set()

    def poll_once(self, now: float | None = None) -> None:
        """Run one observation/control cycle: tick, apply inbox, write state, append alerts."""
        now = self._clock() if now is None else now
        self._controller.tick()
        self._process_command_inbox(now)

        snapshot = self._controller.latest_snapshot
        liveness = self._controller.last_liveness_wall_time
        self._write_state_line(snapshot, liveness, now)

        alerts: list[dict[str, object]] = list(self._log_watch_alerts(now))
        if snapshot is not None:
            alerts.extend(self._threshold_alerts(snapshot, liveness, now))
            guard_alert = self._evaluate_guard(snapshot, liveness, now)
            if guard_alert is not None:
                alerts.append(guard_alert)
        for alert in alerts:
            self._append_jsonl(self._alerts_path, alert)

    # endregion

    # region observation

    def _state_dict(
        self,
        snapshot: WorkerStateSnapshot | None,
        liveness: float | None,
        now: float,
    ) -> dict[str, object]:
        """Project the latest snapshot into the compact per-interval state record."""
        if snapshot is None:
            return {"t": round(now, 3), "worker_up": False}

        procs = [
            {
                "id": process.process_id,
                "type": process.process_type,
                "state": process.last_process_state,
                "busy": process.is_busy,
                "model": process.loaded_horde_model_name,
            }
            for process in snapshot.processes
        ]
        download: dict[str, object] | None = None
        if snapshot.downloads is not None:
            current = snapshot.downloads.current
            download = {
                "phase": str(snapshot.downloads.phase),
                "file": current.model_name if current is not None else None,
                "done_bytes": current.downloaded_bytes if current is not None else None,
                "total_bytes": current.total_bytes if current is not None else None,
            }
        duty: float | None = None
        if snapshot.latest_stats_sample is not None:
            duty = snapshot.latest_stats_sample.gpu_duty_percent
        if duty is None:
            duty = snapshot.gpu_utilization_busy_fraction

        return {
            "t": round(now, 3),
            "worker_up": True,
            "snap_t": round(snapshot.timestamp, 3),
            "liveness_age": None if liveness is None else round(now - liveness, 1),
            "procs": procs,
            "queue": snapshot.jobs_pending_inference,
            "in_progress": snapshot.jobs_in_progress,
            "popped": snapshot.num_jobs_popped,
            "submitted": snapshot.num_jobs_submitted,
            "faulted": snapshot.num_jobs_faulted,
            "kudos_hr": snapshot.kudos_per_hour,
            "maintenance": snapshot.maintenance_mode,
            "server_maintenance": snapshot.worker_details_maintenance,
            "download": download,
            "gpu_duty": duty,
        }

    def _write_state_line(
        self,
        snapshot: WorkerStateSnapshot | None,
        liveness: float | None,
        now: float,
    ) -> None:
        """Write one bounded state line, shedding per-process rows if it would exceed the byte cap."""
        record = self._state_dict(snapshot, liveness, now)
        line = json.dumps(record, separators=(",", ":"), default=str)
        procs = record.get("procs")
        while len(line.encode("utf-8")) > STATE_LINE_MAX_BYTES and isinstance(procs, list) and procs:
            procs.pop()
            record["procs_truncated"] = True
            line = json.dumps(record, separators=(",", ":"), default=str)
        self._append_line(self._state_path, line)

    def _log_watch_alerts(self, now: float) -> list[dict[str, object]]:
        """Run one incremental live-log watch pass and turn its change-only strings into alert records."""
        if self._log_dir is None or not self._log_dir.exists():
            return []
        try:
            bundle = LogBundle.from_path(self._log_dir)
            messages, self._watch_state = watch_pass(bundle, self._watch_state)
        except Exception:  # noqa: BLE001 - a malformed/rotating log must not break the observation loop
            logger.exception("Live-log watch pass failed; skipping this interval.")
            return []
        return [self._log_alert(now, message) for message in messages]

    @staticmethod
    def _log_alert(now: float, message: str) -> dict[str, object]:
        """Wrap a watch_pass string as an alert record, lifting its embedded severity when present."""
        lowered = message.lower()
        severity = _SEVERITY_INFO
        if "[critical]" in lowered:
            severity = _SEVERITY_CRITICAL
        elif "[warning]" in lowered:
            severity = _SEVERITY_WARNING
        return {"t": round(now, 3), "rule": "log_watch", "severity": severity, "summary": message}

    # endregion

    # region threshold rules

    def _threshold_alerts(
        self,
        snapshot: WorkerStateSnapshot,
        liveness: float | None,
        now: float,
    ) -> list[dict[str, object]]:
        """Evaluate every snapshot threshold rule; each yields at most one alert per rising edge."""
        alerts: list[dict[str, object]] = []
        for candidate in (
            self._rule_frozen_parent(liveness, now),
            self._rule_consecutive_failure_pause(snapshot, now),
            self._rule_fault_burst(snapshot, now),
            self._rule_gpu_idle_with_pending(snapshot, now),
            self._rule_download_no_progress(snapshot, now),
        ):
            if candidate is not None:
                alerts.append(candidate)
        return alerts

    def _edge(self, rule_id: str, active: bool) -> bool:
        """Return True only on the rising edge of ``active`` for ``rule_id`` (re-arms once it goes false)."""
        was_fired = self._fired.get(rule_id, False)
        if active and not was_fired:
            self._fired[rule_id] = True
            return True
        if not active:
            self._fired[rule_id] = False
        return False

    def _sustained(self, rule_id: str, condition: bool, now: float, threshold: float) -> bool:
        """Whether ``condition`` has held continuously for ``threshold`` seconds (tracking its onset)."""
        if not condition:
            self._since[rule_id] = None
            return False
        since = self._since.get(rule_id)
        if since is None:
            since = now
            self._since[rule_id] = since
        return (now - since) >= threshold

    def _rule_frozen_parent(self, liveness: float | None, now: float) -> dict[str, object] | None:
        """The worker's control-loop liveness stamp has aged past the frozen-parent threshold."""
        age = None if liveness is None else now - liveness
        active = age is not None and age > LIVENESS_STALE_SECONDS
        if not self._edge("frozen_parent", active):
            return None
        return {
            "t": round(now, 3),
            "rule": "frozen_parent",
            "severity": _SEVERITY_CRITICAL,
            "summary": f"Worker control-loop liveness is {age:.0f}s stale (> {LIVENESS_STALE_SECONDS:.0f}s): "
            "the parent may be frozen.",
            "liveness_age_seconds": round(age or 0.0, 1),
        }

    def _rule_consecutive_failure_pause(self, snapshot: WorkerStateSnapshot, now: float) -> dict[str, object] | None:
        """The worker has armed its consecutive-failure pop pause."""
        if not self._edge("consecutive_failure_pause", bool(snapshot.too_many_consecutive_failed_jobs)):
            return None
        return {
            "t": round(now, 3),
            "rule": "consecutive_failure_pause",
            "severity": _SEVERITY_WARNING,
            "summary": f"Worker armed its consecutive-failure pause ({snapshot.consecutive_failed_jobs} "
            "consecutive faults); it stopped popping to protect itself.",
            "consecutive_failed_jobs": snapshot.consecutive_failed_jobs,
        }

    def _rule_fault_burst(self, snapshot: WorkerStateSnapshot, now: float) -> dict[str, object] | None:
        """More than the allowed number of job faults landed inside the trailing fault window."""
        faults_in_window = self._update_fault_window(now, snapshot.num_jobs_faulted)
        if not self._edge("fault_burst", faults_in_window >= FAULT_BURST_COUNT):
            return None
        return {
            "t": round(now, 3),
            "rule": "fault_burst",
            "severity": _SEVERITY_WARNING,
            "summary": f"{faults_in_window} job fault(s) in the last {FAULT_BURST_WINDOW_SECONDS / 60:.0f} "
            f"minutes (>= {FAULT_BURST_COUNT}).",
            "faults_in_window": faults_in_window,
            "window_seconds": FAULT_BURST_WINDOW_SECONDS,
        }

    def _update_fault_window(self, now: float, cumulative_faulted: int) -> int:
        """Track the cumulative fault counter and return how many faults fell inside the trailing window.

        The snapshot's ``num_jobs_faulted`` is monotonic, so faults-in-window is the current value minus the
        value at the window's start (the newest sample at or before it, or the session's first sample early
        on). Samples older than the window are pruned, keeping one anchor just before it.
        """
        self._faults.append((now, cumulative_faulted))
        window_start = now - FAULT_BURST_WINDOW_SECONDS
        anchor = self._faults[0][1]
        for sample_time, sample_value in self._faults:
            if sample_time <= window_start:
                anchor = sample_value
            else:
                break
        in_window = [sample for sample in self._faults if sample[0] >= window_start]
        older = [sample for sample in self._faults if sample[0] < window_start]
        self._faults = ([older[-1]] if older else []) + in_window
        return cumulative_faulted - anchor

    def _rule_gpu_idle_with_pending(self, snapshot: WorkerStateSnapshot, now: float) -> dict[str, object] | None:
        """Jobs are pending while every inference process has sat idle past the stall threshold."""
        condition = snapshot.jobs_pending_inference > 0 and _inference_processes_all_idle(snapshot)
        active = self._sustained("gpu_idle_with_pending", condition, now, GPU_IDLE_PENDING_SECONDS)
        if not self._edge("gpu_idle_with_pending", active):
            return None
        since = self._since.get("gpu_idle_with_pending")
        idle_seconds = now - (now if since is None else since)
        return {
            "t": round(now, 3),
            "rule": "gpu_idle_with_pending",
            "severity": _SEVERITY_CRITICAL,
            "summary": f"{snapshot.jobs_pending_inference} job(s) pending while every inference process has "
            f"been idle for {idle_seconds:.0f}s (> {GPU_IDLE_PENDING_SECONDS:.0f}s): the GPU is not being fed.",
            "pending_jobs": snapshot.jobs_pending_inference,
            "idle_seconds": round(idle_seconds, 1),
        }

    def _rule_download_no_progress(self, snapshot: WorkerStateSnapshot, now: float) -> dict[str, object] | None:
        """An active download whose byte count has not advanced past the stall threshold."""
        downloads = snapshot.downloads
        active = (
            downloads is not None and downloads.phase is DownloadPhase.DOWNLOADING and downloads.current is not None
        )
        if not active or downloads is None or downloads.current is None:
            self._download_last_bytes = None
            self._download_stall_since = None
            self._fired["download_no_progress"] = False
            return None

        current_bytes = downloads.current.downloaded_bytes
        if self._download_last_bytes is None or current_bytes != self._download_last_bytes:
            # Fresh progress (or the first sighting): reset the stall clock and re-arm the rule.
            self._download_last_bytes = current_bytes
            self._download_stall_since = now
            self._fired["download_no_progress"] = False
            return None

        stall_since = self._download_stall_since
        stall_seconds = now - (now if stall_since is None else stall_since)
        if not self._edge("download_no_progress", stall_seconds >= DOWNLOAD_STALL_SECONDS):
            return None
        return {
            "t": round(now, 3),
            "rule": "download_no_progress",
            "severity": _SEVERITY_WARNING,
            "summary": f"Download of {downloads.current.model_name} made no byte progress for "
            f"{stall_seconds:.0f}s (> {DOWNLOAD_STALL_SECONDS:.0f}s).",
            "model": downloads.current.model_name,
            "downloaded_bytes": current_bytes,
            "stall_seconds": round(stall_seconds, 1),
        }

    # endregion

    # region auto-guard

    def _evaluate_guard(
        self,
        snapshot: WorkerStateSnapshot,
        liveness: float | None,
        now: float,
    ) -> dict[str, object] | None:
        """Set server maintenance on (once) when a guard-eligible condition persists past confirmation.

        The two guard conditions are the frozen-parent stall and GPU-idle-with-pending. When either holds
        continuously for :data:`AUTO_GUARD_CONFIRM_SECONDS`, the guard sends a single
        ``SET_SERVER_MAINTENANCE=true`` so the horde stops routing jobs to a worker that cannot serve them,
        and records why. It never restarts or shuts the worker down; an inbox command lifting maintenance
        re-arms it (see :meth:`_process_command_inbox`).
        """
        age = None if liveness is None else now - liveness
        frozen = age is not None and age > LIVENESS_STALE_SECONDS
        gpu_idle = snapshot.jobs_pending_inference > 0 and _inference_processes_all_idle(snapshot)
        if not (frozen or gpu_idle):
            self._guard_since = None
            return None
        if self._guard_since is None:
            self._guard_since = now
        if self._guard_fired or (now - self._guard_since) < AUTO_GUARD_CONFIRM_SECONDS:
            return None

        self._guard_fired = True
        self._controller.send_command(
            SupervisorControlMessage(
                command=SupervisorCommand.SET_SERVER_MAINTENANCE,
                server_maintenance_enabled=True,
            ),
        )
        reasons: list[str] = []
        if frozen:
            reasons.append(f"the parent has been frozen for {age:.0f}s")
        if gpu_idle:
            reasons.append(f"{snapshot.jobs_pending_inference} job(s) pending while the GPU sat idle")
        elapsed = now - self._guard_since
        return {
            "t": round(now, 3),
            "rule": "auto_guard_server_maintenance",
            "severity": _SEVERITY_CRITICAL,
            "summary": "Auto-guard set server-side maintenance ON (one shot): "
            + " and ".join(reasons)
            + f", persisting past the {AUTO_GUARD_CONFIRM_SECONDS:.0f}s confirmation window "
            f"({elapsed:.0f}s). The worker will not restart or shut down; lift maintenance via the inbox to "
            "re-arm the guard.",
            "confirm_seconds": AUTO_GUARD_CONFIRM_SECONDS,
            "elapsed_seconds": round(elapsed, 1),
        }

    def _rearm_guard(self) -> None:
        """Re-arm the auto-guard after an operator lifts maintenance, so it must re-confirm before acting."""
        self._guard_fired = False
        self._guard_since = None

    # endregion

    # region command inbox

    def _process_command_inbox(self, now: float) -> None:
        """Apply every not-yet-seen line in the command inbox exactly once (malformed lines are skipped)."""
        if not self._commands_path.exists():
            return
        try:
            lines = self._commands_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            logger.exception("Could not read the command inbox; skipping this interval.")
            return
        new_lines = lines[self._commands_consumed :]
        # Advance the offset first so a line that somehow errors is never retried on the next interval.
        self._commands_consumed = len(lines)
        for raw in new_lines:
            self._apply_command_line(raw, now)

    def _apply_command_line(self, raw: str, now: float) -> None:
        """Parse one inbox line and forward it to the worker, logging and skipping anything malformed."""
        text = raw.strip()
        if not text:
            return
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            self._reject_command(now, text, "not valid JSON")
            return
        if not isinstance(data, dict):
            self._reject_command(now, text, "not a JSON object")
            return
        name = data.get("command")
        if not isinstance(name, str):
            self._reject_command(now, text, "missing a string 'command'")
            return
        command = _resolve_command(name)
        if command is None:
            self._reject_command(now, text, f"unknown command '{name}'")
            return

        if command is SupervisorCommand.SHUTDOWN:
            # Route shutdown through the supervisor's orphan-proof graceful path rather than a raw command.
            logger.info("Command inbox requested a graceful shutdown.")
            self._controller.request_graceful_stop()
            return

        try:
            message = _build_control_message(command, data)
        except ValidationError as validation_error:
            self._reject_command(now, text, f"invalid fields ({validation_error.error_count()} error(s))")
            return

        if not self._controller.send_command(message):
            logger.warning(f"Worker rejected command {command.name} (transport unusable).")
            return
        logger.info(f"Applied inbox command {command.name}.")
        if command is SupervisorCommand.SET_SERVER_MAINTENANCE and message.server_maintenance_enabled is False:
            self._rearm_guard()

    def _reject_command(self, now: float, text: str, reason: str) -> None:
        """Log a malformed/unsupported inbox line and surface it as an alert, without ever failing the loop."""
        clipped = text if len(text) <= 200 else text[:197] + "..."
        logger.warning(f"Skipping inbox command line ({reason}): {clipped}")
        self._append_jsonl(
            self._alerts_path,
            {
                "t": round(now, 3),
                "rule": "command_rejected",
                "severity": _SEVERITY_WARNING,
                "summary": f"Skipped an inbox command line ({reason}).",
                "line": clipped,
            },
        )

    # endregion

    # region file io

    def _append_jsonl(self, path: Path, record: dict[str, object]) -> None:
        """Append one compact JSON record as a line."""
        self._append_line(path, json.dumps(record, separators=(",", ":"), default=str))

    @staticmethod
    def _append_line(path: Path, line: str) -> None:
        """Append a single line (best-effort; an I/O error is logged, never raised into the loop)."""
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            logger.exception(f"Could not append to {path}.")

    # endregion


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the attach-supervisor command line."""
    parser = argparse.ArgumentParser(
        prog="horde-worker-attach",
        description="Run the AI Horde worker headless, observing it through files and controlling it via an inbox.",
    )
    parser.add_argument(
        "--session-dir",
        type=str,
        required=True,
        help="Directory for state.jsonl, alerts.jsonl, and the commands.jsonl inbox.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help=f"Seconds between polls (default {DEFAULT_INTERVAL_SECONDS}).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to bridgeData.yaml; its directory becomes the working directory (config + logs live there).",
    )
    parser.add_argument("-e", "--load-config-from-env-vars", action="store_true", help="Load config from env vars.")
    parser.add_argument("--amd", "--amd-gpu", action="store_true", help="Enable AMD GPU optimisations.")
    parser.add_argument("-n", "--worker-name", type=str, default=None, help="Override the worker name.")
    parser.add_argument("--directml", type=int, default=None, help="Enable directml on the given device index.")
    parser.add_argument("--no-auto-restart", action="store_true", help="Do not relaunch the worker if it crashes.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Console entry point (``horde-worker-attach``): launch the worker and run the file-driven supervisor."""
    multiprocessing.freeze_support()
    args = _parse_args(argv)

    # Resolve the session dir before any chdir so the observation files always land where the caller asked.
    session_dir = Path(args.session_dir).resolve()

    # The worker reads bridgeData.yaml from the working directory and writes logs/ there; pointing the working
    # directory at the config's folder keeps the config, the log sink, and the live-log watch all consistent.
    if args.config:
        config_path = Path(args.config).resolve()
        if config_path.parent != Path.cwd():
            os.chdir(config_path.parent)

    from horde_worker_regen.app_state import default_app_state_dir
    from horde_worker_regen.process_management.lifecycle.owned_process_registry import OwnedProcessRegistry
    from horde_worker_regen.tui.logging_setup import setup_supervisor_file_logging
    from horde_worker_regen.tui.worker_launcher import WorkerProcessMode, WorkerSupervisor

    setup_supervisor_file_logging("attach")

    options = WorkerLaunchOptions(
        load_config_from_env_vars=args.load_config_from_env_vars,
        amd=args.amd,
        worker_name=args.worker_name,
        directml=args.directml,
    )

    # Reap a worker tree a prior attach session orphaned, then own this session's worker pid so a successor
    # can do the same. Skipped under test (it would touch real OS processes and a shared on-disk file).
    owned_registry: OwnedProcessRegistry | None = None
    if not os.environ.get("AI_HORDE_TESTING"):
        owned_registry = OwnedProcessRegistry(path=default_app_state_dir() / "attach_owned_pids.json")
        reaped = owned_registry.reap_orphans_from_previous_run(kill_tree=True)
        if reaped:
            logger.warning(f"Reaped an orphaned worker tree left by a previous attach session: {reaped}")

    supervisor = WorkerSupervisor(
        options,
        mode=WorkerProcessMode.REAL,
        auto_restart=not args.no_auto_restart,
        owned_registry=owned_registry,
    )
    attach = AttachSupervisor(supervisor, session_dir=session_dir, log_dir=Path("logs"), interval=args.interval)

    def _handle_signal(_signum: int, _frame: FrameType | None) -> None:
        logger.info("Attach supervisor received a stop signal; shutting down.")
        attach.stop()

    for signal_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        handler_signal = getattr(signal, signal_name, None)
        if handler_signal is not None:
            with contextlib.suppress(ValueError, OSError):
                signal.signal(handler_signal, _handle_signal)

    supervisor.start()
    try:
        attach.run_forever()
    finally:
        # The launcher's own stop is the orphan-proof teardown (graceful drain, then tree-kill on overrun).
        supervisor.stop()


if __name__ == "__main__":
    main()
