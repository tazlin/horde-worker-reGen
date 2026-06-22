"""Durable record of the OS processes this worker owns, so orphans can be reaped after a hard crash.

The worker's multiprocessing children are normally ended cleanly on shutdown. But if the *parent*
dies hard (SIGKILL, OOM-kill, power loss, an unhandled crash that skips ``atexit``), its children
are orphaned: they keep a GPU resident and a model loaded, and a relaunched worker then contends
with its own zombies. ``atexit``/signal handlers cover the graceful and most-signal cases; this
registry covers the rest by persisting which OS pids the worker started, so the *next* startup can
find and kill any that are still alive.

The single hazard with pid-based reaping is pid reuse: by the time we look, the recorded pid may
belong to an unrelated process. Each record therefore stores the child's ``create_time`` (and a
name fragment); a survivor is only killed when both still match, which is unforgeable in practice
(creation timestamps are effectively unique per pid).

The file lives at ``.horde_worker_regen/owned_pids.json`` (the same grouped state dir as
``app_state``). Reads never raise (a missing or corrupt file yields an empty registry) and writes
are atomic (temp file + ``os.replace``), so this can never block worker startup.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path

import psutil
from loguru import logger
from pydantic import BaseModel

from horde_worker_regen.app_state import default_app_state_dir

OWNED_PIDS_FILENAME = "owned_pids.json"

_CREATE_TIME_TOLERANCE_SECONDS = 1.0
"""How close a live process's create_time must be to the recorded one to count as the same process.

psutil's create_time is sub-second but can differ slightly from the value sampled right after spawn,
so an exact match is too strict; one second is far tighter than any realistic pid-reuse window.
"""


class OwnedProcessRecord(BaseModel):
    """One OS process the worker started and is responsible for cleaning up."""

    os_pid: int
    create_time: float
    """``psutil.Process.create_time()`` sampled just after launch; guards against pid reuse."""
    launch_identifier: int
    process_type: str
    """The ``HordeProcessType`` name (inference/safety/download), for diagnostics."""
    name_hint: str = ""
    """A fragment of the process name at launch (e.g. 'python'), a secondary pid-reuse guard."""


class OwnedProcessRegistry:
    """Persists the worker's owned child pids and reaps orphans left by a prior crashed run."""

    def __init__(self, path: Path | None = None) -> None:
        """Initialize the registry, defaulting to ``.horde_worker_regen/owned_pids.json`` in the working dir."""
        self._path = path if path is not None else (default_app_state_dir() / OWNED_PIDS_FILENAME)
        self._records: dict[int, OwnedProcessRecord] = {}

    @property
    def path(self) -> Path:
        """The on-disk registry file path."""
        return self._path

    def _load(self) -> list[OwnedProcessRecord]:
        """Read the persisted records, returning an empty list on any error (never raises)."""
        try:
            raw = self._path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return []
        try:
            data = json.loads(raw)
            return [OwnedProcessRecord.model_validate(entry) for entry in data]
        except Exception as e:
            logger.warning(f"owned-pids registry at {self._path} is unreadable ({type(e).__name__}); ignoring it")
            return []

    def _persist(self) -> None:
        """Atomically write the current records (temp file + replace), swallowing IO errors."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps([r.model_dump() for r in self._records.values()])
            fd, tmp_name = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(payload)
                os.replace(tmp_name, self._path)
            finally:
                with contextlib.suppress(FileNotFoundError, OSError):
                    if os.path.exists(tmp_name):  # noqa: PTH110
                        os.remove(tmp_name)  # noqa: PTH107
        except Exception as e:
            logger.warning(f"Failed to persist owned-pids registry to {self._path}: {type(e).__name__} {e}")

    def record(self, *, os_pid: int | None, launch_identifier: int, process_type: str) -> None:
        """Record a freshly-started child so it can be reaped if this parent dies before ending it."""
        if os_pid is None:
            return
        create_time = 0.0
        name_hint = ""
        with contextlib.suppress(Exception):
            proc = psutil.Process(os_pid)
            create_time = proc.create_time()
            name_hint = proc.name()
        self._records[os_pid] = OwnedProcessRecord(
            os_pid=os_pid,
            create_time=create_time,
            launch_identifier=launch_identifier,
            process_type=process_type,
            name_hint=name_hint,
        )
        self._persist()

    def forget(self, os_pid: int | None) -> None:
        """Drop a child the worker has cleanly ended; it no longer needs reaping."""
        if os_pid is None:
            return
        if self._records.pop(os_pid, None) is not None:
            self._persist()

    def clear(self) -> None:
        """Forget every recorded child (e.g. after reaping a prior run's orphans)."""
        if self._records:
            self._records.clear()
        self._persist()

    def _still_owned(self, record: OwnedProcessRecord) -> psutil.Process | None:
        """Return the live process for a record only if its identity still matches (else None).

        Guards against pid reuse: a different process now holding the pid will have a different
        ``create_time``, so it is left untouched.
        """
        if not psutil.pid_exists(record.os_pid):
            return None
        try:
            proc = psutil.Process(record.os_pid)
            if abs(proc.create_time() - record.create_time) > _CREATE_TIME_TOLERANCE_SECONDS:
                return None
            if record.name_hint and record.name_hint not in proc.name():
                return None
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
        return proc

    def reap_orphans_from_previous_run(self, *, kill_tree: bool = False) -> list[int]:
        """Kill any still-living children recorded by a prior (crashed) run, then reset the registry.

        Call this once at startup, before starting new processes: at that point every record is from
        a previous run. Returns the pids actually killed (verified-identity matches only).

        ``kill_tree`` widens each kill to the recorded process *and its descendants*: a host records only
        the top worker pid, but the GPU-resident processes are that worker's inference/safety children, so
        reaping the worker alone would leave them orphaned. The worker tracks its own children individually
        and so leaves this off.
        """
        killed: list[int] = []
        for record in self._load():
            proc = self._still_owned(record)
            if proc is None:
                continue
            logger.warning(
                f"Reaping orphaned {record.process_type} process from a previous run "
                f"(pid={record.os_pid}, launch={record.launch_identifier})",
            )
            if kill_tree:
                kill_process_tree(record.os_pid)
                killed.append(record.os_pid)
                continue
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                proc.kill()
                killed.append(record.os_pid)
        self._records.clear()
        self.clear()
        return killed

    def kill_all_owned(self) -> list[int]:
        """Best-effort kill of every child currently recorded as owned (for atexit / signal handlers).

        Identity is re-verified per pid so a reused pid is never killed. Returns the pids killed.
        """
        killed: list[int] = []
        for record in list(self._records.values()):
            proc = self._still_owned(record)
            if proc is None:
                continue
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                proc.kill()
                killed.append(record.os_pid)
        return killed


def kill_process_tree(pid: int, *, grace_seconds: float = 5.0) -> list[int]:
    """Kill a process and all of its descendants (best-effort); return the pids targeted.

    Killing only a top-level process orphans its children on platforms (notably Windows) where a
    child's lifetime is not tied to its parent's by a process group or job object. The benchmark stack
    is three deep -- controller -> per-level runner -> worker children (inference/safety/download) -- so
    a cancel or a hung-level kill that targets only the top process leaves the GPU-resident workers
    running long after the user asked them to stop (the exact symptom this addresses). This enumerates
    the whole descendant tree *first* (so nothing is reparented out of reach mid-kill), asks each to
    terminate, then hard-kills any survivor after a short grace.

    This complements -- it does not replace -- :class:`OwnedProcessRegistry`: the registry reaps a prior
    run's orphans on the *next* startup and ``atexit`` kills children on a *graceful* exit, but neither
    can kill a still-living tree the moment a user cancels or a parent is hard-killed (``atexit`` never
    runs on ``TerminateProcess``/``SIGKILL``).
    """
    try:
        root = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []

    try:
        tree = [*root.children(recursive=True), root]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        tree = [root]

    for proc in tree:
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            proc.terminate()

    _gone, alive = psutil.wait_procs(tree, timeout=grace_seconds)
    for proc in alive:
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            proc.kill()
    psutil.wait_procs(alive, timeout=grace_seconds)

    return [proc.pid for proc in tree]
