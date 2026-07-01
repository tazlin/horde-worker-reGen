"""Guarantee a spawned worker process leaves an on-disk trace when it crashes during startup.

A worker is a spawned child. If it dies before its normal loguru file sink (hordelib's
``HordeLog``, which writes ``logs/bridge.log`` / ``logs/bridge_{id}.log``) is open, its traceback
goes only to the inherited stderr. Under the TUI that stderr is the full-screen Textual app's, which
discards it; under a headless run it is buried in terminal scrollback. The result is the worst kind of
failure to debug: the worker "does nothing" and there is no ``logs/bridge*.log`` entry at all.

That window is wide. Every spawn target does ``logger.remove()`` and only re-opens a sink at
``HordeLog.initialise()``; the risky preflight in between (telemetry setup, ``from hordelib.api import
HordeLog``, env loading, the version check) runs with no loguru sink, so its ``logger.*`` calls write
nowhere. A crash there is invisible.

This module installs two cross-platform, loguru-independent backstops so that window is never silent:

- :func:`enable_child_faulthandler` points :mod:`faulthandler` at a per-process file, capturing hard
  faults (segfaults, fatal signals from torch / ComfyUI) that no ``try``/``except`` can catch.
- :func:`write_startup_crash` appends a full, formatted traceback to a discoverable
  ``logs/bridge_{role}_startup.log`` using a plain file write, so a Python exception during preflight is
  recorded even when no loguru sink exists yet (and even if loguru itself is the thing that broke).

Both are deliberately stdlib-only and swallow their own errors: crash capture must never be the reason a
worker fails to start, and it must keep working on Windows and Linux alike (faulthandler and plain file
writes behave the same on both; there are no POSIX-only paths here).
"""

from __future__ import annotations

import faulthandler
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

_LOG_DIR = Path("logs")

# The conventional final "ExceptionClass: message" line of a Python traceback, for lifting the root
# cause out of a startup-crash file. Kept here (not imported from the analysis package) so the worker's
# recovery path stays dependency-light.
_EXCEPTION_LINE_RE = re.compile(r"^(?P<exc>[A-Za-z_][\w.]*(?:Error|Exception|Warning|Interrupt|Exit)): ?(?P<msg>.*)$")

_FAULTHANDLER_FILES: list[object] = []
"""Open faulthandler file handles, kept referenced for the process lifetime.

``faulthandler.enable(file=...)`` writes to the handle when a fatal signal fires, so the handle must
outlive the call; letting it be garbage-collected would close the file and silence the backstop.
"""


def enable_child_faulthandler(role: str) -> None:
    """Point :mod:`faulthandler` at ``logs/bridge_{role}.faulthandler`` for hard-fault capture.

    A hard fault (segfault, abort, fatal signal) cannot be caught by ``except``; faulthandler is the
    only way to get a trace of one. The file is opened eagerly (the handle must exist before the fault)
    and kept open for the process lifetime. Any failure is swallowed so enabling capture can never stop
    the worker from starting.

    Args:
        role: Short identifier for the writing process (e.g. ``"main"``, ``"inference_0"``), used to
            name the per-process file so concurrent children never write to the same handle.
    """
    try:
        _LOG_DIR.mkdir(exist_ok=True)
        fault_file = (_LOG_DIR / f"bridge_{role}.faulthandler").open("a", encoding="utf-8")
        _FAULTHANDLER_FILES.append(fault_file)
        faulthandler.enable(file=fault_file)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001 - crash capture must never block the worker from starting
        return


def neutralize_inherited_argv() -> None:
    """Reduce a spawned child's ``sys.argv`` to just the program name.

    A spawned worker inherits the launcher's full ``sys.argv`` (multiprocessing's ``spawn`` start
    method restores the parent's argv in the child), but the child consumes none of it: its
    configuration arrives through the target function's arguments and IPC, not the command line.

    That inherited argv is not inert. Libraries loaded later can call ``argparse.parse_known_args()``
    against ``sys.argv`` at runtime -- notably some ComfyUI controlnet-annotator preprocessors, whose
    depth/normal path builds an argparse parser and parses the process argv when the annotator model is
    loaded. ``parse_known_args`` still honours prefix abbreviation, so an inherited flag that ambiguously
    (or wrongly) matches one of the parser's options makes argparse print an error and call
    ``sys.exit(2)``. ``SystemExit`` is a ``BaseException``, so the worker's ``except Exception`` guards do
    not catch it: the child exits with code 2 with no fault message and no fatal signal (faulthandler
    stays silent), surfacing only as an unexplained mid-inference process recovery.

    Clearing the argv the child never uses closes that whole class of collision. Call once, early in each
    spawned entry point.
    """
    try:
        sys.argv = sys.argv[:1]
    except Exception:  # noqa: BLE001 - argv hygiene must never block the worker from starting
        return


def write_startup_crash(
    role: str,
    exc: BaseException,
    *,
    os_pid: int | None = None,
    launch_identifier: int | None = None,
) -> None:
    """Append a full traceback for ``exc`` to a discoverable ``logs/bridge_{role}_startup.log``.

    This is the loguru-independent backstop for the no-sink startup window: it writes with a plain
    ``open()`` so it works even before ``HordeLog`` has opened a sink, and even if loguru is itself the
    failure. The file is created lazily (only when a crash is actually recorded), so it appears in the
    TUI Logs tab precisely when there is something to show. The line uses the same ``| LEVEL |`` shape
    as the other bridge logs so the Logs tab's level parser styles it.

    The OS pid and launch identifier are embedded in the line when known. That startup log is appended
    across every relaunch of a slot, so without them an offline triage tool can only tie this crash to
    the parent's recovery diagnostics by timestamp proximity; with them the join is exact (the parent
    logs the same ``os_pid``/``launch`` for the slot it reaps).

    Args:
        role: Short identifier for the writing process (matches :func:`enable_child_faulthandler`).
        exc: The exception to record, including its traceback chain.
        os_pid: This process's OS pid, if known, for an exact parent<->child join.
        launch_identifier: The parent-assigned launch counter for this slot, if known.
    """
    try:
        _LOG_DIR.mkdir(exist_ok=True)
        path = _LOG_DIR / f"bridge_{role}_startup.log"
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        identity = ""
        if os_pid is not None or launch_identifier is not None:
            identity = f" (os_pid={os_pid}, launch={launch_identifier})"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{stamp} | CRITICAL | {role}:startup - worker child{identity} crashed before its log was "
                f"ready:\n{tb}\n",
            )
    except Exception:  # noqa: BLE001 - the emergency writer must never raise over the original crash
        return


def read_last_startup_crash(role: str, *, max_bytes: int = 8192) -> str | None:
    """Return the exception summary of the most recent crash in ``logs/bridge_{role}_startup.log``.

    Lets the parent stamp the *why* of a crash into the structured action ledger at reap time, so the
    root cause survives even when the per-subprocess human logs are not. Only the file's tail is read
    (the file is appended across relaunches and can be large), and every error is swallowed: this runs on
    the recovery path and must never add a failure of its own. Returns None when there is no crash file
    or no recognizable exception line.

    Args:
        role: Short identifier for the crashed process (e.g. ``"inference_1"``).
        max_bytes: How many trailing bytes of the crash file to scan.
    """
    try:
        path = _LOG_DIR / f"bridge_{role}_startup.log"
        if not path.is_file():
            return None
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
            tail = handle.read().decode("utf-8", errors="replace")
        for line in reversed(tail.splitlines()):
            match = _EXCEPTION_LINE_RE.match(line.strip())
            if match is not None:
                message = match.group("msg").strip()
                return f"{match.group('exc')}: {message}" if message else match.group("exc")
        return None
    except Exception:  # noqa: BLE001 - never let crash-cause reading disrupt the recovery path
        return None
