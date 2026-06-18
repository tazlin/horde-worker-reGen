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
import traceback
from datetime import datetime
from pathlib import Path

_LOG_DIR = Path("logs")

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


def write_startup_crash(role: str, exc: BaseException) -> None:
    """Append a full traceback for ``exc`` to a discoverable ``logs/bridge_{role}_startup.log``.

    This is the loguru-independent backstop for the no-sink startup window: it writes with a plain
    ``open()`` so it works even before ``HordeLog`` has opened a sink, and even if loguru is itself the
    failure. The file is created lazily (only when a crash is actually recorded), so it appears in the
    TUI Logs tab precisely when there is something to show. The line uses the same ``| LEVEL |`` shape
    as the other bridge logs so the Logs tab's level parser styles it.

    Args:
        role: Short identifier for the writing process (matches :func:`enable_child_faulthandler`).
        exc: The exception to record, including its traceback chain.
    """
    try:
        _LOG_DIR.mkdir(exist_ok=True)
        path = _LOG_DIR / f"bridge_{role}_startup.log"
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{stamp} | CRITICAL | {role}:startup - worker child crashed before its log was ready:\n{tb}\n",
            )
    except Exception:  # noqa: BLE001 - the emergency writer must never raise over the original crash
        return
