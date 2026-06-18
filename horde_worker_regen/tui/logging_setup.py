"""On-disk logging for the supervisor process (the TUI app and the worker host).

The worker child writes ``logs/bridge*.log`` itself (via hordelib's ``HordeLog``), and the Logs tab
tails those files. The *parent* process that owns the worker had no on-disk log of its own: its
loguru output went only to loguru's default stderr sink, which a full-screen Textual app paints over
and discards on exit. That hid exactly the diagnostics needed when the worker never starts or
crash-loops, none of which the worker can record because it never got far enough to open its own log:

- ``Launched worker (mode=…, pid=…)`` and every restart/backoff line (``worker_launcher``),
- ``Worker exceeded the restart budget; leaving it stopped``,
- any exception raised in the supervisor/TUI process itself.

This module gives the parent process its own loguru file sink. It writes ``logs/bridge_{role}.log``
so the existing Logs tab (which globs ``bridge*.log``) discovers and displays it as its own process
entry, and it never collides with the worker's ``bridge.log`` (separate process, separate file, so no
double-write). The format matches hordelib's plain file format so the Logs tab's level parser styles
it. Writes are synchronous (no ``enqueue``) so a crash never loses the last buffered lines.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

_PLAIN_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
"""Mirror of hordelib's plain file format so the Logs tab parses the level token the same way."""


def setup_supervisor_file_logging(role: str, *, quiet_console: bool = False) -> int | None:
    """Add a loguru file sink for a supervisor process, returning the sink id (or None on failure).

    Args:
        role: Short identifier for the writing process; the sink is ``logs/bridge_{role}.log`` so the
            Logs tab discovers it (e.g. ``"tui"`` -> the "tui" process entry, ``"host"`` -> "host").
        quiet_console: When True, also remove loguru's default stderr handler. Set this for the
            full-screen TUI, where stray stderr writes corrupt the Textual display; leave it False for
            the worker host, whose console output is still useful to its launcher.

    Returns:
        The loguru sink id, or None if the sink could not be created. Logging setup must never prevent
        the supervisor from starting, so all failures are swallowed.
    """
    try:
        log_path = Path("logs") / f"bridge_{role}.log"
        log_path.parent.mkdir(exist_ok=True)
        if quiet_console:
            # The default handler writes to stderr, which a full-screen Textual app owns; remove it so
            # the supervisor logs only to its file and cannot corrupt the rendered UI.
            logger.remove()
        return logger.add(
            log_path,
            level="DEBUG",
            rotation="1 day",
            retention="2 days",
            format=_PLAIN_FORMAT,
            backtrace=True,
            diagnose=True,
        )
    except Exception:  # noqa: BLE001 - logging setup must never block the supervisor from launching
        return None
