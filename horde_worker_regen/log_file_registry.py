"""The single source of truth for what counts as a *worker log file*.

The worker (and the hordelib engine it embeds) scatter their on-disk logs across ``logs/`` under a
handful of stable naming conventions. Several independent code paths produce them: hordelib's
``HordeLog`` loguru sinks, the supervisor's own sink, the raw ``stdout``/``stderr`` redirections, the
pre-sink startup-crash backstops, and the faulthandler dumps. Nothing previously *declared* that set in
one place, so "which files are ours" had to be re-derived by every consumer (the log-tailer, the
support-bundle grouper, and the startup purge).

This module makes that set explicit and discoverable: :data:`WORKER_LOG_FILE_SPECS` names every family,
gives it a base-name pattern, says which component writes it, and classifies it as a loguru sink, a raw
stream redirection, a startup-crash backstop, or a faulthandler dump. Two things build on it:

* The startup purge (:mod:`horde_worker_regen.logging_purge`) deletes only files this registry
  positively recognizes. It is *fail-closed*: a file in ``logs/`` that matches no spec is never deleted,
  so an unfamiliar file is left untouched rather than guessed at.
* A CI check (``tests/test_log_file_registry.py``) introspects the loguru sinks the worker and hordelib
  *actually register at runtime* and asserts every one is described by a spec here. If a new file sink
  is added (in either repo) whose name no registry entry covers, that test fails, so the registry cannot
  silently drift out of step with the code that writes the logs.

A file name is matched against a spec by its *base name*: the rotation timestamp loguru appends and the
``.zip``/``.gz`` compression suffix are stripped first, so ``bridge.2026-06-22_00-55-59_013989.log.zip``
is recognized as the same ``bridge.log`` family as the active file.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal

LogFileKind = Literal["loguru_sink", "raw_stream", "startup_crash", "faulthandler"]
"""How a log-file family is produced.

* ``loguru_sink``: a rotating, ``.zip``-compressing loguru file sink (bounded rotation count).
* ``raw_stream``: a plain ``open()`` redirection with no rotation (one file per run, truncated on open).
* ``startup_crash``: a loguru-independent pre-sink crash backstop, appended to with a plain write.
* ``faulthandler``: a hard-fault (segfault/abort) trace file written by :mod:`faulthandler`.
"""


@dataclass(frozen=True)
class LogFileSpec:
    """One declared family of worker log file."""

    name: str
    """Stable identifier for the family (not a filename); unique across the registry."""
    pattern: re.Pattern[str]
    """Matches the file's *base name* (compression suffix and rotation timestamp already stripped)."""
    kind: LogFileKind
    """How the family is produced (see :data:`LogFileKind`)."""
    writer: str
    """Human-readable note on which component opens it, for discoverability."""
    description: str
    """What the family contains."""


# Order is presentation-only: the patterns are mutually exclusive, so exactly one spec matches any given
# base name (asserted by the registry integrity test). Role tokens in the startup/faulthandler families
# are deliberately open (``.+``): child roles are formed as ``bridge_{role}...`` for an open set of roles
# (inference_<n>, safety_<n>, download_<n>, main, and benchmark prewarm roles), all sharing one shape.
WORKER_LOG_FILE_SPECS: tuple[LogFileSpec, ...] = (
    LogFileSpec(
        name="orchestrator_loop",
        pattern=re.compile(r"bridge\.log"),
        kind="loguru_sink",
        writer="hordelib HordeLog (main/orchestrator process)",
        description="The orchestrator process's primary DEBUG log.",
    ),
    LogFileSpec(
        name="child_loop",
        pattern=re.compile(r"bridge_\d+\.log"),
        kind="loguru_sink",
        writer="hordelib HordeLog (worker child, keyed by process slot)",
        description="A worker child (inference/safety/download slot) process loop log.",
    ),
    LogFileSpec(
        name="orchestrator_trace",
        pattern=re.compile(r"trace\.log"),
        kind="loguru_sink",
        writer="hordelib HordeLog (main/orchestrator process)",
        description="The orchestrator's TRACE-level sink (most verbose; kept at a smaller rotation count).",
    ),
    LogFileSpec(
        name="child_trace",
        pattern=re.compile(r"trace_\d+\.log"),
        kind="loguru_sink",
        writer="hordelib HordeLog (worker child, keyed by process slot)",
        description="A worker child's TRACE-level sink.",
    ),
    LogFileSpec(
        name="child_stdout",
        pattern=re.compile(r"stdout_\d+\.log"),
        kind="raw_stream",
        writer="hordelib HordeLog (child sys.stdout redirect)",
        description="A worker child's redirected standard output.",
    ),
    LogFileSpec(
        name="child_stderr",
        pattern=re.compile(r"stderr_\d+\.log"),
        kind="raw_stream",
        writer="hordelib HordeLog (child sys.stderr redirect)",
        description="A worker child's redirected standard error.",
    ),
    LogFileSpec(
        name="supervisor_loop",
        pattern=re.compile(r"bridge_(?:tui|host)\.log"),
        kind="loguru_sink",
        writer="tui.logging_setup.setup_supervisor_file_logging (TUI app / worker host)",
        description="The supervisor process's own loguru sink, discovered by the Logs tab as its process entry.",
    ),
    LogFileSpec(
        name="main_console",
        pattern=re.compile(r"bridge_main_console\.log"),
        kind="raw_stream",
        writer="run_worker._redirect_streams_to_file (supervised worker)",
        description="A supervised worker's console redirect, kept out of the TUI's own screen.",
    ),
    LogFileSpec(
        name="utilities_console",
        pattern=re.compile(r"bridge_utilities_\d+\.log"),
        kind="raw_stream",
        writer="horde_image_utilities.launcher.CapabilityServerProcess (utilities lane subprocess redirect)",
        description="The out-of-venv image-utilities lane subprocess's stdout/stderr (uvicorn + native stack), "
        "keyed by lane slot; it is not a hordelib child, so it has no per-slot HordeLog sinks.",
    ),
    LogFileSpec(
        name="startup_crash",
        pattern=re.compile(r"bridge_.+_startup\.log"),
        kind="startup_crash",
        writer="process_management.lifecycle.child_crash_capture.write_startup_crash",
        description="Pre-sink startup-crash backstop for a role, written before (or instead of) a loguru sink.",
    ),
    LogFileSpec(
        name="faulthandler",
        pattern=re.compile(r"bridge_.+\.faulthandler"),
        kind="faulthandler",
        writer="process_management.lifecycle.child_crash_capture.enable_child_faulthandler",
        description="Hard-fault (segfault/abort/fatal-signal) trace for a role.",
    ),
)

# A rotated loguru archive carries a timestamp segment before ``.log`` and a ``.zip``/``.gz`` suffix after
# it, e.g. ``bridge.2026-06-22_00-55-59_013989.log.zip``. This mirrors analysis/bundle.py's normalization
# so the two agree on what base a rotation maps to.
_COMPRESSION_SUFFIXES = (".zip", ".gz")
_ROTATION_TS_RE = re.compile(r"\.\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_\d+)?(?=\.log$)")


def base_log_name(filename: str) -> str:
    """Reduce a possibly-rotated, possibly-compressed filename to the base name specs match against.

    ``bridge.2026-06-22_00-55-59_013989.log.zip`` -> ``bridge.log``.
    """
    name = filename
    for suffix in _COMPRESSION_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return _ROTATION_TS_RE.sub("", name)


def classify_log_file(filename: str) -> LogFileSpec | None:
    """Return the spec that describes *filename* (matched on its base name), or None if none does."""
    base = base_log_name(filename)
    for spec in WORKER_LOG_FILE_SPECS:
        if spec.pattern.fullmatch(base):
            return spec
    return None


def is_worker_log_file(filename: str) -> bool:
    """Return whether *filename* is a recognized worker log file (any rotation/compression thereof)."""
    return classify_log_file(filename) is not None


def discover_registered_file_sink_basenames() -> set[str]:
    """Return the basenames of every loguru file sink currently registered on the root logger.

    Introspects loguru's live handler table (private but stable across the pinned 0.7.x line). The drift
    test uses this, after triggering the real logging setup, to compare the sinks the worker and hordelib
    actually open against :data:`WORKER_LOG_FILE_SPECS`. Raw ``stdout``/``stderr`` redirections and the
    crash backstops are not loguru sinks, so they never appear here; they are validated by known-sample
    classification instead.
    """
    from loguru import logger

    try:
        from loguru._file_sink import FileSink
    except Exception:  # noqa: BLE001 - private path; fall back to a duck-typed check below
        file_sink_type: type | None = None
    else:
        file_sink_type = FileSink

    names: set[str] = set()
    core = getattr(logger, "_core", None)
    handlers = getattr(core, "handlers", {}) if core is not None else {}
    for handler in handlers.values():
        sink = getattr(handler, "_sink", None)
        is_file_sink = (file_sink_type is not None and isinstance(sink, file_sink_type)) or type(
            sink
        ).__name__ == "FileSink"
        path = getattr(sink, "_path", None)
        if is_file_sink and path:
            names.add(os.path.basename(str(path)))
    return names
