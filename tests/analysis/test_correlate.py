"""Unit tests for parent<->child correlation: exception extraction, recovery parsing, and the join."""

from __future__ import annotations

from pathlib import Path

from horde_worker_regen.analysis.bundle import LogBundle
from horde_worker_regen.analysis.correlate import (
    build_session_context,
    build_timeline,
    extract_exception,
    find_child_crash,
    parse_recoveries,
)
from horde_worker_regen.analysis.log_ingest import parse_lines
from horde_worker_regen.analysis.sessions import segment_sessions

_TRACEBACK = """\
2026-06-24 18:29:26.383 | CRITICAL | inference_1:startup - worker child crashed before its log was ready:
Traceback (most recent call last):
  File "worker_entry_points.py", line 324, in start_inference_process
    hordelib.initialise(
  File "model_management.py", line 211, in get_torch_device
    return torch.device(torch.cuda.current_device())
AssertionError: Torch not compiled with CUDA enabled
"""

_RECOVERY = (
    "2026-06-24 18:29:31.575 | ERROR    | "
    "horde_worker_regen.process_management.lifecycle.process_lifecycle:_log_recovery_diagnostics:367 - "
    "Recovery diagnostics for process 1 (os_pid=4600, launch=2): "
    "reason='inference process replaced (crashed or hung)'; last_state=PROCESS_STARTING; exitcode=1; "
    "last_heartbeat_type=OTHER; since_last_heartbeat=8.2s; since_last_message=8.2s; last_job=None; recent_actions=[]"
)


class TestExtractException:
    """Lifting the actionable root-cause line out of a traceback block."""

    def test_extracts_assertion(self) -> None:
        """The final ``ExceptionClass: message`` line is returned, not the framing text."""
        assert extract_exception(_TRACEBACK) == "AssertionError: Torch not compiled with CUDA enabled"

    def test_none_when_no_exception(self) -> None:
        """Text with no exception line yields None."""
        assert extract_exception("just some normal log output\nwith no traceback") is None


class TestParseRecoveries:
    """Parsing the parent-side recovery diagnostics that key the child join."""

    def test_parses_fields(self) -> None:
        """process_id, os_pid, launch, reason, and exitcode are all extracted."""
        records = parse_lines([_RECOVERY], Path("bridge.log"))
        (recovery,) = parse_recoveries(records)
        assert recovery.process_id == 1
        assert recovery.os_pid == 4600
        assert recovery.launch == 2
        assert recovery.exitcode == "1"


class TestChildJoin:
    """The cross-process join: a parent recovery to the child traceback that caused it."""

    def _bundle(self, tmp_path: Path) -> LogBundle:
        (tmp_path / "bridge.log").write_text(_RECOVERY + "\n", encoding="utf-8")
        (tmp_path / "bridge_inference_1_startup.log").write_text(_TRACEBACK, encoding="utf-8")
        return LogBundle.from_path(tmp_path)

    def test_find_child_crash_matches_by_time_window(self, tmp_path: Path) -> None:
        """The slot-1 crash 5s before the recovery is found and its exception surfaced."""
        bundle = self._bundle(tmp_path)
        recovery = parse_recoveries(bundle.orchestrator_records())[0]
        crash = find_child_crash(bundle, process_id=1, around=recovery.timestamp)
        assert crash is not None
        assert crash.exception == "AssertionError: Torch not compiled with CUDA enabled"

    def test_os_pid_exact_match_wins_over_time(self, tmp_path: Path) -> None:
        """When the child stamped its os_pid, the exact identity match beats timestamp proximity."""
        # Two crashes for slot 1, both near the recovery time, but only one shares the recovery's os_pid.
        startup = (
            "2026-06-24 18:29:30.000 | CRITICAL | inference_1:startup - worker child (os_pid=9999, launch=1) "
            "crashed before its log was ready:\nValueError: wrong one\n"
            "2026-06-24 18:29:31.400 | CRITICAL | inference_1:startup - worker child (os_pid=4600, launch=2) "
            "crashed before its log was ready:\nAssertionError: right one\n"
        )
        (tmp_path / "bridge.log").write_text(_RECOVERY + "\n", encoding="utf-8")
        (tmp_path / "bridge_inference_1_startup.log").write_text(startup, encoding="utf-8")
        bundle = LogBundle.from_path(tmp_path)
        recovery = parse_recoveries(bundle.orchestrator_records())[0]
        crash = find_child_crash(bundle, 1, recovery.timestamp, os_pid=recovery.os_pid)
        assert crash is not None
        assert crash.exception == "AssertionError: right one"

    def test_timeline_interleaves_child_crash_with_parent(self, tmp_path: Path) -> None:
        """The merged timeline contains both the child CRASH and the parent recovery, in time order."""
        bundle = self._bundle(tmp_path)
        session = segment_sessions(bundle.orchestrator_records())[0]
        timeline = build_timeline(build_session_context(session, bundle))
        sources = [entry.source for entry in timeline]
        assert "child_startup" in sources
        assert "orchestrator" in sources
        # The crash (18:29:26) precedes the parent's recovery (18:29:31).
        assert timeline[0].source == "child_startup"
