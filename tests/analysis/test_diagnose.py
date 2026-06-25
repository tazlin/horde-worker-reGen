"""Tests for the diagnose facade: the active-only/recent scoping and the lightweight view payload.

These cover the efficiency knobs the TUI relies on -- reading only the live ``bridge.log`` for a quick
pass, bounding to the recent few sessions, and returning a record-free view cheap to ship across a
process boundary -- without depending on the TUI itself.
"""

from __future__ import annotations

from pathlib import Path

from horde_worker_regen.analysis.diagnose import SessionDiagnosisView, SessionSummary, diagnose, diagnose_views

_STARTUP = "Setting up logger for main process"


def _session_log(ts_date: str, ts_time: str, body: str) -> str:
    """A one-session log: the main-process startup boundary followed by one body line."""
    return "\n".join(
        [
            f"{ts_date} {ts_time}.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
            f"{ts_date} {ts_time}.500 | ERROR | x:y:1 - {body}",
        ],
    )


def _logs_dir(tmp_path: Path) -> Path:
    """A logs dir with one older rotation and a newer active ``bridge.log`` (one session each)."""
    (tmp_path / "bridge.log").write_text(
        _session_log("2026-06-24", "18:00:00", "CUDA out of memory. Tried to allocate 2.00 GiB"),
        encoding="utf-8",
    )
    (tmp_path / "bridge.2026-06-23_00-00-00.log").write_text(
        _session_log("2026-06-23", "10:00:00", "all good"),
        encoding="utf-8",
    )
    return tmp_path


def test_active_only_skips_rotations(tmp_path: Path) -> None:
    """``active_only`` reads just the live log (one session); the full pass also reads the rotation."""
    logs = _logs_dir(tmp_path)
    assert len(diagnose(logs, active_only=True)) == 1
    assert len(diagnose(logs)) == 2


def test_recent_limits_to_most_recent_sessions(tmp_path: Path) -> None:
    """``recent`` keeps only the most recent N sessions (the newest is the active log's session)."""
    logs = _logs_dir(tmp_path)
    recent = diagnose(logs, recent=1)
    assert len(recent) == 1
    # The active (newer) session carries the OOM finding; the older rotation does not.
    assert any(f.id == "oom" for f in recent[0].findings)


def test_diagnose_views_returns_record_free_summaries(tmp_path: Path) -> None:
    """The view payload is a SessionSummary (no parsed records) plus the findings, ready to pickle."""
    views = diagnose_views(_logs_dir(tmp_path))
    assert views and all(isinstance(view, SessionDiagnosisView) for view in views)
    assert all(isinstance(view.session, SessionSummary) for view in views)
    # The heavy per-record list never crosses into the view (that is the whole point of summarizing).
    assert not hasattr(views[-1].session, "records")
    assert any(f.id == "oom" for view in views for f in view.findings)
