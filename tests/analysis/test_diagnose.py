"""Tests for the diagnose facade: the active-only/recent scoping and the lightweight view payload.

These cover the efficiency knobs the TUI relies on: reading only the live ``bridge.log`` for a quick
pass, bounding to the recent few sessions, and returning a record-free view cheap to ship across a
process boundary, without depending on the TUI itself.
"""

from __future__ import annotations

from pathlib import Path

from horde_worker_regen.analysis.bundle import OrchestratorLogKind
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


def _harness_log(ts_time: str, body: str) -> str:
    """A one-run harness log: the process-manager banner (the harness has no main-process marker), one line."""
    return "\n".join(
        [
            f"2026-06-24 {ts_time}.000 | DEBUG | horde_worker_regen.process_management.process_manager:__init__:1 - "
            "Models to load: []",
            f"2026-06-24 {ts_time}.500 | ERROR | x:y:1 - {body}",
        ],
    )


def _worker_and_benchmark_dir(tmp_path: Path) -> Path:
    """A worker launch at 18:00 and a benchmark run at 18:05 that overlaps it; only the benchmark ran out of VRAM."""
    (tmp_path / "bridge.log").write_text(
        "\n".join([_session_log("2026-06-24", "18:00:00", "all good"), "2026-06-24 18:30:00.000 | INFO | a:b:1 - up"]),
        encoding="utf-8",
    )
    (tmp_path / "bridge_harness.log").write_text(
        _harness_log("18:05:00", "CUDA out of memory. Tried to allocate 2.00 GiB"),
        encoding="utf-8",
    )
    return tmp_path


def test_findings_run_over_a_benchmark_run_as_over_a_worker_session(tmp_path: Path) -> None:
    """The benchmark run is diagnosed on its own records; its fault is not charged to the overlapping worker."""
    results = diagnose(_worker_and_benchmark_dir(tmp_path))

    assert [result.session.log_kind for result in results] == [OrchestratorLogKind.WORKER, OrchestratorLogKind.HARNESS]
    worker, benchmark = results
    assert any(finding.id == "oom" for finding in benchmark.findings)
    assert not any(finding.id == "oom" for finding in worker.findings)


def test_worker_and_benchmark_selectors_each_address_their_own_list(tmp_path: Path) -> None:
    """``last`` stays on the worker list though a benchmark started later; benchmark runs have their own selectors."""
    logs = _worker_and_benchmark_dir(tmp_path)

    def selected(
        *,
        last: bool = False,
        session_index: int | None = None,
        last_benchmark: bool = False,
        benchmark_index: int | None = None,
        recent: int | None = None,
    ) -> list[tuple[OrchestratorLogKind, int]]:
        results = diagnose(
            logs,
            last=last,
            session_index=session_index,
            last_benchmark=last_benchmark,
            benchmark_index=benchmark_index,
            recent=recent,
        )
        return [(result.session.log_kind, result.session.index) for result in results]

    assert selected(last=True) == [(OrchestratorLogKind.WORKER, 0)]
    assert selected(session_index=0) == [(OrchestratorLogKind.WORKER, 0)]
    assert selected(last_benchmark=True) == [(OrchestratorLogKind.HARNESS, 0)]
    assert selected(benchmark_index=0) == [(OrchestratorLogKind.HARNESS, 0)]
    assert selected(recent=1) == [(OrchestratorLogKind.WORKER, 0), (OrchestratorLogKind.HARNESS, 0)]


def test_views_carry_the_kind_and_the_per_kind_index(tmp_path: Path) -> None:
    """The TUI's view names each session's kind and its number within that kind."""
    views = diagnose_views(_worker_and_benchmark_dir(tmp_path), active_only=True)

    assert [(view.session.log_kind, view.session.index) for view in views] == [
        (OrchestratorLogKind.WORKER, 0),
        (OrchestratorLogKind.HARNESS, 0),
    ]
