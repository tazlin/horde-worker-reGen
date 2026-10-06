"""Unit tests for segmenting an appended bridge.log into per-launch sessions with an end-reason."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from horde_worker_regen.analysis.bundle import LogBundle, OrchestratorLogKind
from horde_worker_regen.analysis.log_ingest import parse_lines
from horde_worker_regen.analysis.log_triage_cli import build_parser
from horde_worker_regen.analysis.sessions import (
    SessionEndReason,
    WorkerSession,
    segment_bundle_sessions,
    segment_sessions,
)
from horde_worker_regen.analysis.triage_report import render_findings, render_sessions, session_to_dict

# Two launches in one appended log. Each opens with the main-process logger-setup line (the boundary),
# prints a worker-info line (identity + version + recoveries), then ends differently: the first aborts
# via save-our-ship, the second is stopped by the operator.
_LOG = """\
2026-06-24 18:00:00.000 | DEBUG    | hordelib.utils.logger:set_sinks:269 - Setting up logger for main process
2026-06-24 18:00:05.000 | INFO     | horde_worker_regen.reporting.status_reporter:_print_worker_info:442 -   dreamer_name: tazlin-tui-example | (v12.28.0+dev.gabc.dirty) | hordelib: 7.9.0 (editable hordelib gc3202bd0.modified) | horde user: Tazlin#6572 | num_models: 113 | custom_models: False | max_power: 32 (1024x1024) | max_threads: 1 | queue_size: 3 | safety_on_gpu: True
2026-06-24 18:00:10.000 | INFO     | horde_worker_regen.reporting.status_reporter:_print_job_info:295 -   Session job info: ... | process_recoveries: 17 | 0.00 seconds without jobs
2026-06-24 18:00:20.000 | CRITICAL | horde_worker_regen.process_management.process_manager:_give_up_on_wedged_jobs:2123 - Save-our-ship: the worker cannot restore a working process pool; abandoning ship
2026-06-24 18:00:21.000 | WARNING  | horde_worker_regen.process_management.process_manager:_process_control_loop:2156 - Found .abort file; aborting immediately
2026-06-24 18:00:22.000 | INFO     | horde_worker_regen.run_worker:main:106 - Worker has finished working.
2026-06-24 18:01:00.000 | DEBUG    | hordelib.utils.logger:set_sinks:269 - Setting up logger for main process
2026-06-24 18:01:05.000 | INFO     | horde_worker_regen.reporting.status_reporter:_print_worker_info:442 -   dreamer_name: tazlin-tui-example | (v12.28.0+dev.gabc.dirty) | horde user: Tazlin#6572 | num_models: 113 | custom_models: False | max_power: 32 (1024x1024) | max_threads: 1 | queue_size: 3 | safety_on_gpu: True
2026-06-24 18:05:00.000 | WARNING  | horde_worker_regen.process_management.process_manager:_apply_supervisor_command:2619 - Supervisor requested shutdown.
2026-06-24 18:05:02.000 | INFO     | horde_worker_regen.run_worker:main:106 - Worker has finished working.
"""


def _sessions() -> list[WorkerSession]:
    return segment_sessions(parse_lines(_LOG.splitlines(), Path("bridge.log")))


class TestSegmentation:
    """Splitting on the main-process logger-setup boundary."""

    def test_splits_into_two_sessions(self) -> None:
        """Two logger-setup lines yield two sessions."""
        assert len(_sessions()) == 2

    def test_identity_and_version_extracted(self) -> None:
        """Each session captures the worker identity and version from its info line."""
        session = _sessions()[0]
        assert session.dreamer_name == "tazlin-tui-example"
        assert session.num_models == 113
        assert session.max_threads == 1
        assert session.version == "12.28.0+dev.gabc.dirty"
        assert session.hordelib == "7.9.0 (editable hordelib gc3202bd0.modified)"
        assert _sessions()[1].hordelib is None

    def test_listing_names_the_hordelib(self) -> None:
        """The session listing carries the hordelib identity when the run logged one."""
        listing = render_sessions(_sessions(), root=Path("bridge.log"))
        assert "threads: 1 | hordelib: 7.9.0 (editable hordelib gc3202bd0.modified)" in listing
        assert listing.count("hordelib:") == 1

    def test_peak_recoveries_captured(self) -> None:
        """The peak process_recoveries count for the session is read from its status line."""
        assert _sessions()[0].peak_process_recoveries == 17


class TestStartBounds:
    """A session whose launch line is missing reports its start as a bound, not as a fact."""

    _MID_RUN = (
        "2026-06-24 18:30:00.000 | INFO  | a.b:c:1 - working\n"
        "2026-06-24 19:30:00.000 | INFO  | a.b:c:1 - still working\n"
    )

    def test_start_of_a_bounded_session_is_flagged(self) -> None:
        """A capture that begins mid-run yields a session marked as starting at or before its first line."""
        session = segment_sessions(parse_lines(self._MID_RUN.splitlines(), Path("bridge.log")))[0]
        assert session.start_is_lower_bound is True
        assert session.start_truncated is False

    def test_truncated_capture_names_truncation_as_the_cause(self) -> None:
        """A bundle truncation note ahead of the first line attributes the missing start to the trim."""
        log = "[... truncated to the most recent 15 MB ...]\n" + self._MID_RUN
        session = segment_sessions(parse_lines(log.splitlines(), Path("bridge.log")))[0]
        assert session.start_is_lower_bound is True
        assert session.start_truncated is True
        assert session.start_ts is not None
        assert session.start_ts.hour == 18, "the truncation note must not be parsed as a record"

    def test_a_launch_boundary_gives_an_exact_start(self) -> None:
        """A session opened on its own logger-setup line is not a bound."""
        assert all(session.start_is_lower_bound is False for session in _sessions())

    def test_rendering_discloses_the_bound(self) -> None:
        """The rendered listing shows the span as a bound and says why."""
        log = "[... truncated to the most recent 15 MB ...]\n" + self._MID_RUN
        sessions = segment_sessions(parse_lines(log.splitlines(), Path("bridge.log")))
        text = render_sessions(sessions, root=Path("logs"))
        assert "<=18:30:00" in text
        assert ">=1h0m" in text
        assert "started at or before 18:30:00 (log truncated)" in text

    def test_dict_carries_the_bound(self) -> None:
        """The JSON view exposes the same disclosure for downstream tooling."""
        sessions = segment_sessions(parse_lines(self._MID_RUN.splitlines(), Path("bridge.log")))
        assert session_to_dict(sessions[0])["start_is_lower_bound"] is True


class TestEndReason:
    """Classifying how each session ended."""

    def test_give_up_abort_wins_over_clean_exit(self) -> None:
        """A session that abandoned ship is GAVE_UP_ABORTED even though it also logged a clean exit."""
        assert _sessions()[0].end_reason is SessionEndReason.GAVE_UP_ABORTED

    def test_supervisor_shutdown(self) -> None:
        """An operator-stopped session is SUPERVISOR_SHUTDOWN."""
        assert _sessions()[1].end_reason is SessionEndReason.SUPERVISOR_SHUTDOWN

    def test_truncated_middle_session_is_killed_or_crashed(self) -> None:
        """A non-final session with no exit marker is treated as killed/crashed mid-run."""
        log = (
            "2026-06-24 18:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - Setting up logger for main process\n"
            "2026-06-24 18:00:01.000 | INFO  | a.b:c:1 - working\n"
            "2026-06-24 18:01:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - Setting up logger for main process\n"
            "2026-06-24 18:01:01.000 | INFO  | a.b:c:1 - working\n"
        )
        sessions = segment_sessions(parse_lines(log.splitlines(), Path("bridge.log")))
        assert sessions[0].end_reason is SessionEndReason.KILLED_OR_CRASHED
        assert sessions[1].end_reason is SessionEndReason.STILL_RUNNING

    @pytest.mark.parametrize(
        ("sentinel_name", "expected"),
        [
            (".abort", SessionEndReason.ABORTED),
            (".abort_benchmark", SessionEndReason.ABORTED),
            (".abort_other", SessionEndReason.STILL_RUNNING),
        ],
    )
    def test_the_abort_line_names_either_sentinel(self, sentinel_name: str, expected: SessionEndReason) -> None:
        """The worker's and the benchmark's sentinel both end a session as aborted; another file name does not."""
        log = (
            "2026-06-24 18:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - "
            "Setting up logger for main process\n"
            "2026-06-24 18:00:10.000 | WARNING | "
            "horde_worker_regen.process_management.process_manager:_process_control_loop:1 - "
            f"Found {sentinel_name} file; aborting immediately\n"
        )

        session = segment_sessions(parse_lines(log.splitlines(), Path("bridge.log")))[0]

        assert session.end_reason is expected

    def test_child_teardown_line_alone_is_not_a_clean_process_exit(self) -> None:
        """The manager can reap every child while a gathered sibling still pins the worker process."""
        log = (
            "2026-06-24 18:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - "
            "Setting up logger for main process\n"
            "2026-06-24 18:00:10.000 | INFO  | "
            "horde_worker_regen.process_management.process_manager:_process_control_loop:1 - "
            "Shutting down process manager\n"
        )

        session = segment_sessions(parse_lines(log.splitlines(), Path("bridge.log")))[0]

        assert session.end_reason is SessionEndReason.STILL_RUNNING


# A worker launch that a second launch follows, in ``bridge.log``.
_WORKER_LOG = """\
2026-06-24 18:00:00.000 | DEBUG    | hordelib.utils.logger:set_sinks:269 - Setting up logger for main process
2026-06-24 18:05:00.000 | INFO     | a.b:c:1 - worker busy
2026-06-24 19:00:00.000 | INFO     | a.b:c:1 - worker still busy
2026-06-24 19:30:00.000 | DEBUG    | hordelib.utils.logger:set_sinks:269 - Setting up logger for main process
2026-06-24 19:31:00.000 | INFO     | a.b:c:1 - second launch busy
"""

# A benchmark run, logged by the harness parent while the first worker launch was still up. The harness
# log has no main-process marker; its lines before the process-manager banner are the run's preamble.
_HARNESS_BANNER = (
    "2026-06-24 18:10:01.000 | DEBUG    | horde_worker_regen.process_management.process_manager:__init__:1338 - "
    "Models to load: ['Deliberate']"
)
_HARNESS_BODY = f"""\
{_HARNESS_BANNER}
2026-06-24 18:10:05.000 | INFO     | horde_worker_regen.reporting.status_reporter:_print_worker_info:442 -   dreamer_name: warm-benchmark-worker | (v18.8.1) | horde user: Tazlin#6572 | num_models: 2 | custom_models: False | max_power: 32 (1024x1024) | max_threads: 1 | queue_size: 1 | safety_on_gpu: True
2026-06-24 18:50:00.000 | CRITICAL | horde_worker_regen.process_management.lifecycle.worker_recovery_coordinator:give_up_on_wedged_jobs:1633 - Save-our-ship: the worker cannot restore a working process pool; abandoning ship
"""
_HARNESS_PREAMBLE = (
    "2026-06-24 18:10:00.000 | INFO     | hordelib.beta_models:build_pending_provider:105 - Beta models on\n"
)


def _bundle_dir(tmp_path: Path, *, harness_text: str | None) -> Path:
    """A logs dir holding ``_WORKER_LOG`` and, when given, a ``bridge_harness.log``."""
    (tmp_path / "bridge.log").write_text(_WORKER_LOG, encoding="utf-8")
    if harness_text is not None:
        (tmp_path / "bridge_harness.log").write_text(harness_text, encoding="utf-8")
    return tmp_path


def _bundle_sessions(tmp_path: Path, *, harness_text: str | None) -> list[WorkerSession]:
    return segment_bundle_sessions(LogBundle.from_path(_bundle_dir(tmp_path, harness_text=harness_text)))


class TestBenchmarkRunSessions:
    """``bridge_harness.log`` yields benchmark-run sessions beside, and never inside, the worker's."""

    def test_benchmark_runs_follow_the_worker_sessions_with_their_own_numbering(self, tmp_path: Path) -> None:
        """A run overlapping a worker launch never shifts the worker's numbering; it counts from #0 on its own."""
        sessions = _bundle_sessions(tmp_path, harness_text=_HARNESS_PREAMBLE + _HARNESS_BODY)

        assert [session.log_kind for session in sessions] == [
            OrchestratorLogKind.WORKER,
            OrchestratorLogKind.WORKER,
            OrchestratorLogKind.HARNESS,
        ]
        assert [session.index for session in sessions] == [0, 1, 0]
        assert [session.is_benchmark_run for session in sessions] == [False, False, True]

    def test_an_overlapping_benchmark_run_never_merges_with_the_worker_session(self, tmp_path: Path) -> None:
        """Both logs are live over 18:10 to 18:50; each session holds only its own file's records."""
        sessions = _bundle_sessions(tmp_path, harness_text=_HARNESS_PREAMBLE + _HARNESS_BODY)
        worker, _, benchmark = sessions

        assert {record.source_path.name for record in worker.records} == {"bridge.log"}
        assert {record.source_path.name for record in benchmark.records} == {"bridge_harness.log"}
        assert worker.start_ts is not None and worker.start_ts.hour == 18 and worker.start_ts.minute == 0
        assert worker.end_ts is not None and worker.end_ts.hour == 19
        assert worker.end_reason is SessionEndReason.KILLED_OR_CRASHED
        assert benchmark.end_reason is SessionEndReason.GAVE_UP_ABORTED
        assert benchmark.dreamer_name == "warm-benchmark-worker"

    def test_the_harness_preamble_belongs_to_its_run(self, tmp_path: Path) -> None:
        """Lines logged just before the banner open the run exactly, not a separate mid-run session."""
        sessions = _bundle_sessions(tmp_path, harness_text=_HARNESS_PREAMBLE + _HARNESS_BODY)
        benchmarks = [session for session in sessions if session.is_benchmark_run]

        assert len(benchmarks) == 1
        assert benchmarks[0].start_is_lower_bound is False
        assert benchmarks[0].start_ts is not None and benchmarks[0].start_ts.second == 0

    def test_a_truncated_harness_capture_keeps_its_lower_bound(self, tmp_path: Path) -> None:
        """Records after a truncation note are the tail of an earlier run, whatever follows them."""
        harness_text = "[... truncated to the most recent 15 MB ...]\n" + _HARNESS_PREAMBLE + _HARNESS_BODY
        benchmarks = [s for s in _bundle_sessions(tmp_path, harness_text=harness_text) if s.is_benchmark_run]

        assert len(benchmarks) == 2
        assert benchmarks[0].start_is_lower_bound is True
        assert benchmarks[1].start_is_lower_bound is False

    def test_a_preamble_outside_the_burst_window_is_an_earlier_run(self, tmp_path: Path) -> None:
        """Lines a minute ahead of the banner are not its startup, so they stay a bounded session."""
        early = "2026-06-24 18:09:00.000 | INFO     | a.b:c:1 - tail of an earlier run\n"
        benchmarks = [s for s in _bundle_sessions(tmp_path, harness_text=early + _HARNESS_BODY) if s.is_benchmark_run]

        assert len(benchmarks) == 2
        assert benchmarks[0].start_is_lower_bound is True

    def test_worker_sessions_are_unchanged_without_a_harness_log(self, tmp_path: Path) -> None:
        """A bundle with no harness log segments exactly as ``bridge.log`` alone always has."""
        sessions = _bundle_sessions(tmp_path, harness_text=None)
        direct = segment_sessions(parse_lines(_WORKER_LOG.splitlines(), Path("bridge.log")))

        assert [(s.index, s.start_ts, s.end_reason) for s in sessions] == [
            (s.index, s.start_ts, s.end_reason) for s in direct
        ]
        assert all(session.log_kind is OrchestratorLogKind.WORKER for session in sessions)

    def test_listing_and_findings_label_the_benchmark_run(self, tmp_path: Path) -> None:
        """``sessions`` and ``diagnose`` text name the benchmark run, and the JSON view carries its kind."""
        sessions = _bundle_sessions(tmp_path, harness_text=_HARNESS_PREAMBLE + _HARNESS_BODY)
        listing = render_sessions(sessions, root=Path("logs"))

        assert "2 worker session(s) in logs" in listing
        assert "1 benchmark run(s) in logs" in listing
        assert listing.index("#1  19:30:00") < listing.index("1 benchmark run(s) in logs"), "worker list first"
        assert "#0  [benchmark run: bridge_harness.log]  18:10:00 -> 18:50:00" in listing
        assert "#0  18:00:00" in listing, "worker sessions keep their untagged heading"
        assert render_findings(sessions[2], []).startswith("=== Session #0  [benchmark run: bridge_harness.log]")
        assert session_to_dict(sessions[2])["log_kind"] == "harness"
        assert session_to_dict(sessions[0])["log_kind"] == "worker"

    def test_the_worker_listing_reads_the_same_with_or_without_a_harness_log(self, tmp_path: Path) -> None:
        """The benchmark list is appended below; the worker list above it is the worker-only listing verbatim."""
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        worker_only = render_sessions(_bundle_sessions(tmp_path / "a", harness_text=None), root=Path("logs"))
        with_benchmark = render_sessions(
            _bundle_sessions(tmp_path / "b", harness_text=_HARNESS_PREAMBLE + _HARNESS_BODY),
            root=Path("logs"),
        )

        assert with_benchmark.startswith(worker_only + "\n\n1 benchmark run(s) in logs\n")


def _run_sessions_cli(logs: Path, *flags: str, capsys: pytest.CaptureFixture[str]) -> list[tuple[str, int]]:
    """Run ``horde-log sessions --json`` with ``flags`` and return each listed session's kind and index."""
    args = build_parser().parse_args(["sessions", str(logs), "--json", *flags])
    assert args.func(args) == 0
    return [(entry["log_kind"], entry["index"]) for entry in json.loads(capsys.readouterr().out)]


class TestSessionSelectorsCli:
    """``--last``/``--session`` address worker sessions; ``--last-benchmark``/``--benchmark`` address benchmark runs."""

    @pytest.fixture
    def logs(self, tmp_path: Path) -> Path:
        """Two worker launches and one benchmark run that overlaps the first."""
        return _bundle_dir(tmp_path, harness_text=_HARNESS_PREAMBLE + _HARNESS_BODY)

    def test_no_selector_lists_workers_then_benchmark_runs(
        self, logs: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Every session, the worker list first."""
        assert _run_sessions_cli(logs, capsys=capsys) == [("worker", 0), ("worker", 1), ("harness", 0)]

    def test_session_and_last_address_worker_sessions_only(
        self, logs: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``--last`` is the latest worker launch even though a benchmark run exists."""
        assert _run_sessions_cli(logs, "--session", "0", capsys=capsys) == [("worker", 0)]
        assert _run_sessions_cli(logs, "--last", capsys=capsys) == [("worker", 1)]

    def test_benchmark_selectors_address_benchmark_runs_only(
        self, logs: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Benchmark run #0 is not worker session #0."""
        assert _run_sessions_cli(logs, "--benchmark", "0", capsys=capsys) == [("harness", 0)]
        assert _run_sessions_cli(logs, "--last-benchmark", capsys=capsys) == [("harness", 0)]
        assert _run_sessions_cli(logs, "--benchmark", "1", capsys=capsys) == []

    def test_selectors_of_both_kinds_combine(self, logs: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A worker selector and a benchmark selector together list both, the worker first."""
        assert _run_sessions_cli(logs, "--last-benchmark", "--session", "1", capsys=capsys) == [
            ("worker", 1),
            ("harness", 0),
        ]
