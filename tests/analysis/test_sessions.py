"""Unit tests for segmenting an appended bridge.log into per-launch sessions with an end-reason."""

from __future__ import annotations

from pathlib import Path

from horde_worker_regen.analysis.log_ingest import parse_lines
from horde_worker_regen.analysis.sessions import SessionEndReason, WorkerSession, segment_sessions

# Two launches in one appended log. Each opens with the main-process logger-setup line (the boundary),
# prints a worker-info line (identity + version + recoveries), then ends differently: the first aborts
# via save-our-ship, the second is stopped by the operator.
_LOG = """\
2026-06-24 18:00:00.000 | DEBUG    | hordelib.utils.logger:set_sinks:269 - Setting up logger for main process
2026-06-24 18:00:05.000 | INFO     | horde_worker_regen.reporting.status_reporter:_print_worker_info:442 -   dreamer_name: tazlin-tui-example | (v12.28.0+dev.gabc.dirty) | horde user: Tazlin#6572 | num_models: 113 | custom_models: False | max_power: 32 (1024x1024) | max_threads: 1 | queue_size: 3 | safety_on_gpu: True
2026-06-24 18:00:10.000 | INFO     | horde_worker_regen.reporting.status_reporter:_print_job_info:295 -   Session job info: ... | process_recoveries: 17 | 0.00 seconds without jobs
2026-06-24 18:00:20.000 | CRITICAL | horde_worker_regen.process_management.process_manager:_give_up_on_wedged_jobs:2123 - Save-our-ship: the worker cannot restore a working process pool; abandoning ship
2026-06-24 18:00:21.000 | WARNING  | horde_worker_regen.process_management.process_manager:_process_control_loop:2156 - Found .abort file - aborting immediately
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

    def test_peak_recoveries_captured(self) -> None:
        """The peak process_recoveries count for the session is read from its status line."""
        assert _sessions()[0].peak_process_recoveries == 17


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
