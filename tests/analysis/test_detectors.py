"""Unit tests for the diagnosis detectors over synthetic sessions.

The marquee case is the one from the real incident: an inference pool that crashes on start and a worker
that spins through a recovery storm without ever giving up. The detectors must lift the child's
exception across the process boundary and distinguish "never gave up" (the bug) from "gave up cleanly"
(the healthy bail-out), even though both arise from the same crash cause.
"""

from __future__ import annotations

from pathlib import Path

from horde_worker_regen.analysis.bundle import LogBundle
from horde_worker_regen.analysis.correlate import build_session_context
from horde_worker_regen.analysis.detectors import Finding, Severity, run_detectors
from horde_worker_regen.analysis.sessions import segment_sessions


def _diagnose(tmp_path: Path, bridge_log: str, child_logs: dict[str, str] | None = None) -> dict[str, Finding]:
    """Write a synthetic bundle and return the findings for its single session, keyed by id."""
    (tmp_path / "bridge.log").write_text(bridge_log, encoding="utf-8")
    for name, text in (child_logs or {}).items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    bundle = LogBundle.from_path(tmp_path)
    session = segment_sessions(bundle.orchestrator_records())[0]
    findings = run_detectors(build_session_context(session, bundle))
    return {finding.id: finding for finding in findings}


_STARTUP = "Setting up logger for main process"


def _recovery(ts: str, pid: int, *, reason: str, last_state: str = "PROCESS_STARTING") -> str:
    return (
        f"2026-06-24 {ts} | ERROR    | horde_worker_regen.process_management.process_lifecycle:_log_recovery_diagnostics:367 - "
        f"Recovery diagnostics for process {pid} (os_pid={1000 + pid}, launch={pid}): reason='{reason}'; "
        f"last_state={last_state}; exitcode=1; last_heartbeat_type=OTHER; since_last_heartbeat=8.0s; "
        f"since_last_message=8.0s; last_job=None; recent_actions=[]"
    )


_TRACEBACK = """\
2026-06-24 18:29:26.000 | CRITICAL | inference_1:startup - worker child crashed before its log was ready:
Traceback (most recent call last):
  File "model_management.py", line 211, in get_torch_device
    return torch.device(torch.cuda.current_device())
AssertionError: Torch not compiled with CUDA enabled
"""


class TestCrashOnStart:
    """Lifting the child exception across the process boundary for a crash-on-start loop."""

    def test_reports_child_exception(self, tmp_path: Path) -> None:
        """The crash-on-start finding names the child's exception, joined from the startup log."""
        bridge = "\n".join(
            [
                f"2026-06-24 18:29:20.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                _recovery("18:29:31.000", 1, reason="inference process replaced (crashed or hung)"),
                _recovery("18:29:40.000", 1, reason="inference process replaced (crashed or hung)"),
            ],
        )
        findings = _diagnose(tmp_path, bridge, {"bridge_inference_1_startup.log": _TRACEBACK})
        assert "crash_on_start_loop" in findings
        assert "Torch not compiled with CUDA enabled" in findings["crash_on_start_loop"].verdict
        assert findings["crash_on_start_loop"].severity is Severity.CRITICAL


class TestDoomedPoolNoGiveup:
    """The recovery storm that never gave up vs. the worker that correctly abandoned ship."""

    def _stormy_bridge(self, *, gave_up: bool) -> str:
        lines = [
            f"2026-06-24 18:29:20.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
            "2026-06-24 18:29:21.000 | INFO | horde_worker_regen.reporting.status_reporter:_print_worker_info:442 -   dreamer_name: w | (v12.28.0) | num_models: 113 | max_power: 32 (1024x1024) | max_threads: 1 | queue_size: 3 | safety_on_gpu: True",
            "2026-06-24 18:29:47.000 | CRITICAL | horde_worker_regen.process_management.process_lifecycle:_quarantine_inference_slot:1182 - Inference slot 1 quarantined (crash on start: 3 consecutive failures before reaching readiness); not respawning it.",
            "2026-06-24 18:30:30.000 | ERROR | horde_worker_regen.process_management.process_manager:_perform_soft_reset:2070 - Save-our-ship soft reset #1: rebuilding process pools and limping by (effective max_threads -> 1).",
            "2026-06-24 18:31:00.000 | INFO | horde_worker_regen.process_management.process_manager:_run_recovery_supervisor:2062 - Save-our-ship: pools recovered; restored configured concurrency (limp-by cleared).",
            "2026-06-24 18:31:08.000 | INFO | horde_worker_regen.reporting.status_reporter:_print_job_info:295 -   Session job info: ... | process_recoveries: 24 | 0.00 seconds without jobs",
        ]
        if gave_up:
            lines.append(
                "2026-06-24 18:31:20.000 | CRITICAL | horde_worker_regen.process_management.process_manager:_give_up_on_wedged_jobs:2123 - Save-our-ship: the worker cannot restore a working process pool; abandoning ship",
            )
        else:
            lines.append(
                "2026-06-24 18:31:23.000 | WARNING | horde_worker_regen.process_management.process_manager:_apply_supervisor_command:2619 - Supervisor requested shutdown.",
            )
        return "\n".join(lines)

    def test_fires_when_storm_without_giveup(self, tmp_path: Path) -> None:
        """A quarantined pool that flapped and stormed without abandoning ship trips the bug detector."""
        findings = _diagnose(tmp_path, self._stormy_bridge(gave_up=False))
        assert "doomed_pool_no_giveup" in findings
        assert "gave_up_clean" not in findings

    def test_silent_when_worker_gave_up(self, tmp_path: Path) -> None:
        """The same storm that ended in abandon-ship is the healthy path, not the bug."""
        findings = _diagnose(tmp_path, self._stormy_bridge(gave_up=True))
        assert "doomed_pool_no_giveup" not in findings
        assert "gave_up_clean" in findings


class TestResourceFindings:
    """OOM and the swallowed-OOM classification gap."""

    def test_oom_detected(self, tmp_path: Path) -> None:
        """An explicit CUDA OOM is surfaced as a critical finding."""
        bridge = "\n".join(
            [
                f"2026-06-24 18:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                "2026-06-24 18:00:10.000 | ERROR | x:y:1 - CUDA out of memory. Tried to allocate 2.00 GiB",
            ],
        )
        assert "oom" in _diagnose(tmp_path, bridge)

    def test_swallowed_oom_detected(self, tmp_path: Path) -> None:
        """A generic 'no images produced' fault is flagged as a possible swallowed OOM."""
        bridge = "\n".join(
            [
                f"2026-06-24 18:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                "2026-06-24 18:00:10.000 | WARNING | x:y:1 - Job faulted: no images were produced",
            ],
        )
        assert "swallowed_oom" in _diagnose(tmp_path, bridge)
