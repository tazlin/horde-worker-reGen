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
        f"2026-06-24 {ts} | ERROR    | horde_worker_regen.process_management.lifecycle.process_lifecycle:_log_recovery_diagnostics:367 - "
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
            "2026-06-24 18:29:47.000 | CRITICAL | horde_worker_regen.process_management.lifecycle.process_lifecycle:_quarantine_inference_slot:1182 - Inference slot 1 quarantined (crash on start: 3 consecutive failures before reaching readiness); not respawning it.",
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


def _maintenance_pop(ts: str, *, reason: str = "dropping too many jobs") -> str:
    """The orchestrator line the job popper logs when the horde rejects a pop with maintenance mode."""
    return (
        f"2026-06-25 {ts} | WARNING  | horde_worker_regen.process_management.jobs.job_popper:_handle_pop_error_response:475 - "
        f"Failed to pop job (Maintenance Mode): message='Maintenance mode activated because worker is {reason}.' "
        "object_data=None rc='WorkerMaintenance'"
    )


def _force_admit(ts: str, *, starved_seconds: int, free_vram_mb: int, model: str = "AlbedoBase XL (SDXL)") -> str:
    """The head-of-queue starvation force-admit warning (budget deferred a job on an idle, free device)."""
    return (
        f"2026-06-25 {ts} | WARNING  | horde_worker_regen.process_management.scheduling.inference_scheduler:_log_head_starvation_force_admit:1348 - "
        f"Head-of-queue {model} was budget-deferred on an idle device for {starved_seconds}s (reclamation "
        "exhausted); force-admitting it best-effort to break the wedge before the recovery supervisor "
        f"soft-resets the pools and faults the backlog. slots=[#1:-[WAITING_FOR_JOB]] device_free_vram={free_vram_mb}MB"
    )


def _soft_reset(ts: str, *, level: int = 1) -> str:
    return (
        f"2026-06-25 {ts} | ERROR    | horde_worker_regen.process_management.process_manager:_perform_soft_reset:2070 - "
        f"Save-our-ship soft reset #{level}: rebuilding process pools and limping by (effective max_threads -> 1)."
    )


def _give_up(ts: str, *, jobs: int) -> str:
    return (
        f"2026-06-25 {ts} | CRITICAL | horde_worker_regen.process_management.process_manager:_give_up_on_wedged_jobs:2111 - "
        f"Save-our-ship: gave up on {jobs} unservable job(s) and reported them faulted so the horde reissues them."
    )


def _server_slow_abort(ts: str, *, job_id: str = "0a69c504-fd18-4474-8f99-3b9587a0fed9") -> str:
    """The verbatim server message the submitter logs when the horde aborts a too-slow generation."""
    return (
        f"2026-06-25 {ts} | ERROR    | horde_worker_regen.process_management.jobs.job_submitter:submit_single_generation:291 - "
        f"Processing Generation with ID {job_id} took too long to process and has been aborted! Please check "
        "your worker speed and do not onboard worker which generate slower than 1 it/s!"
    )


def _slowdown_grade(ts: str, *, pid: int = 4, ratio: float = 4.1, free_vram_mb: int = 5395) -> str:
    """The inference grader warning that a job is running N-times its expected sampling time."""
    return (
        f"2026-06-25 {ts} | WARNING  | horde_worker_regen.process_management.lifecycle.process_lifecycle:_grade_running_inference:1564 - "
        f"Inference on process {pid} is {ratio:.1f}x its expected sampling time (37s vs ~9s); watching for a "
        f"hang. slots=[#1:stable_diffusion[WAITING_FOR_JOB]] device_free_vram={free_vram_mb}MB"
    )


def _submit_latency(ts: str, *, popped_ago: float, gen: float) -> str:
    """A successful-submit line reporting pop->submit latency and generation time."""
    return (
        f"2026-06-25 {ts} | SUCCESS  | horde_worker_regen.process_management.jobs.job_submitter:submit_single_generation:343 - "
        f"Submitted generation abcd1234 (model: stable_diffusion) for 5.76 kudos. Job popped {popped_ago} seconds "
        f"ago and took {gen} to generate. (0.8 kudos/second for the whole batch. 0.4 or greater is ideal)"
    )


def _safety_duration(ts: str, *, seconds: float) -> str:
    """A safety-result line reporting how long the safety check took."""
    return (
        f"2026-06-25 {ts} | DEBUG    | horde_worker_regen.process_management.ipc.message_dispatcher:_handle_safety_result:801 - "
        f"Job abcd1234-0000-0000-0000-000000000000 had 0 images censored and took {seconds} seconds to check safety"
    )


def _consecutive_pause(ts: str) -> str:
    return (
        f"2026-06-25 {ts} | ERROR    | horde_worker_regen.process_management.jobs.job_popper:_handle_consecutive_failures:371 - "
        "Too many consecutive failed jobs, pausing job pops. Please look into what happened and let the "
        "devs know. Waiting 180 seconds..."
    )


class TestForcedMaintenance:
    """The horde stepping in and forcing the worker into maintenance for dropping too many jobs.

    This is the incident headline: the symptom the operator actually sees. It is downstream of the
    worker faulting jobs locally, so the finding must name the drops as the cause, not the maintenance
    flag as the problem to clear.
    """

    def _bridge(self, *lines: str) -> str:
        return "\n".join([f"2026-06-25 13:59:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}", *lines])

    def test_forced_for_dropped_jobs_is_critical(self, tmp_path: Path) -> None:
        """A maintenance pop citing dropped jobs, alongside local give-up faults, is a critical finding."""
        bridge = self._bridge(
            _give_up("15:19:08.000", jobs=4),
            _maintenance_pop("15:19:10.000"),
            _give_up("16:09:31.000", jobs=4),
            _maintenance_pop("16:20:22.000"),
        )
        findings = _diagnose(tmp_path, bridge)
        assert "forced_maintenance" in findings
        assert findings["forced_maintenance"].severity is Severity.CRITICAL
        # The verdict accounts for the jobs the worker dropped (4 + 4), the reason the horde stepped in.
        assert "8" in findings["forced_maintenance"].verdict

    def test_generic_maintenance_is_not_critical(self, tmp_path: Path) -> None:
        """Maintenance not attributed to dropped jobs (e.g. operator-set) is informational, not critical."""
        bridge = self._bridge(_maintenance_pop("15:19:10.000", reason="paused by its operator"))
        findings = _diagnose(tmp_path, bridge)
        assert "forced_maintenance" in findings
        assert findings["forced_maintenance"].severity is not Severity.CRITICAL

    def test_silent_without_maintenance(self, tmp_path: Path) -> None:
        """A session that was never put into maintenance produces no maintenance finding."""
        findings = _diagnose(tmp_path, self._bridge(_give_up("15:19:08.000", jobs=1)))
        assert "forced_maintenance" not in findings

    def test_forced_for_server_slow_aborts_names_the_aborts(self, tmp_path: Path) -> None:
        """Maintenance forced for drops that came from server-side slow-aborts (not give-ups) is critical.

        When all dropped jobs came from server-side slow-aborts with no give-ups, the verdict must name
        those aborts as the cause and point at the slow-generation finding, not the scheduler-wedge one.
        """
        bridge = self._bridge(
            _server_slow_abort("06:38:37.000"),
            _server_slow_abort("06:42:12.000"),
            _maintenance_pop("07:33:02.000"),
        )
        finding = _diagnose(tmp_path, bridge)["forced_maintenance"]
        assert finding.severity is Severity.CRITICAL
        assert "2 generation(s) as too slow" in finding.verdict
        assert "save-our-ship" not in finding.verdict
        assert finding.see_also == "slow_generation_drop_spiral"

    def test_forced_for_both_drop_kinds_names_both(self, tmp_path: Path) -> None:
        """When the worker both gave up backlog jobs and had generations aborted, the verdict names both."""
        bridge = self._bridge(
            _give_up("07:30:00.000", jobs=3),
            _server_slow_abort("07:31:00.000"),
            _maintenance_pop("07:33:02.000"),
        )
        finding = _diagnose(tmp_path, bridge)["forced_maintenance"]
        assert finding.severity is Severity.CRITICAL
        assert "3 backlog job(s)" in finding.verdict
        assert "1 generation(s) as too slow" in finding.verdict

    def test_counts_enriched_giveup_phrasing(self, tmp_path: Path) -> None:
        """The dropped-job count survives the worker enriching the give-up line with its wedge cause.

        The worker logs the give-up with a parenthetical cause and a maintenance note; the count parse
        keys on the stable prefix so this worker-log/tool contract does not silently drift.
        """
        enriched = (
            "2026-06-25 15:19:08.000 | CRITICAL | horde_worker_regen.process_management.process_manager:_give_up_on_wedged_jobs:2120 - "
            "Save-our-ship: gave up on 4 unservable job(s) (scheduler wedged with idle processes (queue "
            "deadlock) despite a healthy pool) and reported them faulted so the horde reissues them. "
            "Repeated drops like this can trigger horde-forced maintenance."
        )
        findings = _diagnose(tmp_path, self._bridge(enriched, _maintenance_pop("15:19:10.000")))
        assert findings["forced_maintenance"].severity is Severity.CRITICAL
        assert "4" in findings["forced_maintenance"].verdict


class TestSchedulerStarvationWedge:
    """The root cause: an over-conservative VRAM budget deferring head-of-queue jobs on an idle device.

    The budget refuses to admit the head-of-queue model on a device with ample free VRAM, so the queue
    deadlocks with idle processes; the recovery supervisor soft-resets the pools and faults the backlog.
    The detector must separate this self-inflicted wedge from a transient near-miss that force-admit absorbed.
    """

    def _bridge(self, *lines: str) -> str:
        return "\n".join([f"2026-06-25 13:59:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}", *lines])

    def test_wedge_with_soft_reset_and_giveup_is_critical(self, tmp_path: Path) -> None:
        """Starvation that escalated to a soft reset and faulted jobs is the critical root-cause finding."""
        bridge = self._bridge(
            _force_admit("15:18:52.000", starved_seconds=110, free_vram_mb=19179),
            _soft_reset("15:18:43.000"),
            _give_up("15:19:08.000", jobs=4),
        )
        findings = _diagnose(tmp_path, bridge)
        assert "scheduler_starvation_wedge" in findings
        finding = findings["scheduler_starvation_wedge"]
        assert finding.severity is Severity.CRITICAL
        # It reports the ample free VRAM (the budget's mistake) and the starvation duration.
        assert "19179" in finding.verdict
        assert "110" in finding.verdict

    def test_transient_force_admit_is_warning(self, tmp_path: Path) -> None:
        """A lone force-admit that broke the wedge without a soft reset is a near-miss warning, not critical."""
        bridge = self._bridge(_force_admit("14:21:44.000", starved_seconds=15, free_vram_mb=19829))
        findings = _diagnose(tmp_path, bridge)
        assert "scheduler_starvation_wedge" in findings
        assert findings["scheduler_starvation_wedge"].severity is Severity.WARNING

    def test_silent_for_crash_on_start(self, tmp_path: Path) -> None:
        """A crash-on-start give-up has no budget starvation, so the starvation detector stays silent."""
        bridge = self._bridge(
            _recovery("18:29:31.000", 1, reason="inference process replaced (crashed or hung)"),
            _recovery("18:29:40.000", 1, reason="inference process replaced (crashed or hung)"),
            _give_up("18:30:00.000", jobs=2),
        )
        findings = _diagnose(tmp_path, bridge)
        assert "scheduler_starvation_wedge" not in findings


class TestSlowGenerationDropSpiral:
    """The horde aborting generations as too slow: the drop mechanism behind a slow-worker maintenance.

    The scenario the starvation-wedge detector does not cover: the worker is not wedged, it is generating
    slower than the horde's per-job deadline. The server aborts each late submission and faults it, and a
    sustained run of those aborts draws forced maintenance. The detector must separate a
    sustained spiral (critical) from a couple of isolated slow jobs (warning), and must still fire when
    only the server aborts are present (no worker-side slowdown grading).
    """

    def _bridge(self, *lines: str) -> str:
        return "\n".join([f"2026-06-25 05:09:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}", *lines])

    def test_sustained_aborts_with_maintenance_is_critical(self, tmp_path: Path) -> None:
        """A run of slow-aborts that drew maintenance is the critical root-cause finding, with the metrics."""
        bridge = self._bridge(
            _slowdown_grade("05:15:13.000", ratio=4.0, free_vram_mb=5395),
            _server_slow_abort("06:38:37.000"),
            _server_slow_abort("06:42:12.000"),
            _server_slow_abort("07:03:16.000"),
            _slowdown_grade("07:04:00.000", ratio=4.1, free_vram_mb=5378),
            _maintenance_pop("07:33:02.000"),
        )
        finding = _diagnose(tmp_path, bridge)["slow_generation_drop_spiral"]
        assert finding.severity is Severity.CRITICAL
        assert "3 generation(s)" in finding.verdict
        # It corroborates with the worst slowdown ratio and the lowest free VRAM (the over-commit signature).
        assert "4.1x" in finding.verdict
        assert "5378 MB" in finding.verdict

    def test_three_aborts_without_maintenance_is_critical(self, tmp_path: Path) -> None:
        """A spiral is defined by a sustained abort run; it is critical even before maintenance lands."""
        bridge = self._bridge(
            _server_slow_abort("06:38:37.000"),
            _server_slow_abort("06:42:12.000"),
            _server_slow_abort("07:03:16.000"),
        )
        assert _diagnose(tmp_path, bridge)["slow_generation_drop_spiral"].severity is Severity.CRITICAL

    def test_isolated_aborts_are_a_warning(self, tmp_path: Path) -> None:
        """A couple of stray slow-aborts (below the spiral threshold, no maintenance) is a warning."""
        bridge = self._bridge(_server_slow_abort("06:38:37.000"), _server_slow_abort("06:42:12.000"))
        assert _diagnose(tmp_path, bridge)["slow_generation_drop_spiral"].severity is Severity.WARNING

    def test_fires_without_worker_side_slowdown_grading(self, tmp_path: Path) -> None:
        """The server aborts alone are enough; the worker-side grade only enriches the verdict."""
        bridge = self._bridge(
            _server_slow_abort("06:38:37.000"),
            _server_slow_abort("06:42:12.000"),
            _server_slow_abort("07:03:16.000"),
        )
        finding = _diagnose(tmp_path, bridge)["slow_generation_drop_spiral"]
        assert finding.severity is Severity.CRITICAL
        assert "expected sampling time" not in finding.verdict

    def test_silent_without_aborts(self, tmp_path: Path) -> None:
        """A worker that grades slow jobs but never has one server-aborted produces no spiral finding."""
        bridge = self._bridge(_slowdown_grade("05:15:13.000"))
        assert "slow_generation_drop_spiral" not in _diagnose(tmp_path, bridge)

    def test_queue_aging_is_distinguished_from_slow_gpu(self, tmp_path: Path) -> None:
        """Fast generation but long pop->submit latency is diagnosed as pipeline aging, not a slow GPU.

        When generation is fast but jobs age in the pipeline queue (pop->submit latency well above
        generation time), the cause is a downstream bottleneck -- typically a slow safety stage -- not a
        slow GPU. The detector must say so rather than blaming generation throughput.
        """
        bridge = self._bridge(
            _submit_latency("07:13:09.000", popped_ago=180.0, gen=7.0),
            _submit_latency("07:13:19.000", popped_ago=175.0, gen=8.0),
            _submit_latency("07:13:29.000", popped_ago=170.0, gen=7.5),
            _safety_duration("07:13:30.000", seconds=9.2),
            _server_slow_abort("07:14:00.000"),
            _server_slow_abort("07:18:00.000"),
            _server_slow_abort("07:22:00.000"),
            _maintenance_pop("07:33:02.000"),
        )
        finding = _diagnose(tmp_path, bridge)["slow_generation_drop_spiral"]
        assert finding.severity is Severity.CRITICAL
        assert "aged in the post-inference queue" in finding.verdict
        # The remediation must not blame max_power for a pipeline-balance problem.
        assert "backpressure" in finding.remediation
        assert "max_power will not help" in finding.remediation

    def test_genuinely_slow_generation_keeps_gpu_framing(self, tmp_path: Path) -> None:
        """When generation itself is slow (latency ~ generation time), keep the slow-GPU remediation."""
        bridge = self._bridge(
            _submit_latency("07:13:09.000", popped_ago=40.0, gen=38.0),
            _submit_latency("07:13:59.000", popped_ago=42.0, gen=39.0),
            _slowdown_grade("07:04:00.000", ratio=4.1, free_vram_mb=5378),
            _server_slow_abort("07:14:00.000"),
            _server_slow_abort("07:18:00.000"),
            _server_slow_abort("07:22:00.000"),
        )
        finding = _diagnose(tmp_path, bridge)["slow_generation_drop_spiral"]
        assert "aged in the post-inference queue" not in finding.verdict
        assert "Reduce max_power" in finding.remediation


class TestConsecutiveFailurePause:
    """The worker self-pausing job pops after three consecutive faults."""

    def _bridge(self, *lines: str) -> str:
        return "\n".join([f"2026-06-25 13:59:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}", *lines])

    def test_pause_detected(self, tmp_path: Path) -> None:
        """The self-pause is surfaced as a warning so the fault cause gets investigated."""
        findings = _diagnose(tmp_path, self._bridge(_consecutive_pause("15:19:11.000")))
        assert "consecutive_failure_pause" in findings
        assert findings["consecutive_failure_pause"].severity is Severity.WARNING

    def test_silent_when_no_pause(self, tmp_path: Path) -> None:
        """A healthy session never self-pauses, so there is no pause finding."""
        bridge = self._bridge("2026-06-25 15:19:11.000 | INFO | x:y:1 - all good")
        assert "consecutive_failure_pause" not in _diagnose(tmp_path, bridge)


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


def _safety_lost_result(ts: str, *, job_id: str = "ab3164c9") -> str:
    """The dispatcher line emitted when a safety verdict arrives for no tracked job (a lost result)."""
    return (
        f"2026-06-25 {ts} | ERROR    | horde_worker_regen.process_management.ipc.message_dispatcher:_handle_safety_result:766 - "
        f"Expected to find a completed job with ID {job_id} but none was found. This should only happen when "
        "certain process crashes occur."
    )


def _safety_requeue(ts: str, *, job_id: str = "ab3164c9", aged: int = 46, attempt: int = 1) -> str:
    """The safety-orphan watchdog requeuing a stranded job for a fresh check (recoverable case)."""
    return (
        f"2026-06-25 {ts} | WARNING  | horde_worker_regen.process_management.process_manager:_reconcile_orphaned_safety_jobs:0 - "
        f"Job {job_id} awaited a safety verdict for {aged}s with none returned; requeued it for a fresh safety "
        f"check (attempt {attempt}/3). Its images are re-checked, never submitted unchecked."
    )


def _safety_unrecoverable(ts: str, *, job_id: str = "ab3164c9") -> str:
    """The watchdog faulting a job with no image because safety could not check it (escalation)."""
    return (
        f"2026-06-25 {ts} | CRITICAL | horde_worker_regen.process_management.process_manager:_reconcile_orphaned_safety_jobs:0 - "
        f"Job {job_id} could not be safety-checked (requeued 3 times without a verdict); dropping its images and "
        "faulting it so the horde reissues it (an image the safety check never cleared is never submitted). "
        "Soft-pausing pops until safety recovers."
    )


def _safety_soft_pause(ts: str) -> str:
    """The worker soft-pausing pops because safety could not be relied on to check a result."""
    return (
        f"2026-06-25 {ts} | WARNING  | horde_worker_regen.process_management.process_manager:_engage_safety_soft_pause:0 - "
        "Soft-pausing job pops for 60s: safety could not check a result (requeued 3 times without a verdict). "
        "In-flight checked jobs still submit; pops resume automatically once safety recovers, so the worker does "
        "not keep taking on work it cannot safety-check."
    )


def _safety_backpressure(ts: str, *, backlog: int = 6, cap: int = 2, oldest: int = 145) -> str:
    """The popper withholding pops because the post-inference safety backlog is too deep."""
    return (
        f"2026-06-25 {ts} | WARNING  | horde_worker_regen.process_management.jobs.job_popper:api_job_pop:638 - "
        f"Withholding job pops: post-inference safety backlog {backlog} >= cap {cap} (oldest waiting safety job "
        f"{oldest}s). The safety stage is slower than inference; if this persists, enable safety_on_gpu or speed "
        "safety up."
    )


def _dispatch_stall(ts: str, *, reason: str, model: str = "AlbedoBase XL (SDXL)", parked: int = 30) -> str:
    """The scheduler explaining why a head-of-queue job is not dispatching."""
    return (
        f"2026-06-25 {ts} | WARNING  | horde_worker_regen.process_management.scheduling.inference_scheduler:_log_dispatch_stall_if_needed:0 - "
        f"Inference dispatch stalled: head 4006e936 ({model}) has been parked {parked}s -- {reason}."
    )


_DISPATCH_GATE_REASON = (
    "its model is resident and idle on process 1, but the concurrency cap is reached (in_progress=1, cap=1)"
)
_DISPATCH_BUG_REASON = (
    "its model is resident and idle on process 1 but dispatch was withheld with no matching gate -- this is a "
    "scheduler stall worth reporting"
)


class TestSafetyStageStall:
    """The downstream safety stall that strands jobs and (escalated) drives forced maintenance."""

    def test_lost_result_then_requeue_is_warning(self, tmp_path: Path) -> None:
        """A lost verdict that the watchdog re-checks (no drop) is a warning about the safety bottleneck."""
        bridge = "\n".join(
            [
                f"2026-06-25 13:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                _safety_lost_result("13:01:00.000"),
                _safety_requeue("13:01:46.000"),
            ],
        )
        findings = _diagnose(tmp_path, bridge)
        assert "safety_stage_stall" in findings
        assert findings["safety_stage_stall"].severity is Severity.WARNING

    def test_unrecoverable_and_soft_pause_is_critical(self, tmp_path: Path) -> None:
        """A no-image fault plus a soft-pause is the safety pipeline failing and dropping jobs (critical)."""
        bridge = "\n".join(
            [
                f"2026-06-25 13:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                _safety_lost_result("13:01:00.000"),
                _safety_unrecoverable("13:02:30.000"),
                _safety_soft_pause("13:02:30.500"),
            ],
        )
        findings = _diagnose(tmp_path, bridge)
        assert "safety_stage_stall" in findings
        assert findings["safety_stage_stall"].severity is Severity.CRITICAL
        assert findings["safety_stage_stall"].see_also == "forced_maintenance"

    def test_pure_backpressure_is_warning(self, tmp_path: Path) -> None:
        """Throttling intake to a slow safety stage (no orphan recovery) is the benign, lower-severity case."""
        bridge = "\n".join(
            [
                f"2026-06-25 13:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                _safety_backpressure("13:01:00.000"),
            ],
        )
        findings = _diagnose(tmp_path, bridge)
        assert "safety_stage_stall" in findings
        assert findings["safety_stage_stall"].severity is Severity.WARNING

    def test_silent_without_safety_signals(self, tmp_path: Path) -> None:
        """A healthy session emits none of the safety-stall signals."""
        bridge = "\n".join(
            [
                f"2026-06-25 13:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                "2026-06-25 13:00:10.000 | INFO | x:y:1 - Submitted generation abcd1234 for 50.00 kudos.",
            ],
        )
        assert "safety_stage_stall" not in _diagnose(tmp_path, bridge)


class TestHeadDispatchStall:
    """The scheduler naming why a head-of-queue job is parked, and flagging the gate-less anomaly."""

    def test_no_matching_gate_is_critical(self, tmp_path: Path) -> None:
        """A resident, idle-process head with no blocking gate that still does not dispatch is critical."""
        bridge = "\n".join(
            [
                f"2026-06-25 13:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                _dispatch_stall("13:01:00.000", reason=_DISPATCH_BUG_REASON),
            ],
        )
        findings = _diagnose(tmp_path, bridge)
        assert "head_dispatch_stall" in findings
        assert findings["head_dispatch_stall"].severity is Severity.CRITICAL
        assert findings["head_dispatch_stall"].see_also == "scheduler_starvation_wedge"

    def test_known_gate_is_warning(self, tmp_path: Path) -> None:
        """A head parked by a named gate (concurrency cap) is a throughput warning, not a wedge."""
        bridge = "\n".join(
            [
                f"2026-06-25 13:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                _dispatch_stall("13:01:00.000", reason=_DISPATCH_GATE_REASON),
                _dispatch_stall("13:01:40.000", reason=_DISPATCH_GATE_REASON),
            ],
        )
        findings = _diagnose(tmp_path, bridge)
        assert "head_dispatch_stall" in findings
        assert findings["head_dispatch_stall"].severity is Severity.WARNING

    def test_silent_without_stall(self, tmp_path: Path) -> None:
        """No dispatch-stall line means no finding."""
        bridge = "\n".join(
            [
                f"2026-06-25 13:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                "2026-06-25 13:00:10.000 | INFO | x:y:1 - Starting inference for job 4006e936 on process 1",
            ],
        )
        assert "head_dispatch_stall" not in _diagnose(tmp_path, bridge)
