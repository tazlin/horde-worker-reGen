"""Unit tests for the diagnosis detectors over synthetic sessions.

The marquee case is the one from the real incident: an inference pool that crashes on start and a worker
that spins through a recovery storm without ever giving up. The detectors must lift the child's
exception across the process boundary and distinguish "never gave up" (the bug) from "gave up cleanly"
(the healthy bail-out), even though both arise from the same crash cause.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from horde_worker_regen.analysis.bundle import LogBundle
from horde_worker_regen.analysis.correlate import build_session_context
from horde_worker_regen.analysis.detectors import Finding, Severity, _records_in_window, run_detectors
from horde_worker_regen.analysis.log_ingest import LogRecord
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


def _stuck_step_reap(ts: str, *, slot: int = 3, repeats: int = 3060) -> str:
    """The stuck-step watchdog's reap line for a slot looping on one sampling step."""
    return (
        f"2026-06-26 {ts} | ERROR    | horde_worker_regen.process_management.lifecycle.process_lifecycle:replace_hung_processes:1830 - "
        f"Inference slot {slot} is stuck on a non-advancing sampling step (reported step 24/25 without "
        f"advancing {repeats} times); the ComfyUI generation will not return a result, replacing it "
        f"(stuck-step watchdog)."
    )


class TestStuckInferenceStep:
    """The detector for a slot wedged repeating one sampling step (the non-silent hang)."""

    def _bridge(self, *lines: str) -> str:
        return "\n".join(
            [f"2026-06-26 09:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}", *lines],
        )

    def test_fires_and_points_at_the_lora_shape_cause(self, tmp_path: Path) -> None:
        """The finding surfaces as a WARNING and steers the operator toward the upstream model/LoRA fault."""
        findings = _diagnose(tmp_path, self._bridge(_stuck_step_reap("09:48:02.000")))
        assert "stuck_inference_step" in findings
        finding = findings["stuck_inference_step"]
        assert finding.severity is Severity.WARNING
        assert "lora" in finding.remediation.lower()

    def test_silent_without_a_reap(self, tmp_path: Path) -> None:
        """No reap line means no finding (the detector keys on the watchdog's own emit)."""
        findings = _diagnose(tmp_path, self._bridge())
        assert "stuck_inference_step" not in findings


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
        return "\n".join(
            [
                f"2026-06-25 13:59:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                *lines,
            ]
        )

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
        return "\n".join(
            [
                f"2026-06-25 13:59:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                *lines,
            ]
        )

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
        return "\n".join(
            [
                f"2026-06-25 05:09:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                *lines,
            ]
        )

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
        generation time), the cause is a downstream bottleneck (typically a slow safety stage), not a
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
        return "\n".join(
            [
                f"2026-06-25 13:59:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                *lines,
            ]
        )

    def test_pause_detected(self, tmp_path: Path) -> None:
        """The self-pause is surfaced as a warning so the fault cause gets investigated."""
        findings = _diagnose(tmp_path, self._bridge(_consecutive_pause("15:19:11.000")))
        assert "consecutive_failure_pause" in findings
        assert findings["consecutive_failure_pause"].severity is Severity.WARNING

    def test_silent_when_no_pause(self, tmp_path: Path) -> None:
        """A healthy session never self-pauses, so there is no pause finding."""
        bridge = self._bridge("2026-06-25 15:19:11.000 | INFO | x:y:1 - all good")
        assert "consecutive_failure_pause" not in _diagnose(tmp_path, bridge)


def _oom_coresident(ts: str, *, slot: int = 4, model: str = "Z-Image-Turbo") -> str:
    """A faulted-inference OOM carrying the allocator's co-residency accounting (the over-admission case).

    Mirrors the real emit: the generic wrapper, the faulting model, the free-VRAM figure, and two sibling
    'Process N has X GiB memory in use' lines (so total co-residency is 3 with the faulting process).
    """
    return (
        f"2026-06-24 {ts} | ERROR    | horde_worker_regen.process_management.ipc.message_dispatcher:_handle_faulted_inference_result:912 - "
        f"Job 9cbb045c faulted on process {slot}: RuntimeError: Pipeline failed to run - declared output "
        f"node(s) ['output_image'] produced no results. Model: {model}. Error: sampler (KSampler): "
        f"torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 30.00 MiB. GPU 0 has a total "
        f"capacity of 23.51 GiB of which 193.56 MiB is free. Process 2733578 has 3.97 GiB memory in use. "
        f"Process 2733829 has 3.43 GiB memory in use. Including non-PyTorch memory, this process has 12.85 "
        f"GiB memory in use."
    )


def _fd_fault(
    ts: str,
    *,
    slot: int = 3,
    model: str = "WAI-NSFW-illustrious-SDXL",
    node: str = "sampler",
    resource: str = "/proc/meminfo",
) -> str:
    """A faulted-inference result whose underlying error is EMFILE (errno 24, 'Too many open files')."""
    return (
        f"2026-06-24 {ts} | WARNING  | horde_worker_regen.process_management.ipc.message_dispatcher:_handle_faulted_inference_result:892 - "
        f"Job 597d4471 faulted on process {slot} (RuntimeError: Pipeline failed to run - declared output "
        f"node(s) ['output_image'] produced no results. Model: {model}. Error: {node} (HordeCheckpointLoader): "
        f"OSError: [Errno 24] Too many open files: '{resource}'"
    )


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

    def test_oom_names_model_and_coresidency(self, tmp_path: Path) -> None:
        """The OOM finding names the faulting model and the card's over-admission fingerprint.

        A bare count ("8 OOM faults") does not tell a maintainer whether one model was too large or many
        were over-admitted. The allocator message carries both; the finding must lift them.
        """
        bridge = "\n".join([f"2026-06-24 18:00:00.000 | DEBUG | x:y:1 - {_STARTUP}", _oom_coresident("18:00:10.000")])
        findings = _diagnose(tmp_path, bridge)
        assert "oom" in findings
        verdict = findings["oom"].verdict
        assert "Z-Image-Turbo" in verdict
        assert "slot(s) 4" in verdict
        # Two sibling "Process N has ..." lines + the faulting process itself == 3 co-resident.
        assert "3 processes co-resident" in verdict
        assert "194 MiB free" in verdict

    def test_fd_exhaustion_detected(self, tmp_path: Path) -> None:
        """An EMFILE (errno 24) run of faults is surfaced as its own critical finding.

        This must not be swallowed by the OOM detector: it shares the generic 'produced no results'
        wrapper but is a descriptor leak, needing a different fix. The finding names the slot, the model
        that was running when the ceiling was hit, the refused open, and the slot replacement.
        """
        bridge = "\n".join(
            [
                f"2026-06-24 20:00:00.000 | DEBUG | x:y:1 - {_STARTUP}",
                _fd_fault("20:09:24.000", node="sampler", resource="/proc/meminfo"),
                _fd_fault("20:13:46.000", node="model_loader", resource="/proc/meminfo"),
                _fd_fault("20:16:00.000", node="model_loader", resource="/proc/2733826/stat"),
                _recovery(
                    "20:16:16.000",
                    3,
                    reason="inference process replaced (failed to load model WAI-NSFW-illustrious-SDXL)",
                    last_state="PRELOADING_MODEL",
                ),
            ],
        )
        findings = _diagnose(tmp_path, bridge)
        assert "file_descriptor_exhaustion" in findings
        fd = findings["file_descriptor_exhaustion"]
        assert fd.severity is Severity.CRITICAL
        assert "WAI-NSFW-illustrious-SDXL" in fd.verdict
        assert "slot(s) 3" in fd.verdict
        assert "/proc/meminfo" in fd.verdict
        assert "replaced" in fd.verdict
        # The mechanism note that keeps a maintainer from mistaking it for an OOM.
        assert "descriptor leak" in fd.verdict

    def test_fd_exhaustion_and_oom_are_distinct(self, tmp_path: Path) -> None:
        """A descriptor-exhaustion session raises no OOM finding, and vice versa (no cross-contamination)."""
        fd_bridge = "\n".join(
            [f"2026-06-24 20:00:00.000 | DEBUG | x:y:1 - {_STARTUP}", _fd_fault("20:09:24.000")],
        )
        fd_findings = _diagnose(tmp_path, fd_bridge)
        assert "file_descriptor_exhaustion" in fd_findings
        assert "oom" not in fd_findings

        oom_bridge = "\n".join(
            [f"2026-06-24 20:00:00.000 | DEBUG | x:y:1 - {_STARTUP}", _oom_coresident("20:20:26.000")],
        )
        oom_findings = _diagnose(tmp_path, oom_bridge)
        assert "oom" in oom_findings
        assert "file_descriptor_exhaustion" not in oom_findings


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
        f"Inference dispatch stalled: head 4006e936 ({model}) has been parked {parked}s: {reason}."
    )


def _whole_card_reserve(
    ts: str,
    *,
    model: str = "AlbedoBase XL (SDXL)",
    current: int = 4,
    after: int = 3,
    total: int = 4,
    target: int = 3,
    free_mb: int = 9443,
) -> str:
    """inference_scheduler._establish_whole_card_residency: the worker reserving the device for a model.

    The process-count figures and the residency snapshot's ``device_free_vram`` are what say whether the
    claim reduced anything, so they are parameterized rather than baked into the fixture.
    """
    return (
        f"2026-06-25 {ts} | WARNING  | horde_worker_regen.process_management.scheduling.inference_scheduler:_establish_whole_card_residency:1043 - "
        f"Whole-card residency: reserving the device for {model} (inference processes {current} -> {after} of "
        f"{total}, target {target}) and moving safety off-GPU. Its weights + activations need the whole ~24GB "
        "card; co-resident siblings/safety would force the driver to stream activations to host RAM and run "
        f"several times slower. slots=[#1:{model}[WAITING_FOR_JOB]] device_free_vram={free_mb}MB"
    )


def _stream_forecast(
    ts: str,
    *,
    model: str = "AlbedoBase XL (SDXL)",
    marginal_mb: int = 276,
    source: str = "probe",
    unreclaimable_mb: int = 1450,
    needs_reduction: bool = True,
) -> str:
    """inference_scheduler._log_stream_forecast: the establishment-time arithmetic behind a claim."""
    return (
        f"2026-06-25 {ts} | DEBUG    | horde_worker_regen.process_management.scheduling.inference_scheduler:_log_stream_forecast:1345 - "
        f"Stream forecast for {model}: weights ~9158 MB + 1519 MB reserve do not fit 8679 MB free: exclusive "
        f"[free_now=8679.0, after_model_evict=10129.0, alone=16100.0, unreclaimable={unreclaimable_mb}MB, "
        f"live_procs=2, overhead/proc=275MB, marginal/ctx={marginal_mb}MB(src={source},probe=276,"
        f"idle_floor=143)] -> coresident=False, needs_exclusive=True, "
        f"needs_process_count_reduction={needs_reduction}(max_resident=2), streams_unavoidably=False"
    )


def _fault_report(ts: str, *, job_id: str, popped: float = 419.31) -> str:
    """job_submitter.submit_single_generation: the terminal per-job fault reported to the horde."""
    return (
        f"2026-06-25 {ts} | ERROR    | horde_worker_regen.process_management.jobs.job_submitter:submit_single_generation:449 - "
        f"{job_id} faulted. Reported fault to the horde. Job popped {popped} seconds ago and took 0.00 to generate."
    )


def _popped_job(ts: str, *, job_id: str, model: str = "AlbedoBase XL (SDXL)") -> str:
    """job_popper.api_job_pop: the pop line that binds a job id to the model it was popped for."""
    return (
        f"2026-06-25 {ts} | INFO     | horde_worker_regen.process_management.jobs.job_popper:api_job_pop:2122 - "
        f"Popped job {job_id} (38 eMPS) (model: {model}, batch: 2, loras: False, post_processing: False)"
    )


def _faulted_on_process(
    ts: str,
    *,
    job_id: str,
    slot: int = 3,
    model: str = "AlbedoBase XL (SDXL)",
    error: str = "sampler (KSampler): RuntimeError: boom",
    requeued: bool = False,
) -> str:
    """message_dispatcher._handle_faulted_inference_result: a job faulted on the slot that ran it.

    ``requeued`` produces the bounded-retry wording, which is the same line for an attempt the tracker
    handed back rather than a job the worker lost.
    """
    return (
        f"2026-06-25 {ts} | WARNING  | horde_worker_regen.process_management.ipc.message_dispatcher:_handle_faulted_inference_result:892 - "
        f"Job {job_id} faulted on process {slot} (RuntimeError: Pipeline failed to run - declared output "
        f"node(s) ['output_image'] produced no results. Model: {model}. Error: {error}"
        + ("); requeued for another attempt." if requeued else "")
    )


def _sample_stage_fault(
    ts: str,
    *,
    job_id: str,
    error: str = "Model reference for category image_generation not found or could not be parsed.",
) -> str:
    """inference_process._run_sample_stage: the disaggregated sample stage faulting a job (child log)."""
    return (
        f"2026-06-25 {ts} | ERROR    | horde_worker_regen.process_management.workers.inference_process:_run_sample_stage:1206 - "
        f"Sample stage faulted for job {job_id}: {error}"
    )


def _model_reference_stale(ts: str, *, filename: str = "stable_diffusion.json") -> str:
    """horde_model_reference replica backend: a cached category file going stale under an in-flight job."""
    return (
        f"2026-06-25 {ts} | DEBUG    | horde_model_reference.backends.replica_backend_base:needs_refresh:332 - "
        f"File {filename} mtime changed, needs refresh; the cache is stale"
    )


def _residency_governor_enter(ts: str, *, model: str = "WAI-NSFW-illustrious-SDXL") -> str:
    """PopGovernorRegistry: the whole-card residency spell opening (the hold's ENTER boundary)."""
    return (
        f"2026-06-25 {ts} | INFO     | horde_worker_regen.process_management.scheduling.pop_governor_registry:_default_log:200 - "
        f"Pop governor ENTER: whole_card_residency ({model} holds the card (establishing)); expected ~45s"
    )


def _residency_governor_exit(ts: str) -> str:
    """PopGovernorRegistry: the whole-card residency spell closing (the hold's EXIT boundary)."""
    return (
        f"2026-06-25 {ts} | INFO     | horde_worker_regen.process_management.scheduling.pop_governor_registry:_default_log:200 - "
        "Pop governor EXIT: whole_card_residency after 04m00s (1x this session, 04m00s total)"
    )


def _full_queue_frozen(
    ts: str,
    *,
    frozen_seconds: int = 120,
    waiting: int = 4,
    model: str = "Nova Anime XL",
    blocker: str | None = None,
) -> str:
    """process_manager._full_queue_frozen_line: the local queue full and motionless (the total-stall signal)."""
    blocker_text = f" The scheduler names the block as: {blocker}." if blocker else ""
    return (
        f"2026-06-25 {ts} | ERROR    | horde_worker_regen.process_management.process_manager:_check_full_queue_liveness:5181 - "
        f"Pop liveness: the local job queue has been full and not draining for {frozen_seconds}s "
        f"({waiting} accepted job(s) waiting, head model '{model}'), with nothing dispatched and nothing "
        "completed in that time. A full queue holds pops back legitimately only while it moves, so the worker "
        f"is serving nothing.{blocker_text}"
    )


_FLUX = "Flux.1-Schnell fp8 (Compact)"
"""The heavy model used as the whole-card residency's claimant in the pop-claim fixtures."""


def _pop_claim_engaged(ts: str, *, model: str = _FLUX, max_hold: int = 180) -> str:
    """inference_scheduler._disclose_pop_claim_edge: a residency taking the worker's pop offer."""
    return (
        f"2026-06-25 {ts} | INFO | horde_worker_regen.process_management.scheduling.inference_scheduler:_disclose_pop_claim_edge:3012 - "
        f"Whole-card pop claim engaged for {model}: advertising that model alone while it holds the card, for "
        f"at most {max_hold}s."
    )


def _pop_claim_released(ts: str, *, model: str = _FLUX, release: str = "the maximum hold elapsed") -> str:
    """inference_scheduler._disclose_pop_claim_edge: the claim ending, naming which of its ends fired."""
    return (
        f"2026-06-25 {ts} | INFO | horde_worker_regen.process_management.scheduling.inference_scheduler:_disclose_pop_claim_edge:3027 - "
        f"Whole-card pop claim released for {model}: {release}; advertising the full pool again."
    )


_DISPATCH_GATE_REASON = (
    "its model is resident and idle on process 1, but the concurrency cap is reached (in_progress=1, cap=1)"
)
_DISPATCH_BUG_REASON = (
    "its model is resident and idle on process 1 but dispatch was withheld with no matching gate; this is a "
    "scheduler stall worth reporting"
)
# The dispatch residency-reconciliation attribution (inference_scheduler._classify_dispatch_stall): the head
# is resident and idle but the dispatch gate is holding it while it evicts idle VRAM so the head's on-device
# materialisation fits. A benign, self-clearing swap-churn wait, not the gate-less scheduler-bug stall.
_DISPATCH_RECONCILE_REASON = (
    "its model is resident and idle on process 1, but dispatch is held to reconcile residency (evicting idle "
    "VRAM): its materialisation would over-commit the card until an idle resident is evicted, and it "
    "dispatches once that eviction frees room"
)
# The post-processing co-residency defer attribution (inference_scheduler._classify_dispatch_stall): the head
# is resident and idle but an in-flight post-processing chain's committed VRAM and this job's sampling peak
# cannot share the card, so dispatch is held until the chain finishes. A real named head-park (worth seeing as
# a warning), not the gate-less scheduler-bug stall.
_DISPATCH_POST_PROCESSING_REASON = (
    "its model is resident and idle on process 1, but dispatch is held while an in-flight post-processing "
    "chain finishes: the chain's committed VRAM and this job's sampling peak cannot share the card, and it "
    "dispatches once the chain releases the device"
)
# The whole-card residency convergence attribution (inference_scheduler._diagnose_dispatch_stall): the head is
# resident and idle, but an idle sibling holding a still-queued model was not torn down by the convergence.
_DISPATCH_WHOLE_CARD_REASON = (
    "its model is resident and idle on process 4, but the whole-card residency stuck: cannot reach sole "
    "residency because process 3 holds queued model 'CyberRealistic Pony'; the convergence teardown should "
    "have stopped that idle sibling (only the head's holder is spared), so the shrink has not collapsed the "
    "pool and the head never dispatches"
)
# The non-head whole-card residency attribution (inference_scheduler._diagnose_dispatch_stall): the head's
# model is not resident because a residency is held for a different, deeper-queue model that reserved the card.
_DISPATCH_NONHEAD_REASON = (
    "its model is not resident because a whole-card residency is held for non-head model "
    "'Flux.1-Schnell fp8 (Compact)'; the card is reserved for that model and its siblings were torn down, so "
    "this head cannot load until that residency restores"
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


class TestWholeCardConvergenceWedge:
    """The whole-card residency that cannot collapse to sole residency because a queued-model sibling is pinned."""

    def test_queued_model_sibling_pins_teardown_is_critical(self, tmp_path: Path) -> None:
        """A pre-staged head parked by a queued-model sibling is the convergence deadlock (critical)."""
        bridge = "\n".join(
            [
                f"2026-06-25 13:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                _dispatch_stall(
                    "13:01:00.000", reason=_DISPATCH_WHOLE_CARD_REASON, model="Flux.1-Schnell fp8 (Compact)"
                ),
            ],
        )
        findings = _diagnose(tmp_path, bridge)
        assert "whole_card_convergence_wedge" in findings
        assert findings["whole_card_convergence_wedge"].severity is Severity.CRITICAL
        assert findings["whole_card_convergence_wedge"].see_also == "head_dispatch_stall"

    def test_wedge_line_does_not_also_fire_generic_dispatch_warning(self, tmp_path: Path) -> None:
        """The wedge owns its line: head_dispatch_stall must not double-report it as a generic warning."""
        bridge = "\n".join(
            [
                f"2026-06-25 13:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                _dispatch_stall(
                    "13:01:00.000", reason=_DISPATCH_WHOLE_CARD_REASON, model="Flux.1-Schnell fp8 (Compact)"
                ),
            ],
        )
        findings = _diagnose(tmp_path, bridge)
        assert "whole_card_convergence_wedge" in findings
        assert "head_dispatch_stall" not in findings


class TestWholeCardNonHeadResidencyStarvation:
    """A whole-card residency held for a non-head model that starves the actual head of the queue."""

    def test_starvation_with_soft_reset_is_critical(self, tmp_path: Path) -> None:
        """A head parked behind a non-head residency that escalates to a soft reset is the wedge (critical)."""
        bridge = "\n".join(
            [
                f"2026-06-25 13:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                _dispatch_stall("13:01:00.000", reason=_DISPATCH_NONHEAD_REASON, model="Juggernaut XL"),
                _soft_reset("13:01:30.000"),
            ],
        )
        findings = _diagnose(tmp_path, bridge)
        assert "whole_card_nonhead_residency_starvation" in findings
        assert findings["whole_card_nonhead_residency_starvation"].severity is Severity.CRITICAL
        assert findings["whole_card_nonhead_residency_starvation"].see_also == "scheduler_starvation_wedge"

    def test_starvation_without_escalation_is_warning(self, tmp_path: Path) -> None:
        """A non-head residency stall that did not escalate to a soft reset or drops is the lower-severity case."""
        bridge = "\n".join(
            [
                f"2026-06-25 13:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                _dispatch_stall("13:01:00.000", reason=_DISPATCH_NONHEAD_REASON, model="Juggernaut XL"),
            ],
        )
        findings = _diagnose(tmp_path, bridge)
        assert "whole_card_nonhead_residency_starvation" in findings
        assert findings["whole_card_nonhead_residency_starvation"].severity is Severity.WARNING

    def test_line_does_not_also_fire_generic_dispatch_warning(self, tmp_path: Path) -> None:
        """The non-head residency owns its line: head_dispatch_stall must not double-report it as a warning."""
        bridge = "\n".join(
            [
                f"2026-06-25 13:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                _dispatch_stall("13:01:00.000", reason=_DISPATCH_NONHEAD_REASON, model="Juggernaut XL"),
            ],
        )
        findings = _diagnose(tmp_path, bridge)
        assert "whole_card_nonhead_residency_starvation" in findings
        assert "head_dispatch_stall" not in findings


class TestWholeCardResidencyChurn:
    """Repeated whole-card reservations in a session: the over-eager-reservation signature."""

    def _bridge(self, *lines: str) -> str:
        return "\n".join(
            [f"2026-06-25 07:50:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}", *lines],
        )

    def test_repeated_reservations_are_flagged_as_churn(self, tmp_path: Path) -> None:
        """Three reservations in a session is churn, surfaced as a warning when it did not escalate."""
        findings = _diagnose(
            tmp_path,
            self._bridge(
                _whole_card_reserve("07:54:50.000"),
                _whole_card_reserve("07:55:45.000", model="CyberRealistic Pony"),
                _whole_card_reserve("07:57:46.000"),
            ),
        )
        assert "whole_card_residency_churn" in findings
        assert findings["whole_card_residency_churn"].severity is Severity.WARNING

    def test_churn_with_soft_reset_is_critical(self, tmp_path: Path) -> None:
        """Reservation churn that escalated to a soft reset is the wedge-feeding case (critical)."""
        findings = _diagnose(
            tmp_path,
            self._bridge(
                _whole_card_reserve("07:54:50.000"),
                _whole_card_reserve("07:55:45.000"),
                _whole_card_reserve("07:57:46.000"),
                _soft_reset("07:59:45.000"),
            ),
        )
        assert findings["whole_card_residency_churn"].severity is Severity.CRITICAL

    def test_a_single_reservation_is_not_churn(self, tmp_path: Path) -> None:
        """One deliberate reservation is normal and must not fire (only sustained cycling does)."""
        findings = _diagnose(tmp_path, self._bridge(_whole_card_reserve("07:54:50.000")))
        assert "whole_card_residency_churn" not in findings

    def test_the_claims_own_figures_are_reported(self, tmp_path: Path) -> None:
        """The finding quotes the process-count arithmetic and free-VRAM the reservation line carries."""
        findings = _diagnose(
            tmp_path,
            self._bridge(
                _whole_card_reserve("07:54:50.000", current=4, after=3, target=3, free_mb=9443),
                _whole_card_reserve("07:55:45.000", current=4, after=3, target=3, free_mb=10500),
                _whole_card_reserve("07:57:46.000", current=4, after=3, target=3, free_mb=11000),
            ),
        )
        verdict = findings["whole_card_residency_churn"].verdict
        assert "4 -> 3" in verdict, "the live/after process counts must be reported"
        assert "9443" in verdict and "11000" in verdict, "the device_free_vram range must be reported"

    def test_a_target_at_or_above_the_live_count_is_the_headline(self, tmp_path: Path) -> None:
        """A claim whose target never reduced the pool is structurally incoherent, and leads the verdict."""
        findings = _diagnose(
            tmp_path,
            self._bridge(
                _whole_card_reserve("07:54:50.000", current=2, after=2, total=2, target=21),
                _whole_card_reserve("07:55:45.000", current=2, after=2, total=2, target=21),
                _whole_card_reserve("07:57:46.000", current=2, after=2, total=2, target=21),
            ),
        )
        finding = findings["whole_card_residency_churn"]
        assert finding.verdict.startswith("Every one of the 3 reservation(s)"), finding.verdict
        assert "demanded no reduction" in finding.verdict
        assert "target 21" in finding.verdict
        assert "unmeasured marginal" not in finding.remediation, (
            "an incoherent claim must not be blamed on an unmeasured per-context marginal"
        )
        assert "target" in finding.remediation

    def test_a_measured_marginal_suppresses_the_unmeasured_remediation(self, tmp_path: Path) -> None:
        """With ``src=probe`` in the forecast, the per-context marginal is measured, so that fix is wrong."""
        findings = _diagnose(
            tmp_path,
            self._bridge(
                _stream_forecast("07:54:49.000", source="probe", marginal_mb=276),
                _whole_card_reserve("07:54:50.000"),
                _whole_card_reserve("07:55:45.000"),
                _whole_card_reserve("07:57:46.000"),
            ),
        )
        finding = findings["whole_card_residency_churn"]
        assert "unmeasured marginal" not in finding.verdict
        assert "measured" in finding.remediation
        assert "276MB" in finding.verdict, "the measured marginal must be quoted"
        assert "1450" in finding.verdict, "the unreclaimable charge behind the reduction must be quoted"

    def test_a_seeded_marginal_keeps_the_unmeasured_remediation(self, tmp_path: Path) -> None:
        """Only a seeded source means nothing was measured, which is when that remediation is true."""
        findings = _diagnose(
            tmp_path,
            self._bridge(
                _stream_forecast("07:54:49.000", source="seeded"),
                _whole_card_reserve("07:54:50.000"),
                _whole_card_reserve("07:55:45.000"),
                _whole_card_reserve("07:57:46.000"),
            ),
        )
        assert "measured" in findings["whole_card_residency_churn"].remediation
        assert "unmeasured" in findings["whole_card_residency_churn"].verdict


_GIVEN_UP_JOB = "34ce8495-7820-4af5-9723-b411698ff27f"
_SLOT_FAULT_JOB = "597d4471-b223-4383-b874-86c6a1549594"
_SAFETY_FAULT_JOB = "ebc711be-20b4-4dac-bdab-5b005fdf6c11"
_STAGE_FAULT_JOB = "3217e6c9-30f3-4520-a810-672132a362cc"


class TestFaultedJobCensus:
    """Every job the worker reported faulted, with the cause read off the lines around it."""

    @staticmethod
    def _bridge(*lines: str) -> str:
        return "\n".join(
            [f"2026-06-25 07:50:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}", *lines],
        )

    def test_no_faults_produces_no_finding(self, tmp_path: Path) -> None:
        """A clean session must not carry a census."""
        assert "faulted_job_census" not in _diagnose(tmp_path, self._bridge())

    def test_every_faulted_job_is_enumerated_with_its_model(self, tmp_path: Path) -> None:
        """Each fault is one census entry, named by job id and the model it was popped for."""
        findings = _diagnose(
            tmp_path,
            self._bridge(
                _popped_job("07:51:00.000", job_id=_GIVEN_UP_JOB, model="WAI-NSFW-illustrious-SDXL"),
                _give_up("07:58:00.000", jobs=3),
                _fault_report("07:58:00.500", job_id=_GIVEN_UP_JOB),
            ),
        )
        finding = findings["faulted_job_census"]
        assert "1 job(s) faulted" in finding.verdict
        assert any(_GIVEN_UP_JOB[:8] in line for line in finding.evidence)
        assert any("WAI-NSFW-illustrious-SDXL" in line for line in finding.evidence)

    def test_causes_are_classified_and_counted(self, tmp_path: Path) -> None:
        """The distinct causes and their counts are the census's headline breakdown."""
        findings = _diagnose(
            tmp_path,
            self._bridge(
                _give_up("07:58:00.000", jobs=3),
                _fault_report("07:58:00.500", job_id=_GIVEN_UP_JOB),
                _safety_unrecoverable("07:59:00.000", job_id=_SAFETY_FAULT_JOB),
                _fault_report("07:59:01.000", job_id=_SAFETY_FAULT_JOB),
                _faulted_on_process("08:00:00.000", job_id=_SLOT_FAULT_JOB),
                _fault_report("08:01:01.000", job_id=_STAGE_FAULT_JOB),
            ),
            {"bridge_5.log": _sample_stage_fault("08:01:00.000", job_id=_STAGE_FAULT_JOB)},
        )
        finding = findings["faulted_job_census"]
        assert "4 job(s) faulted" in finding.verdict
        for cause in ("give-up backstop", "safety-unrecoverable", "disaggregation stage fault", "process fault"):
            assert cause in finding.verdict, f"{cause} missing from {finding.verdict}"

    def test_a_malformed_pop_is_attributed_not_left_unexplained(self, tmp_path: Path) -> None:
        """A job popped with no model name is unservable at the boundary, so it is named, not bucketed."""
        findings = _diagnose(
            tmp_path,
            self._bridge(
                _empty_model_pop("07:51:00.000", job_id=_GIVEN_UP_JOB),
                _fault_report("07:51:05.000", job_id=_GIVEN_UP_JOB),
            ),
        )
        finding = findings["faulted_job_census"]
        assert "malformed pop (no model name)" in finding.verdict
        assert "other" not in finding.verdict

    def test_a_requeued_attempt_is_not_a_faulted_job(self, tmp_path: Path) -> None:
        """A slot fault handed back for a retry did not cost the horde a job, so it is not in the census."""
        findings = _diagnose(
            tmp_path,
            self._bridge(
                _faulted_on_process("08:00:00.000", job_id=_SLOT_FAULT_JOB, requeued=True),
                _fault_report("08:01:00.000", job_id=_GIVEN_UP_JOB),
            ),
        )
        finding = findings["faulted_job_census"]
        assert "1 job(s) faulted" in finding.verdict
        assert all(_SLOT_FAULT_JOB[:8] not in line for line in finding.evidence)

    def test_a_job_is_counted_once_across_both_fault_surfaces(self, tmp_path: Path) -> None:
        """A slot fault that is later reported to the horde is one faulted job, not two."""
        findings = _diagnose(
            tmp_path,
            self._bridge(
                _faulted_on_process("08:00:00.000", job_id=_SLOT_FAULT_JOB),
                _fault_report("08:00:02.000", job_id=_SLOT_FAULT_JOB),
            ),
        )
        assert "1 job(s) faulted" in findings["faulted_job_census"].verdict


class TestPopLivenessFullQueue:
    """The worker's most complete stall: a full local queue that stopped moving entirely."""

    @staticmethod
    def _bridge(*lines: str) -> str:
        return "\n".join(
            [f"2026-06-25 07:50:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}", *lines],
        )

    def test_a_frozen_full_queue_is_high_severity(self, tmp_path: Path) -> None:
        """One frozen-queue disclosure is enough: nothing was dispatched and nothing completed."""
        findings = _diagnose(tmp_path, self._bridge(_full_queue_frozen("07:52:00.000")))
        finding = findings["pop_liveness_full_queue"]
        assert finding.severity is Severity.CRITICAL
        assert "Nova Anime XL" in finding.verdict
        assert "120s" in finding.verdict

    def test_a_concurrent_residency_hold_is_named(self, tmp_path: Path) -> None:
        """A whole-card residency governor spell open across the freeze is the correlation that explains it."""
        findings = _diagnose(
            tmp_path,
            self._bridge(
                _residency_governor_enter("07:51:00.000"),
                _full_queue_frozen("07:52:00.000", blocker=_DISPATCH_NONHEAD_REASON),
                _residency_governor_exit("07:55:00.000"),
            ),
        )
        verdict = findings["pop_liveness_full_queue"].verdict
        assert "whole-card residency" in verdict, "the residency holding the card over the freeze must be named"

    def test_no_open_residency_spell_is_not_correlated(self, tmp_path: Path) -> None:
        """A spell that closed before the freeze is not what is holding the queue, so it is not claimed to be."""
        findings = _diagnose(
            tmp_path,
            self._bridge(
                _residency_governor_enter("07:50:10.000"),
                _residency_governor_exit("07:50:40.000"),
                _full_queue_frozen("07:52:00.000"),
            ),
        )
        assert "whole-card residency" not in findings["pop_liveness_full_queue"].verdict

    def test_a_healthy_session_does_not_fire(self, tmp_path: Path) -> None:
        """No frozen-queue disclosure, no finding."""
        assert "pop_liveness_full_queue" not in _diagnose(tmp_path, self._bridge())


class TestModelReferenceSampleFault:
    """A model-reference cache refresh landing under an in-flight sample stage."""

    @staticmethod
    def _bridge(*lines: str) -> str:
        return "\n".join(
            [f"2026-06-25 07:50:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}", *lines],
        )

    def test_the_child_fault_signature_fires(self, tmp_path: Path) -> None:
        """The fault is in a slot log, so the detector must read the child records, not just the parent."""
        findings = _diagnose(
            tmp_path,
            self._bridge("2026-06-25 07:59:00.000 | INFO | x:y:1 - working"),
            {"bridge_5.log": _sample_stage_fault("07:55:00.000", job_id=_STAGE_FAULT_JOB)},
        )
        finding = findings["model_reference_sample_fault"]
        assert finding.severity is Severity.WARNING
        assert _STAGE_FAULT_JOB[:8] in finding.verdict or any(
            _STAGE_FAULT_JOB[:8] in line for line in finding.evidence
        )
        assert "model-reference cache refresh" in finding.remediation.lower()

    def test_a_concurrent_stale_cache_is_reported(self, tmp_path: Path) -> None:
        """The staleness lines around the fault are what attribute it to a refresh, so they are counted."""
        findings = _diagnose(
            tmp_path,
            self._bridge("2026-06-25 07:59:00.000 | INFO | x:y:1 - working"),
            {
                "bridge_5.log": "\n".join(
                    [
                        _model_reference_stale("07:54:59.000"),
                        _sample_stage_fault("07:55:00.000", job_id=_STAGE_FAULT_JOB),
                    ],
                ),
            },
        )
        assert "stale" in findings["model_reference_sample_fault"].verdict

    def test_an_unrelated_stage_fault_does_not_fire(self, tmp_path: Path) -> None:
        """Only the model-reference signature counts; other sample-stage faults have their own causes."""
        findings = _diagnose(
            tmp_path,
            self._bridge("2026-06-25 07:59:00.000 | INFO | x:y:1 - working"),
            {
                "bridge_5.log": _sample_stage_fault(
                    "07:55:00.000",
                    job_id=_STAGE_FAULT_JOB,
                    error="model_loader (HordeCheckpointLoader): ValueError: too many values to unpack",
                ),
            },
        )
        assert "model_reference_sample_fault" not in findings


class TestWholeCardPopClaim:
    """The claim's engage/release edges: the episodes, and a cap-ended claim squeezing a mixed queue."""

    @staticmethod
    def _bridge(*lines: str) -> str:
        """A single-session bridge log opened by the startup boundary."""
        return "\n".join(
            [f"2026-06-25 07:50:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}", *lines],
        )

    def test_episodes_are_summarised_with_their_ends(self, tmp_path: Path) -> None:
        """The informational rollup names the claimed model, how long it held, and how the claim ended."""
        findings = _diagnose(
            tmp_path,
            self._bridge(
                _pop_claim_engaged("07:51:00.000"),
                _pop_claim_released("07:53:00.000", release="the horde had no further work for it"),
            ),
        )
        finding = findings["whole_card_pop_claim_episodes"]
        assert finding.severity is Severity.INFO
        assert _FLUX in finding.verdict
        assert "no further work" in finding.verdict
        assert "120s" in finding.verdict

    def test_a_claim_still_standing_at_the_end_is_reported(self, tmp_path: Path) -> None:
        """A session that ended inside a claim still says the offer was narrowed and by what."""
        findings = _diagnose(tmp_path, self._bridge(_pop_claim_engaged("07:51:00.000")))
        assert "still standing" in findings["whole_card_pop_claim_episodes"].verdict

    def test_no_claim_lines_report_nothing(self, tmp_path: Path) -> None:
        """A worker whose residencies never claim the offer produces no episode finding at all."""
        findings = _diagnose(tmp_path, self._bridge(_whole_card_reserve("07:54:50.000")))
        assert "whole_card_pop_claim_episodes" not in findings

    def test_repeated_cap_ends_with_foreign_heads_parked_is_a_warning(self, tmp_path: Path) -> None:
        """Claims run to their cap while another model's head waits: the monopoly squeezing a mixed queue."""
        findings = _diagnose(
            tmp_path,
            self._bridge(
                _pop_claim_engaged("07:51:00.000"),
                _dispatch_stall("07:52:00.000", reason=_DISPATCH_NONHEAD_REASON, model="Juggernaut XL"),
                _pop_claim_released("07:54:00.000", release="the maximum hold elapsed"),
                _pop_claim_engaged("07:55:00.000"),
                _dispatch_stall("07:56:00.000", reason=_DISPATCH_NONHEAD_REASON, model="Juggernaut XL"),
                _pop_claim_released("07:58:00.000", release="the maximum hold elapsed"),
            ),
        )
        finding = findings["whole_card_pop_claim_monopoly"]
        assert finding.severity is Severity.WARNING
        assert "Juggernaut XL" in finding.verdict
        assert finding.see_also == "whole_card_pop_claim_episodes"

    def test_cap_ends_with_nothing_else_queued_are_not_a_monopoly(self, tmp_path: Path) -> None:
        """A worker serving only the claimed model rides its cap without squeezing anything."""
        findings = _diagnose(
            tmp_path,
            self._bridge(
                _pop_claim_engaged("07:51:00.000"),
                _pop_claim_released("07:54:00.000", release="the maximum hold elapsed"),
                _pop_claim_engaged("07:55:00.000"),
                _pop_claim_released("07:58:00.000", release="the maximum hold elapsed"),
            ),
        )
        assert "whole_card_pop_claim_monopoly" not in findings

    def test_claims_that_release_themselves_are_not_a_monopoly(self, tmp_path: Path) -> None:
        """Ending on the empty-pop evidence is the claim giving the intake back on its own."""
        findings = _diagnose(
            tmp_path,
            self._bridge(
                _pop_claim_engaged("07:51:00.000"),
                _dispatch_stall("07:52:00.000", reason=_DISPATCH_NONHEAD_REASON, model="Juggernaut XL"),
                _pop_claim_released("07:54:00.000", release="the horde had no further work for it"),
                _pop_claim_engaged("07:55:00.000"),
                _dispatch_stall("07:56:00.000", reason=_DISPATCH_NONHEAD_REASON, model="Juggernaut XL"),
                _pop_claim_released("07:58:00.000", release="the residency released"),
            ),
        )
        assert "whole_card_pop_claim_monopoly" not in findings


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

    def test_post_processing_defer_is_a_warning_not_critical(self, tmp_path: Path) -> None:
        """A post-processing co-residency defer is a named gate: a parked-head warning, never the bug-critical.

        The PP-defer line names its gate, so it must not match the gate-less ``no matching gate`` bug phrase.
        It is kept as a named-gate warning (real head-parked time worth seeing), not excluded like the benign
        reconcile hold.
        """
        bridge = "\n".join(
            [
                f"2026-06-25 13:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                _dispatch_stall("13:01:00.000", reason=_DISPATCH_POST_PROCESSING_REASON),
                _dispatch_stall("13:01:40.000", reason=_DISPATCH_POST_PROCESSING_REASON),
            ],
        )
        findings = _diagnose(tmp_path, bridge)
        assert "head_dispatch_stall" in findings
        assert findings["head_dispatch_stall"].severity is Severity.WARNING

    def test_reconcile_hold_is_not_a_head_dispatch_stall(self, tmp_path: Path) -> None:
        """A residency-reconciliation hold must not masquerade as the gate-less scheduler-bug stall.

        The reconcile-hold line carries its own attribution now; it is a benign swap-churn wait, so it must
        neither trip the CRITICAL ``no matching gate`` finding nor the generic parked-head warning.
        """
        bridge = "\n".join(
            [
                f"2026-06-25 13:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                _dispatch_stall("13:01:00.000", reason=_DISPATCH_RECONCILE_REASON, parked=11),
                _dispatch_stall("13:01:40.000", reason=_DISPATCH_RECONCILE_REASON, parked=11),
            ],
        )
        assert "head_dispatch_stall" not in _diagnose(tmp_path, bridge)


class TestResidencyReconciliationHolds:
    """The dispatch residency-reconciliation gate holding a resident head while it evicts idle VRAM.

    This is a real GPU-uptime signal (swap-churn duty cost), not a scheduler bug: benign at low volume,
    worth a warning when the holds pile up (by rate or by the share of the session spent parked).
    """

    def _reconcile(self, ts: str, *, model: str = "AlbedoBase XL (SDXL)", parked: int = 11) -> str:
        return _dispatch_stall(ts, reason=_DISPATCH_RECONCILE_REASON, model=model, parked=parked)

    def test_low_volume_reconcile_holds_are_info(self, tmp_path: Path) -> None:
        """A handful of reconcile holds across a long session is a benign, informational swap-churn note."""
        bridge = "\n".join(
            [
                f"2026-06-25 13:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                self._reconcile("13:10:00.000"),
                self._reconcile("13:30:00.000", model="CyberRealistic Pony"),
                self._reconcile("13:50:00.000"),
            ],
        )
        findings = _diagnose(tmp_path, bridge)
        assert "residency_reconciliation_holds" in findings
        assert findings["residency_reconciliation_holds"].severity is Severity.INFO
        verdict = findings["residency_reconciliation_holds"].verdict
        assert "AlbedoBase XL (SDXL)" in verdict
        assert "CyberRealistic Pony" in verdict

    def test_high_rate_reconcile_holds_are_warning(self, tmp_path: Path) -> None:
        """Many reconcile holds per hour is a throughput-shaping swap-churn cost worth a warning."""
        lines = [f"2026-06-25 14:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}"]
        # 40 holds inside a ~20 minute window is ~120/hour, well past the rate threshold.
        for i in range(40):
            minute, second = divmod(i * 30, 60)
            lines.append(self._reconcile(f"14:{minute:02d}:{second:02d}.000"))
        findings = _diagnose(tmp_path, "\n".join(lines))
        assert "residency_reconciliation_holds" in findings
        assert findings["residency_reconciliation_holds"].severity is Severity.WARNING

    def test_high_parked_share_reconcile_holds_are_warning(self, tmp_path: Path) -> None:
        """Few holds, but each parked long enough that they dominate a share of the session, is a warning."""
        bridge = "\n".join(
            [
                f"2026-06-25 15:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                self._reconcile("15:02:00.000", parked=90),
                self._reconcile("15:04:00.000", parked=90),
                self._reconcile("15:06:00.000", parked=90),
                self._reconcile("15:08:00.000", parked=90),
                self._reconcile("15:10:00.000", parked=90),
            ],
        )
        findings = _diagnose(tmp_path, bridge)
        assert "residency_reconciliation_holds" in findings
        assert findings["residency_reconciliation_holds"].severity is Severity.WARNING

    def test_silent_without_reconcile_holds(self, tmp_path: Path) -> None:
        """A session with no reconcile-hold stall lines produces no finding."""
        bridge = "\n".join(
            [
                f"2026-06-25 13:00:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                _dispatch_stall("13:01:00.000", reason=_DISPATCH_GATE_REASON),
            ],
        )
        assert "residency_reconciliation_holds" not in _diagnose(tmp_path, bridge)


class TestPostProcessingVramStall:
    """A slot reaped silent in post-processing, plus the breaker advisory that escalates the session.

    The watchdog reap line is the base signal (a single one is an actionable warning); the breaker-trip
    advisory and any forced maintenance escalate it to critical. The detector must also fire on a
    breaker-only session, since the planner's unhostable-peak faults can trip the breaker with no watchdog
    stall line of their own.

    Bare co-occurrence of dedicated post-processing activity and child low-free-VRAM readings is no longer
    a warning on its own: the scheduler now admits sampling/post-processing co-residency when measured
    device truth affords it, so a tight-but-healthy overlap is an admitted operating mode. Such a case
    downgrades to an informational, audit-only finding unless a corroborating signal is present: a
    post-processing watchdog reap, WDDM demand-paging in the window, or a reading below the inference
    reserve (not merely low). Any of those keeps the warning semantics and text.
    """

    def _bridge(self, *lines: str) -> str:
        return "\n".join(
            [f"2026-06-28 16:53:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}", *lines],
        )

    @staticmethod
    def _stall(ts: str, *, slot: int = 3) -> str:
        """process_lifecycle._check_and_replace_process: the post-processing-stage watchdog reaping a slot."""
        return (
            f"2026-06-28 {ts} | ERROR    | horde_worker_regen.process_management.lifecycle.process_lifecycle:_check_and_replace_process:1618 - "
            f"HordeProcessInfo(process_id={slot}, last_process_state=HordeProcessState.INFERENCE_POST_PROCESSING, "
            f"loaded_horde_model_name=AAM XL AnimeMix) seems to be stuck post processing, replacing it"
        )

    @staticmethod
    def _breaker(ts: str) -> str:
        """process_manager._apply_post_processing_fault_breaker: the breaker tripping (operator advisory)."""
        return (
            f"2026-06-28 {ts} | WARNING | horde_worker_regen.process_management.process_manager:_apply_post_processing_fault_breaker:1090 - "
            "Post-processing fault breaker tripped: 5 post-processing over-commit fault(s) in the last 1800s "
            "(threshold 4). Disabling post-processing on this worker for the rest of the session."
        )

    def test_lone_stall_is_warning(self, tmp_path: Path) -> None:
        """A single watchdog reap with no escalation is an actionable warning."""
        findings = _diagnose(tmp_path, self._bridge(self._stall("16:53:42.000")))
        assert "post_processing_vram_stall" in findings
        assert findings["post_processing_vram_stall"].severity is Severity.WARNING

    def test_breaker_trip_escalates_to_critical(self, tmp_path: Path) -> None:
        """The breaker advisory escalates the stall finding to critical and is named in the verdict."""
        findings = _diagnose(tmp_path, self._bridge(self._stall("16:53:42.000"), self._breaker("16:55:00.000")))
        finding = findings["post_processing_vram_stall"]
        assert finding.severity is Severity.CRITICAL
        assert "breaker tripped" in finding.verdict

    def test_breaker_only_still_fires(self, tmp_path: Path) -> None:
        """The detector fires on a breaker-only session (the planner-fault path leaves no stall line)."""
        findings = _diagnose(tmp_path, self._bridge(self._breaker("16:55:00.000")))
        assert "post_processing_vram_stall" in findings
        assert findings["post_processing_vram_stall"].severity is Severity.CRITICAL

    def test_uncorroborated_co_residency_is_info(self, tmp_path: Path) -> None:
        """Dedicated post-processing plus a low readout, with nothing corroborating, is admitted co-residency.

        This is the live false-positive: post-processing active, a low child free-VRAM reading, but no
        watchdog reap, no WDDM paging, and nothing below the inference reserve. The scheduler admits this
        overlap deliberately, so the finding must downgrade to an informational, audit-only note rather than
        a warning, and its text must name it as admitted co-residency.
        """
        child = "\n".join(
            [
                "2026-06-28 16:53:12.000 | INFO | horde_worker_regen.process_management.workers.post_process_process:_run_post_processing:190 - Post-processing job 9bccbf84",
                "2026-06-28 16:53:14.000 | WARNING | comfy.model_management:free_memory:1110 - Free VRAM: 316 MB (+5272 MB reclaimable torch cache; comfy sees 5588)",
            ],
        )
        bridge = self._bridge("2026-06-28 16:54:00.000 | INFO | x:y:1 - Session still active")
        findings = _diagnose(tmp_path, bridge, {"bridge_1.log": child})
        finding = findings["post_processing_vram_stall"]
        assert finding.severity is Severity.INFO
        assert "admitted co-residency" in finding.verdict
        assert "low-free-VRAM" in finding.verdict

    def test_wddm_paging_corroborates_warning(self, tmp_path: Path) -> None:
        """A WDDM demand-paging verdict in the window corroborates the overlap and keeps the warning."""
        child = "\n".join(
            [
                "2026-06-28 16:53:12.000 | INFO | horde_worker_regen.process_management.workers.post_process_process:_run_post_processing:190 - Post-processing job 9bccbf84",
                "2026-06-28 16:53:14.000 | WARNING | comfy.model_management:free_memory:1110 - Free VRAM: 316 MB (+5272 MB reclaimable torch cache; comfy sees 5588)",
            ],
        )
        bridge = self._bridge(self._wddm_paging("16:53:20.000"))
        finding = _diagnose(tmp_path, bridge, {"bridge_1.log": child})["post_processing_vram_stall"]
        assert finding.severity is Severity.WARNING

    def test_no_finding_without_post_processing_activity(self, tmp_path: Path) -> None:
        """Low child free-VRAM readings without any dedicated post-processing activity produce no finding."""
        child = "\n".join(
            [
                self._free_vram_readout("16:53:13.000", free_mb=316),
                self._free_vram_readout("16:53:14.000", free_mb=800),
            ],
        )
        bridge = self._bridge("2026-06-28 16:54:00.000 | INFO | x:y:1 - Session still active")
        assert "post_processing_vram_stall" not in _diagnose(tmp_path, bridge, {"bridge_1.log": child})

    def test_empty_generation_field_is_not_dedicated_post_processing_activity(self, tmp_path: Path) -> None:
        """An inference request with an empty post-processing list does not prove downstream activity."""
        child = "\n".join(
            [
                "2026-06-28 16:53:12.000 | DEBUG | inference_process:start_inference:1048 - "
                "{'width': 1024, 'height': 1024, 'post_processing': []}",
                self._reserve_warning("16:53:15.000"),
            ],
        )
        bridge = self._bridge(
            "2026-06-28 16:53:10.000 | INFO | x:y:1 - allow_post_processing: False | dedicated_post_processing: off",
            "2026-06-28 16:54:00.000 | INFO | x:y:1 - Session still active",
        )

        assert "post_processing_vram_stall" not in _diagnose(tmp_path, bridge, {"bridge_1.log": child})

    @staticmethod
    def _free_vram_readout(ts: str, *, free_mb: int) -> str:
        """The routine device-wide free-VRAM readout hordelib emits on every log_free_ram call (DEBUG)."""
        return (
            f"2026-06-28 {ts} | DEBUG | hordelib.comfy_horde:log_free_ram:600 - "
            f"Free VRAM: {free_mb} MB (+5272 MB reclaimable torch cache; comfy sees {free_mb + 5272}), "
            "Free RAM: 40000 MB"
        )

    @staticmethod
    def _wddm_paging(ts: str, *, pid: int = 1234, shared_mb: int = 640) -> str:
        """inference_scheduler.note_wddm_paging: the parent's WDDM demand-paging verdict (bridge WARNING)."""
        return (
            f"2026-06-28 {ts} | WARNING | horde_worker_regen.process_management.scheduling.inference_scheduler:note_wddm_paging:6964 - "
            f"WDDM demand-paging detected on worker processes (pid {pid}: {shared_mb}MB shared); the driver "
            "demoted their VRAM allocations to system memory. Denying model retention and reclaiming idle "
            "resident VRAM (newest idle resident first)."
        )

    @staticmethod
    def _reserve_warning(ts: str, *, free_mb: int = 316, reserve_mb: int = 2048) -> str:
        """The throttled, genuinely-alarming below-inference-reserve streaming warning (WARNING)."""
        return (
            f"2026-06-28 {ts} | WARNING | hordelib.comfy_horde:log_free_ram:624 - "
            f"Free VRAM {free_mb} MB is below the {reserve_mb} MB inference reserve: sampling activations will "
            "stream from host RAM and run several times slower (ComfyUI reports no offload, so the GPU driver's "
            "system-memory fallback is the likely cause)."
        )

    def test_benign_high_vram_readouts_do_not_count(self, tmp_path: Path) -> None:
        """Routine ~9.8GB-free readouts must not be counted as low-VRAM warnings (the miscount bug).

        Every log_free_ram readout carries the ``reclaimable torch cache`` note, so counting readouts flagged
        the whole session; only readings that are genuinely low should count. With dedicated activity but no
        genuinely-low reading, the overlap detector must stay silent.
        """
        child = "\n".join(
            [
                "2026-06-28 16:53:12.000 | INFO | horde_worker_regen.process_management.workers.post_process_process:_run_post_processing:190 - Post-processing job 9bccbf84",
                self._free_vram_readout("16:53:13.000", free_mb=9800),
                self._free_vram_readout("16:53:14.000", free_mb=9820),
                self._free_vram_readout("16:53:15.000", free_mb=9790),
                self._free_vram_readout("16:53:16.000", free_mb=9805),
            ],
        )
        bridge = self._bridge("2026-06-28 16:54:00.000 | INFO | x:y:1 - Session still active")
        assert "post_processing_vram_stall" not in _diagnose(tmp_path, bridge, {"bridge_1.log": child})

    def test_low_vram_dips_counted_not_readouts(self, tmp_path: Path) -> None:
        """The summary reports the count of genuinely-low readings, not the flood of routine readouts.

        The low readings here are merely-low readouts (above the inference reserve), so nothing corroborates
        a stall and the finding is the audit-only informational note; the count must still be surfaced.
        """
        child = "\n".join(
            [
                "2026-06-28 16:53:12.000 | INFO | horde_worker_regen.process_management.workers.post_process_process:_run_post_processing:190 - Post-processing job 9bccbf84",
                self._free_vram_readout("16:53:13.000", free_mb=9800),
                self._free_vram_readout("16:53:14.000", free_mb=9820),
                self._free_vram_readout("16:53:15.000", free_mb=9790),
                self._free_vram_readout("16:53:16.000", free_mb=316),
                self._free_vram_readout("16:53:17.000", free_mb=800),
            ],
        )
        bridge = self._bridge("2026-06-28 16:54:00.000 | INFO | x:y:1 - Session still active")
        finding = _diagnose(tmp_path, bridge, {"bridge_1.log": child})["post_processing_vram_stall"]
        assert finding.severity is Severity.INFO
        assert "2 child low-free-VRAM" in finding.verdict

    def test_reserve_warning_is_alarming_and_counts(self, tmp_path: Path) -> None:
        """The below-inference-reserve streaming warning corroborates the overlap and keeps the warning."""
        child = "\n".join(
            [
                "2026-06-28 16:53:12.000 | INFO | horde_worker_regen.process_management.workers.post_process_process:_run_post_processing:190 - Post-processing job 9bccbf84",
                self._free_vram_readout("16:53:13.000", free_mb=9800),
                self._free_vram_readout("16:53:14.000", free_mb=9820),
                self._reserve_warning("16:53:15.000"),
            ],
        )
        bridge = self._bridge("2026-06-28 16:54:00.000 | INFO | x:y:1 - Session still active")
        finding = _diagnose(tmp_path, bridge, {"bridge_1.log": child})["post_processing_vram_stall"]
        assert finding.severity is Severity.WARNING
        assert "1 child low-free-VRAM" in finding.verdict

    def test_silent_without_signals(self, tmp_path: Path) -> None:
        """A crash-on-start recovery is not a post-processing stall, so the detector stays silent."""
        bridge = self._bridge(_recovery("16:53:31.000", 1, reason="inference process replaced (crashed or hung)"))
        assert "post_processing_vram_stall" not in _diagnose(tmp_path, bridge)


class TestPostProcessingDeferralStarvation:
    """The admission gate deferring the same job in a loop, with and without lane completions."""

    _JOB = "4e17ddbd-a9cc-494d-b668-8f6fcb6d08aa"

    def _defer(self, ts: str, job_id: str) -> str:
        return (
            f"2026-06-28 {ts} | WARNING | horde_worker_regen.process_management.workers."
            f"post_process_orchestrator:_has_post_processing_headroom:102 - Deferring post-processing for "
            f"job {job_id}: estimated peak 8533 MB plus reserve 2048 MB exceeds free VRAM after commitments "
            "(6675 MB available on card 0). No idle VRAM reclaim was available."
        )

    def _finished(self, ts: str, job_id: str) -> str:
        return (
            f"2026-06-28 {ts} | INFO | horde_worker_regen.process_management.ipc."
            f"message_dispatcher:_handle_post_process_result:803 - Post-processing finished for job "
            f"{job_id[:8]} in 2.12 seconds on process 1."
        )

    def _bridge(self, *lines: str) -> str:
        return "\n".join(
            [f"2026-06-28 16:50:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}", *lines],
        )

    def _defer_storm(self, count: int, *, start_minute: int = 53) -> list[str]:
        return [self._defer(f"16:{start_minute + i // 60:02d}:{i % 60:02d}.000", self._JOB) for i in range(count)]

    def test_starved_lane_is_critical(self, tmp_path: Path) -> None:
        """A long same-job deferral storm with zero completions afterwards is the critical starvation."""
        findings = _diagnose(tmp_path, self._bridge(*self._defer_storm(40)))
        finding = findings["post_processing_deferral_starvation"]
        assert finding.severity is Severity.CRITICAL
        assert self._JOB in finding.verdict
        assert "starved the entire lane" in finding.verdict

    def test_storm_with_completions_is_a_warning(self, tmp_path: Path) -> None:
        """A deferral storm while other jobs still complete is head starvation, reported as a warning."""
        lines = self._defer_storm(40) + [self._finished("16:56:30.000", "1bef2bda-0000-0000-0000-000000000000")]
        findings = _diagnose(tmp_path, self._bridge(*lines))
        finding = findings["post_processing_deferral_starvation"]
        assert finding.severity is Severity.WARNING

    def test_transient_deferral_does_not_fire(self, tmp_path: Path) -> None:
        """A handful of deferrals (healthy backpressure across a VRAM spike) produces no finding."""
        lines = self._defer_storm(4) + [self._finished("16:53:30.000", self._JOB)]
        findings = _diagnose(tmp_path, self._bridge(*lines))
        assert "post_processing_deferral_starvation" not in findings


class TestSessionWindowBisect:
    """The binary-search session-window slice must equal the naive per-record filter for the same input."""

    @staticmethod
    def _record(timestamp: datetime | None) -> LogRecord:
        return LogRecord(
            timestamp=timestamp,
            level="INFO",
            name="child",
            function="loop",
            lineno=1,
            message="m",
            source_path=Path("bridge_0.log"),
            raw_lineno=1,
        )

    @staticmethod
    def _naive(records: list[LogRecord], start: datetime | None, end: datetime | None) -> list[LogRecord]:
        """The pre-bisect filter: exclude timestamp-less records and clamp inclusively to the open-ended window."""
        kept: list[LogRecord] = []
        for record in records:
            if record.timestamp is None:
                continue
            if start is not None and record.timestamp < start:
                continue
            if end is not None and record.timestamp > end:
                continue
            kept.append(record)
        return kept

    def test_bisect_matches_naive_across_windows(self) -> None:
        """Over a multi-session set with None-timestamp and on-boundary records, both slices agree exactly."""
        base = datetime(2026, 6, 24, 18, 0, 0)
        later = datetime(2026, 6, 24, 19, 0, 0)  # a second session an hour on
        offsets = [0, 5, 10, 10, 15, 20]  # a duplicate exactly on a boundary candidate
        # Ordered as read_records yields: timestamp-less records first, then ascending timestamps.
        records = [self._record(None), self._record(None)]
        records += [self._record(base + timedelta(seconds=s)) for s in offsets]
        records += [self._record(later + timedelta(seconds=s)) for s in (0, 5, 10)]

        windows: list[tuple[datetime | None, datetime | None]] = [
            (None, None),
            (base + timedelta(seconds=5), base + timedelta(seconds=15)),  # inclusive both ends, spans the dup
            (None, base + timedelta(seconds=10)),  # open start
            (later, None),  # open end into the second session
            (base + timedelta(seconds=7), base + timedelta(seconds=12)),  # straddles the duplicate boundary
            (base, base),  # a single instant, exactly on a record
            (base - timedelta(seconds=1), base - timedelta(seconds=1)),  # empty window before any record
        ]
        for start, end in windows:
            assert _records_in_window(records, start, end) == self._naive(records, start, end), (start, end)


def _empty_model_pop(ts: str, *, job_id: str = "ac0df470-17fd-4916-9ae2-56bb3f998caf") -> str:
    """job_popper.api_job_pop: a pop whose model field arrived empty (the older, uncontained form)."""
    return (
        f"2026-06-25 {ts} | INFO     | horde_worker_regen.process_management.jobs.job_popper:api_job_pop:2122 - "
        f"Popped job {job_id} (14 eMPS) (model: , batch: 1, loras: False, post_processing: False)"
    )


def _blank_preload(ts: str, *, slot: int = 4) -> str:
    """inference_scheduler._send_preload: the blank identity being sent to a slot as if it were a model."""
    return (
        f"2026-06-25 {ts} | DEBUG    | horde_worker_regen.process_management.scheduling.inference_scheduler:_send_preload:5713 - "
        f"Preloading model  on process {slot}"
    )


def _load_failure_recovery(ts: str, *, slot: int = 4, model: str = "") -> str:
    """process_lifecycle: a slot replaced because the model it was told to load ended the process."""
    return _recovery(
        ts,
        slot,
        reason=f"inference process replaced (failed to load model {model})",
        last_state="PROCESS_ENDED",
    ).replace("2026-06-24", "2026-06-25")


def _blank_model_quarantine(ts: str) -> str:
    """process_lifecycle.record_model_incident: the blank identity crossing the quarantine threshold."""
    return (
        f"2026-06-25 {ts} | ERROR    | horde_worker_regen.process_management.lifecycle.process_lifecycle:record_model_incident:3318 - "
        "Model  caused 3 load_failure incident(s) within 600s; quarantining it (its jobs will be reissued, it "
        "will not be preloaded, and it will not be advertised in pops) to stop it churning the inference pool."
    )


def _blank_quarantine_skip(ts: str) -> str:
    """inference_scheduler._attempt_preload_for_job: a job refused because the blank identity is quarantined."""
    return (
        f"2026-06-25 {ts} | WARNING  | horde_worker_regen.process_management.scheduling.inference_scheduler:_attempt_preload_for_job:7981 - "
        "Skipping preload of quarantined model ; faulting its job for reissue."
    )


def _malformed_pop_rejected(ts: str, *, job_id: str = "ac0df470-17fd-4916-9ae2-56bb3f998caf") -> str:
    """job_popper._reject_malformed_pop: the contained form, handing a model-less pop straight back."""
    return (
        f"2026-06-25 {ts} | ERROR    | horde_worker_regen.process_management.jobs.job_popper:_reject_malformed_pop:1693 - "
        f"Popped job {job_id} carries no model name (got ''); returning it to the horde for reissue without "
        "queueing it. This is a malformed pop response, not a model failure."
    )


def _blank_preload_refused(ts: str, *, job_id: str = "ac0df470-17fd-4916-9ae2-56bb3f998caf") -> str:
    """inference_process._preload_model: the child refusing a blank name instead of ending itself."""
    return (
        f"2026-06-25 {ts} | ERROR    | horde_worker_regen.process_management.workers.inference_process:_preload_model:705 - "
        f"Refusing to preload a blank model name (got '') for job {job_id}; reporting a load failure and "
        "staying available."
    )


_ACCOUNT_LIMIT_POP_ERROR = "To avoid abuse, untrusted users can only have up to 3 distinct workers."


def _pop_api_error(ts: str, *, message: str = _ACCOUNT_LIMIT_POP_ERROR, code: str = "TooManyWorkers") -> str:
    """job_popper._handle_pop_error_response: the horde rejecting a pop, quoting its own message."""
    return (
        f"2026-06-25 {ts} | ERROR    | horde_worker_regen.process_management.jobs.job_popper:_handle_pop_error_response:1593 - "
        f"Failed to pop job (API Error): message='{message}' object_data=None rc='{code}'"
    )


def _safety_placement_cycle(ts: str, *, owner: str = "Reclaim ladder", restoring: bool = False) -> str:
    """process_lifecycle.pause_safety_on_gpu / restore_safety_on_gpu: the parent moving safety off/on the GPU."""
    action = (
        "restoring the safety process to the GPU."
        if restoring
        else "moving the safety process off-GPU to free its VRAM context."
    )
    function = "restore_safety_on_gpu:2913" if restoring else "pause_safety_on_gpu:2878"
    return (
        f"2026-06-25 {ts} | INFO     | horde_worker_regen.process_management.lifecycle.process_lifecycle:{function} - "
        f"{owner}: {action}"
    )


def _retired_safety_result(ts: str, *, slot: int = 0, launch: int = 13) -> str:
    """message_dispatcher._classify_retired_launch_message: a verdict dropped because its launch was retired."""
    return (
        f"2026-06-25 {ts} | WARNING  | horde_worker_regen.process_management.ipc.message_dispatcher:_classify_retired_launch_message:813 - "
        f"Ignoring result message from retired safety process {slot} launch {launch} "
        "(safety process replacement): HordeSafetyResultMessage"
    )


_GIT_ENV_TRACEBACK = (
    "2026-06-25 18:29:26.000 | CRITICAL | inference_5:startup - worker child crashed before its log was ready:\n"
    "Traceback (most recent call last):\n"
    '  File "installer.py", line 46, in _run_git\n'
    "    raise GitCommandError(message)\n"
    "hordelib.installation.installer.GitCommandError: git clone https://example.invalid/ComfyQR ComfyQR "
    "failed in /data/comfyui_env/ComfyUI/custom_nodes: Cloning into 'ComfyQR'...\n"
    "error: Untracked working tree file '.github/workflows/publish.yml' would be overwritten by merge.\n"
    "fatal: unable to checkout working tree\n"
    "warning: Clone succeeded, but checkout failed.\n"
)
"""A startup crash whose cause is a contended shared environment directory, not a broken torch install."""


class TestEmptyModelPopCascade:
    """A pop carrying no model name, and what each worker generation does with it."""

    @staticmethod
    def _bridge(*lines: str) -> str:
        return "\n".join(
            [f"2026-06-25 07:50:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}", *lines],
        )

    def test_the_uncontained_cascade_is_reported(self, tmp_path: Path) -> None:
        """On an older capture the blank identity reaches preload, kills slots, and poisons the quarantine."""
        findings = _diagnose(
            tmp_path,
            self._bridge(
                _empty_model_pop("07:51:00.000"),
                _empty_model_pop("07:51:10.000"),
                _blank_preload("07:51:20.000"),
                _load_failure_recovery("07:51:25.000", slot=4),
                _blank_preload("07:51:30.000", slot=5),
                _load_failure_recovery("07:51:35.000", slot=5),
                _blank_model_quarantine("07:51:40.000"),
                _blank_quarantine_skip("07:52:00.000"),
            ),
        )
        finding = findings["empty_model_pop_cascade"]
        assert finding.severity is Severity.CRITICAL
        assert "2 pop(s)" in finding.verdict
        assert "2 child death(s)" in finding.verdict
        assert "quarantin" in finding.verdict
        assert "upgrade" in finding.remediation.lower()

    def test_the_contained_form_is_reported_with_its_rate(self, tmp_path: Path) -> None:
        """On a newer capture the same input is rejected at the boundary, and only the rate matters."""
        findings = _diagnose(
            tmp_path,
            self._bridge(
                _malformed_pop_rejected("07:51:00.000"),
                _malformed_pop_rejected("07:51:10.000"),
                _blank_preload_refused("07:51:20.000"),
            ),
        )
        finding = findings["empty_model_pop_cascade"]
        assert finding.severity is Severity.WARNING
        assert "contained" in finding.verdict
        assert "2 malformed pop(s)" in finding.verdict
        assert "upgrade" not in finding.remediation.lower()

    def test_a_clean_session_does_not_fire(self, tmp_path: Path) -> None:
        """No blank identity anywhere means no finding."""
        assert "empty_model_pop_cascade" not in _diagnose(tmp_path, self._bridge())


class TestPreloadKillsChildLoop:
    """A named model that ends the slot it is preloaded onto, over and over."""

    @staticmethod
    def _bridge(*lines: str) -> str:
        return "\n".join(
            [f"2026-06-25 07:50:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}", *lines],
        )

    def test_repeated_deaths_on_one_model_fire(self, tmp_path: Path) -> None:
        """Three slot replacements attributed to loading the same model is a loop, whatever the model is."""
        findings = _diagnose(
            tmp_path,
            self._bridge(
                _load_failure_recovery("07:51:00.000", slot=3, model="AlbedoBase XL (SDXL)"),
                _load_failure_recovery("07:52:00.000", slot=4, model="AlbedoBase XL (SDXL)"),
                _load_failure_recovery("07:53:00.000", slot=5, model="AlbedoBase XL (SDXL)"),
            ),
        )
        finding = findings["preload_kills_child_loop"]
        assert "AlbedoBase XL (SDXL)" in finding.verdict
        assert "3" in finding.verdict

    def test_deaths_spread_across_models_do_not_fire(self, tmp_path: Path) -> None:
        """One death each for three models is pool churn, not a poisoned checkpoint."""
        findings = _diagnose(
            tmp_path,
            self._bridge(
                _load_failure_recovery("07:51:00.000", slot=3, model="A"),
                _load_failure_recovery("07:52:00.000", slot=4, model="B"),
                _load_failure_recovery("07:53:00.000", slot=5, model="C"),
            ),
        )
        assert "preload_kills_child_loop" not in findings

    def test_the_blank_identity_is_left_to_its_own_detector(self, tmp_path: Path) -> None:
        """A blank name is the malformed-pop cascade, which says more; it must not be reported twice."""
        findings = _diagnose(
            tmp_path,
            self._bridge(
                _empty_model_pop("07:50:30.000"),
                _load_failure_recovery("07:51:00.000", slot=3),
                _load_failure_recovery("07:52:00.000", slot=4),
                _load_failure_recovery("07:53:00.000", slot=5),
            ),
        )
        assert "preload_kills_child_loop" not in findings
        assert "empty_model_pop_cascade" in findings


class TestPopApiErrorDominance:
    """One pop rejection repeating for a long stretch: the horde's own words are the diagnosis."""

    @staticmethod
    def _bridge(*lines: str) -> str:
        return "\n".join(
            [f"2026-06-25 07:50:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}", *lines],
        )

    @staticmethod
    def _storm(message: str | None = None, *, code: str = "TooManyWorkers", count: int = 12) -> list[str]:
        """A run of identical pop rejections spread over several minutes."""
        kwargs = {"code": code} if message is None else {"message": message, "code": code}
        return [_pop_api_error(f"07:5{1 + i // 6}:{(i * 10) % 60:02d}.000", **kwargs) for i in range(count)]

    def test_a_dominant_message_is_quoted_verbatim(self, tmp_path: Path) -> None:
        """The finding names the message the horde sent, its count, and how long it persisted."""
        findings = _diagnose(tmp_path, self._bridge(*self._storm()))
        finding = findings["pop_api_error_dominance"]
        assert _ACCOUNT_LIMIT_POP_ERROR in finding.verdict
        assert "12" in finding.verdict

    def test_an_operator_fixable_message_says_so(self, tmp_path: Path) -> None:
        """An account-limit rejection will not clear on its own, so the remediation must not say to wait."""
        findings = _diagnose(tmp_path, self._bridge(*self._storm()))
        assert "will not clear on its own" in findings["pop_api_error_dominance"].remediation

    def test_a_transient_message_is_keyed_differently(self, tmp_path: Path) -> None:
        """A server-side error is expected to clear, so it gets the transient remediation."""
        findings = _diagnose(
            tmp_path,
            self._bridge(*self._storm("Internal Server Error", code="ServerError")),
        )
        remediation = findings["pop_api_error_dominance"].remediation
        assert "will not clear on its own" not in remediation
        assert "transient" in remediation

    def test_a_handful_of_errors_does_not_fire(self, tmp_path: Path) -> None:
        """Occasional pop errors are ordinary API noise."""
        assert "pop_api_error_dominance" not in _diagnose(tmp_path, self._bridge(_pop_api_error("07:51:00.000")))


class TestSafetyStallActuatorNarration:
    """When the log says who cycled safety, the finding must say so instead of listing candidates."""

    @staticmethod
    def _bridge(*lines: str) -> str:
        return "\n".join(
            [f"2026-06-25 07:50:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}", *lines],
        )

    def test_a_parent_ordered_cycle_is_asserted(self, tmp_path: Path) -> None:
        """The pause/restore pair plus the dropped verdicts are the chain, so they are stated as the cause."""
        findings = _diagnose(
            tmp_path,
            self._bridge(
                _safety_placement_cycle("07:51:00.000"),
                _retired_safety_result("07:51:30.000"),
                _safety_unrecoverable("07:52:00.000"),
                _safety_placement_cycle("07:52:30.000", restoring=True),
            ),
        )
        finding = findings["safety_stage_stall"]
        assert "Reclaim ladder" in finding.verdict
        assert "retired" in finding.verdict
        assert "check the bridge_safety_*.log for crashes" not in finding.remediation
        assert any("off-GPU" in line for line in finding.evidence)

    def test_without_the_markers_the_candidate_list_remains(self, tmp_path: Path) -> None:
        """With no actuator evidence the finding must not invent one; the candidate causes stay."""
        findings = _diagnose(tmp_path, self._bridge(_safety_unrecoverable("07:52:00.000")))
        finding = findings["safety_stage_stall"]
        assert "Reclaim ladder" not in finding.verdict
        assert "bridge_safety_*.log" in finding.remediation


class TestCrashOnStartGuidance:
    """The remediation must follow the failure text, not offer a fixed environment fix."""

    @staticmethod
    def _bridge() -> str:
        return "\n".join(
            [
                f"2026-06-25 18:29:20.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                "2026-06-25 18:29:31.000 | ERROR    | horde_worker_regen.process_management.lifecycle.process_lifecycle:_log_recovery_diagnostics:367 - "
                "Recovery diagnostics for process 5 (os_pid=1005, launch=5): reason='inference process replaced "
                "(crashed or hung)'; last_state=PROCESS_STARTING; exitcode=1; last_heartbeat_type=OTHER; "
                "since_last_heartbeat=8.0s; since_last_message=8.0s; last_job=None; recent_actions=[]",
                "2026-06-25 18:29:40.000 | ERROR    | horde_worker_regen.process_management.lifecycle.process_lifecycle:_log_recovery_diagnostics:367 - "
                "Recovery diagnostics for process 5 (os_pid=1005, launch=6): reason='inference process replaced "
                "(crashed or hung)'; last_state=PROCESS_STARTING; exitcode=1; last_heartbeat_type=OTHER; "
                "since_last_heartbeat=8.0s; since_last_message=8.0s; last_job=None; recent_actions=[]",
            ],
        )

    def test_a_git_failure_points_at_the_shared_environment(self, tmp_path: Path) -> None:
        """A clone/checkout failure is a contended shared ComfyUI environment, not a torch problem."""
        findings = _diagnose(tmp_path, self._bridge(), {"bridge_inference_5_startup.log": _GIT_ENV_TRACEBACK})
        remediation = findings["crash_on_start_loop"].remediation
        assert "environment directory" in remediation
        assert "torch" not in remediation.lower()

    def test_an_import_failure_keeps_the_torch_guidance(self, tmp_path: Path) -> None:
        """A CUDA import failure is exactly what the torch/CUDA reinstall advice is for."""
        bridge = "\n".join(
            [
                f"2026-06-24 18:29:20.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}",
                _recovery("18:29:31.000", 1, reason="inference process replaced (crashed or hung)"),
                _recovery("18:29:40.000", 1, reason="inference process replaced (crashed or hung)"),
            ],
        )
        findings = _diagnose(tmp_path, bridge, {"bridge_inference_1_startup.log": _TRACEBACK})
        assert "torch" in findings["crash_on_start_loop"].remediation.lower()


class TestDedicatedPostProcessMarkers:
    """The dedicated-lane activity markers, as the parent actually relays them."""

    _JOB = "a8c1023b-2420-4bc2-bd36-2dd0adc45826"

    @staticmethod
    def _bridge(*lines: str) -> str:
        return "\n".join(
            [f"2026-06-25 07:50:00.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}", *lines],
        )

    def _relayed_state_change(self, ts: str) -> str:
        """message_dispatcher: the lane's stage transition, relayed with the dispatcher's own prefix."""
        return (
            f"2026-06-25 {ts} | DEBUG    | horde_worker_regen.process_management.ipc.message_dispatcher:_dispatch_buffered_message:590 - "
            f"Received HordeProcessStateChangeMessage from process 1: Post-processing job {self._JOB}"
        )

    @staticmethod
    def _below_reserve(ts: str) -> str:
        """hordelib.comfy_horde.log_free_ram: the throttled below-inference-reserve streaming warning."""
        return (
            f"2026-06-25 {ts} | WARNING  | hordelib.comfy_horde:log_free_ram:655 - "
            "Free VRAM 11 MB is below the 1219 MB inference reserve: sampling activations will stream from "
            "host RAM and run several times slower (ComfyUI reports no offload, so the GPU driver's "
            "system-memory fallback is the likely cause)."
        )

    def test_a_relayed_stage_marker_counts_as_lane_activity(self, tmp_path: Path) -> None:
        """The parent prefixes the relayed message, so anchoring at the start of it saw no lane at all."""
        findings = _diagnose(
            tmp_path,
            self._bridge(
                self._relayed_state_change("07:51:00.000"),
                "2026-06-25 07:51:30.000 | INFO | x:y:1 - still working",
            ),
            {"bridge_1.log": self._below_reserve("07:51:05.000")},
            # The child reading has to fall inside the session, which ends at the last parent record.
        )
        assert "post_processing_vram_stall" in findings

    def test_prose_mentioning_post_processing_is_not_lane_activity(self, tmp_path: Path) -> None:
        """RAM accounting and download lines name post-processing without any lane running."""
        findings = _diagnose(
            tmp_path,
            self._bridge(
                "2026-06-25 07:51:00.000 | INFO | x:y:1 -   Worker: 6.1 GB total (inference 4.0 GB | "
                "post-processing 1.2 GB | safety 0.6 GB | orchestrator 300 MB)",
                "2026-06-25 07:51:02.000 | INFO | x:y:1 -   Now: GFPGAN [post-processing (GFPGAN)] -> /models/gfpgan",
                "2026-06-25 07:51:30.000 | INFO | x:y:1 - still working",
            ),
            {"bridge_1.log": self._below_reserve("07:51:05.000")},
            # The child reading has to fall inside the session, which ends at the last parent record.
        )
        assert "post_processing_vram_stall" not in findings
