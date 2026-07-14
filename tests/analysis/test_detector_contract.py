"""The logging<->detector contract: every detector must fire on a representative real log signature.

The fragile seam in the triage subsystem is between a detector's regex (in ``detectors.py``) and the
f-string the worker actually logs (scattered across ``process_management/``). Nothing links them, so a
reworded log line can silently retire a detector. This test pins that seam: each detector is paired
with a *golden* log line that mirrors the real emit, and the test asserts the detector fires on it.

It also guards against omission. The no-orphan test fails if a detector is added to
:data:`~horde_worker_regen.analysis.detectors.DETECTORS` without a fixture here, so a new incident
class cannot ship without a representative log signature on record. And the id-convention test pins the
``detect_X`` -> finding-id ``X`` mapping the TUI and CLI both rely on.

The golden lines reuse the helpers in :mod:`tests.analysis.test_detectors` wherever one exists (so the
log format lives in one place), and define the few missing ones here, each annotated with the worker
source that emits it. When a detector stops firing because the worker reworded its log, this is the
test that goes red and names which one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from horde_worker_regen.analysis.detectors import DETECTORS, Detector, Severity
from tests.analysis.test_detectors import (
    _DISPATCH_BUG_REASON,
    _DISPATCH_NONHEAD_REASON,
    _DISPATCH_WHOLE_CARD_REASON,
    _STARTUP,
    _TRACEBACK,
    _consecutive_pause,
    _diagnose,
    _dispatch_stall,
    _force_admit,
    _give_up,
    _maintenance_pop,
    _recovery,
    _safety_lost_result,
    _safety_requeue,
    _server_slow_abort,
    _soft_reset,
    _whole_card_reserve,
)
from tests.analysis.test_unsatisfiable_head_starvation import _starvation_diagnostic

# --- Golden lines for detectors whose trigger is not already a reusable helper in test_detectors. ---
# Each mirrors a specific worker emit; the source is named so a reworded log line is traceable here.


def _quarantine(ts: str, *, slot: int = 1) -> str:
    """process_lifecycle._quarantine_inference_slot: a slot quarantined after crashing on start."""
    return (
        f"2026-06-24 {ts} | CRITICAL | horde_worker_regen.process_management.lifecycle.process_lifecycle:_quarantine_inference_slot:1182 - "
        f"Inference slot {slot} quarantined (crash on start: 3 consecutive failures before reaching readiness); not respawning it."
    )


def _pools_recovered(ts: str) -> str:
    """process_manager._run_recovery_supervisor: save-our-ship recovered the pool and cleared limp-by."""
    return (
        f"2026-06-24 {ts} | INFO | horde_worker_regen.process_management.process_manager:_run_recovery_supervisor:2062 - "
        "Save-our-ship: pools recovered; restored configured concurrency (limp-by cleared)."
    )


def _abandon_ship(ts: str) -> str:
    """process_manager._give_up_on_wedged_jobs: the worker self-terminates an unrecoverable pool."""
    return (
        f"2026-06-24 {ts} | CRITICAL | horde_worker_regen.process_management.process_manager:_give_up_on_wedged_jobs:2123 - "
        "Save-our-ship: the worker cannot restore a working process pool; abandoning ship"
    )


def _orphan_punt(ts: str, *, job_id: str, stuck_seconds: int = 42) -> str:
    """process_manager._reconcile_orphaned_in_progress_jobs: the in-progress orphan watchdog punting a job."""
    return (
        f"2026-06-25 {ts} | ERROR    | horde_worker_regen.process_management.process_manager:_reconcile_orphaned_in_progress_jobs:1992 - "
        f"Job {job_id} has been in progress with no live inference slot for {stuck_seconds}s; punting it so "
        "the queue can drain (orphaned-job watchdog)."
    )


def _stuck_step(ts: str, *, slot: int = 3, repeats: int = 3060) -> str:
    """process_lifecycle.replace_hung_processes: the stuck-step watchdog reaping a wedged sampling slot."""
    return (
        f"2026-06-26 {ts} | ERROR    | horde_worker_regen.process_management.lifecycle.process_lifecycle:replace_hung_processes:1830 - "
        f"Inference slot {slot} is stuck on a non-advancing sampling step (reported step 24/25 without "
        f"advancing {repeats} times); the ComfyUI generation will not return a result, replacing it "
        f"(stuck-step watchdog)."
    )


def _post_processing_stall(ts: str, *, slot: int = 3) -> str:
    """process_lifecycle._check_and_replace_process: the post-processing-stage watchdog reaping a slot."""
    return (
        f"2026-06-28 {ts} | ERROR    | horde_worker_regen.process_management.lifecycle.process_lifecycle:_check_and_replace_process:1618 - "
        f"HordeProcessInfo(process_id={slot}, last_process_state=HordeProcessState.INFERENCE_POST_PROCESSING, "
        f"loaded_horde_model_name=AAM XL AnimeMix) seems to be stuck post processing, replacing it"
    )


def _pp_deferral(ts: str, *, job_id: str) -> str:
    """post_process_orchestrator._has_post_processing_headroom: the admission gate deferring a job."""
    return (
        f"2026-06-24 {ts} | WARNING | horde_worker_regen.process_management.workers.post_process_orchestrator:_has_post_processing_headroom:102 - "
        f"Deferring post-processing for job {job_id}: estimated peak 8533 MB plus reserve 2048 MB exceeds "
        "free VRAM after commitments (6675 MB available on card 0). No idle VRAM reclaim was available."
    )


def _oom(ts: str) -> str:
    """An explicit CUDA out-of-memory fault surfaced from an inference slot."""
    return f"2026-06-24 {ts} | ERROR | x:y:1 - CUDA out of memory. Tried to allocate 2.00 GiB"


def _fd_exhaustion(ts: str, *, slot: int = 3, model: str = "WAI-NSFW-illustrious-SDXL") -> str:
    """message_dispatcher._handle_faulted_inference_result: a job faulted by EMFILE (errno 24).

    Mirrors the real emit: the generic 'Pipeline failed to run ... produced no results' wrapper, the
    faulting model, and the underlying 'Too many open files' the process hit once over RLIMIT_NOFILE.
    """
    return (
        f"2026-06-24 {ts} | WARNING  | horde_worker_regen.process_management.ipc.message_dispatcher:_handle_faulted_inference_result:892 - "
        f"Job 597d4471-b223-4383-b874-86c6a1549594 faulted on process {slot} (RuntimeError: Pipeline failed "
        f"to run - declared output node(s) ['output_image'] produced no results. Model: {model}. Error: "
        f"sampler (KSampler): OSError: [Errno 24] Too many open files: '/proc/meminfo'"
    )


def _no_images(ts: str) -> str:
    """A generic 'no images produced' fault (the swallowed-OOM classification gap)."""
    return f"2026-06-24 {ts} | WARNING | x:y:1 - Job faulted: no images were produced"


def _governor_enter(ts: str, *, name: str = "large_model_reentry") -> str:
    """PopGovernorRegistry: a governor spell opening (the ENTER boundary the dominance detector pairs)."""
    return (
        f"2026-06-24 {ts} | INFO | horde_worker_regen.process_management.scheduling.pop_governor_registry:_default_log:200 - "
        f"Pop governor ENTER: {name} (cooling down before serving any very-large model after the last drained); "
        "expected ~600s"
    )


def _governor_exit(ts: str, *, name: str = "large_model_reentry") -> str:
    """PopGovernorRegistry: a governor spell closing (the EXIT boundary carrying the session totals)."""
    return (
        f"2026-06-24 {ts} | INFO | horde_worker_regen.process_management.scheduling.pop_governor_registry:_default_log:200 - "
        f"Pop governor EXIT: {name} after 10m00s (1x this session, 10m00s total)"
    )


def _startup_line() -> str:
    """The main-process logger-setup line that opens a session (the segmentation boundary)."""
    return f"2026-06-24 18:29:20.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - {_STARTUP}"


def _bridge(*lines: str) -> str:
    """A single-session bridge log: the startup boundary followed by the given lines."""
    return "\n".join([_startup_line(), *lines])


@dataclass
class Contract:
    """A detector paired with a golden log that must make it fire, and the severity it must report."""

    bridge: str
    severity: Severity
    child_logs: dict[str, str] = field(default_factory=dict)


_REPLACED = "inference process replaced (crashed or hung)"

# One contract per detector, keyed by the detector's function name. The no-orphan test asserts this
# mapping covers every entry in DETECTORS, so a new detector forces a fixture to be added here.
CONTRACTS: dict[str, Contract] = {
    "detect_crash_on_start_loop": Contract(
        bridge=_bridge(
            _recovery("18:29:31.000", 1, reason=_REPLACED),
            _recovery("18:29:40.000", 1, reason=_REPLACED),
        ),
        child_logs={"bridge_inference_1_startup.log": _TRACEBACK},
        severity=Severity.CRITICAL,
    ),
    "detect_doomed_pool_no_giveup": Contract(
        bridge=_bridge(_quarantine("18:29:47.000"), _pools_recovered("18:31:00.000")),
        severity=Severity.CRITICAL,
    ),
    "detect_gave_up_clean": Contract(
        bridge=_bridge(_abandon_ship("18:31:20.000")),
        severity=Severity.INFO,
    ),
    "detect_forced_maintenance": Contract(
        bridge=_bridge(_give_up("15:19:08.000", jobs=4), _maintenance_pop("15:19:10.000")),
        severity=Severity.CRITICAL,
    ),
    "detect_pop_governor_dominance": Contract(
        # A 10-minute re-entry-cooldown spell over a ~10.5-minute session: well past the dominance share.
        bridge=_bridge(_governor_enter("18:30:00.000"), _governor_exit("18:40:00.000")),
        severity=Severity.INFO,
    ),
    "detect_post_processing_deferral_starvation": Contract(
        # The same job deferred every scheduling tick with no lane completion afterwards: the
        # admission-gate starvation storm.
        bridge=_bridge(
            *[
                _pp_deferral(
                    f"18:{30 + i // 60}:{i % 60:02d}.000",
                    job_id="4e17ddbd-a9cc-494d-b668-8f6fcb6d08aa",
                )
                for i in range(35)
            ],
        ),
        severity=Severity.CRITICAL,
    ),
    "detect_scheduler_starvation_wedge": Contract(
        bridge=_bridge(
            _force_admit("15:18:52.000", starved_seconds=110, free_vram_mb=19179),
            _soft_reset("15:18:43.000"),
            _give_up("15:19:08.000", jobs=4),
        ),
        severity=Severity.CRITICAL,
    ),
    "detect_unsatisfiable_head_starvation": Contract(
        # The same head model deferred with no verified progress repeatedly over a >120s window, with no
        # give-up or consecutive-failure pause ever clearing it: the silent, unsatisfiable-head wedge.
        bridge=_bridge(
            _starvation_diagnostic("18:30:00.000", starved_seconds=130, free_vram_mb=19000),
            _starvation_diagnostic("18:32:30.000", starved_seconds=205, free_vram_mb=19100),
            _starvation_diagnostic("18:35:00.000", starved_seconds=280, free_vram_mb=19200),
        ),
        severity=Severity.CRITICAL,
    ),
    "detect_slow_generation_drop_spiral": Contract(
        bridge=_bridge(
            _server_slow_abort("06:38:37.000"),
            _server_slow_abort("06:42:12.000"),
            _server_slow_abort("07:03:16.000"),
        ),
        severity=Severity.CRITICAL,
    ),
    "detect_safety_stage_stall": Contract(
        bridge=_bridge(_safety_lost_result("13:01:00.000"), _safety_requeue("13:01:46.000")),
        severity=Severity.WARNING,
    ),
    "detect_head_dispatch_stall": Contract(
        bridge=_bridge(_dispatch_stall("13:01:00.000", reason=_DISPATCH_BUG_REASON)),
        severity=Severity.CRITICAL,
    ),
    "detect_whole_card_convergence_wedge": Contract(
        bridge=_bridge(
            _dispatch_stall(
                "13:01:00.000",
                reason=_DISPATCH_WHOLE_CARD_REASON,
                model="Flux.1-Schnell fp8 (Compact)",
            ),
        ),
        severity=Severity.CRITICAL,
    ),
    "detect_whole_card_nonhead_residency_starvation": Contract(
        bridge=_bridge(
            _dispatch_stall("13:01:00.000", reason=_DISPATCH_NONHEAD_REASON, model="Juggernaut XL"),
            _soft_reset("13:01:30.000"),
        ),
        severity=Severity.CRITICAL,
    ),
    "detect_whole_card_residency_churn": Contract(
        bridge=_bridge(
            _whole_card_reserve("07:54:50.000"),
            _whole_card_reserve("07:55:45.000", model="CyberRealistic Pony"),
            _whole_card_reserve("07:57:46.000"),
        ),
        severity=Severity.WARNING,
    ),
    "detect_consecutive_failure_pause": Contract(
        bridge=_bridge(_consecutive_pause("15:19:11.000")),
        severity=Severity.WARNING,
    ),
    "detect_stuck_inference_step": Contract(
        bridge=_bridge(_stuck_step("09:48:02.000")),
        severity=Severity.WARNING,
    ),
    "detect_post_processing_vram_stall": Contract(
        bridge=_bridge(_post_processing_stall("16:53:42.000")),
        severity=Severity.WARNING,
    ),
    "detect_oom": Contract(
        bridge=_bridge(_oom("18:00:10.000")),
        severity=Severity.CRITICAL,
    ),
    "detect_file_descriptor_exhaustion": Contract(
        bridge=_bridge(_fd_exhaustion("20:09:24.000")),
        severity=Severity.CRITICAL,
    ),
    "detect_swallowed_oom": Contract(
        bridge=_bridge(_no_images("18:00:10.000")),
        severity=Severity.WARNING,
    ),
    "detect_orphan_wedge": Contract(
        bridge=_bridge(*(_orphan_punt(f"12:0{i}:00.000", job_id=f"job{i}") for i in range(6))),
        severity=Severity.WARNING,
    ),
    "detect_session_summary": Contract(
        bridge=_bridge(),
        severity=Severity.INFO,
    ),
}


def _finding_id(detector: Detector) -> str:
    """The primary finding id a detector emits, derived from its name (``detect_X`` -> ``X``)."""
    return detector.__name__.removeprefix("detect_")


@pytest.mark.parametrize("detector", DETECTORS, ids=lambda d: d.__name__)
def test_detector_fires_on_its_golden_signature(detector: Detector, tmp_path: Path) -> None:
    """Each detector produces its finding (at the expected severity) from a representative log line.

    This is the live half of the logging<->detector contract: if the worker rewords an emit so a
    detector no longer matches it, the detector's golden line stops firing and this test names it.
    """
    contract = CONTRACTS[detector.__name__]
    findings = _diagnose(tmp_path, contract.bridge, contract.child_logs or None)
    finding_id = _finding_id(detector)
    assert finding_id in findings, f"{detector.__name__} did not fire on its golden signature"
    assert findings[finding_id].severity is contract.severity


def test_every_detector_has_a_contract_fixture() -> None:
    """No detector ships without a golden-signature fixture (the omission guard).

    Adding a detector to ``DETECTORS`` without a contract here fails this test, forcing the author to
    record the log signature the detector keys off, which is the single manual step the contract requires.
    """
    registered = {detector.__name__ for detector in DETECTORS}
    assert registered == set(CONTRACTS), {
        "detectors_missing_a_fixture": sorted(registered - set(CONTRACTS)),
        "fixtures_for_unknown_detectors": sorted(set(CONTRACTS) - registered),
    }
