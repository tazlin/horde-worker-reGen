"""Turn a correlated session into ranked, actionable findings: what went wrong and what to do.

Each detector recognizes one incident class from the signals the worker already emits (recovery
diagnostics, the action ledger, child tracebacks) and returns a :class:`Finding` with a plain-language
verdict, the evidence that supports it, and a remediation. This is the automated form of the manual log
archeology a maintainer would otherwise do: the crash-on-start detector lifts the child's exception
across the process boundary; the doomed-pool detector recognizes the save-our-ship loop that spins
without ever giving up.

Detectors are independent and registered in :data:`DETECTORS`, so a new incident class is one function
plus one list entry. They never raise: a detector that cannot make sense of a session returns no
findings rather than aborting the report.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .correlate import SessionContext, find_child_crash
from .log_ingest import LogRecord
from .sessions import SessionEndReason

# Signatures over orchestrator message text.
_QUARANTINE_RE = re.compile(r"quarantined \(crash on start")
_SOFT_RESET_RE = re.compile(r"Save-our-ship soft reset")
_POOLS_RECOVERED_RE = re.compile(r"pools recovered.*limp-by cleared")
_ABANDON_SHIP_RE = re.compile(r"abandoning ship|cannot restore a working process pool")
_OOM_RE = re.compile(
    r"CUDA out of memory|OutOfMemoryError|torch\.cuda\.OutOfMemoryError|RuntimeError: .*out of memory",
)
_NO_IMAGES_RE = re.compile(r"no images were produced|no images produced")
_ORPHAN_RE = re.compile(r"orphaned? in-progress|punt(?:ing|ed) (?:an? )?orphan")

# A pool that flapped through at least this many soft resets is "stuck recovering", not a one-off blip.
_SOFT_RESET_FLAP_THRESHOLD = 2
# A recovery count at or above this is a storm worth surfacing on its own.
_RECOVERY_STORM_THRESHOLD = 5


class Severity(enum.StrEnum):
    """How urgent a finding is; also its sort key (critical first)."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


_SEVERITY_ORDER = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}


@dataclass
class Finding:
    """One diagnosis of a session: the verdict, the evidence, and what to do about it."""

    id: str
    severity: Severity
    title: str
    verdict: str
    remediation: str
    evidence: list[str] = field(default_factory=list)
    see_also: str | None = None


Detector = Callable[[SessionContext], "list[Finding]"]


def _matching(records: list[LogRecord], pattern: re.Pattern[str]) -> list[LogRecord]:
    """Orchestrator records whose message matches ``pattern``."""
    return [record for record in records if pattern.search(record.message)]


def _evidence(record: LogRecord) -> str:
    """A one-line evidence reference: timestamp, source location, and a clipped message."""
    ts = record.timestamp.strftime("%H:%M:%S") if record.timestamp else "--:--:--"
    message = record.message if len(record.message) <= 140 else record.message[:137] + "..."
    return f"{ts}  {record.source_path.name}:{record.raw_lineno}  {message}"


def detect_crash_on_start_loop(context: SessionContext) -> list[Finding]:
    """Inference children that crash during startup, with the child's exception as the root cause."""
    crashing = [r for r in context.recoveries if r.last_state == "PROCESS_STARTING" and r.exitcode not in ("0", "")]
    if len(crashing) < _SOFT_RESET_FLAP_THRESHOLD:
        return []

    # Lift the actual exception from each affected slot's child startup log (the cross-process join).
    exceptions: dict[str, int] = {}
    evidence: list[str] = []
    for process_id in sorted({r.process_id for r in crashing}):
        first = next(r for r in crashing if r.process_id == process_id)
        crash = find_child_crash(context.bundle, process_id, first.timestamp, os_pid=first.os_pid)
        if crash is not None:
            exceptions[crash.exception] = exceptions.get(crash.exception, 0) + 1
            evidence.append(f"slot {process_id}: {_evidence(crash.record).split('  ', 1)[1]} -> {crash.exception}")

    cause = max(exceptions, key=lambda exc: exceptions[exc]) if exceptions else None
    verdict = (
        f"{len(crashing)} inference start(s) across {len({r.process_id for r in crashing})} slot(s) crashed before "
        "reaching readiness"
    )
    if cause is not None:
        verdict += f"; the child raised `{cause}`."
        remediation = (
            f"The inference subprocess fails during hordelib/ComfyUI init with `{cause}`. Fix that "
            "environment fault (e.g. reinstall a CUDA-enabled torch if it reports torch was not compiled "
            "with CUDA); the worker cannot serve until the children start."
        )
    else:
        verdict += " (no child traceback found to attribute)."
        remediation = (
            "Inspect the affected slot's bridge_inference_<N>_startup.log for the failing import/exception; "
            "the parent only sees a nonzero exit code."
        )
    return [
        Finding(
            id="crash_on_start_loop",
            severity=Severity.CRITICAL,
            title="Inference pool crashes on start",
            verdict=verdict,
            remediation=remediation,
            evidence=evidence[:6] or [_evidence(crashing[0].record)],
        ),
    ]


def detect_doomed_pool_no_giveup(context: SessionContext) -> list[Finding]:
    """The save-our-ship loop spun on an unrecoverable pool instead of giving up (the observed bug)."""
    session = context.session
    soft_resets = _matching(session.records, _SOFT_RESET_RE)
    recovered = _matching(session.records, _POOLS_RECOVERED_RE)
    quarantined = _matching(session.records, _QUARANTINE_RE)
    gave_up = bool(_matching(session.records, _ABANDON_SHIP_RE))

    # The defining symptom: the pool reached full quarantine (proved unrecoverable) yet the worker kept
    # going (a soft-reset recovery and/or a recovery storm) instead of abandoning ship. Soft-reset count
    # is supporting evidence, not a gate: the slow-restart variant of the bug flaps with as few as one
    # soft reset per episode because each episode closes on a long clean window.
    unrecoverable_seen = bool(quarantined)
    kept_going = bool(recovered) or session.peak_process_recoveries >= _RECOVERY_STORM_THRESHOLD
    if gave_up or not (unrecoverable_seen and kept_going):
        return []

    verdict = (
        f"The pool quarantined and was soft-reset {len(soft_resets)} time(s), recovering "
        f"{len(recovered)} time(s), and reached {session.peak_process_recoveries} process recoveries, but the "
        f"worker never abandoned ship (it ended via {session.end_reason}). A deterministically-doomed pool "
        "flapped between soft reset and re-crash instead of self-terminating."
    )
    return [
        Finding(
            id="doomed_pool_no_giveup",
            severity=Severity.CRITICAL,
            title="Recovery storm never gave up",
            verdict=verdict,
            remediation=(
                "Pair with the crash-on-start root cause: the pool cannot recover, so it should give up "
                "fast. The give-up abort only fires when every slot is quarantined at the exact give-up "
                "tick, but a soft reset's transient un-quarantine (and a clean window longer than the "
                "recovery clean streak) keeps that from coinciding, so the worker spins. Make the abort "
                "latch 'was fully quarantined this episode' rather than sampling the instantaneous state."
            ),
            evidence=[_evidence(r) for r in (soft_resets[:2] + recovered[:1] + quarantined[:1])],
            see_also="recovery_supervisor give-up phase mismatch",
        ),
    ]


def detect_gave_up_clean(context: SessionContext) -> list[Finding]:
    """The worker correctly abandoned ship on an unrecoverable pool (the healthy terminal path)."""
    abandon = _matching(context.session.records, _ABANDON_SHIP_RE)
    if not abandon:
        return []
    return [
        Finding(
            id="gave_up_clean",
            severity=Severity.INFO,
            title="Worker gave up on an unrecoverable pool",
            verdict=(
                "Save-our-ship abandoned ship and self-terminated after soft resets could not restore a "
                "working pool. This is the intended bail-out, not a hang; see the crash-on-start finding "
                "for why the pool was unrecoverable."
            ),
            remediation="No worker action needed beyond fixing the underlying crash cause; the bail-out worked.",
            evidence=[_evidence(abandon[0])],
        ),
    ]


def detect_oom(context: SessionContext) -> list[Finding]:
    """Out-of-memory faults (explicit CUDA OOM)."""
    oom = _matching(context.session.records, _OOM_RE)
    if not oom:
        return []
    return [
        Finding(
            id="oom",
            severity=Severity.CRITICAL,
            title="GPU out-of-memory faults",
            verdict=f"{len(oom)} out-of-memory fault(s) during the session.",
            remediation=(
                "Reduce concurrency/queue or enable a more conservative VRAM budget; if these recur under "
                "a budget that should fit, suspect over-admission of a heavy head (Flux fp8 / SDXL)."
            ),
            evidence=[_evidence(r) for r in oom[:4]],
        ),
    ]


def detect_swallowed_oom(context: SessionContext) -> list[Finding]:
    """The 'no images were produced' classification gap (an OOM ComfyUI swallowed)."""
    no_images = _matching(context.session.records, _NO_IMAGES_RE)
    if not no_images:
        return []
    return [
        Finding(
            id="swallowed_oom",
            severity=Severity.WARNING,
            title="Jobs faulted with 'no images produced'",
            verdict=(
                f"{len(no_images)} job(s) faulted with a generic 'no images produced' message. ComfyUI can "
                "swallow a CUDA OOM into this generic failure, so the resource-failure breaker may never "
                "fire even though the cause was memory pressure."
            ),
            remediation=(
                "Check VRAM headroom around these faults; if memory-bound, treat 'no images produced' as a "
                "resource failure so the self-throttle/breaker engages."
            ),
            evidence=[_evidence(r) for r in no_images[:4]],
        ),
    ]


def detect_orphan_wedge(context: SessionContext) -> list[Finding]:
    """A storm of orphaned in-progress jobs (a flaky GPU stranding each inference)."""
    orphans = _matching(context.session.records, _ORPHAN_RE)
    if len(orphans) < _RECOVERY_STORM_THRESHOLD:
        return []
    return [
        Finding(
            id="orphan_wedge",
            severity=Severity.WARNING,
            title="Orphaned in-progress jobs",
            verdict=(
                f"{len(orphans)} job(s) were punted as orphaned in-progress with no owning live slot. A "
                "recurring orphan storm means something upstream keeps stranding jobs (often a GPU that "
                "hangs each inference)."
            ),
            remediation=(
                "Inspect the inference slots for hangs/OOM around these punts; a sustained storm should "
                "escalate to a soft reset and reduced concurrency."
            ),
            evidence=[_evidence(r) for r in orphans[:4]],
        ),
    ]


def detect_session_summary(context: SessionContext) -> list[Finding]:
    """An always-present rollup: how the session ended and its recovery/fault headline numbers."""
    session = context.session
    duration = session.duration_seconds
    duration_text = f"{duration / 60:.1f} min" if duration is not None else "unknown duration"
    severity = Severity.WARNING if session.end_reason is SessionEndReason.KILLED_OR_CRASHED else Severity.INFO
    return [
        Finding(
            id="session_summary",
            severity=severity,
            title="Session summary",
            verdict=(
                f"Ended via {session.end_reason} after {duration_text}; peak process recoveries "
                f"{session.peak_process_recoveries}; {len(context.recoveries)} recovery diagnostic(s); "
                f"version v{session.version or '?'}, {session.num_models or '?'} models, "
                f"{session.max_threads or '?'} thread(s)."
            ),
            remediation="",
        ),
    ]


DETECTORS: list[Detector] = [
    detect_crash_on_start_loop,
    detect_doomed_pool_no_giveup,
    detect_gave_up_clean,
    detect_oom,
    detect_swallowed_oom,
    detect_orphan_wedge,
    detect_session_summary,
]


def run_detectors(context: SessionContext) -> list[Finding]:
    """Run all detectors over a session and return their findings, most-severe first."""
    findings: list[Finding] = []
    for detector in DETECTORS:
        try:
            findings.extend(detector(context))
        except Exception:  # noqa: BLE001 - a broken detector must not sink the whole report.
            continue
    findings.sort(key=lambda finding: _SEVERITY_ORDER[finding.severity])
    return findings
