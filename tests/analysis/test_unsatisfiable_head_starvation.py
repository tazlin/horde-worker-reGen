"""Tests for the unsatisfiable/starved-head detector and its change-only live-watch behavior.

The signature is the VRAM arbiter's head-of-queue starvation diagnostic: the same model deferred with no
verified progress repeatedly across a long window. The critical case is the one nothing (give-up or
pop-hold) ever resolves; a give-up or a consecutive-failure pause downgrades it to a warning.
"""

from __future__ import annotations

from pathlib import Path

from horde_worker_regen.analysis.bundle import LogBundle
from horde_worker_regen.analysis.detectors import Severity
from horde_worker_regen.analysis.watch import WatchState, watch_pass
from tests.analysis.test_detectors import _consecutive_pause, _diagnose, _give_up

_STARTUP_LINE = (
    "2026-06-24 18:29:20.000 | DEBUG | hordelib.utils.logger:set_sinks:269 - Setting up logger for main process"
)
_MODEL = "AlbedoBase XL (SDXL)"


def _starvation_diagnostic(ts: str, *, starved_seconds: int, free_vram_mb: int, model: str = _MODEL) -> str:
    """The arbiter's head-of-queue starvation diagnostic, verbatim from a live worker log."""
    available = free_vram_mb - 819
    return (
        f"2026-06-25 {ts} | WARNING  | "
        "horde_worker_regen.process_management.resources.vram_arbiter:_note_starvation_diagnostic:843 - "
        f"Head-of-queue {model} deferred {starved_seconds}s >= 60s with no verified progress; it stays queued "
        "for the structural-wedge recovery supervisor to reroute. Measured: candidate 14573 MB vs available "
        f"(device-free {free_vram_mb} - reservations 0 - noise 819) = {available} MB: does NOT fit."
    )


def _starvation_lines() -> list[str]:
    """Three force-admit diagnostics for one model spanning five minutes on an idle device."""
    return [
        _starvation_diagnostic("18:30:00.000", starved_seconds=130, free_vram_mb=19000, model=_MODEL),
        _starvation_diagnostic("18:32:30.000", starved_seconds=205, free_vram_mb=19100, model=_MODEL),
        _starvation_diagnostic("18:35:00.000", starved_seconds=280, free_vram_mb=19200, model=_MODEL),
    ]


def _bridge(*extra: str) -> str:
    return "\n".join([_STARTUP_LINE, *_starvation_lines(), *extra])


class TestUnsatisfiableHeadStarvation:
    """The persistent, unschedulable head with no corrective action is the critical case."""

    def test_critical_when_nothing_resolves(self, tmp_path: Path) -> None:
        """Repeated same-model starvation over > 120s with no give-up/pause is critical and names the model."""
        findings = _diagnose(tmp_path, _bridge())
        assert "unsatisfiable_head_starvation" in findings
        finding = findings["unsatisfiable_head_starvation"]
        assert finding.severity is Severity.CRITICAL
        assert "AlbedoBase XL" in finding.verdict
        assert "280s" in finding.verdict  # the measured starvation arithmetic from the log line

    def test_warns_when_give_up_resolves_it(self, tmp_path: Path) -> None:
        """A save-our-ship give-up within the window downgrades the finding to a warning."""
        findings = _diagnose(tmp_path, _bridge(_give_up("18:36:00.000", jobs=4)))
        assert findings["unsatisfiable_head_starvation"].severity is Severity.WARNING

    def test_warns_when_pop_hold_resolves_it(self, tmp_path: Path) -> None:
        """A consecutive-failure pop-hold within the window also downgrades it to a warning."""
        findings = _diagnose(tmp_path, _bridge(_consecutive_pause("18:36:00.000")))
        assert findings["unsatisfiable_head_starvation"].severity is Severity.WARNING

    def test_silent_on_a_single_short_deferral(self, tmp_path: Path) -> None:
        """A single, short deferral is transient backpressure, not a persistent unschedulable head."""
        bridge = "\n".join(
            [
                _STARTUP_LINE,
                _starvation_diagnostic("18:30:00.000", starved_seconds=40, free_vram_mb=19000, model=_MODEL),
                _starvation_diagnostic("18:30:30.000", starved_seconds=50, free_vram_mb=19000, model=_MODEL),
            ],
        )
        assert "unsatisfiable_head_starvation" not in _diagnose(tmp_path, bridge)


class TestWatchIntegration:
    """The finding must surface once through the change-only watch pass and stay quiet afterward."""

    def test_alerts_once_then_quiet(self, tmp_path: Path) -> None:
        """watch_pass emits the starvation finding on first sight and not again while unchanged."""
        (tmp_path / "bridge.log").write_text(_bridge(), encoding="utf-8")
        bundle = LogBundle.from_path(tmp_path)
        alerts, state = watch_pass(bundle, WatchState())
        assert any("persistently starved" in alert.lower() for alert in alerts)
        alerts_again, _ = watch_pass(bundle, state)
        assert not any("persistently starved" in alert.lower() for alert in alerts_again)

    def test_absent_on_healthy_logs(self, tmp_path: Path) -> None:
        """A healthy session (no head starvation) produces no starvation alert."""
        (tmp_path / "bridge.log").write_text(
            _STARTUP_LINE + "\n2026-06-24 18:30:00.000 | INFO | x:y:1 - Job submitted successfully\n",
            encoding="utf-8",
        )
        alerts, _ = watch_pass(LogBundle.from_path(tmp_path), WatchState())
        assert not any("persistently starved" in alert.lower() for alert in alerts)
