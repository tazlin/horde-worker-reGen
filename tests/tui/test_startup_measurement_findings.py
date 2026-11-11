"""The Insights tab's startup measurement finding: which readings raise it, and the copy it shows."""

from __future__ import annotations

from horde_worker_regen.analysis.finding_kinds import Finding, FindingKind, Severity
from horde_worker_regen.process_management.ipc.supervisor_channel import (
    StartupContextReading,
    StartupContextRun,
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.tui.recommendations import analyze
from tests.analysis.test_finding_copy_style import _emit_problems

_NOW = 1_800_000_000.0


def _run(first_mb: int, marginal_mb: int) -> StartupContextRun:
    return StartupContextRun(
        first_context_mb=first_mb,
        marginal_mb=marginal_mb,
        baseline_mb=3600,
        after_context_mb=3600 + first_mb,
        after_holder_mb=3600 + first_mb + marginal_mb,
    )


def _reading(device_index: int, verdict: str, *, override_mb: int | None = None) -> StartupContextReading:
    if verdict != "inconsistent":
        return StartupContextReading(
            device_index=device_index,
            chosen=_run(487, 488),
            verdict=verdict,
            probe_runs=1,
            chosen_run=1,
            first_context_override_mb=override_mb,
        )
    runs = [_run(1161, 426), _run(900, 300), _run(1000, 200)]
    return StartupContextReading(
        device_index=device_index,
        chosen=runs[2],
        verdict=verdict,
        probe_runs=3,
        chosen_run=3,
        first_context_override_mb=override_mb,
        runs=runs,
    )


def _snapshot(readings: list[StartupContextReading]) -> WorkerStateSnapshot:
    return WorkerStateSnapshot(
        timestamp=_NOW,
        config=WorkerConfigSummary(dreamer_name="Measured", worker_version="12.0.0"),
        startup_context_readings=readings,
    )


def _measurement_findings(snapshot: WorkerStateSnapshot) -> list[Finding]:
    return [finding for finding in analyze(snapshot) if finding.kind is FindingKind.STARTUP_MEMORY_FIGURES_DISAGREE]


def _describe(finding: Finding) -> str:
    return "\n".join(
        [
            f"[{finding.badge}] {finding.title}",
            f"Headline: {finding.headline}",
            f"Do this: {finding.action}",
            f"Details: {finding.detail}",
            *(f"  - {line}" for line in finding.evidence),
        ],
    )


def test_an_inconsistent_card_gets_a_finding_with_its_tries() -> None:
    """A single card whose figures never agreed is a warning with the try count, every try and the figure in use."""
    findings = _measurement_findings(_snapshot([_reading(0, "inconsistent")]))
    print(*(_describe(finding) for finding in findings), sep="\n\n")

    (finding,) = findings
    assert finding.severity is Severity.WARNING
    assert finding.badge == "Check"
    assert finding.title == "The startup memory figures did not agree"
    assert finding.headline == "The two startup memory figures for the card did not agree on any of 3 tries."
    assert finding.action == (
        "Restart the worker while no other program is using the card. "
        "Or set `vram_per_process_overhead_mb` to the figure you know for this card. "
        "A value above 0 replaces the measured first-process figure on every card."
    )
    assert "subtracts the figure from the card's capacity when it checks what work fits" in finding.detail
    assert "A figure that is too high holds back work that would fit." in finding.detail
    assert "A figure that is too low lets work start that does not fit" in finding.detail
    assert finding.evidence == [
        "Try 1: first process 1161 MB, second process 426 MB.",
        "Try 2: first process 900 MB, second process 300 MB.",
        "Try 3: first process 1000 MB, second process 200 MB.",
        "The worker uses 1000 MB, the middle figure of the tries.",
        "That figure is from try 3.",
    ]
    assert finding.reference_page == "docs/how-to/use-the-dashboard.md"
    assert _emit_problems(finding) == []


def test_consistent_and_unchecked_cards_get_no_finding() -> None:
    """Only a card whose figures never agreed raises the finding."""
    readings = [_reading(0, "consistent"), _reading(1, "unchecked"), _reading(2, "something_newer")]

    assert _measurement_findings(_snapshot(readings)) == []
    assert _measurement_findings(_snapshot([])) == []


def test_two_cards_name_the_card_that_disagreed() -> None:
    """On a host with several cards the headline names the card, and only the inconsistent one fires."""
    findings = _measurement_findings(_snapshot([_reading(0, "consistent"), _reading(1, "inconsistent")]))

    (finding,) = findings
    assert finding.headline == "The two startup memory figures for card 1 did not agree on any of 3 tries."
    assert _emit_problems(finding) == []


def test_a_set_override_is_named_as_the_figure_in_use() -> None:
    """With `vram_per_process_overhead_mb` above 0 the evidence names the setting as the figure in use."""
    (finding,) = _measurement_findings(_snapshot([_reading(0, "inconsistent", override_mb=600)]))

    assert finding.evidence[-1] == "The worker uses 600 MB, set in `vram_per_process_overhead_mb`."
