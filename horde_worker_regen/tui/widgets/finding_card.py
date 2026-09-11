"""One finding as a dashboard card, shared by the Diagnostics and Insights tabs.

The card is rendered generically from a :class:`~horde_worker_regen.analysis.finding_kinds.Finding`'s
fields (badge, title, headline, action, detail, evidence, cross-references), so a newly added kind
appears with no change here and a log diagnosis and a live insight read the same way.
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Collapsible, Static

from horde_worker_regen.analysis.finding_kinds import Finding, Severity

# Presentation-only mapping of a finding's severity to badge and border colours. The badge word itself
# comes from the analysis layer (``Finding.badge``) so the CLI and the dashboard say the same thing; the
# colours are a display choice, so a new severity surfaces here in the fallback grey until styled.
_SEVERITY_BADGE_STYLE: dict[Severity, str] = {
    Severity.CRITICAL: "bold white on red",
    Severity.WARNING: "black on yellow",
    Severity.SUGGESTION: "black on green",
    Severity.INFO: "black on cyan",
}
_SEVERITY_BORDER: dict[Severity, str] = {
    Severity.CRITICAL: "red",
    Severity.WARNING: "yellow",
    Severity.SUGGESTION: "green",
    Severity.INFO: "cyan",
}


class FindingCard(Vertical):
    """One finding: the badge and title, the headline and the "Do this" line, and its details folded away.

    The plain layer is always visible so a reader has the answer without a click; the detail layer (the
    mechanism, the evidence, the cross-references) sits in a collapsed section under it. Enter or a click
    on the section's header opens it, and the focus order walks from one finding to the next.
    """

    DEFAULT_CSS = """
    FindingCard {
        height: auto;
        margin: 0 0 1 0;
        padding: 0 1;
        border: round $panel;
    }
    FindingCard .finding-header {
        height: auto;
    }
    FindingCard .finding-headline {
        height: auto;
        margin: 0 0 0 1;
    }
    FindingCard .finding-action {
        height: auto;
        margin: 0 0 0 1;
    }
    FindingCard Collapsible {
        margin: 0;
        padding: 0;
        border: none;
    }
    FindingCard CollapsibleTitle {
        padding: 0 1;
        color: $text-muted;
    }
    FindingCard .finding-detail {
        height: auto;
        margin: 0 0 0 1;
    }
    """

    def __init__(self, finding: Finding) -> None:
        """Hold the finding this card shows."""
        super().__init__()
        self.finding = finding

    def compose(self) -> ComposeResult:
        """Lay out the plain layer, then the collapsed detail layer when the finding has one."""
        finding = self.finding
        badge_style = _SEVERITY_BADGE_STYLE.get(finding.severity, "bold")
        yield Static(
            Text.assemble((f" {finding.badge} ", badge_style), ("  ", ""), (finding.title, "bold")),
            classes="finding-header",
        )
        yield Static(Text(finding.headline), classes="finding-headline")
        if finding.action:
            yield Static(
                Text.assemble(("Do this: ", "bold green"), (finding.action, "green")),
                classes="finding-action",
            )
        detail = self._detail_text(finding)
        if detail is not None:
            with Collapsible(title="Details", collapsed=True):
                yield Static(detail, classes="finding-detail")

    def on_mount(self) -> None:
        """Colour the border by severity once the widget has styles to set."""
        self.styles.border = ("round", _SEVERITY_BORDER.get(self.finding.severity, "grey50"))

    @staticmethod
    def _detail_text(finding: Finding) -> Text | None:
        """The detail layer as one block: the prose, the evidence, the cross-reference and the docs page."""
        parts: list[Text] = []
        if finding.detail:
            parts.append(Text(finding.detail))
        if finding.evidence:
            evidence = Text("Evidence", style="grey46 bold")
            for line in finding.evidence:
                evidence.append(f"\n  • {line}", style="grey58")
            parts.append(evidence)
        if finding.see_also:
            parts.append(Text(f"See also: {finding.see_also}", style="grey50 italic"))
        if finding.reference_page:
            parts.append(Text(f"More: {finding.reference_page}", style="grey50 italic"))
        if not parts:
            return None
        joined = Text()
        for index, part in enumerate(parts):
            if index:
                joined.append("\n\n")
            joined.append_text(part)
        return joined
