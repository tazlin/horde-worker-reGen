"""The Overview layout registry and the "Customize layout" modal.

The registry ([`OVERVIEW_ELEMENTS`][horde_worker_regen.tui.widgets.overview_layout.OVERVIEW_ELEMENTS]) is the
single source of truth for which Overview panels an operator can hide. It is consumed in three places that
must never drift apart: the customize modal (what the operator toggles), ``OverviewView.update_view`` (which
masks a hidden element's node), and the persisted ``WorkerAppState.overview_hidden_elements`` list (keyed by
the stable ``key`` field). A regression test asserts every mode-managed node id has a registry entry.

Hiding is an *additional* mask layered on top of the density-mode and conditional-guard logic: an element
that is already suppressed by the current view mode (or because its feature is inactive) stays suppressed;
hiding only ever removes an element that would otherwise be shown, never forces one on.
"""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Checkbox, Footer, Label, Static


@dataclass(frozen=True)
class OverviewElement:
    """One independently hideable Overview panel.

    Attributes:
        key: Stable identifier persisted in ``WorkerAppState.overview_hidden_elements``. Never reuse a key
            for a different panel: an operator's saved preference is keyed by it.
        node_id: The Textual query id of the panel's ``Static`` node (with the leading ``#``).
        label: Human label shown in the customize modal.
        group: Section header the modal groups the element under.
    """

    key: str
    node_id: str
    label: str
    group: str


# The display order here is also the modal's order. Grouped so related toggles sit together; the group
# strings double as the modal's section headers.
OVERVIEW_ELEMENTS: tuple[OverviewElement, ...] = (
    OverviewElement("hero", "#overview-hero", "Status hero", "Status & health"),
    OverviewElement("health", "#overview-health", "Health checklist", "Status & health"),
    OverviewElement("trends", "#overview-trends", "Trends", "Status & health"),
    OverviewElement("gpus", "#overview-gpus", "GPUs", "Workload"),
    OverviewElement("pipeline", "#overview-pipeline", "Job pipeline", "Workload"),
    OverviewElement("intent", "#overview-intent", "Now / Next / Why", "Workload"),
    OverviewElement("queue", "#overview-queue", "Queue", "Workload"),
    OverviewElement("governance", "#overview-governance", "Governance", "Governance"),
    OverviewElement("work", "#overview-work", "Work ledger", "Jobs"),
    OverviewElement("processes", "#overview-processes", "Processes", "Jobs"),
    OverviewElement("recent", "#overview-recent", "Recent jobs", "Jobs"),
    OverviewElement("worker", "#overview-worker", "Worker config", "Worker"),
    OverviewElement("alchemy", "#overview-alchemy", "Alchemy", "Worker"),
    OverviewElement("residency", "#overview-residency", "Whole-card residency", "Worker"),
)

_ELEMENTS_BY_KEY: dict[str, OverviewElement] = {element.key: element for element in OVERVIEW_ELEMENTS}
_ELEMENTS_BY_NODE: dict[str, OverviewElement] = {element.node_id: element for element in OVERVIEW_ELEMENTS}


def element_for_node(node_id: str) -> OverviewElement | None:
    """Return the registry entry that owns ``node_id`` (with leading ``#``), or None if untracked."""
    return _ELEMENTS_BY_NODE.get(node_id)


def valid_hidden_keys(keys: object) -> set[str]:
    """Return only the keys from ``keys`` that name a known element, tolerating unknown/renamed keys.

    Persisted preferences may outlive a renamed or removed element; those keys are dropped rather than
    raising, so an old state file never blocks the Overview from rendering.
    """
    if not isinstance(keys, (list, tuple, set, frozenset)):
        return set()
    return {key for key in keys if key in _ELEMENTS_BY_KEY}


class OverviewLayoutModal(ModalScreen[frozenset[str] | None]):
    """Toggle which Overview panels are hidden; dismisses with the chosen hidden-key set (or None).

    A checked box means *hidden*. Escape saves and closes (there is no separate cancel: the operator is
    editing a persistent preference, so the intuitive exit is to keep what they see).
    """

    BINDINGS = [
        Binding("escape", "save", "Save & close"),
        Binding("a", "hide_all", "Hide all"),
        Binding("n", "show_all", "Show all"),
    ]

    DEFAULT_CSS = """
    OverviewLayoutModal {
        align: center middle;
    }
    OverviewLayoutModal #layout-dialog {
        width: 70%;
        max-width: 90;
        height: 80%;
        max-height: 40;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    OverviewLayoutModal .dialog-title {
        text-style: bold;
        margin-bottom: 1;
    }
    OverviewLayoutModal .layout-group {
        text-style: bold;
        color: $text-muted;
        margin-top: 1;
    }
    OverviewLayoutModal Checkbox {
        border: none;
        height: 1;
        margin: 0 0 0 1;
        padding: 0;
    }
    OverviewLayoutModal #layout-hint {
        color: $text-muted;
        margin-top: 1;
    }
    """

    def __init__(self, hidden_keys: frozenset[str]) -> None:
        """Create the modal with the elements in ``hidden_keys`` pre-checked as hidden."""
        super().__init__()
        self._hidden_keys = valid_hidden_keys(hidden_keys)

    def compose(self) -> ComposeResult:
        """Lay out one checkbox per element, grouped under its section header."""
        with VerticalScroll(id="layout-dialog"):
            yield Static("Customize overview layout", classes="dialog-title")
            last_group: str | None = None
            for element in OVERVIEW_ELEMENTS:
                if element.group != last_group:
                    yield Label(element.group, classes="layout-group")
                    last_group = element.group
                yield Checkbox(
                    element.label,
                    value=element.key in self._hidden_keys,
                    id=f"layout-cb-{element.key}",
                )
            yield Static("space: toggle   a: hide all   n: show all   esc: save & close", id="layout-hint")
            yield Footer()

    def _set_all(self, hidden: bool) -> None:
        """Set every checkbox to ``hidden`` (True hides the element)."""
        for checkbox in self.query(Checkbox):
            checkbox.value = hidden

    def action_hide_all(self) -> None:
        """Mark every element hidden."""
        self._set_all(True)

    def action_show_all(self) -> None:
        """Mark every element visible."""
        self._set_all(False)

    def action_save(self) -> None:
        """Collect the checked (hidden) elements and dismiss with their keys."""
        hidden = {
            element.key for element in OVERVIEW_ELEMENTS if self.query_one(f"#layout-cb-{element.key}", Checkbox).value
        }
        self.dismiss(frozenset(hidden))
