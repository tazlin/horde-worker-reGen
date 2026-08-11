"""Widgets specific to the progressive experience levels.

The levels change how much implementation detail each destination renders; they never change which
destinations exist. Every tab is present at every level, so nothing here hides or reorders navigation.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence

from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import Button, Label, Rule, Select, Static

from horde_worker_regen.app_state import DisplayDensity, ExperienceLevel
from horde_worker_regen.tui.responsive import ResponsiveModalScreen

_LEVEL_BUTTON_IDS: dict[ExperienceLevel, str] = {
    ExperienceLevel.SIMPLE: "experience-simple",
    ExperienceLevel.ADVANCED: "experience-advanced",
    ExperienceLevel.DEVELOPER: "experience-developer",
}
"""The button that selects each level, shared by the composer and the active-state styling."""

_LEVEL_SUMMARIES: dict[ExperienceLevel, tuple[str, str]] = {
    ExperienceLevel.SIMPLE: (
        "Simple",
        "Plain-language status, live progress, and the settings most contributors need. Every tab is "
        "still here if you want to look; they just lead with what the worker is doing rather than how.",
    ),
    ExperienceLevel.ADVANCED: (
        "Advanced",
        "The full operator surface: queues, per-process state, scheduler behaviour, download detail, "
        "and the complete configuration.",
    ),
    ExperienceLevel.DEVELOPER: (
        "Developer",
        "Everything in Advanced plus forensic detail and settings that can stop a healthy worker. "
        "Intended for debugging and tuning rather than everyday contribution.",
    ),
}


_THEME_CHOICES = (
    ("Horde Dark", "horde-dark"),
    ("Horde Light", "horde-light"),
    ("Terminal colours (ANSI)", "horde-ansi"),
)

_DENSITY_CHOICES = (
    ("Comfortable", DisplayDensity.COMFORTABLE.value),
    ("Compact", DisplayDensity.COMPACT.value),
)


class DashboardPreferencesView(Vertical):
    """Dashboard appearance controls: experience level, density, and theme.

    These are frontend preferences held in the durable app state. They sit on the Config tab because
    that is where people look to change settings, but nothing here reaches ``bridgeData.yaml``, and the
    surrounding editor never sees the form as dirty: every raw widget event is stopped at this widget and
    re-emitted as a typed message, beyond the reach of the editor's save tracking.
    """

    class ExperienceLevelChanged(Message):
        """Posted when a different experience level is chosen."""

        def __init__(self, level: ExperienceLevel) -> None:
            """Store the requested level."""
            super().__init__()
            self.level = level

    class DensityChanged(Message):
        """Posted when a different display density is chosen."""

        def __init__(self, density: DisplayDensity) -> None:
            """Store the requested density."""
            super().__init__()
            self.density = density

    class ThemeChanged(Message):
        """Posted when a different theme is chosen."""

        def __init__(self, theme_name: str) -> None:
            """Store the requested theme name."""
            super().__init__()
            self.theme_name = theme_name

    DEFAULT_CSS = """
    DashboardPreferencesView {
        height: auto;
    }
    DashboardPreferencesView .experience-row {
        height: 3;
    }
    DashboardPreferencesView .experience-row Button {
        margin-right: 1;
        min-width: 16;
    }
    DashboardPreferencesView .experience-choice {
        height: 3;
    }
    DashboardPreferencesView .experience-choice Label {
        width: 20;
        height: 3;
        content-align: left middle;
    }
    DashboardPreferencesView .experience-choice Select {
        width: 34;
    }
    DashboardPreferencesView #experience-summary {
        height: auto;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(self, level: ExperienceLevel, density: DisplayDensity, theme_name: str) -> None:
        """Render the controls pre-set to the persisted preferences."""
        super().__init__()
        self._level = level
        self._density = density
        self._theme_name = theme_name

    def compose(self) -> ComposeResult:
        """Lay out the level buttons and the density and theme selectors."""
        yield Label("Dashboard", classes="config-section")
        yield Rule()
        yield Static(
            "How much detail this dashboard shows. Every tab is available at every level; the level "
            "changes how each one is presented, not which ones exist.",
            classes="config-guidance",
        )
        with Horizontal(classes="experience-row"):
            for level, button_id in _LEVEL_BUTTON_IDS.items():
                yield Button(_LEVEL_SUMMARIES[level][0], id=button_id)
        yield Static(id="experience-summary")
        with Horizontal(classes="experience-choice"):
            yield Label("Spacing")
            yield Select(_DENSITY_CHOICES, value=self._density.value, id="experience-density", allow_blank=False)
        with Horizontal(classes="experience-choice"):
            yield Label("Theme")
            yield Select(_THEME_CHOICES, value=self._theme_name, id="experience-theme", allow_blank=False)

    def on_mount(self) -> None:
        """Reflect the starting level."""
        self.set_experience_level(self._level)

    def set_experience_level(self, level: ExperienceLevel) -> None:
        """Highlight ``level`` as active and describe what it shows.

        Tolerates being called before ``compose`` has mounted the controls: the level is recorded either
        way and ``on_mount`` re-applies it, so the only effect of an early call is that there is nothing
        yet to restyle.
        """
        self._level = level
        for candidate, button_id in _LEVEL_BUTTON_IDS.items():
            with contextlib.suppress(NoMatches):
                self.query_one(f"#{button_id}", Button).variant = "primary" if candidate is level else "default"
        _name, summary = _LEVEL_SUMMARIES[level]
        with contextlib.suppress(NoMatches):
            self.query_one("#experience-summary", Static).update(summary)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Request a level change, keeping the press away from the surrounding config form."""
        by_button_id = {button_id: level for level, button_id in _LEVEL_BUTTON_IDS.items()}
        requested = by_button_id.get(event.button.id or "")
        if requested is None:
            return
        event.stop()
        self.post_message(self.ExperienceLevelChanged(requested))

    def on_select_changed(self, event: Select.Changed) -> None:
        """Request a density or theme change without touching worker-config state."""
        if event.select.id == "experience-density":
            event.stop()
            self.post_message(self.DensityChanged(DisplayDensity(str(event.value))))
        elif event.select.id == "experience-theme":
            event.stop()
            self.post_message(self.ThemeChanged(str(event.value)))


class ExperienceIntroductionModal(ResponsiveModalScreen[ExperienceLevel | None]):
    """One-time notice that the dashboard now opens in Simple.

    Shown only to an installation whose durable state predates the experience levels. Its purpose is to
    make the changed default visible, so a contributor who was using the full dashboard is never left
    hunting for detail that appears to have vanished.
    """

    DEFAULT_CSS = """
    ExperienceIntroductionModal {
        align: center middle;
    }
    ExperienceIntroductionModal #experience-intro-dialog {
        width: 70;
        max-width: 90%;
        height: auto;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    ExperienceIntroductionModal #experience-intro-dialog Button {
        width: 100%;
        margin-top: 1;
    }
    """

    BINDINGS = [("escape", "keep_current", "Keep Advanced")]

    def compose(self) -> ComposeResult:
        """Lay out the notice and the two choices."""
        with Vertical(id="experience-intro-dialog"):
            yield Static(self._message(), id="experience-intro-message")
            yield Button("Keep the full dashboard (Advanced)", id="experience-intro-advanced", variant="primary")
            yield Button("Try the simplified view (Simple)", id="experience-intro-simple")

    @staticmethod
    def _message() -> Text:
        """Build the notice body."""
        return Text.assemble(
            ("The dashboard now has experience levels\n\n", "bold"),
            (
                "New installs open in Simple, which leads with plain-language status and live progress. "
                "Because this worker was set up before that change, you can keep the full dashboard "
                "instead.\n\n"
                "Every tab exists at every level, so nothing is hidden either way. You can change this "
                "any time in Config, or from the command palette with Ctrl+P.",
                "grey70",
            ),
        )

    def action_keep_current(self) -> None:
        """Escape keeps the full dashboard, matching what this installation had before."""
        self.dismiss(ExperienceLevel.ADVANCED)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Record the chosen level."""
        if event.button.id == "experience-intro-advanced":
            self.dismiss(ExperienceLevel.ADVANCED)
        elif event.button.id == "experience-intro-simple":
            self.dismiss(ExperienceLevel.SIMPLE)


class DeveloperWarningModal(ResponsiveModalScreen[bool]):
    """Confirm the first entry into the Developer level."""

    DEFAULT_CSS = """
    DeveloperWarningModal {
        align: center middle;
    }
    DeveloperWarningModal #developer-warning-dialog {
        width: 68;
        max-width: 90%;
        height: auto;
        padding: 1 2;
        border: thick $warning;
        background: $surface;
    }
    DeveloperWarningModal #developer-warning-dialog Button {
        width: 100%;
        margin-top: 1;
    }
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        """Lay out the warning and the two choices."""
        with Vertical(id="developer-warning-dialog"):
            yield Static(self._message(), id="developer-warning-message")
            yield Button("I understand, use Developer", id="developer-warning-accept", variant="warning")
            yield Button("Stay on Advanced", id="developer-warning-cancel", variant="primary")

    @staticmethod
    def _message() -> Text:
        """Build the warning body."""
        return Text.assemble(
            ("Developer level exposes settings that can stop a working worker\n\n", "bold"),
            (
                "It adds forensic detail and uncommon tuning controls. Several of them will interrupt "
                "generation, evict loaded models, or put the worker into a state the horde treats as "
                "unhealthy if set carelessly.\n\n"
                "You only see this notice once.",
                "grey70",
            ),
        )

    def action_cancel(self) -> None:
        """Escape declines the level change."""
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Accept or decline the level change."""
        if event.button.id == "developer-warning-accept":
            self.dismiss(True)
        elif event.button.id == "developer-warning-cancel":
            self.dismiss(False)


class HelpModal(ResponsiveModalScreen[None]):
    """Explain the dashboard at the level currently in effect."""

    DEFAULT_CSS = """
    HelpModal {
        align: center middle;
    }
    HelpModal #help-dialog {
        width: 76;
        max-width: 95%;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    HelpModal #help-dialog Button {
        width: 100%;
        margin-top: 1;
    }
    """

    BINDINGS = [("escape", "close", "Close"), ("question_mark", "close", "Close")]

    def __init__(self, level: ExperienceLevel, shortcuts: Sequence[tuple[str, str, str]] = ()) -> None:
        """Render help for ``level``, listing every ``(keys, description, action)`` shortcut given."""
        super().__init__()
        self._level = level
        self._shortcuts = tuple(shortcuts)

    def compose(self) -> ComposeResult:
        """Lay out the level explanation, then the complete shortcut list."""
        with Vertical(id="help-dialog"):
            yield Static(self._message(self._level), id="help-message")
            yield Static(self._shortcut_table(self._shortcuts), id="help-shortcuts")
            yield Button("Close", id="help-close", variant="primary")

    @staticmethod
    def _message(level: ExperienceLevel) -> Text:
        """Build help text describing ``level`` and how to navigate."""
        name, summary = _LEVEL_SUMMARIES[level]
        return Text.assemble(
            (f"You are using the {name} experience\n\n", "bold"),
            (f"{summary}\n\n", "grey70"),
            ("Finding things\n", "bold"),
            (
                "Ctrl+P opens the command palette: every tab, every shortcut, and the experience levels "
                "are listed there by name. The bar at the bottom shows as many shortcuts as fit, so on a "
                "narrow terminal it shows only the first few; the palette and this list are always "
                "complete.\n\n",
                "grey70",
            ),
            ("Changing level\n", "bold"),
            ("The Dashboard section of the Config tab, or Ctrl+P.\n", "grey70"),
        )

    @staticmethod
    def _shortcut_table(shortcuts: Sequence[tuple[str, str, str]]) -> Table | Text:
        """Render every shortcut, so this list is complete regardless of terminal width."""
        if not shortcuts:
            return Text("")
        table = Table(expand=True, box=None, pad_edge=False)
        table.add_column("Key", style="bold cyan", no_wrap=True)
        table.add_column("Action")
        for keys, description, _action in shortcuts:
            table.add_row(keys, description)
        return table

    def action_close(self) -> None:
        """Dismiss the help."""
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dismiss the help."""
        if event.button.id == "help-close":
            self.dismiss(None)


__all__ = [
    "DashboardPreferencesView",
    "DeveloperWarningModal",
    "ExperienceIntroductionModal",
    "HelpModal",
]
