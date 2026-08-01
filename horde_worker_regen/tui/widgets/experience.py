"""Widgets specific to the progressive experience levels.

The levels change how much implementation detail each destination renders; they never change which
destinations exist. Every tab is present at every level, so nothing here hides or reorders navigation.
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Rule, Select, Static

from horde_worker_regen.app_state import DisplayDensity, ExperienceLevel

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

    These are frontend preferences held in the durable app state, not worker configuration. It lives on
    the Config tab because that is where people look to change things, but it deliberately writes nothing
    to ``bridgeData.yaml`` and never marks the config form dirty: every raw widget event is stopped here
    and re-emitted as a typed message, so the surrounding editor's save tracking cannot observe it.
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
            yield Button("Simple", id="experience-simple")
            yield Button("Advanced", id="experience-advanced")
            yield Button("Developer", id="experience-developer")
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
        """Highlight ``level`` as active and describe what it shows."""
        self._level = level
        active = {
            ExperienceLevel.SIMPLE: "experience-simple",
            ExperienceLevel.ADVANCED: "experience-advanced",
            ExperienceLevel.DEVELOPER: "experience-developer",
        }
        for candidate, button_id in active.items():
            try:
                button = self.query_one(f"#{button_id}", Button)
            except Exception:  # noqa: BLE001 - the control may not be mounted yet
                continue
            button.variant = "primary" if candidate is level else "default"
        _name, summary = _LEVEL_SUMMARIES[level]
        try:
            self.query_one("#experience-summary", Static).update(summary)
        except Exception:  # noqa: BLE001 - the summary may not be mounted yet
            return

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Request a level change, keeping the press away from the surrounding config form."""
        requested = {
            "experience-simple": ExperienceLevel.SIMPLE,
            "experience-advanced": ExperienceLevel.ADVANCED,
            "experience-developer": ExperienceLevel.DEVELOPER,
        }.get(event.button.id or "")
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


class ExperienceIntroductionModal(ModalScreen[ExperienceLevel | None]):
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


class DeveloperWarningModal(ModalScreen[bool]):
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


class HelpModal(ModalScreen[None]):
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

    def __init__(self, level: ExperienceLevel) -> None:
        """Render help for ``level``."""
        super().__init__()
        self._level = level

    def compose(self) -> ComposeResult:
        """Lay out the level explanation and key bindings."""
        with Vertical(id="help-dialog"):
            yield Static(self._message(self._level), id="help-message")
            yield Button("Close", id="help-close", variant="primary")

    @staticmethod
    def _message(level: ExperienceLevel) -> Text:
        """Build help text describing ``level`` and the always-available keys."""
        name, summary = _LEVEL_SUMMARIES[level]
        return Text.assemble(
            (f"You are using the {name} experience\n\n", "bold"),
            (f"{summary}\n\n", "grey70"),
            ("Changing level\n", "bold"),
            ("Config tab, or Ctrl+P and search for the level you want.\n\n", "grey70"),
            ("Keys\n", "bold"),
            (
                "F3 start or stop the worker\n"
                "Ctrl+P command palette, including every tab by name\n"
                "?  this help\n"
                "F6 cycle how densely the Overview renders\n"
                "Ctrl+Q quit\n",
                "grey70",
            ),
        )

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
