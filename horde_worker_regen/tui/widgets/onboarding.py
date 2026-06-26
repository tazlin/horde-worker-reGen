"""A first-run modal that offers to benchmark the worker.

Shown by the app on launch when no current benchmark exists and the user has not stickily declined (see
[`should_prompt_onboarding`][horde_worker_regen.app_state.should_prompt_onboarding]). Dismisses with the
chosen [`OnboardingChoice`][horde_worker_regen.app_state.OnboardingChoice]; the app persists it and, for
``ACCEPTED``, enters the benchmark flow.
"""

from __future__ import annotations

import enum

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from horde_worker_regen.app_state import BenchmarkAvailability, OnboardingChoice


class WorkerStartChoice(enum.StrEnum):
    """The user's response to the first-run "start the worker?" prompt."""

    START_NOW = "start_now"
    """Start the worker for this session only."""
    START_AND_REMEMBER = "start_and_remember"
    """Start the worker now and auto-start it on every future launch."""
    DOWNLOAD_ONLY = "download_only"
    """Start only the download subsystem (download-only hold): fetch models without committing the GPU."""
    STAY_STOPPED = "stay_stopped"
    """Leave the worker stopped; the user can start it later (F3)."""


class WorkerStartModal(ModalScreen[WorkerStartChoice]):
    """A first-run modal asking whether to start the worker (it does real GPU work)."""

    DEFAULT_CSS = """
    WorkerStartModal {
        align: center middle;
    }
    WorkerStartModal #worker-start-dialog {
        width: 72;
        height: auto;
        padding: 1 2;
        border: thick $accent;
        background: $surface;
    }
    WorkerStartModal #worker-start-dialog Button {
        width: 100%;
        margin-top: 1;
    }
    """

    _BUTTON_CHOICES: dict[str, WorkerStartChoice] = {
        "worker-start-now": WorkerStartChoice.START_NOW,
        "worker-start-remember": WorkerStartChoice.START_AND_REMEMBER,
        "worker-start-download-only": WorkerStartChoice.DOWNLOAD_ONLY,
        "worker-start-stay-stopped": WorkerStartChoice.STAY_STOPPED,
    }

    def compose(self) -> ComposeResult:
        """Lay out the explanatory message and the choice buttons."""
        with Vertical(id="worker-start-dialog"):
            yield Static(self._message(), id="worker-start-message")
            yield Button("Start worker now", id="worker-start-now", variant="success")
            yield Button("Start & auto-start from now on", id="worker-start-remember", variant="primary")
            yield Button("Download models only (no GPU)", id="worker-start-download-only", variant="default")
            yield Button("Stay stopped", id="worker-start-stay-stopped", variant="default")

    @staticmethod
    def _message() -> Text:
        """Explain that the worker does real work and that starting is opt-in."""
        return Text.assemble(
            ("Start the worker?\n\n", "bold"),
            (
                "Starting the worker begins real work for the AI Horde: it spawns inference "
                "processes and uses your GPU. It does not start automatically. You can start or "
                "stop it any time with F3, and manage auto-start from the Control tab.",
                "grey70",
            ),
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dismiss with the choice the pressed button represents."""
        choice = self._BUTTON_CHOICES.get(event.button.id or "")
        if choice is not None:
            self.dismiss(choice)


class BenchmarkOnboardingModal(ModalScreen[OnboardingChoice]):
    """A modal offering to run a benchmark on first launch (or after a version bump)."""

    DEFAULT_CSS = """
    BenchmarkOnboardingModal {
        align: center middle;
    }
    BenchmarkOnboardingModal #onboarding-dialog {
        width: 72;
        height: auto;
        padding: 1 2;
        border: thick $accent;
        background: $surface;
    }
    BenchmarkOnboardingModal #onboarding-dialog Button {
        width: 100%;
        margin-top: 1;
    }
    """

    _BUTTON_CHOICES: dict[str, OnboardingChoice] = {
        "onboarding-run": OnboardingChoice.ACCEPTED,
        "onboarding-skip": OnboardingChoice.DEFERRED,
        "onboarding-decline": OnboardingChoice.DECLINED,
    }

    def __init__(self, availability: BenchmarkAvailability) -> None:
        """Store why the prompt is shown (no benchmark yet vs. a stale one) for the message."""
        super().__init__()
        self._availability = availability

    def compose(self) -> ComposeResult:
        """Lay out the explanatory message and the three choice buttons."""
        with Vertical(id="onboarding-dialog"):
            yield Static(self._message(), id="onboarding-message")
            yield Button("Run benchmark now", id="onboarding-run", variant="success")
            yield Button("Skip for now", id="onboarding-skip", variant="default")
            yield Button("Don't ask again", id="onboarding-decline", variant="warning")

    def _message(self) -> Text:
        """Explain what a benchmark does and why the prompt appeared."""
        if self._availability is BenchmarkAvailability.STALE:
            headline = "The worker version changed since the last benchmark."
        else:
            headline = "This worker has not been benchmarked yet."
        return Text.assemble(
            (headline + "\n\n", "bold"),
            (
                "A benchmark ramps the worker through safe difficulty levels and suggests a tuned "
                "bridgeData. It stops the worker while it runs (it needs the GPU).",
                "grey70",
            ),
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dismiss with the choice the pressed button represents."""
        choice = self._BUTTON_CHOICES.get(event.button.id or "")
        if choice is not None:
            self.dismiss(choice)
