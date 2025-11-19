"""Custom widgets for the Horde Worker TUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Label, ProgressBar, Static


class StatusPanel(Container):
    """A panel showing a labeled status value."""

    DEFAULT_CSS = """
    StatusPanel {
        height: auto;
        border: solid $primary;
        padding: 1;
        margin: 0 1;
    }

    StatusPanel > .label {
        color: $text-muted;
        text-style: bold;
    }

    StatusPanel > .value {
        color: $success;
        text-style: bold;
        padding: 0 0 0 2;
    }

    StatusPanel.warning > .value {
        color: $warning;
    }

    StatusPanel.error > .value {
        color: $error;
    }
    """

    def __init__(
        self,
        label: str,
        value: str = "",
        *,
        name: str | None = None,
        id: str | None = None,  # noqa: A002
        classes: str | None = None,
    ) -> None:
        """Initialize the status panel.

        Args:
            label: The label text
            value: The value text
            name: Widget name
            id: Widget ID
            classes: CSS classes
        """
        super().__init__(name=name, id=id, classes=classes)
        self._label = label
        self._value = value

    def compose(self) -> ComposeResult:
        """Compose the panel widgets."""
        yield Label(self._label, classes="label")
        yield Label(self._value, classes="value")

    def update_value(self, value: str, status: str = "normal") -> None:
        """Update the value and optionally the status color.

        Args:
            value: New value text
            status: Status level (normal, warning, error)
        """
        self.remove_class("warning", "error")
        if status == "warning":
            self.add_class("warning")
        elif status == "error":
            self.add_class("error")

        value_widget = self.query_one(".value", Label)
        value_widget.update(value)


class MetricDisplay(Static):
    """A simple metric display with label and value."""

    DEFAULT_CSS = """
    MetricDisplay {
        height: 1;
        width: auto;
    }

    MetricDisplay .metric-label {
        color: $text-muted;
    }

    MetricDisplay .metric-value {
        color: $accent;
        text-style: bold;
    }
    """

    def __init__(
        self,
        label: str,
        value: str,
        *,
        name: str | None = None,
        id: str | None = None,  # noqa: A002
        classes: str | None = None,
    ) -> None:
        """Initialize the metric display.

        Args:
            label: Metric label
            value: Metric value
            name: Widget name
            id: Widget ID
            classes: CSS classes
        """
        super().__init__(name=name, id=id, classes=classes)
        self._label = label
        self._value = value

    def render(self) -> str:
        """Render the metric."""
        return f"[dim]{self._label}:[/dim] [{self.styles.color}]{self._value}[/]"

    def update_metric(self, value: str) -> None:
        """Update the metric value.

        Args:
            value: New value
        """
        self._value = value
        self.refresh()


class ProcessCard(Container):
    """A card displaying information about a single process."""

    DEFAULT_CSS = """
    ProcessCard {
        height: auto;
        border: round $primary;
        padding: 1;
        margin: 1;
    }

    ProcessCard.safety {
        border: round $warning;
    }

    ProcessCard .process-header {
        text-style: bold;
        color: $accent;
    }

    ProcessCard .process-state {
        color: $success;
        padding: 0 0 1 0;
    }

    ProcessCard .process-detail {
        color: $text-muted;
        padding: 0 0 0 2;
    }

    ProcessCard ProgressBar {
        margin: 1 0;
    }
    """

    def __init__(
        self,
        process_id: int,
        *,
        name: str | None = None,
        id: str | None = None,  # noqa: A002
        classes: str | None = None,
    ) -> None:
        """Initialize the process card.

        Args:
            process_id: The process ID
            name: Widget name
            id: Widget ID
            classes: CSS classes
        """
        super().__init__(name=name, id=id, classes=classes)
        self.process_id = process_id

    def compose(self) -> ComposeResult:
        """Compose the card widgets."""
        yield Label(f"Process {self.process_id}", classes="process-header")
        yield Label("State: UNKNOWN", id=f"state-{self.process_id}", classes="process-state")
        yield Label("Model: None", id=f"model-{self.process_id}", classes="process-detail")
        yield ProgressBar(total=100, id=f"progress-{self.process_id}", show_eta=False)
        yield Label("RAM: N/A | VRAM: N/A", id=f"memory-{self.process_id}", classes="process-detail")

    def update_process(self, process_data: dict) -> None:
        """Update the process card with new data.

        Args:
            process_data: Dictionary containing process information
        """
        # Update safety class
        if process_data.get("is_safety_process"):
            self.add_class("safety")
        else:
            self.remove_class("safety")

        # Update state
        state_label = self.query_one(f"#state-{self.process_id}", Label)
        state = process_data.get("state", "UNKNOWN")
        state_label.update(f"State: {state}")

        # Update model
        model_label = self.query_one(f"#model-{self.process_id}", Label)
        model = process_data.get("model", "None")
        model_state = process_data.get("model_state", "NONE")
        model_label.update(f"Model: {model} ({model_state})")

        # Update progress
        progress_bar = self.query_one(f"#progress-{self.process_id}", ProgressBar)
        progress = process_data.get("progress", 0)
        progress_bar.update(progress=progress)

        # Update memory
        memory_label = self.query_one(f"#memory-{self.process_id}", Label)
        ram = process_data.get("ram_usage", "N/A")
        vram = process_data.get("vram_usage", "N/A")
        job_id = process_data.get("job_id")
        job_text = f" | Job: {job_id[:8]}..." if job_id else ""
        memory_label.update(f"RAM: {ram} | VRAM: {vram}{job_text}")


class StatBox(Container):
    """A box displaying a statistic with label and value."""

    DEFAULT_CSS = """
    StatBox {
        width: 1fr;
        height: 5;
        border: solid $primary;
        padding: 1;
        content-align: center middle;
    }

    StatBox .stat-value {
        text-style: bold;
        color: $accent;
        width: 100%;
        text-align: center;
    }

    StatBox .stat-label {
        color: $text-muted;
        width: 100%;
        text-align: center;
    }
    """

    def __init__(
        self,
        label: str,
        value: str,
        *,
        name: str | None = None,
        id: str | None = None,  # noqa: A002
        classes: str | None = None,
    ) -> None:
        """Initialize the stat box.

        Args:
            label: Stat label
            value: Stat value
            name: Widget name
            id: Widget ID
            classes: CSS classes
        """
        super().__init__(name=name, id=id, classes=classes)
        self._label = label
        self._value = value

    def compose(self) -> ComposeResult:
        """Compose the stat box."""
        yield Label(self._value, classes="stat-value")
        yield Label(self._label, classes="stat-label")

    def update_stat(self, value: str) -> None:
        """Update the stat value.

        Args:
            value: New value
        """
        value_label = self.query_one(".stat-value", Label)
        value_label.update(value)
