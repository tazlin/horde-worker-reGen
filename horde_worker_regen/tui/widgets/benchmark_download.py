"""A modal that explains, then performs, the model downloads a benchmark needs.

The whole point is to answer a non-technical operator's questions *before* anything happens: which models
does this benchmark need, how big are they, do I already have them, where will the new ones be stored, and
do I have room? Only then does pressing Download fetch the missing checkpoints, with live per-model progress.

It drives the ``horde-benchmark download`` subcommand (the single executor shared with the CLI): once with
``--dry-run`` to compute the plan to show, and once for real to fetch. Running it out-of-process keeps the
heavy inference-stack import off the TUI process and reuses one code path for both surfaces.
"""

from __future__ import annotations

import subprocess

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from horde_worker_regen.benchmark.download_progress import (
    DownloadEvent,
    DownloadModelRow,
    decode_download_events,
)
from horde_worker_regen.tui.benchmark_launcher import BenchmarkOptions

_PLAN_TIMEOUT_SECONDS = 240.0
"""Cap on the dry-run plan subprocess: it imports the inference stack and probes the GPU (slow, cold)."""


def _gb(num_bytes: int | None) -> str:
    """Render a byte count as ``N.N GB``, or ``?`` when unknown."""
    if num_bytes is None:
        return "?"
    return f"{num_bytes / 1024**3:.1f} GB"


class BenchmarkDownloadModal(ModalScreen[bool]):
    """Show exactly which models a benchmark needs (size, on-disk, destination), then download the missing ones.

    Dismisses with ``True`` when at least one model was downloaded (so the caller can refresh its plan
    preview), otherwise ``False``.
    """

    DEFAULT_CSS = """
    BenchmarkDownloadModal {
        align: center middle;
    }
    BenchmarkDownloadModal #download-dialog {
        width: 90%;
        height: 90%;
        padding: 1 2;
        border: thick $accent;
        background: $surface;
    }
    BenchmarkDownloadModal #download-intro {
        height: auto;
        margin-bottom: 1;
    }
    BenchmarkDownloadModal #download-body-scroll {
        height: 1fr;
        border: round $panel;
        padding: 0 1;
        margin-bottom: 1;
    }
    BenchmarkDownloadModal #download-actions {
        height: 3;
    }
    BenchmarkDownloadModal #download-actions Button {
        margin-right: 1;
    }
    """

    BINDINGS = [("escape", "dismiss_modal", "Close")]

    def __init__(self, options: BenchmarkOptions) -> None:
        """Store the benchmark options whose models this modal will plan and download."""
        super().__init__()
        self._options = options
        self._plan: DownloadEvent | None = None
        self._downloaded_any = False
        self._progress: dict[str, str] = {}
        """Per-model status (``downloading`` / ``done`` / ``failed``) accumulated during a download run."""
        self._busy = False

    def compose(self) -> ComposeResult:
        """Lay out the intro line, the scrollable plan/progress body, and the action buttons."""
        with Vertical(id="download-dialog"):
            yield Static(
                Text(
                    "These are the models this benchmark needs. Models already on your machine are reused; "
                    "only the missing ones are downloaded, once, before the timed run.",
                    style="bold",
                ),
                id="download-intro",
            )
            with VerticalScroll(id="download-body-scroll"):
                yield Static(id="download-body")
            with Horizontal(id="download-actions"):
                yield Button("Download missing models", id="download-start", variant="success", disabled=True)
                yield Button("Close", id="download-close", variant="warning")

    def on_mount(self) -> None:
        """Kick off the (slow, cold) dry-run plan computation and show a working state meanwhile."""
        self._set_body(
            Text(
                "Working out which models are needed (this starts no benchmark and downloads nothing; "
                "it can take up to a few minutes the first time while the hardware is detected)…",
                style="yellow",
            ),
        )
        self.run_worker(self._compute_plan, thread=True, exclusive=True, group="bench-dl-plan")

    def _compute_plan(self) -> None:
        """(Worker thread) run ``download --dry-run`` and hand the parsed plan back to the UI thread."""
        try:
            result = subprocess.run(
                self._options.build_download_command(dry_run=True),
                capture_output=True,
                text=True,
                timeout=_PLAN_TIMEOUT_SECONDS,
                check=False,
            )
        except Exception as e:  # noqa: BLE001 - best-effort preview; surface it, never crash the TUI
            self.app.call_from_thread(self._render_error, f"{type(e).__name__}: {e}")
            return
        events = decode_download_events(result.stdout)
        planned = next((event for event in events if event.kind == "planned"), None)
        if planned is None:
            tail = (result.stderr or result.stdout or "no output").strip().splitlines()
            self.app.call_from_thread(self._render_error, tail[-1] if tail else "no plan returned")
            return
        self.app.call_from_thread(self._render_plan, planned)

    def _render_plan(self, plan: DownloadEvent) -> None:
        """(UI thread) render the per-model plan and enable Download when there is something that fits."""
        self._plan = plan
        missing = [model for model in plan.models if not model.on_disk]
        self._set_body(Group(_plan_table(plan.models), Text(""), _plan_footer(plan)))

        start = self.query_one("#download-start", Button)
        if not missing:
            start.disabled = True
            start.label = "Nothing to download"
        elif not plan.fits:
            start.disabled = True
            start.label = "Not enough disk space"
        else:
            start.disabled = False
            start.label = f"Download {len(missing)} model(s) ({_gb(plan.to_download_bytes)})"

    def _render_error(self, message: str) -> None:
        """(UI thread) explain why the plan could not be computed."""
        self._set_body(Text(f"Could not work out the download plan: {message}", style="red"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route the action buttons to start a download or close the modal."""
        if event.button.id == "download-start":
            self._start_download()
        elif event.button.id == "download-close":
            self.dismiss(self._downloaded_any)

    def action_dismiss_modal(self) -> None:
        """Close the modal (Escape), reporting whether anything was downloaded."""
        if not self._busy:
            self.dismiss(self._downloaded_any)

    def _start_download(self) -> None:
        """Begin fetching the missing models, streaming live per-model progress into the body."""
        if self._plan is None or self._busy:
            return
        self._busy = True
        self._progress = {model.name: "downloading" for model in self._plan.models if not model.on_disk}
        self.query_one("#download-start", Button).disabled = True
        self.query_one("#download-close", Button).disabled = True
        self._render_progress(header="Downloading models…")
        self.run_worker(self._run_download, thread=True, exclusive=True, group="bench-dl-run")

    def _run_download(self) -> None:
        """(Worker thread) stream the real download subprocess, parsing progress events line by line."""
        try:
            process = subprocess.Popen(  # noqa: S603 - argv is built from our own options, not user input
                self._options.build_download_command(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:  # noqa: BLE001 - surface the failure rather than crashing the TUI
            self.app.call_from_thread(self._finish_download, f"{type(e).__name__}: {e}")
            return

        assert process.stdout is not None
        for line in process.stdout:
            for download_event in decode_download_events(line):
                self.app.call_from_thread(self._apply_event, download_event)
        process.wait()
        self.app.call_from_thread(self._finish_download, None if process.returncode == 0 else "see worker log")

    def _apply_event(self, event: DownloadEvent) -> None:
        """(UI thread) fold one streamed progress event into the per-model status and re-render."""
        if event.kind == "model_started":
            self._progress[event.name] = "downloading"
        elif event.kind == "model_finished":
            self._progress[event.name] = "done" if event.ok else "failed"
            if event.ok:
                self._downloaded_any = True
        self._render_progress(header="Downloading models…")

    def _finish_download(self, error: str | None) -> None:
        """(UI thread) render the final summary and re-enable closing the modal."""
        self._busy = False
        self.query_one("#download-close", Button).disabled = False
        failed = sum(1 for status in self._progress.values() if status == "failed")
        done = sum(1 for status in self._progress.values() if status == "done")
        if error is not None:
            header = Text(f"Download stopped: {error}", style="red")
        elif failed:
            header = Text(f"Finished with problems: {done} downloaded, {failed} failed.", style="yellow")
        else:
            header = Text(
                f"Done: {done} model(s) downloaded. You can close this and run the benchmark.",
                style="green",
            )
        self._render_progress(header=header)

    def _render_progress(self, *, header: RenderableType) -> None:
        """(UI thread) render the header plus a per-model status table from the accumulated progress."""
        table = Table(expand=True)
        table.add_column("Model")
        table.add_column("Status")
        for name, status in self._progress.items():
            table.add_row(name, _status_text(status))
        body: RenderableType = Group(header, Text(""), table) if self._progress else header
        self._set_body(body)

    def _set_body(self, renderable: RenderableType) -> None:
        """Replace the scrollable body's contents."""
        self.query_one("#download-body", Static).update(renderable)


def _plan_table(models: list[DownloadModelRow]) -> Table:
    """Render every needed model with its size, on-disk state, and where it lives or will be written."""
    table = Table(expand=True)
    table.add_column("Model")
    table.add_column("Size", justify="right")
    table.add_column("Status")
    table.add_column("Location")
    for model in models:
        status = Text("on disk", style="green") if model.on_disk else Text("will download", style="yellow")
        table.add_row(
            model.name,
            _gb(model.size_bytes),
            status,
            model.target_path or "(destination undetermined)",
        )
    return table


def _plan_footer(plan: DownloadEvent) -> Text:
    """Render the disk budget: already present, still to download, free space, and any shortfall."""
    free = "unknown" if plan.free_disk_bytes is None else _gb(plan.free_disk_bytes)
    footer = Text()
    footer.append(f"Already on disk: {_gb(plan.present_bytes)}", style="green")
    footer.append("    ")
    footer.append(f"To download: {_gb(plan.to_download_bytes)}", style="yellow")
    footer.append("    ")
    footer.append(f"Free on model volume: {free}")
    if not plan.fits:
        footer.append("\n")
        footer.append(
            f"Not enough space: about {_gb(plan.shortfall_bytes)} short. Free up disk or choose fewer tiers.",
            style="bold red",
        )
    return footer


def _status_text(status: str) -> Text:
    """Colour a per-model download status."""
    if status == "done":
        return Text("done", style="green")
    if status == "failed":
        return Text("failed", style="red")
    return Text("downloading…", style="yellow")


__all__ = ["BenchmarkDownloadModal"]
