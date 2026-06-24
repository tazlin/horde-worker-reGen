"""A modal that explains, then performs, the model downloads a benchmark needs.

The whole point is to answer a non-technical operator's questions *before* anything happens: which models
does this benchmark need, how big are they, do I already have them, where will the new ones be stored, and
do I have room? Only then does pressing Download fetch the missing checkpoints, with live per-model progress.

It drives the ``horde-benchmark download`` subcommand (the single executor shared with the CLI): once with
``--dry-run`` to compute the plan to show, and once for real to fetch. Running it out-of-process keeps the
heavy inference-stack import off the TUI process and reuses one code path for both surfaces.
"""

from __future__ import annotations

import contextlib
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from horde_worker_regen.benchmark.download_progress import (
    DownloadControl,
    DownloadEvent,
    DownloadModelRow,
    decode_download_events,
    encode_download_control,
)
from horde_worker_regen.tui.benchmark_launcher import BenchmarkOptions
from horde_worker_regen.tui.formatters import human_bytes, human_duration
from horde_worker_regen.tui.widgets.downloads import DownloadsView

if TYPE_CHECKING:
    from collections.abc import Callable

_PLAN_TIMEOUT_SECONDS = 240.0
"""Cap on the dry-run plan subprocess: it imports the inference stack and probes the GPU (slow, cold)."""

_STATUS_PRESENT = "present"
_STATUS_DOWNLOADING = "downloading"
_STATUS_TO_DOWNLOAD = "to_download"


@dataclass(frozen=True)
class DownloadLiveState:
    """A running worker's authoritative view of which models are on disk versus being fetched right now.

    Lets the benchmark plan reflect what a live worker is actually doing instead of an independent disk scan.
    A model the worker reports present is shown present; one it is fetching shows as *downloading* (neither
    "ready" nor "to download") and is excluded from the missing set, so the operator is never told a
    mid-download model is ready, nor asked to re-fetch something already in flight.
    """

    present: frozenset[str] = frozenset()
    """Names the worker reports fully on disk."""
    in_flight: frozenset[str] = frozenset()
    """Names the worker is downloading or has queued to download."""


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
    BenchmarkDownloadModal #download-controls {
        height: 3;
    }
    BenchmarkDownloadModal #download-controls Button {
        margin-right: 1;
    }
    BenchmarkDownloadModal #download-rate {
        width: 26;
        margin-right: 1;
    }
    """

    BINDINGS = [("escape", "dismiss_modal", "Close")]

    def __init__(
        self,
        options: BenchmarkOptions,
        *,
        delegate: Callable[[list[str]], bool] | None = None,
        live_state: Callable[[], DownloadLiveState | None] | None = None,
    ) -> None:
        """Store the benchmark options to plan, an optional delegate, and an optional live-worker state reader.

        When ``delegate`` is set (a worker is live), confirming hands the missing models to the worker's
        download process instead of spawning a second, contending out-of-process downloader; the operator
        then tracks progress on the Downloads tab. When None, the modal self-downloads out-of-process.

        ``live_state`` (when a worker is live) returns the worker's authoritative present/in-flight model set
        so the plan reflects reality: a model the worker is fetching shows as downloading, not "ready" or "to
        download", and is excluded from the missing set the confirm would request. Read lazily on each render
        so it tracks the latest snapshot; it must never raise (a flaky read falls back to the disk scan).
        """
        super().__init__()
        self._options = options
        self._delegate = delegate
        self._live_state = live_state
        self._plan: DownloadEvent | None = None
        self._downloaded_any = False
        self._progress: dict[str, str] = {}
        """Per-model status (``downloading`` / ``done`` / ``failed``) accumulated during a download run."""
        self._busy = False
        self._paused = False
        self._process: subprocess.Popen[str] | None = None
        """The running download subprocess, kept so control commands can be written to its stdin."""
        self._current_progress: DownloadEvent | None = None
        """The latest ``model_progress`` event, rendered as a live progress bar for the current model."""

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
            with Horizontal(id="download-controls"):
                yield Button("Pause", id="download-pause", variant="primary")
                yield Input(placeholder="rate limit KB/s (0 = off)", id="download-rate", type="integer")
                yield Button("Apply limit", id="download-rate-apply")
            with Horizontal(id="download-actions"):
                yield Button("Download missing models", id="download-start", variant="success", disabled=True)
                yield Button("Close", id="download-close", variant="warning")

    def on_mount(self) -> None:
        """Kick off the (slow, cold) dry-run plan computation and show a working state meanwhile."""
        self.query_one("#download-controls").display = False
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

    def _resolve_live_state(self) -> DownloadLiveState | None:
        """Read the live-worker state for this render, or None (no worker / read failed: fall back to the scan)."""
        if self._live_state is None:
            return None
        try:
            return self._live_state()
        except Exception:  # noqa: BLE001 - a flaky snapshot read must never break the plan view
            return None

    @staticmethod
    def _row_status(row: DownloadModelRow, live: DownloadLiveState | None) -> str:
        """Classify a plan row as present / downloading / to-download, with a live worker overriding the scan.

        The live worker is authoritative: a model it reports present is present even if the offline scan
        disagreed, and a model it is fetching is *downloading* (so it is neither offered for download again
        nor mislabelled "ready"). With no live state, the row's own scanned ``on_disk`` decides.
        """
        if live is not None:
            if row.name in live.present:
                return _STATUS_PRESENT
            if row.name in live.in_flight:
                return _STATUS_DOWNLOADING
        return _STATUS_PRESENT if row.on_disk else _STATUS_TO_DOWNLOAD

    def _missing_model_names(self) -> list[str]:
        """The names a confirm would actually fetch: rows that are neither present nor already in flight."""
        if self._plan is None:
            return []
        live = self._resolve_live_state()
        return [row.name for row in self._plan.models if self._row_status(row, live) == _STATUS_TO_DOWNLOAD]

    def _render_plan(self, plan: DownloadEvent) -> None:
        """(UI thread) render the per-model plan and enable Download when there is something that fits.

        When a worker is live its snapshot overlays the offline scan, so an in-flight model reads as
        downloading and drops out of the missing set rather than appearing as a fresh fetch the operator
        could redundantly request.
        """
        self._plan = plan
        live = self._resolve_live_state()
        statuses = [(row, self._row_status(row, live)) for row in plan.models]
        missing = [row for row, status in statuses if status == _STATUS_TO_DOWNLOAD]
        in_flight = [row for row, status in statuses if status == _STATUS_DOWNLOADING]
        self._set_body(Group(_plan_table(statuses), Text(""), _plan_footer(plan)))

        start = self.query_one("#download-start", Button)
        if not missing:
            start.disabled = True
            # Distinguish "the worker is already fetching these" from "you genuinely have everything".
            start.label = "Worker is fetching these" if in_flight else "Nothing to download"
        elif not plan.fits:
            start.disabled = True
            start.label = "Not enough disk space"
        else:
            start.disabled = False
            verb = "Request" if self._delegate is not None else "Download"
            to_download_bytes = sum(row.size_bytes or 0 for row in missing)
            start.label = f"{verb} {len(missing)} model(s) ({_gb(to_download_bytes)})"

    def _render_error(self, message: str) -> None:
        """(UI thread) explain why the plan could not be computed."""
        self._set_body(Text(f"Could not work out the download plan: {message}", style="red"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route the action buttons to start a download, toggle controls, or close the modal."""
        if event.button.id == "download-start":
            self._start_download()
        elif event.button.id == "download-close":
            self.dismiss(self._downloaded_any)
        elif event.button.id == "download-pause":
            self._toggle_pause()
        elif event.button.id == "download-rate-apply":
            self._apply_rate_limit()

    def _toggle_pause(self) -> None:
        """Pause or resume the running download by writing a control line to its stdin."""
        self._paused = not self._paused
        self._write_control(DownloadControl(cmd="pause" if self._paused else "resume"))
        self.query_one("#download-pause", Button).label = "Resume" if self._paused else "Pause"

    def _apply_rate_limit(self) -> None:
        """Apply the rate-limit field to the running download (0 or blank clears the cap)."""
        raw = self.query_one("#download-rate", Input).value.strip()
        try:
            kbps = max(int(raw), 0) if raw else 0
        except ValueError:
            return
        self._write_control(DownloadControl(cmd="rate", kbps=kbps))

    def _write_control(self, control: DownloadControl) -> None:
        """Write one control command to the download subprocess's stdin (a no-op if it is not running)."""
        process = self._process
        if process is None or process.stdin is None:
            return
        with contextlib.suppress(Exception):
            process.stdin.write(encode_download_control(control) + "\n")
            process.stdin.flush()

    def action_dismiss_modal(self) -> None:
        """Close the modal (Escape), reporting whether anything was downloaded."""
        if not self._busy:
            self.dismiss(self._downloaded_any)

    def _start_download(self) -> None:
        """Fetch the missing models: delegate to a running worker when one is live, else self-download."""
        if self._plan is None or self._busy:
            return
        missing = self._missing_model_names()
        if not missing:
            return
        if self._delegate is not None:
            self._delegate_to_worker(missing)
            return
        self._busy = True
        self._paused = False
        self._current_progress = None
        self._progress = dict.fromkeys(missing, "downloading")
        self.query_one("#download-start", Button).disabled = True
        self.query_one("#download-close", Button).disabled = True
        controls = self.query_one("#download-controls")
        controls.display = True
        self.query_one("#download-pause", Button).label = "Pause"
        self._render_progress(header="Downloading models…")
        self.run_worker(self._run_download, thread=True, exclusive=True, group="bench-dl-run")

    def _delegate_to_worker(self, missing: list[str]) -> None:
        """Hand the missing models to the running worker's download process and point at the Downloads tab.

        The live worker keeps serving (a download takes no GPU); it just adds these to its background
        download set, fetched into the same shared cache the benchmark reads, so no second downloader
        contends for bandwidth or the same files. The benchmark reuses them on disk when it later runs.
        """
        delegate = self._delegate
        if delegate is None:
            return
        sent = delegate(missing)
        if sent:
            self.query_one("#download-start", Button).disabled = True
            self._set_body(
                Text(
                    f"Requested {len(missing)} model(s) from the running worker. They download in the "
                    "background while the worker keeps serving; track progress on the Downloads tab, then "
                    "close this window. The benchmark reuses them once they finish.",
                    style="green",
                ),
            )
        else:
            self._set_body(Text("Could not reach the worker to delegate the download; try again.", style="red"))

    def _run_download(self) -> None:
        """(Worker thread) stream the real download subprocess, parsing progress events line by line."""
        try:
            process = subprocess.Popen(  # noqa: S603 - argv is built from our own options, not user input
                self._options.build_download_command(control_stdin=True),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:  # noqa: BLE001 - surface the failure rather than crashing the TUI
            self.app.call_from_thread(self._finish_download, f"{type(e).__name__}: {e}")
            return

        self._process = process
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
            self._current_progress = None
        elif event.kind == "model_progress":
            self._current_progress = event
        elif event.kind == "model_finished":
            self._progress[event.name] = "done" if event.ok else "failed"
            if event.ok:
                self._downloaded_any = True
            self._current_progress = None
        self._render_progress(header="Downloading models…")

    def _finish_download(self, error: str | None) -> None:
        """(UI thread) render the final summary and re-enable closing the modal."""
        self._busy = False
        self._current_progress = None
        with contextlib.suppress(Exception):
            self.query_one("#download-controls").display = False
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
        """(UI thread) render the header, a live bar for the current model, and the per-model status table."""
        table = Table(expand=True)
        table.add_column("Model")
        table.add_column("Status")
        for name, status in self._progress.items():
            table.add_row(name, _status_text(status))
        if not self._progress:
            self._set_body(header)
            return
        parts: list[RenderableType] = [header, Text("")]
        if self._current_progress is not None:
            parts.extend([self._progress_line(self._current_progress), Text("")])
        parts.append(table)
        self._set_body(Group(*parts))

    @staticmethod
    def _progress_line(event: DownloadEvent) -> Text:
        """Render a one-line progress bar (bar, sizes, speed, ETA) for the in-flight model."""
        percent = (event.downloaded_bytes / event.total_bytes * 100) if event.total_bytes else None
        bar = DownloadsView._progress_bar(percent)
        sizes = f"{human_bytes(event.downloaded_bytes)} / {human_bytes(event.total_bytes)}"
        speed = f"{human_bytes(event.speed_bps)}/s" if event.speed_bps else "-"
        eta = human_duration(event.eta_seconds) if event.eta_seconds is not None else "-"
        return Text.assemble(
            (f"{event.name}  ", "bold"),
            (bar, "green"),
            ("   ", ""),
            (sizes, "grey70"),
            ("   ⇣ ", "grey50"),
            (speed, "grey70"),
            ("   ETA ", "grey50"),
            (eta, "grey70"),
        )

    def _set_body(self, renderable: RenderableType) -> None:
        """Replace the scrollable body's contents."""
        self.query_one("#download-body", Static).update(renderable)


def _plan_table(statuses: list[tuple[DownloadModelRow, str]]) -> Table:
    """Render every needed model with its size, resolved status, and where it lives or will be written."""
    table = Table(expand=True)
    table.add_column("Model")
    table.add_column("Size", justify="right")
    table.add_column("Status")
    table.add_column("Location")
    for model, status in statuses:
        table.add_row(
            model.name,
            _gb(model.size_bytes),
            _plan_status_text(status),
            model.target_path or "(destination undetermined)",
        )
    return table


def _plan_status_text(status: str) -> Text:
    """Colour a resolved plan-row status (present / downloading / to-download)."""
    if status == _STATUS_PRESENT:
        return Text("on disk", style="green")
    if status == _STATUS_DOWNLOADING:
        return Text("downloading…", style="cyan")
    return Text("will download", style="yellow")


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


__all__ = ["BenchmarkDownloadModal", "DownloadLiveState"]
