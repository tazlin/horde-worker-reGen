"""A modal that explains the model downloads a benchmark needs, then hands them to the download subsystem.

The whole point is to answer a non-technical operator's questions *before* anything happens: which models
does this benchmark need, how big are they, do I already have them, where will the new ones be stored, and do
I have room? Confirming does not download anything here: it delegates the missing models to the worker's own
download orchestration (the same path the Downloads tab uses), so there is one downloader, one progress
surface, and no second process contending for bandwidth or files. Progress is then tracked on the Downloads
tab and the Benchmark tab's waiting banner.

It drives the ``horde-benchmark download --dry-run`` subcommand once to compute the plan to show; the real
fetch is the worker's job. Running the plan out-of-process keeps the heavy inference-stack import off the TUI
process and reuses one code path with the CLI.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from collections.abc import Callable

    from horde_worker_regen.tui.benchmark_launcher import BenchmarkOptions

_PLAN_TIMEOUT_SECONDS = 240.0
"""Cap on the dry-run plan subprocess: it imports the inference stack and probes the GPU (slow, cold)."""

_STATUS_PRESENT = "present"
_STATUS_DOWNLOADING = "downloading"
_STATUS_TO_DOWNLOAD = "to_download"


@dataclass(frozen=True)
class DownloadLiveState:
    """Represents a running worker's authoritative view of which models are on disk versus fetching now.

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
    """Return a byte count rendered as ``N.N GB``, or ``?`` when unknown."""
    if num_bytes is None:
        return "?"
    return f"{num_bytes / 1024**3:.1f} GB"


class BenchmarkDownloadModal(ModalScreen[bool]):
    """Show exactly which models a benchmark needs (size, on-disk, destination), then request the missing ones.

    Confirming delegates the missing models to the worker's download orchestration rather than downloading
    here, so there is a single downloader and progress surface. Dismisses with ``True`` when a download was
    requested (so the caller can refresh its plan preview and enter its waiting state), otherwise ``False``.
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

    def __init__(
        self,
        options: BenchmarkOptions,
        *,
        delegate: Callable[[list[str]], bool],
        live_state: Callable[[], DownloadLiveState | None] | None = None,
    ) -> None:
        """Store the benchmark options to plan, the download delegate, and an optional live-worker state reader.

        Args:
            options: The benchmark selection whose model needs are planned and (on confirm) requested.
            delegate: Hands the missing model names to the download orchestration and returns whether the
                request was accepted. Always provided: a live worker fetches them in the background while it
                keeps serving; a stopped worker is started into a download-only hold to fetch them, GPU idle.
            live_state: When a worker is live, returns its authoritative present/in-flight model set so the
                plan reflects reality: a model the worker is fetching shows as downloading, not "ready" or
                "to download", and is excluded from the missing set the confirm requests. Read lazily on each
                render so it tracks the latest snapshot; it must never raise (a flaky read falls back to the
                disk scan).
        """
        super().__init__()
        self._options = options
        self._delegate = delegate
        self._live_state = live_state
        self._plan: DownloadEvent | None = None
        self._requested_download = False

    def compose(self) -> ComposeResult:
        """Lay out the intro line, the scrollable plan body, and the request/close buttons."""
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
                yield Button("Request missing models", id="download-start", variant="success", disabled=True)
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
        """Run ``download --dry-run`` on a worker thread and hand the parsed plan back to the UI thread."""
        try:
            result = subprocess.run(
                self._options.build_download_command(dry_run=True),
                capture_output=True,
                text=True,
                timeout=_PLAN_TIMEOUT_SECONDS,
                check=False,
            )
        except Exception as plan_error:  # noqa: BLE001 - best-effort preview; surface it, never crash the TUI
            self.app.call_from_thread(self._render_error, f"{type(plan_error).__name__}: {plan_error}")
            return
        events = decode_download_events(result.stdout)
        planned = next((event for event in events if event.kind == "planned"), None)
        if planned is None:
            tail = (result.stderr or result.stdout or "no output").strip().splitlines()
            self.app.call_from_thread(self._render_error, tail[-1] if tail else "no plan returned")
            return
        self.app.call_from_thread(self._render_plan, planned)

    def _resolve_live_state(self) -> DownloadLiveState | None:
        """Return the live-worker state for this render, or None (no worker / read failed: fall back to scan)."""
        if self._live_state is None:
            return None
        try:
            return self._live_state()
        except Exception:  # noqa: BLE001 - a flaky snapshot read must never break the plan view
            return None

    @staticmethod
    def _row_status(row: DownloadModelRow, live: DownloadLiveState | None) -> str:
        """Return a plan row classified present / downloading / to-download, a live worker overriding the scan.

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
        """Return the names a confirm would fetch: rows that are neither present nor already in flight."""
        if self._plan is None:
            return []
        live = self._resolve_live_state()
        return [row.name for row in self._plan.models if self._row_status(row, live) == _STATUS_TO_DOWNLOAD]

    def _missing_image_model_names(self) -> list[str]:
        """Return the missing *image* model names only -- the set requested as image models by name.

        Feature files (controlnet checkpoints, post-processors, annotators) are deliberately excluded: they are
        fetched through the download subsystem's aux pass (each via its own model manager), so requesting them
        by name as image models would route them to the image manager, which has no record of them, and fail.
        """
        if self._plan is None:
            return []
        live = self._resolve_live_state()
        return [
            row.name
            for row in self._plan.models
            if not row.is_aux and self._row_status(row, live) == _STATUS_TO_DOWNLOAD
        ]

    def _render_plan(self, plan: DownloadEvent) -> None:
        """Render the per-model plan on the UI thread and enable Request when there is something that fits.

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
            to_download_bytes = sum(row.size_bytes or 0 for row in missing)
            start.label = f"Request {len(missing)} model(s) ({_gb(to_download_bytes)})"

    def _render_error(self, message: str) -> None:
        """Explain why the plan could not be computed (UI thread)."""
        self._set_body(Text(f"Could not work out the download plan: {message}", style="red"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route the request button to delegate the download, or the close button to dismiss the modal."""
        if event.button.id == "download-start":
            self._request_download()
        elif event.button.id == "download-close":
            self.dismiss(self._requested_download)

    def action_dismiss_modal(self) -> None:
        """Close the modal (Escape), reporting whether a download was requested."""
        self.dismiss(self._requested_download)

    def _request_download(self) -> None:
        """Hand the missing models to the download orchestration and point the operator at the Downloads tab.

        Nothing downloads inside this modal: the worker's own download process fetches the models into the
        shared cache (a live worker keeps serving meanwhile; a stopped one is started GPU-idle for it), so
        there is no second downloader contending for bandwidth or files. The benchmark reuses them on disk
        when it later runs.
        """
        if self._plan is None:
            return
        missing_total = len(self._missing_model_names())
        if not missing_total:
            return
        # Request the missing *image* models by name; the delegate also enables the aux pass, which fetches the
        # missing feature files (controlnet checkpoints, post-processors, annotators) through their own managers.
        accepted = self._delegate(self._missing_image_model_names())
        start = self.query_one("#download-start", Button)
        if accepted:
            self._requested_download = True
            start.disabled = True
            self._set_body(
                Text(
                    f"Requested {missing_total} model(s) from the download subsystem. They download in the "
                    "background; track progress on the Downloads tab and the Benchmark tab's waiting banner. ",
                    style="green",
                ),
            )
        else:
            self._set_body(
                Text("Could not reach the download subsystem to request the download; try again.", style="red"),
            )

    def _set_body(self, renderable: RenderableType) -> None:
        """Replace the scrollable body's contents."""
        self.query_one("#download-body", Static).update(renderable)


def _plan_table(statuses: list[tuple[DownloadModelRow, str]]) -> Table:
    """Return a table of every needed model with its size, resolved status, and where it lives or will land."""
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
    """Return a coloured label for a resolved plan-row status (present / downloading / to-download)."""
    if status == _STATUS_PRESENT:
        return Text("on disk", style="green")
    if status == _STATUS_DOWNLOADING:
        return Text("downloading…", style="cyan")
    return Text("will download", style="yellow")


def _plan_footer(plan: DownloadEvent) -> Text:
    """Return the disk budget: already present, still to download, free space, and any shortfall."""
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


__all__ = ["BenchmarkDownloadModal", "DownloadLiveState"]
