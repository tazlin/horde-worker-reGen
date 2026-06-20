"""The live view: one panel per child process with step progress, throughput, and memory."""

from __future__ import annotations

import time

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from horde_worker_regen.process_management.supervisor_channel import ProcessSnapshot, WorkerStateSnapshot
from horde_worker_regen.tui.formatters import (
    STATE_LABELS,
    format_its,
    human_mb,
    job_id_text,
    label_state,
    shorten,
)

_BAR_WIDTH = 36

_ACTIVE_STATES = frozenset(
    {
        "INFERENCE_STARTING",
        "INFERENCE_POST_PROCESSING",
        "ALCHEMY_STARTING",
        "PRELOADING_MODEL",
        "DOWNLOADING_MODEL",
        "DOWNLOADING_AUX_MODEL",
        "JOB_RECEIVED",
        "EVALUATING_SAFETY",
    },
)
# Only these states have a live, meaningful sampling step/it-s; the snapshot may still carry the last
# job's numbers, so the panel renders the progress row only while the process is genuinely sampling.
_SAMPLING_STATES = frozenset({"INFERENCE_STARTING", "INFERENCE_POST_PROCESSING", "ALCHEMY_STARTING"})
_FAILED_STATES = frozenset({"INFERENCE_FAILED", "ALCHEMY_FAILED", "SAFETY_FAILED", "PROCESS_ENDED"})
_EXPECTED_QUIET_STATES = frozenset(
    {
        "PROCESS_STARTING",
        "PRELOADING_MODEL",
        "DOWNLOADING_MODEL",
        "DOWNLOADING_AUX_MODEL",
    },
)
"""States whose child-side work can block without emitting per-process heartbeats."""

_STALE_AFTER_SECONDS = 4.0
"""Beyond this snapshot age the live view is no longer trustworthy; it dims and flags the panels."""


def _progress_bar(fraction: float) -> Text:
    """Render a unicode progress bar coloured by fill fraction."""
    fraction = max(0.0, min(fraction, 1.0))
    filled = int(round(fraction * _BAR_WIDTH))
    colour = "green" if fraction >= 0.999 else "cyan"
    return Text.assemble(
        ("█" * filled, colour),
        ("░" * (_BAR_WIDTH - filled), "grey37"),
        (f" {fraction * 100:5.1f}%", "bold"),
    )


class LiveView(VerticalScroll):
    """A scrollable column of per-process panels, refreshed from each snapshot."""

    def compose(self) -> ComposeResult:
        """Hold a single Static that renders all process panels."""
        yield Static(id="live-body")

    def update_snapshot(
        self,
        snapshot: WorkerStateSnapshot,
        snapshot_age: float | None = None,
        *,
        detailed: bool = False,
    ) -> None:
        """Rebuild the process panels from a worker-state snapshot.

        ``snapshot_age`` (seconds since the snapshot was produced) drives the staleness banner: a frozen
        or dead worker keeps showing its last frame, so when the data is old we say so explicitly rather
        than let confident-but-stale numbers mislead the operator. ``detailed`` reveals the more
        technical rows (raw job ID, heartbeat age/type) that the F6 toggle gates.
        """
        body = self.query_one("#live-body", Static)
        if not snapshot.processes:
            body.update(Text("Waiting for the first worker snapshot…", style="italic grey62"))
            return

        stale = snapshot_age is not None and snapshot_age > _STALE_AFTER_SECONDS
        panels = [
            self._render_process_panel(process, stale=stale, detailed=detailed) for process in snapshot.processes
        ]
        if stale:
            banner = Text(
                f"⚠ Live data is {snapshot_age:.0f}s old; the worker may be busy, hung, or restarting.",
                style="bold yellow",
            )
            body.update(Group(banner, Text(""), *panels))
        else:
            body.update(Group(*panels))

    def _render_process_panel(
        self,
        process: ProcessSnapshot,
        *,
        stale: bool = False,
        detailed: bool = False,
    ) -> RenderableType:
        """Render one process as a bordered panel with progress and resource detail."""
        state_colour = "grey50" if stale else self._state_colour(process.last_process_state)
        heartbeat_age = time.time() - process.last_heartbeat_timestamp if process.last_heartbeat_timestamp else None

        body = Table.grid(padding=(0, 2))
        body.add_column(justify="right", style="bold cyan", no_wrap=True)
        body.add_column(ratio=1)

        state_label = label_state(process.last_process_state)
        body.add_row("State", Text(state_label, style=state_colour))
        body.add_row("Model", shorten(process.loaded_horde_model_name, 40))
        if process.loaded_horde_model_baseline:
            body.add_row("Baseline", process.loaded_horde_model_baseline)
        if process.current_job_width and process.current_job_height:
            size = f"{process.current_job_width}×{process.current_job_height}"
            if process.batch_amount > 1:
                size += f"   (batch ×{process.batch_amount})"
            body.add_row("Resolution", size)
        if detailed and process.current_job_id:
            # The first UUID group is colour-coded (matching the overview tables) so the same job is
            # recognisable at a glance across views; the remainder stays dim so the full id is still here.
            job_cell = job_id_text(process.current_job_id)
            remainder = process.current_job_id[len(job_cell.plain) :]
            if remainder:
                job_cell.append(remainder, style="grey50")
            body.add_row("Job", job_cell)

        if process.current_job_features is not None and not process.current_job_features.is_empty():
            body.add_row("Features", ", ".join(process.current_job_features.as_tags()))

        # Sampling progress is only meaningful while the process is actually sampling; otherwise the
        # step/it-s carried in the snapshot are last-job residue, so suppress the row when idle.
        if (
            process.last_process_state in _SAMPLING_STATES
            and process.last_current_step is not None
            and process.last_total_steps
        ):
            fraction = process.last_current_step / process.last_total_steps
            body.add_row(
                "Sampling",
                Text.assemble(
                    _progress_bar(fraction),
                    (f"  {process.last_current_step}/{process.last_total_steps} steps", "grey62"),
                ),
            )
            body.add_row("Throughput", format_its(process.last_iterations_per_second))
        elif process.last_process_state in _ACTIVE_STATES:
            # Show a stable placeholder row so the layout doesn't jump when sampling starts/stops.
            working_label = STATE_LABELS.get(process.last_process_state, "working")
            body.add_row("Working", Text(working_label + "…", style="yellow"))

        body.add_row(
            "GPU VRAM",
            f"{human_mb(process.vram_usage_mb)} / {human_mb(process.total_vram_mb)}"
            f"   (peak {human_mb(process.vram_used_high_water_mb)})",
        )
        body.add_row(
            "RAM",
            f"{human_mb(process.ram_usage_bytes / 1024 / 1024)}   (peak {human_mb(process.ram_used_high_water_mb)})",
        )
        if detailed:
            body.add_row(
                "Heartbeat",
                self._heartbeat_text(heartbeat_age, process.is_alive, process.last_process_state),
            )
            if process.is_busy:
                body.add_row("HB type", process.last_heartbeat_type.replace("_", " ").title())
        # A running tally so a healthy-but-quiet process (the safety process especially, whose checks
        # are each over in milliseconds) visibly does work rather than looking parked.
        work_label = "Checked" if process.process_type == "SAFETY" else "Completed"
        body.add_row(work_label, f"{process.num_jobs_completed:,} jobs")

        title = Text.assemble(
            (f" Process {process.process_id} ", "bold"),
            (f"· {process.process_type.title()} ", "grey62"),
        )
        return Panel(body, title=title, border_style=state_colour, title_align="left", padding=(0, 1))

    @staticmethod
    def _state_colour(state: str) -> str:
        """Map a process state to a panel/border colour."""
        if state in _ACTIVE_STATES:
            return "green"
        if state in _FAILED_STATES:
            return "red"
        if state == "WAITING_FOR_JOB":
            return "grey62"
        return "yellow"

    @staticmethod
    def _heartbeat_text(age: float | None, is_alive: bool, state: str) -> Text:
        """Render heartbeat freshness, coloured by staleness."""
        if not is_alive:
            return Text("process not alive", style="bold red")
        if age is None:
            return Text("-", style="grey62")
        if state in _EXPECTED_QUIET_STATES:
            return Text(f"working quietly for {age:.1f}s", style="grey70" if age < 30 else "yellow")
        if age < 5:
            colour = "green"
        elif age < 15:
            colour = "yellow"
        else:
            colour = "red"
        return Text(f"{age:.1f}s ago", style=colour)
