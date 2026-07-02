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

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    ProcessSnapshot,
    SchedulingGovernanceSnapshot,
    WorkerStateSnapshot,
)
from horde_worker_regen.process_management.lifecycle.process_temperature import (
    ProcessTemperature,
    classify_process_temperature,
    temperature_phrase,
)
from horde_worker_regen.tui.formatters import (
    STATE_LABELS,
    format_its,
    human_bytes,
    human_duration,
    human_mb,
    is_low_fidelity,
    job_id_text,
    label_state,
    shorten,
    temperature_colour,
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
    """Render a progress bar coloured by fill fraction."""
    fraction = max(0.0, min(fraction, 1.0))
    filled = int(round(fraction * _BAR_WIDTH))
    colour = "green" if fraction >= 0.999 else "cyan"
    fill_char, empty_char = ("#", "-") if is_low_fidelity() else ("█", "░")
    return Text.assemble(
        (fill_char * filled, colour),
        (empty_char * (_BAR_WIDTH - filled), "grey37"),
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
        governance = self._render_governance_strip(snapshot.scheduling_governance, detailed=detailed)
        if not snapshot.processes:
            body.update(
                Group(governance, Text(""), Text("Waiting for the first worker snapshot…", style="italic grey62"))
            )
            return

        stale = snapshot_age is not None and snapshot_age > _STALE_AFTER_SECONDS
        pending_models = frozenset(entry.model for entry in snapshot.pending_jobs if entry.model)
        panels = [
            self._render_process_panel(process, stale=stale, detailed=detailed, pending_models=pending_models)
            for process in snapshot.processes
        ]
        content: list[RenderableType] = [governance, Text("")]
        # The per-role worker RAM breakdown lives here (the Overview keeps only the overall figure): this tab
        # already carries per-process RAM, so the role split is its natural, deeper home.
        ram_panel = self._render_ram_by_role(snapshot)
        if ram_panel is not None:
            content.extend((ram_panel, Text("")))
        content.extend(panels)
        if stale:
            banner = Text(
                f"⚠ Live data is {snapshot_age:.0f}s old; the worker may be busy, hung, or restarting.",
                style="bold yellow",
            )
            body.update(Group(banner, Text(""), *content))
        else:
            body.update(Group(*content))

    @staticmethod
    def _render_ram_by_role(snapshot: WorkerStateSnapshot) -> RenderableType | None:
        """Render the worker's resident-RAM split by role, or None when no memory sample has arrived yet.

        The overall system figure headlines the title; the rows break the worker's own footprint down by
        role (inference, safety, and so on) so a memory balloon can be traced to the responsible role rather
        than read as one opaque total.
        """
        wire = snapshot.system_memory
        if wire is None or wire.total_bytes <= 0:
            return None
        from horde_worker_regen.process_management.resources.system_memory import ROLE_LABELS

        summary = wire.to_summary()
        role_items = summary.nonzero_role_items()
        if not role_items:
            return None

        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold cyan", no_wrap=True)
        grid.add_column(justify="right", no_wrap=True)
        for role, value in role_items:
            grid.add_row(ROLE_LABELS.get(role, role), human_bytes(value))

        used_fraction = summary.used_fraction
        used_pct = f" ({used_fraction * 100:.0f}%)" if used_fraction is not None else ""
        title = (
            f"Worker RAM by role · {human_bytes(summary.worker_total_bytes)} of "
            f"{human_bytes(summary.used_bytes)} used{used_pct}"
        )
        return Panel(grid, title=title, title_align="left", border_style="grey37", padding=(0, 1))

    @staticmethod
    def _preload_decision_label(decision: str) -> str:
        """Humanize a preload-admission decision key for compact live-view text."""
        if not decision:
            return "no decision yet"
        return decision.replace("_", " ").title()

    @staticmethod
    def _preload_decision_style(decision: str) -> str:
        """Style preload decisions by whether they admitted, skipped, or deferred work."""
        if not decision:
            return "grey50"
        if decision in {"admit", "prestage", "terminal_admit", "already_loaded"}:
            return "green"
        if decision in {"next_job", "quarantined"}:
            return "grey62"
        return "yellow"

    @staticmethod
    def _render_governance_strip(governance: SchedulingGovernanceSnapshot, *, detailed: bool = False) -> Panel:
        """Render a compact scheduler-governance strip above the process cards."""
        ram = governance.ram
        preload = governance.preload
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="bold cyan", no_wrap=True)
        table.add_column(ratio=1)
        table.add_column(justify="right", style="bold cyan", no_wrap=True)
        table.add_column(ratio=1)

        if not ram.measured:
            ram_label = Text("pending", style="grey50")
            ram_detail = "waiting for governor tick"
        elif ram.under_pressure:
            ram_label = Text("pressure", style="red")
            ram_detail = ram.reason
        elif ram.pop_hold_active:
            ram_label = Text("holding", style="yellow")
            ram_detail = ram.reason or "pop hold active"
        else:
            ram_label = Text("ok", style="green")
            ram_detail = ram.reason or "above RAM floor"

        decision_style = LiveView._preload_decision_style(preload.decision)
        decision = LiveView._preload_decision_label(preload.decision)
        target = f"p{preload.process_id}" if preload.process_id is not None else "no target"
        age = f" · {human_duration(time.time() - preload.timestamp)} ago" if preload.timestamp else ""
        preload_detail = f"{shorten(preload.model, 30) if preload.model else '-'} · {target}{age}"
        table.add_row(
            "RAM",
            Text.assemble(ram_label, (f"  {ram_detail}", "grey70")),
            "Preload",
            Text.assemble((decision, decision_style), (f"  {preload_detail}", "grey70")),
        )

        if detailed:
            reclaim: list[str] = []
            if ram.draining_process_ids:
                reclaim.append("draining p" + ", p".join(str(pid) for pid in ram.draining_process_ids))
            if ram.shed_card_indices:
                reclaim.append("shed GPU " + ", ".join(str(index) for index in ram.shed_card_indices))
            intake = "pop hold on" if ram.pop_hold_active else "pop hold off"
            if ram.pop_pause_active:
                intake += f"; hard pause {human_duration(ram.pop_pause_remaining_seconds)}"
            gate = preload.reason or "-"
            table.add_row("Intake", Text(intake, style="grey62"), "Gate", Text(gate, style="grey62"))
            table.add_row("Reclaim", Text("; ".join(reclaim) if reclaim else "none active", style="grey62"), "", "")

        border = "yellow" if ram.under_pressure or ram.pop_hold_active or decision_style == "yellow" else "grey37"
        return Panel(table, title="Scheduling", title_align="left", border_style=border, padding=(0, 1))

    def _render_process_panel(
        self,
        process: ProcessSnapshot,
        *,
        stale: bool = False,
        detailed: bool = False,
        pending_models: frozenset[str] = frozenset(),
    ) -> RenderableType:
        """Render one process as a bordered panel with progress and resource detail."""
        temperature = classify_process_temperature(
            state=process.last_process_state,
            loaded_model=process.loaded_horde_model_name,
            pending_models=pending_models,
        )
        heartbeat_age = time.time() - process.last_heartbeat_timestamp if process.last_heartbeat_timestamp else None

        body = Table.grid(padding=(0, 2))
        body.add_column(justify="right", style="bold cyan", no_wrap=True)
        body.add_column(ratio=1)

        # A temperature-led state row so a primed slot reads as primed, not idle (see the overview table).
        if temperature == ProcessTemperature.DOWN:
            state_label = label_state(process.last_process_state)
            state_colour = "grey50" if stale else self._state_colour(process.last_process_state)
        else:
            phrase = temperature_phrase(temperature, process.last_process_state)
            state_label = f"{temperature.value.title()} · {phrase}"
            state_colour = "grey50" if stale else temperature_colour(temperature)
        body.add_row("State", Text(state_label, style=state_colour))
        # A process running an alchemy form holds no horde image model, so the Model row would read blank;
        # label it as alchemy work instead so the role is clear rather than looking idle/misconfigured.
        if process.last_process_state == "ALCHEMY_STARTING" and not process.loaded_horde_model_name:
            body.add_row("Work", Text("⚗ Alchemy", style="magenta"))
        else:
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
