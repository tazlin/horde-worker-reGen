"""The benchmark view: launch a ramp, watch it live, and apply its recommended config.

Renders from the [`BenchmarkRunState`][horde_worker_regen.tui.benchmark_launcher.BenchmarkRunState] the
supervisor accumulates by tailing the run's progress file. The widget itself owns no process: it posts
``RunRequested`` / ``CancelRequested`` / ``ApplyConfigRequested`` messages and lets the app coordinate the
GPU-exclusive worker/benchmark hand-off.
"""

from __future__ import annotations

import contextlib
import subprocess
import typing

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import Button, Collapsible, Input, Label, Static, Switch

from horde_worker_regen import __version__
from horde_worker_regen.app_state import (
    AppStateStore,
    BenchmarkAvailability,
    OverviewViewMode,
    benchmark_status_summary,
)
from horde_worker_regen.benchmark.enums import BenchAxis, BenchTier
from horde_worker_regen.benchmark.progress_channel import LevelPlanRow, decode_plan_rows
from horde_worker_regen.tui.benchmark_launcher import (
    BenchmarkOptions,
    BenchmarkRunState,
    BenchmarkSupervisorStatus,
    LevelState,
)

_OUTCOME_COLOURS: dict[str, str] = {
    "passed": "green",
    "failed": "red",
    "crashed": "red",
    "crashed_hang": "red",
    "skipped": "grey50",
}

_BASIS_STYLES: dict[str, str] = {
    "proven": "green",
    "disabled_failed": "red",
    "untested_skipped": "grey50",
    "not_in_ladder": "grey50",
    "capped_vram": "yellow",
    "capped_soak": "yellow",
}
"""Colour for each recommendation basis: proven is grounded (green), failed is a real negative (red),
untested is unknown (grey), and capped is a deliberate headroom/stability hold-back (yellow)."""

_PLAN_PREVIEW_TIMEOUT_SECONDS = 180.0
"""Cap on the `plan` subprocess: it imports the inference stack and probes the GPU (slow, cold)."""

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
"""Braille spinner frames, advanced once per app tick, so an in-progress benchmark always shows motion
even between the (1-2s apart) live-metric samples."""


class _TierToggle(typing.NamedTuple):
    """One selectable model tier as presented in the primary controls."""

    tier: BenchTier
    label: str
    help_text: str
    default: bool


_TIER_TOGGLES: tuple[_TierToggle, ...] = (
    _TierToggle(BenchTier.SD15, "SD 1.5", "Smallest and fastest; the recommended starting point.", True),
    _TierToggle(BenchTier.SDXL, "SDXL", "Larger SDXL checkpoints; needs more VRAM than SD 1.5.", True),
    _TierToggle(
        BenchTier.FLUX, "Flux", "Very large (17-20 GB download, 13-16 GB VRAM); auto-skips if it does not fit.", False
    ),
    _TierToggle(
        BenchTier.QWEN, "Qwen", "Very large beta model; needs the pending reference, auto-skips if absent.", False
    ),
)
"""The model tiers an operator can select. flux/qwen default off: they are large and opt-in, and the
ramp pre-flight auto-skips them when the machine cannot hold them."""


def _tier_switch_id(tier: BenchTier) -> str:
    """The widget id for a tier toggle (kept derivable so collection and layout cannot drift)."""
    return f"benchmark-tier-{tier.value}"


class _AxisToggle(typing.NamedTuple):
    """One individually selectable ramp axis as presented in the Advanced panel."""

    axis: BenchAxis
    label: str
    help_text: str


_AXIS_GROUPS: tuple[tuple[str, tuple[_AxisToggle, ...]], ...] = (
    (
        "Concurrency",
        (
            _AxisToggle(BenchAxis.QUEUE_SIZE, "Queue depth", "Preload the next job while one samples (queue_size)."),
            _AxisToggle(BenchAxis.THREADS, "Thread count", "Run two inference jobs at once (max_threads)."),
            _AxisToggle(BenchAxis.BATCH, "Batch size", "Sample several images per step (n_iter / max_batch)."),
        ),
    ),
    (
        "Features",
        (
            _AxisToggle(BenchAxis.HIRES_FIX, "Hires-fix", "A second, upscaled sampling pass."),
            _AxisToggle(
                BenchAxis.POST_PROCESSING, "Post-processing", "Upscalers and face-fixers on generated images."
            ),
            _AxisToggle(BenchAxis.CONTROLNET, "Controlnet", "Classic preprocessor controlnet (SD1.5)."),
            _AxisToggle(BenchAxis.QR_CODE, "QR-code controlnet", "The QR-code workflow (the SDXL controlnet path)."),
        ),
    ),
    (
        "Alchemy",
        (
            _AxisToggle(
                BenchAxis.ALCHEMY_CLIP, "Alchemy: CLIP lane", "Caption / interrogation / NSFW (safety process)."
            ),
            _AxisToggle(BenchAxis.ALCHEMY_GRAPH, "Alchemy: graph lane", "Upscale / face-fix / strip-background."),
            _AxisToggle(BenchAxis.ALCHEMY_CONCURRENT, "Alchemy: concurrent", "Alchemy forms alongside image jobs."),
        ),
    ),
)
"""The per-axis toggles shown in the Advanced panel, grouped by stage for a clear visual hierarchy.

Each axis is independently selectable: deselecting one drops only its levels (see
`LadderOptions.excluded_axes`), so an operator can benchmark, say, post-processing without controlnet."""


def _axis_switch_id(axis: BenchAxis) -> str:
    """The widget id for an axis toggle (kept derivable so collection and layout cannot drift)."""
    return f"benchmark-axis-{axis.value}"


class BenchmarkView(VerticalScroll):
    """Launch and monitor a benchmark ramp, then apply the recommended bridgeData."""

    DEFAULT_CSS = """
    BenchmarkView #benchmark-actions {
        height: 3;
        padding: 0 1;
    }
    BenchmarkView #benchmark-actions Button {
        margin-right: 1;
    }
    BenchmarkView .adv-group {
        text-style: bold;
        color: $text;
        padding: 1 1 0 1;
    }
    BenchmarkView .adv-section-help {
        padding: 0 1;
    }
    BenchmarkView .adv-row {
        height: 3;
    }
    BenchmarkView .adv-row .adv-label {
        content-align: left middle;
        height: 3;
        width: 18;
        padding: 0 1;
    }
    BenchmarkView .adv-row .adv-input {
        width: 12;
    }
    BenchmarkView .adv-row .adv-help {
        content-align: left middle;
        height: 3;
        padding: 0 1;
    }
    BenchmarkView .benchmark-body {
        padding: 1 1;
    }
    BenchmarkView #benchmark-setup {
        height: auto;
    }
    """

    class RunRequested(Message):
        """Posted when the user asks to start a benchmark (the app supplies the process mode)."""

        def __init__(self, options: BenchmarkOptions) -> None:
            """Carry the user-chosen ramp options."""
            super().__init__()
            self.options = options

    class CancelRequested(Message):
        """Posted when the user asks to cancel the running benchmark."""

    class ApplyConfigRequested(Message):
        """Posted when the user accepts the suggested bridgeData."""

    class RestoreKnownGoodRequested(Message):
        """Posted when the user asks to restore the last known-good configuration."""

    def __init__(self, *, worker_mode: str) -> None:
        """Store the worker's process mode, which decides the benchmark's default process mode."""
        super().__init__()
        self._worker_mode = worker_mode
        self._app_state_summary: Text = Text("Loading benchmark status…", style="grey70")
        self._has_known_good = False

    def compose(self) -> ComposeResult:
        """Lay out the guided steps, the primary controls, the collapsed advanced options, and the body.

        The order encodes the recommended path top-to-bottom: read the steps, set the tiers, optionally
        open Advanced, then act from the button bar (Preview plan leads, since it needs no GPU).
        """
        with Horizontal(id="benchmark-actions"):
            yield Button("Preview plan", id="benchmark-preview", variant="primary")
            yield Button("Download models", id="benchmark-download", variant="default")
            yield Button("Run benchmark", id="benchmark-run", variant="success")
            yield Button("History", id="benchmark-history", variant="default")
            yield Button("Cancel", id="benchmark-cancel", variant="warning")
            yield Button("Apply suggested config", id="benchmark-apply", variant="primary")
            yield Button("Restore last-known-good", id="benchmark-restore", variant="default")
        # The setup chrome (steps, tiers, advanced, persisted status) is wrapped so the thin density mode
        # can collapse it to just the live run / result body, leaving the action bar and result in view.
        with Vertical(id="benchmark-setup"):
            yield Static(self._guided_steps(), id="benchmark-steps", classes="benchmark-body")
            yield Label("Model tiers", classes="adv-group")
            yield Static(
                Text(
                    "Model families to benchmark, in order. sd15/sdxl are the common path; flux/qwen are very "
                    "large and auto-skip if they do not fit this machine.",
                    style="grey50",
                ),
                classes="adv-section-help",
            )
            for tier_toggle in _TIER_TOGGLES:
                yield self._switch_row(
                    tier_toggle.label,
                    _tier_switch_id(tier_toggle.tier),
                    default=tier_toggle.default,
                    help_text=tier_toggle.help_text,
                )
            with Collapsible(title="Advanced options", collapsed=True, id="benchmark-advanced"):
                yield self._number_row(
                    "Soak (min)",
                    "benchmark-soak",
                    "5",
                    "How long the post-ramp sustained-load soak runs.",
                )
                yield self._switch_row(
                    "Validate",
                    "benchmark-validate",
                    default=True,
                    help_text="Run the post-ramp soak that proves the suggested config holds under load.",
                )
                yield self._switch_row(
                    "Downloads",
                    "benchmark-downloads",
                    default=False,
                    help_text="Include the level that fetches a lora from CivitAI (needs network + a token).",
                )
                yield Static(
                    Text(
                        "Capabilities to measure. Each is separate: turn off any you do not run, and only its "
                        "levels are skipped.",
                        style="grey50",
                    ),
                    classes="adv-section-help",
                )
                for group_name, toggles in _AXIS_GROUPS:
                    yield Label(group_name, classes="adv-group")
                    for toggle in toggles:
                        yield self._switch_row(
                            toggle.label,
                            _axis_switch_id(toggle.axis),
                            default=True,
                            help_text=toggle.help_text,
                        )
                yield self._switch_row(
                    "Warm worker",
                    "benchmark-warm",
                    default=True,
                    help_text="Reuse one warm worker across levels (faster). Off isolates each level.",
                )
                yield self._switch_row(
                    "Force",
                    "benchmark-force",
                    default=False,
                    help_text="Attempt levels that do not fit this machine (insufficient VRAM/disk) or lack a token.",
                )
                yield self._switch_row(
                    "Verbose",
                    "benchmark-verbose",
                    default=False,
                    help_text="Write extra per-process detail to the run's console.log.",
                )
            yield Static(self._app_state_summary, id="benchmark-status", classes="benchmark-body")
        yield Static(id="benchmark-plan", classes="benchmark-body")
        yield Static(id="benchmark-body", classes="benchmark-body")

    @staticmethod
    def _switch_row(label: str, switch_id: str, *, default: bool, help_text: str) -> Horizontal:
        """One advanced-option row: a name, a switch, and a visible plain-language explanation."""
        return Horizontal(
            Label(label, classes="adv-label"),
            Switch(value=default, id=switch_id),
            Static(Text(help_text, style="grey50"), classes="adv-help"),
            classes="adv-row",
        )

    @staticmethod
    def _number_row(label: str, input_id: str, default: str, help_text: str) -> Horizontal:
        """One advanced-option row backed by a numeric input rather than a switch."""
        return Horizontal(
            Label(label, classes="adv-label"),
            Input(value=default, id=input_id, type="number", classes="adv-input"),
            Static(Text(help_text, style="grey50"), classes="adv-help"),
            classes="adv-row",
        )

    @staticmethod
    def _guided_steps() -> Panel:
        """The plan-first guided path shown above the controls, in plain language."""
        body = Text.assemble(
            ("New to benchmarking? Follow these steps:\n\n", "bold"),
            ("1. ", "bold cyan"),
            ("Preview plan", "bold"),
            (" - see what each level needs and what will run on this machine. No GPU, no risk.\n", "grey70"),
            ("2. ", "bold cyan"),
            ("Download models", "bold"),
            (
                " - fetch any checkpoints the plan says you are missing, so the timed run is not slowed by "
                "downloading mid-benchmark. Skip this if the plan shows nothing to download.\n",
                "grey70",
            ),
            ("3. ", "bold cyan"),
            ("Run benchmark", "bold"),
            (
                " - measures the worker (this stops the worker; it needs the GPU) and suggests a tuned config.\n",
                "grey70",
            ),
            ("4. ", "bold cyan"),
            ("Apply suggested config", "bold"),
            (" - write the recommendation into bridgeData.yaml.\n\n", "grey70"),
            ("Open ", "grey70"),
            ("Advanced options", "italic"),
            (" only if you want to narrow the run. ", "grey70"),
            ("History", "italic"),
            (" reviews and compares past runs.", "grey70"),
        )
        return Panel(body, title="How this works", title_align="left", border_style="cyan")

    def on_mount(self) -> None:
        """Load the persisted benchmark status and render the initial idle view."""
        self.refresh_app_state_summary()
        self.update_view(BenchmarkRunState(), BenchmarkSupervisorStatus.IDLE)

    def refresh_app_state_summary(self) -> None:
        """Re-read durable app state and update the persisted-status line."""
        self._app_state_summary = self._build_app_state_summary()
        with contextlib.suppress(NoMatches):
            self.query_one("#benchmark-status", Static).update(self._app_state_summary)

    def update_view(
        self,
        run_state: BenchmarkRunState,
        status: BenchmarkSupervisorStatus,
        *,
        frame: int = 0,
        mode: OverviewViewMode = OverviewViewMode.NORMAL,
    ) -> None:
        """Refresh the action buttons, the plan pane, and the body panel from the supervisor's latest state.

        ``frame`` is the app's monotonically increasing tick counter, used only to animate the spinner so
        a running level reads as live even when its metrics have not changed since the last sample.

        ``mode`` follows the shared F6 density contract: thin collapses the setup chrome (guided steps,
        tier toggles, advanced options, persisted status) so only the action bar and the live run / result
        remain; normal and detailed keep the full launch pad (detailed never shows less than normal).
        """
        with contextlib.suppress(NoMatches):
            self.query_one("#benchmark-setup", Vertical).display = mode is not OverviewViewMode.THIN
        self._update_buttons(status, run_state)
        self.query_one("#benchmark-body", Static).update(self._render_body(run_state, status, frame))
        # A run's RampPlanned event overrides any idle preview with the authoritative plan; an empty
        # plan (before RampPlanned) leaves a previously-previewed plan in place rather than clearing it.
        if run_state.plan_rows:
            self.query_one("#benchmark-plan", Static).update(self._plan_panel(run_state.plan_rows))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Translate the action buttons into messages the app coordinates."""
        if event.button.id == "benchmark-run":
            self.post_message(self.RunRequested(self._collect_options()))
        elif event.button.id == "benchmark-preview":
            self._start_plan_preview()
        elif event.button.id == "benchmark-download":
            self._open_download()
        elif event.button.id == "benchmark-history":
            self._open_history()
        elif event.button.id == "benchmark-cancel":
            self.post_message(self.CancelRequested())
        elif event.button.id == "benchmark-apply":
            self.post_message(self.ApplyConfigRequested())
        elif event.button.id == "benchmark-restore":
            self.post_message(self.RestoreKnownGoodRequested())

    def _collect_options(self) -> BenchmarkOptions:
        """Build ramp options from the option widgets (the app overrides the process mode)."""
        tiers = [
            toggle.tier.value
            for toggle in _TIER_TOGGLES
            if self.query_one(f"#{_tier_switch_id(toggle.tier)}", Switch).value
        ] or ["sd15"]
        try:
            soak_minutes = float(self.query_one("#benchmark-soak", Input).value or "5")
        except ValueError:
            soak_minutes = 5.0
        excluded_axes = [
            toggle.axis.value
            for _group_name, toggles in _AXIS_GROUPS
            for toggle in toggles
            if not self.query_one(f"#{_axis_switch_id(toggle.axis)}", Switch).value
        ]
        return BenchmarkOptions(
            tiers=tiers,
            process_mode=self._worker_mode,
            validate=self.query_one("#benchmark-validate", Switch).value,
            soak_minutes=soak_minutes,
            include_downloads=self.query_one("#benchmark-downloads", Switch).value,
            excluded_axes=excluded_axes,
            warm=self.query_one("#benchmark-warm", Switch).value,
            force=self.query_one("#benchmark-force", Switch).value,
            verbose=self.query_one("#benchmark-verbose", Switch).value,
        )

    def _open_history(self) -> None:
        """Open the past-runs history/compare modal.

        Imported lazily: the modal pulls the report/history models, which are not worth loading until
        the user actually asks to review prior runs.
        """
        from horde_worker_regen.tui.widgets.benchmark_history import BenchmarkHistoryModal

        self.app.push_screen(BenchmarkHistoryModal())

    def _open_download(self) -> None:
        """Open the Download models modal for the current tier/option selection.

        When the modal reports it downloaded something, re-run the plan preview so the operator sees the
        levels flip from "to download" to RUN: visible proof the problem is solved. Imported lazily to keep
        the modal's subprocess plumbing off the TUI's hot path.
        """
        from horde_worker_regen.tui.widgets.benchmark_download import BenchmarkDownloadModal

        def _after(downloaded_any: bool | None) -> None:
            if downloaded_any:
                self._start_plan_preview()

        self.app.push_screen(BenchmarkDownloadModal(self._collect_options()), _after)

    def _start_plan_preview(self) -> None:
        """Shell ``horde-benchmark plan --json`` in a worker thread and render the result.

        The plan starts no worker and never touches the GPU, so it is safe to run while idle without the
        worker/benchmark GPU hand-off the Run path needs.
        """
        self.query_one("#benchmark-plan", Static).update(
            Panel(
                Text("Computing plan (detecting hardware; no worker is started)…", style="yellow"),
                title="Resource plan",
                title_align="left",
                border_style="grey37",
            ),
        )
        options = self._collect_options()
        self.run_worker(
            lambda: self._compute_plan_preview(options),
            thread=True,
            exclusive=True,
            group="benchmark-plan",
        )

    def _compute_plan_preview(self, options: BenchmarkOptions) -> None:
        """(Worker thread) run the plan subcommand and hand the parsed rows back to the UI thread."""
        try:
            result = subprocess.run(
                options.build_plan_command(),
                capture_output=True,
                text=True,
                timeout=_PLAN_PREVIEW_TIMEOUT_SECONDS,
                check=False,
            )
        except Exception as e:  # noqa: BLE001 - the preview is best-effort; surface it, never crash the TUI
            self.app.call_from_thread(self._render_plan_error, f"{type(e).__name__}: {e}")
            return
        if result.returncode != 0 or not result.stdout.strip():
            tail = (result.stderr or result.stdout or "no output").strip().splitlines()
            self.app.call_from_thread(self._render_plan_error, tail[-1] if tail else "no output")
            return
        try:
            rows = decode_plan_rows(result.stdout)
        except ValueError as e:
            self.app.call_from_thread(self._render_plan_error, f"could not parse plan output: {e}")
            return
        self.app.call_from_thread(self._render_plan_preview, rows)

    def _render_plan_preview(self, rows: list[LevelPlanRow]) -> None:
        """(UI thread) render the previewed plan rows."""
        self.query_one("#benchmark-plan", Static).update(self._plan_panel(rows))

    def _render_plan_error(self, message: str) -> None:
        """(UI thread) show why the plan preview could not be produced."""
        self.query_one("#benchmark-plan", Static).update(
            Panel(
                Text(f"Plan preview failed: {message}", style="red"),
                title="Resource plan",
                title_align="left",
                border_style="red",
            ),
        )

    @staticmethod
    def _plan_disk_cell(row: LevelPlanRow) -> str:
        """Render the disk cell as ``free / needed`` GB when free space is known, else just ``needed``."""
        needed = f"{row.min_disk_free_gb:.0f}G"
        if row.free_disk_bytes is None:
            return needed
        return f"{row.free_disk_bytes / 1024**3:.0f} / {needed}"

    @staticmethod
    def _plan_panel(rows: list[LevelPlanRow]) -> Panel:
        """Render the per-level resource plan and predicted run/skip verdicts as a table.

        When any level still needs models, a plain-language banner sits above the table pointing at the
        Download models button, so a novice sees the one concrete next step rather than a wall of verdicts.
        """
        table = Table(expand=True)
        table.add_column("Level")
        table.add_column("Stage/Tier")
        table.add_column("VRAM", justify="right")
        table.add_column("Disk (free/need)", justify="right")
        table.add_column("Net")
        table.add_column("Key")
        table.add_column("Controlnet")
        table.add_column("Verdict")
        for row in rows:
            vram = "-" if row.estimated_vram_mb is None else f"{row.estimated_vram_mb / 1024:.1f}G"
            verdict = Text("RUN", style="green") if row.will_run else Text(f"SKIP ({row.verdict})", style="grey50")
            table.add_row(
                row.level_id,
                f"{row.stage}/{row.tier}",
                vram,
                BenchmarkView._plan_disk_cell(row),
                "yes" if row.requires_network else "-",
                "civitai" if row.requires_civitai_key else "-",
                BenchmarkView._plan_controlnet_cell(row),
                verdict,
            )

        body: RenderableType = table
        needs_models = any(row.num_models_missing for row in rows)
        # Only nag when annotators are confirmed absent on disk (present is False); a static ROM size is
        # not evidence they are missing. Unknown presence stays silent (the level pre-warms before timing).
        needs_annotators = any(row.requires_controlnet and row.controlnet_annotators_present is False for row in rows)
        if needs_models or needs_annotators:
            what = "models" if needs_models and not needs_annotators else "models or controlnet annotators"
            banner = Text(
                f"Some of these levels need {what} you have not downloaded yet. Press “Download models” "
                "below to fetch them first: otherwise the benchmark will download them mid-run, which makes the "
                "timing slower and less accurate.",
                style="yellow",
            )
            body = Group(banner, Text(""), table)
        return Panel(body, title="Resource plan", title_align="left", border_style="grey37")

    @staticmethod
    def _plan_controlnet_cell(row: LevelPlanRow) -> Text:
        """Render the controlnet cell: ``-`` (n/a), red ``missing`` (extra absent), or the annotator ROM size.

        When the extra is absent the prospective annotator size is still shown (when known) so an operator
        weighing whether to install it sees both the gap and its disk cost.
        """
        if not row.requires_controlnet:
            return Text("-")
        size = f"~{row.controlnet_annotator_bytes / 1024**3:.1f}G" if row.controlnet_annotator_bytes > 0 else ""
        if row.controlnet_installed is False:
            return Text(f"missing {size}".rstrip(), style="red")
        if row.controlnet_annotators_present:
            return Text("ok", style="green")
        return Text(size or "ok", style="green" if not size else "")

    def _update_buttons(self, status: BenchmarkSupervisorStatus, run_state: BenchmarkRunState) -> None:
        """Enable only the actions valid for the current status."""
        running = status is BenchmarkSupervisorStatus.RUNNING
        # PREPARING (worker being stopped, before the subprocess launches) is busy too: block a second
        # Run and the worker-restarting Restore, but there is no subprocess yet to Cancel.
        active = status in (BenchmarkSupervisorStatus.PREPARING, BenchmarkSupervisorStatus.RUNNING)
        has_suggestion = bool(run_state.suggested_bridge_data_yaml)
        self.query_one("#benchmark-run", Button).disabled = active
        self.query_one("#benchmark-preview", Button).disabled = active
        # Downloading contends with the GPU-exclusive run and stops a level mid-flight, so gate it while busy.
        self.query_one("#benchmark-download", Button).disabled = active
        # History only reads completed runs from disk, so it is safe (and useful) at any time.
        self.query_one("#benchmark-history", Button).disabled = False
        self.query_one("#benchmark-cancel", Button).disabled = not running
        self.query_one("#benchmark-apply", Button).disabled = not (
            status is BenchmarkSupervisorStatus.FINISHED and has_suggestion
        )
        self.query_one("#benchmark-restore", Button).disabled = active or not self._has_known_good

    def _build_app_state_summary(self) -> Text:
        """Render the persisted last-benchmark status (and any known-good config) as a short summary."""
        state = AppStateStore().load()
        self._has_known_good = state.last_known_good_settings is not None
        availability = benchmark_status_summary(state, current_version=__version__)

        if availability is BenchmarkAvailability.NONE:
            summary = Text("No benchmark on record; run one to auto-tune this worker.", style="yellow")
        else:
            benchmark = state.last_benchmark
            assert benchmark is not None  # NONE is the only case with no record
            badge_style = "black on yellow" if availability is BenchmarkAvailability.STALE else "black on green"
            badge_text = " STALE (version changed) " if availability is BenchmarkAvailability.STALE else " CURRENT "
            summary = Text.assemble(
                Text(badge_text, style=badge_style),
                "  ",
                Text.from_markup(f"[grey62]last run[/] {benchmark.run_id}"),
                "   ",
                Text.from_markup(f"[grey62]passed[/] {benchmark.levels_passed}/{benchmark.levels_total}"),
                "   ",
                Text.from_markup(f"[grey62]version[/] {benchmark.worker_version}"),
            )

        known_good = state.last_known_good_settings
        if known_good is not None:
            summary.append("\n")
            summary.append_text(
                Text.from_markup(
                    f"[grey62]known-good[/] {known_good.source.value} on v{known_good.worker_version} "
                    "- restorable below",
                ),
            )
        return summary

    def _render_body(
        self, run_state: BenchmarkRunState, status: BenchmarkSupervisorStatus, frame: int
    ) -> RenderableType:
        """Render the headline, current-level card, per-level table, and (when done) the recommendation."""
        if status is BenchmarkSupervisorStatus.IDLE and not run_state.level_order:
            return self._render_idle_hint()

        # Before any level exists (worker stop, then the subprocess's import + hardware-probe window),
        # show the startup phase so the slow hand-off reads as motion rather than a frozen blank tab.
        if not run_state.level_order and run_state.startup_phase:
            return self._render_starting(run_state, status, frame)

        sections: list[RenderableType] = [self._render_headline(run_state, status, frame)]
        current = run_state.current_level_id
        if current is not None and current in run_state.levels:
            sections.append(self._render_current_level(run_state.levels[current], frame))
        if run_state.level_order:
            sections.append(self._render_level_table(run_state))
        if run_state.finished and run_state.suggested_bridge_data_yaml:
            sections.append(self._render_suggestion(run_state))
        return Group(*sections)

    @staticmethod
    def _render_idle_hint() -> Panel:
        """The pre-run message: recommend Preview plan first, before committing the GPU."""
        body = Text.assemble(
            (
                "A benchmark ramps the worker through safe difficulty levels, suggests a tuned bridgeData, "
                "and flags robustness problems.\n\n",
                "grey70",
            ),
            ("Start with ", "grey70"),
            ("Preview plan", "bold cyan"),
            (
                ": it shows what each level needs and what will run on this machine, without starting the "
                "worker or using the GPU. When you are ready, ",
                "grey70",
            ),
            ("Run benchmark", "bold green"),
            (" stops the worker and measures it for real.", "grey70"),
        )
        return Panel(body, title="Benchmark", title_align="left", border_style="cyan")

    @staticmethod
    def _render_starting(run_state: BenchmarkRunState, status: BenchmarkSupervisorStatus, frame: int) -> Panel:
        """The pre-level startup card: shows the current phase during the worker-stop and import window."""
        run_id = run_state.run_id or "-"
        spinner = _SPINNER[frame % len(_SPINNER)]
        body = Text.assemble(
            (f" {status.value.upper()} ", "black on yellow"),
            "  ",
            (run_id, "bold"),
            "\n\n",
            (f"{spinner} ", "bold yellow"),
            (run_state.startup_phase or "Starting…", "yellow"),
            "\n\n",
            ("This can take a minute on a cold start (importing the inference stack and probing the GPU). ", "grey70"),
            ("Live detail is written to the run's ", "grey70"),
            ("console.log", "grey85"),
            (".", "grey70"),
        )
        return Panel(body, title="Benchmark starting", title_align="left", border_style="yellow")

    @staticmethod
    def _render_headline(run_state: BenchmarkRunState, status: BenchmarkSupervisorStatus, frame: int) -> Panel:
        """A one-line summary of the run's identity, mode, and overall progress."""
        finished_levels = sum(1 for level in run_state.levels.values() if level.outcome is not None)
        total = run_state.num_levels or run_state.levels_total or len(run_state.level_order)
        gpu = run_state.gpu_name or "unknown GPU"
        status_colour = {
            BenchmarkSupervisorStatus.PREPARING: "yellow",
            BenchmarkSupervisorStatus.RUNNING: "yellow",
            BenchmarkSupervisorStatus.FINISHED: "green",
            BenchmarkSupervisorStatus.FAILED: "red",
            BenchmarkSupervisorStatus.CANCELLED: "grey50",
            BenchmarkSupervisorStatus.IDLE: "grey50",
        }.get(status, "white")
        active = status in (BenchmarkSupervisorStatus.PREPARING, BenchmarkSupervisorStatus.RUNNING)
        badge = f"{_SPINNER[frame % len(_SPINNER)]} {status.value.upper()} " if active else f" {status.value.upper()} "
        body = Text.assemble(
            (badge, f"black on {status_colour}"),
            "  ",
            (f"{run_state.run_id or '-'}", "bold"),
            (f"   mode={run_state.process_mode or '-'}   {gpu}   ", "grey62"),
            (f"levels {finished_levels}/{total}", "bold cyan"),
        )
        return Panel(body, border_style=status_colour)

    @staticmethod
    def _jobs_cell(level: LevelState) -> RenderableType:
        """The Jobs row: a filled progress bar when the job count is known, else a bare counter.

        A bar gives an at-a-glance sense of how far through the level we are, which a raw ``3/8`` does not.
        """
        suffix = Text(f"  ({level.jobs_faulted} faulted)", style="red") if level.jobs_faulted else Text("")
        if not level.jobs_expected:
            return Text.assemble((str(level.jobs_completed), "white"), suffix)
        width = 20
        fraction = max(0.0, min(1.0, level.jobs_completed / level.jobs_expected))
        filled = int(round(fraction * width))
        bar = Text("█" * filled, style="green")
        bar.append("░" * (width - filled), style="grey37")
        bar.append(f"  {level.jobs_completed}/{level.jobs_expected}  {fraction * 100:.0f}%", style="grey70")
        bar.append_text(suffix)
        return bar

    @staticmethod
    def _render_current_level(level: LevelState, frame: int) -> Panel:
        """A live metric card for the level currently running, led by a spinner so it always reads as live."""
        spinner = _SPINNER[frame % len(_SPINNER)]
        title = Text.assemble((f"{spinner} ", "bold cyan"), ("Current level", "bold"))
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="bold cyan")
        table.add_column()
        table.add_row("Level", f"{level.level_id} ({level.stage}/{level.tier}/{level.axis})")
        if level.phase:
            table.add_row("Status", Text(level.phase, style="yellow"))
        table.add_row("Jobs", BenchmarkView._jobs_cell(level))
        table.add_row("it/s", "-" if level.iterations_per_second is None else f"{level.iterations_per_second:.2f}")
        table.add_row("VRAM", "-" if level.vram_used_mb is None else f"{level.vram_used_mb} MB")
        table.add_row("GPU busy", "-" if level.gpu_busy_percent is None else f"{level.gpu_busy_percent:.0f}%")
        table.add_row("Elapsed", f"{level.elapsed_seconds:.0f}s")
        if level.num_process_recoveries:
            table.add_row("Restarts", Text(f"{level.num_process_recoveries} (!)", style="bold red"))
        if level.process_summary:
            table.add_row("Processes", Text(level.process_summary, style="grey62"))
        return Panel(table, title=title, title_align="left", border_style="cyan")

    def _render_level_table(self, run_state: BenchmarkRunState) -> Panel:
        """A per-level verdict table built up as levels finish."""
        table = Table(expand=True)
        table.add_column("Level")
        table.add_column("Stage/Tier")
        table.add_column("Outcome")
        table.add_column("it/s p50", justify="right")
        table.add_column("Notes")
        for level in run_state.ordered_levels():
            outcome = level.outcome or ("running" if run_state.current_level_id == level.level_id else "pending")
            colour = _OUTCOME_COLOURS.get(level.outcome or "", "yellow")
            its = "-" if level.iterations_per_second is None else f"{level.iterations_per_second:.2f}"
            notes = "; ".join(level.reasons + level.advisories)
            table.add_row(
                level.level_id,
                f"{level.stage}/{level.tier}",
                Text(outcome, style=colour),
                its,
                notes,
            )
        return Panel(table, title="Levels", title_align="left", border_style="grey37")

    @staticmethod
    def _render_suggestion(run_state: BenchmarkRunState) -> Panel:
        """The recommended bridgeData, its run totals, and the per-setting provenance behind each value."""
        findings = f"  ·  {run_state.num_findings} robustness findings" if run_state.num_findings else ""
        header = Text.from_markup(
            f"[bold]{run_state.levels_passed}/{run_state.levels_total}[/] levels passed{findings}\n"
            "Press [bold green]Apply suggested config[/] to write these into bridgeData.yaml.\n",
        )
        sections: list[RenderableType] = [header, Text(run_state.suggested_bridge_data_yaml, style="grey82")]
        if run_state.suggestion_decisions:
            sections.append(Text("\nWhy each value:", style="bold"))
            sections.append(BenchmarkView._provenance_table(run_state))
        for warning in run_state.consistency_warnings:
            sections.append(Text(f"(!) {warning}", style="yellow"))
        return Panel(Group(*sections), title="Suggested bridgeData", title_align="left", border_style="green")

    @staticmethod
    def _provenance_table(run_state: BenchmarkRunState) -> Table:
        """Render why each suggested setting holds its value, colour-coded by the strength of evidence."""
        table = Table(expand=True, show_edge=False, pad_edge=False)
        table.add_column("Setting")
        table.add_column("Value")
        table.add_column("Basis")
        table.add_column("Detail")
        for decision in run_state.suggestion_decisions:
            table.add_row(
                decision.setting,
                decision.value_text,
                Text(decision.basis_label, style=_BASIS_STYLES.get(decision.basis, "grey70")),
                Text(decision.detail, style="grey50"),
            )
        return table
