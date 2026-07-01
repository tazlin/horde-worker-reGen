"""Tests for the live view: sampling-row gating and the staleness banner."""

from __future__ import annotations

from rich.console import Console

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    PreloadAdmissionSnapshot,
    ProcessSnapshot,
    RamGovernanceSnapshot,
    SchedulingGovernanceSnapshot,
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.tui.widgets.live_view import LiveView


def _render(renderable: object) -> str:
    """Render a Rich renderable to plain text."""
    console = Console(width=100)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


def _process(state: str) -> ProcessSnapshot:
    """A process carrying populated sampling fields, in the given state."""
    return ProcessSnapshot(
        process_id=0,
        process_type="INFERENCE",
        last_process_state=state,
        is_alive=True,
        is_busy=state != "WAITING_FOR_JOB",
        loaded_horde_model_name="Deliberate",
        last_heartbeat_timestamp=0.0,
        last_current_step=30,
        last_total_steps=30,
        last_iterations_per_second=9.1,
        vram_usage_mb=8000,
        total_vram_mb=24000,
    )


def test_idle_process_hides_stale_sampling_row() -> None:
    """A WAITING_FOR_JOB process must not show a sampling/throughput row, even with step fields set."""
    panel = LiveView()._render_process_panel(_process("WAITING_FOR_JOB"))
    text = _render(panel)
    assert "Sampling" not in text
    assert "it/s" not in text


def test_active_process_shows_sampling_row() -> None:
    """An actively-sampling process shows the sampling row built from its step fields."""
    panel = LiveView()._render_process_panel(_process("INFERENCE_STARTING"))
    text = _render(panel)
    assert "Sampling" in text
    assert "30/30 steps" in text


def test_resolution_row_shows_by_default_and_job_id_is_gated() -> None:
    """Resolution/batch show without the detail toggle; the raw job ID and heartbeat are gated."""
    process = _process("INFERENCE_STARTING")
    process.current_job_width = 832
    process.current_job_height = 1216
    process.batch_amount = 2
    process.current_job_id = "job-xyz"

    lean = _render(LiveView()._render_process_panel(process))
    assert "832×1216" in lean
    assert "batch ×2" in lean
    assert "job-xyz" not in lean
    assert "Heartbeat" not in lean

    detailed = _render(LiveView()._render_process_panel(process, detailed=True))
    assert "job-xyz" in detailed
    assert "Heartbeat" in detailed


class _RecordingBody:
    """Stands in for the live-body Static, capturing the renderable handed to update()."""

    def __init__(self) -> None:
        self.renderable: object | None = None

    def update(self, renderable: object) -> None:
        self.renderable = renderable


def _snapshot(state: str = "WAITING_FOR_JOB") -> WorkerStateSnapshot:
    config = WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0")
    return WorkerStateSnapshot(config=config, processes=[_process(state)])


def test_stale_snapshot_shows_banner() -> None:
    """When the snapshot is old, the view prepends a clear staleness banner."""
    view = LiveView()
    body = _RecordingBody()
    view.query_one = lambda *args, **kwargs: body  # type: ignore[method-assign,assignment]

    view.update_snapshot(_snapshot(), snapshot_age=12.0)
    text = _render(body.renderable)
    assert "old" in text
    assert "12s" in text


def test_fresh_snapshot_has_no_banner() -> None:
    """A fresh snapshot renders the panels without a staleness banner."""
    view = LiveView()
    body = _RecordingBody()
    view.query_one = lambda *args, **kwargs: body  # type: ignore[method-assign,assignment]

    view.update_snapshot(_snapshot(), snapshot_age=0.5)
    text = _render(body.renderable)
    assert "old" not in text


def test_aux_download_heartbeat_is_rendered_as_expected_quiet_work() -> None:
    """AUX downloads can block without child heartbeats, so the panel should not imply a dead process."""
    text = LiveView._heartbeat_text(12.0, True, "DOWNLOADING_AUX_MODEL").plain
    assert text == "working quietly for 12.0s"


def test_sampling_stale_heartbeat_still_warns() -> None:
    """Sampling silence remains suspicious because INFERENCE_STEP heartbeats are the liveness signal."""
    text = LiveView._heartbeat_text(16.0, True, "INFERENCE_STARTING")
    assert text.plain == "16.0s ago"
    assert text.style == "red"


def test_governance_strip_surfaces_scheduler_decisions() -> None:
    """The Live view starts with a compact RAM/preload decision strip."""
    governance = SchedulingGovernanceSnapshot(
        ram=RamGovernanceSnapshot(
            measured=True,
            reason="available 8192 MB above danger floor 4096 MB",
            pop_hold_active=True,
            draining_process_ids=[2],
        ),
        preload=PreloadAdmissionSnapshot(
            decision="defer_concurrency",
            model="AlbedoBase XL",
            process_id=2,
            reason="preload concurrency gate",
        ),
    )

    text = _render(LiveView._render_governance_strip(governance, detailed=True))

    assert "Scheduling" in text
    assert "holding" in text
    assert "Defer Concurrency" in text
    assert "AlbedoBase XL" in text
    assert "p2" in text
    assert "preload concurrency gate" in text
    assert "draining p2" in text
