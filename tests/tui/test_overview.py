"""Tests for the overview dashboard rendering helpers."""

from __future__ import annotations

from rich.console import Console

from horde_worker_regen.process_management.supervisor_channel import (
    JobQueueEntry,
    ProcessSnapshot,
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.tui.health import derive
from horde_worker_regen.tui.widgets.overview import OverviewView
from horde_worker_regen.tui.worker_launcher import SupervisorStatus


def _render(renderable: object) -> str:
    """Render a Rich renderable to plain text."""
    console = Console(width=160)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


def _busy_process() -> ProcessSnapshot:
    """A sampling inference process carrying resolution/batch/step detail."""
    return ProcessSnapshot(
        process_id=1,
        process_type="INFERENCE",
        last_process_state="INFERENCE_STARTING",
        is_alive=True,
        is_busy=True,
        loaded_horde_model_name="AlbedoBase XL",
        batch_amount=2,
        current_job_width=832,
        current_job_height=1216,
        current_job_steps=28,
        last_current_step=14,
        last_total_steps=28,
        last_iterations_per_second=8.0,
        vram_usage_mb=8000,
        total_vram_mb=24000,
    )


def test_overview_shows_lora_pause_when_background_download_blocks_pops() -> None:
    """The overview explains temporary LoRA pop suppression."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(
            dreamer_name="Tester",
            worker_version="12.0.0",
            allow_lora=True,
            effective_allow_lora=False,
            allow_controlnet=True,
        ),
        worker_registered=True,
        lora_pops_blocked_by_downloads=True,
    )
    report = derive(snapshot, SupervisorStatus.RUNNING, 0.5)

    hero = _render(OverviewView()._render_hero(report, snapshot, frame=0))

    assert "LoRA pops paused while background downloads are active." in hero
    assert OverviewView._allow_summary(snapshot) == "img2img, lora paused, controlnet, post"


def test_process_table_shows_resolution_and_batch_by_default() -> None:
    """The Size column surfaces the active job's resolution and batch without the detail toggle."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        processes=[_busy_process()],
    )
    text = _render(OverviewView()._render_process_table(snapshot))
    assert "832×1216 ×2" in text
    # The lean view does not include the heartbeat-type column.
    assert "HB type" not in text


def test_process_table_detailed_adds_technical_columns() -> None:
    """The F6 detail view reveals per-job steps and heartbeat columns."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        processes=[_busy_process()],
    )
    text = _render(OverviewView()._render_process_table(snapshot, detailed=True))
    assert "HB type" in text
    assert "Steps" in text


def test_pipeline_strip_shows_lifecycle_stages() -> None:
    """The job-pipeline strip labels each lifecycle stage with its live count."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        jobs_pending_inference=3,
        jobs_in_progress=1,
        jobs_pending_safety_check=0,
        jobs_pending_submit=2,
        num_jobs_submitted=42,
    )
    text = _render(OverviewView()._render_pipeline_strip(snapshot))
    assert "Queue" in text and "Inference" in text and "Safety" in text and "Submit" in text
    assert "42 submitted" in text


def test_queue_lane_renders_upcoming_blocks() -> None:
    """The queue lane renders an 'Up next' line of blocks for pending jobs."""
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        pending_jobs=[
            JobQueueEntry(job_id="a", model="Deliberate", steps=30, width=1024, height=1024),
            JobQueueEntry(job_id="b", model="SDXL", steps=25, width=512, height=768),
        ],
    )
    text = _render(OverviewView()._render_queue_table(snapshot))
    assert "Up next" in text
    assert "1024²" in text


def test_momentum_sparklines_track_recorded_trends() -> None:
    """Recorded GPU-duty/kudos history renders as a non-empty sparkline panel."""
    view = OverviewView()
    snapshot = WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        gpu_utilization_mean_percent=70.0,
        kudos_per_hour=12000.0,
    )
    for percent in (40.0, 55.0, 80.0):
        snapshot.gpu_utilization_mean_percent = percent
        view._record_trends(snapshot)
    text = _render(view._render_momentum(snapshot))
    assert "GPU duty" in text and "Kudos/hr" in text
