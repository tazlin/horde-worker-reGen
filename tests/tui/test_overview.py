"""Tests for the overview dashboard rendering helpers."""

from __future__ import annotations

from rich.console import Console

from horde_worker_regen.process_management.supervisor_channel import WorkerConfigSummary, WorkerStateSnapshot
from horde_worker_regen.tui.health import derive
from horde_worker_regen.tui.widgets.overview import OverviewView
from horde_worker_regen.tui.worker_launcher import SupervisorStatus


def _render(renderable: object) -> str:
    """Render a Rich renderable to plain text."""
    console = Console(width=120)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


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
