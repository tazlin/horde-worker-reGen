"""Tests for the multi-GPU per-card dashboard surfaces: the GPUs tab, the Overview strip, and health.

These cover the rendering helpers directly (no mounted app), at fixed console widths, mirroring the other
TUI rendering tests. A single-GPU host renders one collapsed card by design (presentational consistency).
"""

from __future__ import annotations

import time
from collections import deque

from rich.console import Console

from horde_worker_regen.process_management.supervisor_channel import (
    CardSnapshot,
    ProcessSnapshot,
    WorkerConfigSummary,
    WorkerStateSnapshot,
)
from horde_worker_regen.tui.health import HealthStatus, derive
from horde_worker_regen.tui.widgets.gpus import GpusView
from horde_worker_regen.tui.widgets.overview import OverviewView
from horde_worker_regen.tui.worker_launcher import SupervisorStatus


def _render(renderable: object, width: int = 200) -> str:
    """Render a Rich renderable to plain text at the given console width."""
    console = Console(width=width)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


def _card(
    device_index: int,
    *,
    device_name: str | None = None,
    free_vram_mb: float | None = 18432.0,  # 18.0 GiB
    total_vram_mb: float | None = 24576.0,  # 24.0 GiB
    loaded_contexts: int = 2,
    busy_contexts: int = 1,
    target_process_count: int = 2,
    residency_model: str | None = None,
    residency_phase: str = "",
    unservable_models: list[str] | None = None,
) -> CardSnapshot:
    """A CardSnapshot with healthy defaults; override fields to exercise pressure/residency/fault states."""
    return CardSnapshot(
        device_index=device_index,
        device_name=device_name if device_name is not None else f"NVIDIA GeForce RTX 409{device_index}",
        kind="cuda",
        free_vram_mb=free_vram_mb,
        total_vram_mb=total_vram_mb,
        loaded_contexts=loaded_contexts,
        busy_contexts=busy_contexts,
        target_process_count=target_process_count,
        max_concurrent_inference=1,
        residency_model=residency_model,
        residency_phase=residency_phase,
        unservable_models=unservable_models or [],
    )


def _snapshot(cards: list[CardSnapshot], processes: list[ProcessSnapshot] | None = None) -> WorkerStateSnapshot:
    """A worker snapshot carrying the given per-card section (and optional processes)."""
    return WorkerStateSnapshot(
        config=WorkerConfigSummary(dreamer_name="Tester", worker_version="12.0.0"),
        per_card=cards,
        processes=processes or [],
    )


def _inference_process(
    process_id: int,
    *,
    device_index: int,
    its: float | None,
    model_name: str | None = None,
) -> ProcessSnapshot:
    """A busy inference process on a given card, carrying a sampling rate (and optional model name)."""
    return ProcessSnapshot(
        process_id=process_id,
        process_type="INFERENCE",
        device_index=device_index,
        last_process_state="INFERENCE_STARTING",
        is_alive=True,
        is_busy=True,
        loaded_horde_model_name=model_name,
        last_iterations_per_second=its,
    )


def test_gpus_tab_renders_a_row_per_card() -> None:
    """The GPUs table shows one row per card with its trimmed name, VRAM, and context counts."""
    snapshot = _snapshot([_card(0), _card(1)])
    out = _render(GpusView()._render_table(snapshot, detailed=False, available_width=200))

    assert "RTX 4090" in out  # the NVIDIA GeForce prefix is trimmed
    assert "RTX 4091" in out
    assert "18.0/24.0G free" in out
    assert "1/2▸2" in out  # busy/loaded▸target


def test_single_gpu_renders_one_collapsed_card() -> None:
    """A single-GPU host renders exactly one card row (the collapsed card)."""
    out = _render(GpusView()._render_table(_snapshot([_card(0)]), detailed=False, available_width=200))
    assert out.count("RTX 4090") == 1


def test_gpus_details_shows_residency_and_unservable() -> None:
    """The details view adds the per-card residency model/phase and any unservable-model flag."""
    snapshot = _snapshot([_card(0, residency_model="Flux.1", residency_phase="holding", unservable_models=["BadXL"])])
    out = _render(GpusView()._render_table(snapshot, detailed=True, available_width=240))

    assert "Flux.1 (holding)" in out
    assert "unservable" in out


def test_card_columns_shed_on_a_narrow_terminal() -> None:
    """A narrow width keeps the essentials (GPU/VRAM/Contexts) and sheds the NORMAL throughput columns."""
    table = GpusView()._render_table(_snapshot([_card(0)]), detailed=False, available_width=40)
    out = _render(table, width=200)

    assert "VRAM" in out  # essential, always shown
    assert "it/s" not in out  # NORMAL, shed at this width
    assert "Residency" not in out  # DETAILS, never shown without the details intent


def test_thin_aggregate_line_summarizes_the_fleet() -> None:
    """The thin view collapses the whole tab to one line: card count, free VRAM, contexts, and active ones."""
    out = _render(GpusView()._render_aggregate(_snapshot([_card(0), _card(1)])))

    assert "2 GPUs" in out
    assert "36.0G free" in out  # 18 + 18
    assert "4 ctx" in out  # 2 + 2 loaded
    assert "2 active" in out  # 1 + 1 busy


def test_thin_aggregate_flags_pressured_cards() -> None:
    """A card under VRAM pressure is called out in the thin aggregate line."""
    out = _render(GpusView()._render_aggregate(_snapshot([_card(0), _card(1, free_vram_mb=300.0)])))
    assert "pressured" in out


def test_card_jobs_per_hour_derives_from_history() -> None:
    """The per-card jobs/hr rate is derived from successive completion counts over wall-time."""
    view = GpusView()
    now = time.time()
    view._jobs_history[0] = deque([(now - 3600.0, 0), (now, 50)])

    rate, deltas = view._card_jobs_per_hour(0)
    assert rate is not None
    assert round(rate) == 50  # 50 completions over one hour
    assert deltas == [50.0]


def test_card_its_sums_only_its_own_busy_processes() -> None:
    """A card's it/s is the sum across its busy processes, excluding other cards' processes."""
    snapshot = _snapshot(
        [_card(0), _card(1)],
        processes=[
            _inference_process(0, device_index=0, its=4.0),
            _inference_process(1, device_index=0, its=3.0),
            _inference_process(2, device_index=1, its=9.0),
        ],
    )
    view = GpusView()
    assert view._card_its(snapshot, 0) == 7.0
    assert view._card_its(snapshot, 1) == 9.0


def test_overview_strip_renders_each_card() -> None:
    """The Overview per-card strip names each card with its VRAM bar and context counts."""
    snapshot = _snapshot([_card(0), _card(1)])
    out = _render(OverviewView()._render_gpus_strip(snapshot, detailed=False))

    assert "RTX 4090" in out
    assert "RTX 4091" in out
    assert "2/2 ctx" in out


def test_overview_strip_details_flags_residency_and_unservable() -> None:
    """In details mode the strip annotates a card's residency and unservable-model count."""
    snapshot = _snapshot([_card(0, residency_model="Flux.1", residency_phase="holding", unservable_models=["BadXL"])])
    out = _render(OverviewView()._render_gpus_strip(snapshot, detailed=True))

    assert "Flux.1" in out
    assert "unservable" in out


def test_process_table_shows_gpu_column_and_groups_by_card() -> None:
    """The process table carries a GPU column and orders rows by card, then slot."""
    # Slot 0 is on card 1 and slot 1 is on card 0; grouping by device must render card 0's slot first even
    # though it has the higher slot id and the alphabetically-later model, so neither slot-id nor model order
    # could produce this sequence by accident.
    snapshot = _snapshot(
        [_card(0), _card(1)],
        processes=[
            _inference_process(0, device_index=1, its=5.0, model_name="aaa_on_card1"),
            _inference_process(1, device_index=0, its=5.0, model_name="zzz_on_card0"),
        ],
    )
    out = _render(OverviewView()._render_process_table(snapshot, detailed=False, available_width=200))

    assert "GPU" in out
    assert out.index("zzz_on_card0") < out.index("aaa_on_card1")


def test_per_card_health_warns_on_pressure_only_when_multi_gpu() -> None:
    """A pressured card on a multi-GPU host raises a named WARN; a single-GPU host has no per-card check."""
    pressured = _snapshot([_card(0), _card(1, free_vram_mb=200.0)])
    report = derive(pressured, SupervisorStatus.RUNNING, 0.5)
    pressure_checks = [check for check in report.checks if check.name == "GPU 1"]
    assert len(pressure_checks) == 1
    assert pressure_checks[0].status is HealthStatus.WARN
    assert "VRAM pressure" in pressure_checks[0].detail

    single = derive(_snapshot([_card(0)]), SupervisorStatus.RUNNING, 0.5)
    assert not [check for check in single.checks if check.name.startswith("GPU ") or check.name == "GPUs"]


def test_per_card_health_summarizes_a_healthy_fleet() -> None:
    """A healthy multi-GPU fleet gets one reassuring summary check rather than per-card noise."""
    report = derive(_snapshot([_card(0), _card(1)]), SupervisorStatus.RUNNING, 0.5)
    summary = [check for check in report.checks if check.name == "GPUs"]
    assert len(summary) == 1
    assert summary[0].status is HealthStatus.OK
    assert "2 cards healthy" in summary[0].detail
