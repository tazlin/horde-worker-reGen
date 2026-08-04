"""Tests for model-pool state in the headless periodic status block."""

from __future__ import annotations

from collections.abc import Callable

from horde_worker_regen.process_management.ipc.supervisor_channel import (
    ModelPoolBenchRow,
    ModelPoolSeatReadiness,
    ModelPoolSeatRow,
    ModelPoolSnapshot,
)
from horde_worker_regen.reporting.status_reporter import StatusReporter


def _collector() -> tuple[Callable[..., None], list[str]]:
    """Return a logging stand-in and the lines it records.

    Payloads the reporter cannot let a colorizer parse are passed as formatting arguments, so the stand-in
    renders them the way the logger would; recording the bare template would hide the line's real content.
    """
    lines: list[str] = []

    def _record(message: str, *args: object, **kwargs: object) -> None:
        lines.append(message.format(*args, **kwargs) if args or kwargs else message)

    return _record, lines


def test_disabled_pool_names_normal_advertising_behavior() -> None:
    """The default state is explicit in logs instead of looking like absent observability."""
    record, lines = _collector()
    StatusReporter._print_model_pool(record, enabled=False, pool=None)

    assert "Model pool: off" in lines[0]
    assert "eligible-model advertising" in lines[0]


def test_enabled_pool_prints_seats_lanes_demand_and_budget() -> None:
    """Headless output carries the same compact runtime vocabulary as the TUI."""
    record, lines = _collector()
    pool = ModelPoolSnapshot(
        enabled=True,
        seats=[
            ModelPoolSeatRow(
                model="Deliberate",
                source="MANUAL",
                state="ACTIVE",
                readiness=ModelPoolSeatReadiness.RESIDENT,
                resident_device_indices=[0],
            ),
            ModelPoolSeatRow(
                model="AlbedoBase XL",
                state="PENDING_DOWNLOAD",
                pending_model="Flux.1-dev",
                source="RANKER",
            ),
            ModelPoolSeatRow(model="Starved", source="RESCUE", state="ACTIVE"),
        ],
        bench=[ModelPoolBenchRow(model="Old", reason="EMPTY_POPS", cooldown_remaining_seconds=30.0)],
        current_lane="FIXED",
        demand_age_seconds=42.0,
        download_budget_gb=50.0,
        download_bytes_charged=5 * 1024**3,
        fixed_pops=8,
        fixed_fulfilled=6,
        fixed_resident_hits=5,
        free_pops=2,
        free_fulfilled=1,
        free_resident_hits=0,
    )

    StatusReporter._print_model_pool(record, enabled=True, pool=pool)

    line = lines[0]
    assert "Deliberate[M] resident(gpu 0)" in line
    assert "AlbedoBase XL->Flux.1-dev[R] downloading" in line
    assert "Starved[S] cold" in line
    assert "fixed 6/8 matched (75%); 5 resident" in line
    assert "free 1/2 matched (50%); 0 resident" in line
    assert "bench 1" in line
    assert "demand age 42s" in line
    assert "5.0 GB/50 GB" in line
