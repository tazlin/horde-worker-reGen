"""Tests for the system-RAM summary: the derived figures and the wire-model round trip."""

from __future__ import annotations

from horde_worker_regen.process_management.supervisor_channel import SystemMemorySnapshot
from horde_worker_regen.process_management.system_memory import (
    ROLE_DOWNLOAD,
    ROLE_INFERENCE,
    ROLE_ORCHESTRATOR,
    ROLE_SAFETY,
    SystemMemorySummary,
    build_system_memory_summary,
)

_GB = 1024**3


def _summary(**roles: int) -> SystemMemorySummary:
    return build_system_memory_summary(
        total_bytes=64 * _GB,
        available_bytes=20 * _GB,
        worker_rss_by_role=roles,
    )


def test_used_is_total_minus_available() -> None:
    """The system-wide used figure is derived independently of the per-role sum."""
    summary = _summary(inference=10 * _GB)
    assert summary.used_bytes == 44 * _GB


def test_worker_total_sums_roles() -> None:
    """The worker subtotal is the sum of every role's RSS."""
    summary = _summary(orchestrator=1 * _GB, inference=18 * _GB, safety=2 * _GB, download=_GB // 2)
    assert summary.worker_total_bytes == 21 * _GB + _GB // 2


def test_other_bytes_is_used_minus_worker() -> None:
    """'Other' RAM is the used figure with the worker's own footprint removed."""
    summary = _summary(inference=10 * _GB)  # used = 44 GB, worker = 10 GB
    assert summary.other_bytes == 34 * _GB


def test_other_bytes_clamped_when_worker_rss_overcounts() -> None:
    """RSS over-counting (worker subtotal > system used) yields zero 'other', never a negative."""
    summary = build_system_memory_summary(
        total_bytes=64 * _GB,
        available_bytes=60 * _GB,  # used = 4 GB
        worker_rss_by_role={"inference": 30 * _GB},  # over-counts (shared pages)
    )
    assert summary.other_bytes == 0


def test_fractions() -> None:
    """Used and worker fractions are computed against total; worker is capped at 1.0."""
    summary = _summary(inference=16 * _GB)
    assert summary.used_fraction == (44 * _GB) / (64 * _GB)
    assert summary.worker_fraction == (16 * _GB) / (64 * _GB)


def test_fractions_none_without_total() -> None:
    """A degenerate zero-total sample yields None fractions rather than dividing by zero."""
    summary = build_system_memory_summary(total_bytes=0, available_bytes=0, worker_rss_by_role={})
    assert summary.used_fraction is None
    assert summary.worker_fraction is None


def test_builder_clamps_negatives() -> None:
    """Negative or stray figures are normalised to zero so downstream math never sees them."""
    summary = build_system_memory_summary(
        total_bytes=-5,
        available_bytes=-1,
        worker_rss_by_role={"inference": -100, "safety": 5},
    )
    assert summary.total_bytes == 0
    assert summary.available_bytes == 0
    assert summary.worker_rss_by_role == {"inference": 0, "safety": 5}


def test_nonzero_role_items_orders_and_filters() -> None:
    """Only non-zero roles appear, ordered inference-first with download last."""
    summary = _summary(orchestrator=1 * _GB, inference=18 * _GB, safety=0, download=_GB)
    assert summary.nonzero_role_items() == [
        (ROLE_INFERENCE, 18 * _GB),
        (ROLE_ORCHESTRATOR, 1 * _GB),
        (ROLE_DOWNLOAD, _GB),
    ]
    # safety contributed nothing, so it is omitted.
    assert all(role != ROLE_SAFETY for role, _ in summary.nonzero_role_items())


def test_wire_roundtrip_preserves_derived_figures() -> None:
    """The wire snapshot rebuilds an equivalent summary via ``to_summary`` (math lives in one place)."""
    summary = _summary(orchestrator=1 * _GB, inference=18 * _GB, safety=2 * _GB)
    wire = SystemMemorySnapshot.from_summary(summary)

    # Survives a JSON round trip (the transport is pickle, but JSON proves it is plain data).
    restored = SystemMemorySnapshot.model_validate_json(wire.model_dump_json()).to_summary()

    assert restored.total_bytes == summary.total_bytes
    assert restored.available_bytes == summary.available_bytes
    assert restored.worker_total_bytes == summary.worker_total_bytes
    assert restored.other_bytes == summary.other_bytes
    assert restored.nonzero_role_items() == summary.nonzero_role_items()
