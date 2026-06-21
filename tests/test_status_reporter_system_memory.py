"""Tests for the console StatusReporter's System RAM section."""

from __future__ import annotations

from collections.abc import Callable

from horde_worker_regen.process_management.system_memory import build_system_memory_summary
from horde_worker_regen.reporting.status_reporter import StatusReporter

_GB = 1024**3


def _collector() -> tuple[Callable[..., None], list[str]]:
    """Return a logging-function stand-in that records the lines it is given."""
    lines: list[str] = []

    def _record(message: str, *_args: object, **_kwargs: object) -> None:
        lines.append(message)

    return _record, lines


def test_print_system_memory_returns_false_without_sample() -> None:
    """With no sample there is nothing to show and nothing printed."""
    record, lines = _collector()
    assert StatusReporter._print_system_memory(record, None) is False
    assert lines == []


def test_print_system_memory_returns_false_with_zero_total() -> None:
    """A degenerate zero-total sample (cold start) prints nothing."""
    record, lines = _collector()
    summary = build_system_memory_summary(total_bytes=0, available_bytes=0, worker_rss_by_role={})
    assert StatusReporter._print_system_memory(record, summary) is False
    assert lines == []


def test_print_system_memory_renders_total_breakdown_and_other() -> None:
    """The section shows the in-use/total figure, the per-role worker breakdown, and the 'other' line."""
    record, lines = _collector()
    summary = build_system_memory_summary(
        total_bytes=64 * _GB,
        available_bytes=20 * _GB,  # used = 44 GB
        worker_rss_by_role={
            "orchestrator": 1 * _GB,
            "inference": 18 * _GB,
            "safety": 2 * _GB,
            "download": 0,
        },
    )

    assert StatusReporter._print_system_memory(record, summary) is True
    blob = "\n".join(lines)
    assert "System RAM" in blob
    assert "/ 64.0 GB" in blob  # total
    assert "44.0 GB" in blob  # used
    assert "available 20.0 GB" in blob
    assert "inference 18.0 GB" in blob
    assert "safety 2.0 GB" in blob
    # used (44) minus worker subtotal (21) = 23 GB attributed to the rest of the machine.
    assert "Other (OS + apps): 23.0 GB" in blob
    # The download role contributed nothing, so it is omitted from the breakdown.
    assert "download" not in blob
