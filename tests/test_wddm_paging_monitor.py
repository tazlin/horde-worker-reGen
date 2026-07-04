"""Tests for the WDDM demand-paging telemetry and its scheduler verdict.

The monitor reads the per-process ``GPU Process Memory`` counters (the data behind Task Manager's
"Shared GPU memory" column). A worker child whose shared usage climbs while sampling is being
demand-paged, with PID-level attribution separating the worker's own over-commit from external VRAM
pressure. The source is Windows-only and the contract is graceful absence: on any host without the
counters the monitor collects nothing and every consumer reads None/False, never an exception.
"""

from __future__ import annotations

import threading

from horde_worker_regen.utils import wddm_paging_monitor as wddm_module
from horde_worker_regen.utils.wddm_paging_monitor import (
    GpuProcessMemorySample,
    WddmPagingMonitor,
    assess_worker_paging,
    parse_counter_instance_pid,
)


class TestInstanceParsing:
    """PID extraction from GPU Process Memory counter instance names."""

    def test_parses_pid_from_counter_instance(self) -> None:
        """The PID embeds in the instance name between adapter/segment qualifiers."""
        assert parse_counter_instance_pid("pid_20624_luid_0x00000000_0x0000E1DE_phys_0") == 20624

    def test_non_matching_instance_yields_none(self) -> None:
        """Aggregate instances (e.g. _Total) carry no PID and are skipped."""
        assert parse_counter_instance_pid("total") is None


class TestWorkerPagingAssessment:
    """The paging verdict: worker attribution and threshold behavior."""

    def test_only_worker_pids_over_threshold_count(self) -> None:
        """A foreign process's demoted memory is external pressure, not the worker's verdict."""
        sample = GpuProcessMemorySample(
            timestamp=1.0,
            shared_mb_by_pid={101: 900.0, 202: 4000.0, 303: 80.0},
        )

        elevated = assess_worker_paging(sample, {101, 303}, shared_threshold_mb=256.0)

        assert elevated == {101: 900.0}

    def test_runtime_bookkeeping_stays_below_threshold(self) -> None:
        """The tens-of-MB shared mapping every CUDA process holds must not read as paging."""
        sample = GpuProcessMemorySample(timestamp=1.0, shared_mb_by_pid={101: 82.0})

        assert assess_worker_paging(sample, {101}, shared_threshold_mb=256.0) == {}


class TestMonitorLifecycle:
    """Graceful-absence and background-sampling contracts of the monitor."""

    def test_no_reader_means_no_thread_and_no_sample(self, monkeypatch) -> None:  # noqa: ANN001
        """On a host without the counter source the monitor is inert: no thread, latest() stays None."""
        monkeypatch.setattr(wddm_module, "_make_pdh_reader", lambda: None)
        monitor = WddmPagingMonitor()

        monitor.start()

        assert monitor._thread is None
        assert monitor.latest() is None
        monitor.stop()

    def test_injected_reader_updates_latest(self) -> None:
        """With a working reader the newest sample is exposed; a failing read never propagates."""
        produced = GpuProcessMemorySample(timestamp=5.0, shared_mb_by_pid={7: 512.0})
        delivered = threading.Event()

        def read() -> GpuProcessMemorySample:
            delivered.set()
            return produced

        monitor = WddmPagingMonitor(interval_seconds=0.01, read_sample=read)
        monitor.start()
        try:
            assert delivered.wait(timeout=2.0)
            deadline_reached = threading.Event()
            for _ in range(200):
                if monitor.latest() is produced:
                    break
                deadline_reached.wait(0.01)
            assert monitor.latest() is produced
        finally:
            monitor.stop()

    def test_reader_exception_is_swallowed(self) -> None:
        """A reader that raises must not kill the sampling thread or escape to the caller."""
        calls = threading.Event()

        def read() -> GpuProcessMemorySample | None:
            calls.set()
            raise RuntimeError("pdh went away")

        monitor = WddmPagingMonitor(interval_seconds=0.01, read_sample=read)
        monitor.start()
        try:
            assert calls.wait(timeout=2.0)
            assert monitor.latest() is None
        finally:
            monitor.stop()
