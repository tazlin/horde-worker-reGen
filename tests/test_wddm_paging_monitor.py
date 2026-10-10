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
    adapter_dedicated_used_mb,
    assess_worker_paging,
    parse_counter_instance_adapter,
    parse_counter_instance_pid,
    sum_by_adapter,
    sum_by_pid,
    sum_by_pid_and_adapter,
)


class TestInstanceParsing:
    """PID extraction from GPU Process Memory counter instance names."""

    def test_parses_pid_from_counter_instance(self) -> None:
        """The PID embeds in the instance name between adapter/segment qualifiers."""
        assert parse_counter_instance_pid("pid_20624_luid_0x00000000_0x0000E1DE_phys_0") == 20624

    def test_non_matching_instance_yields_none(self) -> None:
        """Aggregate instances (e.g. _Total) carry no PID and are skipped."""
        assert parse_counter_instance_pid("total") is None


class TestAdapterDedicatedUsage:
    """The adapter a set of processes runs on, and the dedicated memory every process holds on it."""

    _CARD = "0x00000000_0x000165f8"
    _OTHER = "0x00000000_0x0001778a"

    def test_parses_the_adapter_from_both_instance_shapes(self) -> None:
        """An adapter instance and a process instance name the adapter the same way, in either case."""
        assert parse_counter_instance_adapter("luid_0x00000000_0x000165F8_phys_0") == self._CARD
        assert parse_counter_instance_adapter("pid_26856_luid_0x00000000_0x000165f8_phys_0") == self._CARD
        assert parse_counter_instance_adapter("total") is None

    def test_instances_are_summed_by_what_they_name(self) -> None:
        """One process can hold several instances on an adapter, and they are summed for each key."""
        by_instance = {
            f"pid_101_luid_{self._CARD}_phys_0": 4000.0,
            f"pid_101_luid_{self._CARD}_phys_1": 500.0,
            f"pid_202_luid_{self._OTHER}_phys_0": 300.0,
            "total": 9999.0,
        }

        assert sum_by_pid(by_instance) == {101: 4500.0, 202: 300.0}
        assert sum_by_pid_and_adapter(by_instance) == {(101, self._CARD): 4500.0, (202, self._OTHER): 300.0}
        assert sum_by_adapter({f"luid_{self._CARD}_phys_0": 6200.0, "total": 1.0}) == {self._CARD: 6200.0}

    def test_the_adapter_is_found_from_the_processes_on_it(self) -> None:
        """The figure is the whole adapter's, other programs included, for the adapter the processes hold."""
        sample = GpuProcessMemorySample(
            timestamp=1.0,
            local_mb_by_pid_and_adapter={
                (101, self._CARD): 4000.0,
                (102, self._CARD): 1200.0,
                (900, self._CARD): 950.0,
                (901, self._OTHER): 80.0,
            },
            adapter_dedicated_mb={self._CARD: 6200.0, self._OTHER: 80.0},
        )

        assert adapter_dedicated_used_mb(sample, {101, 102}) == 6200.0
        assert adapter_dedicated_used_mb(sample, {901}) == 80.0

    def test_no_figure_before_the_processes_hold_memory_or_without_the_counter_set(self) -> None:
        """Processes that hold nothing yet name no adapter, and a host without the counters has no figure."""
        held = {(101, self._CARD): 4000.0}

        assert (
            adapter_dedicated_used_mb(GpuProcessMemorySample(1.0, adapter_dedicated_mb={self._CARD: 1.0}), {101})
            is None
        )
        assert adapter_dedicated_used_mb(GpuProcessMemorySample(1.0, local_mb_by_pid_and_adapter=held), {101}) is None


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
