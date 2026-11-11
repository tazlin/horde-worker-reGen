"""Tests for the WDDM demand-paging telemetry and its scheduler verdict.

The monitor reads the per-process ``GPU Process Memory`` counters (the data behind Task Manager's
"Shared GPU memory" column). A worker child whose shared usage climbs while its card's dedicated memory
is short is being demand-paged, with PID-level attribution separating the worker's own over-commit from
external VRAM pressure. Shared usage while dedicated memory is free is host memory registered with the
driver by the process, and it does not count. The source is Windows-only, and the contract is graceful
absence. On a host without the counters the monitor does not collect anything, and every consumer reads
None/False and never an exception.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import Mock

from loguru import logger

from horde_worker_regen.process_management._internal import util as log_util
from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
from horde_worker_regen.process_management.resources.device_memory_source import (
    child_reported_reading,
    nvml_reading,
)
from horde_worker_regen.process_management.resources.run_metrics import ResourceStateKind
from horde_worker_regen.utils import wddm_paging_monitor as wddm_module
from horde_worker_regen.utils.wddm_paging_monitor import (
    GpuProcessMemorySample,
    SharedUsageVerdict,
    WddmPagingMonitor,
    WorkerPagingAssessment,
    WorkerPagingFigures,
    adapter_dedicated_used_mb,
    assess_worker_paging,
    parse_counter_instance_adapter,
    parse_counter_instance_pid,
    sum_by_adapter,
    sum_by_pid,
    sum_by_pid_and_adapter,
)
from tests.process_management.conftest import make_mock_process_info, make_testable_process_manager


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


def _figures(
    pid: int,
    *,
    shared_mb: float,
    device_index: int | None = 0,
    device_free_mb: float | None = 3130.0,
) -> WorkerPagingFigures:
    """Return one process's figures on a 16 GB card, with 3130 MB free unless stated."""
    return WorkerPagingFigures(
        pid=pid,
        shared_mb=shared_mb,
        local_mb=11281.0,
        adapter_dedicated_used_mb=12952.0,
        device_index=device_index,
        device_free_mb=device_free_mb,
        device_total_mb=None if device_free_mb is None else 16376.0,
    )


class TestWorkerPagingAssessment:
    """The paging verdict: threshold behavior and the card's dedicated memory."""

    _MARGINS = {0: 819.0}

    def _assess(self, *figures: WorkerPagingFigures) -> WorkerPagingAssessment:
        return assess_worker_paging(
            {each.pid: each for each in figures},
            shared_threshold_mb=256.0,
            margin_mb_by_device=self._MARGINS,
        )

    def test_runtime_bookkeeping_stays_below_threshold(self) -> None:
        """Every CUDA process's tens-of-MB shared mapping is below the threshold and is not judged."""
        assessment = self._assess(_figures(101, shared_mb=82.0, device_free_mb=100.0))

        assert assessment.verdict_by_pid == {}
        assert assessment.elevated_shared_mb_by_pid == {}

    def test_shared_usage_with_free_dedicated_memory_does_not_count(self) -> None:
        """Host memory registered for offloaded weights raises shared usage while the card has gigabytes free."""
        assessment = self._assess(_figures(101, shared_mb=4762.0, device_free_mb=3130.0))

        assert assessment.verdict_by_pid == {101: SharedUsageVerdict.NOT_COUNTED_CARD_HAS_ROOM}
        assert assessment.elevated_shared_mb_by_pid == {}

    def test_shared_usage_with_dedicated_memory_at_or_under_the_margin_counts(self) -> None:
        """A demotion needs the card to be short, so a process counts when free memory is at or under the margin."""
        assessment = self._assess(
            _figures(101, shared_mb=900.0, device_free_mb=212.0),
            _figures(102, shared_mb=400.0, device_free_mb=819.0),
        )

        assert assessment.verdict_by_pid == {
            101: SharedUsageVerdict.COUNTED_DEDICATED_SHORT,
            102: SharedUsageVerdict.COUNTED_DEDICATED_SHORT,
        }
        assert assessment.elevated_shared_mb_by_pid == {101: 900.0, 102: 400.0}

    def test_without_a_card_reading_shared_usage_counts_alone(self) -> None:
        """With a missing reading, an unknown device or a card without a margin, shared usage counts alone."""
        assessment = self._assess(
            _figures(101, shared_mb=900.0, device_free_mb=None),
            _figures(102, shared_mb=900.0, device_index=None),
            _figures(103, shared_mb=900.0, device_index=1),
        )

        assert set(assessment.verdict_by_pid.values()) == {SharedUsageVerdict.COUNTED_CARD_UNREAD}
        assert assessment.elevated_shared_mb_by_pid == {101: 900.0, 102: 900.0, 103: 900.0}


class TestWorkerPagingFigures:
    """The figures recorded beside a worker process's shared usage at each sample."""

    def test_describe_and_inputs_hold_every_figure(self) -> None:
        """One process's shared, local, card and adapter figures read in one span of text and one flat row."""
        figures = WorkerPagingFigures(
            pid=23320,
            shared_mb=4470.4,
            local_mb=11980.0,
            adapter_dedicated_used_mb=15219.0,
            device_index=0,
            device_free_mb=1156.0,
            device_total_mb=16375.0,
        )

        assert figures.describe() == (
            "pid 23320: 4470MB shared, 11980MB local, card 0 free 1156MB of 16375MB, adapter dedicated 15219MB"
        )
        assert figures.as_inputs() == {
            "pid": 23320,
            "shared_mb": 4470.4,
            "local_mb": 11980.0,
            "adapter_dedicated_used_mb": 15219.0,
            "device_index": 0,
            "device_free_mb": 1156.0,
            "device_total_mb": 16375.0,
        }

    def test_missing_figures_read_as_unread(self) -> None:
        """A figure missing from the sample or the parent's readings is printed as unread and recorded as None."""
        figures = WorkerPagingFigures(
            pid=7,
            shared_mb=300.0,
            local_mb=None,
            adapter_dedicated_used_mb=None,
            device_index=1,
            device_free_mb=None,
            device_total_mb=None,
        )

        assert figures.describe() == (
            "pid 7: 300MB shared, unread local, card 1 free unread of unread, adapter dedicated unread"
        )
        assert figures.as_inputs()["device_free_mb"] is None
        assert figures.as_inputs()["local_mb"] is None


class TestPagingSampleFigures:
    """The parent records each sample's figures in the log and on the verdict's stats rows."""

    _PID = 4242
    _ADAPTER = "0x00000000_0x000165f8"

    _SHORT_FREE_MB = 212.0
    """Free dedicated memory under the admission margin of a 16 GB card on any platform."""
    _ROOMY_FREE_MB = 1156.0
    """Free dedicated memory over the admission margin of a 16 GB card on any platform."""

    @staticmethod
    def _manager(*, device_free_mb: float | None = _SHORT_FREE_MB) -> HordeWorkerProcessManager:
        pm = make_testable_process_manager()
        process = make_mock_process_info(0, state=HordeProcessState.INFERENCE_STARTING)
        process.device_index = 0
        process.os_pid = TestPagingSampleFigures._PID
        pm._process_map.clear()
        pm._process_map[0] = process  # type: ignore[index]
        pm._last_device_reading_by_device.clear()
        if device_free_mb is not None:
            pm._last_device_reading_by_device[0] = nvml_reading(free_mb=device_free_mb, total_mb=16375.0)
        pm._run_metrics.record_resource_state = Mock(return_value=None)  # type: ignore[method-assign]
        return pm

    _samples_taken = 0

    @classmethod
    def _sample(cls, *, shared_mb: float, pid: int | None = None) -> GpuProcessMemorySample:
        """Return a fresh sample, stamped later than the last because the clock may not move between two calls."""
        pid = cls._PID if pid is None else pid
        cls._samples_taken += 1
        return GpuProcessMemorySample(
            timestamp=time.time() + cls._samples_taken * 0.001,
            shared_mb_by_pid={pid: shared_mb},
            local_mb_by_pid={pid: 11980.0},
            local_mb_by_pid_and_adapter={(pid, cls._ADAPTER): 11980.0},
            adapter_dedicated_mb={cls._ADAPTER: 15219.0},
        )

    @staticmethod
    def _evaluate(pm: HordeWorkerProcessManager, sample: GpuProcessMemorySample) -> None:
        pm._wddm_paging_monitor.latest = Mock(return_value=sample)  # type: ignore[method-assign]
        pm._evaluate_wddm_paging()

    @contextmanager
    def _capture(self) -> Iterator[list[tuple[str, str]]]:
        records: list[tuple[str, str]] = []
        handler_id = logger.add(
            lambda m: records.append((m.record["level"].name, m.record["message"])),
            level="TRACE",
        )
        try:
            yield records
        finally:
            logger.remove(handler_id)

    @staticmethod
    def _sample_lines(records: list[tuple[str, str]]) -> list[tuple[str, str]]:
        return [(level, message) for level, message in records if message.startswith("WDDM paging sample")]

    @staticmethod
    def _detections(records: list[tuple[str, str]]) -> list[str]:
        return [
            message for level, message in records if level == "WARNING" and "WDDM demand-paging detected" in message
        ]

    def test_a_short_card_fires_after_two_samples_and_the_detection_holds_the_card_figures(self) -> None:
        """Shared usage over the threshold with the card short counts, and the verdict fires on the second sample."""
        pm = self._manager()

        with self._capture() as records:
            self._evaluate(pm, self._sample(shared_mb=4470.0))
            assert not self._detections(records)
            assert pm._inference_scheduler.retention.wddm_paging_active is False
            self._evaluate(pm, self._sample(shared_mb=4470.0))

        assert pm._inference_scheduler.retention.wddm_paging_active is True
        sample_lines = self._sample_lines(records)
        assert [level for level, _ in sample_lines] == ["DEBUG", "DEBUG"]
        figures_text = (
            f"pid {self._PID}: 4470MB shared, 11980MB local, card 0 free 212MB of 16375MB, adapter dedicated 15219MB"
        )
        assert sample_lines[1][1].endswith(f"elevated streak 2): {figures_text} (counted, dedicated memory short)")
        detections = self._detections(records)
        assert len(detections) == 1
        assert f"({figures_text}): shared GPU usage over the threshold" in detections[0]

        record_state = pm._run_metrics.record_resource_state
        assert isinstance(record_state, Mock)
        record_state.assert_called_once()
        assert record_state.call_args.kwargs["state_kind"] is ResourceStateKind.WDDM_PAGING
        assert record_state.call_args.kwargs["state"] == "active"
        assert record_state.call_args.kwargs["inputs"] == {
            "elevated_streak": 2,
            "elevated_pids": 1,
            "pid": self._PID,
            "shared_mb": 4470.0,
            "local_mb": 11980.0,
            "adapter_dedicated_used_mb": 15219.0,
            "device_index": 0,
            "device_free_mb": 212.0,
            "device_total_mb": 16375.0,
        }

    def test_shared_usage_on_a_card_with_room_does_not_count(self) -> None:
        """Gigabytes of shared usage with dedicated memory free is host memory, so the verdict stays off."""
        pm = self._manager(device_free_mb=self._ROOMY_FREE_MB)

        with self._capture() as records:
            for _ in range(3):
                self._evaluate(pm, self._sample(shared_mb=4762.0))

        assert pm._wddm_elevated_streak == 0
        assert pm._inference_scheduler.retention.wddm_paging_active is False
        assert not self._detections(records)
        sample_lines = self._sample_lines(records)
        assert [level for level, _ in sample_lines] == ["DEBUG", "DEBUG", "DEBUG"]
        assert all(message.endswith("(not counted, card has room)") for _, message in sample_lines)
        record_state = pm._run_metrics.record_resource_state
        assert isinstance(record_state, Mock)
        record_state.assert_not_called()

    def test_a_stale_reading_with_room_falls_back_to_shared_usage_alone(self) -> None:
        """A reading older than the sample interval cannot suppress the verdict, and the line shows its age."""
        pm = self._manager(device_free_mb=self._ROOMY_FREE_MB)
        stale = pm._last_device_reading_by_device[0]
        pm._last_device_reading_by_device[0] = replace(stale, measured_at=time.monotonic() - 60.0)

        with self._capture() as records:
            self._evaluate(pm, self._sample(shared_mb=4762.0))
            assert pm._inference_scheduler.retention.wddm_paging_active is False
            self._evaluate(pm, self._sample(shared_mb=4762.0))

        assert pm._inference_scheduler.retention.wddm_paging_active is True
        assert len(self._detections(records)) == 1
        last_line = self._sample_lines(records)[1][1]
        assert "card 0 free unread of unread (reading 60." in last_line
        assert last_line.endswith("s old), adapter dedicated 15219MB (counted, card unread)")

    def test_without_a_device_wide_card_reading_shared_usage_counts_alone(self) -> None:
        """A missing reading, or a process report counting a sibling's memory as free, keeps the cautious verdict."""
        for reading in (
            None,
            child_reported_reading(
                reported_free_mb=self._ROOMY_FREE_MB,
                reported_total_mb=16375.0,
                committed_mb=0.0,
                platform="win32",
            ),
        ):
            pm = self._manager(device_free_mb=None)
            if reading is not None:
                pm._last_device_reading_by_device[0] = reading

            with self._capture() as records:
                self._evaluate(pm, self._sample(shared_mb=900.0))
                self._evaluate(pm, self._sample(shared_mb=900.0))

            assert pm._inference_scheduler.retention.wddm_paging_active is True
            assert len(self._detections(records)) == 1
            assert self._sample_lines(records)[1][1].endswith(
                "card 0 free unread of unread, adapter dedicated 15219MB (counted, card unread)"
            )

    def test_samples_below_the_threshold_are_time_boxed(self, monkeypatch) -> None:  # noqa: ANN001
        """A quiet sample logs at DEBUG once per interval and at TRACE in between."""
        monkeypatch.setattr(log_util, "_log_throttle_last_emission", {})
        pm = self._manager()

        with self._capture() as records:
            self._evaluate(pm, self._sample(shared_mb=80.0))
            self._evaluate(pm, self._sample(shared_mb=80.0))

        assert [level for level, _ in self._sample_lines(records)] == ["DEBUG", "TRACE"]

    def test_the_cleared_row_holds_the_figures_of_the_fall(self) -> None:
        """The clearing sample's figures go on the cleared row, so the row shows the shared usage dropping."""
        pm = self._manager()
        self._evaluate(pm, self._sample(shared_mb=4470.0))
        self._evaluate(pm, self._sample(shared_mb=4470.0))

        self._evaluate(pm, self._sample(shared_mb=80.0))

        record_state = pm._run_metrics.record_resource_state
        assert isinstance(record_state, Mock)
        cleared = record_state.call_args.kwargs
        assert cleared["state"] == "cleared"
        assert cleared["inputs"]["elevated_pids"] == 0
        assert cleared["inputs"]["shared_mb"] == 80.0
        assert cleared["inputs"]["device_free_mb"] == 212.0

    def test_a_sample_without_a_worker_process_logs_nothing_and_does_not_count(self) -> None:
        """A foreign process's shared usage is external pressure, even on a short card."""
        pm = self._manager()

        with self._capture() as records:
            self._evaluate(pm, self._sample(shared_mb=4470.0, pid=900))
            self._evaluate(pm, self._sample(shared_mb=4470.0, pid=900))

        assert not self._sample_lines(records)
        assert pm._wddm_elevated_streak == 0
        assert not self._detections(records)


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
