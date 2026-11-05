"""The worker footprint subtracted by the foreign reading, and the log line written when the floor moves."""

from __future__ import annotations

from loguru import logger

from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.device_memory_source import (
    DeviceMemoryReading,
    DeviceMemorySourceKind,
    ReadingProvenance,
)
from tests.process_management.conftest import make_mock_process_info
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


def _reading(process_used_mb: dict[int, float] | None) -> DeviceMemoryReading:
    """A device-wide reading of a 24 GB card with 12 GB free and the given per-process usage."""
    return DeviceMemoryReading(
        free_mb=12000.0,
        total_mb=24000.0,
        source=DeviceMemorySourceKind.NVML,
        provenance=ReadingProvenance.DEVICE_WIDE,
        process_used_mb=process_used_mb,
    )


def _scheduler_with_worker_pids(*pids: int):  # noqa: ANN202
    """A scheduler whose process map holds one inference process per pid, all on card 0."""
    processes = {}
    for index, pid in enumerate(pids):
        process_info = make_mock_process_info(index, model_name="stable_diffusion", device_index=0)
        process_info.os_pid = pid
        processes[index] = process_info
    return _make_inference_scheduler(process_map=ProcessMap(processes))


class TestWorkerFootprintSource:
    """Each branch of the footprint reading records which figure it is."""

    def test_the_driver_usage_over_listed_worker_pids_is_the_footprint(self) -> None:
        """With worker pids listed by the driver, the footprint is the sum of their usage, from the driver."""
        scheduler = _scheduler_with_worker_pids(101, 102)
        footprint = scheduler._foreign_reading_worker_footprint(
            0,
            device_reading=_reading({101: 3000.0, 102: 2000.0, 999: 7000.0}),
            ledger_footprint_mb=1234.0,
        )
        assert footprint.footprint_mb == 5000.0
        assert footprint.source == "the driver's per-process usage"
        assert footprint.listed_pids == 2

    def test_the_ledger_stands_in_when_the_driver_reports_no_per_process_usage(self) -> None:
        """Without per-process usage the ledger figure is the footprint and the source records why."""
        scheduler = _scheduler_with_worker_pids(101)
        footprint = scheduler._foreign_reading_worker_footprint(
            0,
            device_reading=_reading(None),
            ledger_footprint_mb=1234.0,
        )
        assert footprint.footprint_mb == 1234.0
        assert footprint.source == "the ledger, since the driver reports no per-process usage"
        assert footprint.listed_pids == 0

    def test_the_ledger_stands_in_when_no_worker_pid_is_listed(self) -> None:
        """With per-process usage that lists no worker pid, the ledger figure is the footprint."""
        scheduler = _scheduler_with_worker_pids(101)
        footprint = scheduler._foreign_reading_worker_footprint(
            0,
            device_reading=_reading({999: 7000.0}),
            ledger_footprint_mb=1234.0,
        )
        assert footprint.footprint_mb == 1234.0
        assert footprint.source == "the ledger, since the driver lists no worker pid"
        assert footprint.listed_pids == 0


class TestFloorMoveLog:
    """The floor-move line is written once per move with the reading behind it."""

    def test_a_move_is_logged_once_with_the_footprint_source(self) -> None:
        """A repeat or sub-megabyte reading is not logged, and each move states the footprint and its source."""
        scheduler = _scheduler_with_worker_pids(101)
        footprint = scheduler._foreign_reading_worker_footprint(
            0,
            device_reading=_reading(None),
            ledger_footprint_mb=2598.0,
        )
        lines: list[str] = []
        sink_id = logger.add(lambda message: lines.append(str(message)), level="DEBUG")
        try:
            for floor_mb in (2953.0, 2953.0, 2953.4, 3272.0, 3272.0):
                scheduler._log_foreign_floor_move(
                    0,
                    floor_mb,
                    total_vram_mb=24074.0,
                    device_free_mb=12204.0,
                    footprint=footprint,
                    ledger_footprint_mb=2598.0,
                )
        finally:
            logger.remove(sink_id)
        moves = [line for line in lines if "Foreign floor on device 0 moved" in line]
        assert len(moves) == 2
        assert "moved from None to 2953MB" in moves[0]
        assert "moved from 2953MB to 3272MB" in moves[1]
        assert "worker footprint 2598MB from the ledger, since the driver reports no per-process usage" in moves[1]
        assert "ledger 2598MB, reading 9272MB." in moves[1]

    def test_an_unknown_floor_is_logged_when_it_appears_and_when_it_goes(self) -> None:
        """The first reading is logged even when unknown, and a floor that becomes unknown again is logged too."""
        scheduler = _scheduler_with_worker_pids(101)
        footprint = scheduler._foreign_reading_worker_footprint(0, device_reading=None, ledger_footprint_mb=0.0)
        lines: list[str] = []
        sink_id = logger.add(lambda message: lines.append(str(message)), level="DEBUG")
        try:
            for floor_mb in (None, None, 400.0, None):
                scheduler._log_foreign_floor_move(
                    0,
                    floor_mb,
                    total_vram_mb=None,
                    device_free_mb=None,
                    footprint=footprint,
                    ledger_footprint_mb=0.0,
                )
        finally:
            logger.remove(sink_id)
        moves = [line for line in lines if "Foreign floor on device 0 moved" in line]
        assert [line.split("moved ")[1].split(":")[0] for line in moves] == [
            "from None to None",
            "from None to 400MB",
            "from 400MB to None",
        ]
        assert "total None, device free None" in moves[1]
        assert "reading None." in moves[1]
