"""Tests that reclaim LIFO ranking uses the dedicated VRAM-materialization stamp, not the report-time proxy."""

from __future__ import annotations

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources.reclaim_ladder import ReclaimRungKind, build_reclaim_ladder
from tests.process_management.conftest import make_mock_process_info
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


def _resident(process_id: int, *, materialized: float | None, last_received: float) -> object:
    info = make_mock_process_info(process_id, model_name=f"m{process_id}", state=HordeProcessState.WAITING_FOR_JOB)
    info.process_type = HordeProcessType.INFERENCE
    info.process_reserved_mb = 6000
    info.vram_materialized_monotonic = materialized
    info.last_received_timestamp = last_received
    return info


def test_materialization_stamp_drives_lifo_over_report_time() -> None:
    """The more-recently-materialized resident is evicted first even when its report time is older.

    ``last_received_timestamp`` refreshes on report traffic unrelated to materialization, so ranking by it
    scrambles eviction order; the dedicated stamp fixes it.
    """
    # A: materialized more recently (200) but its last report is OLD (10). B: materialized earlier (100) but
    # its last report is NEW (9999). Report-time ranking would put B first; the stamp must put A first.
    resident_a = _resident(1, materialized=200.0, last_received=10.0)
    resident_b = _resident(2, materialized=100.0, last_received=9999.0)
    scheduler = _make_inference_scheduler(process_map=ProcessMap({1: resident_a, 2: resident_b}))  # type: ignore[dict-item]

    ladder = build_reclaim_ladder(scheduler.build_reclaim_ladder_candidates(None))
    unloads = [rung.target_process_id for rung in ladder if rung.kind is ReclaimRungKind.UNLOAD_IDLE_MODEL]

    assert unloads == [1, 2], "the newest-materialized resident (pid 1) must be the first unload target"


def test_note_vram_materialized_sets_the_stamp() -> None:
    """The parent's VRAM-materialization observation stamps a monotonic time on the process."""
    proc = make_mock_process_info(1, model_name="m", state=HordeProcessState.WAITING_FOR_JOB)
    process_map = ProcessMap({1: proc})
    assert proc.vram_materialized_monotonic is None

    process_map.note_vram_materialized(1)

    assert proc.vram_materialized_monotonic is not None


def test_unset_stamp_falls_back_to_the_report_time_proxy() -> None:
    """A resident with no stamp still ranks by its report-time proxy, comparably with stamped residents."""
    # Neither resident has a stamp; ranking falls back to report time, newest-first.
    resident_old = _resident(1, materialized=None, last_received=10.0)
    resident_new = _resident(2, materialized=None, last_received=20.0)
    scheduler = _make_inference_scheduler(process_map=ProcessMap({1: resident_old, 2: resident_new}))  # type: ignore[dict-item]

    ladder = build_reclaim_ladder(scheduler.build_reclaim_ladder_candidates(None))
    unloads = [rung.target_process_id for rung in ladder if rung.kind is ReclaimRungKind.UNLOAD_IDLE_MODEL]

    assert unloads == [2, 1], "with no stamp, the newest report time (pid 2) ranks first"
