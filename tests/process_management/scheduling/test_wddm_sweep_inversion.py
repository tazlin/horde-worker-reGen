"""Tests for the WDDM paging rising-edge sweep: LIFO newest-idle-first, active-sampler immune, PDH not protected."""

from __future__ import annotations

from horde_worker_regen.process_management.ipc.messages import HordeProcessState
from horde_worker_regen.process_management.lifecycle.horde_process import HordeProcessType
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from tests.process_management.conftest import make_mock_process_info
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler


def _idle_resident(process_id: int, *, model: str, reserved_mb: int, materialized: float) -> object:
    info = make_mock_process_info(process_id, model_name=model, state=HordeProcessState.WAITING_FOR_JOB)
    info.process_type = HordeProcessType.INFERENCE
    info.process_reserved_mb = reserved_mb
    info.vram_materialized_monotonic = materialized
    return info


def test_pdh_flagged_idle_newcomer_is_the_first_sweep_target() -> None:
    """The newest idle resident (the PDH-flagged newcomer) is evicted first; it is never protected."""
    newcomer = _idle_resident(1, model="new", reserved_mb=6000, materialized=100.0)  # newest
    older = _idle_resident(2, model="old", reserved_mb=6000, materialized=50.0)  # older
    scheduler = _make_inference_scheduler(process_map=ProcessMap({1: newcomer, 2: older}))  # type: ignore[dict-item]

    swept: list[int] = []
    scheduler.unload_idle_model = lambda process_id, device_index=None: swept.append(process_id) or True  # type: ignore[assignment,method-assign]

    # PDH flags the idle newcomer (pid 1), exactly the process the old sweep would have protected.
    scheduler.note_wddm_paging({newcomer.os_pid: 512.0}, active=True)

    assert swept, "the sweep must reclaim idle residents"
    assert swept[0] == 1, "the newest idle resident (the PDH-flagged newcomer) must be the first target"
    assert set(swept) == {1, 2}


def test_actively_sampling_process_is_never_swept() -> None:
    """A busy (sampling) process is never a sweep target, whatever PDH flagged."""
    idle = _idle_resident(1, model="idle", reserved_mb=6000, materialized=50.0)
    busy = make_mock_process_info(2, model_name="busy", state=HordeProcessState.INFERENCE_STARTING)
    busy.process_type = HordeProcessType.INFERENCE  # type: ignore[attr-defined]
    busy.process_reserved_mb = 6000  # type: ignore[attr-defined]
    busy.vram_materialized_monotonic = 100.0  # type: ignore[attr-defined]
    scheduler = _make_inference_scheduler(process_map=ProcessMap({1: idle, 2: busy}))  # type: ignore[dict-item]

    swept: list[int] = []
    scheduler.unload_idle_model = lambda process_id, device_index=None: swept.append(process_id) or True  # type: ignore[assignment,method-assign]

    # PDH flags the busy sampler (pid 2) AND the idle one; the busy one must still never be swept.
    scheduler.note_wddm_paging({busy.os_pid: 512.0, idle.os_pid: 300.0}, active=True)

    assert 2 not in swept, "an actively-sampling process must never be swept"
    assert swept == [1]


def test_paging_active_flag_still_denies_retention() -> None:
    """The rising-edge sweep rework preserves the retention-denial behavior while paging is active."""
    idle = _idle_resident(1, model="idle", reserved_mb=6000, materialized=50.0)
    scheduler = _make_inference_scheduler(process_map=ProcessMap({1: idle}))  # type: ignore[dict-item]
    scheduler.unload_idle_model = lambda process_id, device_index=None: True  # type: ignore[assignment,method-assign]

    scheduler.note_wddm_paging({idle.os_pid: 512.0}, active=True)
    assert scheduler._wddm_paging_active is True

    scheduler.note_wddm_paging({}, active=False)
    assert scheduler._wddm_paging_active is False
