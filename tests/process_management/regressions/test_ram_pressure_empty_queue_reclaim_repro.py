"""RAM-pressure reclaim must reach idle resident footprint even when the pending queue is empty.

While the host is under its RAM danger floor the worker holds job pops so intake stops adding pressure.
Held pops let the queue drain, and once no job is pending or in flight the only remaining footprint to
reclaim is the idle resident model (and the allocator pages a process retains after unloading it). The
degrade response must still reclaim that footprint with an empty queue: otherwise system RAM is never
returned, the host stays under its floor, the pop hold never lifts, and its skip counter climbs forever.

The contract these tests pin:

* Under pressure, an idle inference process holding a resident model is unloaded from RAM even when no
  job is pending or in flight (the reclaim is not gated on there being other queued work).
* Under pressure, the degrade response cycles an idle model-less process that still holds RAM after an
  unload, returning the allocator-retained pages to the OS, again without requiring queued work.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState, ModelLoadState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.models.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.scheduling.governance.actions import EvictIdleModels
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import make_mock_process_info
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_TOTAL_RAM_MB = 64000.0
_CRITICAL_AVAILABLE_RAM_MB = 500.0


def _pin_available_ram(scheduler: InferenceScheduler, monkeypatch: pytest.MonkeyPatch, available_mb: float) -> None:
    """Pin measured system RAM so the danger-floor verdict is deterministic on any host."""
    monkeypatch.setattr(scheduler, "_measured_available_ram_mb", lambda: available_mb)
    monkeypatch.setattr(scheduler, "_measured_total_ram_mb", lambda: _TOTAL_RAM_MB)


class TestUnderPressureReclaimWithEmptyQueue:
    """The under-pressure reclaim path is reached with no pending or in-flight work."""

    def test_idle_resident_model_is_unloaded_under_pressure_with_empty_queue(self) -> None:
        """A drained queue must not stop the pressure eviction from unloading an idle resident model."""
        model_name = "WAI-NSFW-illustrious-SDXL"
        process_info = make_mock_process_info(
            0,
            model_name=model_name,
            state=HordeProcessState.WAITING_FOR_JOB,
        )
        process_map = ProcessMap({0: process_info})
        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(
            horde_model_name=model_name,
            load_state=ModelLoadState.LOADED_IN_RAM,
            process_id=0,
        )
        scheduler = _make_inference_scheduler(
            process_map=process_map,
            horde_model_map=horde_model_map,
            job_tracker=JobTracker(),
        )

        reclaimed = scheduler.unload_models(under_pressure=True)

        assert reclaimed is True, "an idle resident model must be reclaimable under pressure with an empty queue"
        assert process_info.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_RAM

    def test_degrade_response_cycles_stale_ram_slot_with_empty_queue(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When no idle model remains to unload, the degrade response cycles an allocator-stuck idle slot."""
        process_info = make_mock_process_info(
            0,
            model_name=None,
            state=HordeProcessState.WAITING_FOR_JOB,
        )
        process_info.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_RAM
        process_info.ram_usage_bytes = 3 * 1024 * 1024 * 1024
        process_map = ProcessMap({0: process_info})
        scheduler = _make_inference_scheduler(process_map=process_map, job_tracker=JobTracker())
        _pin_available_ram(scheduler, monkeypatch, _CRITICAL_AVAILABLE_RAM_MB)

        scheduler._execute_governance_actions([EvictIdleModels()])

        scheduler._process_lifecycle._replace_inference_process.assert_called_once()
        _, kwargs = scheduler._process_lifecycle._replace_inference_process.call_args
        assert kwargs.get("intentional_reclaim") is True
