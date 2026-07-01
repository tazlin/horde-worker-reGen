"""Regression guard: an already-resident whole-card model must re-establish sole residency before it samples.

The whole-card residency machinery (evict sibling models / stop idle sibling processes, move safety off-GPU)
is reached only on the *preload* path, when the heavy model still has to be loaded. ``preload_models`` skips a
model that is already resident, and the dispatch path (``get_next_job_and_process``) launches an already-
resident model's job with only the concurrency and overlap gates.

That leaves a gap. A whole-card model (an EXTRA_LARGE baseline such as Flux fp8 on a 16GB card) whose forecast
is ``needs_exclusive_residency`` can be resident on one process while sibling processes still hold their own
models resident (for example after a prior whole-card job drained and the siblings were restored). A fresh job
for the resident heavy model is then dispatched to sample alongside those sibling weights. On a card too small
to co-reside them, free VRAM collapses to zero the moment the heavy model's sampling activations allocate, the
driver streams activations to host RAM, and the step rate drops by an order of magnitude: the job overruns the
server's processing deadline and is dropped, which repeated across such jobs escalates to forced maintenance.

The heavy model is entitled to the same sole residency whether it is being loaded or is already resident. These
tests pin that a scheduling cycle for a resident whole-card head evicts the co-resident siblings and does not
launch the head to co-sample.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.ipc.messages import HordeControlFlag, HordeProcessState
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from horde_worker_regen.process_management.resources import resource_budget
from horde_worker_regen.process_management.scheduling.inference_scheduler import InferenceScheduler
from tests.process_management.conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    track_popped_job_async,
)
from tests.process_management.scheduling.test_inference_scheduling import _make_inference_scheduler

_DEVICE_TOTAL_VRAM_MB = 16375
_FLUX_WEIGHTS_MB = 11500.0
_FLUX_SAMPLING_PEAK_MB = _FLUX_WEIGHTS_MB + 2500  # reserve above the flat floor -> sole residency on 16GB
_PER_PROCESS_OVERHEAD_MB = 1288
_VRAM_RESERVE_MB = 2048
_RAM_RESERVE_MB = 4096

_FLUX_MODEL = "Flux.1-Schnell fp8 (Compact)"
_RESIDENT_SD15 = "Deliberate"
_RESIDENT_SDXL = "CyberRealistic Pony"


def _build_resident_whole_card_scheduler(
    *,
    free_mb: float,
) -> tuple[InferenceScheduler, ProcessMap, JobTracker, object, object]:
    """The heavy whole-card model already resident on one process, two siblings resident with SD/SDXL.

    Device-free reads ``free_mb`` (the siblings' models occupy the card). This is the post-restore state a
    resident-model dispatch lands in: nothing needs loading, so the preload path never re-examines residency.
    """
    proc_sd15 = make_mock_process_info(1, model_name=_RESIDENT_SD15, state=HordeProcessState.WAITING_FOR_JOB)
    proc_sdxl = make_mock_process_info(2, model_name=_RESIDENT_SDXL, state=HordeProcessState.WAITING_FOR_JOB)
    proc_flux = make_mock_process_info(3, model_name=_FLUX_MODEL, state=HordeProcessState.WAITING_FOR_JOB)
    for proc in (proc_sd15, proc_sdxl, proc_flux):
        proc.total_vram_mb = _DEVICE_TOTAL_VRAM_MB
        proc.vram_usage_mb = _DEVICE_TOTAL_VRAM_MB - free_mb
    process_map = ProcessMap({1: proc_sd15, 2: proc_sdxl, 3: proc_flux})

    job_tracker = JobTracker()
    bridge_data = make_mock_bridge_data(
        enable_vram_budget=True,
        whole_card_exclusive_residency=True,
        vram_reserve_mb=_VRAM_RESERVE_MB,
        ram_reserve_mb=_RAM_RESERVE_MB,
        vram_per_process_overhead_mb=_PER_PROCESS_OVERHEAD_MB,
        overbudget_exclusive_mode=True,
        safety_on_gpu=True,
        image_models_to_load=[_RESIDENT_SD15, _RESIDENT_SDXL, _FLUX_MODEL],
        max_threads=1,
    )
    scheduler = _make_inference_scheduler(
        process_map=process_map,
        job_tracker=job_tracker,
        bridge_data=bridge_data,
        max_concurrent=1,
        max_inference=3,
    )
    scheduler._process_lifecycle.is_safety_gpu_paused = False
    return scheduler, process_map, job_tracker, proc_sd15, proc_sdxl


class TestResidentWholeCardReDispatch:
    """A resident whole-card head must claim the card before sampling, exactly as a to-be-loaded one does."""

    async def test_resident_flux_head_reserves_card_not_cosamples(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The resident Flux head evicts the co-resident siblings and is not dispatched to co-sample."""
        monkeypatch.setattr(resource_budget, "predict_job_weight_mb", lambda job, baseline: _FLUX_WEIGHTS_MB)
        monkeypatch.setattr(
            resource_budget,
            "predict_job_sampling_vram_mb",
            lambda job, baseline: _FLUX_SAMPLING_PEAK_MB,
        )

        scheduler, _process_map, job_tracker, proc_sd15, proc_sdxl = _build_resident_whole_card_scheduler(
            free_mb=2000.0,
        )
        head_job = make_job_pop_response(_FLUX_MODEL)
        await track_popped_job_async(job_tracker, head_job)

        scheduler.preload_models()
        dispatch = await scheduler.get_next_job_and_process()

        assert scheduler._sibling_teardown_for_model == _FLUX_MODEL, (
            "a resident whole-card head must establish sole residency, not run co-resident with sibling models"
        )
        assert HordeControlFlag.UNLOAD_MODELS_FROM_VRAM in {
            proc_sd15.last_control_flag,
            proc_sdxl.last_control_flag,
        }, "the resident siblings must be evicted to clear the card for the heavy head"
        assert dispatch is None, "the head must defer until the card is cleared, not co-sample and thrash"
