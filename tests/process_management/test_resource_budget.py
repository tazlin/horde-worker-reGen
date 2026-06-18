"""Tests for the worker-owned VRAM budget and its scheduler gating."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from horde_worker_regen.process_management import resource_budget
from horde_worker_regen.process_management.horde_model_map import HordeModelMap
from horde_worker_regen.process_management.job_tracker import JobTracker
from horde_worker_regen.process_management.messages import HordeControlFlag, HordeProcessState, ModelLoadState
from horde_worker_regen.process_management.process_map import ProcessMap
from horde_worker_regen.process_management.resource_budget import BudgetVerdict, RamBudget, VramBudget

from .conftest import (
    make_job_pop_response,
    make_mock_bridge_data,
    make_mock_process_info,
    track_popped_job_async,
)
from .test_inference_scheduling import _make_inference_scheduler


class TestVramBudget:
    """Unit tests for the VramBudget accountant itself (prediction stubbed)."""

    def test_cold_start_admits_when_no_telemetry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no VRAM telemetry yet, the budget admits so a cold worker never wedges."""
        monkeypatch.setattr(resource_budget, "predict_job_vram_mb", lambda job, baseline: 9999.0)
        budget = VramBudget(reserve_mb=2048.0)
        job = make_job_pop_response("stable_diffusion")
        verdict = budget.check_job(job, "stable_diffusion_1", free_vram_mb=None)
        assert verdict.fits is True
        assert verdict.available_mb is None

    def test_admits_when_estimate_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A None estimate means unknown cost; the budget admits rather than blocking blindly."""
        monkeypatch.setattr(resource_budget, "predict_job_vram_mb", lambda job, baseline: None)
        budget = VramBudget(reserve_mb=2048.0)
        job = make_job_pop_response("stable_diffusion")
        verdict = budget.check_job(job, None, free_vram_mb=500.0)
        assert verdict.fits is True
        assert verdict.predicted_mb is None

    def test_fits_when_free_covers_predicted_plus_reserve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Free VRAM at or above predicted + reserve fits."""
        monkeypatch.setattr(resource_budget, "predict_job_vram_mb", lambda job, baseline: 4000.0)
        budget = VramBudget(reserve_mb=2000.0)
        job = make_job_pop_response("stable_diffusion")
        assert budget.check_job(job, "x", free_vram_mb=6000.0).fits is True
        assert budget.check_job(job, "x", free_vram_mb=5999.0).fits is False

    def test_set_reserve_is_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Updating the reserve changes the verdict immediately (live config reload)."""
        monkeypatch.setattr(resource_budget, "predict_job_vram_mb", lambda job, baseline: 4000.0)
        budget = VramBudget(reserve_mb=2000.0)
        job = make_job_pop_response("stable_diffusion")
        assert budget.check_job(job, "x", free_vram_mb=5000.0).fits is False
        budget.set_reserve_mb(1000.0)
        assert budget.check_job(job, "x", free_vram_mb=5000.0).fits is True

    def test_ram_budget_fits_logic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RamBudget admits when available RAM covers predicted RAM plus reserve, else defers."""
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 6000.0)
        budget = RamBudget(reserve_mb=4096.0)
        job = make_job_pop_response("stable_diffusion")
        assert budget.check_job(job, "x", available_ram_mb=11000.0).fits is True
        assert budget.check_job(job, "x", available_ram_mb=9000.0).fits is False
        assert budget.check_job(job, "x", available_ram_mb=None).fits is True

    def test_verdict_reason_strings(self) -> None:
        """The verdict reason renders the relevant branch for logging."""
        assert (
            "cold start" in BudgetVerdict(fits=True, predicted_mb=None, available_mb=None, reserve_mb=2048.0).reason()
        )
        assert (
            "no burden estimate"
            in BudgetVerdict(fits=True, predicted_mb=None, available_mb=1000.0, reserve_mb=2048.0).reason()
        )
        assert (
            "does NOT fit"
            in BudgetVerdict(fits=False, predicted_mb=4000.0, available_mb=1000.0, reserve_mb=2048.0).reason()
        )
        assert "fits" in BudgetVerdict(fits=True, predicted_mb=1000.0, available_mb=8000.0, reserve_mb=2048.0).reason()


def _budget_bridge_data() -> Mock:
    """Mock bridge data with the VRAM budget enabled and real numeric reserves."""
    return make_mock_bridge_data(
        enable_vram_budget=True,
        vram_reserve_mb=2000,
        ram_reserve_mb=4096,
        image_models_to_load=["model_a", "model_b"],
    )


class TestPreloadBudgetGate:
    """Integration tests for the preload-time VRAM budget gate inside the scheduler."""

    async def test_preload_deferred_and_reclaims_when_over_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the next model will not fit, preload is deferred and idle VRAM is reclaimed."""
        monkeypatch.setattr(resource_budget, "predict_job_vram_mb", lambda job, baseline: 8000.0)

        spare = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        # A second, idle process holding a different resident model: the eviction candidate. It reports
        # the (low) device-wide free VRAM the budget reads.
        resident = make_mock_process_info(1, model_name="model_b", state=HordeProcessState.WAITING_FOR_JOB)
        resident.total_vram_mb = 16000
        resident.vram_usage_mb = 15000  # 1000 MB free, well under 8000 + 2000
        process_map = ProcessMap({0: spare, 1: resident})

        job_tracker = JobTracker()
        job = make_job_pop_response("model_a")
        await track_popped_job_async(job_tracker, job)

        scheduler = _make_inference_scheduler(
            process_map=process_map,
            job_tracker=job_tracker,
            bridge_data=_budget_bridge_data(),
            max_concurrent=2,
            max_inference=2,
        )

        assert scheduler.preload_models() is False
        # The spare process was NOT told to preload...
        assert spare.last_control_flag != HordeControlFlag.PRELOAD_MODEL
        # ...and the idle resident model was evicted to reclaim VRAM (residency overridden under pressure).
        assert resident.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_VRAM

    async def test_preload_proceeds_when_within_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With ample free VRAM and RAM the budget admits and the preload is sent."""
        monkeypatch.setattr(resource_budget, "predict_job_vram_mb", lambda job, baseline: 4000.0)
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 1000.0)

        spare = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        spare.total_vram_mb = 16000
        spare.vram_usage_mb = 2000  # 14000 MB free, covers 4000 + 2000
        process_map = ProcessMap({0: spare})

        job_tracker = JobTracker()
        job = make_job_pop_response("model_a")
        await track_popped_job_async(job_tracker, job)

        scheduler = _make_inference_scheduler(
            process_map=process_map,
            job_tracker=job_tracker,
            bridge_data=_budget_bridge_data(),
            max_concurrent=2,
            max_inference=2,
        )

        assert scheduler.preload_models() is True
        assert spare.last_control_flag == HordeControlFlag.PRELOAD_MODEL

    async def test_preload_deferred_when_over_ram_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """VRAM fits but RAM does not: the preload is deferred and idle RAM is reclaimed."""
        monkeypatch.setattr(resource_budget, "predict_job_vram_mb", lambda job, baseline: 1000.0)
        monkeypatch.setattr(resource_budget, "predict_job_ram_mb", lambda job, baseline: 50000.0)

        spare = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        spare.total_vram_mb = 16000
        spare.vram_usage_mb = 1000  # ample free VRAM, so the VRAM gate passes
        # A second idle process holding a resident model: the RAM eviction candidate.
        resident = make_mock_process_info(1, model_name="model_b", state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: spare, 1: resident})

        job_tracker = JobTracker()
        job = make_job_pop_response("model_a")
        await track_popped_job_async(job_tracker, job)

        horde_model_map = HordeModelMap(root={})
        horde_model_map.update_entry(
            horde_model_name="model_b",
            load_state=ModelLoadState.LOADED_IN_RAM,
            process_id=1,
        )

        scheduler = _make_inference_scheduler(
            process_map=process_map,
            horde_model_map=horde_model_map,
            job_tracker=job_tracker,
            bridge_data=_budget_bridge_data(),
            max_concurrent=2,
            max_inference=2,
        )
        # Force a low available-RAM reading so the RAM budget defers deterministically.
        monkeypatch.setattr(scheduler, "_measured_available_ram_mb", lambda: 8000.0)

        assert scheduler.preload_models() is False
        assert spare.last_control_flag != HordeControlFlag.PRELOAD_MODEL
        assert resident.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_RAM

    async def test_disabled_budget_ignores_low_vram(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the budget disabled, a low-VRAM device does not defer the preload."""
        monkeypatch.setattr(resource_budget, "predict_job_vram_mb", lambda job, baseline: 8000.0)

        spare = make_mock_process_info(0, model_name=None, state=HordeProcessState.WAITING_FOR_JOB)
        spare.total_vram_mb = 16000
        spare.vram_usage_mb = 15500  # only 500 MB free
        process_map = ProcessMap({0: spare})

        job_tracker = JobTracker()
        job = make_job_pop_response("model_a")
        await track_popped_job_async(job_tracker, job)

        scheduler = _make_inference_scheduler(
            process_map=process_map,
            job_tracker=job_tracker,
            bridge_data=make_mock_bridge_data(enable_vram_budget=False, image_models_to_load=["model_a"]),
            max_concurrent=2,
            max_inference=2,
        )

        assert scheduler.preload_models() is True
        assert spare.last_control_flag == HordeControlFlag.PRELOAD_MODEL
