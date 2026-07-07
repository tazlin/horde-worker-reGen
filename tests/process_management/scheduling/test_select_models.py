"""Tests for _select_models_for_pop."""

from __future__ import annotations

from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE

from horde_worker_regen.process_management.jobs.job_popper import _select_models_for_pop
from horde_worker_regen.process_management.jobs.job_tracker import JobTracker
from horde_worker_regen.process_management.lifecycle.process_map import ProcessMap
from tests.process_management.conftest import (
    make_mock_bridge_data,
    make_mock_job,
    make_mock_model_reference_record,
    make_mock_process_info,
    make_test_card_runtimes,
    make_test_model_metadata,
    track_popped_job_async,
)


class TestSelectModelsForPopBasic:
    """Basic model selection behavior."""

    def test_returns_configured_models(self) -> None:
        """All configured models should be returned when there are no constraints."""
        bridge_data = make_mock_bridge_data(image_models_to_load=["model_a", "model_b"])
        process_map = ProcessMap({})
        job_tracker = JobTracker()

        result = _select_models_for_pop(
            bridge_data,
            process_map,
            job_tracker,
            max_inference_processes=2,
            last_pop_had_no_jobs=False,
        )

        assert result == {"model_a", "model_b"}

    def test_empty_models_returns_none(self) -> None:
        """No configured models → no pop should happen."""
        bridge_data = make_mock_bridge_data(image_models_to_load=[])
        process_map = ProcessMap({})
        job_tracker = JobTracker()

        result = _select_models_for_pop(
            bridge_data,
            process_map,
            job_tracker,
            max_inference_processes=2,
            last_pop_had_no_jobs=False,
        )

        assert result is None

    def test_single_model(self) -> None:
        """With only one model configured, that model should be returned."""
        bridge_data = make_mock_bridge_data(image_models_to_load=["stable_diffusion"])
        process_map = ProcessMap({})
        job_tracker = JobTracker()

        result = _select_models_for_pop(
            bridge_data,
            process_map,
            job_tracker,
            max_inference_processes=1,
            last_pop_had_no_jobs=False,
        )

        assert result == {"stable_diffusion"}


class TestDuplicateModelFiltering:
    """Models with >=2 jobs already queued should be excluded."""

    async def test_model_with_two_queued_jobs_excluded(self) -> None:
        """A model that already has 2 jobs queued should be removed from pop candidates."""
        bridge_data = make_mock_bridge_data(image_models_to_load=["model_a", "model_b"])
        process_map = ProcessMap({})
        job_tracker = JobTracker()

        await track_popped_job_async(job_tracker, make_mock_job(model="model_a"))
        await track_popped_job_async(job_tracker, make_mock_job(model="model_a"))

        result = _select_models_for_pop(
            bridge_data,
            process_map,
            job_tracker,
            max_inference_processes=2,
            last_pop_had_no_jobs=False,
        )

        assert result is not None
        assert "model_a" not in result
        assert "model_b" in result

    async def test_model_with_one_queued_job_not_excluded(self) -> None:
        """A model that has only 1 job queued should still be a candidate."""
        bridge_data = make_mock_bridge_data(image_models_to_load=["model_a", "model_b"])
        process_map = ProcessMap({})
        job_tracker = JobTracker()

        await track_popped_job_async(job_tracker, make_mock_job(model="model_a"))

        result = _select_models_for_pop(
            bridge_data,
            process_map,
            job_tracker,
            max_inference_processes=2,
            last_pop_had_no_jobs=False,
        )

        assert result is not None
        assert "model_a" in result

    async def test_all_models_excluded_returns_none(self) -> None:
        """If all models have >=2 queued jobs, no models are eligible → returns None."""
        bridge_data = make_mock_bridge_data(image_models_to_load=["model_a"])
        process_map = ProcessMap({})
        job_tracker = JobTracker()

        for _ in range(2):
            await track_popped_job_async(job_tracker, make_mock_job(model="model_a"))

        result = _select_models_for_pop(
            bridge_data,
            process_map,
            job_tracker,
            max_inference_processes=2,
            last_pop_had_no_jobs=False,
        )

        assert result is None

    async def test_three_queued_jobs_also_excluded(self) -> None:
        """Model with 3+ jobs should also be excluded (threshold is >= 2)."""
        bridge_data = make_mock_bridge_data(image_models_to_load=["model_a", "model_b"])
        process_map = ProcessMap({})
        job_tracker = JobTracker()

        for _ in range(3):
            await track_popped_job_async(job_tracker, make_mock_job(model="model_a"))

        result = _select_models_for_pop(
            bridge_data,
            process_map,
            job_tracker,
            max_inference_processes=2,
            last_pop_had_no_jobs=False,
        )

        assert result is not None
        assert "model_a" not in result


class TestStickyModels:
    """Model stickiness: prefer already-loaded models to avoid disk I/O."""

    def test_no_stickiness_returns_all_models(self) -> None:
        """With stickiness=0, all models should be returned."""
        bridge_data = make_mock_bridge_data(
            image_models_to_load=["model_a", "model_b", "model_c"],
            horde_model_stickiness=0,
        )
        process_info = make_mock_process_info(0, model_name="model_a")
        process_map = ProcessMap({0: process_info})
        job_tracker = JobTracker()

        result = _select_models_for_pop(
            bridge_data,
            process_map,
            job_tracker,
            max_inference_processes=1,
            last_pop_had_no_jobs=False,
        )

        assert result == {"model_a", "model_b", "model_c"}

    def test_stickiness_one_always_sticks(self) -> None:
        """With stickiness=1.0, the random check always passes → only loaded models."""
        bridge_data = make_mock_bridge_data(
            image_models_to_load=["model_a", "model_b", "model_c"],
            horde_model_stickiness=1.0,
        )
        from horde_worker_regen.process_management.ipc.messages import HordeProcessState

        process_info = make_mock_process_info(0, model_name="model_a", state=HordeProcessState.WAITING_FOR_JOB)
        process_map = ProcessMap({0: process_info})
        job_tracker = JobTracker()

        # Models to load (3) > max_inference_processes (1) and loaded == max → sticky path
        result = _select_models_for_pop(
            bridge_data,
            process_map,
            job_tracker,
            max_inference_processes=1,
            last_pop_had_no_jobs=False,
        )

        assert result is not None
        # Should only contain the free (non-busy) loaded model
        assert "model_a" in result
        # May or may not contain others depending on free_models logic
        # The key invariant: it's a subset of loaded models
        for model in result:
            assert model in {"model_a"}

    def test_stickiness_skipped_when_last_pop_had_no_jobs(self) -> None:
        """When last pop returned no jobs, stickiness is bypassed to try different models."""
        bridge_data = make_mock_bridge_data(
            image_models_to_load=["model_a", "model_b", "model_c"],
            horde_model_stickiness=1.0,
        )
        process_info = make_mock_process_info(0, model_name="model_a")
        process_map = ProcessMap({0: process_info})
        job_tracker = JobTracker()

        result = _select_models_for_pop(
            bridge_data,
            process_map,
            job_tracker,
            max_inference_processes=1,
            last_pop_had_no_jobs=True,  # bypass stickiness
        )

        assert result is not None
        # All models returned since stickiness was bypassed
        assert result == {"model_a", "model_b", "model_c"}

    def test_stickiness_not_applied_when_fewer_models_than_processes(self) -> None:
        """When models_to_load <= max_inference_processes, stickiness path isn't entered."""
        bridge_data = make_mock_bridge_data(
            image_models_to_load=["model_a"],
            horde_model_stickiness=1.0,
        )
        process_info = make_mock_process_info(0, model_name="model_a")
        process_map = ProcessMap({0: process_info})
        job_tracker = JobTracker()

        result = _select_models_for_pop(
            bridge_data,
            process_map,
            job_tracker,
            max_inference_processes=2,
            last_pop_had_no_jobs=False,
        )

        assert result == {"model_a"}

    def test_stickiness_not_applied_when_not_all_slots_loaded(self) -> None:
        """Stickiness requires loaded_models == max_inference_processes."""
        bridge_data = make_mock_bridge_data(
            image_models_to_load=["model_a", "model_b", "model_c"],
            horde_model_stickiness=1.0,
        )
        # Only 1 of 2 processes has a model loaded
        proc0 = make_mock_process_info(0, model_name="model_a")
        proc1 = make_mock_process_info(1, model_name=None)
        process_map = ProcessMap({0: proc0, 1: proc1})
        job_tracker = JobTracker()

        result = _select_models_for_pop(
            bridge_data,
            process_map,
            job_tracker,
            max_inference_processes=2,
            last_pop_had_no_jobs=False,
        )

        # Should get all models since stickiness condition isn't met
        assert result == {"model_a", "model_b", "model_c"}


class TestCustomModels:
    """Custom model injection."""

    def test_custom_models_added(self) -> None:
        """Custom models should be added to the set on top of configured models."""
        bridge_data = make_mock_bridge_data(
            image_models_to_load=["model_a"],
            custom_models=[{"name": "custom_model_1"}, {"name": "custom_model_2"}],
        )
        process_map = ProcessMap({})
        job_tracker = JobTracker()

        result = _select_models_for_pop(
            bridge_data,
            process_map,
            job_tracker,
            max_inference_processes=2,
            last_pop_had_no_jobs=False,
        )

        assert result is not None
        assert "model_a" in result
        assert "custom_model_1" in result
        assert "custom_model_2" in result

    def test_empty_custom_models_no_effect(self) -> None:
        """Empty custom models list should have no effect on results."""
        bridge_data = make_mock_bridge_data(
            image_models_to_load=["model_a"],
            custom_models=[],
        )
        process_map = ProcessMap({})
        job_tracker = JobTracker()

        result = _select_models_for_pop(
            bridge_data,
            process_map,
            job_tracker,
            max_inference_processes=2,
            last_pop_had_no_jobs=False,
        )

        assert result == {"model_a"}

    def test_none_custom_models_no_effect(self) -> None:
        """None custom models should have no effect on results."""
        bridge_data = make_mock_bridge_data(
            image_models_to_load=["model_a"],
            custom_models=None,
        )
        process_map = ProcessMap({})
        job_tracker = JobTracker()

        result = _select_models_for_pop(
            bridge_data,
            process_map,
            job_tracker,
            max_inference_processes=2,
            last_pop_had_no_jobs=False,
        )

        assert result == {"model_a"}

    def test_custom_model_duplicates_regular_model(self) -> None:
        """Custom model with same name as configured model doesn't create duplicates (it's a set)."""
        bridge_data = make_mock_bridge_data(
            image_models_to_load=["model_a"],
            custom_models=[{"name": "model_a"}],
        )
        process_map = ProcessMap({})
        job_tracker = JobTracker()

        result = _select_models_for_pop(
            bridge_data,
            process_map,
            job_tracker,
            max_inference_processes=2,
            last_pop_had_no_jobs=False,
        )

        assert result == {"model_a"}


class TestSelectModelsForPopCombinations:
    """Combined scenarios that exercise multiple code paths."""

    async def test_custom_model_survives_duplicate_filter(self) -> None:
        """A custom model without queued jobs should survive duplicate filtering."""
        bridge_data = make_mock_bridge_data(
            image_models_to_load=["model_a"],
            custom_models=[{"name": "custom_1"}],
        )
        process_map = ProcessMap({})
        job_tracker = JobTracker()

        # model_a has 2 jobs, should be filtered
        for _ in range(2):
            await track_popped_job_async(job_tracker, make_mock_job(model="model_a"))

        result = _select_models_for_pop(
            bridge_data,
            process_map,
            job_tracker,
            max_inference_processes=2,
            last_pop_had_no_jobs=False,
        )

        assert result is not None
        assert "model_a" not in result
        assert "custom_1" in result

    async def test_many_models_with_partial_queue(self) -> None:
        """With many models, only the ones with <2 queue slots should be in the result."""
        models = [f"model_{i}" for i in range(5)]
        bridge_data = make_mock_bridge_data(image_models_to_load=models)
        process_map = ProcessMap({})
        job_tracker = JobTracker()

        # model_0 and model_2 have 2 jobs each
        for model_name in ["model_0", "model_0", "model_2", "model_2"]:
            await track_popped_job_async(job_tracker, make_mock_job(model=model_name))

        result = _select_models_for_pop(
            bridge_data,
            process_map,
            job_tracker,
            max_inference_processes=5,
            last_pop_had_no_jobs=False,
        )

        assert result is not None
        assert "model_0" not in result
        assert "model_2" not in result
        assert "model_1" in result
        assert "model_3" in result
        assert "model_4" in result


class TestModelServiceabilityFiltering:
    """Model serviceability filters the offered model set before popping."""

    def test_unserviceable_model_excluded_from_offer(self) -> None:
        """A model whose minimum footprint exceeds the scoped card is not offered."""
        sdxl = "sdxl_model"
        sd15 = "sd15_model"
        bridge_data = make_mock_bridge_data(image_models_to_load=[sdxl, sd15])
        reference = {
            sdxl: make_mock_model_reference_record(
                sdxl,
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
            ),
            sd15: make_mock_model_reference_record(
                sd15,
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            ),
        }
        metadata = make_test_model_metadata(reference)
        card_runtimes = make_test_card_runtimes(config=bridge_data, total_vram_mb=8192.0)

        result = _select_models_for_pop(
            bridge_data,
            ProcessMap({}),
            JobTracker(),
            max_inference_processes=2,
            last_pop_had_no_jobs=False,
            card_runtimes=card_runtimes,
            model_metadata=metadata,
            admission_baseline_provider=lambda _device: 1024.0,
            serviceability_logged=set(),
        )

        assert result == {sd15}

    def test_fitting_model_is_never_excluded_from_offer(self) -> None:
        """A model whose minimum footprint fits the card remains in the offer."""
        sdxl = "sdxl_model"
        bridge_data = make_mock_bridge_data(image_models_to_load=[sdxl])
        reference = {
            sdxl: make_mock_model_reference_record(
                sdxl,
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
            ),
        }
        metadata = make_test_model_metadata(reference)
        card_runtimes = make_test_card_runtimes(config=bridge_data, total_vram_mb=24576.0)

        result = _select_models_for_pop(
            bridge_data,
            ProcessMap({}),
            JobTracker(),
            max_inference_processes=2,
            last_pop_had_no_jobs=False,
            card_runtimes=card_runtimes,
            model_metadata=metadata,
            admission_baseline_provider=lambda _device: 1024.0,
            serviceability_logged=set(),
        )

        assert result == {sdxl}
