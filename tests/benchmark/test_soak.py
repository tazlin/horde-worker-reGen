"""Unit tests for the sustained-load validation (soak) phase."""

from __future__ import annotations

from horde_worker_regen.benchmark.report import SuggestedBridgeData
from horde_worker_regen.benchmark.scenarios import CannedAlchemyFormSpec, CannedImageJobSpec, ScenarioSpec
from horde_worker_regen.benchmark.soak import build_soak_scenario, build_validation_level
from horde_worker_regen.process_management._canned_scenarios import (
    GeneratingAlchemySource,
    GeneratingJobSource,
    SoakImageTemplate,
)


class TestScenarioSoakTemplates:
    """`ScenarioSpec.to_soak_templates` maps specs to weighted generation templates."""

    def test_count_becomes_weight(self) -> None:
        """Each spec's `count` becomes its generation weight; feature fields carry over."""
        scenario = ScenarioSpec(
            name="mix",
            image_jobs=[
                CannedImageJobSpec(model="Deliberate", count=1),
                CannedImageJobSpec(model="Deliberate", n_iter=4, control_type="canny", count=3),
            ],
            alchemy_forms=[CannedAlchemyFormSpec(form="caption", count=2)],
            soak_seconds=10.0,
        )
        image_templates, alchemy_templates = scenario.to_soak_templates()

        assert [t.weight for t in image_templates] == [1.0, 3.0]
        assert image_templates[1].n_iter == 4
        assert image_templates[1].control_type == "canny"
        assert alchemy_templates == [("caption", 2.0)]


class TestBuildSoakScenario:
    """The soak workload reflects exactly the capabilities the recommendation enables."""

    def test_weights_dominated_by_heavy_specs_when_enabled(self) -> None:
        """Enabled capabilities produce heavier-weighted job types than the light baseline."""
        suggested = SuggestedBridgeData(
            max_batch=4,
            allow_controlnet=True,
            allow_post_processing=True,
            alchemist=True,
            models_to_load=["Deliberate"],
        )
        scenario = build_soak_scenario(suggested, "sd15", soak_seconds=120.0)

        assert scenario.soak_seconds == 120.0
        # A light baseline job plus the three heavy job types (batch, controlnet, post-processing).
        n_iters = sorted(spec.n_iter for spec in scenario.image_jobs)
        assert 4 in n_iters  # max batch present
        assert any(spec.control_type == "canny" for spec in scenario.image_jobs)
        assert any(spec.post_processing for spec in scenario.image_jobs)
        # Heavy specs carry more weight than the single light baseline job.
        light_weight = min(spec.count for spec in scenario.image_jobs)
        heavy_weight = max(spec.count for spec in scenario.image_jobs)
        assert heavy_weight > light_weight
        assert [f.form for f in scenario.alchemy_forms]  # alchemy included

    def test_minimal_when_nothing_extra_enabled(self) -> None:
        """A bare config soaks only a single plain job type and no alchemy."""
        suggested = SuggestedBridgeData(models_to_load=["Deliberate"])
        scenario = build_soak_scenario(suggested, "sd15", soak_seconds=60.0)

        assert len(scenario.image_jobs) == 1
        assert scenario.image_jobs[0].n_iter == 1
        assert scenario.alchemy_forms == []

    def test_model_pool_spreads_profiles_across_models(self) -> None:
        """A multi-model pool replicates every job profile across the distinct models.

        This is the correctness fix for the 2-per-model in-flight cap: with one model the soak
        could only ever keep two jobs running, starving extra processes.
        """
        suggested = SuggestedBridgeData(max_batch=4, models_to_load=["Deliberate"])
        pool = ["Deliberate", "Dreamshaper", "ICBINP"]
        scenario = build_soak_scenario(suggested, "sd15", soak_seconds=60.0, model_pool=pool)

        # Two profiles (baseline + batch) replicated across three models.
        assert sorted(scenario.models_referenced()) == sorted(pool)
        assert len(scenario.image_jobs) == 2 * len(pool)
        # The relative weighting between profiles is preserved within each model.
        for model in pool:
            weights = sorted(spec.count for spec in scenario.image_jobs if spec.model == model)
            assert weights == [1, 3]

    def test_single_model_pool_matches_default(self) -> None:
        """A one-entry pool produces the same shape as the default single-model soak."""
        suggested = SuggestedBridgeData(models_to_load=["Deliberate"])
        scenario = build_soak_scenario(suggested, "sd15", soak_seconds=60.0, model_pool=["Deliberate"])
        assert len(scenario.image_jobs) == 1
        assert scenario.image_jobs[0].model == "Deliberate"


class TestBuildValidationLevel:
    """The validation level carries the soak scenario, the synthesized config, and soak criteria."""

    def test_level_shape(self) -> None:
        """The stage-V level carries the soak scenario, recommended config, and soak criteria."""
        suggested = SuggestedBridgeData(
            max_threads=2,
            queue_size=2,
            max_batch=4,
            allow_controlnet=True,
            alchemist=True,
            alchemy_allow_concurrent=True,
            models_to_load=["Deliberate"],
        )
        level = build_validation_level(suggested, "sd15", soak_seconds=120.0)

        assert level.stage == "V"
        assert level.axis == "validation"
        assert level.scenario.soak_seconds == 120.0
        # The soak runs the recommended worker config (batch is applied via job templates, not here).
        assert level.bridge_data_overrides["max_threads"] == 2
        assert level.bridge_data_overrides["allow_controlnet"] is True
        assert "max_batch" not in level.bridge_data_overrides
        # Soak criteria: retention gate on, baseline gate off, a minimum job floor.
        assert level.criteria.min_its_retention == 0.85
        assert level.criteria.gate_its_against_baseline is False
        assert level.criteria.min_completed_jobs >= 1
        # The duty-cycle metric of record gates the soak.
        assert level.criteria.min_gpu_duty_cycle_percent == 90.0
        # Timeout must comfortably exceed the soak period.
        assert level.timeout_seconds > 120.0

    def test_model_pool_loads_all_models(self) -> None:
        """A multi-model validation level loads every pool model and spreads the soak over them."""
        suggested = SuggestedBridgeData(max_threads=2, queue_size=2, models_to_load=["Deliberate"])
        pool = ["Deliberate", "Dreamshaper", "ICBINP", "Anything Diffusion"]
        level = build_validation_level(suggested, "sd15", soak_seconds=120.0, model_pool=pool)

        assert level.bridge_data_overrides["models_to_load"] == pool
        assert sorted(level.scenario.models_referenced()) == sorted(pool)

    def test_residency_expectation_flows_to_criteria(self) -> None:
        """The residency-defeated advisory is enabled only when residency is expected."""
        suggested = SuggestedBridgeData(models_to_load=["Deliberate"])
        level = build_validation_level(suggested, "sd15", soak_seconds=60.0, expect_vram_residency=True)
        assert level.criteria.expect_vram_residency is True


class TestGeneratingSources:
    """The generating sources mint fresh IDs and stop on request."""

    def test_job_source_unique_ids_then_stops(self) -> None:
        """Every generated job gets a unique ID; stop() makes the source report exhausted."""
        source = GeneratingJobSource([SoakImageTemplate(model="Deliberate", n_iter=2, weight=1.0)], seed=7)
        ids = {source.next_pop_response().id_ for _ in range(40)}
        assert len(ids) == 40
        assert source.exhausted is False

        source.stop()
        assert source.exhausted is True
        assert source.next_pop_response().id_ is None

    def test_alchemy_source_unique_ids_then_stops(self) -> None:
        """Every generated alchemy form gets a unique ID; stop() ends generation."""
        source = GeneratingAlchemySource([("caption", 1.0), ("RealESRGAN_x4plus", 2.0)], seed=7)
        form_ids = {form.form_id for _ in range(30) if (form := source.next_form()) is not None}
        assert len(form_ids) == 30

        source.stop()
        assert source.exhausted is True
        assert source.next_form() is None

    def test_weighting_is_respected(self) -> None:
        """A heavily weighted template dominates the generated stream."""
        # An overwhelmingly weighted template should dominate the generated stream.
        source = GeneratingJobSource(
            [
                SoakImageTemplate(model="Deliberate", width=512, weight=0.0),
                SoakImageTemplate(model="Deliberate", width=768, weight=100.0),
            ],
            seed=3,
        )
        widths = [source.next_pop_response().payload.width for _ in range(50)]
        assert widths.count(768) > widths.count(512)
