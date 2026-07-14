"""Unit tests for the sustained-load validation (soak) phase."""

from __future__ import annotations

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopRequest, LorasPayloadEntry, TIPayloadEntry

from horde_worker_regen.benchmark.capabilities.result import SuggestedBridgeData
from horde_worker_regen.benchmark.enums import BenchTier
from horde_worker_regen.benchmark.scenarios import CannedAlchemyFormSpec, CannedImageJobSpec, Scenario
from horde_worker_regen.benchmark.soak import (
    PRODUCTION_REPLAY_FLUX_MODELS,
    PRODUCTION_REPLAY_SD15_MODELS,
    PRODUCTION_REPLAY_SDXL_MODELS,
    build_lora_storm_soak_scenario,
    build_production_replay_soak_scenario,
    build_soak_scenario,
)
from horde_worker_regen.process_management.simulation._canned_scenarios import (
    GeneratingAlchemySource,
    GeneratingJobSource,
    SoakImageTemplate,
)


class TestScenarioSoakTemplates:
    """`Scenario.to_soak_templates` maps specs to weighted generation templates."""

    def test_count_becomes_weight(self) -> None:
        """Each spec's `count` becomes its generation weight; feature fields carry over."""
        scenario = Scenario(
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

    def test_lora_and_ti_references_propagate(self) -> None:
        """A spec's `lora_names`/`ti_names` become concrete payload entries on the soak template."""
        scenario = Scenario(
            name="lora-mix",
            image_jobs=[
                CannedImageJobSpec(model="Deliberate", lora_names=["ref_a", "ref_b"], ti_names=["ti_a"], count=2),
                CannedImageJobSpec(model="Deliberate", count=1),
            ],
            soak_seconds=10.0,
        )
        image_templates, _ = scenario.to_soak_templates()

        assert [entry.name for entry in image_templates[0].loras] == ["ref_a", "ref_b"]
        assert [entry.name for entry in image_templates[0].tis] == ["ti_a"]
        # A spec with no auxiliaries carries none, so the soak keeps a plain control group.
        assert image_templates[1].loras == []
        assert image_templates[1].tis == []


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
        scenario = build_soak_scenario(suggested, BenchTier.SD15, soak_seconds=120.0)

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
        scenario = build_soak_scenario(suggested, BenchTier.SD15, soak_seconds=60.0)

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
        scenario = build_soak_scenario(suggested, BenchTier.SD15, soak_seconds=60.0, model_pool=pool)

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
        scenario = build_soak_scenario(suggested, BenchTier.SD15, soak_seconds=60.0, model_pool=["Deliberate"])
        assert len(scenario.image_jobs) == 1
        assert scenario.image_jobs[0].model == "Deliberate"

    def test_zimage_soak_applies_fixed_steps_and_cfg(self) -> None:
        """All ZIMAGE soak jobs use steps=9 and cfg_scale=1.0 (locked inference parameters)."""
        suggested = SuggestedBridgeData(max_batch=4, models_to_load=["Z-Image-Turbo"])
        scenario = build_soak_scenario(suggested, BenchTier.ZIMAGE, soak_seconds=60.0)

        for spec in scenario.image_jobs:
            assert spec.steps == 9, f"expected steps=9 for all ZIMAGE jobs, got {spec.steps}"
            assert spec.cfg_scale == 1.0, f"expected cfg_scale=1.0 for all ZIMAGE jobs, got {spec.cfg_scale}"
        # No controlnet or hires_fix profiles should appear in a ZIMAGE soak.
        assert all(spec.control_type is None for spec in scenario.image_jobs)
        assert all(not spec.hires_fix for spec in scenario.image_jobs)


class TestGeneratingSources:
    """The generating sources mint fresh IDs and stop on request."""

    @staticmethod
    def _pop_request(**overrides: object) -> ImageGenerateJobPopRequest:
        """Build a permissive request whose individual eligibility axes a test can override."""
        values: dict[str, object] = {
            "apikey": "0000000000",
            "name": "test-worker",
            "models": ["Deliberate"],
            "max_pixels": 1024 * 1024,
            "allow_img2img": True,
            "allow_post_processing": True,
            "allow_controlnet": True,
            "allow_extended_controlnet": True,
            "allow_sdxl_controlnet": True,
            "allow_lora": True,
        }
        values.update(overrides)
        return ImageGenerateJobPopRequest(**values)  # type: ignore[arg-type]

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

    def test_minted_jobs_carry_template_loras_and_tis(self) -> None:
        """Every job a LoRA/TI template mints carries the same configured references (repeated per template)."""
        source = GeneratingJobSource(
            [
                SoakImageTemplate(
                    model="Deliberate",
                    loras=[LorasPayloadEntry(name="shared_ref"), LorasPayloadEntry(name="second_ref")],
                    tis=[TIPayloadEntry(name="ti_ref", inject_ti="prompt")],
                    weight=1.0,
                ),
            ],
            seed=1,
        )
        for _ in range(5):
            response = source.next_pop_response()
            assert [entry.name for entry in response.payload.loras or []] == ["shared_ref", "second_ref"]
            assert [entry.name for entry in response.payload.tis or []] == ["ti_ref"]

    def test_lora_references_are_deterministic_per_seed(self) -> None:
        """Two sources with the same seed mint the same model/LoRA sequence with fresh IDs each pop."""
        templates = [
            SoakImageTemplate(model="Deliberate", loras=[LorasPayloadEntry(name="cache_ref")], weight=2.0),
            SoakImageTemplate(model="Deliberate", loras=[LorasPayloadEntry(name="fresh_ref")], weight=1.0),
            SoakImageTemplate(model="Deliberate", weight=1.0),
        ]

        def _lora_sequence(seed: int) -> list[list[str]]:
            source = GeneratingJobSource([SoakImageTemplate(**vars(t)) for t in templates], seed=seed)
            return [[entry.name for entry in source.next_pop_response().payload.loras or []] for _ in range(30)]

        first = _lora_sequence(11)
        assert _lora_sequence(11) == first
        # The stream visits both LoRA-carrying and plain templates, so the control group is exercised.
        assert any(names for names in first)
        assert any(not names for names in first)

    def test_minted_lora_jobs_get_fresh_ids(self) -> None:
        """Reusing a reference does not reuse generation IDs: every pop is a distinct job."""
        source = GeneratingJobSource(
            [SoakImageTemplate(model="Deliberate", loras=[LorasPayloadEntry(name="shared_ref")], weight=1.0)],
            seed=5,
        )
        ids = {source.next_pop_response().id_ for _ in range(25)}
        assert len(ids) == 25

    def test_pop_request_filters_models_and_resolution_before_weighted_draw(self) -> None:
        """Targeted/idle-fill shaping cannot mint a model or resolution outside the advertised slice."""
        source = GeneratingJobSource(
            [
                SoakImageTemplate(model="Deliberate", width=512, height=512, weight=0.0),
                SoakImageTemplate(model="Deliberate", width=1024, height=1024, weight=100.0),
                SoakImageTemplate(model="Other", width=512, height=512, weight=100.0),
            ],
            seed=1,
        )

        response = source.next_pop_response(self._pop_request(max_pixels=512 * 512))

        assert response.id_ is not None
        assert response.model == "Deliberate"
        assert response.payload.width == 512
        assert response.payload.height == 512

    @pytest.mark.parametrize(
        ("template", "request_override"),
        [
            (SoakImageTemplate(loras=[LorasPayloadEntry(name="lora")]), {"allow_lora": False}),
            (SoakImageTemplate(tis=[TIPayloadEntry(name="ti", inject_ti="prompt")]), {"allow_lora": False}),
            (SoakImageTemplate(post_processing=["GFPGAN"]), {"allow_post_processing": False}),
            (SoakImageTemplate(control_type="canny"), {"allow_controlnet": False}),
            (SoakImageTemplate(control_type="mlsd"), {"allow_extended_controlnet": False}),
            (SoakImageTemplate(workflow="qr_code"), {"allow_sdxl_controlnet": False}),
            (SoakImageTemplate(control_type="canny"), {"allow_img2img": False}),
            (SoakImageTemplate(workflow="custom_workflow"), {"allow_img2img": False}),
        ],
    )
    def test_pop_request_withheld_feature_returns_no_job(
        self,
        template: SoakImageTemplate,
        request_override: dict[str, object],
    ) -> None:
        """A generated source reports no work when its only template needs a withheld capability."""
        source = GeneratingJobSource([template], seed=1)

        response = source.next_pop_response(self._pop_request(**request_override))

        assert response.id_ is None

    def test_pop_request_redistributes_weights_over_eligible_templates(self) -> None:
        """An excluded high-weight template cannot starve a zero-weight but eligible fallback."""
        source = GeneratingJobSource(
            [
                SoakImageTemplate(loras=[LorasPayloadEntry(name="lora")], weight=100.0),
                SoakImageTemplate(weight=0.0),
            ],
            seed=1,
        )

        response = source.next_pop_response(self._pop_request(allow_lora=False))

        assert response.id_ is not None
        assert not response.payload.loras


class TestLoraStormMix:
    """The `lora_storm` named mix exercises download pressure, gate liveness, and backoff behaviour."""

    def test_carries_the_designed_lora_fraction(self) -> None:
        """Roughly two thirds of the weighted stream carries LoRAs; the rest is the plain control group."""
        scenario = build_lora_storm_soak_scenario(soak_seconds=300.0)
        assert scenario.name == "lora_storm"
        assert scenario.soak_seconds == 300.0

        total_weight = sum(spec.count for spec in scenario.image_jobs)
        lora_weight = sum(spec.count for spec in scenario.image_jobs if spec.lora_names)
        plain_weight = total_weight - lora_weight
        lora_fraction = lora_weight / total_weight

        assert 0.6 <= lora_fraction <= 0.7, f"lora fraction {lora_fraction:.3f} outside 60-70%"
        # The plain jobs are a real control group, not a token entry.
        assert plain_weight > 0

    def test_blends_cache_hit_and_download_pressure_pools(self) -> None:
        """A small pool of repeated references and a larger pool of unique references both appear."""
        scenario = build_lora_storm_soak_scenario(soak_seconds=60.0)
        ref_usage: dict[str, int] = {}
        for spec in scenario.image_jobs:
            for name in spec.lora_names:
                ref_usage[name] = ref_usage.get(name, 0) + 1

        repeated = {name for name, uses in ref_usage.items() if uses > 1}
        singletons = {name for name, uses in ref_usage.items() if uses == 1}
        # Cache-hit pool: a few references reused across several specs.
        assert repeated
        # Download-pressure pool: a strictly larger set of references each used once.
        assert len(singletons) > len(repeated)

    def test_spans_a_light_and_a_heavy_base_model(self) -> None:
        """LoRA jobs run on both a lighter and a heavier base model from the roster."""
        models_with_loras = {spec.model for spec in scenario_specs_with_loras()}
        assert len(models_with_loras) >= 2

    def test_lora_count_per_job_stays_in_one_to_four(self) -> None:
        """Every LoRA-carrying job requests between one and four references."""
        counts = [len(spec.lora_names) for spec in scenario_specs_with_loras()]
        assert counts
        assert min(counts) >= 1
        assert max(counts) == 4


def scenario_specs_with_loras() -> list[CannedImageJobSpec]:
    """The LoRA-carrying image specs of the `lora_storm` mix (helper for the mix tests)."""
    scenario = build_lora_storm_soak_scenario(soak_seconds=60.0)
    return [spec for spec in scenario.image_jobs if spec.lora_names]


class TestLoraStormReferenceOverride:
    """A real-mode operator supplies resolvable references; the mix must carry exactly those."""

    def test_supplied_references_replace_the_synthetic_pools(self) -> None:
        """Every LoRA-carrying spec draws only from the supplied pools, none from the synthetic defaults."""
        shared = ("real-shared-a", "real-shared-b", "real-shared-c")
        unique = tuple(f"real-unique-{index}" for index in range(8))
        scenario = build_lora_storm_soak_scenario(
            soak_seconds=60.0,
            shared_lora_references=shared,
            unique_lora_references=unique,
        )
        carried = {name for spec in scenario.image_jobs for name in spec.lora_names}
        assert carried
        assert carried <= set(shared) | set(unique)
        assert not any(name.startswith("lora_storm_") for name in carried)

    def test_undersized_pools_are_rejected(self) -> None:
        """Pools smaller than the mix's indexing requirements fail fast instead of building a partial storm."""
        with pytest.raises(ValueError, match="at least 3 shared and 8 unique"):
            build_lora_storm_soak_scenario(
                soak_seconds=60.0,
                shared_lora_references=("only", "two"),
                unique_lora_references=tuple(f"u{index}" for index in range(8)),
            )


_ALL_PRODUCTION_REPLAY_MODELS = (
    set(PRODUCTION_REPLAY_SDXL_MODELS) | set(PRODUCTION_REPLAY_SD15_MODELS) | set(PRODUCTION_REPLAY_FLUX_MODELS)
)


def _replay_family_share(specs: list[CannedImageJobSpec], family: tuple[str, ...]) -> float:
    """The weighted pop-count fraction of the `production_replay` mix drawn from a family's checkpoints."""
    total_weight = sum(spec.count for spec in specs)
    family_weight = sum(spec.count for spec in specs if spec.model in set(family))
    return family_weight / total_weight


class TestProductionReplayMix:
    """The `production_replay` named mix reproduces the shape of measured production traffic."""

    def test_scenario_name_duration_and_feature_shares(self) -> None:
        """The mix is named, timed, and its weighted LoRA and post-processing shares match the fingerprint."""
        scenario = build_production_replay_soak_scenario(soak_seconds=300.0)
        assert scenario.name == "production_replay"
        assert scenario.soak_seconds == 300.0

        total_weight = sum(spec.count for spec in scenario.image_jobs)
        lora_fraction = sum(spec.count for spec in scenario.image_jobs if spec.lora_names) / total_weight
        pp_fraction = sum(spec.count for spec in scenario.image_jobs if spec.post_processing) / total_weight

        assert 0.18 <= lora_fraction <= 0.30, f"lora fraction {lora_fraction:.3f} outside 18-30%"
        assert 0.30 <= pp_fraction <= 0.45, f"post-processing fraction {pp_fraction:.3f} outside 30-45%"

    def test_family_weight_shares_track_the_fingerprint(self) -> None:
        """SDXL dominates the stream and Flux is the rare heavy-load minority, as measured."""
        specs = build_production_replay_soak_scenario(soak_seconds=60.0).image_jobs
        sdxl_share = _replay_family_share(specs, PRODUCTION_REPLAY_SDXL_MODELS)
        flux_share = _replay_family_share(specs, PRODUCTION_REPLAY_FLUX_MODELS)

        assert 0.6 <= sdxl_share <= 0.8, f"sdxl share {sdxl_share:.3f} outside 60-80%"
        assert 0.01 <= flux_share <= 0.06, f"flux share {flux_share:.3f} outside 1-6%"

    def test_batched_jobs_are_a_meaningful_minority(self) -> None:
        """Multi-image jobs (batch>=2) are present at the measured minority share, not a token entry."""
        specs = build_production_replay_soak_scenario(soak_seconds=60.0).image_jobs
        total_weight = sum(spec.count for spec in specs)
        batch_share = sum(spec.count for spec in specs if spec.n_iter >= 2) / total_weight
        assert 0.18 <= batch_share <= 0.35, f"batch>=2 share {batch_share:.3f} outside 18-35%"

    def test_every_template_uses_a_listed_production_model(self) -> None:
        """No template invents a checkpoint outside the seven measured production models."""
        specs = build_production_replay_soak_scenario(soak_seconds=60.0).image_jobs
        assert {spec.model for spec in specs} <= _ALL_PRODUCTION_REPLAY_MODELS

    def test_undersized_pools_are_rejected(self) -> None:
        """Partial LoRA pools fail fast, the same fast-fail the storm mix enforces."""
        with pytest.raises(ValueError, match="at least 3 shared and 8 unique"):
            build_production_replay_soak_scenario(
                soak_seconds=60.0,
                shared_lora_references=("only", "two"),
                unique_lora_references=tuple(f"u{index}" for index in range(8)),
            )

    def test_minted_jobs_carry_a_templates_post_processing(self) -> None:
        """A post-processing template drives real minted jobs that carry its post-processing list."""
        specs = build_production_replay_soak_scenario(soak_seconds=60.0).image_jobs
        pp_spec = next(spec for spec in specs if spec.post_processing)
        scenario = Scenario(name="pp-probe", image_jobs=[pp_spec], soak_seconds=60.0)
        image_templates, _ = scenario.to_soak_templates()
        source = GeneratingJobSource(image_templates, seed=1)

        for _ in range(5):
            response = source.next_pop_response()
            assert list(response.payload.post_processing or []) == pp_spec.post_processing
