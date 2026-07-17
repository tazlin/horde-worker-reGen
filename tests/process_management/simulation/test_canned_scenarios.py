"""Tests for the canned scenario factories, arrival control, and Scenario (no GPU)."""

from __future__ import annotations

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopRequest

from horde_worker_regen.benchmark.scenarios import (
    CannedAlchemyFormSpec,
    CannedImageJobSpec,
    Scenario,
)
from horde_worker_regen.process_management.simulation._canned_scenarios import (
    ArrivalSchedule,
    CannedAlchemySource,
    GeneratingJobSource,
    TimedJobSource,
    make_alchemy_scenario,
    make_canned_job,
    make_controlnet_scenario,
    make_hires_fix_scenario,
    make_lora_scenario,
    make_post_processing_scenario,
    make_ti_scenario,
)

_PINNED_SEED = "123456789"
"""The seed ``dummy_job_factory`` has always pinned; the deterministic knobs must default to it."""
_PINNED_PROMPT = "a man walking in the snow"
"""The prompt ``dummy_job_factory`` has always pinned; the deterministic knobs must default to it."""


def _restrictive_pop_request() -> ImageGenerateJobPopRequest:
    """Build a request that rejects the feature carried by the fixed-source contract test."""
    return ImageGenerateJobPopRequest(
        apikey="0000000000",
        name="test-worker",
        models=["Deliberate"],
        max_pixels=512 * 512,
        allow_lora=False,
    )


class TestFeatureFactories:
    """The feature scenario factories produce well-formed payloads."""

    def test_lora_scenario_round_robins_names(self) -> None:
        """Lora names are assigned round-robin across the jobs."""
        jobs = make_lora_scenario(3, ["lora-a", "lora-b"])
        names = [job.payload.loras[0].name for job in jobs if job.payload.loras]
        assert names == ["lora-a", "lora-b", "lora-a"]

    def test_ti_scenario_injects_prompt(self) -> None:
        """TI jobs carry the embedding name with prompt injection."""
        jobs = make_ti_scenario(1, ["some-ti"])
        assert jobs[0].payload.tis is not None
        assert jobs[0].payload.tis[0].name == "some-ti"
        assert jobs[0].payload.tis[0].inject_ti == "prompt"

    def test_controlnet_scenario_has_source_image(self) -> None:
        """Controlnet jobs get a synthetic source image automatically."""
        jobs = make_controlnet_scenario(1, control_type="canny")
        assert jobs[0].payload.control_type is not None
        assert jobs[0].source_image, "controlnet jobs need a source image"

    def test_post_processing_scenario_defaults(self) -> None:
        """The default post-processing set is one upscaler plus one facefixer."""
        jobs = make_post_processing_scenario(1)
        assert jobs[0].payload.post_processing == ["RealESRGAN_x4plus", "GFPGAN"]

    def test_hires_fix_scenario(self) -> None:
        """Hires-fix jobs carry the flag and the requested resolution."""
        jobs = make_hires_fix_scenario(1, width=1024, height=1024)
        assert jobs[0].payload.hires_fix
        assert jobs[0].payload.width == 1024

    def test_batched_job_gets_per_image_ids(self) -> None:
        """Batched jobs get one generation ID and R2 slot per image, like the live API."""
        job = make_canned_job(n_iter=3)
        assert job.ids is not None and len(job.ids) == 3
        assert job.r2_uploads is not None and len(job.r2_uploads) == 3

    def test_empty_name_lists_rejected(self) -> None:
        """Factories refuse empty lora/ti name lists."""
        with pytest.raises(ValueError):
            make_lora_scenario(1, [])
        with pytest.raises(ValueError):
            make_ti_scenario(1, [])

    def test_fixed_source_replays_script_despite_pop_request(self) -> None:
        """Deterministic fixed scenarios do not silently discard a scripted job on request shaping."""
        source, _alchemy_source = Scenario(
            name="fixed",
            image_jobs=[CannedImageJobSpec(model="Deliberate", lora_names=["lora-a"])],
        ).to_canned_sources()

        response = source.next_pop_response(_restrictive_pop_request())

        assert response.id_ is not None
        assert response.payload.loras


class TestDeterministicPayloadKnobs:
    """Seed/prompt/source_processing knobs override the pinned canned payload, defaulting to today's pins."""

    def test_defaults_reproduce_pinned_seed_and_prompt(self) -> None:
        """With the knobs unset, a canned job carries the exact historical seed and prompt pins."""
        job = make_canned_job()
        assert job.payload.seed == _PINNED_SEED
        assert job.payload.prompt == _PINNED_PROMPT

    def test_overrides_flow_into_payload(self) -> None:
        """An explicit seed/prompt/source_processing replaces the pinned values on the payload."""
        job = make_canned_job(seed="777", prompt="a lighthouse at dusk", source_processing="img2img")
        assert job.payload.seed == "777"
        assert job.payload.prompt == "a lighthouse at dusk"
        assert str(job.source_processing) == "img2img"

    def test_spec_defaults_preserve_pinned_values(self) -> None:
        """A spec without the knobs expands to jobs carrying the pinned seed/prompt (behaviour-neutral)."""
        jobs = Scenario(
            name="defaults",
            image_jobs=[CannedImageJobSpec(count=2)],
        ).expand_image_jobs()
        assert [job.payload.seed for job in jobs] == [_PINNED_SEED, _PINNED_SEED]
        assert all(job.payload.prompt == _PINNED_PROMPT for job in jobs)

    def test_spec_overrides_flow_through_expansion(self) -> None:
        """The spec knobs reach every expanded fixed-list job."""
        jobs = Scenario(
            name="fixed",
            image_jobs=[CannedImageJobSpec(count=2, seed="555", prompt="a red door", source_processing="img2img")],
        ).expand_image_jobs()
        assert all(job.payload.seed == "555" for job in jobs)
        assert all(job.payload.prompt == "a red door" for job in jobs)
        assert all(str(job.source_processing) == "img2img" for job in jobs)

    def test_spec_knobs_reach_generated_soak_jobs(self) -> None:
        """The knobs survive the soak-template path so a generating source mints them onto every job."""
        image_templates, _alchemy = Scenario(
            name="soak",
            image_jobs=[CannedImageJobSpec(seed="999", prompt="a snowy owl")],
        ).to_soak_templates()
        source = GeneratingJobSource(image_templates, seed=0)

        job = source.next_pop_response()

        assert job.payload.seed == "999"
        assert job.payload.prompt == "a snowy owl"


class TestImg2ImgSourceImage:
    """A plain img2img/remix source-processing job carries a deterministic synthetic start image."""

    def test_img2img_job_gets_a_source_image_at_its_resolution(self) -> None:
        """An img2img job with no control/workflow still ships a source image sized to the job."""
        import base64
        import struct

        job = make_canned_job(width=768, height=512, source_processing="img2img", seed="42")
        assert job.source_image, "an img2img job must carry a start image so it does not degrade to txt2img"
        raw = base64.b64decode(job.source_image)
        width, height = struct.unpack(">II", raw[16:24])
        assert (width, height) == (768, 512)

    def test_remix_also_gets_a_source_image(self) -> None:
        """The remix mode is treated as img2img-class and likewise carries a start image."""
        job = make_canned_job(source_processing="remix", seed="7")
        assert job.source_image

    def test_source_image_is_deterministic_per_seed(self) -> None:
        """Two builds with the same seed and size embed a byte-identical source image."""
        first = make_canned_job(width=256, height=256, source_processing="img2img", seed="99")
        second = make_canned_job(width=256, height=256, source_processing="img2img", seed="99")
        assert first.source_image == second.source_image

    def test_txt2img_job_carries_no_source_image(self) -> None:
        """A job with no source_processing (and no control/workflow) stays a plain txt2img job."""
        assert make_canned_job(seed="1").source_image is None


class TestAlchemyScenario:
    """Canned alchemy form generation and the alchemy source."""

    def test_forms_cycle_and_have_ids(self) -> None:
        """Form names cycle and every form gets a unique ID and source image."""
        forms = make_alchemy_scenario(["caption", "RealESRGAN_x4plus"], 3)
        assert [form.form for form in forms] == ["caption", "RealESRGAN_x4plus", "caption"]
        assert len({form.form_id for form in forms}) == 3
        assert all(form.source_image_bytes for form in forms)

    def test_canned_source_exhausts(self) -> None:
        """The source hands out each form once and then reports exhaustion."""
        source = CannedAlchemySource(make_alchemy_scenario(["caption"], 2))
        assert source.total_forms == 2
        assert source.next_form() is not None
        assert not source.exhausted
        assert source.next_form() is not None
        assert source.exhausted
        assert source.next_form() is None


class TestTimedJobSource:
    """Arrival-schedule gating with an injected clock."""

    def test_steady_schedule_gates_release(self) -> None:
        """Steady arrivals release one job per interval and report empty in between."""
        now = [0.0]
        jobs = [make_canned_job() for _ in range(3)]
        # 60/min = one job per second
        source = TimedJobSource(jobs, ArrivalSchedule(kind="steady", rate_per_minute=60), clock=lambda: now[0])

        assert source.next_pop_response().id_ == jobs[0].id_
        # Second job arrives at t=1; not yet.
        assert source.next_pop_response().id_ is None
        now[0] = 1.0
        assert source.next_pop_response().id_ == jobs[1].id_
        now[0] = 1.5
        assert source.next_pop_response().id_ is None
        now[0] = 2.0
        assert source.next_pop_response().id_ == jobs[2].id_
        assert source.exhausted

    def test_burst_schedule(self) -> None:
        """Burst arrivals release burst_size jobs together, then wait the interval."""
        now = [0.0]
        jobs = [make_canned_job() for _ in range(4)]
        schedule = ArrivalSchedule(kind="bursts", burst_size=2, burst_interval_seconds=10.0)
        source = TimedJobSource(jobs, schedule, clock=lambda: now[0])

        assert source.next_pop_response().id_ is not None
        assert source.next_pop_response().id_ is not None
        assert source.next_pop_response().id_ is None, "third job is in the second burst"
        now[0] = 10.0
        assert source.next_pop_response().id_ is not None
        assert source.next_pop_response().id_ is not None

    def test_all_at_once_releases_immediately(self) -> None:
        """The all_at_once kind imposes no gating."""
        jobs = [make_canned_job() for _ in range(2)]
        source = TimedJobSource(jobs, ArrivalSchedule(kind="all_at_once"), clock=lambda: 0.0)
        assert source.next_pop_response().id_ is not None
        assert source.next_pop_response().id_ is not None


class TestScenario:
    """The declarative spec expands consistently for the harness path."""

    def _spec(self) -> Scenario:
        return Scenario(
            name="mixed",
            image_jobs=[
                CannedImageJobSpec(model="Deliberate", count=2, lora_names=["lora-a"]),
                CannedImageJobSpec(model="AlbedoBase XL (SDXL)", width=1024, height=1024, count=1),
            ],
            alchemy_forms=[CannedAlchemyFormSpec(form="caption", count=2)],
        )

    def test_totals(self) -> None:
        """Job/form totals account for per-spec counts."""
        spec = self._spec()
        assert spec.total_image_jobs == 3
        assert spec.total_alchemy_forms == 2

    def test_expansion(self) -> None:
        """Expansion produces unique jobs with the requested features."""
        spec = self._spec()
        jobs = spec.expand_image_jobs()
        assert len(jobs) == 3
        assert jobs[0].payload.loras is not None and jobs[0].payload.loras[0].name == "lora-a"
        assert jobs[2].model == "AlbedoBase XL (SDXL)"
        assert len({job.id_ for job in jobs}) == 3, "every expanded job needs a unique id"

        forms = spec.expand_alchemy_forms()
        assert [form.form for form in forms] == ["caption", "caption"]

    def test_to_canned_sources(self) -> None:
        """The harness-path sources cover all jobs and forms."""
        job_source, alchemy_source = self._spec().to_canned_sources()
        assert job_source.total_jobs == 3
        assert alchemy_source is not None
        assert alchemy_source.total_forms == 2

    def test_timed_source_when_scheduled(self) -> None:
        """A non-trivial arrival kind yields a TimedJobSource."""
        spec = self._spec()
        spec.arrival_kind = "steady"
        spec.arrival_rate_per_minute = 30
        job_source, _ = spec.to_canned_sources()
        assert isinstance(job_source, TimedJobSource)

    def test_models_referenced(self) -> None:
        """The distinct model list is sorted and deduplicated."""
        assert self._spec().models_referenced() == ["AlbedoBase XL (SDXL)", "Deliberate"]
