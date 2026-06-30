"""Tests for the canned scenario factories, arrival control, and Scenario (no GPU)."""

from __future__ import annotations

import pytest

from horde_worker_regen.benchmark.scenarios import (
    CannedAlchemyFormSpec,
    CannedImageJobSpec,
    Scenario,
)
from horde_worker_regen.process_management.simulation._canned_scenarios import (
    ArrivalSchedule,
    CannedAlchemySource,
    TimedJobSource,
    make_alchemy_scenario,
    make_canned_job,
    make_controlnet_scenario,
    make_hires_fix_scenario,
    make_lora_scenario,
    make_post_processing_scenario,
    make_ti_scenario,
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


class TestAlchemyScenario:
    """Canned alchemy form generation and the alchemy source."""

    def test_forms_cycle_and_have_ids(self) -> None:
        """Form names cycle and every form gets a unique ID and source image."""
        forms = make_alchemy_scenario(["caption", "RealESRGAN_x4plus"], 3)
        assert [form.form for form in forms] == ["caption", "RealESRGAN_x4plus", "caption"]
        assert len({form.form_id for form in forms}) == 3
        assert all(form.source_image_base64 for form in forms)

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
