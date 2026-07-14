"""Declarative workload descriptions shared by the harness and live load-gen paths.

A :class:`Scenario` describes *what* work a benchmark level offers (image jobs with
features, alchemy forms, and an arrival schedule) independently of *how* it is driven:
``to_canned_sources()`` produces the pop-side sources consumed by the e2e harness
(``skip_api=True``), and the live path translates the same spec into SDK submit requests.
Keeping one spec for both guarantees the two execution paths stay in lockstep.
"""

from __future__ import annotations

from horde_sdk.ai_horde_api.apimodels import (
    ImageGenerateJobPopResponse,
    LorasPayloadEntry,
    TIPayloadEntry,
)
from pydantic import BaseModel, Field

from horde_worker_regen.process_management.ipc.messages import AlchemyFormSpec
from horde_worker_regen.process_management.simulation._canned_scenarios import (
    ArrivalSchedule,
    CannedAlchemySource,
    CannedJobSource,
    SoakImageTemplate,
    TimedJobSource,
    make_alchemy_scenario,
    make_canned_job,
)


class CannedImageJobSpec(BaseModel):
    """One image job description, expandable to a pop response or a submit request."""

    model: str = "Deliberate"
    width: int = 512
    height: int = 512
    steps: int = 30
    cfg_scale: float | None = None
    n_iter: int = 1
    hires_fix: bool = False
    lora_names: list[str] = Field(default_factory=list)
    ti_names: list[str] = Field(default_factory=list)
    control_type: str | None = None
    workflow: str | None = None
    """A named hordelib workflow (e.g. ``qr_code``); the SDXL controlnet capability is this, not
    a preprocessor control type. Sets ``payload.workflow``."""
    post_processing: list[str] = Field(default_factory=list)
    count: int = 1
    """How many identical jobs this spec expands to."""


class CannedAlchemyFormSpec(BaseModel):
    """One alchemy form description (form name x count)."""

    form: str
    count: int = 1


class Scenario(BaseModel):
    """A complete benchmark workload: image jobs + alchemy forms + arrival structure.

    This is the single source of truth for a workload across every driver: the benchmark CLI,
    the e2e harness (``pytest -m e2e``, fake mode), and the gpu catalog (``pytest -m gpu``).
    It owns *what* work runs; *how* it is perturbed (faults, simulated VRAM, arrival overrides)
    stays as harness kwargs on the low-level path.
    """

    name: str
    image_jobs: list[CannedImageJobSpec] = Field(default_factory=list)
    alchemy_forms: list[CannedAlchemyFormSpec] = Field(default_factory=list)
    arrival_kind: str = "all_at_once"
    """One of "all_at_once", "steady", or "bursts" (see ArrivalSchedule)."""
    arrival_rate_per_minute: float = 0.0
    arrival_burst_size: int = 0
    arrival_burst_interval_seconds: float = 0.0
    soak_seconds: float | None = None
    """When set, this is a sustained-load soak: jobs/forms are *generated* continuously from
    the specs (their `count` becomes a relative weight) for this many seconds rather than the
    specs being expanded into a fixed list. See `to_soak_templates`."""

    @property
    def total_image_jobs(self) -> int:
        """The total number of image jobs this scenario expands to."""
        return sum(spec.count for spec in self.image_jobs)

    @property
    def total_alchemy_forms(self) -> int:
        """The total number of alchemy forms this scenario expands to."""
        return sum(spec.count for spec in self.alchemy_forms)

    def arrival_schedule(self) -> ArrivalSchedule:
        """Build the arrival schedule described by this spec."""
        return ArrivalSchedule(
            kind=self.arrival_kind,
            rate_per_minute=self.arrival_rate_per_minute,
            burst_size=self.arrival_burst_size,
            burst_interval_seconds=self.arrival_burst_interval_seconds,
        )

    def expand_image_jobs(self) -> list[ImageGenerateJobPopResponse]:
        """Expand the image job specs into concrete canned pop responses."""
        jobs: list[ImageGenerateJobPopResponse] = []
        for spec in self.image_jobs:
            for _ in range(spec.count):
                jobs.append(
                    make_canned_job(
                        spec.model,
                        width=spec.width,
                        height=spec.height,
                        ddim_steps=spec.steps,
                        n_iter=spec.n_iter,
                        hires_fix=spec.hires_fix,
                        cfg_scale=spec.cfg_scale,
                        loras=(
                            [LorasPayloadEntry(name=name) for name in spec.lora_names] if spec.lora_names else None
                        ),
                        tis=(
                            [TIPayloadEntry(name=name, inject_ti="prompt") for name in spec.ti_names]
                            if spec.ti_names
                            else None
                        ),
                        control_type=spec.control_type,
                        workflow=spec.workflow,
                        post_processing=spec.post_processing if spec.post_processing else None,
                    ),
                )
        return jobs

    def expand_alchemy_forms(self) -> list[AlchemyFormSpec]:
        """Expand the alchemy form specs into concrete form specs with source images."""
        forms: list[AlchemyFormSpec] = []
        for spec in self.alchemy_forms:
            forms.extend(make_alchemy_scenario([spec.form], spec.count))
        return forms

    def to_canned_sources(self) -> tuple[CannedJobSource, CannedAlchemySource | None]:
        """Build the harness-path pop sources for this scenario."""
        jobs = self.expand_image_jobs()
        schedule = self.arrival_schedule()
        job_source: CannedJobSource = (
            TimedJobSource(jobs, schedule) if schedule.kind != "all_at_once" else CannedJobSource(jobs)
        )

        alchemy_forms = self.expand_alchemy_forms()
        alchemy_source = CannedAlchemySource(alchemy_forms) if alchemy_forms else None
        return job_source, alchemy_source

    def to_soak_templates(self) -> tuple[list[SoakImageTemplate], list[tuple[str, float]]]:
        """Convert the specs into weighted soak templates (their `count` becomes the weight).

        Returns the image templates and the ``(form_name, weight)`` alchemy pairs the harness's
        generating sources mint fresh jobs/forms from during a soak.
        """
        image_templates = [
            SoakImageTemplate(
                model=spec.model,
                width=spec.width,
                height=spec.height,
                steps=spec.steps,
                cfg_scale=spec.cfg_scale,
                n_iter=spec.n_iter,
                control_type=spec.control_type,
                workflow=spec.workflow,
                post_processing=list(spec.post_processing),
                hires_fix=spec.hires_fix,
                loras=[LorasPayloadEntry(name=name) for name in spec.lora_names],
                tis=[TIPayloadEntry(name=name, inject_ti="prompt") for name in spec.ti_names],
                weight=float(spec.count),
            )
            for spec in self.image_jobs
        ]
        alchemy_templates = [(spec.form, float(spec.count)) for spec in self.alchemy_forms]
        return image_templates, alchemy_templates

    def models_referenced(self) -> list[str]:
        """Return the distinct image models this scenario uses."""
        return sorted({spec.model for spec in self.image_jobs})
