"""Deterministic e2e canary simulations for representative worker lifecycles.

These are deliberately not exhaustive permutations. Each case is a named, foreseeable
volunteer-host profile that runs the real process manager against fake child processes,
so the scheduler, process lifecycle, IPC, job tracker, safety, submit, and audit paths
all move together under simulated load.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import pytest
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse

from horde_worker_regen.benchmark.scenarios import CannedImageJobSpec, Scenario
from horde_worker_regen.harness import HarnessConfig, HarnessResult, run_harness_async
from horde_worker_regen.process_management.ipc.messages import PipelineStageTag
from horde_worker_regen.process_management.process_manager import SystemResources
from horde_worker_regen.process_management.resources.device_info import TorchDeviceInfo, TorchDeviceMap
from horde_worker_regen.process_management.simulation._canned_scenarios import (
    ArrivalSchedule,
    SoakImageTemplate,
    make_alchemy_scenario,
    make_canned_job,
)
from horde_worker_regen.process_management.simulation.fault_injection import FaultProfile

ScenarioFactory = Callable[[], list[ImageGenerateJobPopResponse]]

_PLAIN_CARD_MODEL = "Deliberate"
_CONTROL_CARD_MODEL = "Anything Diffusion"


@dataclass(frozen=True)
class CanaryCase:
    """One representative worker lifecycle canary."""

    scenario_factory: ScenarioFactory
    system_resources: SystemResources
    bridge_data_overrides: dict[str, object]
    arrival: ArrivalSchedule | None = None
    alchemy_forms: int = 0
    job_delay_seconds: float = 0.0
    timeout_seconds: float = 75.0
    inference_fault_profile: FaultProfile | None = None
    fake_initially_available_models: list[str] | None = None
    fake_download_delay_seconds: float = 0.0
    min_disaggregated_sample_stages: int = 0
    """How many jobs the case requires to have been sampled through the disaggregated pipeline.

    Zero for a case that does not enable disaggregation. A case that does needs a floor, or a run where the
    pipeline never engaged would satisfy the discarded-sample oracle by never producing a sample to discard."""


def _system_resources(
    *,
    ram_gb: int,
    cards: tuple[tuple[int, int, str], ...],
    per_process_overhead_mb: int = 0,
    marginal_process_overhead_mb: int = 0,
) -> SystemResources:
    """Build synthetic hardware resources from ``(index, vram_gb, kind)`` card tuples."""
    return SystemResources(
        total_ram_bytes=ram_gb * 1024 * 1024 * 1024,
        device_map=TorchDeviceMap(
            root={
                index: TorchDeviceInfo(
                    device_name=f"Canary {kind.upper()} {index}",
                    device_index=index,
                    total_memory=vram_gb * 1024 * 1024 * 1024,
                    kind=kind,
                )
                for index, vram_gb, kind in cards
            },
        ),
        per_process_overhead_mb=per_process_overhead_mb,
        marginal_process_overhead_mb=marginal_process_overhead_mb,
    )


def _heterogeneous_card_resources() -> SystemResources:
    """Return the two-card topology shared by the capability lifecycle canaries."""
    return _system_resources(
        ram_gb=32,
        cards=((0, 8, "cuda"), (1, 12, "cuda")),
        per_process_overhead_mb=1000,
    )


def _heterogeneous_card_overrides() -> dict[str, object]:
    """Return two non-equivalent card offers with exactly one model assigned to each card."""
    return {
        "gpu_device_indices": [0, 1],
        "max_threads": 1,
        "queue_size": 1,
        "gpu_overrides": {
            0: {
                "models_to_load": [_PLAIN_CARD_MODEL],
                "allow_img2img": False,
                "allow_controlnet": False,
            },
            1: {
                "models_to_load": [_CONTROL_CARD_MODEL],
                "allow_img2img": True,
                "allow_controlnet": True,
            },
        },
    }


def _heterogeneous_routeable_queue() -> list[ImageGenerateJobPopResponse]:
    """Return two jobs per card so each original child reaches its scripted second-job boundary."""
    return [
        make_canned_job(_PLAIN_CARD_MODEL),
        make_canned_job(_CONTROL_CARD_MODEL, control_type="canny"),
        make_canned_job(_PLAIN_CARD_MODEL),
        make_canned_job(_CONTROL_CARD_MODEL, control_type="canny"),
    ]


def _feature_mix_scenario() -> list[ImageGenerateJobPopResponse]:
    """Queue with features that touch routing, safety, submit, and aux/preload metadata paths."""
    return Scenario(
        name="canary-feature-mix",
        image_jobs=[
            CannedImageJobSpec(model="Deliberate", width=512, height=512, steps=20),
            CannedImageJobSpec(model="Deliberate", control_type="canny"),
            CannedImageJobSpec(model="Deliberate", width=768, height=768, hires_fix=True),
            CannedImageJobSpec(model="Deliberate", post_processing=["RealESRGAN_x4plus"]),
            CannedImageJobSpec(model="Deliberate", width=640, height=832, steps=25),
        ],
    ).expand_image_jobs()


def _mainstream_mixed_queue() -> list[ImageGenerateJobPopResponse]:
    """Mixed model and image-size pressure without making the run long.

    The two models alternate (rather than grouping) to keep the scheduler preloading, swapping, and
    unloading between jobs; the explicit per-job specs preserve that interleaving.
    """
    return Scenario(
        name="canary-mainstream-mixed",
        image_jobs=[
            CannedImageJobSpec(model="Deliberate"),
            CannedImageJobSpec(model="Anything Diffusion"),
            CannedImageJobSpec(model="Deliberate"),
            CannedImageJobSpec(model="Anything Diffusion"),
            CannedImageJobSpec(model="Deliberate", width=512, height=512, steps=20),
            CannedImageJobSpec(model="Deliberate", width=1024, height=1024, steps=50),
            CannedImageJobSpec(model="Deliberate", width=768, height=768, steps=30),
            CannedImageJobSpec(model="Anything Diffusion", n_iter=2),
        ],
    ).expand_image_jobs()


def _aux_feature_churn_queue() -> list[ImageGenerateJobPopResponse]:
    """Aux/model-feature queue inspired by LoRA, TI, ControlNet, and post-processing repros."""
    return Scenario(
        name="canary-aux-feature-churn",
        image_jobs=[
            CannedImageJobSpec(model="Deliberate", lora_names=["canary-lora-a"]),
            CannedImageJobSpec(model="Deliberate", lora_names=["canary-lora-b"]),
            CannedImageJobSpec(model="Deliberate", ti_names=["canary-ti"]),
            CannedImageJobSpec(model="Deliberate", control_type="canny"),
            CannedImageJobSpec(model="Deliberate", post_processing=["GFPGAN"]),
            CannedImageJobSpec(model="Deliberate", n_iter=2),
        ],
    ).expand_image_jobs()


def _transient_resource_fault_queue() -> list[ImageGenerateJobPopResponse]:
    """Short queue where one fake OOM should retry without losing later work."""
    return [
        make_canned_job("Deliberate", width=512, height=512, ddim_steps=20),
        make_canned_job("Deliberate", width=1024, height=1024, ddim_steps=35),
        make_canned_job("Anything Diffusion", width=768, height=768, ddim_steps=25),
        make_canned_job("Deliberate", width=640, height=640, ddim_steps=20),
    ]


def _explicit_gpu_selection_queue() -> list[ImageGenerateJobPopResponse]:
    """Queue for a worker explicitly pinned to one card on a heterogeneous host."""
    return [
        make_canned_job("Deliberate", width=512, height=512, ddim_steps=20),
        make_canned_job("Anything Diffusion", width=512, height=768, ddim_steps=25),
        make_canned_job("Deliberate", width=768, height=512, ddim_steps=25),
        make_canned_job("Anything Diffusion", width=768, height=768, ddim_steps=30),
        make_canned_job("Deliberate", width=1024, height=1024, ddim_steps=35),
        make_canned_job("Anything Diffusion", width=640, height=640, ddim_steps=20),
    ]


def _whole_card_mixed_queue() -> list[ImageGenerateJobPopResponse]:
    """Large-model residency queue with smaller sibling models before and after the Flux head."""
    return [
        make_canned_job("CyberRealistic Pony", width=768, height=768, ddim_steps=25),
        make_canned_job("Flux.1-Schnell fp8 (Compact)", width=1216, height=1216, ddim_steps=4),
        make_canned_job("Juggernaut XL", width=1024, height=1024, ddim_steps=30),
        make_canned_job("Flux.1-Schnell fp8 (Compact)", width=1024, height=1024, ddim_steps=4),
        make_canned_job("CyberRealistic Pony", width=640, height=896, ddim_steps=25),
    ]


def _whole_card_disagg_mixed_queue() -> list[ImageGenerateJobPopResponse]:
    """Whole-card heads interleaved with SDXL jobs the pipeline serves disaggregated.

    Eligibility is per job: the SDXL entries run through the encode, sample and decode lanes while the Flux
    entries take the whole card. The heads sit at odd ordinals so that, under the paired burst arrival, each
    arrives while the SDXL job ahead of it is still in flight and the whole-card claim has to decide what to
    do about a staged job that is mid-pipeline.
    """
    return [
        make_canned_job("CyberRealistic Pony", width=768, height=768, ddim_steps=25),
        make_canned_job("Juggernaut XL", width=1024, height=1024, ddim_steps=30),
        make_canned_job("Flux.1-Schnell fp8 (Compact)", width=1216, height=1216, ddim_steps=4),
        make_canned_job("Juggernaut XL", width=832, height=1216, ddim_steps=30),
        make_canned_job("CyberRealistic Pony", width=640, height=896, ddim_steps=25),
        make_canned_job("Flux.1-Schnell fp8 (Compact)", width=1024, height=1024, ddim_steps=4),
        make_canned_job("Juggernaut XL", width=1024, height=1024, ddim_steps=30),
    ]


def _cold_start_download_queue() -> list[ImageGenerateJobPopResponse]:
    """Queue whose later configured models appear through the fake download process."""
    return [
        make_canned_job("Deliberate", width=512, height=512, ddim_steps=20),
        make_canned_job("Anything Diffusion", width=768, height=768, ddim_steps=25),
        make_canned_job("CyberRealistic Pony", width=1024, height=1024, ddim_steps=30),
        make_canned_job("Deliberate", width=640, height=832, ddim_steps=25),
        make_canned_job("Anything Diffusion", width=512, height=768, ddim_steps=20),
    ]


_CANARY_CASES: dict[str, CanaryCase] = {
    "constrained_single_card_feature_mix": CanaryCase(
        scenario_factory=_feature_mix_scenario,
        system_resources=_system_resources(ram_gb=12, cards=((0, 6, "cuda"),), per_process_overhead_mb=900),
        bridge_data_overrides={
            "max_threads": 1,
            "queue_size": 2,
            "cycle_process_on_model_change": True,
            "allow_controlnet": True,
            "allow_post_processing": True,
            "high_performance_mode": True,
        },
        timeout_seconds=75.0,
    ),
    "mainstream_mixed_queue_bursts": CanaryCase(
        scenario_factory=_mainstream_mixed_queue,
        system_resources=_system_resources(
            ram_gb=32,
            cards=((0, 12, "cuda"),),
            per_process_overhead_mb=1500,
            marginal_process_overhead_mb=600,
        ),
        bridge_data_overrides={
            "max_threads": 2,
            "queue_size": 3,
            "gpu_sampling_lease_enabled": True,
            "gpu_sampling_lease_slots": 1,
            "post_process_job_overlap": True,
        },
        arrival=ArrivalSchedule(kind="bursts", burst_size=3, burst_interval_seconds=0.2),
        alchemy_forms=2,
        job_delay_seconds=0.03,
        timeout_seconds=90.0,
    ),
    "aux_feature_churn_under_queue_pressure": CanaryCase(
        scenario_factory=_aux_feature_churn_queue,
        system_resources=_system_resources(ram_gb=24, cards=((0, 10, "cuda"),), per_process_overhead_mb=1200),
        bridge_data_overrides={
            "max_threads": 1,
            "queue_size": 3,
            "allow_lora": True,
            "allow_controlnet": True,
            "allow_post_processing": True,
            "cycle_process_on_model_change": False,
        },
        job_delay_seconds=0.02,
        timeout_seconds=90.0,
    ),
    "transient_oom_retry_keeps_queue_flowing": CanaryCase(
        scenario_factory=_transient_resource_fault_queue,
        system_resources=_system_resources(ram_gb=16, cards=((0, 8, "cuda"),), per_process_overhead_mb=1100),
        bridge_data_overrides={
            "max_threads": 1,
            "queue_size": 2,
            "max_inference_attempts": 2,
            "overbudget_exclusive_mode": True,
        },
        job_delay_seconds=0.02,
        timeout_seconds=90.0,
        inference_fault_profile=FaultProfile(oom_on_job_n=2),
    ),
    "explicit_secondary_gpu_selection": CanaryCase(
        scenario_factory=_explicit_gpu_selection_queue,
        system_resources=_system_resources(
            ram_gb=64,
            cards=((0, 8, "cuda"), (1, 24, "cuda")),
            per_process_overhead_mb=1800,
            marginal_process_overhead_mb=700,
        ),
        bridge_data_overrides={
            "gpu_device_indices": [1],
            "gpu_overrides": {
                1: {"max_threads": 2, "queue_size": 2},
            },
            "max_threads": 1,
            "queue_size": 1,
            "gpu_pop_balance_threshold": 0.25,
        },
        job_delay_seconds=0.02,
        timeout_seconds=90.0,
    ),
    "whole_card_flux_sdxl_residency_churn": CanaryCase(
        scenario_factory=_whole_card_mixed_queue,
        system_resources=_system_resources(
            ram_gb=32,
            cards=((0, 24, "cuda"),),
            per_process_overhead_mb=4200,
            marginal_process_overhead_mb=2000,
        ),
        bridge_data_overrides={
            "enable_vram_budget": True,
            "whole_card_exclusive_residency": True,
            "whole_card_residency_cooldown_seconds": 0,
            "overbudget_exclusive_mode": True,
            "vram_reserve_mb": 2048,
            "ram_reserve_mb": 4096,
            "max_threads": 2,
            "queue_size": 3,
        },
        arrival=ArrivalSchedule(kind="bursts", burst_size=2, burst_interval_seconds=0.15),
        job_delay_seconds=0.02,
        timeout_seconds=90.0,
    ),
    "whole_card_disagg_residency_churn": CanaryCase(
        scenario_factory=_whole_card_disagg_mixed_queue,
        system_resources=_system_resources(
            ram_gb=32,
            cards=((0, 24, "cuda"),),
            per_process_overhead_mb=4200,
            marginal_process_overhead_mb=2000,
        ),
        bridge_data_overrides={
            "enable_vram_budget": True,
            "whole_card_exclusive_residency": True,
            "whole_card_residency_cooldown_seconds": 0,
            "overbudget_exclusive_mode": True,
            "vram_reserve_mb": 2048,
            "ram_reserve_mb": 4096,
            "max_threads": 2,
            "queue_size": 3,
            "enable_pipeline_disaggregation": True,
            "gpu_sampling_lease_enabled": True,
            "gpu_sampling_lease_slots": 1,
            # Retention leaves a sampler's weights on the card between its same-model jobs; the eager evictor
            # would return them after every one and the pipeline would never hold a pinned sampler across the
            # whole-card window this case exists to cross.
            "unload_models_from_vram_often": False,
        },
        arrival=ArrivalSchedule(kind="bursts", burst_size=2, burst_interval_seconds=0.2),
        job_delay_seconds=0.5,
        timeout_seconds=180.0,
        min_disaggregated_sample_stages=2,
    ),
    "cold_start_download_availability_queue": CanaryCase(
        scenario_factory=_cold_start_download_queue,
        system_resources=_system_resources(
            ram_gb=24,
            cards=((0, 12, "cuda"),),
            per_process_overhead_mb=1400,
            marginal_process_overhead_mb=500,
        ),
        bridge_data_overrides={
            "max_threads": 1,
            "queue_size": 2,
            "allow_post_processing": True,
        },
        arrival=ArrivalSchedule(kind="bursts", burst_size=2, burst_interval_seconds=0.1),
        job_delay_seconds=0.02,
        timeout_seconds=90.0,
        fake_initially_available_models=["Deliberate"],
        fake_download_delay_seconds=0.03,
    ),
}


_MAX_IDLE_FREE_FRACTION = 0.9
"""Free VRAM, as a share of a card's total, a driven card must at some point have fallen below.

A simulated card that never gets charged reads as empty for the whole run, and every memory oracle over
such a run passes without judging anything: the governor stays HEALTHY because nothing was ever resident,
and "never saturated" is a statement about arithmetic that never moved. Requiring a real dip is what makes
those oracles mean something. Deliberately loose, because how far a card is driven is the case's business;
this only establishes that it was driven at all."""


def _assert_governed_a_loaded_card(result: HarnessResult) -> None:
    """Assert the run's device-free governor judged a card carrying real load, and never saturated it.

    Two statements in one, because either alone is weak. The occupancy floor says the governor had
    something to govern, and the band trajectory says what it made of it. A card whose free reading never
    left its total satisfies the second for free, which is why the first comes with it.
    """
    assert result.observed_device_total_mb, (
        f"the run recorded no device capacity, so its governor read no card at all: {result.failure_summary()}"
    )
    for device_index, total_mb in sorted(result.observed_device_total_mb.items()):
        min_free_mb = result.min_observed_device_free_mb.get(device_index)
        assert min_free_mb is not None, f"card {device_index} reported a total but no free reading"
        assert min_free_mb < total_mb * _MAX_IDLE_FREE_FRACTION, (
            f"card {device_index} never fell below {_MAX_IDLE_FREE_FRACTION:.0%} free "
            f"({min_free_mb:.0f}MB of {total_mb:.0f}MB): it carried no simulated residency, so every "
            f"memory judgement this run made was against an empty card"
        )
    saturations = [step for step in result.governor_state_transitions if step.endswith("saturated")]
    assert not saturations, (
        f"the device-free governor crossed the paging cliff: {saturations} "
        f"(trajectory {result.governor_state_transitions}): {result.failure_summary()}"
    )


def _assert_clean_canary_result(
    result: HarnessResult,
    *,
    scenario: list[ImageGenerateJobPopResponse],
) -> None:
    expected_jobs = len(scenario)
    assert not result.timed_out, result.failure_summary()
    assert result.exit_reason == "completed", result.failure_summary()
    assert result.all_jobs_accounted_for, result.failure_summary()
    assert result.num_jobs_completed == expected_jobs, result.failure_summary()
    assert result.num_jobs_faulted == 0, result.failure_summary()
    assert result.num_jobs_submitted_faulted == 0, result.failure_summary()
    assert result.audit_failures == [], result.failure_summary()
    assert result.diagnostics == [], result.failure_summary()
    # No finished sample is ever discarded: a job rerouted monolithically after its sample result had arrived
    # throws away a completed denoise and runs the whole job again. The queue still drains, so completion
    # counts alone cannot see it; only this counter can. Trivially satisfied by a case with no pipeline.
    assert result.disaggregation_discarded_sample_reroutes == 0, result.failure_summary()
    assert result.metrics is not None
    assert result.metrics.process_crash_events == []
    _assert_governed_a_loaded_card(result)

    image_records = [record for record in result.metrics.jobs if not record.is_alchemy]
    assert len(image_records) == expected_jobs
    assert Counter(record.model_name for record in image_records) == Counter(job.model for job in scenario)
    assert all(not record.faulted for record in image_records)
    # A disaggregated job's phase metrics arrive per stage, from each lane that ran one, and are kept as
    # standalone stage records rather than folded into the whole-job record. Its whole-job record therefore
    # carries none, so the whole-job requirement applies to jobs that ran monolithically.
    staged_job_ids = {record.job_id for record in result.metrics.stage_metrics}
    assert all(record.phase_metrics is not None for record in image_records if record.job_id not in staged_job_ids)
    for record in image_records:
        assert "INFERENCE_IN_PROGRESS" in record.stage_timestamps
        assert "PENDING_SAFETY_CHECK" in record.stage_timestamps
        assert "PENDING_SUBMIT" in record.stage_timestamps
        assert "FINALIZED" in record.stage_timestamps


# Each case boots a real worker manager and spawns real OS child processes through the harness, so the whole
# module is opt-in via -m slow (skipped in a default sweep).
pytestmark = pytest.mark.slow


@pytest.mark.e2e
@pytest.mark.parametrize("case_name", sorted(_CANARY_CASES))
async def test_representative_worker_lifecycle_canaries(case_name: str) -> None:
    """Representative volunteer-host profiles should drain cleanly under simulated load."""
    case = _CANARY_CASES[case_name]
    scenario = case.scenario_factory()
    alchemy_forms = make_alchemy_scenario(["caption", "RealESRGAN_x4plus"], case.alchemy_forms)
    result = await run_harness_async(
        HarnessConfig(
            scenario=scenario,
            alchemy_forms=alchemy_forms,
            arrival=case.arrival,
            process_mode="fake",
            skip_api=True,
            job_delay_seconds=case.job_delay_seconds,
            timeout_seconds=case.timeout_seconds,
            bridge_data_overrides=case.bridge_data_overrides,
            system_resources=case.system_resources,
            inference_fault_profile=case.inference_fault_profile,
            fake_initially_available_models=case.fake_initially_available_models,
            fake_download_delay_seconds=case.fake_download_delay_seconds,
        ),
    )

    _assert_clean_canary_result(result, scenario=scenario)
    if case.min_disaggregated_sample_stages:
        assert result.metrics is not None
        sample_stages = sum(1 for record in result.metrics.stage_metrics if record.stage is PipelineStageTag.SAMPLE)
        assert sample_stages >= case.min_disaggregated_sample_stages, (
            f"only {sample_stages} job(s) reached the disaggregated sample stage, so the pipeline barely ran "
            f"and the discarded-sample oracle judged almost nothing: {result.failure_summary()}"
        )
    if case.fake_initially_available_models is not None:
        expected_models = {job.model for job in scenario if job.model is not None}
        assert set(case.fake_initially_available_models) < expected_models
        assert result.model_availability_known, result.failure_summary()
        assert expected_models <= set(result.available_model_names), result.failure_summary()
        assert result.failed_download_model_names == [], result.failure_summary()

    assert result.num_alchemy_forms_completed == case.alchemy_forms, result.failure_summary()
    assert result.num_alchemy_forms_faulted == 0, result.failure_summary()


@pytest.mark.e2e
async def test_heterogeneous_card_offer_and_routing_survive_spawned_worker_lifecycle() -> None:
    """Card-scoped SDK offers feed only routeable work through a spawned multi-card worker.

    The third template is deliberately a cross-card combination: its model belongs to the plain card while
    its ControlNet requirement belongs to the other card. It has all of the global generation weight, so an
    unsafe union offer selects it deterministically. Complete card-scoped offers exclude it and leave one
    valid zero-weight template each; the source then normalizes that singleton and exercises both cards.
    """
    result = await run_harness_async(
        HarnessConfig(
            process_mode="fake",
            skip_api=True,
            soak_seconds=10.0,
            soak_drain_timeout_seconds=30.0,
            timeout_seconds=90.0,
            job_delay_seconds=0.02,
            soak_image_templates=[
                SoakImageTemplate(model=_PLAIN_CARD_MODEL, weight=0.0),
                SoakImageTemplate(model=_CONTROL_CARD_MODEL, control_type="canny", weight=0.0),
                SoakImageTemplate(model=_PLAIN_CARD_MODEL, control_type="canny", weight=1.0),
            ],
            system_resources=_heterogeneous_card_resources(),
            bridge_data_overrides=_heterogeneous_card_overrides(),
        ),
    )

    assert not result.timed_out, result.failure_summary()
    assert result.exit_reason == "completed", result.failure_summary()
    assert result.num_jobs_faulted == 0, result.failure_summary()
    assert result.num_jobs_submitted_faulted == 0, result.failure_summary()
    assert result.audit_failures == [], result.failure_summary()
    assert result.diagnostics == [], result.failure_summary()
    assert result.metrics is not None, result.failure_summary()
    assert result.metrics.process_crash_events == [], result.failure_summary()

    records = [record for record in result.metrics.jobs if not record.is_alchemy and not record.faulted]
    completed_models = Counter(record.model_name for record in records)
    assert completed_models[_PLAIN_CARD_MODEL] >= 1, result.failure_summary()
    assert completed_models[_CONTROL_CARD_MODEL] >= 1, result.failure_summary()
    assert all(record.control_type is None for record in records if record.model_name == _PLAIN_CARD_MODEL)
    assert all(record.control_type == "canny" for record in records if record.model_name == _CONTROL_CARD_MODEL)


@pytest.mark.e2e
async def test_heterogeneous_card_routing_recovers_after_spawned_children_exit() -> None:
    """Per-card eligibility and placement remain valid after each card's original child is replaced."""
    scenario = _heterogeneous_routeable_queue()
    result = await run_harness_async(
        HarnessConfig(
            scenario=scenario,
            process_mode="fake",
            skip_api=True,
            timeout_seconds=90.0,
            job_delay_seconds=0.02,
            inference_fault_profile=FaultProfile(crash_on_job_n=2),
            system_resources=_heterogeneous_card_resources(),
            bridge_data_overrides=_heterogeneous_card_overrides(),
        ),
    )

    assert not result.timed_out, result.failure_summary()
    assert result.exit_reason == "completed", result.failure_summary()
    assert result.all_jobs_accounted_for, result.failure_summary()
    assert result.num_jobs_completed == len(scenario), result.failure_summary()
    assert result.num_jobs_faulted == 0, result.failure_summary()
    assert result.num_jobs_submitted_faulted == 0, result.failure_summary()
    assert result.audit_failures == [], result.failure_summary()
    assert result.diagnostics == [], result.failure_summary()
    assert result.metrics is not None, result.failure_summary()
    assert result.metrics.process_crash_events, result.failure_summary()

    completed_models = Counter(record.model_name for record in result.metrics.jobs if not record.faulted)
    assert completed_models == Counter({_PLAIN_CARD_MODEL: 2, _CONTROL_CARD_MODEL: 2})
