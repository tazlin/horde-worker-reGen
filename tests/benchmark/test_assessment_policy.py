"""Evidence boundaries for adaptive capability selection and enforceable configuration export."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_model_reference.model_reference_records import ImageGenerationModelRecord
from horde_sdk.ai_horde_api.apimodels import ImageGenerateJobPopResponse, ImageStatsModelsResponse
from ruamel.yaml import YAML

from horde_worker_regen.benchmark import assessment_policy
from horde_worker_regen.benchmark.assessment_policy import (
    AssessmentCase,
    _eligible_model_records,
    combined_case,
    installed_models_for_tiers,
    measured_recommendation,
    model_guidance,
    next_boundary,
    representative_cases,
    select_installed_models,
)
from horde_worker_regen.benchmark.capabilities.capability import Capability, CapabilityKind
from horde_worker_regen.benchmark.capabilities.catalog import CatalogOptions
from horde_worker_regen.benchmark.capabilities.report_render import render_markdown
from horde_worker_regen.benchmark.capabilities.result import (
    CapabilityReport,
    EvidenceStatus,
    MachineInfo,
    OperatingProfile,
    WorkloadEvidence,
)
from horde_worker_regen.benchmark.enums import BenchTier
from horde_worker_regen.benchmark.ladder import HUGE_TIERS
from horde_worker_regen.process_management.resources.admission_identity import admission_margin_mb
from horde_worker_regen.tui.benchmark_launcher import apply_suggested_to_config


def test_installed_checkpoint_substitution_stays_within_family(monkeypatch: pytest.MonkeyPatch) -> None:
    """An available checkpoint may replace its family's absent canonical model only."""
    records = {
        "installed-sd15": ImageGenerationModelRecord.model_construct(baseline="stable_diffusion_1"),
        "installed-sdxl": ImageGenerationModelRecord.model_construct(baseline="stable_diffusion_xl"),
    }
    monkeypatch.setattr(
        "horde_worker_regen.benchmark.assessment_policy.is_model_present", lambda name, records, **kwargs: True
    )
    assert select_installed_models([BenchTier.SD15, BenchTier.SDXL], records) == {
        BenchTier.SD15: "installed-sd15",
        BenchTier.SDXL: "installed-sdxl",
    }


def test_explicit_batch_selection_preserves_requested_workload() -> None:
    """A targeted quantitative probe keeps its magnitude despite representative-category collapsing."""
    cases = representative_cases(
        CatalogOptions(tiers=[BenchTier.SD15]),
        {BenchTier.SD15: "installed-model"},
        [0],
        only_probe="sd15-batch-4",
    )
    selected = next(case for case in cases if case.capability.slug == "sd15-batch-4")
    assert selected.profile.batch == selected.scenario().image_jobs[0].n_iter == 4


@pytest.mark.parametrize("exclusions", [["installed-sd15"], ["all sd15"], ["top 1"]])
def test_model_selection_respects_literal_and_meta_exclusions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, exclusions: list[str]
) -> None:
    """The normal resolver removes excluded checkpoints before installed-model substitution."""
    records = {
        "installed-sd15": ImageGenerationModelRecord.model_construct(
            name="installed-sd15", baseline="stable_diffusion_1"
        ),
        "installed-sdxl": ImageGenerationModelRecord.model_construct(
            name="installed-sdxl", baseline="stable_diffusion_xl"
        ),
    }
    manager = Mock(spec=ModelReferenceManager)
    manager.get_all_model_references.return_value = {MODEL_REFERENCE_CATEGORY.image_generation: records}
    monkeypatch.setattr("horde_worker_regen.bridge_data.load_config.beta_aware_image_records", lambda manager: records)
    monkeypatch.setattr(
        "horde_sdk.ai_horde_api.ai_horde_clients.AIHordeAPIManualClient.submit_request",
        lambda *args, **kwargs: ImageStatsModelsResponse(
            day={"installed-sd15": 10}, month={"installed-sd15": 10}, total={"installed-sd15": 10}
        ),
    )
    monkeypatch.setattr(
        "horde_worker_regen.benchmark.assessment_policy.is_model_present", lambda *args, **kwargs: True
    )
    config = {"models_to_skip": exclusions}
    eligible = _eligible_model_records(records, config, manager)
    assert select_installed_models([BenchTier.SD15, BenchTier.SDXL], eligible) == {BenchTier.SDXL: "installed-sdxl"}
    assert config["models_to_skip"] == exclusions
    monkeypatch.chdir(tmp_path)
    with Path("bridgeData.yaml").open("w", encoding="utf-8") as stream:
        YAML().dump(config, stream)
    monkeypatch.setattr("horde_worker_regen.reference_helper.ensure_offline_reference_manager", lambda: manager)
    assert installed_models_for_tiers([BenchTier.SD15, BenchTier.SDXL]) == {BenchTier.SDXL: "installed-sdxl"}
    assert installed_models_for_tiers([BenchTier.SD15]) == {}


@pytest.mark.parametrize("card_key", [0, "0"])
def test_model_selection_ignores_exclusions_on_unselected_cards(
    monkeypatch: pytest.MonkeyPatch, card_key: int | str
) -> None:
    """Only participating cards constrain a shared representative checkpoint."""
    records = {
        "installed-sd15": ImageGenerationModelRecord.model_construct(
            name="installed-sd15", baseline="stable_diffusion_1"
        )
    }
    manager = Mock(spec=ModelReferenceManager)
    manager.get_all_model_references.return_value = {MODEL_REFERENCE_CATEGORY.image_generation: records}
    monkeypatch.setattr("horde_worker_regen.bridge_data.load_config.beta_aware_image_records", lambda manager: records)
    config = {
        "gpu_device_indices": [0],
        "gpu_overrides": {card_key: {"models_to_skip": ["installed-sd15"]}, 1: {"models_to_skip": ["other-model"]}},
    }
    assert not _eligible_model_records(records, config, manager)
    config["gpu_device_indices"] = [1]
    assert set(_eligible_model_records(records, config, manager)) == {"installed-sd15"}


def _evidence(
    *,
    batch: int = 1,
    threads: int = 1,
    device: int = 0,
    kind: CapabilityKind = CapabilityKind.BASELINE,
    status: EvidenceStatus = EvidenceStatus.MEASURED_SUCCESS,
    combined: bool = False,
    model: str = "installed-model",
    tier: BenchTier = BenchTier.SD15,
) -> WorkloadEvidence:
    return WorkloadEvidence(
        evidence_id=f"{device}-{kind}-{batch}-{threads}",
        capability=Capability(tier=tier, kind=kind, magnitude=batch),
        profile=OperatingProfile(device_index=device, tier=tier, model_name=model, batch=batch, threads=threads),
        status=status,
        reason="submitted",
        retry_provenance_complete=True,
        combined_validation=combined,
    )


def test_independent_maxima_need_a_joint_execution() -> None:
    """Separate passing batch and thread checks cannot validate their combined settings."""
    results = [
        _evidence(),
        _evidence(batch=4, kind=CapabilityKind.BATCH),
        _evidence(threads=2, kind=CapabilityKind.THREADS),
    ]
    suggestion = measured_recommendation(results, per_card=True)
    assert not suggestion.safe_to_apply
    assert not suggestion.models_to_load
    case = combined_case(results, device=0, tier=BenchTier.SD15)
    assert case is not None and case.combined
    assert case.profile.batch == 4 and case.profile.threads == 2
    results.append(_evidence(batch=4, threads=2, kind=CapabilityKind.SUSTAINED, combined=True))
    suggestion = measured_recommendation(results, per_card=True)
    assert suggestion.safe_to_apply
    assert suggestion.gpu_overrides[0].max_batch == 4
    assert suggestion.gpu_overrides[0].max_threads == 2


def test_predictions_and_failed_confirmation_cannot_enable_settings() -> None:
    """Predictions and failed combined checks cannot enable configuration application."""
    suggestion = measured_recommendation(
        [
            _evidence(),
            _evidence(batch=8, status=EvidenceStatus.PREDICTED, combined=True),
            _evidence(batch=4, status=EvidenceStatus.MEASURED_FAILURE, combined=True),
        ]
    )
    assert not suggestion.safe_to_apply
    assert not suggestion.gpu_overrides
    assert not suggestion.models_to_load


def test_thread_profile_below_one_lane_of_throughput_cannot_raise_threads() -> None:
    """A combined thread profile the throughput criterion left unknown does not confirm its thread count."""
    below_one_lane = _evidence(
        threads=2, kind=CapabilityKind.THREADS, status=EvidenceStatus.UNKNOWN, combined=True
    ).model_copy(
        update={"attempted": True, "reason_code": "throughput_below_single_lane", "lane_equivalent_throughput": 0.76}
    )
    suggestion = measured_recommendation([_evidence(), _evidence(combined=True), below_one_lane], per_card=True)
    assert suggestion.safe_to_apply
    assert suggestion.gpu_overrides[0].max_threads == 1


def test_confirmation_and_recommendation_carry_the_lease_the_thread_count_was_measured_under() -> None:
    """A thread count confirmed under the lease candidate is recommended together with that lease."""
    thread_check = _evidence(threads=2, kind=CapabilityKind.THREADS)
    thread_check.profile.lease_slots = 1
    case = combined_case([_evidence(), thread_check], device=0, tier=BenchTier.SD15)
    assert case is not None and case.profile.threads == 2 and case.profile.lease_slots == 1
    # The confirmation runs only the thread count its thread check ranked against one lane.
    assert case.check_work_count == case.work_count
    thread_case = AssessmentCase(thread_check.capability, thread_check.profile.model_copy(update={"work_jobs": 4}))
    assert thread_case.check_work_count == 2 * thread_case.work_count == 8
    confirmed = _evidence(threads=2, combined=True)
    confirmed.profile.lease_slots = 1
    suggestion = measured_recommendation([confirmed], per_card=True)
    assert suggestion.gpu_overrides[0].max_threads == 2
    assert suggestion.gpu_sampling_lease_enabled is True and suggestion.gpu_sampling_lease_slots == 1
    assert any(decision.setting == "gpu_sampling_lease_slots" for decision in suggestion.decisions)
    exported = YAML(typ="safe").load(suggestion.as_yaml_block())
    assert exported["gpu_sampling_lease_enabled"] is True and exported["gpu_sampling_lease_slots"] == 1


def test_operator_lease_settings_are_left_as_written() -> None:
    """A profile measured under the operator's own lease settings does not set the lease fields."""
    suggestion = measured_recommendation([_evidence(threads=2, combined=True)], per_card=True)
    assert suggestion.safe_to_apply and suggestion.gpu_overrides[0].max_threads == 2
    assert suggestion.gpu_sampling_lease_enabled is None and suggestion.gpu_sampling_lease_slots is None
    assert "gpu_sampling_lease" not in suggestion.as_yaml_block()
    single_lane = combined_case([_evidence()], device=0, tier=BenchTier.SD15)
    assert single_lane is not None and single_lane.profile.lease_slots is None
    assert single_lane.check_work_count == single_lane.work_count


def test_card_limits_are_separate_and_unmeasured_cards_are_not_offered() -> None:
    """Cards with different capacities retain their separately measured limits."""
    suggestion = measured_recommendation(
        [
            _evidence(batch=1, device=0, combined=True),
            _evidence(batch=4, threads=2, device=1, combined=True),
            _evidence(device=2, status=EvidenceStatus.UNKNOWN),
        ]
    )
    assert suggestion.gpu_overrides[0].max_batch == 1
    assert suggestion.gpu_overrides[1].max_batch == 4
    assert suggestion.gpu_device_indices == [0, 1]


def test_different_model_envelopes_are_not_merged_into_new_limits() -> None:
    """Configuration groups use exact jointly confirmed limits for the offered models."""
    suggestion = measured_recommendation(
        [
            _evidence(batch=4, threads=1, combined=True, model="batch-model"),
            _evidence(batch=1, threads=2, combined=True, model="threads-model"),
        ],
        per_card=True,
    )
    limits = suggestion.gpu_overrides[0]
    assert (limits.max_batch, limits.max_threads) in ((4, 1), (1, 2))
    assert not limits.models_to_load


def test_boundary_growth_requires_success_and_exceeds_old_batch_ceiling() -> None:
    """A measured pass can explore beyond four while failures stop that boundary."""
    boundary = next_boundary(_evidence(batch=4, kind=CapabilityKind.BATCH), include_concurrency=True)
    assert boundary is not None and boundary.profile.batch == 8
    assert next_boundary(_evidence(status=EvidenceStatus.MEASURED_FAILURE), include_concurrency=True) is None


def test_report_keeps_history_readable_and_labels_prediction() -> None:
    """Older reports load with empty evidence and new reports retain explicit provenance."""
    legacy = CapabilityReport.model_validate({"report_schema_version": 5, "run_id": "old"})
    assert not legacy.operator_assessment and not legacy.evidence
    report = CapabilityReport(operator_assessment=True, evidence=[_evidence(status=EvidenceStatus.PREDICTED)])
    rendered = render_markdown(report)
    assert "Estimated" in rendered and "installed-model" in rendered
    assert "Sustained" not in rendered or "unresolved" in rendered


@pytest.mark.parametrize("card_key", [0, "0"])
@pytest.mark.parametrize("painting_key", ["allow_painting", "allow_inpainting"])
def test_apply_writes_card_limits_and_keeps_other_card_policies(
    tmp_path: Path, card_key: int | str, painting_key: str
) -> None:
    """Applying measured limits preserves safety placement and writes every confirmed limit."""
    path = tmp_path / "bridgeData.yaml"
    key = "'0'" if isinstance(card_key, str) else "0"
    path.write_text(
        f"allow_lora: true\nallow_img2img: true\n{painting_key}: true\n"
        f"gpu_overrides:\n  {key}:\n    safety_on_gpu: false\n    allow_lora: true\n"
        f"    allow_img2img: true\n    {painting_key}: true\nmax_power: 64\n",
        encoding="utf-8",
    )
    suggestion = measured_recommendation([_evidence(batch=2, threads=2, combined=True)], per_card=True)
    apply_suggested_to_config(suggestion, path)
    applied = YAML(typ="safe").load(path)
    assert len(applied["gpu_overrides"]) == 1
    assert applied["gpu_overrides"][card_key]["safety_on_gpu"] is False
    assert applied["gpu_overrides"][card_key]["max_threads"] == 2
    assert applied["gpu_overrides"][card_key]["max_batch"] == 2
    assert applied["max_power"] == 8
    assert applied["gpu_device_indices"] == [0]
    assert applied["dynamic_models"] is False
    from horde_worker_regen.bridge_data.data_model import reGenBridgeData
    from horde_worker_regen.bridge_data.gpu_config import resolve_effective_gpu_config
    from horde_worker_regen.process_management.gpu.gpu_eligibility import image_worker_feature_flags

    config = reGenBridgeData.model_validate(applied)
    card = resolve_effective_gpu_config(config, config.gpu_overrides[0])
    assert not config.allow_lora and not config.allow_img2img and not config.allow_inpainting
    assert not card.allow_lora and not card.allow_img2img and not card.allow_inpainting
    features = image_worker_feature_flags(card).image_generation_feature_flags
    assert not features.loras
    assert [str(mode) for mode in features.source_processing] == ["txt2img"]
    exported = YAML(typ="safe").load(suggestion.as_yaml_block())
    assert exported["gpu_overrides"][0]["allow_lora"] is False
    assert exported["allow_img2img"] is False and exported["allow_painting"] is False


def test_older_operator_report_remains_readable_but_cannot_be_applied() -> None:
    """Reports without retry history and feature exclusions cannot enable configuration application."""
    report = CapabilityReport.model_validate(
        {
            "report_schema_version": 6,
            "operator_assessment": True,
            "suggested_bridge_data": {"safe_to_apply": True, "models_to_load": ["installed-model"]},
        }
    )
    assert not report.suggested_bridge_data.safe_to_apply
    assert "Rerun before applying settings." in render_markdown(report)
    with pytest.raises(ValueError, match="No combined configuration"):
        apply_suggested_to_config(report.suggested_bridge_data)


def test_retry_evidence_cannot_produce_a_recommendation_even_if_labeled_successful() -> None:
    """The recommendation boundary independently rejects a recovered or provenance-free joint result."""
    for update in ({"inference_failures": 1}, {"degraded_retries": 1}, {"retry_provenance_complete": False}):
        result = _evidence(combined=True).model_copy(update=update)
        assert not measured_recommendation([result]).safe_to_apply


@pytest.mark.parametrize("card_key", [0, "0"])
@pytest.mark.parametrize("painting_key", ["allow_painting", "allow_inpainting"])
def test_single_card_global_apply_and_restore_remove_only_replaced_fields(
    tmp_path: Path,
    card_key: int | str,
    painting_key: str,
) -> None:
    """Global recommendations remain effective after Apply and known-good restoration."""
    from horde_worker_regen.tui.benchmark_launcher import apply_known_good_to_config, suggested_config_overrides

    path = tmp_path / "bridgeData.yaml"
    key = "'0'" if isinstance(card_key, str) else "0"
    path.write_text(
        f"models_to_skip: [excluded-global]\ngpu_overrides:\n  {key}:\n"
        "    models_to_skip: [excluded-card]\n    safety_on_gpu: false\n    max_threads: 9\n"
        f"    allow_lora: true\n    {painting_key}: true\n  1:\n    max_threads: 7\n",
        encoding="utf-8",
    )
    evidence = _evidence(batch=2, threads=2, combined=True)
    evidence.profile.high_performance_mode = True
    suggestion = measured_recommendation([evidence])
    assert suggestion.safe_to_apply and not suggestion.gpu_overrides
    assert suggestion.high_performance_mode is True
    assert "gpu_overrides" not in suggestion.as_yaml_block()
    exported = suggestion.as_yaml_block()
    assert "Before merging" in exported and "Card 0: max_threads, queue_size" in exported
    assert "allow_inpainting, the alias for allow_painting" in exported
    assert "Keep unrelated fields and other cards" in exported
    assert YAML(typ="safe").load(exported)["max_threads"] == 2
    snapshot = suggested_config_overrides(suggestion)
    for _ in range(2):
        apply_known_good_to_config(snapshot, path)
        config = YAML(typ="safe").load(path)
        assert config["max_threads"] == 2 and config["high_performance_mode"] is True
        assert config["models_to_skip"] == ["excluded-global"]
        assert config["gpu_overrides"][card_key] == {"safety_on_gpu": False, "models_to_skip": ["excluded-card"]}
        assert config["gpu_overrides"][1] == {"max_threads": 7}


def test_feature_resolution_does_not_raise_plain_image_confirmation_shape() -> None:
    """A hires feature's larger output does not establish a larger plain-image sampling limit."""
    baseline = _evidence()
    hires = _evidence(kind=CapabilityKind.HIRES_FIX)
    hires.profile.width = hires.profile.height = 2048
    hires.profile.hires_fix = True
    case = combined_case([baseline, hires], device=0, tier=BenchTier.SD15)
    assert case is not None and case.profile.width == case.profile.height == 512


def test_queue_confirmation_offers_enough_jobs_to_fill_the_queue() -> None:
    """Combined settings must exercise queued work instead of validating the queue on one job."""
    baseline = _evidence()
    queued = _evidence(kind=CapabilityKind.QUEUE_SIZE)
    queued.profile.queue_size = 2
    case = combined_case([baseline, queued], device=0, tier=BenchTier.SD15)
    assert case is not None and case.profile.queue_size == 2
    assert case.work_count == len(case.scenario().expand_image_jobs()) == 3


def _tile_forecast(job: ImageGenerateJobPopResponse, baseline: str | None) -> float:
    """Price a job at one MB per 64x64 latent tile per image, so each test can size a card exactly."""
    return float((job.payload.width // 64) * (job.payload.height // 64) * job.payload.n_iter)


def _sdxl_climbs() -> list[WorkloadEvidence]:
    """Return SDXL evidence whose resolution climb reached 1984 and whose batch climb reached four, separately."""
    baseline = _evidence(tier=BenchTier.SDXL)
    baseline.profile.width = baseline.profile.height = 1024
    climbs = [baseline]
    for width in (1408, 1984):
        rung = _evidence(tier=BenchTier.SDXL, kind=CapabilityKind.RESOLUTION)
        rung.profile.width = rung.profile.height = width
        climbs.append(rung)
    for batch in (2, 4):
        rung = _evidence(tier=BenchTier.SDXL, kind=CapabilityKind.BATCH, batch=batch)
        rung.profile.width = rung.profile.height = 1024
        climbs.append(rung)
    return climbs


# Below 10240 MB the admission margin is its 512 MB floor on every platform, so these totals are exact.
_MARGIN_FLOOR_MB = 512


def test_joint_confirmation_without_a_card_total_keeps_the_independent_maxima(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no card total the joint takes the largest passed resolution and batch whatever they are forecast at."""
    monkeypatch.setattr(assessment_policy, "predict_job_sampling_vram_mb", _tile_forecast)
    case = combined_case(_sdxl_climbs(), device=0, tier=BenchTier.SDXL)
    assert case is not None
    assert (case.profile.width, case.profile.batch) == (1984, 4)
    assert case.notes == ()


def test_joint_confirmation_that_fits_the_card_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A joint whose forecast fits the usable card is confirmed as built, with nothing recorded."""
    monkeypatch.setattr(assessment_policy, "predict_job_sampling_vram_mb", _tile_forecast)
    case = combined_case(_sdxl_climbs(), device=0, tier=BenchTier.SDXL, card_total_mb=_MARGIN_FLOOR_MB + 3844)
    assert case is not None
    assert (case.profile.width, case.profile.batch) == (1984, 4)
    assert case.notes == ()


def test_joint_confirmation_over_the_card_gives_back_resolution_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """A joint forecast above the card steps the resolution down a passed rung before touching the batch."""
    monkeypatch.setattr(assessment_policy, "predict_job_sampling_vram_mb", _tile_forecast)
    # At batch four 1984x1984 forecasts 3844 MB and 1408x1408 forecasts 1936 MB.
    case = combined_case(_sdxl_climbs(), device=0, tier=BenchTier.SDXL, card_total_mb=_MARGIN_FLOOR_MB + 3000)
    assert case is not None
    assert (case.profile.width, case.profile.height, case.profile.batch) == (1408, 1408, 4)
    assert len(case.notes) == 1 and "1984x1984 to 1408x1408" in case.notes[0]


def test_joint_confirmation_too_large_at_the_smallest_resolution_steps_the_batch_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the resolution is at its smallest passed rung the batch steps down one passed rung at a time."""
    monkeypatch.setattr(assessment_policy, "predict_job_sampling_vram_mb", _tile_forecast)
    # 1024x1024 at batch four forecasts 1024 MB and at batch two 512 MB.
    case = combined_case(_sdxl_climbs(), device=0, tier=BenchTier.SDXL, card_total_mb=_MARGIN_FLOOR_MB + 700)
    assert case is not None
    assert (case.profile.width, case.profile.height, case.profile.batch) == (1024, 1024, 2)
    assert [note.split(":")[0] for note in case.notes] == [
        "Reduced the resolution 1984x1984 to 1408x1408",
        "Reduced the resolution 1408x1408 to 1024x1024",
        "Reduced the batch 4 to 2",
    ]


def test_joint_confirmation_that_never_fits_confirms_the_baseline_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """When even the baseline shape is forecast over the card the confirmation still runs it, and says so."""
    monkeypatch.setattr(assessment_policy, "predict_job_sampling_vram_mb", _tile_forecast)
    case = combined_case(_sdxl_climbs(), device=0, tier=BenchTier.SDXL, card_total_mb=_MARGIN_FLOOR_MB + 100)
    assert case is not None and case.combined
    assert (case.profile.width, case.profile.height, case.profile.batch) == (1024, 1024, 1)
    assert "baseline shape" in case.notes[-1]


def test_joint_confirmation_prices_every_lane_together(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lanes sample beside each other, so the joint is priced as the sum of each lane's job."""
    monkeypatch.setattr(assessment_policy, "predict_job_sampling_vram_mb", _tile_forecast)
    baseline = _evidence(tier=BenchTier.SDXL)
    baseline.profile.width = baseline.profile.height = 1024
    batch = _evidence(tier=BenchTier.SDXL, kind=CapabilityKind.BATCH, batch=2)
    batch.profile.width = batch.profile.height = 1024
    threads = _evidence(tier=BenchTier.SDXL, kind=CapabilityKind.THREADS, threads=2)
    threads.profile.width = threads.profile.height = 1024
    threads.profile.companion_model = "other-model"
    # One lane at batch two forecasts 512 MB and fits alone; two lanes forecast 1024 MB and do not.
    case = combined_case(
        [baseline, batch, threads], device=0, tier=BenchTier.SDXL, card_total_mb=_MARGIN_FLOOR_MB + 800
    )
    assert case is not None
    assert (case.profile.threads, case.profile.batch) == (2, 1)
    assert case.profile.companion_model == "other-model"


def test_worker_forecast_keeps_the_sdxl_batch_on_a_16_gb_card() -> None:
    """Priced with the worker's own forecast, the SDXL joint on a 16 GB card keeps batch four at 1024x1024."""
    card_total_mb = 16375.0
    case = combined_case(_sdxl_climbs(), device=0, tier=BenchTier.SDXL, card_total_mb=card_total_mb)
    assert case is not None and case.notes
    # Batch four forecasts above the usable room at 1408x1408 and inside it at 1024x1024.
    assert (case.profile.width, case.profile.height, case.profile.batch) == (1024, 1024, 4)
    usable_mb = card_total_mb - admission_margin_mb(card_total_mb)
    lane_jobs = AssessmentCase(case.capability, case.profile.model_copy(update={"work_jobs": 1})).scenario()
    forecast = assessment_policy.predict_job_sampling_vram_mb(lane_jobs.expand_image_jobs()[0], "stable_diffusion_xl")
    assert forecast is not None and forecast <= usable_mb


@pytest.mark.parametrize("per_card", [False, True])
def test_suggestion_leaves_the_operator_model_list_untouched(tmp_path: Path, per_card: bool) -> None:
    """The benchmark representatives are not offered as models, and Apply keeps every existing model list."""
    from horde_worker_regen.tui.benchmark_launcher import suggested_config_overrides

    suggestion = measured_recommendation([_evidence(batch=2, threads=2, combined=True)], per_card=per_card)
    assert suggestion.safe_to_apply
    assert not suggestion.models_to_load
    assert all(not settings.models_to_load for settings in suggestion.gpu_overrides.values())
    assert all("models_to_load" not in fields for fields in suggestion.clear_card_fields.values())
    assert all(decision.setting != "models_to_load" for decision in suggestion.decisions)
    assert "models_to_load" not in suggestion.as_yaml_block()
    assert "models_to_load" not in suggested_config_overrides(suggestion)
    path = tmp_path / "bridgeData.yaml"
    path.write_text(
        "models_to_load: [top 5, Deliberate]\ngpu_overrides:\n  0:\n    models_to_load: [card-model]\n"
        "    max_threads: 9\n",
        encoding="utf-8",
    )
    apply_suggested_to_config(suggestion, path)
    applied = YAML(typ="safe").load(path)
    assert applied["models_to_load"] == ["top 5", "Deliberate"]
    assert applied["gpu_overrides"][0]["models_to_load"] == ["card-model"]
    assert applied["gpu_overrides"][0].get("max_threads", applied.get("max_threads")) == 2


def test_model_guidance_reports_each_card_family_by_evidence() -> None:
    """Passed, failed, estimated and untested families are named per card with the machine's card details."""
    evidence = [
        _evidence(tier=BenchTier.SD15),
        _evidence(tier=BenchTier.SDXL, status=EvidenceStatus.PREDICTED, kind=CapabilityKind.BATCH),
        _evidence(tier=BenchTier.FLUX, status=EvidenceStatus.MEASURED_FAILURE),
        _evidence(tier=BenchTier.SDXL, device=1),
    ]
    machine = MachineInfo(gpu_name="Test GPU", total_vram_mb=16384)
    options = CatalogOptions(tiers=[BenchTier.SD15, BenchTier.SDXL, BenchTier.FLUX])
    guidance = model_guidance(evidence, machine, options, installed_models=1)
    assert any("do not change models_to_load" in line for line in guidance)
    card0 = next(line for line in guidance if line.startswith("Card 0"))
    assert card0.startswith("Card 0 (Test GPU, 16384 MB VRAM): Passed a measured baseline: SD15.")
    assert "Failed the measured baseline: FLUX." in card0
    assert "Estimated only: SDXL." in card0
    assert "Not tested: QWEN, ZIMAGE." in card0
    card1 = next(line for line in guidance if line.startswith("Card 1"))
    assert "Passed a measured baseline: SDXL." in card1 and "Not tested: SD15, FLUX, QWEN, ZIMAGE." in card1
    assert any("less commonly offered models" in line for line in guidance)


def test_model_guidance_large_families_come_from_the_ladder() -> None:
    """A passing large family is supported, a selected one that failed is named, an unselected one can be rerun."""
    large = [tier for tier in BenchTier if tier in HUGE_TIERS]
    assert len(large) >= 3
    passed, failed, *unselected = large
    evidence = [_evidence(tier=passed), _evidence(tier=failed, status=EvidenceStatus.MEASURED_FAILURE)]
    guidance = model_guidance(
        evidence, MachineInfo(), CatalogOptions(tiers=[BenchTier.SD15, passed, failed]), installed_models=2
    )
    assert any(line.startswith("Card 0: ") for line in guidance)
    assert any(f"({', '.join(tier.upper() for tier in large)})" in line for line in guidance)
    assert f"{passed.upper()} passed a measured baseline on card 0 and is supported." in guidance
    assert f"{failed.upper()} was selected but did not pass a measured baseline on any card." in guidance
    rerun = next(line for line in guidance if "not selected for this run" in line)
    assert all(tier.upper() in rerun for tier in unselected) and "rerun" in rerun

    only_one = model_guidance([], MachineInfo(), CatalogOptions(tiers=[*large[:-1]]), installed_models=1)
    assert (
        f"{large[-1].upper()} was not selected for this run. Select it for a rerun if the card has the VRAM for it."
        in (only_one)
    )
    assert "No card completed a check, so no model family was measured." in only_one


def test_model_guidance_states_the_thread_limit_from_the_installed_model_count() -> None:
    """The concurrency limit line carries the installed-model count and is absent when concurrency is excluded."""
    evidence = [_evidence()]
    two = model_guidance(evidence, MachineInfo(), CatalogOptions(), installed_models=2)
    limit = next(line for line in two if line.startswith("Concurrency checks test at most"))
    assert limit == (
        "Concurrency checks test at most 2 sampling threads because each lane needs its own installed model. "
        "Installing a checkpoint from another family raises that limit."
    )
    one = model_guidance(evidence, MachineInfo(), CatalogOptions(), installed_models=1)
    assert any(line.startswith("Concurrency checks test at most 1 sampling thread because") for line in one)
    excluded = model_guidance(evidence, MachineInfo(), CatalogOptions(include_concurrency=False), installed_models=2)
    assert not any(line.startswith("Concurrency checks") for line in excluded)


def test_report_renders_model_guidance_before_the_suggestion() -> None:
    """The operator assessment report carries the guidance as its own section ahead of the bridgeData."""
    evidence = [_evidence(combined=True, kind=CapabilityKind.SUSTAINED), _evidence()]
    report = CapabilityReport(
        operator_assessment=True,
        evidence=evidence,
        suggested_bridge_data=measured_recommendation(evidence),
        model_guidance=model_guidance(evidence, MachineInfo(), CatalogOptions(), installed_models=1),
    )
    rendered = render_markdown(report)
    assert "## Model guidance" in rendered and "## Suggested bridgeData" in rendered
    assert rendered.index("## Model guidance") < rendered.index("## Suggested bridgeData")
    assert f"- {report.model_guidance[0]}" in rendered
