"""Select capability checks within a work budget and export settings confirmed together."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from math import isqrt
from pathlib import Path

from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_model_reference.model_reference_records import ImageGenerationModelRecord
from ruamel.yaml import YAML

from horde_worker_regen.benchmark.capabilities.capability import Capability, CapabilityKind
from horde_worker_regen.benchmark.capabilities.catalog import CatalogOptions, build_capability_catalog
from horde_worker_regen.benchmark.capabilities.result import (
    EvidenceStatus,
    MachineInfo,
    OperatingProfile,
    SuggestedBridgeData,
    SuggestedCardSettings,
    SuggestionBasis,
    SuggestionDecision,
    WorkloadEvidence,
)
from horde_worker_regen.benchmark.enums import BenchTier
from horde_worker_regen.benchmark.ladder import _TIER_BASELINES, _TIER_RESOLUTIONS, BENCH_TIER_MODELS, HUGE_TIERS
from horde_worker_regen.benchmark.scenarios import CannedAlchemyFormSpec, CannedImageJobSpec, Scenario
from horde_worker_regen.model_download_plan import is_model_present
from horde_worker_regen.process_management.resources.admission_identity import admission_margin_mb
from horde_worker_regen.process_management.resources.resource_budget import predict_job_sampling_vram_mb


def _eligible_model_records(
    records: Mapping[str, ImageGenerationModelRecord],
    config: Mapping[str, object],
    manager: ModelReferenceManager,
) -> dict[str, ImageGenerationModelRecord]:
    """Retain models allowed by global exclusions and every selected card's exclusions."""
    from horde_worker_regen.bridge_data.data_model import reGenBridgeData
    from horde_worker_regen.bridge_data.load_config import BridgeDataLoader

    policies = [config]
    overrides = config.get("gpu_overrides")
    selected = config.get("gpu_device_indices")
    selected_indices = {str(index) for index in selected} if isinstance(selected, list) else None
    if isinstance(overrides, dict):
        policies.extend(
            policy
            for index, policy in overrides.items()
            if isinstance(policy, dict) and (selected_indices is None or str(index) in selected_indices)
        )
    eligible = dict(records)
    for policy in policies:
        exclusions = policy.get("models_to_skip")
        if not exclusions:
            continue
        # The worker resolver also expands popularity and family rules. Matching only literal names
        # here would benchmark a checkpoint that ordinary startup subsequently removes.
        bridge = reGenBridgeData.model_validate(
            {
                "models_to_load": list(eligible),
                "models_to_skip": exclusions,
                "load_large_models": config.get("load_large_models", False),
                "only_models_on_disk": False,
            }
        )
        allowed = set(BridgeDataLoader._resolve_meta_instructions(bridge, manager))
        eligible = {name: record for name, record in eligible.items() if name in allowed}
    return eligible


def select_installed_models(
    tiers: list[BenchTier],
    records: Mapping[str, ImageGenerationModelRecord],
    *,
    cache_home: str | None = None,
    extra_model_directories: list[str] | None = None,
) -> dict[BenchTier, str]:
    """Prefer the canonical checkpoint when present, then an installed checkpoint of the same family."""
    selected: dict[BenchTier, str] = {}
    for tier in tiers:
        candidates = [BENCH_TIER_MODELS[tier], *sorted(records)]
        for name in dict.fromkeys(candidates):
            record = records.get(name)
            if record is None or str(record.baseline) != _TIER_BASELINES[tier]:
                continue
            if is_model_present(name, records, cache_home=cache_home, extra_model_directories=extra_model_directories):
                selected[tier] = name
                break
    return selected


def installed_models_for_tiers(tiers: list[BenchTier]) -> dict[BenchTier, str]:
    """Resolve installed family representatives from offline reference metadata."""
    from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY

    from horde_worker_regen.reference_helper import ensure_offline_reference_manager

    manager = ensure_offline_reference_manager()
    available = manager.get_all_model_references()
    records = {
        name: record
        for name, record in (available.get(MODEL_REFERENCE_CATEGORY.image_generation) or {}).items()
        if isinstance(record, ImageGenerationModelRecord)
    }
    source = Path("bridgeData.yaml")
    raw = YAML(typ="safe").load(source) if source.exists() else {}
    config = raw if isinstance(raw, dict) else {}
    records = _eligible_model_records(records, config, manager)
    cache_home = (
        os.environ.get("AIWORKER_CACHE_HOME") or config.get("cache_home") or config.get("models_folder_parent")
    )
    if not isinstance(cache_home, str):
        data_root = os.environ.get("HORDE_WORKER_DATA_DIR")
        cache_home = str(Path(data_root) / "models") if data_root else "models"
    extra_roots = config.get("extra_model_directories")
    directories = (
        [str(Path(root).resolve()) for root in extra_roots if isinstance(root, str)]
        if isinstance(extra_roots, list)
        else None
    )
    if os.environ.get("AIWORKER_EXTRA_MODEL_DIRECTORIES") is not None:
        directories = None  # The presence helper reads the environment at higher precedence.
    return select_installed_models(
        tiers, records, cache_home=str(Path(cache_home).resolve()), extra_model_directories=directories
    )


@dataclass(frozen=True)
class AssessmentCase:
    """A test workload and the settings used to run it."""

    capability: Capability
    profile: OperatingProfile
    combined: bool = False
    notes: tuple[str, ...] = ()
    """What the policy changed about the workload while building it, for the report."""

    @property
    def case_id(self) -> str:
        """Return the stable check identifier, including the card index."""
        suffix = "-combined" if self.combined else ""
        return f"card{self.profile.device_index}-{self.capability.slug}{suffix}"

    def scenario(self) -> Scenario:
        """Create fresh generation ids whenever this check is executed."""
        profile = self.profile

        def spec(model: str, count: int) -> CannedImageJobSpec:
            return CannedImageJobSpec(
                model=model,
                width=profile.width,
                height=profile.height,
                n_iter=profile.batch,
                count=count,
                steps=9 if profile.tier is BenchTier.ZIMAGE else profile.steps,
                cfg_scale=1.0 if profile.tier is BenchTier.ZIMAGE else None,
                control_type=profile.control_type,
                workflow=profile.workflow,
                hires_fix=profile.hires_fix,
                post_processing=profile.post_processing,
                lora_names=profile.lora_names,
            )

        if profile.companion_model is None:
            image_jobs = [spec(profile.model_name, self.image_work_count)]
        else:
            # Alternating models makes work available to both lanes from the first pop.
            models = (profile.model_name, profile.companion_model)
            image_jobs = [spec(models[index % 2], 1) for index in range(self.image_work_count)]
        return Scenario(
            name=self.case_id,
            image_jobs=image_jobs,
            alchemy_forms=[CannedAlchemyFormSpec(form=name) for name in profile.alchemy_forms],
        )

    @property
    def image_work_count(self) -> int:
        """Return the offered image-job count, separate from batch generations and service forms."""
        return self.profile.work_jobs or (
            self.profile.threads
            + (self.profile.queue_size if self.capability.kind is CapabilityKind.QUEUE_SIZE else 0)
        )

    @property
    def work_count(self) -> int:
        """Return the number of image jobs and forms charged to this check."""
        return self.image_work_count + len(self.profile.alchemy_forms)

    @property
    def check_work_count(self) -> int:
        """Return the jobs the whole check offers, the budget's charge for it.

        A concurrency check offers its jobs once at one lane and again at its thread count, since the
        single-lane run is the wall-clock reference the thread count is ranked against. A combined
        confirmation runs only the thread count, which its thread check already ranked.
        """
        return self.work_count * 2 if self.profile.threads > 1 and not self.combined else self.work_count


def representative_cases(
    options: CatalogOptions, models: dict[BenchTier, str], devices: list[int], *, only_probe: str | None = None
) -> list[AssessmentCase]:
    """Keep every selected category visible while choosing one representative per card and family."""
    catalog = build_capability_catalog(options)
    cases: list[AssessmentCase] = []
    seen: set[tuple[int, BenchTier, CapabilityKind]] = set()
    for probe in catalog:
        tier = probe.capability.tier
        if tier not in models or (
            probe.capability.kind in (CapabilityKind.BATCH, CapabilityKind.THREADS)
            and probe.capability.slug != only_probe
        ):
            continue
        image = probe.scenario.image_jobs[0] if probe.scenario.image_jobs else None
        forms = probe.scenario.alchemy_forms[:1]
        for device in devices:
            key = (device, tier, probe.capability.kind)
            if key in seen:
                continue
            seen.add(key)
            cases.append(
                AssessmentCase(
                    capability=probe.capability,
                    profile=OperatingProfile(
                        device_index=device,
                        tier=tier,
                        model_name=models[tier],
                        batch=image.n_iter if image else 1,
                        threads=probe.capability.magnitude if probe.capability.kind is CapabilityKind.THREADS else 1,
                        queue_size=2 if probe.capability.kind is CapabilityKind.QUEUE_SIZE else 1,
                        width=image.width if image else _TIER_RESOLUTIONS[tier],
                        height=image.height if image else _TIER_RESOLUTIONS[tier],
                        control_type=image.control_type if image else None,
                        workflow=image.workflow if image else None,
                        hires_fix=image.hires_fix if image else False,
                        post_processing=image.post_processing if image else [],
                        lora_names=image.lora_names if image else [],
                        alchemy_forms=[form.form for form in forms],
                    ),
                )
            )
    # Run the least expensive family baselines on all cards before spending the budget on feature checks.
    return sorted(cases, key=lambda item: item.capability.kind is not CapabilityKind.BASELINE)


def next_boundary(result: WorkloadEvidence, *, include_concurrency: bool) -> AssessmentCase | None:
    """Choose one larger boundary only after the preceding workload completed successfully."""
    if result.status is not EvidenceStatus.MEASURED_SUCCESS:
        return None
    profile = result.profile
    kind = result.capability.kind
    if kind is not CapabilityKind.RESOLUTION and not include_concurrency:
        return None
    if kind is CapabilityKind.BASELINE:
        return AssessmentCase(
            Capability(tier=profile.tier, kind=CapabilityKind.BATCH, magnitude=2),
            profile.model_copy(update={"batch": 2}),
        )
    if kind is CapabilityKind.BATCH and profile.batch < 16:
        batch = profile.batch * 2
        return AssessmentCase(
            Capability(tier=profile.tier, kind=kind, magnitude=batch),
            profile.model_copy(update={"batch": batch}),
        )
    if kind is CapabilityKind.THREADS and profile.threads < 16:
        threads = profile.threads * 2
        return AssessmentCase(
            Capability(tier=profile.tier, kind=kind, magnitude=threads),
            profile.model_copy(update={"threads": threads}),
        )
    if kind is CapabilityKind.RESOLUTION and profile.width < 4096:
        width = min(4096, (isqrt(profile.width * profile.width * 2) // 64) * 64)
        return AssessmentCase(
            Capability(tier=profile.tier, kind=kind, magnitude=width),
            profile.model_copy(update={"width": width, "height": width}),
        )
    return None


def assessment_priority(case: AssessmentCase) -> int:
    """Run family baselines and initial capacity checks before features and further capacity searches."""
    kind = case.capability.kind
    if kind is CapabilityKind.BASELINE:
        return 0
    if kind in (CapabilityKind.BATCH, CapabilityKind.THREADS):
        return 1 if case.capability.magnitude == 2 else 3
    if kind is CapabilityKind.RESOLUTION:
        first_boundary = (isqrt(_TIER_RESOLUTIONS[case.profile.tier] ** 2 * 2) // 64) * 64
        return 1 if case.profile.width <= first_boundary else 3
    return 2


def combined_case(
    evidence: list[WorkloadEvidence], *, device: int, tier: BenchTier, card_total_mb: float | None = None
) -> AssessmentCase | None:
    """Build a combined workload from successful checks, ready for an independent confirmation.

    Args:
        evidence: Every check run so far.
        device: The card the confirmation runs on.
        tier: The model family being confirmed.
        card_total_mb: The card's total VRAM. When given, a joint the worker's own forecast cannot admit
            is stepped down through passed rungs until it fits, and each reduction is recorded in
            ``AssessmentCase.notes``. None leaves the joint at the largest passed limits.
    """
    passed = [
        item
        for item in evidence
        if item.profile.device_index == device
        and item.profile.tier is tier
        and item.status is EvidenceStatus.MEASURED_SUCCESS
        and not item.combined_validation
    ]
    baselines = [item for item in passed if item.capability.kind is CapabilityKind.BASELINE]
    if not baselines:
        return None
    anchor = baselines[0].profile
    limits = [
        item
        for item in passed
        if item.capability.kind
        in {
            CapabilityKind.BASELINE,
            CapabilityKind.BATCH,
            CapabilityKind.THREADS,
            CapabilityKind.RESOLUTION,
            CapabilityKind.QUEUE_SIZE,
        }
    ]
    resolution = max(limits, key=lambda item: item.profile.width * item.profile.height).profile
    widest = max(limits, key=lambda item: item.profile.threads).profile
    threads = widest.threads
    queue_size = max(item.profile.queue_size for item in limits)
    case = AssessmentCase(
        Capability(tier=tier, kind=CapabilityKind.SUSTAINED),
        anchor.model_copy(
            update={
                "width": resolution.width,
                "height": resolution.height,
                "batch": max(item.profile.batch for item in limits),
                "threads": threads,
                "companion_model": widest.companion_model if threads > 1 else None,
                "lease_slots": widest.lease_slots if threads > 1 else None,
                "queue_size": queue_size,
                "work_jobs": max(threads * 2 if threads > 1 else 1, threads + queue_size if queue_size > 1 else 1),
            }
        ),
        combined=True,
    )
    if card_total_mb is None:
        return case
    baselines = {item.profile.model_name: _TIER_BASELINES[item.profile.tier] for item in evidence}
    return _fit_to_card(case, [item.profile for item in limits], baselines, card_total_mb)


def _joint_sampling_forecast_mb(profile: OperatingProfile, baselines: Mapping[str, str]) -> float | None:
    """Return the worker's sampling forecast for one job per lane, summed, or None when any lane is unpriced.

    The lanes of a confirmation sample beside each other, so the card holds every lane's job at once.
    """
    lanes = AssessmentCase(
        Capability(tier=profile.tier, kind=CapabilityKind.SUSTAINED),
        profile.model_copy(update={"work_jobs": profile.threads}),
    )
    total_mb = 0.0
    for job in lanes.scenario().expand_image_jobs():
        forecast_mb = predict_job_sampling_vram_mb(job, baselines.get(job.model or "", _TIER_BASELINES[profile.tier]))
        if forecast_mb is None:
            return None
        total_mb += forecast_mb
    return total_mb


def _fit_to_card(
    case: AssessmentCase,
    passed: list[OperatingProfile],
    baselines: Mapping[str, str],
    card_total_mb: float,
) -> AssessmentCase:
    """Step a joint workload down through passed rungs until the worker's own forecast fits the card.

    Each limit passed alone, but the joint of the largest resolution and the largest batch can exceed what
    the worker admits. The worker then holds the job for the whole lease-acquire timeout and samples it
    unpriced, so the confirmation measures that wait rather than a workload the worker serves. The resolution
    steps down one passed rung at a time first, because batch jobs are a large share of production pops while
    jobs near the largest resolutions are rare. The batch steps down only once the resolution is at its
    smallest passed rung. A forecast that is not available leaves the workload as built.
    """
    usable_mb = card_total_mb - admission_margin_mb(card_total_mb)
    batches = sorted({profile.batch for profile in passed})
    sizes = sorted({(profile.width, profile.height) for profile in passed}, key=lambda size: size[0] * size[1])
    profile = case.profile
    notes: list[str] = []
    while True:
        forecast_mb = _joint_sampling_forecast_mb(profile, baselines)
        if forecast_mb is None or forecast_mb <= usable_mb:
            break
        reason = (
            f"the worker forecasts {forecast_mb:.0f} MB against {usable_mb:.0f} MB usable "
            f"on a {card_total_mb:.0f} MB card"
        )
        smaller_sizes = [size for size in sizes if size[0] * size[1] < profile.width * profile.height]
        if smaller_sizes:
            width, height = smaller_sizes[-1]
            notes.append(f"Reduced the resolution {profile.width}x{profile.height} to {width}x{height}: {reason}.")
            profile = profile.model_copy(update={"width": width, "height": height})
            continue
        smaller_batches = [batch for batch in batches if batch < profile.batch]
        if smaller_batches:
            notes.append(f"Reduced the batch {profile.batch} to {smaller_batches[-1]}: {reason}.")
            profile = profile.model_copy(update={"batch": smaller_batches[-1]})
            continue
        # The operator still needs a confirmation of what passed, so the smallest passed shape runs anyway.
        notes.append(f"Confirming the baseline shape although {reason}.")
        break
    if not notes:
        return case
    return AssessmentCase(case.capability, profile, combined=case.combined, notes=(*case.notes, *notes))


def measured_recommendation(evidence: list[WorkloadEvidence], *, per_card: bool = False) -> SuggestedBridgeData:
    """Export per-card settings only where a joint workload confirmed the offered limits."""
    suggestion = SuggestedBridgeData(safe_to_apply=False)
    devices = sorted({item.profile.device_index for item in evidence})
    lease_slot_counts: list[int] = []
    for device in devices:
        confirmed = [
            item.profile
            for item in evidence
            if item.profile.device_index == device
            and item.combined_validation
            and item.status is EvidenceStatus.MEASURED_SUCCESS
            and item.retry_provenance_complete
            and not (
                item.recoveries
                or item.inference_failures
                or item.resource_failures
                or item.degraded_retries
                or item.host_commit_refusals
            )
        ]
        if not confirmed:
            continue
        # Configuration cannot express separate resolution/batch/concurrency limits per model.
        # Group exact operating limits rather than manufacturing a combination of independent minima.
        groups: dict[tuple[int, int, int, int, int, bool, bool, int | None], list[OperatingProfile]] = {}
        for profile in confirmed:
            if profile.control_type or profile.workflow or profile.post_processing or profile.alchemy_forms:
                continue
            key = (
                profile.width,
                profile.height,
                profile.batch,
                profile.threads,
                profile.queue_size,
                profile.high_performance_mode,
                profile.moderate_performance_mode,
                profile.lease_slots,
            )
            groups.setdefault(key, []).append(profile)
        if not groups:
            continue
        limits = max(groups.items(), key=lambda pair: (len(pair[1]), pair[0][0] * pair[0][1]))[0]
        width, height, batch, threads, queue_size, high_performance, moderate_performance, lease_slots = limits
        if lease_slots is not None:
            lease_slot_counts.append(lease_slots)
        settings = SuggestedCardSettings(
            max_threads=threads,
            queue_size=queue_size,
            max_batch=batch,
            max_power=max(1, width * height // (8 * 64 * 64)),
            high_performance_mode=high_performance,
            moderate_performance_mode=moderate_performance,
        )
        suggestion.gpu_overrides[device] = settings
    settings_list = list(suggestion.gpu_overrides.values())
    if settings_list:
        suggestion.safe_to_apply = True
        suggestion.allow_img2img = False
        suggestion.allow_painting = False
        suggestion.high_performance_mode = all(item.high_performance_mode for item in settings_list)
        suggestion.moderate_performance_mode = all(item.moderate_performance_mode for item in settings_list)
        suggestion.dynamic_models = False
        suggestion.gpu_device_indices = sorted(suggestion.gpu_overrides)
        suggestion.max_threads = min(item.max_threads for item in settings_list)
        suggestion.queue_size = min(item.queue_size for item in settings_list)
        suggestion.max_batch = min(item.max_batch for item in settings_list)
        suggestion.max_power = min(item.max_power for item in settings_list)
        if lease_slot_counts:
            # The lease settings are global, so the card with the fewest confirmed slots bounds every card.
            suggestion.gpu_sampling_lease_enabled = True
            suggestion.gpu_sampling_lease_slots = min(lease_slot_counts)
        if len(devices) == 1 and not per_card:
            device = devices[0]
            # The tested checkpoints represent model families. Preserve the operator's per-card model
            # list when removing overrides that conflict with the suggested global limits.
            suggestion.clear_card_fields = {
                device: [name for name in SuggestedCardSettings.model_fields if name != "models_to_load"]
            }
            suggestion.gpu_overrides.clear()
            suggestion.gpu_device_indices = None
    suggestion.notes = [
        "Only jointly confirmed card/model profiles are exported. "
        "Other measured and predicted limits remain in the report.",
        "Representative processor checks do not establish every processor or chain. "
        "Sustained reliability was not measured.",
    ]
    basis = SuggestionBasis.PROVEN if settings_list else SuggestionBasis.UNTESTED_SKIPPED
    for setting, value in (
        ("max_threads", suggestion.max_threads),
        ("queue_size", suggestion.queue_size),
        ("max_batch", suggestion.max_batch),
    ):
        suggestion.decisions.append(
            SuggestionDecision(
                setting=setting,
                value=value,
                basis=basis,
                detail="jointly confirmed profiles only"
                if settings_list
                else "no combined configuration was confirmed",
            )
        )
    if suggestion.gpu_sampling_lease_slots is not None:
        suggestion.decisions.append(
            SuggestionDecision(
                setting="gpu_sampling_lease_slots",
                value=suggestion.gpu_sampling_lease_slots,
                basis=SuggestionBasis.PROVEN,
                detail="the thread count was confirmed with the sampling lease enabled",
            )
        )
    return suggestion


def _family_names(tiers: list[BenchTier]) -> str:
    """Join family labels using the same spelling as the evidence table (``Capability.label``)."""
    return ", ".join(tier.upper() for tier in tiers)


def _card_description(device: int, machine: MachineInfo) -> str:
    """Format a card label with the GPU name and total VRAM when available."""
    details = [
        detail
        for detail in (
            machine.gpu_name,
            f"{machine.total_vram_mb} MB VRAM" if machine.total_vram_mb is not None else None,
        )
        if detail
    ]
    return f"Card {device} ({', '.join(details)})" if details else f"Card {device}"


def model_guidance(
    evidence: list[WorkloadEvidence], machine: MachineInfo, options: CatalogOptions, *, installed_models: int
) -> list[str]:
    """Describe which model families each card ran and which models add the most to the network.

    The suggested bridgeData leaves ``models_to_load`` unchanged because the tested checkpoints represent
    model families. The offline reference does not include worker counts, so the popularity advice
    does not give numerical estimates. A concurrency check gives each lane its own installed model, so the thread limit
    it can test is the installed-model count.
    """
    lines = [
        "The suggested settings do not change models_to_load. "
        "The checked checkpoints represent their families, so choose the models to offer yourself.",
    ]
    families = list(BenchTier)
    devices = sorted({item.profile.device_index for item in evidence})
    passed_cards: dict[BenchTier, list[int]] = {}
    if not devices:
        lines.append("No card completed a check, so no model family was measured.")
    for device in devices:
        card_evidence = [item for item in evidence if item.profile.device_index == device]
        passed = [
            tier
            for tier in families
            if any(
                item.profile.tier is tier
                and item.capability.kind is CapabilityKind.BASELINE
                and item.status is EvidenceStatus.MEASURED_SUCCESS
                for item in card_evidence
            )
        ]
        failed = [
            tier
            for tier in families
            if tier not in passed
            and any(
                item.profile.tier is tier
                and item.capability.kind is CapabilityKind.BASELINE
                and item.status is EvidenceStatus.MEASURED_FAILURE
                for item in card_evidence
            )
        ]
        estimated = [
            tier
            for tier in families
            if tier not in passed
            and tier not in failed
            and any(item.profile.tier is tier and item.status is EvidenceStatus.PREDICTED for item in card_evidence)
        ]
        untested = [tier for tier in families if tier not in (*passed, *failed, *estimated)]
        for tier in passed:
            passed_cards.setdefault(tier, []).append(device)
        parts = [
            f"Passed a measured baseline: {_family_names(passed)}."
            if passed
            else "No family passed a measured baseline.",
        ]
        if failed:
            parts.append(f"Failed the measured baseline: {_family_names(failed)}.")
        if estimated:
            parts.append(f"Estimated only: {_family_names(estimated)}.")
        if untested:
            parts.append(f"Not tested: {_family_names(untested)}.")
        lines.append(f"{_card_description(device, machine)}: {' '.join(parts)}")
    lines.append(
        "Consider model availability when choosing what to serve. "
        "Serving less commonly offered models from a family that passed here helps cover gaps in availability."
    )
    # HUGE_TIERS is the ladder's encoding of the families whose checkpoints need the most VRAM.
    large_families = [tier for tier in families if tier in HUGE_TIERS]
    lines.append(
        f"The large families ({_family_names(large_families)}) can earn more kudos per job and require the most VRAM."
    )
    not_selected = [tier for tier in large_families if tier not in options.tiers]
    for tier in large_families:
        if tier in passed_cards:
            cards = passed_cards[tier]
            card_text = f"card {cards[0]}" if len(cards) == 1 else f"cards {', '.join(str(card) for card in cards)}"
            lines.append(f"{tier.upper()} passed a measured baseline on {card_text} and is supported.")
        elif tier in options.tiers:
            lines.append(f"{tier.upper()} was selected but did not pass a measured baseline on any card.")
    if len(not_selected) == 1:
        lines.append(
            f"{not_selected[0].upper()} was not selected for this run. "
            "Select it for a rerun if the card has the VRAM for it."
        )
    elif not_selected:
        lines.append(
            f"{_family_names(not_selected)} were not selected for this run. "
            "Select them for a rerun if the card has the VRAM for them."
        )
    if options.include_concurrency and CapabilityKind.THREADS not in options.excluded_kinds:
        thread_noun = "sampling thread" if installed_models == 1 else "sampling threads"
        lines.append(
            f"Concurrency checks test at most {installed_models} {thread_noun} because each lane needs its own "
            "installed model. Installing a checkpoint from another family raises that limit."
        )
    return lines
