"""Per-level resource requirements and the machine-fit verdict, computed read-only from a level.

A benchmark level declares *what* work it offers (see :mod:`horde_worker_regen.benchmark.scenarios`).
This module derives *what that work needs* (VRAM, disk, downloads, a CivitAI token) without ever
touching the scenario's content, so the same numbers drive three surfaces that must agree:

- the ``horde-benchmark plan`` subcommand (a dry preview, no worker boot),
- the controller's runtime pre-flight (``_pre_flight_skip_reason``), and
- the TUI's plan pane.

Keeping the computation here (and out of the controller) is what guarantees the preview an operator
sees matches the skip decision the ramp actually makes. The benchmark stays apples-to-apples across
machines: requirements only decide *whether* a level runs, never *what* it runs.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, Field

from horde_worker_regen.benchmark.ladder import (
    BETA_TIERS,
    HUGE_TIERS,
    RampLevel,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from horde_model_reference.model_reference_records import GenericModelRecord

    from horde_worker_regen.benchmark.capabilities.probe import CapabilityProbe
    from horde_worker_regen.benchmark.enums import BenchTier
    from horde_worker_regen.benchmark.report import MachineInfo
    from horde_worker_regen.benchmark.scenarios import Scenario
    from horde_worker_regen.model_download_plan import DownloadPlan

_CIVITAI_TOKEN_ENV_VARS = ("CIVIT_API_TOKEN", "AIWORKER_CIVITAI_API_TOKEN")
"""Env vars that carry a CivitAI token. ``load_env_vars`` populates ``CIVIT_API_TOKEN`` from the
config's ``civitai_api_token``; the Docker images use ``AIWORKER_CIVITAI_API_TOKEN``."""


class MissingModel(BaseModel):
    """A model a level needs that is not yet on disk, with the facts an operator needs to act on it."""

    name: str
    size_bytes: int | None = None
    """Declared download size, or None when the record carries no size metadata."""
    target_path: str = ""
    """Where the model's primary file will be written when downloaded; empty when undeterminable."""


class LevelRequirements(BaseModel):
    """What one benchmark level needs to run, derived read-only from its scenario."""

    level_id: str
    stage: str
    tier: str
    axis: str
    baseline: str
    estimated_vram_mb: int | None = None
    """Estimated peak VRAM for the level's heaviest job (hordelib burden), or None when unavailable."""
    min_disk_free_gb: float = 0.0
    estimated_download_bytes: int | None = None
    """Informational: the tier checkpoint's on-disk size for huge tiers (what a fresh fetch would cost)."""
    models_required: list[str] = Field(default_factory=list)
    models_missing: list[str] = Field(default_factory=list)
    """The subset of ``models_required`` confirmed absent on disk (indeterminate ones are omitted)."""
    missing_models: list[MissingModel] = Field(default_factory=list)
    """The same absent models as :attr:`models_missing`, enriched with size and download target path.

    Empty when the on-disk picture could not be sized (e.g. the cheap ``present_resolver`` path), even if
    ``models_missing`` is non-empty; consumers should treat sizes as unknown then.
    """
    download_bytes_needed: int = 0
    """Total declared bytes the absent models would download (0 when sizes are unknown or nothing is missing)."""
    present_bytes: int = 0
    """Declared bytes already on disk for this level's models (0 when the picture could not be sized)."""
    free_disk_bytes: int | None = None
    """Free space on the model volume when known, so a surface can show ``free`` without re-probing."""
    requires_network: bool = False
    requires_civitai_key: bool = False
    """True when a job pulls loras/TIs, which are fetched from CivitAI and may need a token."""
    requires_controlnet: bool = False
    """True when a job exercises a classic controlnet preprocessor (has a ``control_type``)."""
    controlnet_installed: bool | None = None
    """Whether the controlnet extra (onnxruntime annotators) is installed, or None when undeterminable."""
    controlnet_annotators_present: bool | None = None
    """Whether the annotator checkpoints are already downloaded on disk, or None when undeterminable.

    Distinct from :attr:`controlnet_installed` (the onnxruntime *package*): the annotator *weights* are
    fetched lazily on first use, so a worker with the extra installed can still be missing them. Drives
    whether a surface should prompt to pre-download them (only when this is ``False``)."""
    controlnet_annotator_bytes: int = 0
    """ROM annotator-checkpoint download size for the level's control types (0 when not a controlnet level).

    These annotator weights are fetched lazily on first use and are *not* part of ``download_bytes_needed``
    (which counts only image checkpoints); a surface that wants the full "to fetch" figure must add them.
    """
    controlnet_install_hint: str = ""
    """How to enable controlnet when the level needs it but the extra is absent (empty otherwise)."""
    controlnet_checkpoints_missing: list[str] = Field(default_factory=list)
    """The level's SD1.5 ``control_<type>`` checkpoint records confirmed absent on disk (empty when none, or
    when the presence-only path skipped the reference read). These are feature models the image-model disk
    plan does not cover, so a level that fits the machine but lacks them is "download first", not "ready"."""
    features: list[str] = Field(default_factory=list)
    """Human-readable feature tags exercised by the level (hires_fix, controlnet, post_processing, ...)."""


def civitai_token_available() -> bool:
    """Whether a CivitAI token is configured in this process's environment (best-effort)."""
    return any(os.getenv(var) for var in _CIVITAI_TOKEN_ENV_VARS)


def model_present_on_disk(model_name: str) -> bool | None:
    """Whether *model_name*'s files are all on disk, or None when it cannot be determined.

    Reads the reference offline (disk-only, never downloaded) via the existing
    :func:`is_model_present` existence check. This runs inside short-lived benchmark subprocesses, which
    must never fetch references over the network: only the parent/TUI does that, writing the converted
    JSON to disk that this then reads. An online manager here would block the subprocess on a PRIMARY-API
    round-trip whose latency it cannot bound, so the read is forced offline. Fails open (returns None) on
    any error.
    """
    try:
        from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY

        from horde_worker_regen.model_download_plan import is_model_present
        from horde_worker_regen.reference_helper import ensure_offline_reference_manager

        manager = ensure_offline_reference_manager()
        records = manager.get_all_model_references().get(MODEL_REFERENCE_CATEGORY.image_generation) or {}
        return is_model_present(model_name, records)
    except Exception as e:  # noqa: BLE001 - presence is best-effort; fail open
        logger.debug(f"Could not determine on-disk presence of {model_name!r}: {e}")
        return None


_CONTROLNET_RECORD_PREFIX = "control_"
"""SD1.5 controlnet records are named ``control_<type>`` (canny -> ``control_canny``); the ``_sd2``/``_xl``
suffixed siblings are other baselines the benchmark's SD1.5 controlnet sweep does not exercise."""

_POST_PROCESSOR_CATEGORIES = ("esrgan", "gfpgan", "codeformer")
"""The model-reference categories a post-processor's weights can live in (rembg/strip_background has none)."""


class FeatureModelFile(BaseModel):
    """One model a feature needs, with where it lives and whether it is on disk (tri-state).

    Unlike :class:`MissingModel`, a feature plan must also surface files that ARE present (so the operator
    sees the whole picture) and an undeterminable state (``on_disk=None`` when the reference or record is
    unavailable, never a confident-but-wrong claim). ``category`` is the model-reference category (e.g.
    ``controlnet``, ``esrgan``); for a real model it doubles as the model-manager attribute the
    self-download fetches it through.
    """

    name: str
    category: str
    size_bytes: int | None = None
    on_disk: bool | None = None
    target_path: str = ""


def _coerce_root_paths(extra_model_directories: Sequence[str | os.PathLike[str]] | None) -> list[Path]:
    """Normalise extra model directories to a list of Paths (empty when none)."""
    return [Path(directory) for directory in (extra_model_directories or ())]


def _offline_category_reference(category: str) -> Mapping[str, GenericModelRecord] | None:
    """Read a category's reference offline (disk-only), or None when it cannot be loaded (torch-free).

    Mirrors :func:`model_present_on_disk`'s offline discipline: a short-lived benchmark subprocess must
    never block on a PRIMARY-API fetch, so the read is forced offline. Fails open to None ("unknown"), so a
    surface shows the feature's files as undeterminable rather than wrongly claiming them missing or present.
    """
    try:
        from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY

        from horde_worker_regen.reference_helper import ensure_offline_reference_manager

        manager = ensure_offline_reference_manager()
        return manager.get_all_model_references().get(MODEL_REFERENCE_CATEGORY(category)) or {}
    except Exception as e:  # noqa: BLE001 - reference is best-effort; fail open to "unknown"
        logger.debug(f"Could not load the offline {category!r} reference: {e}")
        return None


def _feature_model_file(
    record_name: str,
    reference: Mapping[str, GenericModelRecord],
    category: str,
    *,
    cache_home: str | None,
    extra_model_directories: Sequence[str | os.PathLike[str]] | None,
) -> FeatureModelFile:
    """Build a :class:`FeatureModelFile` for *record_name* with its size, on-disk state, and primary path."""
    from horde_model_reference.on_disk_layout import file_paths_for, resolve_weights_root

    from horde_worker_regen.model_download_plan import is_model_present

    record = reference[record_name]
    size = getattr(record, "size_on_disk_bytes", None) or (
        sum(download.size_bytes or 0 for download in record.config.download) or None
    )
    on_disk = is_model_present(
        record_name,
        reference,
        cache_home=cache_home,
        extra_model_directories=extra_model_directories,
    )
    paths = file_paths_for(
        record,
        resolve_weights_root(cache_home),
        extra_roots=_coerce_root_paths(extra_model_directories),
    )
    return FeatureModelFile(
        name=record_name,
        category=category,
        size_bytes=size,
        on_disk=on_disk,
        target_path=str(paths[0]) if paths else "",
    )


def controlnet_checkpoint_files(
    control_types: list[str],
    *,
    cache_home: str | None = None,
    extra_model_directories: Sequence[str | os.PathLike[str]] | None = None,
) -> list[FeatureModelFile]:
    """The SD1.5 controlnet checkpoint records the given control types need, with on-disk presence (torch-free).

    A control type whose ``control_<type>`` record is absent from the reference (e.g. an SDXL-only tier with
    no matching SD1.5 record) is dropped: the benchmark cannot plan a file the reference does not describe.
    When the whole reference cannot be read, each row is returned with ``on_disk=None`` (undeterminable),
    never a false "missing".
    """
    reference = _offline_category_reference("controlnet")
    rows: list[FeatureModelFile] = []
    seen: set[str] = set()
    for control_type in control_types:
        record_name = f"{_CONTROLNET_RECORD_PREFIX}{control_type}"
        if record_name in seen:
            continue
        seen.add(record_name)
        if reference is None:
            rows.append(FeatureModelFile(name=record_name, category="controlnet", on_disk=None))
        elif record_name in reference:
            rows.append(
                _feature_model_file(
                    record_name,
                    reference,
                    "controlnet",
                    cache_home=cache_home,
                    extra_model_directories=extra_model_directories,
                ),
            )
    return rows


def post_processor_model_files(
    post_processors: list[str],
    *,
    cache_home: str | None = None,
    extra_model_directories: Sequence[str | os.PathLike[str]] | None = None,
) -> list[FeatureModelFile]:
    """The post-processing model records the given post-processors need, with on-disk presence (torch-free).

    Each post-processor name is looked up across the esrgan/gfpgan/codeformer references (its name IS its
    record name). A post-processor with no model record (``strip_background``/rembg, whose weights are
    fetched lazily by the library at first use) contributes no row: there is no horde-managed file to plan.
    """
    references = {category: _offline_category_reference(category) for category in _POST_PROCESSOR_CATEGORIES}
    rows: list[FeatureModelFile] = []
    seen: set[str] = set()
    for post_processor in post_processors:
        if post_processor in seen:
            continue
        seen.add(post_processor)
        for category in _POST_PROCESSOR_CATEGORIES:
            reference = references[category]
            if reference is not None and post_processor in reference:
                rows.append(
                    _feature_model_file(
                        post_processor,
                        reference,
                        category,
                        cache_home=cache_home,
                        extra_model_directories=extra_model_directories,
                    ),
                )
                break
    return rows


def annotators_present_offline(
    control_types: list[str],
    *,
    cache_home: str | None = None,
    extra_model_directories: Sequence[str | os.PathLike[str]] | None = None,
) -> bool | None:
    """Whether the controlnet annotators for *control_types* are on disk, torch-free via HMR (no hordelib).

    Prefers hordelib's torch-free, existence-based ``annotators_resolvable`` (which consults the checkpoint
    directory the engine actually uses *and* the HuggingFace hub cache a fetch would resolve from), and falls
    back to the model reference's own existence check only when hordelib is too old to expose the resolver.
    Both are torch-free, so the dry-run preview stays cheap. Weightless control types are vacuously present;
    returns None only when presence cannot be determined.
    """
    try:
        from hordelib.preload import annotators_resolvable

        resolved = annotators_resolvable(control_types)
        if resolved is not None:
            return resolved
    except Exception as e:  # noqa: BLE001 - prefer hordelib, but fall back rather than fail
        logger.debug(f"hordelib annotator resolver unavailable; falling back to reference existence check: {e}")

    try:
        from horde_model_reference.on_disk_layout import annotators_present_for_control_types, resolve_weights_root

        return annotators_present_for_control_types(
            control_types,
            resolve_weights_root(cache_home),
            extra_roots=_coerce_root_paths(extra_model_directories),
        )
    except Exception as e:  # noqa: BLE001 - presence is best-effort; fail open to "unknown"
        logger.debug(f"Could not determine controlnet annotator presence offline: {e}")
        return None


def models_disk_plan(model_names: list[str]) -> DownloadPlan | None:
    """The on-disk/size/free-space picture for ``model_names``, or None when it cannot be determined.

    Reuses the same planner the config model-picker uses (so the benchmark's disk story matches the rest of
    the worker) over the on-disk reference, read offline (never downloaded). This is what the download
    modal's dry-run preview calls; running it in a short-lived subprocess means an online manager would
    block on a PRIMARY-API fetch per process (the in-memory freshness state is never inherited, so a warm
    on-disk cache does not spare the round-trip), which can outlast the caller's timeout. Forcing the read
    offline keeps the preview bounded to local disk; the parent/TUI keeps that disk copy current. Fails
    open (returns None) so callers fall back to a cheap, presence-only path offline or in tests.
    """
    try:
        from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY

        from horde_worker_regen.model_download_plan import compute_download_plan
        from horde_worker_regen.reference_helper import ensure_offline_reference_manager

        manager = ensure_offline_reference_manager()
        records = manager.get_all_model_references().get(MODEL_REFERENCE_CATEGORY.image_generation) or {}
        return compute_download_plan(model_names, records)
    except Exception as e:  # noqa: BLE001 - disk sizing is best-effort; fail open to the presence-only path
        logger.debug(f"Could not compute a disk plan for {model_names!r}: {e}")
        return None


def _estimate_vram_mb(scenario: Scenario, baseline: str, *, label: str) -> int | None:
    """Estimate the scenario's heaviest-job VRAM via the hordelib burden registry, or None on error."""
    try:
        from hordelib.feature_impact import estimate_job_burden

        burden = estimate_job_burden(
            baseline=baseline,
            width=max((job.width for job in scenario.image_jobs), default=512),
            height=max((job.height for job in scenario.image_jobs), default=512),
            batch=max((job.n_iter for job in scenario.image_jobs), default=1),
        )
        return burden.vram_mb
    except Exception as e:  # noqa: BLE001 - estimate is informational; never blocks
        logger.debug(f"Burden estimate unavailable for {label}: {e}")
        return None


def _tier_download_bytes(tier: BenchTier) -> int | None:
    """The tier checkpoint's declared download size (huge tiers only), or None when unavailable."""
    if tier not in HUGE_TIERS:
        return None
    try:
        from hordelib.feature_impact import estimate_job_burden

        from horde_worker_regen.benchmark.ladder import _TIER_BASELINES, _TIER_RESOLUTIONS

        resolution = _TIER_RESOLUTIONS[tier]
        burden = estimate_job_burden(baseline=_TIER_BASELINES[tier], width=resolution, height=resolution, batch=1)
        return burden.disk_bytes_needed
    except Exception as e:  # noqa: BLE001 - informational only
        logger.debug(f"Download-size estimate unavailable for {tier}: {e}")
        return None


def _scenario_control_types(scenario: Scenario) -> list[str]:
    """Return the distinct controlnet ``control_type``s the scenario's image jobs exercise (may be empty)."""
    return sorted({job.control_type for job in scenario.image_jobs if job.control_type})


def controlnet_installed() -> bool | None:
    """Whether the controlnet extra (onnxruntime annotators) is installed, or None when undeterminable.

    Fails open (None) so the cheap offline ``plan`` preview and the unit tests, which may not import
    hordelib, do not turn an unknown into a false "missing".
    """
    try:
        from horde_worker_regen.capabilities import controlnet_available

        return controlnet_available()
    except Exception as e:  # noqa: BLE001 - capability probe is best-effort; fail open to "unknown"
        logger.debug(f"Could not determine controlnet availability: {e}")
        return None


def controlnet_annotators_present(control_types: list[str]) -> bool | None:
    """Whether the controlnet annotators for *control_types* are on disk/cached, or None when undeterminable.

    A thin default-rooted alias over the single offline annotator-presence entry point,
    :func:`annotators_present_offline`, kept for the in-process callers that do not pass a cache root.
    Existence-based (hordelib's torch-free ``annotators_resolvable`` with a model-reference fallback), *not*
    the pin-keyed preload marker: a level is nagged to pre-download annotators only when the files are
    genuinely absent, never merely because a full preload-verify has not yet run for the current pin (the
    bug that made a machine with the annotators on disk read as missing). Fails open (None) when presence
    cannot be determined, so a surface treats an unknown as "do not claim missing".
    """
    return annotators_present_offline(control_types)


def _controlnet_annotator_bytes(control_types: list[str]) -> int:
    """ROM annotator-download size for *control_types* via hordelib, or 0 when unavailable/none."""
    if not control_types:
        return 0
    try:
        from hordelib.pipeline.constants import controlnet_annotator_download_bytes

        return controlnet_annotator_download_bytes(control_types)
    except Exception as e:  # noqa: BLE001 - sizing is informational; fail open to 0
        logger.debug(f"Could not size controlnet annotators for {control_types}: {e}")
        return 0


def _scenario_features(scenario: Scenario) -> list[str]:
    """Human-readable feature tags the scenario exercises, derived from its image jobs and forms."""
    jobs = scenario.image_jobs
    features: list[str] = []
    if any(job.hires_fix for job in jobs):
        features.append("hires_fix")
    if any(job.control_type for job in jobs):
        features.append("controlnet")
    if any(job.workflow for job in jobs):
        features.append("qr_code")
    if any(job.post_processing for job in jobs):
        features.append("post_processing")
    if any(job.lora_names for job in jobs):
        features.append("loras")
    if any(job.ti_names for job in jobs):
        features.append("ti")
    if any(job.n_iter > 1 for job in jobs):
        features.append("batch")
    if scenario.alchemy_forms:
        features.append("alchemy")
    return features


def compute_level_requirements(
    level: RampLevel,
    *,
    present_resolver: Callable[[str], bool | None] | None = None,
) -> LevelRequirements:
    """Derive the read-only resource requirements of *level* (the ladder adapter over the shared core)."""
    return _compute_requirements(
        identifier=level.id,
        stage=str(level.stage),
        tier=level.tier,
        axis=str(level.axis),
        baseline=level.baseline_hordelib,
        scenario=level.scenario,
        requires_network=level.requires_network,
        min_disk_free_gb=level.criteria.min_disk_free_gb,
        present_resolver=present_resolver,
    )


def compute_probe_requirements(
    probe: CapabilityProbe,
    *,
    present_resolver: Callable[[str], bool | None] | None = None,
) -> LevelRequirements:
    """Derive the read-only resource requirements of a capability *probe* (the capability adapter).

    The same machine-fit numbers as :func:`compute_level_requirements`, read from a
    :class:`~horde_worker_regen.benchmark.capabilities.probe.CapabilityProbe`, so the gpu probe catalog
    and the executor reuse the one :func:`requirement_skip_reason` gate.
    """
    return _compute_requirements(
        identifier=probe.probe_id,
        stage="",
        tier=probe.capability.tier,
        axis=str(probe.capability.kind),
        baseline=probe.baseline_hordelib,
        scenario=probe.scenario,
        requires_network=probe.requires_network,
        min_disk_free_gb=probe.criteria.min_disk_free_gb,
        present_resolver=present_resolver,
    )


def _compute_requirements(
    *,
    identifier: str,
    stage: str,
    tier: BenchTier,
    axis: str,
    baseline: str,
    scenario: Scenario,
    requires_network: bool,
    min_disk_free_gb: float,
    present_resolver: Callable[[str], bool | None] | None,
) -> LevelRequirements:
    """Derive the read-only resource requirements of a scenario, shared by the level and probe adapters.

    By default this loads the real disk plan (presence, per-model size, download target, free space) via
    :func:`models_disk_plan`, so every surface can name *which* model is missing, *how big* it is, and
    *where* it will land. When a ``present_resolver`` is supplied the cheap presence-only path is used
    instead (no reference load), keeping the ``plan`` preview's offline mode and the unit tests fast.

    Args:
        identifier: The level id or probe slug, used only for labelling.
        stage: The level stage string, or empty for a probe (which has no stage).
        tier: The model tier, for the huge/beta download gates.
        axis: The axis string or the capability kind, for display only.
        baseline: The ``KNOWN_IMAGE_GENERATION_BASELINE`` value, for the burden estimate.
        scenario: The workload to inspect (never mutated).
        requires_network: Whether the work needs network access.
        min_disk_free_gb: The runtime disk-free floor from the criteria.
        present_resolver: When given, a presence-only resolver (True/False/None=unknown) used in place of
            the sized disk plan. Injectable so tests and offline previews avoid touching the reference
            manager; sizes are reported as unknown in that mode.
    """
    models_required = scenario.models_referenced()
    requires_civitai_key = any(job.lora_names or job.ti_names for job in scenario.image_jobs)

    control_types = _scenario_control_types(scenario)
    requires_controlnet = bool(control_types)
    cn_installed = controlnet_installed() if requires_controlnet else None
    # Only meaningful when the extra is present: without onnxruntime the annotators are never fetched, so
    # "present" stays None (the install gate, not the download gate, is what a surface should surface).
    cn_annotators_present = (
        controlnet_annotators_present(control_types) if requires_controlnet and cn_installed else None
    )
    cn_install_hint = ""
    if requires_controlnet and cn_installed is False:
        try:
            from horde_worker_regen.capabilities import controlnet_install_hint

            cn_install_hint = controlnet_install_hint()
        except Exception as e:  # noqa: BLE001 - hint is advisory; a generic message covers the gap
            logger.debug(f"Could not build controlnet install hint: {e}")
            cn_install_hint = "install the `horde-worker-reGen[controlnet]` extra to enable controlnet"

    plan = models_disk_plan(models_required) if present_resolver is None else None
    if plan is not None:
        missing_models = [
            MissingModel(name=info.name, size_bytes=info.size_bytes, target_path=info.target_path)
            for info in plan.models
            if not info.on_disk
        ]
        models_missing = [model.name for model in missing_models]
        download_bytes_needed = plan.to_download_bytes
        present_bytes = plan.present_bytes
        free_disk_bytes = plan.free_disk_bytes
    else:
        resolver = present_resolver if present_resolver is not None else model_present_on_disk
        models_missing = [name for name in models_required if resolver(name) is False]
        missing_models = [MissingModel(name=name) for name in models_missing]
        download_bytes_needed = 0
        present_bytes = 0
        free_disk_bytes = None

    # Controlnet checkpoints are feature models the image-model plan above does not cover; a level that fits
    # the machine but lacks them must read as "download first", not "ready". The cheap present_resolver path
    # (tests/offline) skips the reference read, leaving this empty.
    controlnet_checkpoints_missing = (
        [checkpoint.name for checkpoint in controlnet_checkpoint_files(control_types) if checkpoint.on_disk is False]
        if requires_controlnet and present_resolver is None
        else []
    )

    return LevelRequirements(
        level_id=identifier,
        stage=stage,
        tier=str(tier),
        axis=axis,
        baseline=baseline,
        estimated_vram_mb=_estimate_vram_mb(scenario, baseline, label=identifier),
        min_disk_free_gb=min_disk_free_gb,
        estimated_download_bytes=_tier_download_bytes(tier),
        models_required=models_required,
        models_missing=models_missing,
        missing_models=missing_models,
        download_bytes_needed=download_bytes_needed,
        present_bytes=present_bytes,
        free_disk_bytes=free_disk_bytes,
        requires_network=requires_network,
        requires_civitai_key=requires_civitai_key,
        requires_controlnet=requires_controlnet,
        controlnet_installed=cn_installed,
        controlnet_annotators_present=cn_annotators_present,
        controlnet_annotator_bytes=_controlnet_annotator_bytes(control_types),
        controlnet_install_hint=cn_install_hint,
        controlnet_checkpoints_missing=controlnet_checkpoints_missing,
        features=_scenario_features(scenario),
    )


def requirement_skip_reason(
    req: LevelRequirements,
    *,
    machine: MachineInfo,
    process_mode: str,
    civitai_available: bool,
    force: bool = False,
) -> str | None:
    """Return why *req* cannot run on this machine, or None to proceed.

    Covers only the per-level *resource* gates (model presence, VRAM, CivitAI key); the dynamic
    ramp gates (failed-baseline/axis cascades, ``--skip-downloads``, the empty-weights-root guard) stay
    with the controller, which calls this after them. ``force`` bypasses the machine-fit and key gates
    (insufficient VRAM, missing CivitAI key) but never the absent-checkpoint gate: there is simply
    nothing to run when the weights are not present.

    Disk space is not gated here: if the model is already on disk nothing additional is needed, and
    real-mode benchmarking never downloads checkpoints mid-run. The runtime ``min_disk_free_gb``
    criterion in :class:`~horde_worker_regen.benchmark.criteria.LevelCriteria` catches genuine disk
    exhaustion during a level run.

    Resource gates apply only in ``real`` mode; ``fake``/``dry_run`` download and infer nothing.
    """
    if process_mode != "real":
        return None

    # A genuinely-absent huge/beta checkpoint is a hard skip even under --force: real-mode benchmarking
    # never downloads checkpoints, so there is nothing to run.
    if req.models_missing and _tier_is_huge(req.tier):
        beta_hint = (
            " (a beta model: set HORDE_MODEL_REFERENCE_PRIMARY_API_URL and await publication)"
            if _tier_is_beta(req.tier)
            else " (real-mode benchmarking does not download checkpoints; use Download models first)"
        )
        missing = ", ".join(_missing_label(req, name) for name in req.models_missing)
        return f"{req.tier} model {missing} is not present on disk{beta_hint}"

    # Mirror the runtime coercion in capabilities.coerce_bridge_data_to_capabilities: a worker without the
    # controlnet extra advertises controlnet off, so a controlnet level cannot run there. Surfacing it as a
    # skip (with the install remedy) keeps the preview honest rather than letting the level "run" and fault.
    if not force and req.requires_controlnet and req.controlnet_installed is False:
        hint = req.controlnet_install_hint or "install the `horde-worker-reGen[controlnet]` extra"
        return f"controlnet not installed: {hint}"

    if (
        not force
        and req.estimated_vram_mb is not None
        and machine.total_vram_mb
        and req.estimated_vram_mb > machine.total_vram_mb
    ):
        return f"insufficient VRAM: estimated {req.estimated_vram_mb} MB needed, {machine.total_vram_mb} MB available"

    if not force and req.requires_civitai_key and not civitai_available:
        return (
            "requires a CivitAI API token for lora/TI downloads (set `civitai_api_token` in bridgeData.yaml "
            "or export CIVIT_API_TOKEN)"
        )

    return None


def _missing_label(req: LevelRequirements, name: str) -> str:
    """Render a missing model as ``'name' (N.N GB)`` when its size is known, else just ``'name'``."""
    size = next((model.size_bytes for model in req.missing_models if model.name == name), None)
    return f"{name!r} ({size / 1024**3:.1f} GB)" if size else repr(name)


def _tier_is_huge(tier: str) -> bool:
    """Whether the (stringified) tier is one of the huge-download tiers."""
    return any(tier == str(huge) for huge in HUGE_TIERS)


def _tier_is_beta(tier: str) -> bool:
    """Whether the (stringified) tier is sourced from the beta/pending reference."""
    return any(tier == str(beta) for beta in BETA_TIERS)


__all__ = [
    "LevelRequirements",
    "MissingModel",
    "civitai_token_available",
    "compute_level_requirements",
    "compute_probe_requirements",
    "controlnet_annotators_present",
    "controlnet_installed",
    "model_present_on_disk",
    "models_disk_plan",
    "requirement_skip_reason",
]
