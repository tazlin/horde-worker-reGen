"""Runtime detection of optional, backend-blocking features, and config coercion to match.

Some worker features depend on native packages that have no wheels for every accelerator ComfyUI
supports: ``onnxruntime`` backs the controlnet annotators (only Openpose/DWPose among the
horde-exposed control types actually needs it) and ``rembg`` backs ``strip_background``. Those
packages live in ``horde-engine`` extras, re-exported as the worker's ``controlnet`` and
``post-processing`` extras (see ``pyproject.toml``), so a lean base install on Intel XPU / Apple MPS
/ Ascend simply lacks them. Everything else (core SD/Flux inference, the NSFW/CSAM safety
classifier, ESRGAN upscalers, CodeFormer/GFPGAN face-fixers, LoRA, img2img) is pure PyTorch and runs
on every backend.

This module reads hordelib's typed capability registry (``hordelib.feature_requirements``) and
coerces the loaded bridge data so the worker never advertises a feature it cannot actually run: a job
that requested it would otherwise fault. Image-generation post-processing is one atomic switch
(``allow_post_processing``): the AI Horde API has no per-job way to accept upscale/face-fix while
refusing ``strip_background``, so when ``rembg`` is absent the whole switch is coerced off even though
the upscalers/face-fixers themselves would run. Alchemy forms are enumerated per-form, so there
``strip_background`` alone is dropped (see :func:`strip_background_available`) and the pure-torch
forms stay on offer.

hordelib is imported lazily inside each probe, and from its torch-free ``feature_requirements`` submodule
rather than the ``hordelib.api`` facade (which would drag torch into the orchestrator). So importing this
module (and the process manager) stays cheap, requires no torch, and preserves the no-GPU/no-network
dry-run test path.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from hordelib.feature_impact import FEATURE_KIND

    from horde_worker_regen.bridge_data.data_model import reGenBridgeData
    from horde_worker_regen.process_management.scheduling.workload_flow import WorkloadKind

# The one worker-local fact about feature extras: how a horde-engine feature extra is re-exported under
# the worker's own ``[project.optional-dependencies]`` name. horde-engine's ``rembg`` extra is surfaced
# to operators as ``post-processing``; ``controlnet`` keeps its name. Everything else about gated features
# (which FEATURE_KINDs are gated, the packages each needs, the labels) is owned by hordelib's
# ``feature_requirements`` registry and read from it below, so there is a single source of truth.
_HORDE_ENGINE_EXTRA_TO_WORKER_EXTRA: dict[str, str] = {
    "rembg": "post-processing",
    "controlnet": "controlnet",
}


@functools.lru_cache(maxsize=1)
def _worker_extra_for_feature() -> dict[str, str]:
    """Map each gated ``FEATURE_KIND`` value to the worker extra that installs it.

    Derived from hordelib's typed requirement registry (the source of truth for which features are
    backend-gated and the horde-engine extra each needs) via the worker-local re-export aliases in
    :data:`_HORDE_ENGINE_EXTRA_TO_WORKER_EXTRA`. A horde-engine extra with no worker alias falls back to
    its own name, so a newly gated feature is still named usefully in install hints before an alias is
    added. Cached because hordelib's registry does not change during a run.
    """
    from hordelib.feature_requirements import get_feature_requirement_registry

    return {
        requirement.feature.value: _HORDE_ENGINE_EXTRA_TO_WORKER_EXTRA.get(requirement.extra, requirement.extra)
        for requirement in get_feature_requirement_registry().values()
    }


@functools.lru_cache(maxsize=1)
def _available_features() -> frozenset[FEATURE_KIND]:
    """Return the features hordelib reports as runnable in this environment (cached for the process).

    Installed packages do not change during a run, so the probe is memoized. Returns an empty set if
    hordelib cannot be imported at all: in that case the worker cannot run inference regardless, so
    treating every gated feature as unavailable is the safe reading.
    """
    try:
        from hordelib.feature_requirements import available_features
    except Exception as exc:  # pragma: no cover - hordelib is a hard dependency in practice
        logger.debug("Could not import hordelib to probe feature availability; assuming none: {}", exc)
        return frozenset()
    return frozenset(available_features())


def strip_background_available() -> bool:
    """Return whether background removal (``rembg``) is installed and runnable here."""
    from hordelib.feature_impact import FEATURE_KIND

    return FEATURE_KIND.strip_background in _available_features()


def controlnet_available() -> bool:
    """Return whether controlnet preprocessing (``onnxruntime`` annotators) is installed and runnable."""
    from hordelib.feature_impact import FEATURE_KIND

    return FEATURE_KIND.controlnet in _available_features()


@functools.lru_cache(maxsize=1)
def vectorize_available() -> bool:
    """Return whether the image vectorizer (``vtracer``) is installed and runnable here.

    Unlike the other probes, vectorization is a worker-only alchemy op (raster -> SVG) that does
    not route through hordelib, so its dependency lives in the worker's own ``vectorize`` extra
    rather than hordelib's feature registry. A plain import probe is therefore the right check.
    """
    try:
        import vtracer  # noqa: F401
    except Exception as exc:
        logger.debug("Could not import vtracer to probe vectorize availability; assuming unavailable: {}", exc)
        return False
    return True


@functools.lru_cache(maxsize=1)
def describe_available() -> bool:
    """Return whether the technical-metadata ``describe`` form can run here.

    Like :func:`vectorize_available`, this is a worker-only alchemy op that does not route through
    hordelib; its blurhash/perceptual-hash pieces come from the worker's own dependencies, so a
    plain import probe is the right check. The plain dimensions/dominant-colour parts use Pillow and
    are always available, but the form is only offered when the full bundle can be produced.
    """
    try:
        import blurhash  # noqa: F401
        import imagehash  # noqa: F401
    except Exception as exc:
        logger.debug("Could not import describe deps (blurhash/imagehash); assuming unavailable: {}", exc)
        return False
    return True


def controlnet_install_hint() -> str:
    """Build the actionable "install the controlnet extra" fragment naming the missing packages.

    Public wrapper around :func:`_install_hint` for the benchmark planner, so a controlnet level that
    cannot run on this install surfaces the same remedy the runtime coercion logs.
    """
    from hordelib.feature_impact import FEATURE_KIND

    return _install_hint(FEATURE_KIND.controlnet)


def post_processing_install_hint() -> str:
    """Build the actionable "install the post-processing extra" fragment naming the missing packages.

    Public wrapper around :func:`_install_hint` for the feature-readiness table, so a post-processing
    feature that cannot run on this install surfaces the same remedy the runtime coercion logs.
    """
    from hordelib.feature_impact import FEATURE_KIND

    return _install_hint(FEATURE_KIND.strip_background)


def _install_hint(feature: FEATURE_KIND) -> str:
    """Build an actionable "install this extra" fragment naming the missing packages for *feature*."""
    from hordelib.feature_requirements import missing_packages

    missing = missing_packages(feature)
    extra = _worker_extra_for_feature().get(feature.value, feature.value)
    packages = ", ".join(missing) if missing else "the required packages"
    return (
        f"{packages} not installed; install `horde-worker-reGen[{extra}]` (or use an install profile "
        "that includes it) to enable this on your backend"
    )


def enabled_workloads(bridge_data: reGenBridgeData) -> frozenset[WorkloadKind]:
    """Return the workloads this worker actually serves, derived from config and the install.

    The single source of truth that turns the operator-facing role flags (``dreamer``, ``alchemist``)
    into the internal :class:`WorkloadKind` vocabulary the rest of the worker reasons in (process
    sizing, the flow registry, the dashboard). A CPU-only install cannot serve image generation
    regardless of ``dreamer`` (CPU inference is impractically slow), so image generation is dropped
    there. A future worker type adds a single membership rule here and is then first-class everywhere
    downstream rather than threading another boolean through every site.

    ``WorkloadKind`` is imported lazily so this module's import stays torch-free (the benchmark planner
    imports it); the import chain behind ``WorkloadKind`` is only pulled in when a caller actually needs
    the served-workload set, which only happens in contexts that already tolerate it.
    """
    from horde_worker_regen.compute_mode import is_cpu_only_install
    from horde_worker_regen.process_management.scheduling.workload_flow import WorkloadKind

    workloads: set[WorkloadKind] = set()
    if bridge_data.dreamer and not is_cpu_only_install():
        workloads.add(WorkloadKind.IMAGE_GENERATION)
    if bridge_data.alchemist:
        workloads.add(WorkloadKind.ALCHEMY)
    return frozenset(workloads)


def _coerce_workload_config(bridge_data: reGenBridgeData, *, log: bool) -> list[str]:
    """Disable image generation when this worker does not serve the image-generation workload.

    Returns the coercion descriptions applied (empty when image generation is served or there was
    nothing to disable). Image generation is not served either because this is a CPU-only install
    (where CPU inference is impractically slow) or because the operator deselected the dreamer role
    (``dreamer: false``). In both cases the resolved image model list is cleared and dynamic model
    loading is turned off so the worker never advertises or pops an image job it will not run. Alchemy
    is left untouched: its graph forms (upscale, face-fix) and CLIP forms (interrogation, caption) run
    acceptably without image generation, so an alchemist-enabled worker stays useful. A worker that
    serves no workload at all is surfaced as a warning rather than silently doing nothing.
    """
    from horde_worker_regen.compute_mode import is_cpu_only_install
    from horde_worker_regen.process_management.scheduling.workload_flow import WorkloadKind

    if WorkloadKind.IMAGE_GENERATION in enabled_workloads(bridge_data):
        return []

    coercions: list[str] = []

    # Distinguish the two reasons image generation is off so the remedy in the log is actionable: a
    # CPU install needs a GPU reinstall, a deliberate opt-out just needs dreamer turned back on.
    if is_cpu_only_install():
        reason = "this is a CPU-only (alchemist-only) install (bin/backend is 'cpu')"
        remedy = "Reinstall a GPU build (e.g. update-runtime --cu132) to enable image generation."
    else:
        reason = "the dreamer role is disabled (dreamer: false)"
        remedy = "Set dreamer: true in bridgeData.yaml to enable image generation."

    if bridge_data.image_models_to_load:
        bridge_data.image_models_to_load = []
        message = f"image_models_to_load coerced to empty: {reason}, so image generation is disabled. {remedy}"
        coercions.append(message)
        if log:
            logger.warning(message)

    if bridge_data.dynamic_models:
        bridge_data.dynamic_models = False
        message = f"dynamic_models coerced to False: image generation is disabled because {reason}."
        coercions.append(message)
        if log:
            logger.warning(message)

    if not bridge_data.alchemist and log:
        logger.warning(
            f"Image generation is disabled ({reason}) and alchemist=False, so the worker has nothing to "
            "serve. Set alchemist: true in bridgeData.yaml to run alchemy forms.",
        )

    return coercions


def coerce_bridge_data_to_capabilities(bridge_data: reGenBridgeData, *, log: bool = True) -> list[str]:
    """Coerce advertised features off when the packages backing them are not installed.

    Mutates *bridge_data* in place (mirroring the model's other config-normalisation passes) so that
    every downstream consumer (the poppers, the status reporter) advertises only what this install can
    actually serve. Returns a list of human-readable descriptions of each coercion applied (empty when
    nothing changed), which doubles as the assertion surface for tests.

    Args:
        bridge_data: The loaded config to normalise. Modified in place.
        log: When True, emit a loud warning for each coercion. Callers on the hot reload path leave
            this on; it only fires when something actually changes, so it does not spam.

    Returns:
        The list of coercion descriptions applied.
    """
    # Dry-run skips real inference, so capability constraints do not apply and we avoid importing
    # hordelib in that path.
    if bridge_data.dry_run_skip_inference:
        return []

    coercions: list[str] = []

    # A worker that does not serve image generation (a CPU-only install, or a deliberate dreamer: false
    # opt-out) has its image model list and dynamic loading cleared so it never advertises image work it
    # will not run, while the alchemy forms stay on offer. This is gated on config plus the install's
    # declared intent (the torch-free bin/backend sentinel), not a torch probe, so it stays cheap on the
    # hot reload path. Done before the hordelib feature probes because it needs none of them.
    coercions.extend(_coerce_workload_config(bridge_data, log=log))

    from hordelib.feature_impact import FEATURE_KIND

    if not strip_background_available() and bridge_data.allow_post_processing:
        bridge_data.allow_post_processing = False
        hint = _install_hint(FEATURE_KIND.strip_background)
        message = (
            f"allow_post_processing coerced to False: {hint}. Post-processing is offered as one "
            "bucket because the AI Horde API cannot accept upscale/face-fix while refusing "
            "strip_background, so the whole option is disabled. (Alchemy still offers the pure-torch "
            "upscalers and face-fixers if alchemist is enabled.)"
        )
        coercions.append(message)
        if log:
            logger.warning(message)

    if not controlnet_available() and (bridge_data.allow_controlnet or bridge_data.allow_sdxl_controlnet):
        bridge_data.allow_controlnet = False
        bridge_data.allow_sdxl_controlnet = False
        hint = _install_hint(FEATURE_KIND.controlnet)
        message = f"allow_controlnet and allow_sdxl_controlnet coerced to False: {hint}."
        coercions.append(message)
        if log:
            logger.warning(message)

    return coercions
