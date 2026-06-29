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


def _coerce_cpu_only_mode(bridge_data: reGenBridgeData, *, log: bool) -> list[str]:
    """Disable image generation (the dreamer role) when this is a CPU-only / alchemist-only install.

    Returns the coercion descriptions applied (empty when this is not a CPU install or there was nothing
    to disable). Clears the resolved image model list and turns off dynamic model loading so the worker
    never advertises or pops an image job it would run impractically slowly on CPU. Alchemy is left
    untouched: its graph forms (upscale, face-fix) and CLIP forms (interrogation, caption) run acceptably
    on CPU, so an alchemist-enabled worker stays useful. A CPU install with alchemist disabled would have
    nothing to do, which is surfaced as a warning rather than silently forcing the role on.
    """
    from horde_worker_regen.compute_mode import is_cpu_only_install

    if not is_cpu_only_install():
        return []

    coercions: list[str] = []

    if bridge_data.image_models_to_load:
        bridge_data.image_models_to_load = []
        message = (
            "image_models_to_load coerced to empty: this is a CPU-only (alchemist-only) install "
            "(bin/backend is 'cpu'), so image generation is disabled. Reinstall a GPU build "
            "(e.g. update-runtime --cu132) to enable image generation."
        )
        coercions.append(message)
        if log:
            logger.warning(message)

    if bridge_data.dynamic_models:
        bridge_data.dynamic_models = False
        message = "dynamic_models coerced to False: image generation is disabled on a CPU-only install."
        coercions.append(message)
        if log:
            logger.warning(message)

    if not bridge_data.alchemist and log:
        logger.warning(
            "This is a CPU-only install with image generation disabled and alchemist=False, so the worker "
            "has nothing to serve. Set alchemist: true in bridgeData.yaml to run alchemy forms on CPU.",
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

    # A CPU-only / alchemist-only install cannot serve image generation (CPU inference is impractically
    # slow), so the dreamer role is disabled by definition while the pure-CPU alchemy forms stay on offer.
    # This is gated on the install's declared intent (the torch-free bin/backend sentinel), not a torch
    # probe, so it stays cheap on the hot reload path. Done before the hordelib feature probes because it
    # needs none of them.
    coercions.extend(_coerce_cpu_only_mode(bridge_data, log=log))

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
