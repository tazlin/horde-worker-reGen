"""Resolve a per-card effective worker config from the global config plus a per-GPU override.

A multi-GPU worker is one AI-Horde identity sharing one job queue, but each card may want its own
concurrency, served models, feature flags, and VRAM budget (a small card should not be offered a model
it cannot hold, a fast card may want more threads, and so on). The horde protocol has no way to express
"this card does X, that card does Y" to the server, so per-card configurability is entirely worker-side:
the operator writes a global base config plus an optional per-card override (a
:class:`~horde_worker_regen.bridge_data.data_model.GpuOverride`) keyed by stable device index, and the
orchestrator resolves each card to its own effective config.

The resolved per-card config is itself a :class:`~horde_worker_regen.bridge_data.data_model.reGenBridgeData`,
not a bespoke view type: every scheduler/popper consumer already takes a ``reGenBridgeData``, so a per-card
one is drop-in, and there is no second (and third) place to add a field whenever a new one becomes
per-card-relevant. :func:`resolve_effective_gpu_config` applies the override as a delta
(``model_copy(update=...)`` of the global config with only the operator-set fields) and then re-applies the
same cross-field normalisation the constructor runs (extra-slow clamps, performance-mode timeout scaling,
the ``queue_size`` cap, the controlnet/img2img dependency chain, and meta-instruction extraction when a
model list is replaced), reusing the standalone helpers in
:mod:`horde_worker_regen.bridge_data.data_model` rather than re-deriving the rules here. A card with no
override inherits the global config unchanged (the single-GPU case is bit-for-bit identical), so the
returned config must be treated as read-only by consumers.
"""

from __future__ import annotations

import re

from horde_sdk.worker.dispatch.ai_horde.bridge_data import MetaInstruction

from horde_worker_regen.bridge_data.data_model import (
    GpuOverride,
    apply_extra_slow_clamps,
    cap_queue_size,
    compute_extra_slow_clamps,
    compute_performance_timeout,
    reGenBridgeData,
)

# The base for performance-mode timeout scaling, mirroring reGenBridgeData.process_timeout's field default.
# A card with a performance mode set recomputes its timeout from this base; without one it keeps the global
# (possibly operator-set) value.
_PROCESS_TIMEOUT_DEFAULT = 300


def resolve_effective_gpu_config(
    base: reGenBridgeData,
    override: GpuOverride | None,
) -> reGenBridgeData:
    """Return the effective per-card config: the global *base* with *override*'s set fields applied.

    The override is a delta (only the fields the operator set on it), applied over a copy of the global
    config; the cross-field normalisation the global config went through is then re-applied to the
    per-card combination via :func:`_renormalize_overridden_config`. A card with no override (or an empty
    one) inherits the global config object unchanged, so the single-GPU case is identical to today's.

    Args:
        base: The global, already-validated worker config.
        override: The card's delta, or None to inherit the global config wholesale.

    Returns:
        The effective :class:`reGenBridgeData` for the card. Treat it as read-only: when nothing is
        overridden it is the shared *base* object itself.
    """
    if override is None:
        return base
    # exclude_unset keeps only the fields the operator actually set on the override (by field name, which
    # is what model_copy(update=...) expects); an all-default override changes nothing.
    delta = override.model_dump(exclude_unset=True)
    if not delta:
        return base
    resolved = base.model_copy(update=delta, deep=True)
    _renormalize_overridden_config(resolved, changed_fields=set(delta))
    return resolved


def _renormalize_overridden_config(resolved: reGenBridgeData, *, changed_fields: set[str]) -> None:
    """Re-apply the cross-field normalisation to a freshly merged per-card config (mutated in place).

    ``model_copy(update=...)`` sets the override's fields without re-running validators, so the same passes
    :meth:`reGenBridgeData.validate_performance_modes` (and the SDK's ``validate_model``) perform are redone
    here through the shared standalone helpers, with logging suppressed (the global config already logged
    them). Meta load/skip instructions are re-extracted only when the card replaced the corresponding model
    list, so a ``top 5`` / ``all sdxl`` entry in a per-card list is pulled out as it is for the global config.
    """
    if resolved.extra_slow_worker:
        apply_extra_slow_clamps(
            resolved,
            compute_extra_slow_clamps(
                high_performance_mode=resolved.high_performance_mode,
                moderate_performance_mode=resolved.moderate_performance_mode,
                queue_size=resolved.queue_size,
                max_threads=resolved.max_threads,
                preload_timeout=resolved.preload_timeout,
                log=False,
            ),
        )

    # Recompute from the default base, keeping the global (possibly operator-set) value when no performance
    # mode is active, and rescaling from the default when one is.
    resolved.process_timeout = compute_performance_timeout(
        high_performance_mode=resolved.high_performance_mode,
        moderate_performance_mode=resolved.moderate_performance_mode,
        default_timeout=_PROCESS_TIMEOUT_DEFAULT,
        current_timeout=resolved.process_timeout,
        log=False,
    )
    resolved.queue_size = cap_queue_size(max_threads=resolved.max_threads, queue_size=resolved.queue_size, log=False)

    # ControlNet requires img2img, and SDXL ControlNet requires plain ControlNet (the SDK's validate_model).
    if not resolved.allow_img2img:
        resolved.allow_controlnet = False
    if not resolved.allow_controlnet:
        resolved.allow_sdxl_controlnet = False

    if "image_models_to_load" in changed_fields:
        resolved.image_models_to_load, resolved._meta_load_instructions = _split_meta_instructions(
            resolved.image_models_to_load,
        )
    if "image_models_to_skip" in changed_fields:
        resolved.image_models_to_skip, resolved._meta_skip_instructions = _split_meta_instructions(
            resolved.image_models_to_skip,
        )


def _split_meta_instructions(entries: list[str]) -> tuple[list[str], list[str] | None]:
    """Split a model list into (literal model names, meta instructions), mirroring the SDK's extraction.

    A meta instruction is an entry matching one of the SDK's :class:`MetaInstruction` patterns (``top 5``,
    ``all sdxl``, and so on); the global config strips these out of ``image_models_to_load`` into
    ``meta_load_instructions`` during validation, and a per-card override that replaces a model list needs
    the same treatment. The :class:`MetaInstruction` enum is reused as the single source of the patterns
    rather than re-listing them here. Returns ``None`` for the meta list when there are no meta entries, to
    match the SDK's "unset" representation.
    """
    patterns = list(MetaInstruction.__members__.values())
    literals: list[str] = []
    metas: list[str] = []
    for entry in entries:
        if any(re.match(pattern, entry, re.IGNORECASE) for pattern in patterns):
            metas.append(entry)
        else:
            literals.append(entry)
    return literals, (metas or None)


def resolve_all_effective_gpu_configs(
    base: reGenBridgeData,
    device_indices: list[int],
) -> dict[int, reGenBridgeData]:
    """Resolve every configured card's effective :class:`reGenBridgeData`, keyed by stable device index.

    A device with no entry in ``base.gpu_overrides`` inherits the global config wholesale, so the common
    (and single-GPU) case needs no per-card config at all.
    """
    return {index: resolve_effective_gpu_config(base, base.gpu_overrides.get(index)) for index in device_indices}
