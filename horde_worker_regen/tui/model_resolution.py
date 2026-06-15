"""Resolve a models_to_load / models_to_skip config into the concrete model set the worker will load.

This mirrors, over the already-loaded image-model catalog, the combination the worker performs at
startup (``horde_sdk``'s ``ImageModelLoadResolver`` plus ``load_config._resolve_meta_instructions``):
literal picks and expanded meta commands are unioned, the skip set (literal + meta + the always-skipped
SDXL beta placeholder) is removed, large models are dropped from meta selections unless opted in, and the
result is intersected with the known reference. The output keeps per-model provenance so the editor can
explain *why* each model is (or is not) in the effective set: the entire point is conveying the net effect.

Kept deliberately import-light and free of network/Textual so it is unit-testable in isolation; the
caller supplies the catalog (loaded off-thread) and, for ``top N`` / ``bottom N``, a popularity mapping.
"""

from __future__ import annotations

import dataclasses
import enum

from horde_worker_regen.tui.model_catalog import MetaKind, ModelInfo, parse_meta_instruction

# Baselines the worker excludes from ALL/TOP meta selections unless large models are opted in.
LARGE_BASELINES = frozenset({"flux_1", "stable_cascade"})
_SD21_BASELINES = frozenset({"stable_diffusion_2_512", "stable_diffusion_2_768"})

# The worker unconditionally appends this placeholder to the skip list (horde_sdk bridge_data).
ALWAYS_SKIP = ("SDXL_beta::stability.ai#6901",)

# What an empty models_to_load falls back to worker-side (validate_models_to_load).
DEFAULT_WHEN_EMPTY = "top 2"


class EffectiveStatus(enum.StrEnum):
    """The outcome for one candidate model in the resolved set."""

    ON_DISK = "on_disk"
    """Will load and is already present on disk."""
    TO_DOWNLOAD = "to_download"
    """Will load but must be downloaded first."""
    SKIPPED = "skipped"
    """Would have loaded, but a skip rule removed it."""
    EXCLUDED_LARGE = "excluded_large"
    """A large model dropped from a meta selection because large models are not opted in."""
    UNKNOWN = "unknown"
    """A literal entry that is not a model in the reference (typically a typo)."""


_INCLUDED_STATUSES = frozenset({EffectiveStatus.ON_DISK, EffectiveStatus.TO_DOWNLOAD})


@dataclasses.dataclass(frozen=True)
class EffectiveModel:
    """One row of the resolved view: a model's outcome plus the reason for it."""

    name: str
    baseline: str
    status: EffectiveStatus
    size_bytes: int | None
    on_disk: bool
    reason: str
    """A short, human explanation of provenance (e.g. ``you picked`` / ``via "top 5"``)."""


@dataclasses.dataclass(frozen=True)
class ResolutionResult:
    """The full resolved picture for a load/skip config, ready to render."""

    rows: list[EffectiveModel]
    """Every candidate: included first (alphabetical), then skipped, excluded, and unknown."""
    warnings: list[str]
    """No-op skips, empty selections, and other footguns worth surfacing."""
    needs_resolve: list[str]
    """``top N`` / ``bottom N`` commands that could not be expanded (no popularity supplied yet)."""
    default_applied: bool
    """True when the load list was empty and the worker's ``top 2`` default was assumed."""
    catalog_loaded: bool
    """False when the reference is not loaded yet (so nothing could be resolved)."""

    @property
    def included(self) -> list[EffectiveModel]:
        """The models that will actually load."""
        return [row for row in self.rows if row.status in _INCLUDED_STATUSES]


def _expand_meta(
    kind: MetaKind,
    count: int | None,
    by_name: dict[str, ModelInfo],
    popularity: dict[str, int] | None,
) -> tuple[set[str], bool]:
    """Expand one meta kind to a set of model names. Returns ``(names, resolved)``.

    ``resolved`` is False only for ``top N`` / ``bottom N`` when no popularity mapping is available.
    """
    if kind is MetaKind.ALL:
        return set(by_name), True
    if kind is MetaKind.ALL_SDXL:
        return {n for n, m in by_name.items() if m.baseline == "stable_diffusion_xl"}, True
    if kind is MetaKind.ALL_SD15:
        return {n for n, m in by_name.items() if m.baseline == "stable_diffusion_1"}, True
    if kind is MetaKind.ALL_SD21:
        return {n for n, m in by_name.items() if m.baseline in _SD21_BASELINES}, True
    if kind is MetaKind.ALL_SFW:
        return {n for n, m in by_name.items() if not m.nsfw}, True
    if kind is MetaKind.ALL_NSFW:
        return {n for n, m in by_name.items() if m.nsfw}, True
    if kind is MetaKind.ALL_INPAINTING:
        return {n for n, m in by_name.items() if m.inpainting}, True
    # TOP_N / BOTTOM_N: ordered by usage, intersected with the reference (matching the worker).
    if popularity is None:
        return set(), False
    ranked = sorted(popularity.items(), key=lambda item: item[1], reverse=kind is MetaKind.TOP_N)
    chosen = [name for name, _ in ranked if name in by_name][: count or 0]
    return set(chosen), True


def _quote(entry: str) -> str:
    """Wrap an instruction in typographic quotes for use in a provenance reason."""
    return f"“{entry}”"


def resolve_effective_models(
    load_entries: list[str],
    skip_entries: list[str],
    catalog: list[ModelInfo] | None,
    *,
    load_large_models: bool,
    popularity: dict[str, int] | None = None,
) -> ResolutionResult:
    """Compute the effective model set for a load/skip config against the loaded ``catalog``.

    Args:
        load_entries: The raw ``models_to_load`` list (literal names and/or meta commands).
        skip_entries: The raw ``models_to_skip`` list (literal names and/or meta commands).
        catalog: The image-model reference as ``ModelInfo`` rows, or None when not loaded yet.
        load_large_models: Whether Flux/Cascade are kept in meta selections.
        popularity: Model-name -> last-month usage count, needed only to expand ``top``/``bottom N``.

    Returns:
        A :class:`ResolutionResult` describing each candidate's outcome, with warnings and any
        unexpanded ``top``/``bottom`` commands.
    """
    load_entries = [entry.strip() for entry in load_entries if entry.strip()]
    skip_entries = [entry.strip() for entry in skip_entries if entry.strip()]

    if catalog is None:
        return ResolutionResult(
            rows=[],
            warnings=[],
            needs_resolve=[entry for entry in load_entries + skip_entries if parse_meta_instruction(entry)],
            default_applied=not load_entries,
            catalog_loaded=False,
        )

    by_name = {model.name: model for model in catalog}

    default_applied = not load_entries
    effective_load = [DEFAULT_WHEN_EMPTY] if default_applied else load_entries

    needs_resolve: list[str] = []
    warnings: list[str] = []
    include_source: dict[str, str] = {}
    unknown: dict[str, EffectiveModel] = {}

    # Literal picks first so an explicit pick keeps the "you picked" provenance over a meta match.
    for entry in effective_load:
        if parse_meta_instruction(entry) is not None:
            continue
        if entry in by_name:
            include_source.setdefault(entry, "you picked")
        elif entry not in unknown:
            unknown[entry] = EffectiveModel(
                entry, "", EffectiveStatus.UNKNOWN, None, False, "not in reference (typo?)"
            )

    for entry in effective_load:
        parsed = parse_meta_instruction(entry)
        if parsed is None:
            continue
        names, resolved = _expand_meta(parsed[0], parsed[1], by_name, popularity)
        if not resolved:
            needs_resolve.append(entry)
            continue
        if not names:
            warnings.append(f"{_quote(entry)} matched no models")
        for name in names:
            include_source.setdefault(name, f"via {_quote(entry)}")

    candidates = set(include_source)

    skip_source: dict[str, str] = {}
    for entry in skip_entries:
        parsed = parse_meta_instruction(entry)
        if parsed is None:
            skip_source.setdefault(entry, "skipped by you")
            if entry in by_name and entry not in candidates:
                warnings.append(f"skip {_quote(entry)} removed nothing (not in the load set)")
            elif entry not in by_name:
                warnings.append(f"skip {_quote(entry)} is not a known model")
            continue
        names, resolved = _expand_meta(parsed[0], parsed[1], by_name, popularity)
        if not resolved:
            needs_resolve.append(entry)
            continue
        for name in names:
            skip_source.setdefault(name, f"via skip {_quote(entry)}")
    for placeholder in ALWAYS_SKIP:
        skip_source.setdefault(placeholder, "always skipped (horde placeholder)")

    included: list[EffectiveModel] = []
    skipped: list[EffectiveModel] = []
    excluded_large: list[EffectiveModel] = []
    for name, reason in include_source.items():
        model = by_name[name]
        if name in skip_source:
            skipped.append(
                EffectiveModel(
                    name,
                    model.baseline,
                    EffectiveStatus.SKIPPED,
                    model.size_on_disk_bytes,
                    model.on_disk,
                    skip_source[name],
                ),
            )
            continue
        from_meta = reason.startswith("via ")
        if not load_large_models and from_meta and model.baseline in LARGE_BASELINES:
            excluded_large.append(
                EffectiveModel(
                    name,
                    model.baseline,
                    EffectiveStatus.EXCLUDED_LARGE,
                    model.size_on_disk_bytes,
                    model.on_disk,
                    "large model; enable “Include large models”",
                ),
            )
            continue
        status = EffectiveStatus.ON_DISK if model.on_disk else EffectiveStatus.TO_DOWNLOAD
        included.append(
            EffectiveModel(name, model.baseline, status, model.size_on_disk_bytes, model.on_disk, reason),
        )

    if not included and not needs_resolve:
        warnings.append("No models will load with the current rules.")

    def by_lower_name(model: EffectiveModel) -> str:
        return model.name.lower()

    rows = (
        sorted(included, key=by_lower_name)
        + sorted(skipped, key=by_lower_name)
        + sorted(excluded_large, key=by_lower_name)
        + list(unknown.values())
    )
    return ResolutionResult(
        rows=rows,
        warnings=_dedupe(warnings),
        needs_resolve=_dedupe(needs_resolve),
        default_applied=default_applied,
        catalog_loaded=True,
    )


def _dedupe(values: list[str]) -> list[str]:
    """Return ``values`` with duplicates removed, preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
