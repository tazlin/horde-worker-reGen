"""Unit tests for the load/skip/meta resolution engine behind the unified model panel."""

from __future__ import annotations

from horde_worker_regen.tui.model_catalog import ModelInfo
from horde_worker_regen.tui.model_resolution import EffectiveStatus, ResolutionResult, resolve_effective_models


def _model(
    name: str,
    baseline: str = "stable_diffusion_xl",
    *,
    nsfw: bool = False,
    inpainting: bool = False,
    size: int | None = 1_000,
    on_disk: bool = False,
) -> ModelInfo:
    """Build a minimal ModelInfo for resolution tests."""
    return ModelInfo(
        name=name,
        baseline=baseline,
        nsfw=nsfw,
        inpainting=inpainting,
        size_on_disk_bytes=size,
        on_disk=on_disk,
    )


_CATALOG = [
    _model("AlbedoBase XL", "stable_diffusion_xl", on_disk=True),
    _model("Juggernaut XL", "stable_diffusion_xl"),
    _model("DreamShaper", "stable_diffusion_1"),
    _model("Naughty", "stable_diffusion_1", nsfw=True),
    _model("Flux Schnell", "flux_1", size=20_000),
]


def _statuses(result: ResolutionResult, name: str) -> EffectiveStatus | None:
    for row in result.rows:
        if row.name == name:
            return row.status
    return None


def test_catalog_not_loaded_defers() -> None:
    """With no catalog, resolution defers and reports that the reference is not loaded."""
    result = resolve_effective_models(["top 5"], [], None, load_large_models=False)
    assert result.catalog_loaded is False
    assert result.needs_resolve == ["top 5"]


def test_literal_pick_is_included_with_provenance() -> None:
    """A literal name in the reference is included and labelled as a manual pick."""
    result = resolve_effective_models(["DreamShaper"], [], _CATALOG, load_large_models=True)
    assert [m.name for m in result.included] == ["DreamShaper"]
    assert result.included[0].reason == "you picked"


def test_unknown_literal_is_flagged_not_included() -> None:
    """A literal name absent from the reference shows as unknown and does not load."""
    result = resolve_effective_models(["Definitely Not Real"], [], _CATALOG, load_large_models=True)
    assert result.included == []
    assert _statuses(result, "Definitely Not Real") is EffectiveStatus.UNKNOWN


def test_all_sdxl_expands_from_reference() -> None:
    """'all sdxl' expands to every SDXL model without needing usage stats."""
    result = resolve_effective_models(["all sdxl"], [], _CATALOG, load_large_models=True)
    assert {m.name for m in result.included} == {"AlbedoBase XL", "Juggernaut XL"}
    assert all(m.reason.startswith("via ") and "all sdxl" in m.reason for m in result.included)
    assert result.needs_resolve == []


def test_skip_removes_and_reports_provenance() -> None:
    """A skip rule removes a model from a meta selection and the row records why."""
    result = resolve_effective_models(["all sdxl"], ["Juggernaut XL"], _CATALOG, load_large_models=True)
    assert {m.name for m in result.included} == {"AlbedoBase XL"}
    assert _statuses(result, "Juggernaut XL") is EffectiveStatus.SKIPPED


def test_noop_skip_warns() -> None:
    """A skip for a known model that is not in the load set is surfaced as a no-op."""
    result = resolve_effective_models(["DreamShaper"], ["Juggernaut XL"], _CATALOG, load_large_models=True)
    assert any("removed nothing" in warning for warning in result.warnings)


def test_top_n_needs_stats_then_expands() -> None:
    """'top N' cannot resolve without popularity, then expands to the most-used models in order."""
    deferred = resolve_effective_models(["top 2"], [], _CATALOG, load_large_models=True)
    assert deferred.needs_resolve == ["top 2"]
    assert deferred.included == []

    popularity = {"Juggernaut XL": 500, "DreamShaper": 300, "AlbedoBase XL": 100}
    resolved = resolve_effective_models(["top 2"], [], _CATALOG, load_large_models=True, popularity=popularity)
    assert {m.name for m in resolved.included} == {"Juggernaut XL", "DreamShaper"}
    assert resolved.needs_resolve == []


def test_large_model_excluded_from_meta_unless_opted_in() -> None:
    """Flux/Cascade drop out of meta selections when large models are not opted in."""
    excluded = resolve_effective_models(["all models"], [], _CATALOG, load_large_models=False)
    assert _statuses(excluded, "Flux Schnell") is EffectiveStatus.EXCLUDED_LARGE
    assert "Flux Schnell" not in {m.name for m in excluded.included}

    included = resolve_effective_models(["all models"], [], _CATALOG, load_large_models=True)
    assert "Flux Schnell" in {m.name for m in included.included}


def test_literal_large_pick_is_kept_even_when_not_opted_in() -> None:
    """Explicitly picking a large model keeps it; the large-model gate only filters meta selections."""
    result = resolve_effective_models(["Flux Schnell"], [], _CATALOG, load_large_models=False)
    assert "Flux Schnell" in {m.name for m in result.included}


def test_empty_load_assumes_top_2_default() -> None:
    """An empty load list assumes the worker's 'top 2' default and needs stats to expand it."""
    result = resolve_effective_models([], [], _CATALOG, load_large_models=True)
    assert result.default_applied is True
    assert result.needs_resolve == ["top 2"]
