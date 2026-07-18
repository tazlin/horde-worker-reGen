"""Unit tests for the pure ``disagg_optimized N`` selection logic.

Exercises the ranking (shared-VAE cluster first, then popularity, then name), the record/local hash merge
with local winning, the popularity-only degrade, the candidate predicate, and the N guards, entirely over
fabricated records with no config object, network, or filesystem.
"""

from __future__ import annotations

from horde_model_reference.component_hash import ComponentKind
from horde_model_reference.meta_consts import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecordConfig,
    ImageGenerationModelRecord,
)

from horde_worker_regen.bridge_data.disagg_model_selection import (
    is_disagg_optimized_candidate,
    select_disagg_optimized_models,
)


def _record(
    name: str,
    *,
    baseline: KNOWN_IMAGE_GENERATION_BASELINE = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
    inpainting: bool = False,
    downloadable: bool = True,
    vae_hash: str | None = None,
) -> ImageGenerationModelRecord:
    """Build a minimal image-generation record carrying only the fields the selection reads."""
    download = (
        [DownloadRecord(file_name=f"{name}.safetensors", file_url=f"http://example/{name}")] if downloadable else []
    )
    embedded = {"vae": vae_hash} if vae_hash is not None else None
    return ImageGenerationModelRecord(
        name=name,
        baseline=baseline,
        nsfw=False,
        inpainting=inpainting,
        config=GenericModelRecordConfig(download=download, embedded_component_hashes=embedded),
    )


class TestCandidatePredicate:
    """The static candidate predicate: SD1.5/SDXL family, not inpainting, has a download."""

    def test_sd15_and_sdxl_families_are_candidates(self) -> None:
        """Both v1-validated families qualify."""
        assert is_disagg_optimized_candidate(
            _record("sdxl", baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl),
        )
        assert is_disagg_optimized_candidate(
            _record("sd15", baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1),
        )

    def test_other_families_are_not_candidates(self) -> None:
        """A non-SD1.5/SDXL baseline (Flux, Cascade) is excluded."""
        assert not is_disagg_optimized_candidate(_record("flux", baseline=KNOWN_IMAGE_GENERATION_BASELINE.flux_1))
        assert not is_disagg_optimized_candidate(
            _record("cascade", baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade),
        )

    def test_inpainting_is_excluded(self) -> None:
        """An inpainting-variant checkpoint cannot run the staged txt2img graph."""
        assert not is_disagg_optimized_candidate(_record("paint", inpainting=True))

    def test_non_downloadable_is_excluded(self) -> None:
        """A record with no declared download file is not obtainable, so not a candidate."""
        assert not is_disagg_optimized_candidate(_record("nofile", downloadable=False))


class TestClusterRanking:
    """Shared-VAE cluster membership orders ahead of popularity."""

    def test_cluster_outranks_unclustered_at_equal_popularity(self) -> None:
        """Three models sharing one VAE beat more-popular unclustered models."""
        records = {
            "a": _record("a", vae_hash="V"),
            "b": _record("b", vae_hash="V"),
            "c": _record("c", vae_hash="V"),
            "d": _record("d", vae_hash="D"),
            "e": _record("e", vae_hash="E"),
        }
        # d and e are the most popular, but they are unclustered.
        popularity = ["d", "e", "a", "b", "c"]

        result = select_disagg_optimized_models(records, 3, popularity_order=popularity)

        assert result.selected == ["a", "b", "c"]
        assert result.cluster_sizes == {"a": 3, "b": 3, "c": 3}
        assert result.candidate_count == 5
        assert result.hash_data_available is True

    def test_popularity_orders_within_a_cluster_tier(self) -> None:
        """Among equally-clustered models, the more popular one comes first."""
        records = {
            "x": _record("x", vae_hash="V"),
            "y": _record("y", vae_hash="V"),
        }
        result = select_disagg_optimized_models(records, 2, popularity_order=["y", "x"])
        assert result.selected == ["y", "x"]

    def test_local_hash_wins_and_forms_a_cluster(self) -> None:
        """A local sidecar hash overrides the record hash, turning two record-distinct models into a cluster."""
        records = {
            "a": _record("a", vae_hash="RA"),
            "b": _record("b", vae_hash="RB"),
            "c": _record("c", vae_hash="RC"),
        }
        # By record hashes alone every VAE is distinct, so the most popular (c) would lead.
        record_only = select_disagg_optimized_models(records, 1, popularity_order=["c", "a", "b"])
        assert record_only.selected == ["c"]

        # A local sidecar says a and b actually share a VAE, forming a cluster of two that outranks c.
        local = {
            "a": {ComponentKind.VAE: "SHARED"},
            "b": {ComponentKind.VAE: "SHARED"},
        }
        merged = select_disagg_optimized_models(
            records,
            1,
            popularity_order=["c", "a", "b"],
            local_component_hashes=local,
        )
        assert merged.selected == ["a"]
        assert merged.cluster_sizes == {"a": 2}


class TestPopularityDegrade:
    """With no hash data the ranking is pure popularity, identical to top-N filtered to the eligible pool."""

    def test_degrade_matches_eligible_filtered_top_n(self) -> None:
        """No hashes: the selection equals the popularity order filtered to candidates, truncated to N."""
        records = {name: _record(name) for name in ("a", "b", "c", "d")}
        popularity = ["c", "a", "d", "b"]

        result = select_disagg_optimized_models(records, 3, popularity_order=popularity)

        expected = [name for name in popularity if name in records][:3]
        assert result.selected == expected
        assert result.hash_data_available is False

    def test_no_popularity_and_no_hash_orders_by_name(self) -> None:
        """With neither signal, candidates order by name for determinism."""
        records = {name: _record(name) for name in ("b", "a", "c")}
        result = select_disagg_optimized_models(records, 3)
        assert result.selected == ["a", "b", "c"]


class TestNGuards:
    """Zero, negative, and oversized N are handled without error."""

    def test_zero_selects_nothing(self) -> None:
        """N of zero selects nothing."""
        records = {"a": _record("a")}
        assert select_disagg_optimized_models(records, 0).selected == []

    def test_negative_selects_nothing(self) -> None:
        """A negative N selects nothing."""
        records = {"a": _record("a")}
        assert select_disagg_optimized_models(records, -5).selected == []

    def test_oversized_n_returns_all_candidates(self) -> None:
        """An N larger than the pool returns every candidate, not more."""
        records = {name: _record(name) for name in ("a", "b", "c")}
        result = select_disagg_optimized_models(records, 100, popularity_order=["a", "b", "c"])
        assert result.selected == ["a", "b", "c"]
        assert result.candidate_count == 3

    def test_no_candidates_returns_empty(self) -> None:
        """A reference with no eligible model yields an empty selection."""
        records = {"flux": _record("flux", baseline=KNOWN_IMAGE_GENERATION_BASELINE.flux_1)}
        result = select_disagg_optimized_models(records, 2)
        assert result.selected == []
        assert result.candidate_count == 0


class TestDeterminism:
    """Selection is a pure function of its inputs and stable across calls."""

    def test_repeated_calls_are_identical(self) -> None:
        """The same inputs produce the same ordered selection."""
        records = {name: _record(name, vae_hash="V" if name in ("a", "b") else name) for name in ("a", "b", "c", "d")}
        popularity = ["d", "c", "b", "a"]
        first = select_disagg_optimized_models(records, 3, popularity_order=popularity)
        second = select_disagg_optimized_models(records, 3, popularity_order=popularity)
        assert first.selected == second.selected

    def test_ties_break_by_name(self) -> None:
        """Equal cluster size and no popularity: names decide, ascending."""
        records = {name: _record(name) for name in ("c", "a", "b")}
        result = select_disagg_optimized_models(records, 2)
        assert result.selected == ["a", "b"]
