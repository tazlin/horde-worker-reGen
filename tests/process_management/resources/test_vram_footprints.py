"""Unit tests for the learned VRAM footprint store (Stage 1, shadow-only estimation provider)."""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.resources.vram_footprints import (
    FootprintKey,
    FootprintStage,
    LearnedFootprintStore,
    ResolutionBucket,
)


def _key(
    *,
    baseline: str = "stable_diffusion_xl",
    bucket: ResolutionBucket = ResolutionBucket.LE_1024,
    platform: str = "linux",
    stage: FootprintStage = FootprintStage.SAMPLE,
) -> FootprintKey:
    return FootprintKey(model_baseline=baseline, resolution_bucket=bucket, platform=platform, stage=stage)


class TestResolutionBucketClassifier:
    """The classifier bands by maximum dimension and ignores batch."""

    @pytest.mark.parametrize(
        ("width", "height", "expected"),
        [
            (512, 512, ResolutionBucket.LE_512),
            (256, 512, ResolutionBucket.LE_512),
            (513, 512, ResolutionBucket.LE_768),
            (768, 768, ResolutionBucket.LE_768),
            (1024, 768, ResolutionBucket.LE_1024),
            (1024, 1024, ResolutionBucket.LE_1024),
            (1536, 1024, ResolutionBucket.GT_1024),
            (2048, 2048, ResolutionBucket.GT_1024),
        ],
    )
    def test_bands_by_maximum_dimension(self, width: int, height: int, expected: ResolutionBucket) -> None:
        """The larger of width/height decides the band, so orientation does not matter."""
        assert ResolutionBucket.from_dimensions(width, height) is expected

    def test_orientation_is_collapsed(self) -> None:
        """A landscape and its portrait transpose land in the same band."""
        assert ResolutionBucket.from_dimensions(1024, 512) is ResolutionBucket.from_dimensions(512, 1024)

    def test_batch_does_not_change_the_bucket(self) -> None:
        """Batch size is not folded into the key: same dimensions map to the same band regardless."""
        assert ResolutionBucket.from_dimensions(512, 512, batch=1) is ResolutionBucket.from_dimensions(
            512,
            512,
            batch=8,
        )


class TestEwmaAndWatermark:
    """observe_peak maintains an EWMA (observability) and a max-watermark (the estimate basis)."""

    def test_first_observation_seeds_both_statistics(self) -> None:
        """The first peak initialises the EWMA and the watermark to that value."""
        store = LearnedFootprintStore()
        key = _key()
        store.observe_peak(key, 9000.0)

        observation = store.get_observation(key)
        assert observation is not None
        assert observation.ewma_mb == pytest.approx(9000.0)
        assert observation.watermark_mb == pytest.approx(9000.0)
        assert observation.observation_count == 1

    def test_ewma_tracks_toward_new_observations(self) -> None:
        """A second, higher peak moves the EWMA by alpha (0.3) toward it."""
        store = LearnedFootprintStore()
        key = _key()
        store.observe_peak(key, 8000.0)
        store.observe_peak(key, 12000.0)

        observation = store.get_observation(key)
        assert observation is not None
        # 0.3*12000 + 0.7*8000 = 9200
        assert observation.ewma_mb == pytest.approx(9200.0)
        assert observation.observation_count == 2

    def test_watermark_only_rises(self) -> None:
        """The watermark holds the maximum ever seen; a later, lower peak does not lower it."""
        store = LearnedFootprintStore()
        key = _key()
        store.observe_peak(key, 11000.0)
        store.observe_peak(key, 6000.0)

        observation = store.get_observation(key)
        assert observation is not None
        assert observation.watermark_mb == pytest.approx(11000.0)

    def test_non_positive_peaks_are_ignored(self) -> None:
        """A zero or negative reading carries no footprint information and is dropped."""
        store = LearnedFootprintStore()
        key = _key()
        store.observe_peak(key, 0.0)
        store.observe_peak(key, -5.0)

        assert store.get_observation(key) is None
        assert len(store) == 0


class TestEstimateFloorSemantics:
    """estimate_mb overlays the learned watermark on the static seed and can only raise it."""

    def test_cold_key_returns_the_seed(self) -> None:
        """A never-observed key falls back to the static seed unchanged."""
        store = LearnedFootprintStore()
        assert store.estimate_mb(_key(), static_seed_mb=6158.0) == pytest.approx(6158.0)

    def test_learned_watermark_above_seed_raises_the_estimate(self) -> None:
        """A measured peak exceeding the seed lifts the estimate to the watermark."""
        store = LearnedFootprintStore()
        key = _key()
        store.observe_peak(key, 11000.0)
        assert store.estimate_mb(key, static_seed_mb=6158.0) == pytest.approx(11000.0)

    def test_learned_watermark_below_seed_never_lowers_the_estimate(self) -> None:
        """A measured peak below the seed leaves the seed as the floor (undershoot-proofing)."""
        store = LearnedFootprintStore()
        key = _key()
        store.observe_peak(key, 4000.0)
        assert store.estimate_mb(key, static_seed_mb=6158.0) == pytest.approx(6158.0)

    def test_distinct_keys_are_independent(self) -> None:
        """Observations under one key do not affect the estimate of another."""
        store = LearnedFootprintStore()
        observed = _key(stage=FootprintStage.SAMPLE)
        other = _key(stage=FootprintStage.DECODE)
        store.observe_peak(observed, 11000.0)

        assert store.estimate_mb(observed, static_seed_mb=6158.0) == pytest.approx(11000.0)
        assert store.estimate_mb(other, static_seed_mb=6158.0) == pytest.approx(6158.0)


class TestFootprintKeyIdentity:
    """FootprintKey is frozen and value-hashable so it can key the store."""

    def test_equal_keys_share_a_store_entry(self) -> None:
        """Two keys with identical fields address the same observation population."""
        store = LearnedFootprintStore()
        store.observe_peak(_key(), 8000.0)
        store.observe_peak(_key(), 9000.0)
        assert len(store) == 1

    def test_key_is_hashable(self) -> None:
        """A frozen key can be used directly in a set/dict."""
        assert len({_key(), _key()}) == 1
