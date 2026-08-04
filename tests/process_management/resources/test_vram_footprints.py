"""Unit tests for the learned VRAM footprint store (Stage 1, shadow-only estimation provider)."""

from __future__ import annotations

import pytest

from horde_worker_regen.process_management.resources.vram_footprints import (
    SAFETY_PROCESS_BASELINE,
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


def _resident_key(*, checkpoint: str = "checkpoint-a", platform: str = "linux") -> FootprintKey:
    """A resident-weight key: per checkpoint, with no resolution band."""
    return FootprintKey(
        model_baseline="stable_diffusion_xl",
        resolution_bucket=None,
        platform=platform,
        stage=FootprintStage.RESIDENT,
        checkpoint=checkpoint,
    )


def _safety_key(*, platform: str = "linux") -> FootprintKey:
    """The safety process's key: no baseline, no checkpoint, no resolution band."""
    return FootprintKey(
        model_baseline=SAFETY_PROCESS_BASELINE,
        resolution_bucket=None,
        platform=platform,
        stage=FootprintStage.SAFETY,
    )


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

    def test_omitting_the_checkpoint_matches_an_explicit_none(self) -> None:
        """The baseline-keyed activation stages address one population whether or not the field is passed."""
        store = LearnedFootprintStore()
        store.observe_peak(_key(), 8000.0)
        store.observe_peak(
            FootprintKey(
                model_baseline="stable_diffusion_xl",
                resolution_bucket=ResolutionBucket.LE_1024,
                platform="linux",
                stage=FootprintStage.SAMPLE,
                checkpoint=None,
            ),
            9000.0,
        )
        assert len(store) == 1

    def test_checkpoints_of_one_baseline_are_separate_populations(self) -> None:
        """Two checkpoints sharing a baseline hold different weights, so they never share a watermark."""
        store = LearnedFootprintStore()
        store.observe_peak(_resident_key(checkpoint="checkpoint-a"), 4900.0)
        store.observe_peak(_resident_key(checkpoint="checkpoint-b"), 6800.0)

        assert len(store) == 2
        assert store.estimate_mb(_resident_key(checkpoint="checkpoint-a"), static_seed_mb=0.0) == pytest.approx(4900.0)


class TestResidentAndSafetyStages:
    """The resident-weight and safety stages carry the same raise-only contract on their own keys."""

    def test_a_resident_key_never_collides_with_the_sampling_key(self) -> None:
        """A checkpoint's resident weights and its baseline's sampling peak are separate populations.

        Folding a sampling peak into the resident key would price a merely-loaded slot at the cost of a
        running one, permanently and in the raise-only direction.
        """
        store = LearnedFootprintStore()
        store.observe_peak(_key(), 11000.0)

        assert store.estimate_mb(_resident_key(), static_seed_mb=4900.0) == pytest.approx(4900.0)

    def test_resident_watermark_raises_but_never_lowers_the_seed(self) -> None:
        """A resident observation raises the seed it exceeds and leaves a higher seed alone."""
        store = LearnedFootprintStore()
        key = _resident_key()
        store.observe_peak(key, 6200.0)
        assert store.estimate_mb(key, static_seed_mb=4900.0) == pytest.approx(6200.0)

        store.observe_peak(key, 5100.0)
        assert store.estimate_mb(key, static_seed_mb=4900.0) == pytest.approx(6200.0)
        assert store.estimate_mb(key, static_seed_mb=9000.0) == pytest.approx(9000.0)

    def test_cold_resident_and_safety_keys_return_their_seeds(self) -> None:
        """Before either stage is ever observed, a consumer gets the static seed back unchanged."""
        store = LearnedFootprintStore()
        assert store.estimate_mb(_resident_key(), static_seed_mb=4900.0) == pytest.approx(4900.0)
        assert store.estimate_mb(_safety_key(), static_seed_mb=3044.0) == pytest.approx(3044.0)

    def test_safety_watermark_raises_the_static_charge(self) -> None:
        """A measured safety residency above the static charge becomes the priced figure."""
        store = LearnedFootprintStore()
        store.observe_peak(_safety_key(), 3500.0)
        assert store.estimate_mb(_safety_key(), static_seed_mb=3044.0) == pytest.approx(3500.0)

    def test_the_safety_key_is_independent_of_every_model_key(self) -> None:
        """The safety process belongs to no baseline, so its footprint stands alone in the store."""
        store = LearnedFootprintStore()
        store.observe_peak(_safety_key(), 3500.0)

        assert store.estimate_mb(_key(), static_seed_mb=6158.0) == pytest.approx(6158.0)
        assert store.estimate_mb(_resident_key(), static_seed_mb=4900.0) == pytest.approx(4900.0)
        assert len(store) == 1
