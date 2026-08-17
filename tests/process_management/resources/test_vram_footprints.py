"""Unit tests for the learned VRAM footprint store: keying, both estimate policies, and persistence."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from horde_worker_regen.process_management.resources.resource_budget import platform_context_constant_mb
from horde_worker_regen.process_management.resources.vram_footprints import (
    _MEASURED_ESTIMATE_MARGIN,
    _MIN_OBSERVATIONS_FOR_MEASURED,
    _PERSIST_EVERY_N_OBSERVATIONS,
    _RECENT_WINDOW_SIZE,
    FOOTPRINT_STORE_SCHEMA_VERSION,
    SAFETY_PROCESS_BASELINE,
    FootprintKey,
    FootprintStage,
    LearnedFootprintStore,
    ResolutionBucket,
)


@dataclasses.dataclass(frozen=True)
class _Footprint:
    """A measured per-job footprint in the shape the backend reports one.

    Stands in for ``hordelib.metrics.JobVramFootprint`` so these tests pin the store's own keying rather
    than the backend version installed in the environment; the store consumes the shape structurally.
    """

    peak_resident_weights_mb: float | None = None
    peak_device_used_mb: float | None = None
    resident_weights_after_job_mb: float | None = None
    model_name: str | None = "checkpoint-a"
    baseline: str | None = "stable_diffusion_xl"
    width: int | None = 1024
    height: int | None = 1024
    batch_size: int | None = 1
    stage: str | None = "whole_job"


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


class TestJobFootprintRecording:
    """A backend-measured footprint lands under the store's own keys, or is dropped rather than guessed."""

    def test_whole_job_footprint_writes_only_the_resident_key(self) -> None:
        """A footprint answers what the slot holds; its device-wide high-water is not a per-job activation figure."""
        store = LearnedFootprintStore()
        written = store.observe_job_footprint(
            _Footprint(peak_resident_weights_mb=11000.0, peak_device_used_mb=13500.0),
            baseline=None,
            platform="linux",
            context_constant_mb=144.0,
        )

        assert [key.stage for key in written] == [FootprintStage.RESIDENT]
        (resident,) = written
        assert resident.checkpoint == "checkpoint-a"
        assert resident.resolution_bucket is None
        # The resident population is kept in whole-device terms; the backend measures weights alone.
        assert store.estimate_mb(resident, static_seed_mb=0.0) == pytest.approx(11144.0)
        sample = FootprintKey(
            model_baseline="stable_diffusion_xl",
            resolution_bucket=ResolutionBucket.LE_1024,
            platform="linux",
            stage=FootprintStage.SAMPLE,
        )
        assert store.get_observation(sample) is None

    def test_sample_stage_footprint_without_resident_figure_records_nothing(self) -> None:
        """A device-wide peak alone keys nothing: it would raise the activation population with siblings' weights."""
        store = LearnedFootprintStore()
        written = store.observe_job_footprint(
            _Footprint(peak_device_used_mb=9000.0, stage="sample_stage"),
            baseline=None,
            platform="linux",
        )

        assert written == []
        assert len(store) == 0

    def test_resident_falls_back_to_the_after_job_figure(self) -> None:
        """A run that reports only what it left resident still answers the residency question."""
        store = LearnedFootprintStore()
        written = store.observe_job_footprint(
            _Footprint(resident_weights_after_job_mb=6000.0),
            baseline=None,
            platform="linux",
        )
        assert [key.stage for key in written] == [FootprintStage.RESIDENT]

    def test_baseline_falls_back_to_the_callers_lookup(self) -> None:
        """A backend that could not resolve a baseline is keyed from the parent's model metadata."""
        store = LearnedFootprintStore()
        written = store.observe_job_footprint(
            _Footprint(peak_resident_weights_mb=9000.0, baseline=None),
            baseline="flux_1",
            platform="linux",
        )
        assert [key.model_baseline for key in written] == ["flux_1"]

    def test_unkeyable_footprints_are_dropped(self) -> None:
        """No baseline at all, or no positive resident figure, records nothing."""
        store = LearnedFootprintStore()
        assert store.observe_job_footprint(_Footprint(baseline=None), baseline=None, platform="linux") == []
        assert (
            store.observe_job_footprint(
                _Footprint(peak_device_used_mb=9000.0, resident_weights_after_job_mb=0.0),
                baseline=None,
                platform="linux",
            )
            == []
        )
        assert len(store) == 0


class TestMeasuredEstimate:
    """The bidirectional estimate answers only once a key is observed enough, and carries one margin."""

    def _observe(self, store: LearnedFootprintStore, key: FootprintKey, count: int, mb: float = 13500.0) -> None:
        for _ in range(count):
            store.observe_peak(key, mb)

    def test_under_observed_key_keeps_the_raise_only_contract(self) -> None:
        """One observation below the threshold must not talk a consumer below the static seed."""
        store = LearnedFootprintStore()
        key = _key()
        self._observe(store, key, _MIN_OBSERVATIONS_FOR_MEASURED - 1)

        assert store.measured_estimate_mb(key) is None
        assert store.estimate_mb(key, static_seed_mb=16400.0) == pytest.approx(16400.0)

    def test_threshold_observation_answers_below_the_seed(self) -> None:
        """At the threshold the measurements answer outright, including well under an over-stated seed."""
        store = LearnedFootprintStore()
        key = _key(platform="win32")
        self._observe(store, key, _MIN_OBSERVATIONS_FOR_MEASURED)

        expected = (13500.0 * _MEASURED_ESTIMATE_MARGIN) + platform_context_constant_mb(platform="win32")
        assert store.measured_estimate_mb(key) == pytest.approx(expected)
        assert store.measured_estimate_mb(key) < 16400.0
        # The raise-only policy is untouched by the measured one: they are separate answers.
        assert store.estimate_mb(key, static_seed_mb=16400.0) == pytest.approx(16400.0)

    def test_estimate_tracks_the_recent_window_not_the_all_time_watermark(self) -> None:
        """A figure that has aged out of the window stops holding the estimate up."""
        store = LearnedFootprintStore()
        key = _key(platform="linux")
        store.observe_peak(key, 20000.0)
        self._observe(store, key, _RECENT_WINDOW_SIZE, mb=10000.0)

        expected = (10000.0 * _MEASURED_ESTIMATE_MARGIN) + platform_context_constant_mb(platform="linux")
        assert store.measured_estimate_mb(key) == pytest.approx(expected)
        assert store.estimate_mb(key, static_seed_mb=0.0) == pytest.approx(20000.0)

    def test_observation_count_is_reported(self) -> None:
        """The count is readable so a decision made on measurement can be logged with its evidence."""
        store = LearnedFootprintStore()
        key = _key()
        self._observe(store, key, 3)
        assert store.observation_count(key) == 3
        assert store.observation_count(_key(platform="win32")) == 0


class TestPersistence:
    """Observations survive a restart, and a missing or corrupt file never blocks one."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """A saved store reloads its keys, counts and window, so a restart keeps its calibration."""
        path = tmp_path / "vram_footprints.json"
        store = LearnedFootprintStore(path=path)
        key = _key()
        for mb in (11000.0, 12000.0, 13500.0):
            store.observe_peak(key, mb)
        store.save()

        reloaded = LearnedFootprintStore(path=path)
        observation = reloaded.get_observation(key)
        assert observation is not None
        assert observation.observation_count == 3
        assert observation.watermark_mb == pytest.approx(13500.0)
        assert observation.recent_mb == pytest.approx([11000.0, 12000.0, 13500.0])

    def test_observations_persist_on_the_debounce(self, tmp_path: Path) -> None:
        """The store writes itself out on its own cadence, not only at shutdown."""
        path = tmp_path / "vram_footprints.json"
        store = LearnedFootprintStore(path=path)
        key = _key()
        for _ in range(_PERSIST_EVERY_N_OBSERVATIONS - 1):
            store.observe_peak(key, 100.0)
        assert not path.exists()

        store.observe_peak(key, 100.0)
        assert path.exists()
        assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == FOOTPRINT_STORE_SCHEMA_VERSION

    def test_missing_file_starts_cold(self, tmp_path: Path) -> None:
        """A first run has nothing to read and must start empty rather than fail."""
        assert len(LearnedFootprintStore(path=tmp_path / "absent.json")) == 0

    @pytest.mark.parametrize(
        "content",
        ["not json at all", "{}", '{"schema_version": 999, "observations": []}', '{"schema_version": 1}'],
    )
    def test_unreadable_file_starts_cold(self, tmp_path: Path, content: str) -> None:
        """Corrupt, empty and schema-mismatched files are discarded; the store re-learns from traffic."""
        path = tmp_path / "vram_footprints.json"
        path.write_text(content, encoding="utf-8")
        assert len(LearnedFootprintStore(path=path)) == 0

    def test_entries_the_current_build_cannot_parse_are_skipped(self, tmp_path: Path) -> None:
        """One unreadable entry must not cost the file's other keys."""
        path = tmp_path / "vram_footprints.json"
        good = _key()
        path.write_text(
            json.dumps(
                {
                    "schema_version": FOOTPRINT_STORE_SCHEMA_VERSION,
                    "observations": [
                        {"key": {"model_baseline": "x"}, "observation": {}},
                        {
                            "key": good.model_dump(mode="json"),
                            "observation": {
                                "ewma_mb": 100.0,
                                "watermark_mb": 100.0,
                                "observation_count": 1,
                                "recent_mb": [100.0],
                            },
                        },
                    ],
                },
            ),
            encoding="utf-8",
        )
        store = LearnedFootprintStore(path=path)
        assert len(store) == 1
        assert store.get_observation(good) is not None

    def test_a_pathless_store_never_writes(self, tmp_path: Path) -> None:
        """The in-memory construction (tests, and any consumer that wants no file) writes nothing."""
        store = LearnedFootprintStore()
        store.observe_peak(_key(), 100.0)
        store.save()
        assert list(tmp_path.iterdir()) == []
