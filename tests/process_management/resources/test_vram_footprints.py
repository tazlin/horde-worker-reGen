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
    _PER_READING_SAMPLE_SCHEMA_VERSION,
    _PERSIST_EVERY_N_OBSERVATIONS,
    _POOLED_ONLY_SCHEMA_VERSION,
    _RECENT_WINDOW_SIZE,
    FOOTPRINT_STORE_SCHEMA_VERSION,
    SAFETY_PROCESS_BASELINE,
    FootprintKey,
    FootprintStage,
    LearnedFootprintStore,
    ResolutionBucket,
    learned_post_process_vram_mb,
    plausible_activation_ceiling_mb,
    plausible_sampling_peak_mb,
    post_process_footprint_key,
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

    def test_net_of_context_takes_back_exactly_the_context_it_added(self) -> None:
        """A job-level price is the margined basis alone, whatever the platform's context figure."""
        store = LearnedFootprintStore()
        for platform in ("linux", "win32"):
            key = _key(platform=platform)
            self._observe(store, key, _MIN_OBSERVATIONS_FOR_MEASURED)
            assert store.measured_estimate_net_of_context_mb(key) == pytest.approx(13500.0 * _MEASURED_ESTIMATE_MARGIN)
        assert store.measured_estimate_net_of_context_mb(_key(platform="darwin")) is None

    def test_estimate_tracks_the_recent_window_not_the_all_time_watermark(self) -> None:
        """A figure that has aged out of the window stops holding the estimate up."""
        store = LearnedFootprintStore()
        key = _key(platform="linux")
        store.observe_peak(key, 20000.0)
        self._observe(store, key, _RECENT_WINDOW_SIZE, mb=10000.0)

        expected = (10000.0 * _MEASURED_ESTIMATE_MARGIN) + platform_context_constant_mb(platform="linux")
        assert store.measured_estimate_mb(key) == pytest.approx(expected)
        assert store.estimate_mb(key, static_seed_mb=0.0) == pytest.approx(20000.0)

    def test_a_lone_outlier_job_does_not_set_the_sampling_price(self) -> None:
        """A sampling key prices at its second-highest job, so one inherited high job cannot price the key."""
        store = LearnedFootprintStore()
        key = _key(platform="win32")
        self._observe(store, key, _MIN_OBSERVATIONS_FOR_MEASURED - 1, mb=13000.0)
        store.observe_peak(key, 14260.0)

        expected = (13000.0 * _MEASURED_ESTIMATE_MARGIN) + platform_context_constant_mb(platform="win32")
        assert store.measured_estimate_mb(key) == pytest.approx(expected)
        assert store.estimate_mb(key, static_seed_mb=0.0) == pytest.approx(14260.0)

    def test_two_high_jobs_set_the_sampling_price(self) -> None:
        """A high figure that repeats is the key's cost, so the second-highest carries it."""
        store = LearnedFootprintStore()
        key = _key(stage=FootprintStage.SAMPLE_ISOLATED)
        self._observe(store, key, _MIN_OBSERVATIONS_FOR_MEASURED - 2, mb=9000.0)
        self._observe(store, key, 2, mb=11000.0)

        expected = (11000.0 * _MEASURED_ESTIMATE_MARGIN) + platform_context_constant_mb(platform="linux")
        assert store.measured_estimate_mb(key) == pytest.approx(expected)

    def test_below_the_minimum_the_outlier_keeps_the_raise_only_price(self) -> None:
        """Too few jobs to tell an outlier from the norm, so the watermark (outlier included) governs."""
        store = LearnedFootprintStore()
        key = _key()
        self._observe(store, key, _MIN_OBSERVATIONS_FOR_MEASURED - 2, mb=13000.0)
        store.observe_peak(key, 14260.0)

        assert store.measured_estimate_mb(key) is None
        assert store.estimate_mb(key, static_seed_mb=6158.0) == pytest.approx(14260.0)

    def test_a_resident_key_keeps_the_maximum_of_its_window(self) -> None:
        """Resident readings are one entry per at-rest reading, so the window's maximum still prices them."""
        store = LearnedFootprintStore()
        key = _resident_key()
        self._observe(store, key, _MIN_OBSERVATIONS_FOR_MEASURED - 1, mb=7000.0)
        store.observe_peak(key, 7400.0)

        expected = (7400.0 * _MEASURED_ESTIMATE_MARGIN) + platform_context_constant_mb(platform="linux")
        assert store.measured_estimate_mb(key) == pytest.approx(expected)

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

    def test_a_per_reading_file_rebuilds_its_sampling_windows_from_jobs(self, tmp_path: Path) -> None:
        """A file whose SAMPLE windows hold per-report readings loads with those windows empty.

        Its watermark, EWMA and count are kept, so the raise-only price survives the upgrade. Its readings are
        not jobs, so the measured price waits for the window to refill with job figures.
        """
        path = tmp_path / "vram_footprints.json"
        sample = _key()
        isolated = _key(stage=FootprintStage.SAMPLE_ISOLATED)
        resident = _resident_key()
        readings = [9000.0, 14260.0, 9100.0, 9050.0, 9200.0, 9000.0]

        def _entry(key: FootprintKey) -> dict[str, object]:
            return {
                "key": key.model_dump(mode="json"),
                "observation": {
                    "ewma_mb": 9500.0,
                    "watermark_mb": 14260.0,
                    "observation_count": len(readings),
                    "recent_mb": readings,
                },
            }

        path.write_text(
            json.dumps(
                {
                    "schema_version": _PER_READING_SAMPLE_SCHEMA_VERSION,
                    "observations": [_entry(sample), _entry(isolated), _entry(resident)],
                },
            ),
            encoding="utf-8",
        )

        store = LearnedFootprintStore(path=path)

        loaded_sample = store.get_observation(sample)
        assert loaded_sample is not None
        assert loaded_sample.recent_mb == []
        assert loaded_sample.watermark_mb == pytest.approx(14260.0)
        assert loaded_sample.ewma_mb == pytest.approx(9500.0)
        assert loaded_sample.observation_count == len(readings)
        assert store.measured_estimate_mb(sample) is None
        assert store.estimate_mb(sample, static_seed_mb=6158.0) == pytest.approx(14260.0)
        # The isolated sampler was always fed once per job, and resident windows hold readings by design.
        for kept in (isolated, resident):
            loaded = store.get_observation(kept)
            assert loaded is not None
            assert loaded.recent_mb == pytest.approx(readings)

    def test_a_saved_store_is_read_back_as_per_job_windows(self, tmp_path: Path) -> None:
        """A file this build writes keeps its SAMPLE windows on reload, since they already hold jobs."""
        path = tmp_path / "vram_footprints.json"
        store = LearnedFootprintStore(path=path)
        for mb in (11000.0, 12000.0):
            store.observe_peak(_key(), mb)
        store.save()

        assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] != _PER_READING_SAMPLE_SCHEMA_VERSION
        reloaded = LearnedFootprintStore(path=path).get_observation(_key())
        assert reloaded is not None
        assert reloaded.recent_mb == pytest.approx([11000.0, 12000.0])

    def test_a_pooled_only_file_loads_unchanged(self, tmp_path: Path) -> None:
        """A file from before job-class keys keeps its pooled windows, which price every class until it has its own."""
        path = tmp_path / "vram_footprints.json"
        pooled = _key()
        readings = [9000.0, 12500.0, 9100.0, 12400.0, 9200.0]
        path.write_text(
            json.dumps(
                {
                    "schema_version": _POOLED_ONLY_SCHEMA_VERSION,
                    "observations": [
                        {
                            "key": pooled.model_dump(mode="json"),
                            "observation": {
                                "ewma_mb": 10000.0,
                                "watermark_mb": 12500.0,
                                "observation_count": len(readings),
                                "recent_mb": readings,
                            },
                        },
                    ],
                },
            ),
            encoding="utf-8",
        )

        store = LearnedFootprintStore(path=path)

        loaded = store.get_observation(pooled)
        assert loaded is not None
        assert loaded.recent_mb == pytest.approx(readings)
        single = pooled.for_job_class(batch=1, hires_fix=False)
        assert store.measured_job_estimate_net_of_context_mb(pooled, single) == pytest.approx(
            store.measured_estimate_net_of_context_mb(pooled),
        )

    def test_job_class_keys_are_read_back_apart_from_their_pooled_key(self, tmp_path: Path) -> None:
        """A saved class key reloads as its own key, never folded into the pooled key it narrows."""
        path = tmp_path / "vram_footprints.json"
        store = LearnedFootprintStore(path=path)
        pooled = _key()
        batched = pooled.for_job_class(batch=4, hires_fix=False)
        store.observe_job_peak(pooled, batched, 12500.0)
        store.observe_peak(pooled, 7500.0)
        store.save()

        reloaded = LearnedFootprintStore(path=path)
        pooled_observation = reloaded.get_observation(pooled)
        batched_observation = reloaded.get_observation(batched)
        assert pooled_observation is not None and batched_observation is not None
        assert pooled_observation.recent_mb == pytest.approx([12500.0, 7500.0])
        assert batched_observation.recent_mb == pytest.approx([12500.0])

    def test_a_pathless_store_never_writes(self, tmp_path: Path) -> None:
        """The in-memory construction (tests, and any consumer that wants no file) writes nothing."""
        store = LearnedFootprintStore()
        store.observe_peak(_key(), 100.0)
        store.save()
        assert list(tmp_path.iterdir()) == []


class TestPlausibilityBounds:
    """A reading that cannot describe its key is refused at the seam, never folded into a raise-only watermark."""

    def test_a_reading_below_the_floor_is_dropped(self) -> None:
        """A resident figure under the checkpoint's own weight bytes is not a resident observation."""
        store = LearnedFootprintStore()
        store.observe_peak(_resident_key(), 1456.0, plausible_min_mb=19500.0)
        assert len(store) == 0

    def test_a_reading_above_the_ceiling_is_dropped(self) -> None:
        """An activation figure at the card's size is an allocator artefact, not a job's peak."""
        store = LearnedFootprintStore()
        store.observe_peak(_key(), 23044.0, plausible_max_mb=23347.0)
        assert store.get_observation(_key()) is not None
        store.observe_peak(_key(bucket=ResolutionBucket.GT_1024), 23604.0, plausible_max_mb=23347.0)
        assert store.get_observation(_key(bucket=ResolutionBucket.GT_1024)) is None

    def test_a_reading_inside_the_bounds_is_kept(self) -> None:
        """The bounds only refuse; a plausible reading records exactly as before."""
        store = LearnedFootprintStore()
        store.observe_peak(_key(), 10654.0, plausible_min_mb=4900.0, plausible_max_mb=23347.0)
        observation = store.get_observation(_key())
        assert observation is not None
        assert observation.watermark_mb == 10654.0

    def test_unbounded_observation_is_unchanged(self) -> None:
        """A feeder with nothing to bound by (an unsized card, an unpriced checkpoint) keeps the old contract."""
        store = LearnedFootprintStore()
        store.observe_peak(_key(), 30000.0)
        assert store.get_observation(_key()) is not None

    def test_job_footprint_resident_figure_below_the_floor_writes_no_key(self) -> None:
        """A run that block-swapped most of a file measures only what fit; that is not the file's resident cost."""
        store = LearnedFootprintStore()
        written = store.observe_job_footprint(
            _Footprint(peak_resident_weights_mb=1456.0),
            baseline=None,
            platform="linux",
            resident_floor_mb=19500.0,
        )
        assert written == []
        assert len(store) == 0

    def test_activation_ceiling_is_the_card_net_of_its_noise_buffer(self) -> None:
        """The ceiling is the same margin admission keeps free, and unknown when the card is unsized."""
        assert plausible_activation_ceiling_mb(None) is None
        assert plausible_activation_ceiling_mb(0.0) is None
        assert plausible_activation_ceiling_mb(24576.0) == pytest.approx(24576.0 - 1228.8)


class TestSamplingPlausibilityBound:
    """A sampling reading is bounded by what the heaviest job in its band could plausibly need."""

    def test_card_sized_readings_fall_outside_their_band(self) -> None:
        """Allocator readings near the card's size cannot describe an SDXL or SD1.5 job's peak."""
        sdxl = plausible_sampling_peak_mb(_key(), 24576.0)
        sd15 = plausible_sampling_peak_mb(_key(baseline="stable_diffusion_1", bucket=ResolutionBucket.LE_512), 24576.0)
        assert sdxl is not None and 9256.0 < sdxl < 20584.0
        assert sd15 is not None and 6386.0 < sd15 < 19980.0

    def test_large_models_are_capped_at_the_card(self) -> None:
        """A band whose plausible peak exceeds the card keeps the card's own bound."""
        bound = plausible_sampling_peak_mb(_key(baseline="qwen_image", bucket=ResolutionBucket.GT_1024), 24576.0)
        assert bound == pytest.approx(plausible_activation_ceiling_mb(24576.0))

    def test_keys_without_a_band_or_a_known_baseline_keep_the_card_bound(self) -> None:
        """Resident keys and baselines the burden model does not know are bounded by the card alone."""
        card = plausible_activation_ceiling_mb(24576.0)
        assert plausible_sampling_peak_mb(_resident_key(), 24576.0) == card
        assert plausible_sampling_peak_mb(_key(baseline="no_such_baseline"), 24576.0) == card
        assert plausible_sampling_peak_mb(_key(baseline="no_such_baseline"), None) is None

    def test_a_card_sized_sdxl_reading_is_refused_at_the_seam(self) -> None:
        """The feeder's bound keeps a 20 GB SDXL reading out of the raise-only watermark."""
        store = LearnedFootprintStore()
        store.observe_peak(_key(), 20584.0, plausible_max_mb=plausible_sampling_peak_mb(_key(), 24576.0))
        assert store.get_observation(_key()) is None


class TestSanitizeSamplingObservations:
    """Stored sampling statistics above their bound are clamped; everything else is left as it was."""

    def test_implausible_sampling_statistics_are_clamped_and_kept(self, tmp_path: Path) -> None:
        """The watermark, recent window and EWMA are cut back; the count and genuine readings survive."""
        path = tmp_path / "footprints.json"
        store = LearnedFootprintStore(path=path)
        for reading in (6874.0, 9256.0, 20584.0):
            store.observe_peak(_key(), reading)
        store.observe_peak(_resident_key(), 21000.0)
        store.observe_peak(_key(bucket=ResolutionBucket.LE_512), 7000.0)

        changed = store.sanitize_sampling_observations(lambda key: plausible_sampling_peak_mb(key, None))

        bound = plausible_sampling_peak_mb(_key(), None)
        assert bound is not None
        assert changed == [_key()]
        clamped = store.get_observation(_key())
        assert clamped is not None
        assert clamped.watermark_mb == pytest.approx(bound)
        assert max(clamped.recent_mb) == pytest.approx(bound)
        assert 6874.0 in clamped.recent_mb and 9256.0 in clamped.recent_mb
        assert clamped.ewma_mb <= bound
        assert clamped.observation_count == 3
        resident = store.get_observation(_resident_key())
        assert resident is not None and resident.watermark_mb == 21000.0
        plausible = store.get_observation(_key(bucket=ResolutionBucket.LE_512))
        assert plausible is not None and plausible.watermark_mb == 7000.0
        reloaded = LearnedFootprintStore(path=path)
        reloaded_observation = reloaded.get_observation(_key())
        assert reloaded_observation is not None
        assert reloaded_observation.watermark_mb == pytest.approx(bound)

    def test_a_clean_store_is_untouched(self) -> None:
        """Nothing above its bound means nothing changes and nothing is reported."""
        store = LearnedFootprintStore()
        store.observe_peak(_key(), 9256.0)
        assert store.sanitize_sampling_observations(lambda key: plausible_sampling_peak_mb(key, None)) == []


class TestPostProcessChainPricing:
    """A chain is keyed by its lane operations and image size, and priced from what such chains grew the lane by."""

    @staticmethod
    def _job(forms: list[str], *, width: int = 1024, height: int = 1024):  # noqa: ANN205
        from tests.process_management.conftest import make_job_pop_response

        return make_job_pop_response(width=width, height=height, post_processing=forms)

    def test_the_key_is_the_lane_operations_in_any_order(self) -> None:
        """The same operations in another order are the same chain, and background removal runs on another lane."""
        one = post_process_footprint_key(self._job(["RealESRGAN_x4plus", "GFPGAN"]))
        other = post_process_footprint_key(self._job(["GFPGAN", "RealESRGAN_x4plus", "strip_background"]))
        assert one is not None and one == other
        assert one.stage is FootprintStage.POST_PROCESS
        assert one.checkpoint == "GFPGAN+RealESRGAN_x4plus"
        assert one.resolution_bucket is ResolutionBucket.LE_1024

    def test_a_job_the_lane_runs_nothing_for_has_no_key(self) -> None:
        """No lane operation, no chain to price."""
        assert post_process_footprint_key(self._job([])) is None
        assert post_process_footprint_key(self._job(["strip_background"])) is None

    def test_a_cold_chain_keeps_the_static_estimate(self) -> None:
        """Below the measured minimum the chain is booked at the static estimate."""
        store = LearnedFootprintStore()
        job = self._job(["GFPGAN", "RealESRGAN_x4plus"])
        key = post_process_footprint_key(job)
        assert key is not None
        store.observe_peak(key, 1500.0)
        assert learned_post_process_vram_mb(store, job, 6400.0) == pytest.approx(6400.0)

    def test_a_measured_chain_lowers_the_static_estimate(self) -> None:
        """Five chains price the next one at the margined second-highest growth, net of the context charge."""
        store = LearnedFootprintStore()
        job = self._job(["GFPGAN", "RealESRGAN_x4plus"])
        key = post_process_footprint_key(job)
        assert key is not None
        for growth_mb in (1400.0, 1500.0, 1450.0, 1500.0, 1300.0):
            store.observe_peak(key, growth_mb)
        assert learned_post_process_vram_mb(store, job, 6400.0) == pytest.approx(1500.0 * _MEASURED_ESTIMATE_MARGIN)

    def test_a_chain_above_its_estimate_is_raised_by_its_watermark(self) -> None:
        """A chain that grew the lane past the estimate is never booked below what it was seen to use."""
        store = LearnedFootprintStore()
        job = self._job(["GFPGAN", "RealESRGAN_x4plus"])
        key = post_process_footprint_key(job)
        assert key is not None
        store.observe_peak(key, 7000.0)
        assert learned_post_process_vram_mb(store, job, 6400.0) == pytest.approx(7000.0)

    def test_without_a_store_the_estimate_stands(self) -> None:
        """A parent with no store books the static estimate, None included."""
        job = self._job(["GFPGAN"])
        assert learned_post_process_vram_mb(None, job, 6400.0) == 6400.0
        assert learned_post_process_vram_mb(None, job, None) is None
