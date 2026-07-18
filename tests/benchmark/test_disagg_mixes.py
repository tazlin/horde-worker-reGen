"""Tests for the disagg gate payload mixes."""

from __future__ import annotations

import base64
import math
import struct

import pytest

from horde_worker_regen.benchmark.disagg_mixes import (
    DEFAULT_CHURN_MODEL_POOL,
    DISTINCT_VAE_MODEL_POOL,
    SHARED_VAE_MODEL_POOL,
    DisaggGateMix,
    build_disagg_gate_scenario,
)

_DETERMINISTIC_MIXES = [
    DisaggGateMix.CHURN_DETERMINISTIC,
    DisaggGateMix.CLUSTER_SHARED_VAE,
    DisaggGateMix.CLUSTER_DISTINCT_VAE,
]


class TestDeterminism:
    """Identical inputs must build an identical spec list (generation ids are minted later)."""

    @pytest.mark.parametrize("mix", list(DisaggGateMix))
    def test_same_seed_builds_identical_specs(self, mix: DisaggGateMix) -> None:
        """Two builds with the same inputs produce equal image-job specs for every mix."""
        first = build_disagg_gate_scenario(mix, rung_seconds=300.0, seed=7)
        second = build_disagg_gate_scenario(mix, rung_seconds=300.0, seed=7)
        assert first.image_jobs == second.image_jobs

    @pytest.mark.parametrize("mix", list(DisaggGateMix))
    def test_different_seed_changes_specs(self, mix: DisaggGateMix) -> None:
        """A different seed changes the built specs, so the seed is actually load-bearing."""
        first = build_disagg_gate_scenario(mix, rung_seconds=300.0, seed=1)
        second = build_disagg_gate_scenario(mix, rung_seconds=300.0, seed=2)
        assert first.image_jobs != second.image_jobs


class TestInterleaveShape:
    """The deterministic mixes put a component reload on every job by never repeating a model."""

    @pytest.mark.parametrize("mix", _DETERMINISTIC_MIXES)
    def test_no_two_consecutive_jobs_share_a_model(self, mix: DisaggGateMix) -> None:
        """A strict round-robin means adjacent jobs always target different models."""
        scenario = build_disagg_gate_scenario(mix, rung_seconds=300.0, seed=3)
        models = [spec.model for spec in scenario.image_jobs]
        assert len(models) > len(set(models))  # the list revisits models (it is longer than the pool)
        assert all(earlier != later for earlier, later in zip(models, models[1:], strict=False))

    def test_seeded_random_is_a_soak_scenario(self) -> None:
        """The seeded-random mix is duration-paced (a soak), unlike the fixed-list deterministic mixes."""
        scenario = build_disagg_gate_scenario(DisaggGateMix.CHURN_SEEDED_RANDOM, rung_seconds=120.0, seed=1)
        assert scenario.soak_seconds == 120.0

    @pytest.mark.parametrize("mix", _DETERMINISTIC_MIXES)
    def test_deterministic_mixes_are_fixed_lists(self, mix: DisaggGateMix) -> None:
        """The deterministic mixes expand a fixed job list rather than streaming a soak."""
        scenario = build_disagg_gate_scenario(mix, rung_seconds=120.0, seed=1)
        assert scenario.soak_seconds is None
        assert all(spec.count == 1 for spec in scenario.image_jobs)


class TestFeatureShares:
    """The img2img and LoRA shares land at the requested fraction, deterministically."""

    def test_deterministic_shares_are_exact_floors(self) -> None:
        """The floor-crossing placement yields exactly floor(total * fraction) feature-carrying jobs."""
        scenario = build_disagg_gate_scenario(
            DisaggGateMix.CHURN_DETERMINISTIC,
            rung_seconds=300.0,
            seed=5,
            img2img_fraction=0.2,
            lora_fraction=0.35,
        )
        total = len(scenario.image_jobs)
        img2img_jobs = sum(1 for spec in scenario.image_jobs if spec.source_processing == "img2img")
        lora_jobs = sum(1 for spec in scenario.image_jobs if spec.lora_names)
        assert img2img_jobs == math.floor(total * 0.2)
        assert lora_jobs == math.floor(total * 0.35)

    def test_version_id_lora_pool_flows_into_payload_entries(self) -> None:
        """A version-id pool marks every expanded LoRA entry is_version, so it resolves exactly from cache."""
        scenario = build_disagg_gate_scenario(
            DisaggGateMix.CHURN_DETERMINISTIC,
            rung_seconds=120.0,
            seed=5,
            lora_fraction=1.0,
            lora_pool=("81907",),
            lora_pool_is_version=True,
        )
        jobs = scenario.expand_image_jobs()
        assert jobs
        for job in jobs:
            assert job.payload.loras is not None
            assert [(entry.name, entry.is_version) for entry in job.payload.loras] == [("81907", True)]

    def test_name_lora_pool_stays_non_version(self) -> None:
        """The default name-reference pool keeps is_version False, preserving prior behavior."""
        scenario = build_disagg_gate_scenario(
            DisaggGateMix.CHURN_DETERMINISTIC,
            rung_seconds=120.0,
            seed=5,
            lora_fraction=1.0,
        )
        jobs = scenario.expand_image_jobs()
        assert jobs
        for job in jobs:
            assert job.payload.loras is not None
            assert all(entry.is_version is False for entry in job.payload.loras)

    def test_zero_fractions_produce_no_features(self) -> None:
        """A zero share leaves every job plain txt2img with no LoRA."""
        scenario = build_disagg_gate_scenario(
            DisaggGateMix.CHURN_DETERMINISTIC,
            rung_seconds=300.0,
            seed=5,
            img2img_fraction=0.0,
            lora_fraction=0.0,
        )
        assert all(spec.source_processing is None for spec in scenario.image_jobs)
        assert all(not spec.lora_names for spec in scenario.image_jobs)

    def test_lora_references_come_only_from_the_pool(self) -> None:
        """Every LoRA a job carries is drawn from the supplied pool, none invented."""
        pool = ["ref-a", "ref-b", "ref-c"]
        scenario = build_disagg_gate_scenario(
            DisaggGateMix.CHURN_DETERMINISTIC,
            rung_seconds=300.0,
            seed=5,
            lora_pool=pool,
        )
        carried = {name for spec in scenario.image_jobs for name in spec.lora_names}
        assert carried
        assert carried <= set(pool)

    def test_seeded_random_marginal_shares_track_the_fractions(self) -> None:
        """The weighted template buckets reproduce both marginal shares across the pool."""
        scenario = build_disagg_gate_scenario(
            DisaggGateMix.CHURN_SEEDED_RANDOM,
            rung_seconds=300.0,
            seed=5,
            img2img_fraction=0.2,
            lora_fraction=0.35,
        )
        total_weight = sum(spec.count for spec in scenario.image_jobs)
        img2img_weight = sum(spec.count for spec in scenario.image_jobs if spec.source_processing == "img2img")
        lora_weight = sum(spec.count for spec in scenario.image_jobs if spec.lora_names)
        assert img2img_weight / total_weight == pytest.approx(0.2, abs=0.02)
        assert lora_weight / total_weight == pytest.approx(0.35, abs=0.02)


class TestModelPools:
    """The model pool is operator-supplied data with per-mix defaults."""

    def test_cluster_defaults_are_used_when_no_pool_supplied(self) -> None:
        """Each cluster mix defaults to its measured VAE-sharing pool."""
        shared = build_disagg_gate_scenario(DisaggGateMix.CLUSTER_SHARED_VAE, rung_seconds=120.0, seed=1)
        distinct = build_disagg_gate_scenario(DisaggGateMix.CLUSTER_DISTINCT_VAE, rung_seconds=120.0, seed=1)
        assert set(shared.models_referenced()) == set(SHARED_VAE_MODEL_POOL)
        assert set(distinct.models_referenced()) == set(DISTINCT_VAE_MODEL_POOL)

    def test_churn_defaults_to_the_mixed_baseline_pool(self) -> None:
        """The churn mix defaults to the mixed SD1.5/SDXL pool when none is supplied."""
        scenario = build_disagg_gate_scenario(DisaggGateMix.CHURN_DETERMINISTIC, rung_seconds=120.0, seed=1)
        assert set(scenario.models_referenced()) == set(DEFAULT_CHURN_MODEL_POOL)

    def test_supplied_pool_is_respected(self) -> None:
        """An explicit pool overrides the default and only those models appear."""
        pool = ["Model One", "Model Two", "Model Three"]
        scenario = build_disagg_gate_scenario(
            DisaggGateMix.CHURN_DETERMINISTIC,
            rung_seconds=120.0,
            seed=1,
            model_pool=pool,
        )
        assert set(scenario.models_referenced()) == set(pool)

    def test_single_model_pool_is_rejected(self) -> None:
        """A one-model pool cannot satisfy the no-consecutive-same-model interleave."""
        with pytest.raises(ValueError, match="at least two models"):
            build_disagg_gate_scenario(
                DisaggGateMix.CHURN_DETERMINISTIC,
                rung_seconds=120.0,
                seed=1,
                model_pool=["only-one"],
            )

    @pytest.mark.parametrize("fraction", [-0.1, 1.5])
    def test_out_of_range_fraction_is_rejected(self, fraction: float) -> None:
        """A share fraction outside [0, 1] fails fast."""
        with pytest.raises(ValueError, match="within"):
            build_disagg_gate_scenario(
                DisaggGateMix.CHURN_DETERMINISTIC,
                rung_seconds=120.0,
                seed=1,
                img2img_fraction=fraction,
            )


def _png_dimensions(source_image_b64: str) -> tuple[int, int]:
    """Return the (width, height) read from the IHDR of a base64-encoded PNG."""
    raw = base64.b64decode(source_image_b64)
    # 8-byte signature, then a 4-byte length + 4-byte "IHDR" tag, then width/height as big-endian uint32s.
    width, height = struct.unpack(">II", raw[16:24])
    return width, height


class TestImg2ImgSourceImages:
    """The img2img share carries a real, deterministic source image so the VAE-encode lane fires."""

    def test_img2img_jobs_carry_a_source_image_and_plain_jobs_do_not(self) -> None:
        """Every expanded img2img job ships a source image; every txt2img job ships none."""
        scenario = build_disagg_gate_scenario(
            DisaggGateMix.CHURN_DETERMINISTIC,
            rung_seconds=120.0,
            seed=4,
            img2img_fraction=0.5,
        )
        jobs = scenario.expand_image_jobs()
        img2img_jobs = [job for job in jobs if str(job.source_processing) == "img2img"]
        plain_jobs = [job for job in jobs if job.source_processing is None]

        assert img2img_jobs, "the mix must contain img2img jobs for this to be meaningful"
        assert all(job.source_image for job in img2img_jobs)
        assert all(job.source_image is None for job in plain_jobs)

    def test_source_image_resolution_matches_the_job(self) -> None:
        """The synthetic source image is minted at the job's own resolution (the encode-cost driver)."""
        scenario = build_disagg_gate_scenario(
            DisaggGateMix.CHURN_DETERMINISTIC,
            rung_seconds=120.0,
            seed=4,
            img2img_fraction=0.5,
            model_pool=["Model One", "Model Two"],
        )
        jobs = scenario.expand_image_jobs()
        for job in jobs:
            if job.source_image is None:
                continue
            assert _png_dimensions(job.source_image) == (job.payload.width, job.payload.height)

    def test_source_images_are_byte_identical_across_builds(self) -> None:
        """Same seed builds byte-identical source images, preserving A/B work parity."""
        first = build_disagg_gate_scenario(
            DisaggGateMix.CHURN_DETERMINISTIC,
            rung_seconds=120.0,
            seed=4,
            img2img_fraction=0.5,
        ).expand_image_jobs()
        second = build_disagg_gate_scenario(
            DisaggGateMix.CHURN_DETERMINISTIC,
            rung_seconds=120.0,
            seed=4,
            img2img_fraction=0.5,
        ).expand_image_jobs()

        first_images = [job.source_image for job in first]
        second_images = [job.source_image for job in second]
        assert any(image is not None for image in first_images)
        assert first_images == second_images

    def test_different_seeds_give_different_source_images(self) -> None:
        """A different mix seed changes the source-image content, so the seed drives the image too."""
        first = build_disagg_gate_scenario(
            DisaggGateMix.CHURN_DETERMINISTIC,
            rung_seconds=120.0,
            seed=4,
            img2img_fraction=0.5,
        ).expand_image_jobs()
        second = build_disagg_gate_scenario(
            DisaggGateMix.CHURN_DETERMINISTIC,
            rung_seconds=120.0,
            seed=5,
            img2img_fraction=0.5,
        ).expand_image_jobs()

        first_images = {job.source_image for job in first if job.source_image}
        second_images = {job.source_image for job in second if job.source_image}
        assert first_images and second_images
        assert first_images.isdisjoint(second_images)
