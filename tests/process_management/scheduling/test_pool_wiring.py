"""Tests for the config-to-engine and performance-model wiring glue for the fixed pool.

Covers seat-count resolution (auto and explicit), manual-pin passthrough, and the expected-value adapter's
behaviour when the performance model has no rate for a candidate.
"""

from __future__ import annotations

import pytest

from horde_worker_regen.bridge_data.data_model import ModelPoolConfig, PinnedModelEntry
from horde_worker_regen.process_management.scheduling.model_pool import PinnedModel
from horde_worker_regen.process_management.scheduling.performance_model import JobSignature, baseline_signature
from horde_worker_regen.process_management.scheduling.pool_wiring import (
    build_expected_value_adapter,
    build_pool_params,
)


class TestBuildPoolParams:
    """Config resolves into the engine's frozen parameters, with an auto seat count and pins passed through."""

    def test_auto_seat_count_resolves_to_process_count(self) -> None:
        """Test auto seat count resolves to process count."""
        params = build_pool_params(ModelPoolConfig(seats=0), max_inference_processes=3)
        assert params.seat_count == 3

    def test_explicit_seat_count_passes_through(self) -> None:
        """Test explicit seat count passes through."""
        params = build_pool_params(ModelPoolConfig(seats=2), max_inference_processes=5)
        assert params.seat_count == 2

    def test_pins_pass_through_with_affinity(self) -> None:
        """Test pins pass through with affinity."""
        config = ModelPoolConfig(
            pinned=[
                PinnedModelEntry(name="alpha", affinity=0.9),
                PinnedModelEntry(name="beta", affinity=0.4),
            ],
        )
        params = build_pool_params(config, max_inference_processes=2)
        assert params.pinned == (
            PinnedModel(name="alpha", affinity=0.9),
            PinnedModel(name="beta", affinity=0.4),
        )

    def test_tunables_pass_through(self) -> None:
        """Test tunables pass through."""
        config = ModelPoolConfig(
            ranker_enabled=False,
            rotation_minutes=30.0,
            min_dwell_minutes=5.0,
            rescue_enabled=True,
            rescue_eta_seconds=1234.0,
            rescue_window_minutes=7.0,
        )
        params = build_pool_params(config, max_inference_processes=1)
        assert params.ranker_enabled is False
        assert params.rotation_minutes == 30.0
        assert params.min_dwell_minutes == 5.0
        assert params.rescue_enabled is True
        assert params.rescue_eta_seconds == 1234.0
        assert params.rescue_window_minutes == 7.0


class TestExpectedValueAdapter:
    """The adapter turns a model name into an expected kudos-per-wall-second earning rate."""

    def test_unknown_baseline_returns_none(self) -> None:
        """A model whose baseline cannot be resolved has no estimable value."""
        adapter = build_expected_value_adapter(
            expected_its=lambda _signature: 12.0,
            baseline_resolver=lambda _name: None,
        )
        assert adapter("mystery_model") is None

    def test_absent_perf_data_returns_none(self) -> None:
        """A baseline the performance model has no rate for has no estimable value."""
        adapter = build_expected_value_adapter(
            expected_its=lambda _signature: None,
            baseline_resolver=lambda _name: "stable_diffusion_1",
        )
        assert adapter("some_sd15_model") is None

    def test_value_is_price_over_wall_seconds(self) -> None:
        """The rate is the baseline's canonical job price over sampling seconds plus the per-job overhead."""
        adapter = build_expected_value_adapter(
            expected_its=lambda signature: 10.0,
            baseline_resolver=lambda _name: "stable_diffusion_1",
        )
        value = adapter("an_sd15_model")
        assert value is not None
        signature = baseline_signature(baseline="stable_diffusion_1", resolution=512)
        expected = 7.84 / (signature.total_sampling_iterations / 10.0 + 5.0)
        assert value == pytest.approx(expected)

    def test_better_paying_baseline_outearns_cheaper_at_higher_speed(self) -> None:
        """An SDXL model out-earns an SD1.5 model even when the SD1.5 samples several times faster.

        This is the live-regression contract: a card full of fast cheap jobs pays per-job overhead on every
        one of them, so the earning rate must favour the better-paying queue, not the faster sampler.
        """

        def its_by_baseline(signature: JobSignature) -> float | None:
            return 2.5 if signature.baseline == "stable_diffusion_xl" else 10.0

        adapter = build_expected_value_adapter(
            expected_its=its_by_baseline,
            baseline_resolver=lambda name: "stable_diffusion_xl" if name == "xl" else "stable_diffusion_1",
        )
        xl_value = adapter("xl")
        sd15_value = adapter("sd15")
        assert xl_value is not None and sd15_value is not None
        assert xl_value > sd15_value

    def test_sdxl_and_sd15_use_their_native_resolutions(self) -> None:
        """The SDXL and SD1.5 signatures are built at their different native resolutions."""
        seen: dict[str, JobSignature] = {}

        def record_expected_its(signature: JobSignature) -> float | None:
            seen[signature.baseline] = signature
            return 9.0

        adapter = build_expected_value_adapter(
            expected_its=record_expected_its,
            baseline_resolver=lambda name: "stable_diffusion_xl" if name == "an_xl_model" else "stable_diffusion_1",
        )

        assert adapter("an_xl_model") is not None
        assert adapter("an_sd15_model") is not None
        assert seen["stable_diffusion_xl"].resolution_bucket != seen["stable_diffusion_1"].resolution_bucket
