"""Tests for the offline learned-footprint view the duty report prints."""

from __future__ import annotations

from pathlib import Path

import pytest

from horde_worker_regen.analysis.footprint_view import (
    default_footprint_store_path,
    learned_sampling_footprints,
    render_learned_sampling_footprints,
)
from horde_worker_regen.app_state import default_app_state_dir
from horde_worker_regen.process_management.resources.resource_budget import platform_context_constant_mb
from horde_worker_regen.process_management.resources.vram_footprints import (
    FOOTPRINT_STORE_FILENAME,
    FootprintKey,
    FootprintStage,
    LearnedFootprintStore,
    ResolutionBucket,
)

_PLATFORM = "linux"
_SAMPLE_KEY = FootprintKey(
    model_baseline="stable_diffusion_xl",
    resolution_bucket=ResolutionBucket.LE_1024,
    platform=_PLATFORM,
    stage=FootprintStage.SAMPLE,
)
_ACTIVATION_KEY = _SAMPLE_KEY.model_copy(update={"stage": FootprintStage.SAMPLE_ACTIVATION})


def _resident_key(checkpoint: str) -> FootprintKey:
    return FootprintKey(
        model_baseline="stable_diffusion_xl",
        resolution_bucket=None,
        platform=_PLATFORM,
        stage=FootprintStage.RESIDENT,
        checkpoint=checkpoint,
    )


def _write_store(path: Path) -> LearnedFootprintStore:
    store = LearnedFootprintStore(path=path)
    for peak_mb in (9000.0, 9500.0, 12000.0, 10000.0, 9800.0, 15000.0):
        store.observe_peak(_SAMPLE_KEY, peak_mb)
    store.observe_peak(_ACTIVATION_KEY, 6000.0)
    store.observe_peak(_resident_key("Checkpoint A"), 8000.0)
    store.observe_peak(_resident_key("Checkpoint B"), 7000.0)
    store.observe_peak(
        FootprintKey(
            model_baseline="flux_1",
            resolution_bucket=ResolutionBucket.LE_1024,
            platform=_PLATFORM,
            stage=FootprintStage.SAMPLE,
        ),
        17000.0,
    )
    store.save()
    return store


class TestLearnedSamplingFootprints:
    """Each sampling key's figures come from the store's own estimate methods."""

    def test_rows_carry_the_store_figures(self, tmp_path: Path) -> None:
        """Watermark, measured price, activation watermark and residents match what the store reports."""
        path = tmp_path / FOOTPRINT_STORE_FILENAME
        store = _write_store(path)
        context_mb = platform_context_constant_mb(platform=_PLATFORM)

        rows = learned_sampling_footprints(path)

        assert [(row.baseline, row.stage) for row in rows] == [("flux_1", "sample"), ("stable_diffusion_xl", "sample")]
        sdxl = rows[1]
        assert sdxl.watermark_mb == store.estimate_mb(_SAMPLE_KEY, static_seed_mb=0.0) == 15000.0
        measured_raw_mb = store.measured_estimate_mb(_SAMPLE_KEY)
        assert measured_raw_mb is not None
        assert sdxl.measured_mb == pytest.approx(measured_raw_mb - context_mb)
        assert (sdxl.observation_count, sdxl.window_size) == (6, 6)
        assert sdxl.activation_watermark_mb == 6000.0
        assert [(resident.checkpoint, resident.resident_mb) for resident in sdxl.residents] == [
            ("Checkpoint A", pytest.approx(8000.0 - context_mb)),
            ("Checkpoint B", pytest.approx(7000.0 - context_mb)),
        ]
        flux = rows[0]
        assert flux.measured_mb is None, "one job is under the store's observation floor"
        assert flux.activation_watermark_mb is None
        assert flux.residents == []

    def test_reading_never_rewrites_the_store(self, tmp_path: Path) -> None:
        """A bundle's store is evidence; the view loads it without saving."""
        path = tmp_path / FOOTPRINT_STORE_FILENAME
        _write_store(path)
        before = path.read_bytes()

        learned_sampling_footprints(path)

        assert path.read_bytes() == before

    def test_render_lists_figures_without_a_verdict(self, tmp_path: Path) -> None:
        """The rendering states each figure and the resident range for the baseline."""
        path = tmp_path / FOOTPRINT_STORE_FILENAME
        _write_store(path)

        output = render_learned_sampling_footprints(path, learned_sampling_footprints(path))

        assert "stable_diffusion_xl le_1024 sample [all]: watermark 15000MB, measured " in output
        assert "activation watermark 6000MB" in output
        assert "over 2 checkpoints" in output
        assert "flux_1 le_1024 sample [all]: watermark 17000MB, measured n/a (1 in window)" in output

    def test_missing_store_yields_no_rows(self, tmp_path: Path) -> None:
        """A path with no store lists nothing."""
        assert learned_sampling_footprints(tmp_path / FOOTPRINT_STORE_FILENAME) == []


class TestDefaultFootprintStorePath:
    """The bundle's copy is read beside a bundle's stats, the worker's store otherwise."""

    def test_bundle_copy_beside_stats(self, tmp_path: Path) -> None:
        """``<bundle>/stats`` reads ``<bundle>/config/vram_footprints.json``."""
        bundle_copy = tmp_path / "config" / FOOTPRINT_STORE_FILENAME
        bundle_copy.parent.mkdir()
        bundle_copy.write_text("{}", encoding="utf-8")

        assert default_footprint_store_path(tmp_path / "stats") == bundle_copy

    def test_worker_store_without_a_bundle(self, tmp_path: Path) -> None:
        """A stats directory with no bundle config beside it falls back to the worker's own store."""
        expected = default_app_state_dir() / FOOTPRINT_STORE_FILENAME

        assert default_footprint_store_path(tmp_path / "stats") == expected
        assert default_footprint_store_path(None) == expected
