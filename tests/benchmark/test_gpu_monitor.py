"""Tests for the GPU core-utilization sampler."""

from __future__ import annotations

import sys
import time

import pytest

from horde_worker_regen.utils.gpu_monitor import (
    GpuUtilizationSampler,
    GpuUtilizationSamplers,
    _make_utilization_reader,
)


class TestGpuUtilizationSampler:
    """The sampler summarises injected utilization readings without needing real hardware."""

    def test_mean_and_busy_fraction_math(self) -> None:
        """mean_percent and busy_fraction summarise the collected samples."""
        sampler = GpuUtilizationSampler(busy_threshold_percent=50)
        sampler._samples = [100, 100, 0, 100]  # noqa: SLF001 - exercising the summary math directly
        assert sampler.sample_count == 4
        assert sampler.mean_percent() == 75.0
        assert sampler.busy_fraction() == 0.75

    def test_no_samples_reports_none(self) -> None:
        """Without samples (e.g. no NVML) the figures are None, never a crash."""
        sampler = GpuUtilizationSampler()
        assert sampler.sample_count == 0
        assert sampler.mean_percent() is None
        assert sampler.busy_fraction() is None

    def test_windowed_query_uses_only_recent_samples(self) -> None:
        """A windowed mean/busy considers only samples from the last N seconds (the live rolling view)."""
        sampler = GpuUtilizationSampler(busy_threshold_percent=50)
        now = time.time()
        # Two recent busy samples, one recent idle, and an old busy one that the window must exclude.
        sampler._timeline.extend(  # noqa: SLF001 - injecting the timestamped series directly
            [(now - 1.0, 100), (now - 2.0, 100), (now - 3.0, 0), (now - 500.0, 100)],
        )
        assert sampler.mean_percent(window_seconds=10.0) == (100 + 100 + 0) / 3
        assert sampler.busy_fraction(window_seconds=10.0) == 2 / 3
        # The whole-run figure (no window) reads the separate sample buffer, untouched here.
        assert sampler.mean_percent() is None

    def test_not_before_excludes_pre_first_inference_samples(self) -> None:
        """``not_before`` drops cold-boot warm-up samples so they never dilute the duty figure."""
        sampler = GpuUtilizationSampler(busy_threshold_percent=50)
        now = time.time()
        first_inference = now - 2.5
        # Two idle boot samples before the first inference, then two busy samples after it.
        sampler._timeline.extend(  # noqa: SLF001 - injecting the timestamped series directly
            [(now - 4.0, 0), (now - 3.0, 0), (now - 2.0, 100), (now - 1.0, 100)],
        )
        # Without the cutoff the boot idle halves the mean; with it, only the two busy samples count.
        assert sampler.mean_percent(window_seconds=10.0) == (0 + 0 + 100 + 100) / 4
        assert sampler.mean_percent(window_seconds=10.0, not_before=first_inference) == 100.0
        assert sampler.busy_fraction(window_seconds=10.0, not_before=first_inference) == 1.0

    def test_not_before_after_all_samples_reports_none(self) -> None:
        """A cutoff later than every sample (no inference yet) yields no duty reading rather than zero."""
        sampler = GpuUtilizationSampler(busy_threshold_percent=50)
        now = time.time()
        sampler._timeline.extend([(now - 4.0, 0), (now - 3.0, 0)])  # noqa: SLF001
        assert sampler.mean_percent(window_seconds=10.0, not_before=now) is None

    def test_buffers_are_bounded(self) -> None:
        """A long-lived worker cannot grow the sample buffers without bound."""
        sampler = GpuUtilizationSampler(max_samples=5, read_utilization=lambda: 50)
        for _ in range(20):
            sampler._samples.append(50)  # noqa: SLF001 - exercising the cap directly
        assert sampler.sample_count == 5

    def test_background_sampling_with_injected_reader(self) -> None:
        """A run between start() and stop() collects samples from the injected reader."""
        sampler = GpuUtilizationSampler(interval_seconds=0.002, read_utilization=lambda: 80)
        sampler.start()
        time.sleep(0.05)
        sampler.stop()
        assert sampler.sample_count >= 1
        assert sampler.mean_percent() == 80.0
        assert sampler.busy_fraction() == 1.0

    def test_stop_is_safe_when_never_started(self) -> None:
        """Stopping a sampler that never started (or had no reader) is harmless."""
        sampler = GpuUtilizationSampler(read_utilization=None)
        sampler.stop()  # must not raise
        assert sampler.mean_percent() is None


class TestUtilizationReaderDelegatesToHordelib:
    """The reader consults hordelib's backend-agnostic utilization helper, with no direct NVML in the worker.

    It reads NVML directly via the torch-free ``hordelib.utils.nvml`` submodule (not the torch-importing
    ``get_accelerator_utilization_percent`` nor the ``hordelib.api`` facade), so building a sampler in the
    orchestrator never drags torch into the parent process.
    """

    def test_reader_uses_nvml_helper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When NVML reports a percentage, the built reader returns it for the requested device."""
        import hordelib.utils.nvml as nvml

        seen: dict[str, int] = {}

        def fake_utilization(index: int = 0) -> int | None:
            seen["index"] = index
            return 42

        monkeypatch.setattr(nvml, "get_device_utilization_percent", fake_utilization, raising=False)

        reader = _make_utilization_reader(3)
        assert reader is not None
        assert reader() == 42
        assert seen["index"] == 3

    def test_reader_is_none_when_no_backend_telemetry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When NVML reports None (non-NVIDIA / no telemetry), the sampler builds no reader (no-op)."""
        import hordelib.utils.nvml as nvml

        monkeypatch.setattr(
            nvml,
            "get_device_utilization_percent",
            lambda index=0: None,
            raising=False,
        )
        assert _make_utilization_reader(0) is None

    def test_reader_is_none_when_helper_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A hordelib too old to expose the helper degrades gracefully to no sampling, never a crash."""
        import hordelib.utils.nvml as nvml

        monkeypatch.delattr(nvml, "get_device_utilization_percent", raising=False)
        # Ensure the lazy `from hordelib.utils.nvml import ...` re-resolves against the patched module.
        monkeypatch.setitem(sys.modules, "hordelib.utils.nvml", nvml)
        assert _make_utilization_reader(0) is None

    def test_dump_timeline_writes_valid_json(self, tmp_path) -> None:  # noqa: ANN001
        """The diagnostics dump round-trips through JSON regardless of the internal buffer type."""
        import json

        sampler = GpuUtilizationSampler()
        sampler._timeline.extend([(1000.0, 42), (1001.0, 7)])  # noqa: SLF001
        target = tmp_path / "timeline.json"
        sampler.dump_timeline(target)
        assert json.loads(target.read_text(encoding="utf-8")) == [[1000.0, 42], [1001.0, 7]]


class TestGpuUtilizationSamplers:
    """The multi-card holder keeps one sampler per driven card and reduces across them."""

    def test_samples_every_driven_card_from_its_own_reader(self) -> None:
        """Each card is polled through its own reader; the per-card figures stay separate."""
        samplers = GpuUtilizationSamplers(
            [0, 1],
            interval_seconds=0.002,
            busy_threshold_percent=50,
            readers={0: lambda: 90, 1: lambda: 10},
        )
        samplers.start()
        time.sleep(0.05)
        samplers.stop()

        assert samplers.device_indices == (0, 1)
        assert samplers.mean_percent_per_card() == {0: 90.0, 1: 10.0}
        assert samplers.busy_fraction_per_card() == {0: 1.0, 1: 0.0}
        assert all(count >= 1 for count in samplers.sample_count_per_card().values())

    def test_worker_wide_figures_reduce_across_cards(self) -> None:
        """The scalar figures are the unweighted mean of the per-card ones, so cards count equally."""
        samplers = GpuUtilizationSamplers([0, 1], busy_threshold_percent=50)
        samplers._samplers[0]._samples.extend([100, 100, 100, 100])  # noqa: SLF001 - drive the math directly
        samplers._samplers[1]._samples.extend([0])  # noqa: SLF001

        assert samplers.mean_percent() == 50.0
        assert samplers.busy_fraction() == 0.5
        assert samplers.sample_count == 5

    def test_unmeasured_cards_are_omitted_rather_than_counted_as_zero(self) -> None:
        """A card with no telemetry drops out of the per-card view and out of the reduction."""
        samplers = GpuUtilizationSamplers([0, 1])
        samplers._samplers[0]._samples.extend([60, 80])  # noqa: SLF001

        assert samplers.mean_percent_per_card() == {0: 70.0}
        assert samplers.sample_count_per_card() == {0: 2, 1: 0}
        assert samplers.mean_percent() == 70.0

    def test_no_measured_cards_report_none(self) -> None:
        """With nothing sampled anywhere the reductions are None, never zero."""
        samplers = GpuUtilizationSamplers([0, 1])
        assert samplers.mean_percent() is None
        assert samplers.busy_fraction() is None
        assert samplers.sample_count == 0

    def test_single_card_matches_a_bare_sampler(self) -> None:
        """A single-GPU host's figures are identical to one sampler on index 0."""
        samplers = GpuUtilizationSamplers([0], busy_threshold_percent=50)
        bare = GpuUtilizationSampler(busy_threshold_percent=50)
        for values in (samplers._samplers[0]._samples, bare._samples):  # noqa: SLF001
            values.extend([100, 100, 0, 100])

        assert samplers.mean_percent() == bare.mean_percent()
        assert samplers.busy_fraction() == bare.busy_fraction()
        assert samplers.sample_count == bare.sample_count
        assert samplers.mean_percent_per_card() == {0: bare.mean_percent()}

    def test_stop_is_safe_when_never_started(self) -> None:
        """Stopping cards that never started (no telemetry) is harmless."""
        GpuUtilizationSamplers([0, 1], readers={}).stop()  # must not raise
