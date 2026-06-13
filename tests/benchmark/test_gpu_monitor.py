"""Tests for the GPU core-utilization sampler."""

from __future__ import annotations

import time

from horde_worker_regen.utils.gpu_monitor import GpuUtilizationSampler


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
