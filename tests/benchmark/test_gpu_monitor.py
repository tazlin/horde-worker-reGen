"""Tests for the GPU core-utilization sampler."""

from __future__ import annotations

import sys
import time

import pytest

from horde_worker_regen.utils.gpu_monitor import GpuUtilizationSampler, _make_utilization_reader


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
