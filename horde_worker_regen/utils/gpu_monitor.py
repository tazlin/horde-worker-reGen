"""Background GPU core-utilization sampling for the benchmark.

The benchmark measures *sampling rate* (it/s) while a job is on the GPU, but that says
nothing about how much of the wall clock the GPU is actually busy — the gaps *between*
jobs (VAE decode, post-processing, encode, IPC hand-off, scheduling latency) are exactly
where uptime is lost. This sampler polls NVML for the device's core-utilization percentage
on a background thread for the duration of a run, so the report can state a GPU duty cycle.

It degrades gracefully: if NVML (``pynvml``) is unavailable, the sampler simply collects no
samples and reports ``None``, so fake/CPU runs and CI are unaffected.
"""

from __future__ import annotations

import json
import threading
import time
import warnings
from collections.abc import Callable
from pathlib import Path

from loguru import logger


def _make_nvml_reader(device_index: int) -> Callable[[], int | None] | None:
    """Return a callable that reads device core-utilization %, or None if NVML is unavailable."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence pynvml's deprecation FutureWarning
            import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    except Exception as e:  # noqa: BLE001 - any NVML failure means "no GPU telemetry", not a crash
        logger.debug(f"GPU utilization sampling unavailable (NVML init failed: {e})")
        return None

    def _read() -> int | None:
        try:
            return int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        except Exception:  # noqa: BLE001
            return None

    return _read


class GpuUtilizationSampler:
    """Polls GPU core utilization on a background thread between ``start()`` and ``stop()``."""

    def __init__(
        self,
        *,
        device_index: int = 0,
        interval_seconds: float = 0.1,
        busy_threshold_percent: int = 5,
        read_utilization: Callable[[], int | None] | None = None,
    ) -> None:
        """Initialize the sampler.

        Args:
            device_index: Which CUDA device to watch.
            interval_seconds: How often to poll utilization.
            busy_threshold_percent: Utilization at or above which a sample counts as "busy".
                Defaults low (5%) so ``busy_fraction`` measures the fraction of wall-clock the
                GPU was doing *anything*; ``mean_percent`` remains the duty-cycle headline.
            read_utilization: Override the utilization reader (for tests). When None, an NVML
                reader is created at ``start()``; if NVML is unavailable the sampler no-ops.
        """
        self._device_index = device_index
        self._interval = interval_seconds
        self._busy_threshold = busy_threshold_percent
        self._read = read_utilization
        self._owns_reader = read_utilization is None

        self._samples: list[int] = []
        self._timeline: list[tuple[float, int]] = []
        """``(epoch_seconds, util_percent)`` per sample, for offline correlation with spans."""
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Begin sampling on a background thread (a no-op if no GPU telemetry is available)."""
        if self._read is None:
            self._read = _make_nvml_reader(self._device_index)
        if self._read is None:
            return
        self._thread = threading.Thread(target=self._loop, name="gpu-util-sampler", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self._read is not None
        while not self._stop_event.wait(self._interval):
            value = self._read()
            if value is not None:
                self._samples.append(value)
                self._timeline.append((time.time(), value))

    def stop(self) -> None:
        """Stop sampling and release NVML."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._owns_reader and self._read is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    import pynvml

                    pynvml.nvmlShutdown()
                except Exception:  # noqa: BLE001
                    pass

    @property
    def sample_count(self) -> int:
        """How many utilization samples were collected."""
        return len(self._samples)

    def mean_percent(self) -> float | None:
        """Average GPU core utilization across the run (the duty cycle), or None without samples."""
        if not self._samples:
            return None
        return sum(self._samples) / len(self._samples)

    def busy_fraction(self) -> float | None:
        """Fraction of samples at or above the busy threshold, or None without samples."""
        if not self._samples:
            return None
        busy = sum(1 for sample in self._samples if sample >= self._busy_threshold)
        return busy / len(self._samples)

    def timeline(self) -> list[tuple[float, int]]:
        """The collected ``(epoch_seconds, util_percent)`` samples, in order."""
        return list(self._timeline)

    def dump_timeline(self, path: str | Path) -> None:
        """Write the timestamped utilization series to ``path`` as JSON (diagnostics)."""
        try:
            Path(path).write_text(json.dumps(self._timeline), encoding="utf-8")
        except OSError as e:  # noqa: BLE001 - a diagnostics dump must never break a run
            logger.debug(f"Could not write GPU utilization timeline to {path}: {e}")
