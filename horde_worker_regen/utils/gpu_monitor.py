"""Background GPU core-utilization sampling for the benchmark.

The benchmark measures *sampling rate* (it/s) while a job is on the GPU, but that says
nothing about how much of the wall clock the GPU is actually busy -- the gaps *between*
jobs (VAE decode, post-processing, encode, IPC hand-off, scheduling latency) are exactly
where uptime is lost. This sampler polls the device's core-utilization percentage on a
background thread for the duration of a run, so the report can state a GPU duty cycle.

Utilization is read through hordelib's backend-agnostic accelerator helper
(:func:`hordelib.utils.torch_memory.get_accelerator_utilization_percent`), which returns the figure
for whatever backend can report it (NVIDIA via NVML today) and ``None`` elsewhere. It is imported from
that torch-free submodule rather than the ``hordelib.api`` facade so merely importing this module pulls
no torch. The worker itself never touches NVML/``pynvml`` directly, so it makes no NVIDIA assumption.

It degrades gracefully: when no backend telemetry is available the sampler collects no
samples and reports ``None``, so CPU/fake runs, non-NVIDIA backends, and CI are unaffected.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

from loguru import logger


def _make_utilization_reader(device_index: int) -> Callable[[], int | None] | None:
    """Return a callable reading device core-utilization %, or None when no telemetry source exists.

    Delegates to hordelib's backend-agnostic API so this works on whatever backend reports utilization and
    yields no sampler elsewhere. Probes once: a backend (or machine) with no utilization source returns
    ``None`` here, so the caller skips sampling entirely rather than spinning a thread that only collects
    ``None``.
    """
    try:
        from hordelib.utils.torch_memory import get_accelerator_utilization_percent
    except Exception as import_error:  # noqa: BLE001 - "no telemetry" is expected off-GPU, not a crash
        logger.debug(f"GPU utilization sampling unavailable (hordelib import failed: {import_error})")
        return None

    def _read() -> int | None:
        return get_accelerator_utilization_percent(device_index)

    if _read() is None:
        logger.debug("GPU utilization sampling unavailable (no backend telemetry for this device)")
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
            device_index: Which device to watch.
            interval_seconds: How often to poll utilization.
            busy_threshold_percent: Utilization at or above which a sample counts as "busy".
                Defaults low (5%) so ``busy_fraction`` measures the fraction of wall-clock the
                GPU was doing *anything*; ``mean_percent`` remains the duty-cycle headline.
            read_utilization: Override the utilization reader (for tests). When None, a backend-agnostic
                reader is created at ``start()``; if no telemetry is available the sampler no-ops.
        """
        self._device_index = device_index
        self._interval = interval_seconds
        self._busy_threshold = busy_threshold_percent
        self._read = read_utilization

        self._samples: list[int] = []
        self._timeline: list[tuple[float, int]] = []
        """``(epoch_seconds, util_percent)`` per sample, for offline correlation with spans."""
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Begin sampling on a background thread (a no-op when no GPU telemetry is available)."""
        if self._read is None:
            self._read = _make_utilization_reader(self._device_index)
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
        """Stop sampling. NVML's lifecycle is owned by hordelib, so there is nothing to release here."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def sample_count(self) -> int:
        """The number of utilization samples collected."""
        return len(self._samples)

    def mean_percent(self) -> float | None:
        """Return the average GPU core utilization across the run (the duty cycle), or None without samples."""
        if not self._samples:
            return None
        return sum(self._samples) / len(self._samples)

    def busy_fraction(self) -> float | None:
        """Return the fraction of samples at or above the busy threshold, or None without samples."""
        if not self._samples:
            return None
        busy = sum(1 for sample in self._samples if sample >= self._busy_threshold)
        return busy / len(self._samples)

    def timeline(self) -> list[tuple[float, int]]:
        """Return the collected ``(epoch_seconds, util_percent)`` samples, in order."""
        return list(self._timeline)

    def dump_timeline(self, path: str | Path) -> None:
        """Write the timestamped utilization series to ``path`` as JSON (diagnostics)."""
        try:
            Path(path).write_text(json.dumps(self._timeline), encoding="utf-8")
        except OSError as write_error:  # noqa: BLE001 - a diagnostics dump must never break a run
            logger.debug(f"Could not write GPU utilization timeline to {path}: {write_error}")
