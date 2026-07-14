"""Background GPU core-utilization sampling for the benchmark.

The benchmark measures *sampling rate* (it/s) while a job is on the GPU, but that says
nothing about how much of the wall clock the GPU is actually busy; the gaps *between*
jobs (VAE decode, post-processing, encode, IPC hand-off, scheduling latency) are exactly
where uptime is lost. This sampler polls the device's core-utilization percentage on a
background thread for the duration of a run, so the report can state a GPU duty cycle.

Utilization is read through hordelib's NVML helper
(:func:`hordelib.utils.nvml.get_device_utilization_percent`), which returns the NVIDIA figure and ``None``
on any non-NVIDIA host. This sampler runs in the *orchestrator* process, which must stay torch-free (see
the torch-free orchestrator invariant): the backend-agnostic ``get_accelerator_utilization_percent`` gates
on the active torch backend and so does ``import torch``, pulling torch into the parent and tripping a
partial-init circular import. Core utilization is NVML-only telemetry today regardless (CUDA via NVML;
every other backend reports ``None``), so reading NVML directly is behaviourally identical here while
keeping the parent torch-free. NVML returns ``None`` off NVIDIA, so non-NVIDIA backends still report no
duty cycle rather than erroring.

It degrades gracefully: when no backend telemetry is available the sampler collects no
samples and reports ``None``, so CPU/fake runs, non-NVIDIA backends, and CI are unaffected.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

from loguru import logger


def _make_utilization_reader(device_index: int) -> Callable[[], int | None] | None:
    """Return a callable reading device core-utilization %, or None when no telemetry source exists.

    Reads NVML directly (the only utilization source today) so the orchestrator never imports torch. Probes
    once: a host with no NVML/utilization source returns ``None`` here, so the caller skips sampling entirely
    rather than spinning a thread that only collects ``None``.
    """
    try:
        from hordelib.utils.nvml import get_device_utilization_percent
    except Exception as import_error:  # noqa: BLE001 - "no telemetry" is expected off-GPU, not a crash
        logger.debug(f"GPU utilization sampling unavailable (hordelib import failed: {import_error})")
        return None

    def _read() -> int | None:
        return get_device_utilization_percent(device_index)

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
        max_samples: int = 100_000,
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
            max_samples: Cap on retained samples so a long-lived worker (sampling for days) cannot grow
                the buffers without bound. The benchmark runs only for a level, so this never bites there;
                in the worker it bounds memory while still covering any rolling window the logs query.
        """
        self._device_index = device_index
        self._interval = interval_seconds
        self._busy_threshold = busy_threshold_percent
        self._read = read_utilization

        self._samples: deque[int] = deque(maxlen=max_samples)
        self._timeline: deque[tuple[float, int]] = deque(maxlen=max_samples)
        """``(epoch_seconds, util_percent)`` per sample, for rolling-window queries and offline correlation."""
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
        """The number of utilization samples currently retained."""
        return len(self._samples)

    def _windowed_values(self, window_seconds: float | None, *, not_before: float | None = None) -> list[int]:
        """Utilization values from the last ``window_seconds`` (whole buffer when None).

        ``not_before`` drops any sample taken before that epoch second. The worker passes the first
        inference's start time so the cold-boot model-load window (GPU idle while weights stream in,
        before any job samples) never dilutes the duty figure: that startup time is not inter-job
        inefficiency, just one-time warm-up, and counting it only adds noise to the headline.
        """
        if window_seconds is None and not_before is None:
            return list(self._samples)
        cutoff = time.time() - window_seconds if window_seconds is not None else None
        floor = max(cutoff, not_before) if cutoff is not None and not_before is not None else (cutoff or not_before)
        return [value for timestamp, value in self._timeline if floor is None or timestamp >= floor]

    def mean_percent(self, window_seconds: float | None = None, *, not_before: float | None = None) -> float | None:
        """Average GPU core utilization (the duty cycle), over the whole run or the last window.

        Args:
            window_seconds: When given, average only samples from the last this-many seconds (for a
                live rolling figure); when None, average the whole retained run (the benchmark's use).
            not_before: When given, exclude samples taken before this epoch second (e.g. the first
                inference start, so cold-boot warm-up is not counted against the duty cycle).
        """
        values = self._windowed_values(window_seconds, not_before=not_before)
        if not values:
            return None
        return sum(values) / len(values)

    def busy_fraction(self, window_seconds: float | None = None, *, not_before: float | None = None) -> float | None:
        """Fraction of samples at or above the busy threshold, over the whole run or the last window."""
        values = self._windowed_values(window_seconds, not_before=not_before)
        if not values:
            return None
        busy = sum(1 for value in values if value >= self._busy_threshold)
        return busy / len(values)

    def timeline(self) -> list[tuple[float, int]]:
        """Return the collected ``(epoch_seconds, util_percent)`` samples, in order."""
        return list(self._timeline)

    def dump_timeline(self, path: str | Path) -> None:
        """Write the timestamped utilization series to ``path`` as JSON (diagnostics)."""
        try:
            Path(path).write_text(json.dumps(self.timeline()), encoding="utf-8")
        except OSError as write_error:  # noqa: BLE001 - a diagnostics dump must never break a run
            logger.debug(f"Could not write GPU utilization timeline to {path}: {write_error}")
