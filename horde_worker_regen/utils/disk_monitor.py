"""Disk-space sampling for the main process control loop.

The worker writes model downloads, lora caches, and temporary artifacts to disk but has
never checked free space; a full disk surfaces as opaque download or write failures.
The monitor samples ``shutil.disk_usage`` over the relevant paths, tracks the low-water
mark, mirrors the figures to logfire, and warns (rate-limited) below a floor.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import logfire
from loguru import logger

DEFAULT_WARN_FLOOR_BYTES = 20 * 1024**3
"""Warn when free space drops below this (20 GB ~ one large model + working room)."""

_WARN_INTERVAL_SECONDS = 300.0

disk_free_gauge = logfire.metric_gauge(
    "disk.free_bytes",
    unit="By",
    description="Free disk space on monitored paths",
)


class DiskSpaceMonitor:
    """Samples free disk space for a set of paths and tracks low-water marks."""

    def __init__(self, paths: list[Path], *, warn_floor_bytes: int = DEFAULT_WARN_FLOOR_BYTES) -> None:
        """Initialize the monitor.

        Args:
            paths: The directories to watch (e.g. the model cache dir and the working dir).
                Paths on the same volume are deduplicated by their disk-usage results.
            warn_floor_bytes: Free-space floor below which a warning is logged.
        """
        self._paths = [path.resolve() for path in paths]
        self._warn_floor_bytes = warn_floor_bytes
        self._last_warned: dict[str, float] = {}

        self.current_free_bytes: dict[str, int] = {}
        """Most recent free-space sample per path."""
        self.min_free_bytes: dict[str, int] = {}
        """Lowest free-space figure observed per path since startup."""

    def sample(self) -> dict[str, int]:
        """Sample all monitored paths, update low-water marks, and return current free bytes."""
        for path in self._paths:
            key = str(path)
            try:
                free = shutil.disk_usage(path).free
            except OSError as e:
                logger.warning(f"Failed to sample disk space for {path}: {e}")
                continue

            self.current_free_bytes[key] = free
            previous_min = self.min_free_bytes.get(key)
            if previous_min is None or free < previous_min:
                self.min_free_bytes[key] = free

            disk_free_gauge.set(free, {"path": key})

            if free < self._warn_floor_bytes and (
                time.time() - self._last_warned.get(key, 0.0) > _WARN_INTERVAL_SECONDS
            ):
                self._last_warned[key] = time.time()
                logger.warning(
                    f"Low disk space on {key}: {free / 1024**3:.1f} GB free "
                    f"(floor: {self._warn_floor_bytes / 1024**3:.1f} GB). "
                    "Model downloads and result writes may start failing.",
                )

        return dict(self.current_free_bytes)
