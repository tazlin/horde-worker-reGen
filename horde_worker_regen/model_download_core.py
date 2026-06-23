"""The single source of truth for downloading image-model checkpoints.

Owns the two pieces that were previously copied across the background download process, the standalone
``download_models`` entry point, and the benchmark's ``download`` subcommand:

  * the per-chunk **pause / bandwidth-limit** enforcement (plus speed/ETA smoothing), and
  * the per-model **validate + retry-once** download loop.

It is deliberately import-light: it operates on a duck-typed compvis-style manager (``download_model`` /
``validate_model`` / ``is_model_available``) and never imports hordelib or torch, so a caller that only needs
to fetch checkpoints does not pay for the inference stack.

Pause and rate-limit are read through callables (not a snapshot) so they apply live, mid-download: the worker
process reads its own message-driven flags, while the benchmark reads a :class:`DownloadControls` it updates
from a stdin control channel.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "ChunkPacer",
    "DownloadAborted",
    "DownloadControls",
    "DownloadOutcome",
    "ModelProgress",
    "UNKNOWN_DOWNLOAD_HOST",
    "download_host_for_url",
    "download_one_model",
    "ensure_models_present",
]

UNKNOWN_DOWNLOAD_HOST = "unknown"
"""Host bucket for downloads whose source URL has no parseable hostname.

Grouped under one conservative bucket so they serialize like a single host by default, rather than being
treated as many distinct hosts (which would let an unbounded number run in parallel)."""


def download_host_for_url(url: str | None) -> str:
    """Return the lowercased hostname of *url* for per-host download scheduling.

    The hostname (not the full URL) is the unit the scheduler parallelizes against: two files from the
    same host should be allowed to serialize (don't hammer one server) while different hosts run at once.
    A missing or unparseable host collapses to :data:`UNKNOWN_DOWNLOAD_HOST` so such downloads are paced
    conservatively as one bucket rather than spuriously fanned out.
    """
    if not url:
        return UNKNOWN_DOWNLOAD_HOST
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return UNKNOWN_DOWNLOAD_HOST
    return host.lower() if host else UNKNOWN_DOWNLOAD_HOST


class DownloadAborted(Exception):
    """Raised from inside a chunk callback to abort an in-flight download (e.g. on shutdown)."""


@dataclass
class ModelProgress:
    """Per-chunk progress for the file currently downloading."""

    downloaded_bytes: int
    total_bytes: int
    speed_bps: float | None
    eta_seconds: float | None


@dataclass
class DownloadOutcome:
    """The result of an :func:`ensure_models_present` pass."""

    downloaded: int = 0
    failed: int = 0
    present: list[str] = field(default_factory=list)
    """Models already on disk that were skipped (never re-fetched)."""
    failures: list[str] = field(default_factory=list)


class CompVisLike(Protocol):
    """The subset of a model manager the download core needs (duck-typed; hordelib's compvis satisfies it)."""

    def download_model(
        self,
        model_name: str,
        *,
        callback: Callable[[int, int], None] | None = ...,
        connections: int = ...,
    ) -> bool | None:
        """Fetch *model_name*'s files (skipping any already present); return truthy on success.

        ``connections`` is the max concurrent connections per large file (the engine segments a big file
        across that many ranged connections to raise single-file throughput; 1 keeps the single stream).
        """
        ...

    def validate_model(self, model_name: str, skip_checksum: bool = ...) -> bool | None:
        """Return whether *model_name* on disk matches its reference (checksum)."""
        ...

    def is_model_available(self, model_name: str) -> bool:
        """Return whether *model_name*'s files are all present on disk."""
        ...


class DownloadControls:
    """Thread-safe pause flag and bandwidth cap, read live by :class:`ChunkPacer`.

    The worker download process drives pause/rate from its control pipe and does not need this; it is the
    convenience the benchmark uses, updated from its stdin control channel while a download is in flight.
    """

    def __init__(self, *, paused: bool = False, rate_limit_kbps: int | None = None) -> None:
        """Initialise with an optional starting pause state and bandwidth cap (kB/s; <=0 means no cap)."""
        self._lock = threading.Lock()
        self._paused = paused
        self._rate_limit_kbps = rate_limit_kbps if (rate_limit_kbps or 0) > 0 else None

    def set_paused(self, paused: bool) -> None:
        """Pause or resume; applied to the next chunk."""
        with self._lock:
            self._paused = paused

    def set_rate_limit(self, kbps: int | None) -> None:
        """Set the bandwidth cap in kB/s (0 or negative clears it)."""
        with self._lock:
            self._rate_limit_kbps = kbps if (kbps or 0) > 0 else None

    def is_paused(self) -> bool:
        """Return whether downloads are currently paused."""
        with self._lock:
            return self._paused

    def rate_limit_kbps(self) -> int | None:
        """Return the current bandwidth cap in kB/s, or None when uncapped."""
        with self._lock:
            return self._rate_limit_kbps


class ChunkPacer:
    """Holds the per-download state needed to enforce rate-limit and compute a smoothed speed/ETA.

    One pacer is used per file download; call :meth:`step` from the download's per-chunk callback.

    Both the throttle wait and the pause wait are sliced into short polls rather than one long sleep.
    That keeps three properties a single ``time.sleep`` quietly broke: shutdown (``should_abort``) lands
    within a poll instead of after a multi-minute wait; the UI keeps refreshing mid-wait (``on_wait``);
    and a low cap against a large chunk cannot idle the connection long enough for the server to drop it,
    since the throttle wait is bounded by :data:`MAX_THROTTLE_SECONDS` (a brief rate overshoot is the
    deliberate price of a download that survives the cap).
    """

    MAX_THROTTLE_SECONDS = 5.0
    """Upper bound on a single chunk's throttle wait, so a low cap cannot stall the socket into a drop."""

    def __init__(self) -> None:
        """Start with no observed bytes, time, or speed."""
        self._last_bytes = 0
        self._last_time = 0.0
        self._speed_bps: float | None = None

    def step(
        self,
        downloaded: int,
        total: int,
        *,
        is_paused: Callable[[], bool],
        rate_limit_kbps: Callable[[], int | None],
        should_abort: Callable[[], bool],
        on_wait: Callable[[ModelProgress], None] | None = None,
        poll_seconds: float = 0.2,
    ) -> ModelProgress:
        """Pace one chunk: honour the rate cap, smooth the speed, then block while paused.

        ``on_wait`` (if given) is invoked with the current progress on each poll while the chunk is
        being throttled or held paused, so a caller can keep its display live during the wait.

        Raises:
            DownloadAborted: if ``should_abort`` is true on entry, or observed during a wait.
        """
        if should_abort():
            raise DownloadAborted

        now = time.time()

        # The first observation only establishes a baseline: there is no prior sample to rate-limit
        # against, and the first callback often carries the whole chunk already read, so pacing it would
        # freeze the transfer before a single byte of feedback (the "stuck at 0%" report).
        if self._last_time == 0.0:
            self._last_bytes = downloaded
            self._last_time = now
            progress = ModelProgress(downloaded, total, None, None)
            self._wait_while_paused(
                progress,
                is_paused=is_paused,
                should_abort=should_abort,
                on_wait=on_wait,
                poll_seconds=poll_seconds,
            )
            return progress

        delta = max(0, downloaded - self._last_bytes)
        elapsed = now - self._last_time

        rate = rate_limit_kbps()
        if rate and delta > 0:
            target = delta / (rate * 1024.0)
            sleep_for = min(max(0.0, target - elapsed), self.MAX_THROTTLE_SECONDS)
            self._sleep_in_slices(
                sleep_for,
                lambda: self._progress_at(downloaded, total),
                should_abort=should_abort,
                on_wait=on_wait,
                poll_seconds=poll_seconds,
            )
            now = time.time()
            elapsed = now - self._last_time

        if elapsed > 0 and delta > 0:
            instantaneous = delta / elapsed
            self._speed_bps = (
                instantaneous if self._speed_bps is None else (0.7 * self._speed_bps + 0.3 * instantaneous)
            )
        self._last_bytes = downloaded
        self._last_time = now

        progress = self._progress_at(downloaded, total)
        self._wait_while_paused(
            progress,
            is_paused=is_paused,
            should_abort=should_abort,
            on_wait=on_wait,
            poll_seconds=poll_seconds,
        )
        return progress

    def _progress_at(self, downloaded: int, total: int) -> ModelProgress:
        """Build a progress reading at the current byte count using the latest smoothed speed."""
        eta = (total - downloaded) / self._speed_bps if self._speed_bps and total > downloaded else None
        return ModelProgress(
            downloaded_bytes=downloaded,
            total_bytes=total,
            speed_bps=self._speed_bps,
            eta_seconds=eta,
        )

    def _sleep_in_slices(
        self,
        seconds: float,
        make_progress: Callable[[], ModelProgress],
        *,
        should_abort: Callable[[], bool],
        on_wait: Callable[[ModelProgress], None] | None,
        poll_seconds: float,
    ) -> None:
        """Sleep ``seconds`` in ``poll_seconds`` slices, checking abort and emitting heartbeats each slice."""
        remaining = seconds
        while remaining > 0:
            if should_abort():
                raise DownloadAborted
            if on_wait is not None:
                on_wait(make_progress())
            nap = min(poll_seconds, remaining)
            time.sleep(nap)
            remaining -= nap
        if should_abort():
            raise DownloadAborted

    def _wait_while_paused(
        self,
        progress: ModelProgress,
        *,
        is_paused: Callable[[], bool],
        should_abort: Callable[[], bool],
        on_wait: Callable[[ModelProgress], None] | None,
        poll_seconds: float,
    ) -> None:
        """Block while paused, emitting a heartbeat each poll; raise on abort."""
        while is_paused() and not should_abort():
            if on_wait is not None:
                on_wait(progress)
            time.sleep(poll_seconds)
        if should_abort():
            raise DownloadAborted


def download_one_model(
    compvis: CompVisLike,
    model_name: str,
    *,
    callback: Callable[[int, int], None] | None = None,
    connections: int = 1,
) -> bool:
    """Download *model_name*, re-downloading once if the on-disk checksum does not validate.

    Returns True only when the download (and any forced re-download) succeeded. ``compvis.download_model``
    already short-circuits when the files are present, so a present-and-valid model is a cheap no-op.
    ``connections`` is forwarded to the engine so a large file can be fetched over several ranged
    connections at once (1 keeps the single-stream path).
    """
    succeeded = bool(compvis.download_model(model_name, callback=callback, connections=connections))
    if succeeded and compvis.validate_model(model_name) is False:
        # The record changed or the file is corrupt: fetch it again.
        succeeded = bool(compvis.download_model(model_name, callback=callback, connections=connections))
    return succeeded


def ensure_models_present(
    compvis: CompVisLike,
    model_names: list[str],
    *,
    controls: DownloadControls | None = None,
    should_abort: Callable[[], bool] | None = None,
    on_model_start: Callable[[str, int, int], None] | None = None,
    on_progress: Callable[[str, int, int, ModelProgress], None] | None = None,
    on_model_finish: Callable[[str, int, int, bool], None] | None = None,
    connections: int = 1,
) -> DownloadOutcome:
    """Ensure every name in *model_names* is on disk, downloading the missing ones with live pause/rate-limit.

    Already-present models are skipped (and reported in :attr:`DownloadOutcome.present`); the remainder are
    downloaded one at a time, each with its own :class:`ChunkPacer`. The ``on_*`` callbacks let a caller
    surface progress (the benchmark emits structured events; the worker logs); ``index``/``total`` count the
    models being downloaded (present ones excluded), so a UI can render "k of n". ``connections`` is forwarded
    to the engine to segment each large file across that many ranged connections.
    """
    controls = controls or DownloadControls()
    should_abort = should_abort or (lambda: False)

    pending = [name for name in model_names if not compvis.is_model_available(name)]
    present = [name for name in model_names if name not in pending]
    outcome = DownloadOutcome(present=present)
    total = len(pending)

    for index, name in enumerate(pending, start=1):
        if on_model_start is not None:
            on_model_start(name, index, total)

        pacer = ChunkPacer()

        def _callback(
            downloaded: int,
            total_bytes: int,
            *,
            _pacer: ChunkPacer = pacer,
            _name: str = name,
            _index: int = index,
        ) -> None:
            def _emit(progress: ModelProgress) -> None:
                if on_progress is not None:
                    on_progress(_name, _index, total, progress)

            # ``on_wait`` keeps the heartbeat alive while a chunk is throttled or paused, so a rate-limited
            # benchmark download still ticks instead of looking frozen between chunks.
            progress = _pacer.step(
                downloaded,
                total_bytes,
                is_paused=controls.is_paused,
                rate_limit_kbps=controls.rate_limit_kbps,
                should_abort=should_abort,
                on_wait=_emit,
            )
            _emit(progress)

        succeeded = download_one_model(compvis, name, callback=_callback, connections=connections)
        if succeeded:
            outcome.downloaded += 1
        else:
            outcome.failed += 1
            outcome.failures.append(name)
        if on_model_finish is not None:
            on_model_finish(name, index, total, succeeded)

    return outcome
