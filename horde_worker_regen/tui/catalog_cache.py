"""Process-wide cache of the image-model catalog for the TUI, warmed once and shared by every view.

Loading the model reference (and the usage stats behind ``top N`` / ``bottom N``) is a network round
trip that takes seconds. Each view used to trigger its own load when it opened, so opening the picker or
the Models config panel felt sluggish and re-fetched what another view had already loaded. This module
centralises that work: the app warms the cache once at startup, and every view reads the same in-memory
snapshot instantly, refreshing only on demand. All loads funnel through one lock so concurrent openers
coalesce onto a single fetch instead of stampeding the network.

Reads (:meth:`ModelCatalogCache.snapshot`) take a short-lived lock and never block on the network, so
they are safe to call from the UI thread. The blocking load (:meth:`ModelCatalogCache.ensure_loaded`)
must be called from a worker thread; it is what the views run off-thread and what the app warms with.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from horde_model_reference import resolve_weights_root

from horde_worker_regen.model_download_plan import free_model_bytes
from horde_worker_regen.tui.model_catalog import ModelInfo, fetch_model_popularity, load_image_models


@dataclass(frozen=True)
class CatalogSnapshot:
    """An immutable view of the cache at one instant, safe to read on the UI thread."""

    catalog: list[ModelInfo] | None
    """The loaded image-model catalog, or None when it has not been loaded yet."""
    free_disk_bytes: int | None
    """Free space on the model volume as of the last catalog load, or None when undetermined."""
    weights_root: str | None
    """The resolved models directory the disk figures describe (so the operator can verify the volume)."""
    popularity: dict[str, int] | None
    """Model-name -> last-month usage count, loaded only once a ``top``/``bottom N`` resolve needs it."""
    loaded_at: float | None
    """Epoch seconds of the last successful load, or None when never loaded."""
    loading: bool
    """Whether a load is in flight right now (so a view can show a 'loading…' affordance)."""

    @property
    def is_loaded(self) -> bool:
        """Whether the catalog itself is available (popularity may still be absent)."""
        return self.catalog is not None


class ModelCatalogCache:
    """A thread-safe, lazily-populated cache of the image-model catalog and usage stats."""

    def __init__(self) -> None:
        """Start empty; nothing is fetched until :meth:`ensure_loaded` is called (by the warm or a view)."""
        self._data_lock = threading.Lock()
        """Guards the cached fields; held only momentarily, never across the network fetch."""
        self._load_lock = threading.Lock()
        """Serialises fetches so concurrent openers coalesce onto a single network round trip."""
        self._catalog: list[ModelInfo] | None = None
        self._free_disk_bytes: int | None = None
        self._weights_root: str | None = None
        self._popularity: dict[str, int] | None = None
        self._loaded_at: float | None = None
        self._loading = False

    def _snapshot_locked(self) -> CatalogSnapshot:
        """Build a snapshot. Caller must hold ``_data_lock``."""
        return CatalogSnapshot(
            catalog=self._catalog,
            free_disk_bytes=self._free_disk_bytes,
            weights_root=self._weights_root,
            popularity=self._popularity,
            loaded_at=self._loaded_at,
            loading=self._loading,
        )

    def snapshot(self) -> CatalogSnapshot:
        """Return the current cache contents without ever touching the network (UI-thread safe)."""
        with self._data_lock:
            return self._snapshot_locked()

    def ensure_loaded(self, *, force: bool = False, want_popularity: bool = False) -> CatalogSnapshot:
        """Load whatever is missing and return the resulting snapshot. Blocking; call off the UI thread.

        Args:
            force: Reload the catalog (and, with ``want_popularity``, the stats) even if already cached.
            want_popularity: Also fetch usage stats, needed only to expand ``top``/``bottom N`` commands.

        Concurrent callers coalesce: one holds the load lock and fetches while the others wait and then
        observe the freshly-populated cache. Raises on fetch failure so the caller can surface it.
        """
        with self._data_lock:
            have_catalog = self._catalog is not None
            have_popularity = self._popularity is not None
            if not force and have_catalog and (not want_popularity or have_popularity):
                return self._snapshot_locked()

        with self._load_lock:
            # Re-check under the load lock: another loader may have populated the cache while we waited.
            with self._data_lock:
                need_catalog = force or self._catalog is None
                need_popularity = want_popularity and (force or self._popularity is None)
                if not need_catalog and not need_popularity:
                    return self._snapshot_locked()
                self._loading = True

            try:
                catalog = load_image_models() if need_catalog else None
                free = free_model_bytes() if need_catalog else None
                root = str(resolve_weights_root()) if need_catalog else None
                popularity = fetch_model_popularity() if need_popularity else None
            except Exception:
                with self._data_lock:
                    self._loading = False
                raise

            with self._data_lock:
                if catalog is not None:
                    self._catalog = catalog
                    self._free_disk_bytes = free
                    self._weights_root = root
                if popularity is not None:
                    self._popularity = popularity
                self._loaded_at = time.time()
                self._loading = False
                return self._snapshot_locked()

    def reset(self) -> None:
        """Drop all cached data (used by tests to isolate the process-wide singleton)."""
        with self._data_lock:
            self._catalog = None
            self._free_disk_bytes = None
            self._weights_root = None
            self._popularity = None
            self._loaded_at = None
            self._loading = False


CATALOG_CACHE = ModelCatalogCache()
"""The single, process-wide catalog cache shared by every TUI view."""
