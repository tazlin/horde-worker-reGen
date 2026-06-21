"""Unit tests for the shared, warm-once model catalog cache."""

from __future__ import annotations

import threading
import time

import pytest

from horde_worker_regen.tui.catalog_cache import ModelCatalogCache
from horde_worker_regen.tui.model_catalog import ModelInfo

_MODELS = [ModelInfo("Deliberate", "stable_diffusion_1", nsfw=False, inpainting=False)]


def _install_counters(cache: ModelCatalogCache, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Patch the cache's loaders with call-counting stubs and return the counter dict."""
    counts = {"catalog": 0, "free": 0, "popularity": 0}

    def _load() -> list[ModelInfo]:
        counts["catalog"] += 1
        return list(_MODELS)

    def _free() -> int:
        counts["free"] += 1
        return 123

    def _popularity() -> dict[str, int]:
        counts["popularity"] += 1
        return {"Deliberate": 7}

    monkeypatch.setattr("horde_worker_regen.tui.catalog_cache.load_image_models", _load)
    monkeypatch.setattr("horde_worker_regen.tui.catalog_cache.free_model_bytes", _free)
    monkeypatch.setattr("horde_worker_regen.tui.catalog_cache.fetch_model_popularity", _popularity)
    return counts


def test_snapshot_is_empty_before_loading() -> None:
    """A fresh cache reports nothing loaded and never touches the network for a snapshot."""
    cache = ModelCatalogCache()
    snapshot = cache.snapshot()
    assert snapshot.is_loaded is False
    assert snapshot.catalog is None
    assert snapshot.loading is False


def test_loads_once_then_serves_from_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """The catalog is fetched once; later calls reuse it without re-fetching."""
    cache = ModelCatalogCache()
    counts = _install_counters(cache, monkeypatch)

    first = cache.ensure_loaded()
    assert first.is_loaded and first.catalog == _MODELS
    assert first.free_disk_bytes == 123

    cache.ensure_loaded()
    assert counts["catalog"] == 1  # not re-fetched


def test_popularity_only_fetched_on_demand(monkeypatch: pytest.MonkeyPatch) -> None:
    """Usage stats are fetched only when asked for, then cached."""
    cache = ModelCatalogCache()
    counts = _install_counters(cache, monkeypatch)

    cache.ensure_loaded()
    assert counts["popularity"] == 0

    snapshot = cache.ensure_loaded(want_popularity=True)
    assert snapshot.popularity == {"Deliberate": 7}
    assert counts["popularity"] == 1

    cache.ensure_loaded(want_popularity=True)
    assert counts["popularity"] == 1  # served from cache


def test_force_reloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """A forced load re-fetches the catalog even when already cached."""
    cache = ModelCatalogCache()
    counts = _install_counters(cache, monkeypatch)

    cache.ensure_loaded()
    cache.ensure_loaded(force=True)
    assert counts["catalog"] == 2


def test_concurrent_callers_coalesce(monkeypatch: pytest.MonkeyPatch) -> None:
    """Many threads opening views at once collapse onto a single network fetch."""
    cache = ModelCatalogCache()
    counts = {"catalog": 0}

    def _slow_load() -> list[ModelInfo]:
        counts["catalog"] += 1
        time.sleep(0.2)
        return list(_MODELS)

    monkeypatch.setattr("horde_worker_regen.tui.catalog_cache.load_image_models", _slow_load)
    monkeypatch.setattr("horde_worker_regen.tui.catalog_cache.free_model_bytes", lambda: 0)

    threads = [threading.Thread(target=cache.ensure_loaded) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert counts["catalog"] == 1
    assert cache.snapshot().catalog == _MODELS
