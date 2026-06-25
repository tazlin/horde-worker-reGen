"""A torch-free listing of the model files on disk, for "is model X present / how big / disk full?".

Many worker problems are really storage problems: a half-downloaded checkpoint, a model the operator
thinks is installed but is not, or a full disk that silently fails writes. A maintainer can answer those
in seconds given a listing of what is actually on disk. This walks the model-cache directory and reports
model-like files (name + size), the totals, and the volume's free space.

Deliberately a plain filesystem walk: no hordelib/model-reference resolution (which would pull heavy,
possibly torch-touching, machinery). It answers presence and size, not semantic validity.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

# File suffixes that denote model weights worth listing; everything else (configs, locks, json) is noise.
_MODEL_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx", ".pte"}
# Cap the listing so a cache with a pathological number of model files cannot bloat the bundle.
_MAX_ENTRIES = 5000
# Cap the *walk* itself: the model cache can be a multi-hundred-GB tree on a slow disk (a deep HF-hub
# cache of many tiny files), and an unbounded recursive walk made bundle generation take a minute. We
# visit at most this many filesystem entries, which is plenty to find the weight files, then stop.
_MAX_SCANNED = 40000


def collect_cache_inventory(cache_home: str | None) -> dict[str, Any]:
    """List model-like files under ``cache_home`` with sizes, totals, and the volume's free space.

    Returns a ``present: False`` stub when the cache directory is unknown or missing, so the bundle still
    records *that* (an unset/inaccessible cache is itself a useful signal).
    """
    if not cache_home:
        return {"cache_home": None, "present": False, "reason": "cache_home not resolved"}
    root = Path(cache_home)
    if not root.is_dir():
        return {"cache_home": cache_home, "present": False, "reason": "directory does not exist"}

    files: list[dict[str, Any]] = []
    model_count = 0
    total_bytes = 0
    visited = 0
    walk_truncated = False
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            visited += 1
            if visited > _MAX_SCANNED:
                walk_truncated = True
                break
            if Path(filename).suffix.lower() not in _MODEL_SUFFIXES:
                continue
            entry = Path(dirpath) / filename
            try:
                size = entry.stat().st_size
            except OSError:
                continue
            model_count += 1
            total_bytes += size
            if len(files) < _MAX_ENTRIES:
                files.append({"path": str(entry.relative_to(root)), "size_bytes": size})
        if walk_truncated:
            break

    truncated = walk_truncated or model_count > len(files)
    files.sort(key=lambda item: item["size_bytes"], reverse=True)
    disk = _disk_usage(root)
    return {
        "cache_home": cache_home,
        "present": True,
        "model_file_count": model_count,
        "total_model_bytes": total_bytes,
        "free_bytes": disk["free"] if disk else None,
        "volume_total_bytes": disk["total"] if disk else None,
        "listing_truncated": truncated,
        "files": files,
    }


def _disk_usage(path: Path) -> dict[str, int] | None:
    """Free/total bytes for the volume holding ``path``, or None if it cannot be read."""
    try:
        usage = shutil.disk_usage(path)
        return {"free": usage.free, "total": usage.total}
    except OSError:
        return None
