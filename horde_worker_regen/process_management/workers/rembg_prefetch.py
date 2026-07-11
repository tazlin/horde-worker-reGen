"""Pre-place the rembg ``u2net.onnx`` weight into the image-utilities lane's rembg cache directory.

The image-utilities capability service runs with downloads disabled (``HIU_ALLOW_DOWNLOADS=false``), so it
never fetches the background-removal weight itself: if the file is absent, ``strip_background`` faults in
the service with a missing-model error. The worker's download process therefore pre-places the weight where
the service looks for it, exactly as it ensures the safety/post-processing models on disk.

The cache directory mirrors the image-utilities package's ``get_rembg_cache_dir`` derivation
(``AIWORKER_CACHE_HOME/horde/image-utilities/rembg``) without importing that package: the download process
is torch-free and stdlib-only, and importing the utilities package (which pulls its native stack) into it
would be inappropriate. The path segments are a stable convention shared by both sides.

The weight is fetched from its canonical rembg release asset and verified against the checksum rembg itself
publishes for it. rembg publishes an MD5 (via its ``pooch`` retrieve call), so MD5 is what is verified here;
if a future rembg version publishes a SHA-256 the constant below should be updated to match.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

from loguru import logger

_REMBG_CACHE_SEGMENTS = ("horde", "image-utilities", "rembg")
"""The path segments appended to ``AIWORKER_CACHE_HOME`` for the isolated rembg cache.

These mirror ``horde_image_utilities.config.CachePathSegments`` (ROOT_NAMESPACE / PROJECT_NAME / REMBG_DIR)
so the pre-placed weight lands exactly where the capability service resolves ``U2NET_HOME`` to."""

U2NET_FILENAME = "u2net.onnx"
"""The default rembg background-removal model file; the service resolves ``<cache>/u2net.onnx``."""

U2NET_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
"""The canonical rembg release asset for the u2net model. The worker downloads from here; it does not vendor."""

U2NET_MD5 = "60024c5c889badc19c04ad937298a77b"
"""The MD5 rembg publishes for ``u2net.onnx`` (its ``pooch`` retrieve checksum), verified after download.

MD5 (not SHA-256) because that is the digest rembg itself pins the asset to; a mismatch rejects a
truncated/tampered download rather than handing the service a corrupt weight."""


def rembg_cache_dir() -> Path | None:
    """Return the isolated rembg cache directory, or None when the cache home is unset.

    Reads ``AIWORKER_CACHE_HOME`` from the process environment (the download process inherits it from the
    parent) and appends the shared segments. None when the cache home is unset, in which case the caller
    skips the pre-place (there is no isolated cache to populate).
    """
    cache_home = os.environ.get("AIWORKER_CACHE_HOME")
    if not cache_home:
        return None
    return Path(cache_home, *_REMBG_CACHE_SEGMENTS)


def _md5_of(path: Path) -> str:
    """Return the hex MD5 of a file, streamed in 1 MiB chunks."""
    digest = hashlib.md5()  # noqa: S324 - matching rembg's published digest, not a security boundary
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def u2net_present() -> bool:
    """Return whether a checksum-valid ``u2net.onnx`` is already in the isolated rembg cache."""
    cache_dir = rembg_cache_dir()
    if cache_dir is None:
        return False
    weight_path = cache_dir / U2NET_FILENAME
    return weight_path.is_file() and _md5_of(weight_path) == U2NET_MD5


def ensure_u2net_present() -> Path | None:
    """Return the path to the cached ``u2net.onnx``, downloading and verifying it once if absent.

    A previously-present file is re-verified by MD5 on every call, so a truncated earlier download is
    re-fetched rather than trusted. Returns None when the cache home is unset (nothing to populate). Raises
    on a checksum mismatch after download so a corrupt upstream file faults loudly rather than reaching the
    service silently wrong.
    """
    cache_dir = rembg_cache_dir()
    if cache_dir is None:
        logger.debug("AIWORKER_CACHE_HOME is unset; skipping rembg u2net pre-place.")
        return None

    weight_path = cache_dir / U2NET_FILENAME
    if weight_path.is_file() and _md5_of(weight_path) == U2NET_MD5:
        return weight_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Pre-placing rembg background-removal weight from {U2NET_URL} into {cache_dir}")
    temp_path = weight_path.with_suffix(weight_path.suffix + ".partial")
    urllib.request.urlretrieve(U2NET_URL, temp_path)  # noqa: S310 - fixed canonical https URL

    actual = _md5_of(temp_path)
    if actual != U2NET_MD5:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"rembg u2net.onnx checksum mismatch: expected md5 {U2NET_MD5}, got {actual}")
    temp_path.replace(weight_path)
    return weight_path
