"""Resumable, checksum-verified fetch of the DeepDanbooru safety weight, with live progress.

``horde_safety.download_deep_danbooru_model`` writes straight to the final ``.pt`` path with no range
resume and no progress hook. Three consequences follow, and this module exists to remove all three:

* An interrupted transfer leaves a truncated file at the real filename, and the next attempt deletes it
  and restarts from zero. On a link slow enough that ~640 MB outlasts one worker session, the file can
  never complete no matter how many times the worker retries.
* A truncated ``.pt`` is indistinguishable from a complete one to an existence probe, so the worker
  reports the safety models present and starts the safety process, which fails deserializing a zip
  archive that has no central directory.
* The transfer's only feedback is a tqdm bar on the download process's stderr, so the operator-facing
  download view shows a named download with no bytes, no rate, and no estimate for as long as it runs.

The asset URL and digest stay owned by ``horde_safety``; this module reads both from it rather than
restating them, so an upstream change to either is picked up rather than silently diverged from. What is
added here is the transport: an HTTP range resume onto the partial file, per-chunk progress through the
caller's callback, and a marker file recording the byte size of a hash-verified download so a later
presence probe can distinguish a complete file from a partial one without re-hashing it every start.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import requests
from loguru import logger

__all__ = [
    "DEEP_DANBOORU_MARKER_SUFFIX",
    "deep_danbooru_model_path",
    "deep_danbooru_source_url",
    "deep_danbooru_verified_on_disk",
    "ensure_deep_danbooru_present",
]

DEEP_DANBOORU_MARKER_SUFFIX = ".verified"
"""Suffix of the sidecar recording the byte size of a hash-verified weight, beside the weight itself."""

_CHUNK_BYTES = 1024 * 256
"""Read size per chunk. Large enough that the per-chunk callback is not the bottleneck on a fast link,
small enough that a paused or aborting download notices within a fraction of a second."""

_CONNECT_TIMEOUT_SECONDS = 30.0
_READ_TIMEOUT_SECONDS = 120.0
"""Socket timeouts. The read timeout is generous because the failure this module is written for is a slow
link, and a stalled-but-alive connection must be distinguishable from one that is merely crawling."""


def deep_danbooru_model_path() -> Path:
    """Return the on-disk path horde_safety resolves the DeepDanbooru weight to."""
    from horde_safety.deep_danbooru_model import default_deep_danbooru_model_path

    return Path(default_deep_danbooru_model_path)


def deep_danbooru_source_url() -> str:
    """Return the release-asset URL horde_safety fetches the DeepDanbooru weight from."""
    from horde_safety.deep_danbooru_model import model_url

    return str(model_url)


def _marker_path(weight_path: Path) -> Path:
    return weight_path.with_name(weight_path.name + DEEP_DANBOORU_MARKER_SUFFIX)


def _read_marker(weight_path: Path) -> int | None:
    """Return the byte size a prior verified download recorded, or None when there is no usable marker."""
    try:
        raw = _marker_path(weight_path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _write_marker(weight_path: Path, size_bytes: int) -> None:
    try:
        _marker_path(weight_path).write_text(str(size_bytes), encoding="utf-8")
    except OSError as e:
        # The marker is an optimisation over re-hashing, never a correctness boundary: a worker that
        # cannot write it pays one hash per start instead of failing the download it just completed.
        logger.warning(f"Could not record the verified DeepDanbooru marker: {type(e).__name__} {e}")


def deep_danbooru_verified_on_disk() -> bool:
    """Return whether the weight on disk is a complete, previously hash-verified file.

    Reads the marker rather than the file: hashing ~640 MB on every presence probe would add seconds to
    each start for a fact that does not change. A file whose size no longer matches the marker (a
    truncated re-download, a partial write) reads as unverified, which is what routes it back through
    :func:`ensure_deep_danbooru_present` for a resume rather than being trusted by the safety process.
    """
    weight_path = deep_danbooru_model_path()
    recorded = _read_marker(weight_path)
    if recorded is None:
        return False
    try:
        return weight_path.stat().st_size == recorded
    except OSError:
        return False


def _verify_hash(weight_path: Path) -> bool:
    """Return whether the file matches the digest horde_safety publishes for this asset."""
    from horde_safety.deep_danbooru_model import verify_deep_danbooru_model_hash

    return bool(verify_deep_danbooru_model_hash(weight_path))


def _accept_verified(weight_path: Path) -> bool:
    """Hash the file and, when it matches, record its size so later probes need no hash. Return the match."""
    if not _verify_hash(weight_path):
        return False
    _write_marker(weight_path, weight_path.stat().st_size)
    return True


def _open_transfer(url: str, from_byte: int) -> requests.Response | None:
    """Open the (optionally ranged) streaming GET, or None when the server rejects the range as past the end."""
    headers = {"Range": f"bytes={from_byte}-"} if from_byte > 0 else {}
    response = requests.get(
        url,
        stream=True,
        headers=headers,
        timeout=(_CONNECT_TIMEOUT_SECONDS, _READ_TIMEOUT_SECONDS),
    )
    if from_byte > 0 and response.status_code == requests.codes.requested_range_not_satisfiable:
        response.close()
        return None
    return response


def ensure_deep_danbooru_present(
    *,
    callback: Callable[[int, int], None] | None = None,
) -> None:
    """Place a hash-verified DeepDanbooru weight on disk, resuming a partial file rather than restarting it.

    A file already recorded as verified returns immediately. Otherwise the transfer resumes from whatever
    bytes are already on disk via a range request, reporting ``(downloaded, total)`` to *callback* per
    chunk so the caller can surface real progress. The completed file is hash-verified before it is
    accepted; a mismatch deletes it and raises, so a corrupt asset faults here rather than reaching the
    safety process as an undecodable checkpoint.

    Args:
        callback: Invoked per chunk with the cumulative bytes on disk and the full expected size. It may
            raise (the download core's pacer raises :class:`DownloadAborted` on shutdown) to abort the
            transfer; the partial file is left in place for the next attempt to resume.

    Raises:
        RuntimeError: The completed download did not match the published digest.
        requests.RequestException: The transfer failed at the HTTP layer.
    """
    weight_path = deep_danbooru_model_path()
    if deep_danbooru_verified_on_disk():
        return

    url = deep_danbooru_source_url()
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    existing_bytes = weight_path.stat().st_size if weight_path.is_file() else 0

    response = _open_transfer(url, existing_bytes)
    if response is None:
        # The range was unsatisfiable, so the file on disk is at least as long as the asset. That is the
        # warm-worker case: a complete file written before this module existed, needing only its hash
        # confirmed. Hashing is deferred to exactly here so a partial file is never hashed on every retry.
        if _accept_verified(weight_path):
            logger.info("The DeepDanbooru safety weight was already complete on disk; recorded it as verified.")
            return
        weight_path.unlink(missing_ok=True)
        existing_bytes = 0
        response = _open_transfer(url, existing_bytes)
        if response is None:  # pragma: no cover - a zero-offset range is always satisfiable
            raise RuntimeError("The DeepDanbooru weight source rejected a request for the whole file.")

    with response:
        response.raise_for_status()
        if existing_bytes > 0 and response.status_code != requests.codes.partial_content:
            # The server answered the range with the whole file, so the bytes on disk are not a prefix of
            # what is arriving; overwrite rather than append.
            existing_bytes = 0
        remaining = int(response.headers.get("content-length", 0))
        total_bytes = existing_bytes + remaining

        megabytes = total_bytes / 1024 / 1024
        if existing_bytes > 0:
            logger.info(
                f"Resuming the DeepDanbooru safety weight at {existing_bytes / 1024 / 1024:.0f} MB "
                f"of {megabytes:.0f} MB.",
            )
        else:
            logger.info(f"Downloading the DeepDanbooru safety weight (~{megabytes:.0f} MB) to {weight_path}.")

        downloaded = existing_bytes
        if callback is not None:
            callback(downloaded, total_bytes)
        with weight_path.open("ab" if existing_bytes > 0 else "wb") as handle:
            for chunk in response.iter_content(_CHUNK_BYTES):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if callback is not None:
                    callback(downloaded, total_bytes)

    if _accept_verified(weight_path):
        logger.success("DeepDanbooru safety weight downloaded and verified.")
        return

    # Only reachable once the transfer ran to completion, so the bytes on disk are a full file that is
    # not the asset. Keeping it would make every later resume a no-op against a permanently wrong file.
    weight_path.unlink(missing_ok=True)
    raise RuntimeError(
        "The DeepDanbooru safety weight failed its checksum after downloading; the partial file has been "
        "removed so the next attempt starts clean.",
    )
