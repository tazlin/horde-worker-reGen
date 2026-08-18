"""Tests for the resumable, checksum-verified fetch of the DeepDanbooru safety weight.

The behaviour under test is what makes a slow link survivable: a transfer that picks up where the last
attempt stopped, a presence verdict that a truncated file cannot pass, and a completed file that is rejected
unless it matches the published digest. The upstream fetch this replaces had none of the three, so an
interrupted download restarted from zero forever and its remains loaded as a corrupt archive.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from horde_worker_regen.process_management.workers import safety_model_prefetch as prefetch

_URL = "https://example.invalid/model-resnet_custom_v3.pt"
_BODY = bytes(range(256)) * 4
"""A stand-in payload; only its length and identity matter to the transfer logic."""


class _FakeResponse:
    """The subset of ``requests.Response`` the fetch touches, over an in-memory body."""

    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = {"content-length": str(len(body))}
        self.closed = False

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [self._body[at : at + chunk_size] for at in range(0, len(self._body), chunk_size)]


class _FakeServer:
    """Serves ``_BODY`` with real range semantics, recording every request it answered."""

    def __init__(self, *, honour_range: bool = True) -> None:
        self.honour_range = honour_range
        self.requested_ranges: list[int | None] = []

    def __call__(self, url: str, **kwargs: object) -> _FakeResponse:
        headers = kwargs.get("headers") or {}
        assert isinstance(headers, dict)
        raw_range = headers.get("Range")
        start = int(str(raw_range).removeprefix("bytes=").removesuffix("-")) if raw_range else None
        self.requested_ranges.append(start)
        if start is None or not self.honour_range:
            return _FakeResponse(200, _BODY)
        if start >= len(_BODY):
            return _FakeResponse(requests.codes.requested_range_not_satisfiable, b"")
        return _FakeResponse(requests.codes.partial_content, _BODY[start:])


@pytest.fixture
def weight_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the module at a temporary weight path and a fixed source URL."""
    path = tmp_path / "model-resnet_custom_v3.pt"
    monkeypatch.setattr(prefetch, "deep_danbooru_model_path", lambda: path)
    monkeypatch.setattr(prefetch, "deep_danbooru_source_url", lambda: _URL)
    # Several chunks per body, so the transfer is observable mid-flight rather than arriving whole.
    monkeypatch.setattr(prefetch, "_CHUNK_BYTES", len(_BODY) // 8)
    return path


def _accept_whatever_matches_the_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the published digest: the asset is exactly ``_BODY``."""
    monkeypatch.setattr(prefetch, "_verify_hash", lambda path: path.read_bytes() == _BODY)


def _install_server(monkeypatch: pytest.MonkeyPatch, server: _FakeServer) -> None:
    monkeypatch.setattr(prefetch.requests, "get", server)


def test_a_marked_complete_file_needs_no_request(
    weight_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A weight already recorded as verified is accepted without touching the network."""
    weight_path.write_bytes(_BODY)
    prefetch._write_marker(weight_path, len(_BODY))
    server = _FakeServer()
    _install_server(monkeypatch, server)
    _accept_whatever_matches_the_body(monkeypatch)

    prefetch.ensure_deep_danbooru_present()

    assert server.requested_ranges == []


def test_a_truncated_file_does_not_read_as_verified(weight_path: Path) -> None:
    """The presence verdict is what stopped a partial file from being handed to the safety process."""
    weight_path.write_bytes(_BODY[: len(_BODY) // 2])
    prefetch._write_marker(weight_path, len(_BODY))

    assert prefetch.deep_danbooru_verified_on_disk() is False


def test_a_partial_file_resumes_rather_than_restarting(
    weight_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The transfer asks for the bytes it does not have, and reports progress from where it resumed.

    This is the property the whole module exists for: without it, a link too slow to finish inside one
    worker session re-fetches the same opening megabytes forever.
    """
    already = len(_BODY) // 4
    weight_path.write_bytes(_BODY[:already])
    server = _FakeServer()
    _install_server(monkeypatch, server)
    _accept_whatever_matches_the_body(monkeypatch)

    seen: list[tuple[int, int]] = []
    prefetch.ensure_deep_danbooru_present(callback=lambda done, total: seen.append((done, total)))

    assert server.requested_ranges == [already]
    assert weight_path.read_bytes() == _BODY
    assert seen[0] == (already, len(_BODY))
    assert seen[-1] == (len(_BODY), len(_BODY))
    assert prefetch.deep_danbooru_verified_on_disk() is True


def test_a_complete_unmarked_file_is_verified_not_refetched(
    weight_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A warm worker upgrading into this module pays a hash, never the transfer again."""
    weight_path.write_bytes(_BODY)
    server = _FakeServer()
    _install_server(monkeypatch, server)
    _accept_whatever_matches_the_body(monkeypatch)

    prefetch.ensure_deep_danbooru_present()

    assert server.requested_ranges == [len(_BODY)]
    assert prefetch.deep_danbooru_verified_on_disk() is True


def test_a_full_length_file_that_is_not_the_asset_is_replaced(
    weight_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file as long as the asset but failing its digest is discarded and fetched whole."""
    weight_path.write_bytes(b"x" * len(_BODY))
    server = _FakeServer()
    _install_server(monkeypatch, server)
    _accept_whatever_matches_the_body(monkeypatch)

    prefetch.ensure_deep_danbooru_present()

    assert server.requested_ranges == [len(_BODY), None]
    assert weight_path.read_bytes() == _BODY


def test_a_server_that_ignores_the_range_overwrites_rather_than_appends(
    weight_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Answering a range with the whole file must not concatenate onto the bytes already held."""
    weight_path.write_bytes(_BODY[: len(_BODY) // 4])
    server = _FakeServer(honour_range=False)
    _install_server(monkeypatch, server)
    _accept_whatever_matches_the_body(monkeypatch)

    prefetch.ensure_deep_danbooru_present()

    assert weight_path.read_bytes() == _BODY


def test_a_checksum_failure_removes_the_file_and_raises(
    weight_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed transfer that does not match the digest leaves nothing behind for a resume to trust."""
    server = _FakeServer()
    _install_server(monkeypatch, server)
    monkeypatch.setattr(prefetch, "_verify_hash", lambda _path: False)

    with pytest.raises(RuntimeError):
        prefetch.ensure_deep_danbooru_present()

    assert weight_path.exists() is False
    assert prefetch.deep_danbooru_verified_on_disk() is False


def test_an_aborting_callback_leaves_the_partial_file_for_the_next_attempt(
    weight_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shutdown mid-transfer keeps its bytes, which is what the next start resumes from."""
    server = _FakeServer()
    _install_server(monkeypatch, server)
    _accept_whatever_matches_the_body(monkeypatch)

    class _Stop(Exception):
        pass

    def abort_after_first_chunk(done: int, _total: int) -> None:
        if done > 0:
            raise _Stop

    with pytest.raises(_Stop):
        prefetch.ensure_deep_danbooru_present(callback=abort_after_first_chunk)

    assert 0 < weight_path.stat().st_size < len(_BODY)
    assert prefetch.deep_danbooru_verified_on_disk() is False
