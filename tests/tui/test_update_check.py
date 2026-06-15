"""Tests for the best-effort release update check."""

from __future__ import annotations

import pytest

from horde_worker_regen.tui import update_check
from horde_worker_regen.tui.update_check import UpdateInfo, check_for_update


@pytest.mark.parametrize(
    ("value", "expected"),
    [("v12.0.1", (12, 0, 1)), ("12.0.0", (12, 0, 0)), ("13.2", (13, 2)), ("v1.0.0rc1", (1, 0, 0))],
)
def test_version_tuple_parsing(value: str, expected: tuple[int, ...]) -> None:
    """Versions parse to integer tuples, tolerating a leading v and odd suffixes."""
    assert update_check._version_tuple(value) == expected


def test_is_newer_compares_versions() -> None:
    """A strictly higher version is newer; equal or lower is not."""
    assert update_check._is_newer("12.0.1", "12.0.0") is True
    assert update_check._is_newer("v13.0.0", "12.9.9") is True
    assert update_check._is_newer("v12.0.0", "12.0.0") is False
    assert update_check._is_newer("11.9.9", "12.0.0") is False


def test_check_for_update_reports_a_newer_release(monkeypatch: pytest.MonkeyPatch) -> None:
    """A newer tag yields UpdateInfo with the version (v-stripped) and the release URL."""
    monkeypatch.setattr(
        update_check,
        "_fetch_latest_release",
        lambda: {"tag_name": "v13.0.0", "html_url": "https://example.test/release"},
    )
    assert check_for_update(current="12.0.0") == UpdateInfo(
        latest_version="13.0.0",
        html_url="https://example.test/release",
    )


def test_check_for_update_none_when_up_to_date(monkeypatch: pytest.MonkeyPatch) -> None:
    """No update is reported when the latest tag matches the running version."""
    monkeypatch.setattr(update_check, "_fetch_latest_release", lambda: {"tag_name": "v12.0.0"})
    assert check_for_update(current="12.0.0") is None


def test_check_for_update_none_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreachable API is silent (returns None), so an offline launch shows nothing."""

    def _boom() -> dict[str, object]:
        raise OSError("offline")

    monkeypatch.setattr(update_check, "_fetch_latest_release", _boom)
    assert check_for_update(current="12.0.0") is None


def test_check_for_update_defaults_html_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an html_url in the payload, the canonical releases page is used."""
    monkeypatch.setattr(update_check, "_fetch_latest_release", lambda: {"tag_name": "v13.0.0"})
    info = check_for_update(current="12.0.0")
    assert info is not None
    assert info.html_url == update_check.RELEASES_URL
