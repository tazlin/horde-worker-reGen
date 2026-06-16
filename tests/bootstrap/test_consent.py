"""Unit tests for the install-time disclosure and consent gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from worker_bootstrap import consent


@pytest.fixture
def notice(tmp_path: Path) -> Path:
    """A bundled notice file on disk."""
    path = tmp_path / "INSTALL_NOTICE.txt"
    path.write_text("THE NOTICE TEXT", encoding="utf-8")
    return path


@pytest.fixture
def marker(tmp_path: Path) -> Path:
    """The (initially absent) consent marker path, under a bin/ that does not yet exist."""
    return tmp_path / "bin" / "install-consent"


def test_consent_env_var_detects_each(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any of the three consent env vars is reported; none set reports None."""
    for name in ("HORDE_WORKER_ASSUME_YES", "HORDE_WORKER_NONINTERACTIVE", "HORDE_WORKER_FROM_LAUNCHER"):
        monkeypatch.delenv(name, raising=False)
    assert consent.consent_env_var() is None
    monkeypatch.setenv("HORDE_WORKER_NONINTERACTIVE", "1")
    assert consent.consent_env_var() == "HORDE_WORKER_NONINTERACTIVE"


def test_marker_present_proceeds_silently(notice: Path, marker: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An existing marker means consent was already captured: proceed without reprinting the notice."""
    marker.parent.mkdir(parents=True)
    marker.write_text("x", encoding="utf-8")
    assert consent.ensure_consent(notice_path=notice, marker_path=marker, interactive=True, consent_env=None)
    assert "THE NOTICE TEXT" not in capsys.readouterr().out


def test_consent_env_acknowledges_and_writes_marker(notice: Path, marker: Path) -> None:
    """A consent env var proceeds and records the marker, without reprinting the full notice again."""
    assert consent.ensure_consent(
        notice_path=notice,
        marker_path=marker,
        interactive=False,
        consent_env="HORDE_WORKER_ASSUME_YES",
    )
    assert marker.exists()


def test_interactive_yes_proceeds(notice: Path, marker: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A 'y' at the prompt proceeds, shows the notice, and records the marker."""
    assert consent.ensure_consent(
        notice_path=notice,
        marker_path=marker,
        interactive=True,
        consent_env=None,
        prompt=lambda _: "y",
    )
    assert marker.exists()
    assert "THE NOTICE TEXT" in capsys.readouterr().out


def test_interactive_no_aborts_without_marker(notice: Path, marker: Path) -> None:
    """Declining at the prompt aborts and leaves no marker (a re-run will ask again)."""
    assert not consent.ensure_consent(
        notice_path=notice,
        marker_path=marker,
        interactive=True,
        consent_env=None,
        prompt=lambda _: "",  # empty == default No
    )
    assert not marker.exists()


def test_headless_no_flag_proceeds(notice: Path, marker: Path) -> None:
    """No TTY and no flag (Docker/CI) proceeds so established automation keeps working."""
    assert consent.ensure_consent(notice_path=notice, marker_path=marker, interactive=False, consent_env=None)
    assert marker.exists()


def test_read_notice_falls_back_when_missing(tmp_path: Path) -> None:
    """A missing notice file yields the built-in fallback rather than crashing."""
    text = consent.read_notice(tmp_path / "absent.txt")
    assert "AI Horde Worker" in text
