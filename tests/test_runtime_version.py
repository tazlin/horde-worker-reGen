"""Tests for the runtime version dev-marker annotation.

The release literal ``__version__`` stays clean; ``runtime_version()`` only annotates it with a
``+dev.g<sha>`` suffix when run from a git checkout that is not exactly on the matching release tag.
These tests drive that logic with ``_git`` and the repo-root probe stubbed, so they never touch the real
repository state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import horde_worker_regen.runtime_version as rv
from horde_worker_regen import __version__


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """``runtime_version`` is cached; clear it around every case so stubs take effect."""
    rv.runtime_version.cache_clear()


def _with_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, responses: dict[tuple[str, ...], str | None]) -> None:
    """Point the helper at a tmp repo root that has a .git dir and stub ``_git`` from *responses*."""
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(rv, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(rv, "_git", lambda *args: responses.get(tuple(args)))


def test_no_git_checkout_returns_plain_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A directory with no .git (e.g. an installed/release bundle) reports the clean version."""
    monkeypatch.setattr(rv, "_REPO_ROOT", tmp_path)  # no .git created
    assert rv.runtime_version() == __version__


def test_on_release_tag_clean_returns_plain_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """HEAD exactly on the matching tag with a clean tree is indistinguishable from a release."""
    _with_git(
        monkeypatch,
        tmp_path,
        {
            ("describe", "--tags", "--exact-match"): f"v{__version__}",
            ("status", "--porcelain"): "",
        },
    )
    assert rv.runtime_version() == __version__


def test_non_tag_clean_gets_dev_suffix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A clean checkout that is not on the tag gets a ``+dev.g<sha>`` marker."""
    _with_git(
        monkeypatch,
        tmp_path,
        {
            ("describe", "--tags", "--exact-match"): None,  # not on any tag
            ("status", "--porcelain"): "",
            ("rev-parse", "--short", "HEAD"): "abc1234",
        },
    )
    assert rv.runtime_version() == f"{__version__}+dev.gabc1234"


def test_non_tag_dirty_gets_dirty_suffix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Uncommitted changes off the tag add a ``.modified`` marker."""
    _with_git(
        monkeypatch,
        tmp_path,
        {
            ("describe", "--tags", "--exact-match"): None,
            ("status", "--porcelain"): " M file.py",
            ("rev-parse", "--short", "HEAD"): "abc1234",
        },
    )
    assert rv.runtime_version() == f"{__version__}+dev.gabc1234.modified"


def test_on_tag_but_dirty_still_marked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A dirty tree on the release tag is not the released bits, so it is still marked dev/dirty."""
    _with_git(
        monkeypatch,
        tmp_path,
        {
            ("describe", "--tags", "--exact-match"): f"v{__version__}",
            ("status", "--porcelain"): " M file.py",
            ("rev-parse", "--short", "HEAD"): "abc1234",
        },
    )
    assert rv.runtime_version() == f"{__version__}+dev.gabc1234.modified"


def test_git_unavailable_degrades_to_plain(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """If git fails for everything (binary missing, not a repo), fall back to the clean version."""
    _with_git(monkeypatch, tmp_path, {})  # every _git call returns None
    assert rv.runtime_version() == __version__


class _FakeDistribution:
    """A stand-in for ``importlib.metadata.Distribution`` carrying a version and optional ``direct_url.json``."""

    def __init__(self, direct_url: str | None) -> None:
        self.version = "7.9.0"
        self._direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        return self._direct_url if filename == "direct_url.json" else None


def _with_hordelib(monkeypatch: pytest.MonkeyPatch, distribution: _FakeDistribution | None, dirty: bool) -> None:
    """Stub the hordelib distribution lookup and the checkout's git answers."""
    rv.hordelib_identity.cache_clear()

    def _distribution(name: str) -> _FakeDistribution:
        if distribution is None:
            raise rv.metadata.PackageNotFoundError(name)
        return distribution

    responses = {("rev-parse", "--short", "HEAD"): "c3202bd0", ("status", "--porcelain"): " M x.py" if dirty else ""}
    monkeypatch.setattr(rv.metadata, "distribution", _distribution)
    monkeypatch.setattr(rv, "_git", lambda *args, cwd=None: responses.get(tuple(args)))


def test_hordelib_site_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    """A regular install reports its version from site-packages."""
    _with_hordelib(monkeypatch, _FakeDistribution(None), dirty=False)
    assert rv.hordelib_identity() == "7.9.0 (site-packages)"


@pytest.mark.parametrize(("dirty", "suffix"), [(False, ""), (True, ".modified")])
def test_hordelib_editable_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dirty: bool, suffix: str) -> None:
    """An editable install names the checkout's commit, marked when the tree is dirty."""
    checkout = tmp_path / "hordelib"
    direct_url = f'{{"url": "{checkout.as_uri()}", "dir_info": {{"editable": true}}}}'
    _with_hordelib(monkeypatch, _FakeDistribution(direct_url), dirty=dirty)
    assert rv.hordelib_identity() == f"7.9.0 (editable hordelib gc3202bd0{suffix})"


def test_hordelib_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing distribution degrades to an unknown identity without raising."""
    _with_hordelib(monkeypatch, None, dirty=False)
    assert rv.hordelib_identity() == "unknown (not installed)"
