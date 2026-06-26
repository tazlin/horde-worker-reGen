"""Tests for the self-updater: install-info origin tracking, repo priority, and the --repo CLI flag.

Covers:
- ``resolve_update_repo`` priority chain (env var > file > default)
- ``write_repo_to_install_info`` round-trips and edge cases
- ``_cmd_update --repo``: persist on success, no-persist on check failure, no-persist in read-only mode,
  persist happens before the download attempt (failed download still records the new origin)
- CI packaging guards: release.yml ISCC invocation, Inno ``.iss`` repo define, one-liner write paths
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from worker_bootstrap import updater as updater_mod
from worker_bootstrap.cli import _cmd_update
from worker_bootstrap.updater import (
    _DEFAULT_REPO,
    UpdateInfo,
    UpdateResult,
    _read_install_info,
    resolve_update_repo,
    write_repo_to_install_info,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

_OFFICIAL_REPO = "Haidra-Org/horde-worker-reGen"
_FORK_REPO = "tazlin/horde-worker-reGen"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**overrides: object) -> argparse.Namespace:
    """Minimal Namespace suitable for _cmd_update; defaults to a non-interactive, no-update-available run."""
    base: dict[str, object] = {
        "check": False,
        "yes": True,
        "repo": None,
        # _sync_options reads these via getattr; they are not reached in the tested paths.
        "no_sync_preview": False,
        "hold_torch": None,
        "confirm_above_mb": None,
        "headless_policy": None,
        "no_prune": False,
        "cache_mode": None,
        "backend": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _info(*, latest: str | None, current: str = "12.30.1", available: bool = False) -> UpdateInfo:
    return UpdateInfo(current=current, latest=latest, available=available, bundle_url=None, checksums_url=None)


# ---------------------------------------------------------------------------
# resolve_update_repo: priority chain
# ---------------------------------------------------------------------------


def test_resolve_repo_env_var_wins_over_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """HORDE_WORKER_UPDATE_REPO overrides the recorded install origin in bin/install-info."""
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "install-info").write_text(f"repo={_OFFICIAL_REPO}\n", encoding="utf-8")
    monkeypatch.setenv("HORDE_WORKER_UPDATE_REPO", _FORK_REPO)
    assert resolve_update_repo(tmp_path) == _FORK_REPO


def test_resolve_repo_empty_env_var_falls_through_to_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty HORDE_WORKER_UPDATE_REPO is treated as unset and the file value is used."""
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "install-info").write_text(f"repo={_FORK_REPO}\n", encoding="utf-8")
    monkeypatch.setenv("HORDE_WORKER_UPDATE_REPO", "")
    assert resolve_update_repo(tmp_path) == _FORK_REPO


def test_resolve_repo_file_wins_over_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The recorded 'repo' key in bin/install-info overrides the hardcoded default."""
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "install-info").write_text(f"method=exe\nrepo={_FORK_REPO}\n", encoding="utf-8")
    monkeypatch.delenv("HORDE_WORKER_UPDATE_REPO", raising=False)
    assert resolve_update_repo(tmp_path) == _FORK_REPO


def test_resolve_repo_default_when_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When bin/install-info is absent the updater falls back to the canonical default repo."""
    monkeypatch.delenv("HORDE_WORKER_UPDATE_REPO", raising=False)
    assert resolve_update_repo(tmp_path) == _DEFAULT_REPO


def test_resolve_repo_default_when_file_has_no_repo_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bin/install-info that has 'method' but no 'repo' still falls back to the default.

    This is the pre-fix state for .exe installs built by a fork: the installer wrote 'method=exe'
    but no 'repo', so the updater would silently use the official default.
    """
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "install-info").write_text("method=exe\n", encoding="utf-8")
    monkeypatch.delenv("HORDE_WORKER_UPDATE_REPO", raising=False)
    assert resolve_update_repo(tmp_path) == _DEFAULT_REPO


# ---------------------------------------------------------------------------
# write_repo_to_install_info: round-trips and edge cases
# ---------------------------------------------------------------------------


def test_write_repo_creates_file_from_scratch(tmp_path: Path) -> None:
    """Creates bin/install-info (and the bin/ dir) when neither exist."""
    assert not (tmp_path / "bin").exists()
    write_repo_to_install_info(tmp_path, _FORK_REPO)
    assert _read_install_info(tmp_path).get("repo") == _FORK_REPO


def test_write_repo_updates_repo_and_preserves_other_keys(tmp_path: Path) -> None:
    """Updating 'repo' does not discard other recorded keys such as 'method'."""
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "install-info").write_text(f"method=exe\nrepo={_OFFICIAL_REPO}\n", encoding="utf-8")
    write_repo_to_install_info(tmp_path, _FORK_REPO)
    info = _read_install_info(tmp_path)
    assert info.get("repo") == _FORK_REPO
    assert info.get("method") == "exe"


def test_write_repo_is_idempotent(tmp_path: Path) -> None:
    """Writing the same repo twice produces exactly one 'repo=' line (no duplicates)."""
    write_repo_to_install_info(tmp_path, _FORK_REPO)
    write_repo_to_install_info(tmp_path, _FORK_REPO)
    content = (tmp_path / "bin" / "install-info").read_text(encoding="utf-8")
    assert content.count("repo=") == 1
    assert _read_install_info(tmp_path).get("repo") == _FORK_REPO


def test_write_repo_round_trips_through_resolve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Values written by write_repo_to_install_info are returned correctly by resolve_update_repo."""
    monkeypatch.delenv("HORDE_WORKER_UPDATE_REPO", raising=False)
    write_repo_to_install_info(tmp_path, _FORK_REPO)
    assert resolve_update_repo(tmp_path) == _FORK_REPO


def test_write_repo_strips_bom_from_existing_file(tmp_path: Path) -> None:
    """A UTF-8 BOM in an existing install-info is not propagated to the rewritten file.

    PowerShell may write a BOM; _read_install_info strips it on read, and the rewrite must not
    put it back so a subsequent plain read() never sees the BOM prefix.
    """
    bom = "﻿"
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "install-info").write_text(f"{bom}method=exe\nrepo={_OFFICIAL_REPO}\n", encoding="utf-8")
    write_repo_to_install_info(tmp_path, _FORK_REPO)
    raw = (tmp_path / "bin" / "install-info").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "rewrite must not carry the BOM forward"
    info = _read_install_info(tmp_path)
    assert info.get("repo") == _FORK_REPO
    assert info.get("method") == "exe"


# ---------------------------------------------------------------------------
# _cmd_update --repo flag: persist / no-persist logic
# ---------------------------------------------------------------------------


def test_update_repo_flag_persists_when_already_up_to_date(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--repo persists the new origin even when no update is available.

    This is the canonical fork-to-official (or official-to-fork) channel switch: both repos are at
    the same version, but the user wants future plain 'update' runs to track the new origin.
    """
    monkeypatch.setattr(updater_mod, "self_update_allowed", lambda root: (True, ""))
    monkeypatch.setattr(updater_mod, "check_for_update", lambda root, repo=None, channel=None: _info(latest="12.30.1"))

    rc = _cmd_update(_make_args(repo=_FORK_REPO), tmp_path, "uv")

    assert rc == 0
    assert resolve_update_repo(tmp_path) == _FORK_REPO
    out = capsys.readouterr().out
    assert "Update origin set to" in out
    assert _FORK_REPO in out


def test_update_repo_flag_does_not_persist_when_check_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--repo must NOT persist when the GitHub API call fails (info.latest is None).

    Persisting an unreachable repo would silently redirect all future plain updates to somewhere
    the network could not confirm even exists.
    """
    monkeypatch.setattr(updater_mod, "self_update_allowed", lambda root: (True, ""))
    monkeypatch.setattr(updater_mod, "check_for_update", lambda root, repo=None, channel=None: _info(latest=None))

    rc = _cmd_update(_make_args(repo=_FORK_REPO), tmp_path, "uv")

    assert rc == 1
    assert not (tmp_path / "bin" / "install-info").exists()


def test_update_without_repo_flag_never_touches_install_info(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plain 'update' (no --repo) must not create or modify bin/install-info."""
    monkeypatch.setattr(updater_mod, "self_update_allowed", lambda root: (True, ""))
    monkeypatch.setattr(updater_mod, "check_for_update", lambda root, repo=None, channel=None: _info(latest="12.30.1"))

    _cmd_update(_make_args(), tmp_path, "uv")

    assert not (tmp_path / "bin" / "install-info").exists()


def test_update_check_flag_does_not_persist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'update --check --repo' reports availability but never writes bin/install-info.

    --check is a read-only operation; it must not have side-effects regardless of which other
    flags are supplied alongside it.
    """
    monkeypatch.setattr(updater_mod, "check_for_update", lambda root, repo=None, channel=None: _info(latest="12.30.1"))

    rc = _cmd_update(_make_args(check=True, repo=_FORK_REPO), tmp_path, "uv")

    assert rc == 0
    assert not (tmp_path / "bin" / "install-info").exists()


def test_update_repo_flag_persists_before_download_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--repo is persisted as soon as the remote is confirmed reachable, even if the download later fails.

    The repo proved valid (check succeeded and returned a tag), so recording it now means a
    transient download failure does not leave the user's install-info pointing at the wrong origin.
    """
    monkeypatch.setattr(updater_mod, "self_update_allowed", lambda root: (True, ""))
    monkeypatch.setattr(
        updater_mod,
        "check_for_update",
        lambda root, repo=None, channel=None: _info(latest="13.1.3", available=True),
    )
    monkeypatch.setattr(
        updater_mod,
        "perform_update",
        lambda root, info: UpdateResult(ok=False, message="Download failed", from_version=None, to_version=None),
    )

    rc = _cmd_update(_make_args(repo=_FORK_REPO), tmp_path, "uv")

    assert rc == 1  # download failed, so overall update failed
    # but the origin must already be recorded
    assert resolve_update_repo(tmp_path) == _FORK_REPO


def test_update_repo_flag_persists_on_available_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--repo persists when an update is found and then successfully applied."""
    monkeypatch.setattr(updater_mod, "self_update_allowed", lambda root: (True, ""))
    monkeypatch.setattr(
        updater_mod,
        "check_for_update",
        lambda root, repo=None, channel=None: _info(latest="13.1.3", current="12.30.1", available=True),
    )
    monkeypatch.setattr(
        updater_mod,
        "perform_update",
        lambda root, info: UpdateResult(
            ok=True, message="Updated to 13.1.3.", from_version="12.30.1", to_version="13.1.3"
        ),
    )
    monkeypatch.setattr(updater_mod, "clear_skip", lambda root: None)
    monkeypatch.setattr(updater_mod, "sync_arp_version", lambda root, version: None)
    # _sync would try to run uv; short-circuit it here
    from worker_bootstrap import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_sync", lambda uv, root, *, cli_flag, options: 0)

    rc = _cmd_update(_make_args(repo=_FORK_REPO), tmp_path, "uv")

    assert rc == 0
    assert resolve_update_repo(tmp_path) == _FORK_REPO


# ---------------------------------------------------------------------------
# CI / packaging guards
# ---------------------------------------------------------------------------


def test_release_workflow_passes_repo_to_iscc() -> None:
    """The release workflow must pass /DRepo= to ISCC, derived from GITHUB_REPOSITORY.

    Without this, every .exe bakes in 'Haidra-Org/horde-worker-reGen' regardless of which fork's
    CI built it, causing 'update.cmd' to silently check the wrong repo and always say 'up to date'.
    This test is the regression guard for that bug.
    """
    text = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "/DRepo=" in text, (
        "release.yml ISCC invocation must pass /DRepo= so fork-built .exe installers record the "
        "fork's repo in bin/install-info instead of hardcoding the official repo"
    )
    assert "GITHUB_REPOSITORY" in text, (
        "the Repo value must derive from GITHUB_REPOSITORY so a fork's CI automatically stamps its own slug"
    )


def test_inno_script_has_repo_define_with_official_default() -> None:
    """HordeWorker.iss must have the Repo compile-time define with the official repo as its default.

    CI overrides the default with /DRepo=. The default being the official repo ensures that a local
    build (without /DRepo) still produces a sane install-info rather than an empty or broken one.
    """
    text = (REPO_ROOT / "packaging" / "inno" / "HordeWorker.iss").read_text(encoding="utf-8")
    assert "#ifndef Repo" in text, "HordeWorker.iss must have the Repo compile-time define"
    assert _OFFICIAL_REPO in text, "HordeWorker.iss Repo default must be the official repo slug"


def test_inno_installer_writes_repo_to_install_info() -> None:
    """The graphical installer must write repo={#Repo} into bin/install-info at post-install time.

    The Inno CurStepChanged(ssPostInstall) step is the only point at which the .exe front-end can
    stamp the install origin; if it omits 'repo', the updater silently falls back to the default
    (official) repo regardless of which fork distributed the .exe.
    """
    text = (REPO_ROOT / "packaging" / "inno" / "HordeWorker.iss").read_text(encoding="utf-8")
    # The .iss uses Pascal string literals: 'repo={#Repo}' inside SaveStringToFile
    assert "repo={#Repo}" in text, (
        "HordeWorker.iss CurStepChanged must write 'repo={#Repo}' into bin/install-info; "
        "without it the self-updater cannot know which fork to pull future releases from"
    )


def test_install_sh_records_method_and_repo_in_install_info() -> None:
    """install.sh must write both 'method=one-line' and 'repo=<origin>' into bin/install-info.

    The repo value is taken from HORDE_WORKER_REPO so a fork that bakes its slug into the
    advertised one-liner gets the correct origin recorded without editing the script itself.
    """
    text = (REPO_ROOT / "install.sh").read_text(encoding="utf-8")
    assert "HORDE_WORKER_REPO" in text, "install.sh must read the fork override from HORDE_WORKER_REPO"
    assert "install-info" in text, "install.sh must write to bin/install-info"
    assert "method=one-line" in text, "install.sh must stamp method=one-line in install-info"
    assert "repo=" in text, "install.sh must stamp the repo origin in install-info"


def test_install_ps1_records_method_and_repo_in_install_info() -> None:
    """install.ps1 must write both 'method=one-line' and 'repo=<origin>' into bin/install-info."""
    text = (REPO_ROOT / "install.ps1").read_text(encoding="utf-8")
    assert "HORDE_WORKER_REPO" in text, "install.ps1 must read the fork override from HORDE_WORKER_REPO"
    assert "install-info" in text, "install.ps1 must write to bin/install-info"
    assert "method=one-line" in text
    assert "repo=" in text
