"""Unit tests for the self-updater engine (no real network; uses an in-memory bundle)."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from worker_bootstrap import updater


def test_version_compare_ignores_v_prefix_and_suffix() -> None:
    """Tags compare on their numeric head; a leading v and build metadata are ignored."""
    assert updater.is_newer("v12.26.0", "12.25.0") is True
    assert updater.is_newer("v12.25.0", "12.25.0") is False
    assert updater.is_newer("12.25.0", "12.26.0") is False
    assert updater.is_newer("v12.26.0+dev.gabc", "12.25.0") is True


def test_version_compare_respects_prerelease_precedence() -> None:
    """Semver precedence: a final outranks its own pre-release, and pre-releases order among themselves."""
    # A final release outranks any pre-release of the same core (graduation), and the reverse is not newer.
    assert updater.is_newer("12.26.0", "12.26.0-beta.1") is True
    assert updater.is_newer("12.26.0-beta.1", "12.26.0") is False
    # Pre-releases order numerically and by stage; an older final never "reverts" a higher beta.
    assert updater.is_newer("12.26.0-beta.2", "12.26.0-beta.1") is True
    assert updater.is_newer("12.26.0-rc.1", "12.26.0-beta.9") is True
    assert updater.is_newer("12.26.0", "12.27.0-beta.1") is False


def test_update_channel_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Channel: explicit env wins; otherwise a pre-release build infers beta, else stable."""
    monkeypatch.delenv("HORDE_WORKER_UPDATE_CHANNEL", raising=False)
    pkg = tmp_path / "horde_worker_regen"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('__version__ = "12.26.0"\n', encoding="utf-8")
    assert updater.update_channel(tmp_path) == "stable"

    (pkg / "__init__.py").write_text('__version__ = "12.27.0-beta.1"\n', encoding="utf-8")
    assert updater.update_channel(tmp_path) == "beta"  # inferred from the running pre-release

    monkeypatch.setenv("HORDE_WORKER_UPDATE_CHANNEL", "stable")
    assert updater.update_channel(tmp_path) == "stable"  # explicit env overrides the inference


def _release(tag: str, *, prerelease: bool, draft: bool = False) -> dict[str, object]:
    """A minimal GitHub release object carrying the bundle + checksums assets."""
    return {
        "tag_name": tag,
        "prerelease": prerelease,
        "draft": draft,
        "assets": [
            {"name": "horde-worker-reGen.zip", "browser_download_url": f"https://example/{tag}/bundle.zip"},
            {"name": "SHA256SUMS", "browser_download_url": f"https://example/{tag}/SHA256SUMS"},
        ],
    }


def test_beta_channel_picks_newest_including_prereleases(monkeypatch: pytest.MonkeyPatch) -> None:
    """The beta channel reads the releases list and selects the highest-precedence non-draft release."""
    releases = [
        _release("v12.26.0", prerelease=False),
        _release("v12.27.0-beta.2", prerelease=True),
        _release("v12.27.0-beta.3", prerelease=True),
        _release("v12.27.0-beta.4", prerelease=True, draft=True),  # drafts ignored
    ]
    monkeypatch.setattr(updater, "_fetch_json", lambda url: releases)
    result = updater.latest_release("owner/repo", channel="beta")
    assert result is not None
    tag, _assets, is_prerelease = result
    assert tag == "v12.27.0-beta.3"
    assert is_prerelease is True


def test_stable_channel_never_sees_a_prerelease(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stable channel uses /releases/latest, which the API already filters to non-pre-releases."""
    monkeypatch.setattr(updater, "_fetch_json", lambda url: _release("v12.26.0", prerelease=False))
    result = updater.latest_release("owner/repo", channel="stable")
    assert result is not None
    assert result[0] == "v12.26.0"
    assert result[2] is False


def test_check_for_update_beta_does_not_downgrade(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker on a higher beta is not offered an older stable (no revert)."""
    monkeypatch.delenv("HORDE_WORKER_UPDATE_CHANNEL", raising=False)
    monkeypatch.delenv("HORDE_WORKER_UPDATE_REPO", raising=False)
    pkg = tmp_path / "horde_worker_regen"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('__version__ = "12.27.0-beta.3"\n', encoding="utf-8")
    # Newest thing the beta channel can see is an older final; precedence says it is not an update.
    monkeypatch.setattr(updater, "_fetch_json", lambda url: [_release("v12.26.0", prerelease=False)])
    info = updater.check_for_update(tmp_path)
    assert info.channel == "beta"
    assert info.available is False


def test_resolve_update_repo_prefers_env_then_marker_then_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Origin resolution order: env override > recorded install marker > production default."""
    monkeypatch.delenv("HORDE_WORKER_UPDATE_REPO", raising=False)
    assert updater.resolve_update_repo(tmp_path) == "Haidra-Org/horde-worker-reGen"

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "install-info").write_text("method=one-line\nrepo=tazlin/horde-worker-reGen\n", encoding="utf-8")
    assert updater.resolve_update_repo(tmp_path) == "tazlin/horde-worker-reGen"

    monkeypatch.setenv("HORDE_WORKER_UPDATE_REPO", "someone/fork")
    assert updater.resolve_update_repo(tmp_path) == "someone/fork"


def test_install_info_marker_makes_updates_follow_the_install_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The marker the installers write (``method=one-line``/``repo=<owner>/<repo>``) is the origin updates follow.

    install.ps1/install.sh resolve the download source from ``HORDE_WORKER_REPO`` (canonical default for the
    upstream repo, a fork's one-liner overriding it) and record the resolved value here. This pins those exact
    bytes so a fork install keeps self-updating from the fork, while an explicit env override still wins.
    """
    monkeypatch.delenv("HORDE_WORKER_UPDATE_REPO", raising=False)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "install-info").write_text("method=one-line\nrepo=someone/fork\n", encoding="utf-8")
    assert updater.resolve_update_repo(tmp_path) == "someone/fork"
    assert updater.resolve_install_method(tmp_path) == "one-line"

    monkeypatch.setenv("HORDE_WORKER_UPDATE_REPO", "elsewhere/repo")
    assert updater.resolve_update_repo(tmp_path) == "elsewhere/repo"


def test_install_info_parsing_tolerates_a_bom(tmp_path: Path) -> None:
    """A marker written with a UTF-8 BOM (PowerShell) still parses its first key."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "install-info").write_text("method=exe\nrepo=Haidra-Org/horde-worker-reGen\n", encoding="utf-8-sig")
    assert updater.resolve_update_repo(tmp_path) == "Haidra-Org/horde-worker-reGen"
    assert updater.resolve_install_method(tmp_path) == "exe"


def test_resolve_install_method_and_self_update_gating(tmp_path: Path) -> None:
    """A winget install and dev checkouts are detected and refused; an ordinary install is allowed."""
    # A winget portable path is detected from the path itself, regardless of any marker.
    winget_root = tmp_path / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages" / "Haidra.HordeWorker_x"
    winget_root.mkdir(parents=True)
    assert updater.resolve_install_method(winget_root) == "winget"
    allowed, reason = updater.self_update_allowed(winget_root)
    assert allowed is False and "winget upgrade" in reason

    # A git working tree is a developer checkout: never overlay it.
    dev_root = tmp_path / "checkout"
    (dev_root / ".git").mkdir(parents=True)
    assert updater.resolve_install_method(dev_root) == "dev"
    assert updater.self_update_allowed(dev_root)[0] is False

    # A local path= source in pyproject is also a dev pin.
    pin_root = tmp_path / "pinned"
    pin_root.mkdir()
    (pin_root / "pyproject.toml").write_text(
        '[tool.uv.sources]\nhordelib = { path = "../hordelib" }\n', encoding="utf-8"
    )
    assert updater.resolve_install_method(pin_root) == "dev"

    # An ordinary install (no marker, no dev/winget signal) is unknown and allowed to self-update.
    plain_root = tmp_path / "plain"
    plain_root.mkdir()
    assert updater.resolve_install_method(plain_root) == "unknown"
    assert updater.self_update_allowed(plain_root)[0] is True


def test_skip_version_persists_until_a_newer_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A skipped version is remembered, cleared on update, and only that exact version is suppressed."""
    state = tmp_path / ".update-state.json"
    monkeypatch.setattr(updater.paths, "update_state_file", lambda root=None: state)

    assert updater.is_version_skipped(tmp_path, "v2.0.0") is False
    updater.mark_version_skipped(tmp_path, "v2.0.0")
    assert updater.is_version_skipped(tmp_path, "v2.0.0") is True
    assert updater.is_version_skipped(tmp_path, "v2.1.0") is False  # a newer version is still offered
    updater.clear_skip(tmp_path)
    assert updater.is_version_skipped(tmp_path, "v2.0.0") is False


def test_check_throttle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The launch-time check is due when no check is recorded, and not due right after one is recorded."""
    state = tmp_path / ".update-state.json"
    monkeypatch.setattr(updater.paths, "update_state_file", lambda root=None: state)
    assert updater.should_check_now(tmp_path) is True
    updater.record_check(tmp_path)
    assert updater.should_check_now(tmp_path) is False


def test_installed_version_parses_literal(tmp_path: Path) -> None:
    """The installed version is read from the source literal without importing the package."""
    pkg = tmp_path / "horde_worker_regen"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('__version__ = "12.25.0"\n', encoding="utf-8")
    assert updater.installed_version(tmp_path) == "12.25.0"


def test_installed_version_missing_is_none(tmp_path: Path) -> None:
    """A missing package yields None rather than raising."""
    assert updater.installed_version(tmp_path) is None


def test_expected_hash_matches_asset_line() -> None:
    """The checksum parser finds the sha256 for the named asset (tolerating a binary '*' marker)."""
    text = "abc123  other.zip\nDEADBEEF *horde-worker-reGen.zip\n"
    assert updater._expected_hash(text, "horde-worker-reGen.zip") == "DEADBEEF"
    assert updater._expected_hash("nope  other.zip", "horde-worker-reGen.zip") is None


def test_overlay_preserves_shims_and_user_state(tmp_path: Path) -> None:
    """The overlay updates Python source but never the running shell shims or preserved state."""
    bundle = tmp_path / "bundle"
    install = tmp_path / "install"
    (bundle / "horde_worker_regen").mkdir(parents=True)
    (bundle / "horde_worker_regen" / "__init__.py").write_text('__version__ = "2.0.0"\n', encoding="utf-8")
    (bundle / "pyproject.toml").write_text("new-pyproject\n", encoding="utf-8")
    (bundle / "runtime.cmd").write_text("NEW SHIM\n", encoding="utf-8")  # must be skipped

    install.mkdir()
    (install / "horde_worker_regen").mkdir()
    (install / "horde_worker_regen" / "__init__.py").write_text('__version__ = "1.0.0"\n', encoding="utf-8")
    (install / "runtime.cmd").write_text("OLD SHIM\n", encoding="utf-8")
    (install / "bridgeData.yaml").write_text("api_key: secret\n", encoding="utf-8")

    updater._overlay(bundle, install)

    assert updater.installed_version(install) == "2.0.0"  # source updated
    assert (install / "pyproject.toml").read_text(encoding="utf-8") == "new-pyproject\n"
    assert (install / "runtime.cmd").read_text(encoding="utf-8") == "OLD SHIM\n"  # shim untouched
    assert (install / "bridgeData.yaml").read_text(encoding="utf-8") == "api_key: secret\n"  # user state kept


def _build_bundle_zip(version: str) -> bytes:
    """Return a zip whose contents mimic the release bundle (files at the archive root)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("pyproject.toml", "name = 'x'\n")
        archive.writestr("uv.lock", "lock-new\n")
        archive.writestr("horde_worker_regen/__init__.py", f'__version__ = "{version}"\n')
        archive.writestr("runtime.cmd", "NEW SHIM\n")
    return buffer.getvalue()


def test_perform_update_verifies_then_overlays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A correctly-hashed bundle is applied, bumping the installed version and the lockfile."""
    install = tmp_path / "install"
    (install / "horde_worker_regen").mkdir(parents=True)
    (install / "horde_worker_regen" / "__init__.py").write_text('__version__ = "1.0.0"\n', encoding="utf-8")
    (install / "uv.lock").write_text("lock-old\n", encoding="utf-8")
    (install / "runtime.cmd").write_text("OLD SHIM\n", encoding="utf-8")

    bundle_bytes = _build_bundle_zip("2.0.0")
    checksums = f"{hashlib.sha256(bundle_bytes).hexdigest()}  horde-worker-reGen.zip\n"

    def fake_http_get(url: str) -> bytes:
        return bundle_bytes if url.endswith(".zip") else checksums.encode("utf-8")

    monkeypatch.setattr(updater, "_http_get", fake_http_get)

    info = updater.UpdateInfo(
        current="1.0.0",
        latest="v2.0.0",
        available=True,
        bundle_url="https://example/horde-worker-reGen.zip",
        checksums_url="https://example/SHA256SUMS",
    )
    result = updater.perform_update(install, info)

    assert result.ok is True
    assert result.to_version == "2.0.0"
    assert (install / "uv.lock").read_text(encoding="utf-8") == "lock-new\n"
    assert (install / "runtime.cmd").read_text(encoding="utf-8") == "OLD SHIM\n"  # shim preserved


def test_perform_update_refuses_on_checksum_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bundle whose hash does not match the published checksum is never written to the install."""
    install = tmp_path / "install"
    (install / "horde_worker_regen").mkdir(parents=True)
    (install / "horde_worker_regen" / "__init__.py").write_text('__version__ = "1.0.0"\n', encoding="utf-8")

    def fake_http_get(url: str) -> bytes:
        return _build_bundle_zip("2.0.0") if url.endswith(".zip") else b"0000  horde-worker-reGen.zip\n"

    monkeypatch.setattr(updater, "_http_get", fake_http_get)

    info = updater.UpdateInfo(
        current="1.0.0",
        latest="v2.0.0",
        available=True,
        bundle_url="https://example/horde-worker-reGen.zip",
        checksums_url="https://example/SHA256SUMS",
    )
    result = updater.perform_update(install, info)

    assert result.ok is False
    assert "mismatch" in result.message.lower()
    assert updater.installed_version(install) == "1.0.0"  # untouched


def test_perform_update_refuses_without_checksums(tmp_path: Path) -> None:
    """An update with no published checksums asset is refused rather than applied unverified."""
    info = updater.UpdateInfo(
        current="1.0.0",
        latest="v2.0.0",
        available=True,
        bundle_url="https://example/horde-worker-reGen.zip",
        checksums_url=None,
    )
    result = updater.perform_update(tmp_path, info)
    assert result.ok is False


def test_expected_hash_matches_basename_not_suffix() -> None:
    """An unrelated asset whose name merely ends with the bundle name is not mistaken for it."""
    text = "aaa  prefixed-horde-worker-reGen.zip\nbbb  horde-worker-reGen.zip\n"
    assert updater._expected_hash(text, "horde-worker-reGen.zip") == "bbb"
    # A path-prefixed entry still matches by basename.
    assert updater._expected_hash("ccc  stage/horde-worker-reGen.zip", "horde-worker-reGen.zip") == "ccc"


def test_bundle_root_descends_single_wrapper_dir(tmp_path: Path) -> None:
    """A zip that wraps everything in one top directory is descended into; a flat root is returned as-is."""
    extracted = tmp_path / "extracted"
    inner = extracted / "horde-worker-reGen"
    inner.mkdir(parents=True)
    (inner / "pyproject.toml").write_text("x\n", encoding="utf-8")
    assert updater._bundle_root(extracted) == inner

    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "pyproject.toml").write_text("x\n", encoding="utf-8")
    assert updater._bundle_root(flat) == flat


def test_overlay_prunes_stale_import_root_files_but_merges_other_dirs(tmp_path: Path) -> None:
    """A module dropped from the new bundle is pruned from an import root; non-mirrored dirs are merged."""
    bundle = tmp_path / "bundle"
    install = tmp_path / "install"
    (bundle / "horde_worker_regen").mkdir(parents=True)
    (bundle / "horde_worker_regen" / "__init__.py").write_text('__version__ = "2.0.0"\n', encoding="utf-8")
    (bundle / "docs").mkdir()
    (bundle / "docs" / "new.md").write_text("new\n", encoding="utf-8")

    install.mkdir()
    pkg = install / "horde_worker_regen"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('__version__ = "1.0.0"\n', encoding="utf-8")
    (pkg / "removed_upstream.py").write_text("stale\n", encoding="utf-8")  # absent from the new bundle
    docs = install / "docs"
    docs.mkdir()
    (docs / "old.md").write_text("old\n", encoding="utf-8")  # non-mirrored dir: keep

    updater._overlay(bundle, install)

    assert not (pkg / "removed_upstream.py").exists()  # pruned from the import root
    assert updater.installed_version(install) == "2.0.0"
    assert (docs / "old.md").exists()  # merged dir keeps pre-existing files
    assert (docs / "new.md").exists()


def test_perform_update_invalidates_sync_stamp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The overlay clears the venv sync stamp so an interrupted update still reconciles on the next launch."""
    install = tmp_path / "install"
    (install / "horde_worker_regen").mkdir(parents=True)
    (install / "horde_worker_regen" / "__init__.py").write_text('__version__ = "1.0.0"\n', encoding="utf-8")
    stamp = updater.paths.sync_stamp_file(install)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text("old-fingerprint", encoding="utf-8")

    bundle_bytes = _build_bundle_zip("2.0.0")
    checksums = f"{hashlib.sha256(bundle_bytes).hexdigest()}  horde-worker-reGen.zip\n"
    monkeypatch.setattr(
        updater, "_http_get", lambda url: bundle_bytes if url.endswith(".zip") else checksums.encode("utf-8")
    )

    info = updater.UpdateInfo(
        current="1.0.0",
        latest="v2.0.0",
        available=True,
        bundle_url="https://example/horde-worker-reGen.zip",
        checksums_url="https://example/SHA256SUMS",
    )
    result = updater.perform_update(install, info)

    assert result.ok is True
    assert not stamp.exists()  # cleared so the reconcile sync runs on the next launch
