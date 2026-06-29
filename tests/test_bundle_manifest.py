"""Guards for the release bundle manifest (packaging/bundle-include.txt).

The release workflow stages exactly the files listed in the manifest, so these tests keep it honest:
a new root launcher must be added (or it silently would not ship), and a removed file must be taken out
(or staging would reference a path that no longer exists).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "packaging" / "bundle-include.txt"

# Root scripts that are intentionally NOT bundled: the curl|sh bootstrapper is fetched directly by the
# one-line installer, not shipped inside the zip it downloads.
NOT_BUNDLED = {"install.sh"}

_GLOB_CHARS = "*?["


def _manifest_entries() -> list[str]:
    """Non-comment, non-blank entries from the bundle manifest."""
    lines = MANIFEST.read_text(encoding="utf-8").splitlines()
    return [stripped for line in lines if (stripped := line.strip()) and not stripped.startswith("#")]


def test_manifest_entries_exist() -> None:
    """Every non-glob path in the manifest exists, so staging never references a removed file."""
    for entry in _manifest_entries():
        if any(char in entry for char in _GLOB_CHARS):
            continue
        assert (REPO_ROOT / entry).exists(), f"bundle manifest lists a missing path: {entry}"


def test_all_root_launchers_are_bundled() -> None:
    """Every root .cmd/.sh launcher is listed, so a new launcher fails CI until it is added."""
    entries = set(_manifest_entries())
    launchers = {path.name for path in REPO_ROOT.glob("*.cmd")} | {path.name for path in REPO_ROOT.glob("*.sh")}
    missing = launchers - entries - NOT_BUNDLED
    assert not missing, f"these root launchers are missing from packaging/bundle-include.txt: {sorted(missing)}"


def test_bootstrap_entry_point_is_bundled() -> None:
    """bootstrap.py (the entry the launchers run via `uv run --script`) must ship in the bundle."""
    assert "bootstrap.py" in _manifest_entries()


def test_bootstrap_package_is_staged_by_workflows() -> None:
    """Both staging scripts must copy the separately-bundled worker_bootstrap/ package.

    It is copied like horde_worker_regen/ (not via the manifest), so if a staging script drops it the
    bundle would omit the bootstrap brain and every launcher would fail.
    """
    assert (REPO_ROOT / "worker_bootstrap" / "cli.py").exists(), "worker_bootstrap/cli.py is missing"
    release = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    build_local = (REPO_ROOT / "packaging" / "build-local.ps1").read_text(encoding="utf-8")
    assert "worker_bootstrap" in release, "release.yml must stage the worker_bootstrap package"
    assert "worker_bootstrap" in build_local, "build-local.ps1 must stage the worker_bootstrap package"


def test_detect_backend_script_is_bundled() -> None:
    """detect-backend.ps1 must ship in the bundle for the graphical installer's pre-install wizard.

    The wizard detects the GPU before uv (and thus bootstrap.py) exists, sourcing the script from the same
    staging as the zip, so it can never drift from what ships.
    """
    assert "packaging/detect-backend.ps1" in _manifest_entries()


def test_inno_installer_sources_the_staging() -> None:
    """The graphical installer must build from the same staged bundle as the zip (single source of truth).

    Sourcing from the shared staging means it can never drift from what the manifest ships.
    """
    iss = REPO_ROOT / "packaging" / "inno" / "HordeWorker.iss"
    assert iss.exists(), "packaging/inno/HordeWorker.iss is missing"
    text = iss.read_text(encoding="utf-8")
    assert "{#StageDir}" in text, "HordeWorker.iss should source files from the {#StageDir} staging directory"
    assert "detect-backend.ps1" in text, "HordeWorker.iss should ship/extract detect-backend.ps1"


def test_start_here_orientation_file_is_bundled() -> None:
    """The browser-rendering 'Start Here' orientation page must ship so every install method drops it.

    It is the non-technical user's signpost in the install folder (README.md only opens in Notepad), so if
    it falls out of the manifest the EXE/zip/one-line installs would silently ship without it.
    """
    assert "_Start_Here.html" in _manifest_entries(), "_Start_Here.html must be bundled"


def test_inno_installer_defaults_start_menu_shortcut_on() -> None:
    """The graphical installer must pre-check the Start Menu shortcut.

    This is so a non-technical user is left with a way to relaunch without hunting through the install folder.
    Guards against silently flipping it back to unchecked.
    """
    text = (REPO_ROOT / "packaging" / "inno" / "HordeWorker.iss").read_text(encoding="utf-8")
    start_menu_line = next(
        (line for line in text.splitlines() if 'Name: "startmenuicon"' in line and line.strip().startswith("Name:")),
        None,
    )
    assert start_menu_line is not None, "HordeWorker.iss must declare the startmenuicon task"
    assert "unchecked" not in start_menu_line, "the startmenuicon task must default ON (no 'unchecked' flag)"


def test_disclosure_notices_are_bundled() -> None:
    """Both disclosure files must ship so every front-end can show the same notice and licenses."""
    entries = _manifest_entries()
    assert "INSTALL_NOTICE.txt" in entries, "INSTALL_NOTICE.txt must be bundled"
    assert "THIRD-PARTY-NOTICES.md" in entries, "THIRD-PARTY-NOTICES.md must be bundled"


def test_inno_installer_shows_disclosure_pages() -> None:
    """The graphical installer must surface the notice (Info page) and licenses (License page)."""
    text = (REPO_ROOT / "packaging" / "inno" / "HordeWorker.iss").read_text(encoding="utf-8")
    assert "InfoBeforeFile" in text and "INSTALL_NOTICE.txt" in text, "the .exe must show INSTALL_NOTICE.txt"
    assert "LicenseFile" in text and "THIRD-PARTY-NOTICES.md" in text, "the .exe must show the license notices"
