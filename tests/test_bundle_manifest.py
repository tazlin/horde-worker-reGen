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
