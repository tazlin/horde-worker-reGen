"""Parity guard: every install front-end must prune the worker's import roots on a reinstall.

A reinstall over an older release must not leave a renamed or deleted module behind to shadow the new
code. The self-updater already does this by mirror-pruning ``worker_bootstrap/updater.py:_MIRROR_DIRS``;
these tests pin that the one-line installers and the graphical (Inno) installer do the same, so a new
front-end (or a refactor that drops the step) fails CI instead of silently reintroducing the stale-module
gap. The shell/PowerShell installers reuse the overlay via the ``apply-bundle`` bootstrap command; Inno,
which has no clean bundle reference on the target machine, achieves the same delete-then-relay semantics
with an ``[InstallDelete]`` section.
"""

from __future__ import annotations

import re
from pathlib import Path, PureWindowsPath

from worker_bootstrap import updater
from worker_bootstrap.cli import _HANDLERS

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_apply_bundle_is_a_registered_bootstrap_command() -> None:
    """The installers invoke ``runtime.{sh,cmd} apply-bundle``, so the bootstrap must expose it."""
    assert "apply-bundle" in _HANDLERS, "the apply-bundle command the one-line installers call is missing"


def test_mirror_dirs_are_the_worker_import_roots() -> None:
    """The mirror-pruned set is the worker's two Python import roots (the source dirs a reinstall ships)."""
    assert frozenset({"horde_worker_regen", "worker_bootstrap"}) == updater._MIRROR_DIRS


def test_one_line_installers_apply_through_the_pruning_overlay() -> None:
    """Both one-line installers route a (re)install through ``apply-bundle`` rather than a raw extract.

    A plain ``unzip -o`` / ``Expand-Archive`` only overwrites and never prunes, so requiring the
    apply-bundle hand-off is what guarantees the mirror-prune actually runs for these front-ends.
    """
    for script in ("install.sh", "install.ps1"):
        text = (REPO_ROOT / script).read_text(encoding="utf-8")
        assert "apply-bundle" in text, f"{script} must overlay the bundle via `apply-bundle` (mirror-prune)"


def _inno_install_delete_targets(iss_text: str) -> list[str]:
    """Return the ``Name:`` targets declared in the Inno script's ``[InstallDelete]`` section."""
    match = re.search(r"^\[InstallDelete\](.*?)(?=^\[)", iss_text, re.MULTILINE | re.DOTALL)
    if not match:
        return []
    return re.findall(r'Name:\s*"([^"]+)"', match.group(1))


def test_inno_installer_prunes_the_import_roots_before_copying() -> None:
    """The graphical installer must delete each import root before [Files] relays the new bundle.

    Inno's [Files] never removes files dropped between versions, so the [InstallDelete] entries are the
    graphical path's equivalent of the one-liners' mirror-prune. They are checked against the canonical
    ``_MIRROR_DIRS`` so the two paths cannot drift.
    """
    iss_text = (REPO_ROOT / "packaging" / "inno" / "HordeWorker.iss").read_text(encoding="utf-8")
    targets = {PureWindowsPath(name).name for name in _inno_install_delete_targets(iss_text)}
    missing = updater._MIRROR_DIRS - targets
    assert not missing, f"HordeWorker.iss [InstallDelete] must remove these import roots: {sorted(missing)}"
