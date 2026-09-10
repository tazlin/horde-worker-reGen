"""Every declared finding kind must be catalogued in ``docs/reference/log_findings.md``, and vice versa.

The finding id is the handle an operator, the JSON output, the dashboard, and a ``see also:`` line all
use, so an id with no entry in the catalogue is a dead end for whoever reads it. The declared set is
:data:`~horde_worker_regen.analysis.finding_kinds.FINDING_SPECS`, so this test reads the table rather
than scraping detector source: a kind cannot ship undocumented, and a catalogue row cannot outlive the
kind it describes.
"""

from __future__ import annotations

from pathlib import Path

from horde_worker_regen.analysis.finding_kinds import FINDING_SPECS, FindingKind

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FINDINGS_DOC = _REPO_ROOT / "docs" / "reference" / "log_findings.md"


def _catalogued_ids() -> set[str]:
    """The ids in the first column of the catalogue's tables.

    Prose elsewhere on the page may legitimately mention a concept that is not itself a finding, so only
    table rows count as an entry.
    """
    return {
        line.split("|")[1].strip().strip("`")
        for line in _FINDINGS_DOC.read_text(encoding="utf-8").splitlines()
        if line.startswith("| `")
    }


def test_the_catalogue_page_exists() -> None:
    """The catalogue is a committed reference page, not a generated one."""
    assert _FINDINGS_DOC.is_file(), f"missing {_FINDINGS_DOC}"


def test_every_finding_kind_is_catalogued() -> None:
    """No declared kind is missing an entry that explains it."""
    missing = sorted(kind.value for kind in FINDING_SPECS if kind.value not in _catalogued_ids())
    assert not missing, f"finding ids missing from docs/reference/log_findings.md: {missing}"


def test_the_catalogue_names_no_retired_id() -> None:
    """An id removed from the declared set is removed from the catalogue in the same change."""
    declared = {kind.value for kind in FINDING_SPECS}
    stale = sorted(_catalogued_ids() - declared)
    assert not stale, f"catalogued ids no kind declares: {stale}"


def test_every_reference_page_exists() -> None:
    """A spec's deep-dive page is printed to an operator, so it must be a page that is actually there.

    The path is repo-relative and hand-written, so a page renamed or moved elsewhere in the docs tree
    would otherwise reach the report as a dead link.
    """
    missing = sorted(
        f"{kind.value} -> {spec.reference_page}"
        for kind, spec in FINDING_SPECS.items()
        if spec.reference_page is not None and not (_REPO_ROOT / spec.reference_page).is_file()
    )
    assert not missing, f"FindingSpec.reference_page values that name no file: {missing}"


def test_every_kind_has_a_spec() -> None:
    """The enum and the table cannot drift apart: one entry per kind, each filed under its own kind."""
    assert set(FINDING_SPECS) == set(FindingKind)
    assert all(spec.kind is kind for kind, spec in FINDING_SPECS.items())
