"""Tests for generated API-reference maintenance."""

from pathlib import Path

from docs.build_docs import _remove_stale_generated_stubs


def test_stale_stub_pruning_preserves_current_and_hand_written_pages(tmp_path: Path) -> None:
    """Only exact generated pages absent from the expected source set are removed."""
    docs_root = tmp_path / "docs" / "horde_worker_regen"
    package = docs_root / "sample"
    package.mkdir(parents=True)
    current = package / "current.md"
    stale = package / "stale.md"
    narrative = package / "guide.md"
    current.write_text("# current\n::: horde_worker_regen.sample.current\n", encoding="utf-8")
    stale.write_text("# stale\n::: horde_worker_regen.sample.stale\n", encoding="utf-8")
    narrative.write_text("# guide\n\nHand-written explanation.\n", encoding="utf-8")

    _remove_stale_generated_stubs(docs_root, {current})

    assert current.exists()
    assert narrative.exists()
    assert not stale.exists()
