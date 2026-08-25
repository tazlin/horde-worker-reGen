"""The annotator hub entries a pre-isolation worker fetched are copied into the isolated cache, and nothing else is."""

from __future__ import annotations

from pathlib import Path

from horde_worker_regen.model_download_core import migrate_hub_annotator_repos

_REPOS = ("Intel/zoedepth-nyu-kitti", "shi-labs/oneformer_ade20k_swin_large")


def _seed_repo(hub_dir: Path, repo: str) -> Path:
    repo_dir = hub_dir / ("models--" + repo.replace("/", "--")) / "snapshots" / "abc"
    repo_dir.mkdir(parents=True)
    (repo_dir / "config.json").write_text("{}", encoding="utf-8")
    return repo_dir.parent.parent


def test_named_repos_are_copied_and_the_legacy_cache_is_untouched(tmp_path: Path) -> None:
    """Only the named repos are carried across, and the legacy cache keeps every entry: it may be shared."""
    legacy = tmp_path / "legacy" / "hub"
    target = tmp_path / "isolated" / "hub"
    wanted = _seed_repo(legacy, _REPOS[0])
    foreign = _seed_repo(legacy, "someone/their-llm")

    copied = migrate_hub_annotator_repos(target_hub_dir=target, legacy_hub_dirs=[legacy], repos=_REPOS)

    assert copied == [_REPOS[0]]
    assert (wanted / "snapshots" / "abc" / "config.json").is_file(), "the source stays where it was"
    assert (target / wanted.name / "snapshots" / "abc" / "config.json").is_file()
    assert foreign.is_dir()
    assert not (target / foreign.name).exists()


def test_present_target_entries_are_left_alone(tmp_path: Path) -> None:
    """An entry already in the isolated cache is authoritative; the legacy copy is not moved over it."""
    legacy = tmp_path / "legacy" / "hub"
    target = tmp_path / "isolated" / "hub"
    legacy_copy = _seed_repo(legacy, _REPOS[0])
    _seed_repo(target, _REPOS[0])

    copied = migrate_hub_annotator_repos(target_hub_dir=target, legacy_hub_dirs=[legacy], repos=_REPOS)

    assert copied == []
    assert legacy_copy.is_dir()


def test_first_legacy_location_holding_the_entry_wins(tmp_path: Path) -> None:
    """Legacy caches are consulted in order; the first one holding an entry supplies it."""
    first = tmp_path / "first" / "hub"
    second = tmp_path / "second" / "hub"
    target = tmp_path / "isolated" / "hub"
    _seed_repo(second, _REPOS[1])

    copied = migrate_hub_annotator_repos(target_hub_dir=target, legacy_hub_dirs=[first, second], repos=_REPOS)

    assert copied == [_REPOS[1]]
    assert (target / ("models--" + _REPOS[1].replace("/", "--"))).is_dir()
