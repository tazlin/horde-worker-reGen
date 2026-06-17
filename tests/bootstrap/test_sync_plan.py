"""Unit tests for the dependency-sync preview plan (parsing, classification, sizing, decision)."""

from __future__ import annotations

from pathlib import Path

from worker_bootstrap import sync_plan
from worker_bootstrap.sync_plan import ChangeKind, PackageChange

# A representative uv `sync --dry-run` plan: a torch+torchvision upgrade, a small upgrade, a fresh
# install, and a removal. Indentation and the +/- markers mirror uv's output.
_DRY_RUN = """\
Resolved 312 packages in 1.20s
Would download 4 packages
Would install 4 packages
 - torch==2.11.0+cu132
 + torch==2.12.1+cu132
 - torchvision==0.26.0+cu132
 + torchvision==0.27.1+cu132
 - pydantic==2.9.2
 + pydantic==2.10.0
 + brand-new-pkg==1.0.0
 - retired-pkg==0.1.0
"""


def test_version_tuple_strips_local_and_compares() -> None:
    """Version comparison strips local (+cuXXX) segments and tolerates trailing non-numeric pieces."""
    assert sync_plan.version_at_least("2.12.1+cu132", "2.12.1")
    assert sync_plan.version_at_least("2.12.1+cu132", "2.11.0")
    assert not sync_plan.version_at_least("2.11.0+cu132", "2.12.0")
    # A trailing non-numeric segment (e.g. an rc) is truncated, not crashed on.
    assert sync_plan.version_at_least("2.12.0rc1", "2.12.0")


def test_parse_dry_run_classifies_changes() -> None:
    """A dry-run plan is parsed into upgrade/install/remove changes with from/to versions and largeness."""
    changes = {c.name: c for c in sync_plan.parse_dry_run(_DRY_RUN)}

    assert changes["torch"].kind == ChangeKind.UPGRADE
    assert changes["torch"].from_version == "2.11.0+cu132"
    assert changes["torch"].to_version == "2.12.1+cu132"
    assert changes["torch"].is_large
    assert changes["torchvision"].kind == ChangeKind.UPGRADE
    assert changes["pydantic"].kind == ChangeKind.UPGRADE
    assert not changes["pydantic"].is_large
    assert changes["brand-new-pkg"].kind == ChangeKind.INSTALL
    assert changes["retired-pkg"].kind == ChangeKind.REMOVE
    assert changes["retired-pkg"].est_download_bytes == 0  # removals download nothing


def test_parse_dry_run_ignores_unknown_lines() -> None:
    """Lines that are not +/- package entries are ignored, yielding no changes."""
    assert sync_plan.parse_dry_run("Resolved 1 package\nnonsense line\nAudited 1 package") == []


def test_is_held_upgrade_only_for_torch_pair_upgrades() -> None:
    """Only torch/torchvision upgrades (with a known prior version) count as held candidates."""
    torch_up = sync_plan.parse_dry_run(_DRY_RUN)
    held = [c.name for c in torch_up if sync_plan.is_held_upgrade(c)]
    assert sorted(held) == ["torch", "torchvision"]


def test_installed_versions_reads_dist_info(tmp_path: Path) -> None:
    """Installed versions are read from dist-info dir names and normalized, without importing packages."""
    site = tmp_path / "Lib" / "site-packages"
    site.mkdir(parents=True)
    (site / "torch-2.11.0+cu132.dist-info").mkdir()
    (site / "nvidia_cudnn_cu12-9.1.0.dist-info").mkdir()
    (site / "not-a-package").mkdir()

    versions = sync_plan.installed_versions(tmp_path)

    assert versions["torch"] == "2.11.0+cu132"
    assert versions["nvidia-cudnn-cu12"] == "9.1.0"  # normalized name


def test_held_overrides_text_pins_public_versions() -> None:
    """The override body pins held packages at their installed public version (local segment dropped)."""
    changes = sync_plan.parse_dry_run(_DRY_RUN)
    text = sync_plan.held_overrides_text(changes, {"torch": "2.11.0+cu132", "torchvision": "0.26.0+cu132"})
    assert text == "torch==2.11.0\ntorchvision==0.26.0\n"


def test_held_overrides_text_none_when_nothing_to_hold() -> None:
    """With no torch/torchvision upgrade there is nothing to hold, so the override body is None."""
    changes = sync_plan.parse_dry_run("Resolved\n + pydantic==2.10.0")
    assert sync_plan.held_overrides_text(changes, {}) is None


def test_build_plan_marks_required_when_not_holdable() -> None:
    """When the held dry-run fails, large upgrades are marked REQUIRED (not skippable)."""
    changes = sync_plan.parse_dry_run(_DRY_RUN)
    plan = sync_plan.build_plan(
        changes,
        holdable=False,
        cache_dir="C:/data/uv_cache",
        cache_is_owned=True,
        free_disk_bytes=10**12,
    )
    assert not plan.has_skippable_upgrade
    assert all(c.required for c in plan.large_upgrades)


def test_build_plan_optional_when_holdable() -> None:
    """When the held dry-run resolves, large upgrades are optional and a download total is computed."""
    changes = sync_plan.parse_dry_run(_DRY_RUN)
    plan = sync_plan.build_plan(
        changes,
        holdable=True,
        cache_dir="C:/data/uv_cache",
        cache_is_owned=True,
        free_disk_bytes=10**12,
    )
    assert plan.has_skippable_upgrade
    assert not any(c.required for c in plan.large_upgrades)
    assert plan.total_download_bytes > 0


def _plan(*, holdable: bool, total_bytes: int) -> sync_plan.SyncPlan:
    change = PackageChange(
        name="torch",
        from_version="2.11.0",
        to_version="2.12.1",
        kind=ChangeKind.UPGRADE,
        est_download_bytes=total_bytes,
        is_large=True,
        required=not holdable,
    )
    return sync_plan.SyncPlan(
        changes=(change,),
        total_download_bytes=total_bytes,
        cache_dir="C:/data/uv_cache",
        cache_is_owned=True,
        free_disk_bytes=10**12,
        holdable=holdable,
    )


_BIG = 2 * 1024**3
_SMALL = 10 * 1024**2
_THRESHOLD = 1500 * 1024 * 1024


def test_decide_proceeds_when_no_large_upgrade() -> None:
    """With no large upgrade, decide always proceeds (nothing to hold), even if hold was requested."""
    empty = sync_plan.SyncPlan((), 0, "c", True, None, False)
    assert _run_decide(empty, hold_requested=True) == "proceed_full"


def test_decide_mandatory_upgrade_ignores_hold() -> None:
    """A mandatory (non-resolvable) upgrade proceeds and explains that holding is impossible."""
    plan = _plan(holdable=False, total_bytes=_BIG)
    messages: list[str] = []
    action = _run_decide(plan, hold_requested=True, emit=messages.append)
    assert action == "proceed_full"
    assert any("mandatory" in m for m in messages)


def test_decide_explicit_hold_wins() -> None:
    """An explicit hold request takes the held path for a resolvable optional upgrade."""
    assert _run_decide(_plan(holdable=True, total_bytes=_BIG), hold_requested=True) == "hold"


def test_decide_under_threshold_proceeds_silently() -> None:
    """An optional upgrade below the confirm threshold proceeds without prompting."""
    assert _run_decide(_plan(holdable=True, total_bytes=_SMALL), interactive=True) == "proceed_full"


def test_decide_headless_policy_hold() -> None:
    """A headless run with policy 'hold' holds a big optional upgrade."""
    action = _run_decide(
        _plan(holdable=True, total_bytes=_BIG),
        headless=True,
        headless_policy="hold",
        interactive=False,
    )
    assert action == "hold"


def test_decide_headless_policy_proceed() -> None:
    """A headless run with policy 'proceed' takes a big optional upgrade."""
    action = _run_decide(
        _plan(holdable=True, total_bytes=_BIG),
        headless=True,
        headless_policy="proceed",
        interactive=False,
    )
    assert action == "proceed_full"


def test_decide_interactive_prompt_hold_and_cancel() -> None:
    """Interactive prompt answers map to hold / abort / proceed."""
    plan = _plan(holdable=True, total_bytes=_BIG)
    assert _run_decide(plan, interactive=True, answer="h") == "hold"
    assert _run_decide(plan, interactive=True, answer="c") == "abort"
    assert _run_decide(plan, interactive=True, answer="") == "proceed_full"


def _run_decide(
    plan: sync_plan.SyncPlan,
    *,
    hold_requested: bool = False,
    headless: bool = False,
    headless_policy: str = "proceed",
    interactive: bool = False,
    answer: str = "",
    emit: object = None,
) -> str:
    return sync_plan.decide(
        plan,
        hold_requested=hold_requested,
        headless=headless,
        headless_policy=headless_policy,
        confirm_threshold_bytes=_THRESHOLD,
        interactive=interactive,
        prompt=lambda _: answer,
        emit=emit or (lambda _: None),
    )
