"""Tests for per-level resource requirements and the machine-fit verdict."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from horde_worker_regen.benchmark import requirements as requirements_mod
from horde_worker_regen.benchmark.ladder import LadderOptions, RampLevel, build_default_ladder
from horde_worker_regen.benchmark.report import MachineInfo
from horde_worker_regen.benchmark.requirements import (
    LevelRequirements,
    compute_level_requirements,
    requirement_skip_reason,
)


def _present(_name: str) -> bool | None:
    """A presence resolver that reports every model as on disk (avoids touching the reference manager)."""
    return True


def _sd15_baseline_and_download_levels() -> tuple[RampLevel, RampLevel]:
    """Build a minimal sd15 ladder with the ad-hoc download level, returning (baseline, download)."""
    ladder = build_default_ladder(
        LadderOptions(
            tiers=["sd15"],
            include_concurrency=False,
            include_features=False,
            include_alchemy=False,
            include_downloads=True,
        ),
    )
    baseline = next(level for level in ladder if level.establishes_tier_baseline)
    download = next(level for level in ladder if level.requires_network)
    return baseline, download


def test_compute_requirements_baseline_needs_no_network_or_key() -> None:
    """A plain sd15 baseline needs its checkpoint but neither network nor a CivitAI token."""
    baseline, _ = _sd15_baseline_and_download_levels()
    req = compute_level_requirements(baseline, present_resolver=_present)
    assert req.requires_network is False
    assert req.requires_civitai_key is False
    assert req.models_required
    assert req.models_missing == []


def test_compute_requirements_download_level_needs_network_and_key() -> None:
    """The ad-hoc lora level pulls from CivitAI, so it requires both network and a token."""
    _, download = _sd15_baseline_and_download_levels()
    req = compute_level_requirements(download, present_resolver=_present)
    assert req.requires_network is True
    assert req.requires_civitai_key is True
    assert "loras" in req.features


def _req(**overrides: object) -> LevelRequirements:
    """A minimal LevelRequirements for verdict tests; override only what the test exercises."""
    base: dict[str, object] = {
        "level_id": "A-sd15-baseline",
        "stage": "A",
        "tier": "sd15",
        "axis": "baseline",
        "baseline": "stable_diffusion_1",
    }
    base.update(overrides)
    return LevelRequirements(**base)  # type: ignore[arg-type]


def test_verdict_is_none_outside_real_mode(tmp_path: Path) -> None:
    """fake/dry_run download and infer nothing, so resource gates never apply."""
    req = _req(estimated_vram_mb=999_999, requires_civitai_key=True)
    reason = requirement_skip_reason(
        req,
        machine=MachineInfo(total_vram_mb=1),
        process_mode="fake",
        cache_path=tmp_path,
        civitai_available=False,
    )
    assert reason is None


def test_insufficient_vram_skips_unless_forced(tmp_path: Path) -> None:
    """A level that needs more VRAM than the machine has is skipped, but --force attempts it anyway."""
    req = _req(estimated_vram_mb=24_000)
    machine = MachineInfo(total_vram_mb=8_000)
    skip = requirement_skip_reason(
        req,
        machine=machine,
        process_mode="real",
        cache_path=tmp_path,
        civitai_available=True,
    )
    assert skip is not None
    assert "insufficient VRAM" in skip

    forced = requirement_skip_reason(
        req,
        machine=machine,
        process_mode="real",
        cache_path=tmp_path,
        civitai_available=True,
        force=True,
    )
    assert forced is None


def test_missing_civitai_token_skips_lora_level_unless_forced(tmp_path: Path) -> None:
    """A lora/TI level without a configured token is skipped with a clear reason; --force overrides."""
    req = _req(requires_civitai_key=True, estimated_vram_mb=2_000)
    machine = MachineInfo(total_vram_mb=8_000)
    skip = requirement_skip_reason(
        req,
        machine=machine,
        process_mode="real",
        cache_path=tmp_path,
        civitai_available=False,
    )
    assert skip is not None
    assert "CivitAI" in skip

    assert (
        requirement_skip_reason(
            req,
            machine=machine,
            process_mode="real",
            cache_path=tmp_path,
            civitai_available=True,
        )
        is None
    )
    assert (
        requirement_skip_reason(
            req,
            machine=machine,
            process_mode="real",
            cache_path=tmp_path,
            civitai_available=False,
            force=True,
        )
        is None
    )


def test_insufficient_disk_skips_unless_forced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The live disk-free check skips a level when the cache volume is below the floor; --force overrides."""
    req = _req(min_disk_free_gb=10.0, estimated_vram_mb=2_000)
    machine = MachineInfo(total_vram_mb=8_000)
    monkeypatch.setattr(
        requirements_mod.shutil,
        "disk_usage",
        lambda _path: types.SimpleNamespace(total=0, used=0, free=1 * 1024**3),
    )
    skip = requirement_skip_reason(
        req,
        machine=machine,
        process_mode="real",
        cache_path=tmp_path,
        civitai_available=True,
    )
    assert skip is not None
    assert "insufficient disk" in skip

    assert (
        requirement_skip_reason(
            req,
            machine=machine,
            process_mode="real",
            cache_path=tmp_path,
            civitai_available=True,
            force=True,
        )
        is None
    )


def test_absent_huge_checkpoint_is_a_hard_skip_even_when_forced(tmp_path: Path) -> None:
    """A missing flux/qwen checkpoint cannot be forced: real-mode benchmarking never downloads weights."""
    req = _req(tier="flux", models_missing=["Flux.1-Schnell fp8 (Compact)"])
    reason = requirement_skip_reason(
        req,
        machine=MachineInfo(total_vram_mb=80_000),
        process_mode="real",
        cache_path=tmp_path,
        civitai_available=True,
        force=True,
    )
    assert reason is not None
    assert "not present on disk" in reason
